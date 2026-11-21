"""Fail-fast source completeness check (standard library only, safe to import from a broken checkout).

check_sources(src_root)
  * parses every .py file under <src_root>/wildspeech with `ast` (nothing is imported or executed)
  * resolves every `import wildspeech...` / `from wildspeech... import` / relative `from .x import y` to a module
    path, including `from pkg import name` where `name` is a submodule
  * verifies each resolved module exists as <module>.py or <package>/__init__.py
  * verifies the REQUIRED_PACKAGES (packages V4 depends on directly or transitively) exist
Returns a report dict; assert_complete() raises SourceIncompleteError listing every missing module.

Motivation: an unanchored `models/` pattern in .gitignore once removed src/wildspeech/models from the repository,
and Colab failed only deep inside training with ModuleNotFoundError: No module named 'wildspeech.models'.
"""
from __future__ import annotations

import ast
from pathlib import Path

REQUIRED_PACKAGES = (
    "wildspeech", "wildspeech.acoustic", "wildspeech.analysis", "wildspeech.asr", "wildspeech.correctness",
    "wildspeech.data", "wildspeech.diagnostics", "wildspeech.discovery", "wildspeech.evaluation",
    "wildspeech.features", "wildspeech.foundation", "wildspeech.harmonization", "wildspeech.models",
    "wildspeech.phonetic", "wildspeech.phonetics", "wildspeech.reference", "wildspeech.representations",
    "wildspeech.stages", "wildspeech.v2", "wildspeech.v3", "wildspeech.v4", "wildspeech.validation",
)
REQUIRED_MODULES = (
    "wildspeech.models.zoo", "wildspeech.models.preprocessing", "wildspeech.models.residual_fusion",
    "wildspeech.models.compact_regression", "wildspeech.models.fusion", "wildspeech.models.neural",
    "wildspeech.evaluation.nested_asr_cv", "wildspeech.validation.runner", "wildspeech.v2.validation_registry",
    "wildspeech.v4.pipeline",
)


class SourceIncompleteError(RuntimeError):
    pass


def _exists(src: Path, mod: str) -> bool:
    p = src.joinpath(*mod.split("."))
    return p.with_suffix(".py").is_file() or (p / "__init__.py").is_file()


def _module_of(src: Path, path: Path) -> tuple[str, bool]:
    rel = path.relative_to(src).with_suffix("")
    parts = list(rel.parts)
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def _resolve_relative(cur: str, is_pkg: bool, level: int, module: str | None) -> str:
    base = cur.split(".") if is_pkg else cur.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def check_sources(src_root: Path | str) -> dict:
    src = Path(src_root)
    pkg_root = src / "wildspeech"
    missing: dict[str, list[str]] = {}
    parse_errors: list[str] = []
    n_files = 0
    for f in sorted(pkg_root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        n_files += 1
        cur, is_pkg = _module_of(src, f)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            parse_errors.append(f"{f.relative_to(src)}:{e.lineno}: {e.msg}")
            continue
        for node in ast.walk(tree):
            targets: list[tuple[str, bool]] = []          # (module, optional_name_may_be_attribute)
            if isinstance(node, ast.Import):
                targets = [(a.name, False) for a in node.names if a.name.split(".")[0] == "wildspeech"]
            elif isinstance(node, ast.ImportFrom):
                mod = _resolve_relative(cur, is_pkg, node.level, node.module) if node.level else (node.module or "")
                if mod.split(".")[0] != "wildspeech":
                    continue
                targets.append((mod, False))
                # `from pkg import name`: name is either an attribute of pkg or a submodule pkg.name
                for a in node.names:
                    sub = f"{mod}.{a.name}"
                    if src.joinpath(*mod.split(".")).is_dir() and not _has_attr_definition(src, mod, a.name):
                        targets.append((sub, True))
            for mod, is_name in targets:
                if not _exists(src, mod):
                    missing.setdefault(mod, []).append(str(f.relative_to(src)))
    for mod in (*REQUIRED_PACKAGES, *REQUIRED_MODULES):
        if not _exists(src, mod):
            missing.setdefault(mod, []).append("<required>")
    return {"src_root": str(src), "n_files": n_files, "missing": missing, "parse_errors": parse_errors,
            "complete": not missing and not parse_errors}


def _has_attr_definition(src: Path, pkg: str, name: str) -> bool:
    """True if <pkg>/__init__.py defines or imports `name` (then `from pkg import name` is not a submodule)."""
    init = src.joinpath(*pkg.split(".")) / "__init__.py"
    if not init.is_file():
        return False
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name.split(".")[0]) == name for a in node.names):
            return True
    return False


def assert_complete(src_root: Path | str) -> dict:
    rep = check_sources(src_root)
    if not rep["complete"]:
        lines = [f"  {m}  (imported by {', '.join(sorted(set(v))[:5])})" for m, v in sorted(rep["missing"].items())]
        lines += [f"  syntax error: {e}" for e in rep["parse_errors"]]
        raise SourceIncompleteError("WildSpeech source checkout is incomplete — missing modules:\n" + "\n".join(lines) +
                                    "\nCheck .gitignore (artefact patterns must be anchored, e.g. '/models/') and "
                                    "that the checked-out revision contains src/wildspeech/models/.")
    return rep


if __name__ == "__main__":                                   # python -m wildspeech.v4.source_check <src_root>
    import sys

    r = assert_complete(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2])
    print(f"source complete: {r['n_files']} files checked")
