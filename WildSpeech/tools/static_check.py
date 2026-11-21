"""Static checks that never execute project code: undefined global names and
cross-module attribute references (``module.name`` and ``from .module import name``)."""
from __future__ import annotations

import ast
import builtins
import sys
import symtable
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "wildspeech"


def module_globals(tree: ast.Module) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node in tree.body:
            names.add(node.name)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.For, ast.With, ast.If, ast.Try)):
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    names.add(n.id)
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        names.add((a.asname or a.name).split(".")[0])
    return names


def check_undefined(path: Path) -> list:
    src = path.read_text()
    tree = ast.parse(src)
    comp_targets = {n.id for c in ast.walk(tree) if isinstance(c, ast.comprehension) for n in ast.walk(c.target) if isinstance(n, ast.Name)}
    globs = module_globals(tree) | set(dir(builtins)) | {"__file__", "__name__"} | comp_targets
    problems = []

    def walk(tab):
        for sym in tab.get_symbols():
            if sym.is_referenced() and (sym.is_global() or (tab.get_type() == "module" and not sym.is_assigned() and not sym.is_imported())):
                if sym.get_name() not in globs:
                    problems.append(f"{path.name}:{tab.get_name()}: undefined name {sym.get_name()}")
        for child in tab.get_children():
            walk(child)
    walk(symtable.symtable(src, str(path), "exec"))
    return problems


def defined_names(path: Path) -> set:
    return module_globals(ast.parse(path.read_text()))


def check_cross_refs() -> list:
    modules = {p.stem: p for p in PKG.glob("*.py")}
    defs = {m: defined_names(p) for m, p in modules.items()}
    problems = []
    for m, p in modules.items():
        tree = ast.parse(p.read_text())
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module is None:
                    for a in node.names:
                        aliases[a.asname or a.name] = a.name
                        if a.name not in modules:
                            problems.append(f"{m}: imports missing module {a.name}")
                else:
                    target = node.module.split(".")[0]
                    if target not in modules:
                        problems.append(f"{m}: imports missing module {node.module}")
                        continue
                    for a in node.names:
                        if a.name not in defs[target]:
                            problems.append(f"{m}: '{a.name}' not defined in {target}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                mod = aliases[node.value.id]
                if node.attr not in defs.get(mod, set()):
                    problems.append(f"{m}: {node.value.id}.{node.attr} not defined in {mod}")
    return problems


if __name__ == "__main__":
    issues = []
    for p in sorted(PKG.glob("*.py")):
        issues += check_undefined(p)
    issues += check_cross_refs()
    print("\n".join(issues) if issues else "OK: no undefined names or broken cross-module references")
    sys.exit(1 if issues else 0)
