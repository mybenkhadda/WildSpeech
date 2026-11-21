"""Repository packaging guards: every wildspeech import resolves to a source file, src/wildspeech/models exists,
and .gitignore can never exclude a source package again (regression: unanchored `models/` hid the package)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from wildspeech.v4.source_check import REQUIRED_MODULES, SourceIncompleteError, assert_complete, check_sources

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def test_checkout_is_complete():
    rep = check_sources(SRC)
    assert rep["complete"], rep["missing"]
    assert (SRC / "wildspeech" / "models" / "zoo.py").is_file()


def test_missing_package_is_detected(tmp_path):
    dst = tmp_path / "src"
    shutil.copytree(SRC / "wildspeech", dst / "wildspeech", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(dst / "wildspeech" / "models")
    with pytest.raises(SourceIncompleteError) as e:
        assert_complete(dst)
    assert "wildspeech.models" in str(e.value)


def test_required_modules_import():
    import importlib

    for m in REQUIRED_MODULES:
        importlib.import_module(m)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("git") is None or not (REPO.parent / ".git").exists(), reason="needs the git checkout")
def test_gitignore_never_excludes_source_files():
    py = [str(p.relative_to(REPO)) for d in ("src", "tests", "scripts") for p in (REPO / d).rglob("*.py")
          if "__pycache__" not in p.parts]
    r = _git("check-ignore", "--no-index", *py)
    assert r.stdout.strip() == "", f"ignored source files:\n{r.stdout}"
    # artefact folders at the project root are still ignored
    r2 = _git("check-ignore", "--no-index", "models/x.bin", "cache/x", "outputs/x", "submissions/x.csv")
    assert len(r2.stdout.split()) == 4


def test_gitignore_artifact_patterns_are_anchored():
    lines = [l.strip() for l in (REPO / ".gitignore").read_text().splitlines() if l.strip() and not l.startswith("#")]
    for name in ("models/", "cache/", "outputs/", "experiments/", "submissions/", "reports/"):
        assert name not in lines, f"unanchored pattern {name!r} would also match src/wildspeech/{name}"
        assert "/" + name in lines
