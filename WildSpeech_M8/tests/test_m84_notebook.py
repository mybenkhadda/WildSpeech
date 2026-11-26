"""Notebook structure and executable cell syntax validation (no private data)."""
from __future__ import annotations
import ast
import json
from pathlib import Path

NOTEBOOK=Path(__file__).resolve().parents[1]/"notebooks"/"WILDSPEECH_M84_PHASE0_MASTER.ipynb"


def test_m84_notebook_cells_compile_and_drive_first():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells=nb["cells"]
    assert len(cells)>=25
    assert len([c for c in cells if c["cell_type"]=="code"])>=12
    assert len(set(c["id"] for c in cells))==len(cells)
    code=[]
    for n,c in enumerate(cells):
        if c["cell_type"]=="code":
            s="".join(c["source"])
            ast.parse(s,filename=f"m84_notebook_{n}.py")
            code.append(s)
    all_code="\n".join(code)
    assert all_code.index('drive.mount(')<all_code.index('run_phase0(')
    for token in ("preregister(ROOT)","build_folds(D.ids,D.groups)",
                  "freeze_folds(FOLDS,D.ids,D.groups,ROOT)","run_phase0(",
                  "reference_submission.csv","M84_BASELINE_PARITY.md",
                  "RUN_LEGACY_RESEARCH_ONLY=False","require_parity=REQUIRE_REFERENCE_PARITY"):
        assert token in all_code
    assert "zipfile" not in all_code
