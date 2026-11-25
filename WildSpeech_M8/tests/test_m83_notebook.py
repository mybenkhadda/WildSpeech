"""Fast, offline structure tests for the Drive-first M8.3 notebook."""
from __future__ import annotations
import ast
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "WILDSPEECH_M83_BOTTLENECK_MASTER.ipynb"

def read_cells():
    notebook=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 50, "Expected exhaustive notebook, not a launcher"
    ids=[c["id"] for c in notebook["cells"]]
    assert len(ids)==len(set(ids))
    return notebook["cells"]

def test_notebook_all_python_cells_compile():
    cells=read_cells()
    for i,c in enumerate(cells):
        if c["cell_type"] == "code":
            source="".join(c["source"])
            try: ast.parse(source,filename=f"notebook_cell_{i}.py")
            except SyntaxError as e: raise AssertionError(f"Invalid Python code cell {i}: {e}") from e

def test_drive_first_feature_lineage_and_evidence():
    cells=read_cells()
    code=["".join(c["source"]) for c in cells if c["cell_type"]=="code"]
    combined="\n".join(code)
    must=["drive.mount", "read_official(raw_cfg)", "prepare_dataset(raw_cfg)",
          "run_frame(X_train,y,schools,OUT,CFG)", "oof_predictions_PRIVATE",
          "M83_ATLAS.html", "M83_DEEP_DIVE.md","FULL","SMOKE",
          "RUN_SIGNATURE","feature_quality","domain_auc","neighbor_pairs",
          "residual_probe","negative_control","learning_curve"]
    missing=[token for token in must if token not in combined]
    assert not missing,f"Missing notebook sections: {missing}"
    assert combined.index("drive.mount")<combined.index("read_official(raw_cfg)")
    assert combined.index("prepare_dataset(raw_cfg)")<combined.index("run_frame(X_train,y,schools,OUT,CFG)")
    assert "zipfile" not in combined
    assert "force_remount=True" not in combined
