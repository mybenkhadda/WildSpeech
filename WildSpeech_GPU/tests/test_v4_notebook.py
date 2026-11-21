"""The V4 Colab notebook is valid JSON, carries no outputs, and every code cell is syntactically valid Python that
drives the package (no placeholder cells)."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks" / "WILDSPEECH_V4_MULTIMODAL_MASTER.ipynb"


def _cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb["cells"]


def test_notebook_cells_are_code_and_parse():
    code = [c for c in _cells() if c["cell_type"] == "code"]
    assert len(code) >= 10
    for c in code:
        src = "".join(c["source"])
        assert src.strip(), "empty code cell"
        assert not c.get("outputs"), "notebook must be committed without outputs"
        ast.parse(src)


def test_notebook_uses_v4_pipeline_and_required_steps():
    src = "\n".join("".join(c["source"]) for c in _cells() if c["cell_type"] == "code")
    for needle in ("drive.mount", "git -C", "pip install -q -e", "V4Pipeline", "pipe.prepare", "pipe.features",
                   "pipe.experiments", "pipe.figures", "pipe.submission", "pipe.report", "pinned_commit",
                   "permission_review", "12SzHSmkNbeoS_mO1L80uEd6A2k4jH-8V", "all_models_predictions.parquet",
                   "'rerun_asr': False"):
        assert needle in src, needle


def test_config_keys_match_package_defaults():
    from wildspeech.v4.config import DEFAULT_CONFIG

    src = "".join(_cells()[1]["source"])
    for k in DEFAULT_CONFIG:
        assert f"'{k}'" in src, k
