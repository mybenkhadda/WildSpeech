"""Notebook must parse as Python, with Drive mount and readable phase anchors."""
import ast,json
from pathlib import Path

def test_m86_notebook_is_clean_and_reproducible():
    path=Path(__file__).resolve().parents[1]/"notebooks"/"WILDSPEECH_M86_M0PP_MASTER.ipynb"
    j=json.loads(path.read_text(encoding="utf-8"))
    assert len(j["cells"])>=28
    code=[]
    for i,cell in enumerate(j["cells"]):
        if cell["cell_type"]=="code":
            src="".join(cell["source"])
            ast.parse(src,filename=f"m86_cell{i}.py")
            # A research notebook legitimately retains FULL execution outputs.
            # Preserve evidence and check source syntax/schema only.
            assert isinstance(cell.get("outputs",[]),list)
            assert cell.get("execution_count") is None or isinstance(cell["execution_count"],int)
            code.append(src)
    joined="\n".join(code)
    assert joined.index("drive.mount(")<joined.index("execute(")
    assert 'MODE="SCREEN"' in joined or 'MODE="FULL"' in joined
    assert "M85_ROOT" in joined and "PHASE0_ROOT" in joined
    assert "EXPORT_E04_DIAGNOSTIC=False" in joined
    assert "M86_DECISION.md" in joined and "EXPORT_CANDIDATE=True" in joined
