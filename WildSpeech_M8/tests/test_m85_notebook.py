import ast
import json
from pathlib import Path

NOTEBOOK=Path(__file__).resolve().parents[1]/"notebooks"/"WILDSPEECH_M85_M0PLUS_MASTER.ipynb"


def test_colab_contract_syntax_and_order():
    nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codes=[]
    for idx,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            code="".join(c["source"])
            ast.parse(code,filename=f"m85_cell{idx}.py")
            codes.append(code)
            assert c["outputs"]==[]
    joined="\n".join(codes)
    assert len(codes)>=10 and len(nb["cells"])>=20
    assert joined.index("drive.mount(")<joined.index("execute(")
    assert 'MODE="SCREEN"' in joined
    for term in ("EVAL_HASH_2026","PHASE0_ROOT","M85_M0PLUS_DECISION.md",
                 "locked_plan()","reference_metrics","RESULT[\"decision\"]"):
        assert term in joined
