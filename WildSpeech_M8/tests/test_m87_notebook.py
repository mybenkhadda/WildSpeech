"""M8.7 master checks; repository notebooks retain no fabricated outputs."""
import ast,json
from pathlib import Path
def test_master_python_syntax_and_pipeline_controls():
    p=Path(__file__).resolve().parents[1]/"notebooks"/"WILDSPEECH_M87_SPEECH_EVIDENCE_MASTER.ipynb"
    nb=json.loads(p.read_text(encoding="utf-8"))
    assert nb["nbformat"]==4 and len(nb["cells"])>=45
    code=[]
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":
            src="".join(c["source"])
            ast.parse(src,filename=f"m87_cell_{i}.py")
            assert c["execution_count"] is None and c["outputs"]==[]
            code.append(src)
    all_code="\n".join(code)
    for value in ("drive.mount(","validate_baselines(","extract(","evaluate(",
                  "MODE=\"SCREEN\"","RUN_GPU_EXTRACT=False",
                  "LICENSE_ACKNOWLEDGED=False","qa","M87_DECISION.md",
                  "RUN_SUBMISSION_EXPORT=False"):
        assert value in all_code
