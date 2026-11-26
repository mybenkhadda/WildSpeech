import pytest
from wildspeech_m84.ledger import Ledger,preregister,make_run_id


def test_preregistration_and_append_only_transitions(tmp_path):
    plan=preregister(tmp_path)
    assert len(plan["comparisons"])>=5
    assert preregister(tmp_path)["plan_hash"]==plan["plan_hash"]
    with pytest.raises(RuntimeError,match="PRE-REGISTERED"):
        preregister(tmp_path,[])
    ledger=Ledger(tmp_path)
    rid=make_run_id("REF","sha","features","folds","config")
    ledger.transition(rid,"PLANNED",model_id="REF")
    ledger.transition(rid,"RUNNING")
    ledger.transition(rid,"COMPLETE",mean_fold_r2=.193)
    assert ledger.latest(rid)["status"]=="COMPLETE"
    assert len(ledger.snapshot())==1
    with pytest.raises(RuntimeError,match="Invalid immutable"):
        ledger.transition(rid,"RUNNING")
