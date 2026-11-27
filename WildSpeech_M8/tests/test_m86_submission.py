"""Final train-only fit and raw singleton/permutation test of submission."""
from __future__ import annotations
import numpy as np
import pandas as pd
from test_m85_models import _as_dataset
from wildspeech_m86.submission import export_m86_candidate
from wildspeech_m84.data import digest_bytes

def test_m86_final_submissions_are_test_student_independent(tmp_path):
    data=_as_dataset()
    result=export_m86_candidate(data,"B3",tmp_path,
                                iterations=8,seeds=(42,),rasch_iters=3)
    assert result["all_test_singleton_checked"] and result["test_permutation_checked"]
    assert result["submitted_online"] is False
    file=tmp_path/"m0pp_submission.csv"
    assert digest_bytes(file)==result["sha256"]
    frame=pd.read_csv(file,dtype={"subject_id":str})
    assert list(frame.columns)==["subject_id","outcome_score"]
    assert frame.subject_id.tolist()==data.submission_format.index.tolist()
    assert np.isfinite(frame.outcome_score.to_numpy(float)).all()
    assert (tmp_path/"m0pp_model.joblib").exists()
