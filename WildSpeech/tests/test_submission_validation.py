import numpy as np
import pandas as pd

from wildspeech.submission import validate_submission


def _fmt():
    return pd.DataFrame({"student_id": ["a", "b", "c"], "score": [0.0, 0.0, 0.0]})


def test_valid_submission_passes():
    sub = _fmt().assign(score=[1.0, 2.0, 3.0])
    assert validate_submission(sub, _fmt(), "student_id", "score")["valid"]


def test_detects_missing_nonfinite_duplicates_and_order():
    fmt = _fmt()
    assert not validate_submission(fmt.assign(score=[1.0, np.nan, 2.0]), fmt, "student_id", "score")["no_missing_predictions"]
    assert not validate_submission(fmt.assign(score=[1.0, np.inf, 2.0]), fmt, "student_id", "score")["all_finite"]
    dup = fmt.assign(student_id=["a", "a", "c"], score=1.0)
    chk = validate_submission(dup, fmt, "student_id", "score")
    assert not chk["no_duplicate_ids"] and not chk["exact_id_coverage"]
    reordered = fmt.iloc[[2, 1, 0]].assign(score=1.0).reset_index(drop=True)
    assert not validate_submission(reordered, fmt, "student_id", "score")["order_preserved"]
    renamed = fmt.rename(columns={"score": "prediction"}).assign(prediction=1.0)
    assert not validate_submission(renamed.rename(columns={"prediction": "score"}).iloc[:2], fmt, "student_id", "score")["row_count_matches"]
