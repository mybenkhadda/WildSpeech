import numpy as np
import pandas as pd

from wildspeech.residual_models import nested_two_stage_cv
from wildspeech.validation import iter_folds, make_cv_assignments


class RecordingBuilder:
    """Records every training set used for fitting; features are fixed columns."""

    def __init__(self, base):
        self.base = base
        self.fits = []

    def build(self, train_ids, val_ids, families, key=None):
        self.fits.append((key, set(train_ids)))
        cols = ["theta_blending", "theta_deletion"] if "SCREENING" in families else ["t_active_dur__mean"]
        return self.base.loc[list(train_ids), cols], self.base.loc[list(val_ids), cols]

    def columns_for(self, families):
        return []


def _data(synthetic_students):
    base = synthetic_students[["theta_blending", "theta_deletion"]].copy()
    base["t_active_dur__mean"] = np.random.default_rng(3).normal(size=len(base))
    return base, synthetic_students["target"]


def test_outer_validation_students_never_used_for_fitting(synthetic_students, cfg):
    base, y = _data(synthetic_students)
    assign, _ = make_cv_assignments(synthetic_students, dict(cfg, cv_repeats=1))
    b = RecordingBuilder(base)
    nested_two_stage_cv(b, y, assign, "CV_A_REPEATED_STUDENT_KFOLD", cfg, ["SCREENING"], ["TEMPORAL"])
    for rep, fold, tr, va in iter_folds(assign, "CV_A_REPEATED_STUDENT_KFOLD"):
        key = ("CV_A_REPEATED_STUDENT_KFOLD", rep, fold)
        fits = [s for k, s in b.fits if k is not None and k[:3] == key]
        assert fits and all(not (s & set(va)) for s in fits)


def test_alpha_does_not_depend_on_outer_validation_outcomes(synthetic_students, cfg):
    base, y = _data(synthetic_students)
    assign, _ = make_cv_assignments(synthetic_students, dict(cfg, cv_repeats=1))
    r1 = nested_two_stage_cv(RecordingBuilder(base), y, assign, "CV_A_REPEATED_STUDENT_KFOLD", cfg, ["SCREENING"], ["TEMPORAL"])
    first_val = next(iter_folds(assign, "CV_A_REPEATED_STUDENT_KFOLD"))[3]
    y2 = y.copy()
    y2.loc[first_val] = y2.loc[first_val] + 1000.0
    r2 = nested_two_stage_cv(RecordingBuilder(base), y2, assign, "CV_A_REPEATED_STUDENT_KFOLD", cfg, ["SCREENING"], ["TEMPORAL"])
    assert r1.fold_metrics.loc[0, "alpha"] == r2.fold_metrics.loc[0, "alpha"]
    pd.testing.assert_series_equal(r1.oof[r1.oof["fold"] == 0]["pred"].reset_index(drop=True),
                                   r2.oof[r2.oof["fold"] == 0]["pred"].reset_index(drop=True))
