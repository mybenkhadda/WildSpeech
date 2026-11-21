import numpy as np
import pandas as pd

from wildspeech.student_aggregation import (accuracy_features, conditional_stats, correctness_contrasts, group_stats,
                                            shrunk_slopes)


def _clips():
    return pd.DataFrame({
        "student_id": ["a"] * 4 + ["b"] * 3, "task_family": ["blending", "blending", "deletion", "deletion", "blending", "deletion", "deletion"],
        "correct": [1, 0, 1, 1, 0, 0, 1], "t_active_dur": [1.0, 2.0, 3.0, 4.0, 2.0, 2.5, 3.5], "l_n_phonemes": [3, 4, 5, 6, 3, 4, 6]})


def test_one_row_per_student():
    c = _clips()
    g = group_stats(c, "student_id", ["t_active_dur"], ["mean", "std", "count", "mad", "p25"], min_obs=2)
    assert g.index.is_unique and set(g.index) == {"a", "b"}
    assert g.loc["a", "t_active_dur__mean"] == 2.5
    assert g.loc["b", "t_active_dur__count"] == 3


def test_conditional_and_accuracy_features():
    c = _clips()
    cond = conditional_stats(c, "task_family", "task", ["t_active_dur"], ["mean"], 1)
    assert cond.loc["a", "t_active_dur__task=blending__mean"] == 1.5
    acc = accuracy_features(c)
    assert acc.loc["a", "acc_blending"] == 0.5 and acc.loc["b", "acc_n_attempted_deletion"] == 2


def test_contrasts_require_both_categories():
    c = _clips()
    cc = correctness_contrasts(c, {"duration": "t_active_dur"}, 1)
    col = "cc_correct_minus_incorrect_duration__all"
    assert np.isclose(cc.loc["a", col], (1 + 3 + 4) / 3 - 2)
    blend = "cc_correct_minus_incorrect_duration__blending"
    assert np.isnan(cc.loc["b", blend])


def test_shrunk_slopes_fall_back_to_pooled_with_few_observations():
    c = _clips()
    sl = shrunk_slopes(c, "student_id", "l_n_phonemes", "t_active_dur", k=5.0, min_obs=4)
    assert sl.loc["a", "reliable"] == 1.0 and sl.loc["b", "reliable"] == 0.0
    pooled = sl.loc["b", "slope_shrunk"]
    assert np.isfinite(pooled)
