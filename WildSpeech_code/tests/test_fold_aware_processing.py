import numpy as np
import pandas as pd

from wildspeech.fold_features import FoldFeatureBuilder
from wildspeech.student_aggregation import item_adjusted_features


def test_theta_common_factor_ignores_validation_rows(ctx, synthetic_students):
    base = synthetic_students[["theta_blending", "theta_deletion", "theta_repetition"]]
    train, val = list(base.index[:40]), list(base.index[40:])
    b1 = FoldFeatureBuilder(ctx, base=base)
    tr1, _ = b1.build(train, val, {"SCREENING"})
    perturbed = base.copy()
    perturbed.loc[val] = perturbed.loc[val] * 100 + 50
    b2 = FoldFeatureBuilder(ctx, base=perturbed)
    tr2, _ = b2.build(train, val, {"SCREENING"})
    pd.testing.assert_series_equal(tr1["theta_common_factor"], tr2["theta_common_factor"])


def test_item_reference_uses_fit_students_only():
    clips = pd.DataFrame({"student_id": ["a", "a", "b", "b", "c", "c"], "item_key": ["i1", "i2"] * 3,
                          "task_family": ["blending"] * 6, "t_active_dur": [1.0, 2.0, 1.2, 2.2, 50.0, 60.0]})
    out1 = item_adjusted_features(clips, ["t_active_dur"], ["a", "b"], k=0.0)
    clips2 = clips.copy()
    clips2.loc[clips2["student_id"] == "c", "t_active_dur"] = [500.0, 600.0]
    out2 = item_adjusted_features(clips2, ["t_active_dur"], ["a", "b"], k=0.0)
    pd.testing.assert_series_equal(out1.loc[["a", "b"], "ia_t_active_dur__mean"], out2.loc[["a", "b"], "ia_t_active_dur__mean"])


def test_model_pipeline_preprocessing_is_fitted_inside_folds(cfg):
    from wildspeech.baseline_models import make_model

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(50, 4)), columns=list("abcd"))
    y = X["a"] * 2 + rng.normal(size=50)
    m = make_model("RIDGE", cfg, n_features=4)
    m.fit(X.iloc[:30], y.iloc[:30])
    scaler = m.named_steps["scale"]
    imputed = m.named_steps["impute"].transform(X.iloc[:30])
    assert np.allclose(scaler.mean_[:4], imputed[:, :4].mean(axis=0))
