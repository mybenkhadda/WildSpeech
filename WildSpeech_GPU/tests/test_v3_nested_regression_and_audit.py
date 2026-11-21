"""Nested regression, fusion fallback auditing and the repaired task ensemble."""
import numpy as np
import pandas as pd
import pytest

from wildspeech.config import Config
from wildspeech.evaluation.fusion_audit import classify_v2_results, verdicts
from wildspeech.evaluation.nested_asr_cv import V3FoldModel, V3Spec, _inner_oof_residual


def _cfg():
    return Config({"validation": {"inner_splits": 4}, "models": {"max_missing_frac": 0.6, "gbm_threads": 1},
                   "v2": {"evaluation": {"family_components": 10}}, "v3": {"representations": {}}})


def _data(n=200, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({f"SCREENING__all__s{i}__value": rng.standard_normal(n) for i in range(3)})
    add = pd.DataFrame({f"ASRP__all__a{i}__value": rng.standard_normal(n) for i in range(8)})
    y = base.sum(1) * 2 + signal * 3 * add.iloc[:, 0] + rng.standard_normal(n)
    X = pd.concat([base, add], axis=1)
    X.index = [f"s{i}" for i in range(n)]
    return X, y.to_numpy(), list(base.columns), {"ASRP": list(add.columns)}


def test_inner_residual_is_out_of_fold():
    X, y, bc, _ = _data()
    r = _inner_oof_residual(X[bc], y, 0)
    from sklearn.linear_model import RidgeCV

    ins = y - RidgeCV().fit(X[bc], y).predict(X[bc])
    assert np.var(r) > np.var(ins)                     # OOF residuals are never the optimistic in-sample ones


def test_forced_zero_alpha_is_flagged_as_fallback():
    X, y, bc, g = _data(signal=1.0)
    sp = V3Spec("t", "RIDGE", ["ASRP"], 5, "pca", "residual", alphas=(0.0,), base_model="RIDGE")
    m = V3FoldModel(sp, _cfg(), 0, bc, g, [], "RIDGE").fit(X.iloc[:150], y[:150])
    t = m.trace(X.iloc[150:])
    assert t["fallback"] and "alpha=0" in t["fallback_reason"] and t["stage2_pred_sd"] == 0.0
    base = V3FoldModel(V3Spec("b", "RIDGE", [], None, "none", "early"), _cfg(), 0, bc, {}, [], "RIDGE").fit(X.iloc[:150][bc], y[:150])
    np.testing.assert_allclose(m.predict(X.iloc[150:]), base.predict(X.iloc[150:][bc]), atol=1e-8)


def test_informative_residual_is_used():
    X, y, bc, g = _data(signal=1.0)
    sp = V3Spec("t", "RIDGE", ["ASRP"], 5, "select", "residual", alphas=(0.0, 0.5, 1.0), base_model="RIDGE")
    m = V3FoldModel(sp, _cfg(), 0, bc, g, [], "RIDGE").fit(X.iloc[:150], y[:150])
    t = m.trace(X.iloc[150:])
    assert not t["fallback"] and t["alpha"] > 0 and t["stage2_pred_sd"] > 0
    assert t["n_added_raw"] == 8 and t["n_added"] == 5


def test_no_added_columns_is_fallback():
    X, y, bc, _ = _data()
    sp = V3Spec("t", "RIDGE", ["ASRP"], 5, "pca", "residual", base_model="RIDGE")
    m = V3FoldModel(sp, _cfg(), 0, bc, {}, [], "RIDGE").fit(X.iloc[:150][bc], y[:150])
    t = m.trace(X.iloc[150:][bc])
    assert t["fallback"] and t["n_added"] == 0


def test_shuffled_control_changes_nothing_structurally():
    X, y, bc, g = _data(signal=1.0)
    sp = V3Spec("c", "RIDGE", ["ASRP"], 5, "pca", "early", control="permuted_features")
    m = V3FoldModel(sp, _cfg(), 0, bc, g, [], "RIDGE").fit(X.iloc[:150], y[:150])
    assert m.predict(X.iloc[150:]).shape == (50,)


def test_classify_and_verdicts():
    res = pd.DataFrame([
        {"name": "V2_BASELINE", "scheme": "repeated_kfold", "group": "comparator", "r2_mean": 0.2, "delta_r2": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "budget": None, "n_added": 0},
        {"name": "P21_SINGLE_X", "scheme": "repeated_kfold", "group": "single_family", "r2_mean": 0.2, "delta_r2": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "budget": 20, "n_added": 10},
        {"name": "P21_EARLY", "scheme": "repeated_kfold", "group": "fusion", "r2_mean": 0.18, "delta_r2": -0.02, "delta_ci_low": -0.04, "delta_ci_high": 0.0, "budget": 20, "n_added": 20},
        {"name": "P21_BUDGET_X_50", "scheme": "repeated_kfold", "group": "budget", "r2_mean": 0.2, "delta_r2": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "budget": 50, "n_added": 10},
        {"name": "P21_BUDGET_X_100", "scheme": "repeated_kfold", "group": "budget", "r2_mean": 0.2, "delta_r2": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "budget": 100, "n_added": 10},
    ])
    c = classify_v2_results(res)
    assert c.set_index("name").loc["P21_SINGLE_X", "suspect_fallback"]
    assert not c.set_index("name").loc["P21_EARLY", "suspect_fallback"]
    cap = c.attrs["budget_cap"]
    assert bool(cap.set_index("family").loc["X", "capped"])
    audit = pd.DataFrame([
        {"audited": "A", "scheme": "repeated_kfold", "variant": "reproduce", "status": "fallback_baseline", "delta_r2": 0.0},
        {"audited": "A", "scheme": "repeated_kfold", "variant": "forced_alpha_1", "status": "ok", "delta_r2": 0.004},
        {"audited": "B", "scheme": "repeated_kfold", "variant": "reproduce", "status": "fallback_baseline", "delta_r2": 0.0},
        {"audited": "B", "scheme": "repeated_kfold", "variant": "forced_alpha_1", "status": "ok", "delta_r2": -0.01},
        {"audited": "C", "scheme": "repeated_kfold", "variant": "reproduce", "status": "ok", "delta_r2": -0.02},
        {"audited": "D", "scheme": "repeated_kfold", "variant": "reproduce", "status": "failed", "delta_r2": np.nan},
    ])
    v = verdicts(audit).set_index("audited")["verdict"]
    assert v["A"] == "positive_signal_discarded" and v["B"] == "silent_fallback"
    assert v["C"] == "genuine_negative" and v["D"] == "evaluation_error"


def test_repaired_task_ensemble_is_bounded():
    from wildspeech.models.residual_fusion import RobustTaskEnsemble

    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"SCREENING__all__s__value": rng.standard_normal(n),
                      "BEH__blending__x__value": rng.standard_normal(n) * 1e3,          # wild-scale task block
                      "__avail__blending": (rng.random(n) > 0.3).astype(float)})
    y = X["SCREENING__all__s__value"] * 5 + 100 + rng.standard_normal(n)
    m = RobustTaskEnsemble(base_cols=["SCREENING__all__s__value"], tasks=("blending",)).fit(X.iloc[:90], y[:90])
    Xt = X.iloc[90:].copy()
    Xt["BEH__blending__x__value"] *= 1e6                                                  # extreme validation inputs
    p = m.predict(Xt)
    assert np.isfinite(p).all() and p.min() >= m.y_lo_ and p.max() <= m.y_hi_
    w = m.weights_.to_numpy()
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-6
