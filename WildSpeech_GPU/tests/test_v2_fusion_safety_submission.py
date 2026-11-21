"""Residual stacking, convex fusion, repaired task ensemble with missing modalities, compact reduction,
prediction safety / quarantine and submission correctness."""
import numpy as np
import pandas as pd
import pytest

from wildspeech.diagnostics.prediction_safety import PredictionSafetyChecker
from wildspeech.features.compact_representation import FamilyReducer, IndexTransformer, spearman_vector
from wildspeech.models.residual_fusion import (ConvexLateFusion, DomainPartialPooling, NestedResidualRegressor, RegularizedStack,
                                               RobustTaskEnsemble, convex_weights)
from wildspeech.submission import validate_submission

TASKS = ("blending", "deletion", "sentence_repetition")


def _toy(n=200, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"SCREENING__a": rng.normal(size=n), "SCREENING__b": rng.normal(size=n)})
    for t in TASKS:
        X[f"HARM__{t}__x__mean"] = rng.normal(size=n)
    X["SOFT__all__p__value"] = rng.normal(size=n)
    y = 3 * X["SCREENING__a"] + signal * X["HARM__deletion__x__mean"] + 0.5 * rng.normal(size=n)
    return X, y.to_numpy()


def test_convex_weights_simplex():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 3))
    y = 0.7 * Z[:, 0] + 0.3 * Z[:, 2] + 0.01 * rng.normal(size=300)
    w, b = convex_weights(Z, y)
    assert np.all(w >= -1e-9) and abs(w.sum() - 1) < 1e-6 and w[1] < 0.05


def test_nested_residual_shrinks_noise_and_uses_signal():
    X, y = _toy(signal=0.0)
    m = NestedResidualRegressor(stage1_cols=["SCREENING__a", "SCREENING__b"]).fit(X.iloc[:150], y[:150])
    assert m.alpha_ in (0.0, 0.25, 0.5)
    X2, y2 = _toy(signal=2.0, seed=1)
    m2 = NestedResidualRegressor(stage1_cols=["SCREENING__a", "SCREENING__b"]).fit(X2.iloc[:150], y2[:150])
    assert m2.alpha_ >= 0.5
    base_only = NestedResidualRegressor(stage1_cols=["SCREENING__a", "SCREENING__b"], alphas=(0.0,)).fit(X2.iloc[:150], y2[:150])
    e_full = np.mean((m2.predict(X2.iloc[150:]) - y2[150:]) ** 2)
    e_base = np.mean((base_only.predict(X2.iloc[150:]) - y2[150:]) ** 2)
    assert e_full < e_base


def test_late_fusion_and_stack_are_convex():
    X, y = _toy(signal=2.0)
    lf = ConvexLateFusion(base_cols=["SCREENING__a", "SCREENING__b"]).fit(X, y)
    assert abs(lf.weights_.sum() - 1) < 1e-6 and (lf.weights_ >= -1e-9).all()
    st = RegularizedStack().fit(X, y)
    assert abs(np.sum(st.weights_) - 1) < 1e-6 and np.isfinite(st.predict(X)).all()


def test_repaired_task_ensemble_handles_missing_task_and_extrapolation():
    X, y = _toy(signal=2.0)
    for t in TASKS:
        X[f"__avail__{t}"] = 1.0
    X.loc[X.index[:40], "__avail__deletion"] = 0.0             # 40 students without the deletion task
    X.loc[X.index[:40], "HARM__deletion__x__mean"] = 0.0
    m = RobustTaskEnsemble(base_cols=["SCREENING__a", "SCREENING__b"], tasks=TASKS).fit(X.iloc[:160], y[:160])
    Xt = X.iloc[160:].copy()
    Xt.iloc[0, Xt.columns.get_loc("HARM__blending__x__mean")] = 1e6     # wild extrapolation in one feature
    p = m.predict(Xt)
    assert np.isfinite(p).all()
    assert p.max() <= m.y_hi_ + 1e-9 and p.min() >= m.y_lo_ - 1e-9
    assert abs(m.weights_.sum() - 1) < 1e-6


def test_domain_partial_pooling_runs():
    X, y = _toy()
    X["RECORDING__orig_sr_low"] = (np.arange(len(X)) % 3 == 0).astype(float)
    m = DomainPartialPooling(["RECORDING__orig_sr_low"], min_domain_size=30).fit(X, y)
    assert m.lambda_ in (0.0, 0.25, 0.5, 0.75, 1.0) and np.isfinite(m.predict(X)).all()


def test_family_reducer_budget_and_fold_fit():
    X, y = _toy(n=120)
    for k in range(30):
        X[f"PHON__blending__f{k}__mean"] = np.random.default_rng(k).normal(size=len(X))
    groups = {"HARM": [c for c in X if c.startswith("HARM")], "PHON": [c for c in X if c.startswith("PHON")]}
    red = FamilyReducer(groups, budget=5, family_components=4).fit(X.iloc[:80], y[:80])
    A = red.transform(X.iloc[80:])
    assert A.shape == (40, 5) and np.isfinite(A.to_numpy()).all()
    s = spearman_vector(X[["SCREENING__a"]].to_numpy(), y)
    assert s[0] > 0.5


def test_index_transformer_signs_validated_on_training_fold():
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({"PHON__blending__per__mean": rng.normal(size=n), "PHON__deletion__per__mean": rng.normal(size=n)})
    y = -X["PHON__blending__per__mean"] - X["PHON__deletion__per__mean"] + 0.3 * rng.normal(size=n)
    it = IndexTransformer({"PA": {"include": ["PHON__*__per__mean"]}}, 2).fit(X, y)
    idx = it.transform(X)["IDX__all__PA__value"]
    assert np.corrcoef(idx, y)[0, 1] > 0.8                      # error rates flipped to a positive direction
    assert (it.report_["status"] == "ok").all()


def test_prediction_safety_and_quarantine():
    y = np.random.default_rng(0).normal(100, 10, 300)
    chk = PredictionSafetyChecker(y, _Cfg())
    good = pd.DataFrame({"subject_id": [str(i) for i in range(300)], "repeat": 0, "fold": np.arange(300) % 5,
                         "y_true": y, "y_pred": y + np.random.default_rng(1).normal(0, 5, 300)})
    ok, bad = chk.deployable(chk.check(good, None, 300))
    assert ok and not bad
    broken = good.copy()
    broken.loc[3, "y_pred"] = 5e4                                # one catastrophic prediction
    broken.loc[4, "y_pred"] = np.nan
    ok, bad = chk.deployable(chk.check(broken, None, 300))
    assert not ok and "finite_oof" in bad and "extreme_fraction_oof" in bad


def test_submission_validation_v2_rules():
    fmt = pd.DataFrame({"subject_id": ["a", "b", "c"], "outcome_score": 0.0})
    assert (validate_submission(pd.DataFrame({"subject_id": ["a", "b", "c"], "outcome_score": [1.0, 2.0, 3.0]}), fmt)["status"] == "PASS").all()
    const = validate_submission(pd.DataFrame({"subject_id": ["a", "b", "c"], "outcome_score": [1.0, 1.0, 1.0]}), fmt).set_index("check")
    assert const.loc["non-constant predictions", "status"] == "FAIL"


def test_candidate_selection_skips_quarantined():
    from wildspeech.models.champion_selection import ensemble_members, select_candidates

    res = pd.DataFrame([
        {"name": "A", "r2_mean": 0.25, "r2_sd": 0.01, "delta_ci_low": 0.01, "providers": "HARM", "fusion": "residual"},
        {"name": "B", "r2_mean": 0.30, "r2_sd": 0.01, "delta_ci_low": 0.02, "providers": "HARM", "fusion": "residual"},
        {"name": "V2_BASELINE", "r2_mean": 0.21, "r2_sd": 0.0, "delta_ci_low": 0.0, "providers": "", "fusion": "early"}])
    chosen = select_candidates(res, {}, {"A": True, "B": False, "V2_BASELINE": True})
    assert chosen["harmonized_acoustic"] == "A" and chosen["best_tabular"] == "V2_BASELINE"
    assert ensemble_members(res, chosen) == ["A"]


class _Cfg:
    def get_path(self, key, default=None):
        return {"v2.failure.safety": {"max_abs_z": 6.0, "max_extreme_frac": 0.0, "range_margin_sd": 1.0,
                                      "min_fold_r2": -0.1, "max_repeat_sd": 0.05}}.get(key, default)


def test_budgeted_fold_model_on_synthetic_frames(ctx):
    from wildspeech.models.compact_regression import BudgetedFoldModel, CompactSpec

    X, y = _toy(n=120)
    for t in TASKS:
        X[f"BEHAVIOR__{t}__n_items__value"] = 3.0
    for fusion in ("early", "residual", "late", "task_pool", "stack"):
        spec = CompactSpec(f"t_{fusion}", "RIDGE", providers=["HARM", "SOFT"], budget=3, fusion=fusion, stack_models=("RIDGE", "GAM"))
        m = BudgetedFoldModel(spec, ctx.cfg, 0, ["SCREENING__a", "SCREENING__b"],
                              {"HARM": [c for c in X if c.startswith("HARM")], "SOFT": ["SOFT__all__p__value"]}, list(TASKS), "RIDGE")
        p = m.fit(X.iloc[:90], y[:90]).predict(X.iloc[90:])
        assert p.shape == (30,) and np.isfinite(p).all() and m.n_added_ == 3
    with pytest.raises(ValueError):
        BudgetedFoldModel(CompactSpec("bad", fusion="nope"), ctx.cfg, 0, ["SCREENING__a"], {}, list(TASKS), "RIDGE").fit(X, y)
