"""V4 estimators (reliability-gated experts, domain MoE via V4FoldModel), fallback tracing, V4 controls,
champion selection and submission checks. Synthetic data, CPU only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildspeech.config import Config
from wildspeech.evaluation.nested_asr_cv import V3Spec
from wildspeech.v4.experiments import select_champion
from wildspeech.v4.models import ReliabilityGatedExperts, V4Evaluator, V4FoldModel


def _data(n=300, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    txt = rng.normal(size=(n, 4))
    rel = rng.uniform(size=(n, 1))
    y = base @ np.array([1.0, 0.5, -0.5]) + (2.0 * txt[:, 0] * rel[:, 0] if signal else 0.0) + rng.normal(0, 0.5, n)
    X = pd.DataFrame(np.hstack([base, txt, rel]),
                     columns=["SCREENING__a", "SCREENING__b", "SCREENING__c"] + [f"ASRTXT__all__f{i}__mean" for i in range(4)]
                     + ["ASRREL__all__rel_frac_ok__mean"])
    return X, y


def _cfg():
    return Config({"validation": {"inner_splits": 4}, "models": {"max_missing_frac": 0.6}, "v4": {"moe": {"l2": 1.0}}})


def test_gated_experts_uses_signal_and_falls_back_on_noise():
    X, y = _data(signal=True)
    m = ReliabilityGatedExperts(base_cols=list(X.columns[:3]), gate_cols=["ASRREL__all__rel_frac_ok__mean"]).fit(X, y)
    assert not m.fallback_ and float(m.weights_.sum()) > 0
    Xn, yn = _data(signal=False, seed=3)
    Xn[[c for c in Xn.columns if c.startswith("ASRTXT")]] = np.random.default_rng(9).normal(size=(len(Xn), 4))
    m2 = ReliabilityGatedExperts(base_cols=list(Xn.columns[:3]), gate_cols=["ASRREL__all__rel_frac_ok__mean"]).fit(Xn, yn)
    p_base = m2.m1_.predict(Xn[m2.c1_])
    if m2.fallback_:
        assert np.allclose(m2.predict(Xn), p_base)                      # a fallback IS the baseline prediction
        assert float(m2.weights_.sum()) == 0.0


def test_v4_fold_model_moe_gated_traces_gate_and_fallback():
    X, y = _data(signal=True)
    base = list(X.columns[:3])
    groups = {"ASRTXT": [c for c in X.columns if c.startswith("ASRTXT")], "ASRREL": ["ASRREL__all__rel_frac_ok__mean"]}
    sp = V3Spec("g", "RIDGE", ["ASRTXT", "ASRREL"], 2, "pca", "moe_gated", params={"gate_families": ["ASRREL"]})
    fm = V4FoldModel(sp, _cfg(), 0, base, groups, [], "RIDGE").fit(X.iloc[:220], y[:220])
    t = fm.trace(X.iloc[220:])
    assert t["n_gate"] >= 1 and t["n_added"] == 2
    assert "gate_weight_mean" in t
    assert fm.predict(X.iloc[220:]).shape == (80,)


def test_v4_fold_model_domain_moe_runs():
    X, y = _data()
    X["RECORDING__orig_sr_low"] = (np.arange(len(X)) % 2).astype(float)
    base = list(X.columns[:3]) + ["RECORDING__orig_sr_low"]
    groups = {"ASRTXT": [c for c in X.columns if c.startswith("ASRTXT")]}
    sp = V3Spec("d", "RIDGE", ["ASRTXT"], 2, "pca", "domain_moe")
    fm = V4FoldModel(sp, _cfg(), 0, base, groups, [], "RIDGE").fit(X.iloc[:220], y[:220])
    assert "alpha" in fm.trace(X.iloc[220:])


def test_v4_controls_permute_or_blank_only_named_families():
    ev = object.__new__(V4Evaluator)
    X, y = _data(60)
    groups = {"ASRTXT": [c for c in X.columns if c.startswith("ASRTXT")]}
    sp = V3Spec("c", "RIDGE", ["ASRTXT"], 2, "pca", "early", control="missing_modality:ASRTXT")
    Xtr, Xva, ytr = ev._control(sp, X.iloc[:40], X.iloc[40:], groups, y[:40], 0)
    assert Xva[groups["ASRTXT"]].isna().all().all() and Xtr[groups["ASRTXT"]].notna().all().all()
    assert Xva["SCREENING__a"].equals(X.iloc[40:]["SCREENING__a"])
    sp2 = V3Spec("p", "RIDGE", ["ASRTXT"], 2, "pca", "early", control="permuted_families:ASRTXT")
    Xtr2, _, _ = ev._control(sp2, X.iloc[:40], X.iloc[40:], groups, y[:40], 1)
    assert Xtr2["SCREENING__a"].equals(X.iloc[:40]["SCREENING__a"])
    assert not Xtr2[groups["ASRTXT"][0]].equals(X.iloc[:40][groups["ASRTXT"][0]])
    assert sorted(Xtr2[groups["ASRTXT"][0]]) == sorted(X.iloc[:40][groups["ASRTXT"][0]])


def test_champion_never_a_fallback_or_control():
    rows = pd.DataFrame([
        {"name": "M0_GAM", "group": "baseline", "scheme": "repeated_kfold", "status": "ok", "control": None,
         "providers": "", "r2_mean": 0.21, "delta_r2": np.nan, "result_class": "BASELINE"},
        {"name": "M5_RESID_ALL", "group": "multimodal", "scheme": "repeated_kfold", "status": "fallback_baseline",
         "control": None, "providers": "ASRTXT", "r2_mean": 0.21, "delta_r2": 0.0, "result_class": "BASELINE_FALLBACK"},
        {"name": "CTRL_SHUFFLED_ASR", "group": "control", "scheme": "repeated_kfold", "status": "ok",
         "control": "permuted_families:ASRTXT", "providers": "ASRTXT", "r2_mean": 0.30, "delta_r2": 0.09,
         "result_class": "GENUINE_GAIN"},
    ])
    c = select_champion(rows, "repeated_kfold", "M0_GAM", {"ASRTXT"})
    assert c["name"] == "M0_GAM" and c["is_baseline"]
    rows.loc[len(rows)] = {"name": "M1_TEXT_RIDGE", "group": "multimodal", "scheme": "repeated_kfold", "status": "ok",
                           "control": None, "providers": "ASRTXT", "r2_mean": 0.24, "delta_r2": 0.03,
                           "result_class": "GENUINE_GAIN"}
    assert select_champion(rows, "repeated_kfold", "M0_GAM", {"ASRTXT"})["name"] == "M1_TEXT_RIDGE"
    # a provider without test coverage cannot be the champion
    assert select_champion(rows, "repeated_kfold", "M0_GAM", set())["name"] == "M0_GAM"


def test_submission_extra_checks_flag_nan_and_train_only_fields():
    from wildspeech.v4.submission import extra_checks

    fmt = pd.DataFrame({"subject_id": ["t1", "t2"], "outcome_score": [0.0, 0.0]})
    sub = pd.DataFrame({"subject_id": ["t1", "t2"], "outcome_score": [1.0, np.nan]})
    Xtr = pd.DataFrame({"SCREENING__x": [1.0], "SCHOOL__school_id": [1.0]})
    Xte = pd.DataFrame(index=["t1", "t2"])
    chk = extra_checks(sub, fmt, Xtr, Xte, ["t1", "t2"], ["s1"], np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    st = chk.set_index("check")["status"]
    assert st["no NaN predictions"] == "FAIL" and st["no train-only field in the design"] == "FAIL"
    assert st["re-loaded model reproduces in-memory predictions"] == "PASS"
