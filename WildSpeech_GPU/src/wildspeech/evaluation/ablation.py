"""ASR00-ASR12 experiment definitions and the budget-controlled runner (P28 / P29).

| id    | feature family                         | providers        |
|-------|----------------------------------------|------------------|
| ASR00 | tabular GAM baseline (comparator)      | -                |
| ASR01 | word transcripts                       | ASRW             |
| ASR02 | phoneme error features                 | ASRP             |
| ASR03 | expected-phoneme posterior confidence  | ASRC             |
| ASR04 | phonetic duration and fluency          | ASRT             |
| ASR05 | task-specific pronunciation errors     | ASRK             |
| ASR06 | correct-response-only features         | ASRO             |
| ASR07 | ASR model disagreement                 | ASRD             |
| ASR08 | compact pretrained embeddings          | ASRE (+ASRE_MIL) |
| ASR09 | phonetic consistency                   | ASRS             |
| ASR10 | all VALIDATED phonetic families        | rule below       |
| ASR11 | nested phonetic residual fusion        | ASR10 families   |
| ASR12 | regularized multimodal ensemble        | ASR10 families   |
Supplementary: ASRN (nested soft correctness), ASRX (confusion embeddings).
Validated-family rule (pre-specified, computed from runtime results of the primary scheme): single-family
status == ok (no fallback) and ΔR² > 0; if no family qualifies, ASR10-12 use every available family and the
table records `validation_rule = "none_validated_all_used"`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import v3
from .nested_asr_cv import V3Evaluator, V3Result, V3Spec, result_row

FAMILY_EXPERIMENTS = {"ASR01": [v3.FAM_WORD], "ASR02": [v3.FAM_PHON], "ASR03": [v3.FAM_CONF], "ASR04": [v3.FAM_TIME],
                      "ASR05": [v3.FAM_TASK], "ASR06": [v3.FAM_CORR], "ASR07": [v3.FAM_DIS], "ASR08": [v3.FAM_EMB],
                      "ASR09": [v3.FAM_CONS], "ASR13": [v3.FAM_ENC]}
SUPPLEMENTARY = {"ASR08_MIL": ["ASRE_MIL"], "ASRN_SOFT": [v3.FAM_SOFT], "ASRX_CONFUSION": [v3.FAM_CONFU],
                 "ASR13_W2V": ["ASRF_wav2vec2_base"], "ASR13_HUB": ["ASRF_hubert_base"],
                 "ASR13_WLM": ["ASRF_wavlm_base_plus"]}
DESCRIPTIONS = {"ASR00": "Tabular GAM baseline", "ASR01": "Word transcripts", "ASR02": "Phoneme error features",
                "ASR03": "Expected-phoneme posterior confidence", "ASR04": "Phonetic duration and fluency",
                "ASR05": "Task-specific pronunciation errors", "ASR06": "Correct-response-only features",
                "ASR07": "ASR model disagreement", "ASR08": "Compact pretrained embeddings", "ASR09": "Phonetic consistency",
                "ASR10": "All validated phonetic families", "ASR11": "Nested phonetic residual fusion",
                "ASR12": "Regularized multimodal ensemble", "ASR08_MIL": "Attention-MIL embeddings",
                "ASRN_SOFT": "Nested ASR soft correctness", "ASRX_CONFUSION": "Phoneme confusion embeddings",
                "ASR13": "Frozen self-supervised encoder embeddings (primary encoder)",
                "ASR13_W2V": "wav2vec2-base frozen embeddings", "ASR13_HUB": "HuBERT-base frozen embeddings",
                "ASR13_WLM": "WavLM-base-plus frozen embeddings",
                "ASR10_NO_PASKETTI": "All validated families without Pasketti-derived features (overlap sensitivity)"}
BASE_FUSION_MODEL = {"early": None, "late": "RIDGE", "task_pool": "RIDGE", "residual": "RIDGE", "stack": "RIDGE"}


def family_specs(exp: str, fams: list[str], fusions: list[str], budget: int, method: str, baseline_model: str,
                 alphas: tuple) -> list[V3Spec]:
    out = []
    for fu in fusions:
        model = baseline_model if fu == "early" else BASE_FUSION_MODEL.get(fu, "RIDGE")
        out.append(V3Spec(f"{exp}_{fu.upper()}", model=model, providers=list(fams), budget=budget,
                          method="raw" if fu == "task_pool" else method,
                          fusion=fu, alphas=alphas if fu == "residual" else None, group="family", experiment=exp))
    return out


def validated_families(rows: pd.DataFrame, available: list[str], scheme: str = "repeated_kfold") -> tuple[list[str], str]:
    r = rows[(rows["scheme"] == scheme) & (rows["group"] == "family")] if not rows.empty else rows
    ok = []
    for exp, fams in FAMILY_EXPERIMENTS.items():
        g = r[r["experiment"] == exp] if not r.empty else r
        if g.empty:
            continue
        good = g[(g["status"] == "ok") & (g["delta_r2"] > 0)]
        if not good.empty:
            ok += [f for f in fams if f in available]
    ok = sorted(set(ok))
    if ok:
        return ok, "single_family_ok_and_positive_delta"
    return sorted(available), "none_validated_all_used"


def combined_specs(fams: list[str], budget: int, method: str, baseline_model: str, alphas: tuple) -> list[V3Spec]:
    return [V3Spec("ASR10_EARLY", baseline_model, fams, budget, method, "early", group="combined", experiment="ASR10"),
            V3Spec("ASR11_RESIDUAL", "RIDGE", fams, budget, method, "residual", alphas=alphas, group="combined", experiment="ASR11"),
            V3Spec("ASR12_STACK", "RIDGE", fams, budget, method, "stack", group="combined", experiment="ASR12"),
            V3Spec("ASR12_LATE", "RIDGE", fams, budget, method, "late", group="combined", experiment="ASR12"),
            V3Spec("ASR12_TASK", "RIDGE", fams, budget, "raw", "task_pool", group="combined", experiment="ASR12")]


def model_specs(fams: list[str], models: list[str], budget: int, method: str, alphas: tuple) -> list[V3Spec]:
    out = []
    for m in models:
        out.append(V3Spec(f"ASR10_MODEL_{m}_EARLY", m, fams, budget, method, "early", group="model", experiment="ASR10"))
        out.append(V3Spec(f"ASR10_MODEL_{m}_RESIDUAL", m, fams, budget, method, "residual", alphas=alphas, group="model",
                          experiment="ASR11"))
    return out


def budget_specs(fams_by_exp: dict[str, list[str]], budgets: list[int], methods: list[str], alphas: tuple) -> list[V3Spec]:
    out = []
    for exp, fams in fams_by_exp.items():
        for meth in methods:
            for b in budgets:
                out.append(V3Spec(f"{exp}_B{b}_{meth.upper()}", "RIDGE", fams, int(b), meth, "residual", alphas=alphas,
                                  group="budget", experiment=exp))
    return out


def control_specs(fams: list[str], budget: int, method: str, alphas: tuple, controls: list[str], baseline_model: str) -> list[V3Spec]:
    out = []
    for c in controls:
        if c == "recording_only":
            out.append(V3Spec("CTRL_RECORDING_ONLY", "RIDGE", ["REC"], budget, method, "residual", alphas=alphas,
                              group="control", experiment="CTRL"))
        else:
            out.append(V3Spec(f"CTRL_{c.upper()}", "RIDGE", fams, budget, method, "residual", alphas=alphas, control=c,
                              group="control", experiment="CTRL"))
            out.append(V3Spec(f"CTRL_{c.upper()}_EARLY", baseline_model, fams, budget, method, "early", control=c,
                              group="control", experiment="CTRL"))
    return out


class AblationRunner:
    """Evaluates specs against the baseline on each scheme (paired deltas) under the experiment scheduler."""

    def __init__(self, ev: V3Evaluator, base: dict[str, V3Result], max_evals: int, time_budget_s: float, logger):
        from ..v2.scheduler import ExperimentScheduler

        self.ev, self.base = ev, base
        self.sched = ExperimentScheduler(int(max_evals), float(time_budget_s), logger)
        self.rows: list[dict] = []
        self.results: dict[tuple[str, str], V3Result] = {}

    def add(self, spec: V3Spec, schemes: list[str], priority: int) -> None:
        def task(spec=spec, schemes=tuple(schemes)):
            n = 0
            for sc in schemes:
                if sc not in self.base:
                    continue
                r = self.ev.evaluate(spec, sc, baseline=self.base[sc])
                d = self.ev.compare(self.base[sc], r) if r.status in ("ok", "partial_fallback", "fallback_baseline") else {}
                self.rows.append(result_row(r, d))
                self.results[(spec.name, sc)] = r
                n += 1
            return n
        self.sched.add(spec.name, task, priority, spec.group, cost=len(schemes))

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        sched = self.sched.run()
        return pd.DataFrame(self.rows), sched


def add_inference(rows: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """p from the paired-bootstrap CI (normal approximation), BH-FDR within (scheme, group)."""
    from scipy import stats as ss

    from ..stats import bh_fdr

    if rows.empty:
        return rows
    r = rows.copy()
    for col in ("delta_r2", "delta_ci_low", "delta_ci_high"):
        if col not in r:
            r[col] = np.nan

    def p_of(est, lo, hi):
        if not all(np.isfinite([est, lo, hi])) or hi <= lo:
            return np.nan
        se = (hi - lo) / (2 * 1.959964)
        return float(2 * ss.norm.sf(abs(est) / se))

    r["p"] = [p_of(*v) for v in r[["delta_r2", "delta_ci_low", "delta_ci_high"]].astype(float).to_numpy()] \
        if {"delta_r2", "delta_ci_low", "delta_ci_high"} <= set(r.columns) else np.nan
    r["q"] = np.nan
    for _, idx in r.groupby(["scheme", "group"]).groups.items():
        r.loc[idx, "q"] = bh_fdr(r.loc[idx, "p"].to_numpy(float))
    r["significant"] = (r["q"] < alpha) & (r["delta_ci_low"] > 0)
    r["reportable"] = ~r["status"].isin(["fallback_baseline", "no_added_features", "failed", "unavailable"])
    return r


RESULT_CLASSES = ("BASELINE", "NOT_REPORTABLE", "BASELINE_FALLBACK", "GENUINE_GAIN", "GENUINE_NEGATIVE",
                  "NO_DETECTABLE_EFFECT", "UNDETERMINED")


def classify_results(rows: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Label every ablation row so a genuine negative result is never confused with a silent fallback:
    BASELINE               the reference row itself
    BASELINE_FALLBACK      >= 1 fold fell back to the baseline (or status fallback_baseline / no_added_features):
                           its ΔR² says nothing about the features
    NOT_REPORTABLE         failed / unavailable
    GENUINE_GAIN           all folds used the features, ΔR² CI > 0 and q < alpha
    GENUINE_NEGATIVE       all folds used the features and ΔR² CI < 0 (the features hurt)
    NO_DETECTABLE_EFFECT   all folds used the features, CI spans 0
    UNDETERMINED           no CI (e.g. too few folds)"""
    if rows is None or rows.empty:
        return rows
    r = rows.copy()
    for c in ("status", "group", "fallback_folds", "delta_ci_low", "delta_ci_high", "q", "delta_r2"):
        if c not in r:
            r[c] = np.nan
    fb = pd.to_numeric(r["fallback_folds"], errors="coerce").fillna(0) > 0
    st = r["status"].astype(str)
    lo = pd.to_numeric(r["delta_ci_low"], errors="coerce")
    hi = pd.to_numeric(r["delta_ci_high"], errors="coerce")
    q = pd.to_numeric(r["q"], errors="coerce")
    cls = np.select(
        [r["group"].astype(str).eq("baseline"),
         st.isin(["failed", "unavailable"]),
         fb | st.isin(["fallback_baseline", "no_added_features"]),
         lo.isna() | hi.isna(),
         (lo > 0) & (q < alpha),
         hi < 0],
        ["BASELINE", "NOT_REPORTABLE", "BASELINE_FALLBACK", "UNDETERMINED", "GENUINE_GAIN", "GENUINE_NEGATIVE"],
        default="NO_DETECTABLE_EFFECT")
    r["result_class"] = cls
    return r
