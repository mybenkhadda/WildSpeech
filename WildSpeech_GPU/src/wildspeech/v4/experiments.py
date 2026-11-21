"""V4 experiment programme on identical student-disjoint folds (frozen V2 validation registry).

Model ladder (V3Spec; every fusion weight / shrinkage / reducer fitted inside the training fold):
  M0  screening / tabular baselines: RIDGE, GAM (reference comparator), CATBOOST, LIGHTGBM on
      SCREENING+BEHAVIOR+ITEM (+ a screening-only ridge)
  M1  baseline + ASR recognized-text families (ASRTXT, ASRTASK [+ASRUNC] [+ASRDIS])       early fusion
  M2  baseline + acoustic (ACOU)                                                       early fusion
  M3  baseline + measured phonetics (ASRPHN)                     only when phone predictions exist
  M4  baseline + text + acoustic (+ phonetics)                                         early fusion
  M5  residual correction: speech families fitted to inner-OOF baseline residuals, shrinkage by inner CV
  M6  late fusion (convex weights on inner-OOF per-family predictions) and regularized stacking
  M7  conditional mixture of experts (reliability-gated experts; domain partial pooling)
  M8  compact hierarchical multimodal network — only when an M1-M7 row is a GENUINE_GAIN (gate)
Every row reports ΔR² against the reference baseline (paired bootstrap on repeat-averaged OOF predictions) and
against the same-model M0 (feature contribution with the estimator held fixed), the per-fold trace (fallbacks,
alpha, weights, effective features) and a result class (evaluation.ablation.classify_results). A fallback is
never a multimodal improvement: such rows are BASELINE_FALLBACK and are not eligible as champion.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import ID
from ..evaluation.ablation import add_inference, classify_results
from ..evaluation.nested_asr_cv import V3Result, V3Spec, result_row
from . import FAM_ACOU, FAM_DIS, FAM_PHON, FAM_REL, FAM_TASK, FAM_TEXT, FAM_UNC


# ================================================================================ spec factory
class SpecFactory:
    def __init__(self, cfg, providers: dict, baseline: dict):
        v4 = cfg.get_path("v4", {}) or {}
        self.cfg, self.providers, self.baseline = cfg, providers, baseline
        self.budget, self.method = int(v4.get("budget", 20)), str(v4.get("method", "pca"))
        self.alphas = tuple(float(a) for a in v4.get("residual_alphas", [0, 0.25, 0.5, 0.75, 1.0]))
        self.models_early = list(v4.get("models_early", ["RIDGE", "CATBOOST"]))
        self.base_models = list(v4.get("baseline_models", ["RIDGE", "GAM", "CATBOOST"]))
        self.base_fams = list(baseline["families"])
        self.gate_fams = list((v4.get("moe", {}) or {}).get("gate_families", [FAM_REL]))

    def have(self, *names) -> list[str]:
        return [n for n in names if n in self.providers and self.providers[n].available()]

    def spec(self, name, providers, model="RIDGE", fusion="early", group="multimodal", experiment="", budget=None,
             method=None, control=None, params=None, base_model=None, base_families=None) -> V3Spec:
        return V3Spec(name=name, model=model, providers=list(providers), budget=self.budget if budget is None else budget,
                      method=(self.method if providers else "none") if method is None else method, fusion=fusion,
                      alphas=self.alphas if fusion == "residual" else None, control=control,
                      base_families=list(base_families or self.base_fams), base_model=base_model,
                      params=dict(params or {}), group=group, experiment=experiment or name.split("_")[0])

    # ------------------------------------------------------------------ families
    def text_set(self) -> list[str]:
        return self.have(FAM_TEXT, FAM_TASK, FAM_UNC, FAM_DIS)

    def speech_set(self) -> list[str]:
        return self.text_set() + self.have(FAM_ACOU, FAM_PHON)

    # ------------------------------------------------------------------ ladders
    def baselines(self) -> list[V3Spec]:
        out = [self.spec(f"M0_{m}", [], model=m, group="baseline" if m == self.baseline["model"] else "baseline_alt",
                         experiment="M0", budget=None) for m in self.base_models]
        if not any(s.model == self.baseline["model"] for s in out):
            out.insert(0, self.spec(f"M0_{self.baseline['model']}", [], model=self.baseline["model"], group="baseline",
                                    experiment="M0", budget=None))
        out.append(self.spec("M0_SCREENING_RIDGE", [], model="RIDGE", group="baseline_alt", experiment="M0", budget=None,
                             base_families=["SCREENING", "RECORDING"]))
        return out

    def ladder(self) -> list[V3Spec]:
        S: list[V3Spec] = []
        txt, sp = self.text_set(), self.speech_set()
        ac, ph = self.have(FAM_ACOU), self.have(FAM_PHON)
        for m in self.models_early:
            if txt:
                S.append(self.spec(f"M1_TEXT_{m}", txt, model=m, experiment="M1"))
            if ac:
                S.append(self.spec(f"M2_ACOU_{m}", ac, model=m, experiment="M2"))
            if ph:
                S.append(self.spec(f"M3_PHON_{m}", ph, model=m, experiment="M3"))
            if txt and ac:
                S.append(self.spec(f"M4_TEXT_ACOU_{m}", sp, model=m, experiment="M4"))
        bm = self.baseline["model"]
        for nm, fams in (("TEXT", txt), ("ACOU", ac), ("PHON", ph), ("ALL", sp)):
            if fams:
                S.append(self.spec(f"M5_RESID_{nm}", fams, model="RIDGE", fusion="residual", base_model=bm, experiment="M5"))
        if sp:
            S.append(self.spec("M6_LATE_ALL", sp, model="RIDGE", fusion="late", experiment="M6"))
            S.append(self.spec("M6_STACK_ALL", sp, model="RIDGE", fusion="stack", experiment="M6"))
            gate = self.have(*self.gate_fams)
            S.append(self.spec("M7_GATED_MOE", sp + gate, model="RIDGE", fusion="moe_gated", base_model=bm,
                               params={"gate_families": gate}, experiment="M7"))
            S.append(self.spec("M7_DOMAIN_MOE", sp, model="RIDGE", fusion="domain_moe", experiment="M7"))
        return S

    def hier_net(self) -> list[V3Spec]:
        sp = self.speech_set()
        gate = self.have(*self.gate_fams)
        return [self.spec("M8_HIER_NET", sp + gate, model="RIDGE", fusion="hier_net", params={"gate_families": gate},
                          experiment="M8")] if sp else []

    # ------------------------------------------------------------------ ablations / controls
    def ablations(self, task_providers: dict, model_providers: dict, loo_model_providers: dict,
                  correct_providers: list[str]) -> list[V3Spec]:
        A: list[V3Spec] = []
        sp = self.speech_set()
        for f in self.have(FAM_TEXT, FAM_TASK, FAM_UNC, FAM_DIS, FAM_PHON, FAM_ACOU, FAM_REL):
            A.append(self.spec(f"ABL_FAMILY_{f}", [f], group="family", experiment="ABL"))
        if len(sp) > 1:
            for f in sp:
                A.append(self.spec(f"ABL_LOO_{f}", [x for x in sp if x != f], group="leave_one_out", experiment="ABL"))
        for t, names in task_providers.items():
            ok = self.have(*names)
            if ok:
                A.append(self.spec(f"ABL_TASK_{t}", ok, group="task", experiment="H2"))
        for m, n in model_providers.items():
            if self.have(n):
                A.append(self.spec(f"ABL_ASRMODEL_{m}", [n], group="asr_model", experiment="ABL"))
        for m, n in loo_model_providers.items():
            if self.have(n):
                A.append(self.spec(f"ABL_ASRMODEL_LOO_{m}", [n], group="asr_model_loo", experiment="ABL"))
        cor = self.have(*correct_providers)
        if cor:
            A.append(self.spec("H3_CORRECT_ONLY", cor, group="correct_only", experiment="H3"))
            A.append(self.spec("H3_CORRECT_ONLY_RESID", cor, model="RIDGE", fusion="residual",
                               base_model=self.baseline["model"], group="correct_only", experiment="H3"))
        if self.have(FAM_DIS):
            A.append(self.spec("H4_DISAGREEMENT_RESID", [FAM_DIS], model="RIDGE", fusion="residual",
                               base_model=self.baseline["model"], group="disagreement", experiment="H4"))
        # nonlinear interactions (H9): same inputs, linear vs boosted trees
        if sp:
            for m in ("RIDGE", "CATBOOST", "LIGHTGBM"):
                A.append(self.spec(f"H9_INTERACT_{m}", sp, model=m, group="interaction", experiment="H9"))
        # budgets / reducers
        for b in (self.cfg.get_path("v4.validation.budgets", [5, 10, 20, 40]) or []):
            for meth in ("pca", "pls", "select"):
                if sp:
                    A.append(self.spec(f"ABL_BUDGET_{meth}_{b}", sp, budget=int(b), method=meth, group="budget",
                                       experiment="ABL"))
        return A

    def controls(self) -> list[V3Spec]:
        C: list[V3Spec] = []
        txt, sp = self.text_set(), self.speech_set()
        asr = [f for f in sp if f.startswith("ASR")]
        if asr:
            C.append(self.spec("CTRL_SHUFFLED_ASR", sp, control="permuted_families:" + "+".join(asr), group="control",
                               experiment="CTRL"))
        if sp:
            C.append(self.spec("CTRL_SHUFFLED_OUTCOME", sp, control="shuffled_outcome", group="control", experiment="CTRL"))
            for f in sp:
                C.append(self.spec(f"ROB_MISSING_{f}", sp, control=f"missing_modality:{f}", group="missing_modality",
                                   experiment="ROB"))
        if self.have("REC"):
            C.append(self.spec("CTRL_RECORDING_ONLY", ["REC"], group="control", experiment="CTRL"))
            if txt:
                C.append(self.spec("H6_TEXT_PLUS_REC", txt + ["REC"], group="domain", experiment="H6"))
            if self.have(FAM_ACOU):
                C.append(self.spec("H6_ACOU_PLUS_REC", [FAM_ACOU, "REC"], group="domain", experiment="H6"))
        return C


# ================================================================================ runner
class ExperimentRunner:
    """Evaluates specs on a scheme with the V4Evaluator; resumable (V3Evaluator disk cache keyed by spec,
    provider fingerprints, folds and config)."""

    def __init__(self, ev, logger, n_boot: int = 1000):
        self.ev, self.log, self.n_boot = ev, logger, int(n_boot)
        self.results: dict[tuple[str, str], V3Result] = {}
        self.base: dict[str, V3Result] = {}
        self.same_model_base: dict[tuple[str, str], V3Result] = {}

    def reference(self, scheme: str, spec: V3Spec) -> V3Result:
        if scheme not in self.base:
            self.base[scheme] = self.ev.evaluate(spec, scheme)
            self.results[(spec.name, scheme)] = self.base[scheme]
        return self.base[scheme]

    def run(self, specs: list[V3Spec], scheme: str, ref_spec: V3Spec, m0_by_model: dict[str, V3Spec] | None = None
            ) -> pd.DataFrame:
        base = self.reference(scheme, ref_spec)
        rows = []
        for i, sp in enumerate(specs):
            t0 = time.perf_counter()
            try:
                r = self.results.get((sp.name, scheme)) or self.ev.evaluate(sp, scheme, baseline=base)
            except Exception as e:  # noqa: BLE001 - recorded, never replaced by baseline predictions
                rows.append({"name": sp.name, "experiment": sp.experiment, "group": sp.group, "scheme": scheme,
                             "status": "failed", "reason": f"{type(e).__name__}: {e}"})
                continue
            self.results[(sp.name, scheme)] = r
            ok = r.status not in ("failed", "unavailable", "no_folds")
            d = self.ev.compare(base, r, self.n_boot) if ok and sp.name != ref_spec.name else {}
            row = result_row(r, d)
            if m0_by_model and ok and sp.providers and sp.model in m0_by_model:
                key = (m0_by_model[sp.model].name, scheme)
                if key not in self.results:
                    self.results[key] = self.ev.evaluate(m0_by_model[sp.model], scheme)
                same = self.results[key]
                ds = self.ev.compare(same, r, self.n_boot) if not same.oof.empty else {}
                row.update({"same_model_baseline": same.spec.name,
                            **{f"{k}_same_model": v for k, v in ds.items()}})
            row["seconds_wall"] = time.perf_counter() - t0
            rows.append(row)
            if self.log:
                self.log.info("V4 [%d/%d] %s %s: status=%s r2=%.4f dR2=%s", i + 1, len(specs), scheme, sp.name, r.status,
                              float(r.summary.get("r2_mean") or np.nan), d.get("delta_r2"))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ exports
    def oof_long(self) -> pd.DataFrame:
        parts = [r.oof.assign(name=n, scheme=s) for (n, s), r in self.results.items() if r.oof is not None and not r.oof.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def traces(self) -> pd.DataFrame:
        parts = []
        for (n, s), r in self.results.items():
            if r.trace is not None and not r.trace.empty:
                parts.append(r.trace.assign(name=n, scheme=s))
        if not parts:
            return pd.DataFrame()
        t = pd.concat(parts, ignore_index=True)
        return t.astype({c: str for c in t.columns if t[c].dtype == object})


def finalize(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if rows is None or rows.empty:
        return rows
    r = add_inference(rows, alpha)
    return classify_results(r, alpha)


# ================================================================================ fallback report
def fallback_report(rows: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    """Explicit report of every way a multimodal row can silently equal the baseline."""
    if rows is None or rows.empty:
        return pd.DataFrame()
    out = []
    for r in rows.itertuples(index=False):
        t = traces[(traces["name"] == r.name) & (traces["scheme"] == r.scheme)] if traces is not None and not traces.empty else pd.DataFrame()

        def num(col):
            return pd.to_numeric(t[col], errors="coerce") if col in t else pd.Series(dtype=float)

        reasons = t["fallback_reason"].replace("", np.nan).dropna().value_counts().to_dict() if "fallback_reason" in t else {}
        out.append({"name": r.name, "scheme": r.scheme, "status": r.status, "result_class": getattr(r, "result_class", None),
                    "n_folds": len(t), "fallback_folds": int(t["fallback"].astype(str).isin(["True", "true", "1"]).sum()) if "fallback" in t else None,
                    "zero_effective_features_folds": int((num("n_added") == 0).sum()),
                    "zero_raw_features_folds": int((num("n_added_raw") == 0).sum()),
                    "zero_variance_added_mean": float(num("added_zero_var").mean()) if len(t) else np.nan,
                    "zero_correction_variance_folds": int((num("stage2_pred_sd") == 0).sum()),
                    "alpha_zero_folds": int((num("alpha") == 0).sum()),
                    "zero_fusion_weight_folds": int(t["fallback_reason"].astype(str).str.contains("zero weight|weights set to 0").sum()) if "fallback_reason" in t else 0,
                    "feature_delivery_failure": r.status in ("no_added_features", "unavailable"),
                    "identical_to_baseline": getattr(r, "identical_to_baseline", None),
                    "fallback_reasons": "; ".join(f"{k} x{v}" for k, v in reasons.items()),
                    "is_fallback": (r.status in ("fallback_baseline", "no_added_features")) or bool(reasons)})
    return pd.DataFrame(out)


# ================================================================================ champion
def select_champion(rows: pd.DataFrame, scheme: str, ref_name: str, submittable: set[str], require_gain: bool = True
                    ) -> dict:
    """Highest repeated-CV R² among GENUINE_GAIN multimodal rows without any fallback fold whose providers all
    have test-time coverage; otherwise the validated reference baseline. Never a control / ablation row."""
    r = rows[(rows["scheme"] == scheme)].copy()
    base = r[r["name"] == ref_name]
    cand = r[r["group"].isin(["multimodal"]) & r["control"].isna()]
    cand = cand[cand["status"] == "ok"]
    if require_gain:
        cand = cand[cand.get("result_class", pd.Series(index=cand.index, dtype=str)) == "GENUINE_GAIN"]
    cand = cand[cand["providers"].fillna("").map(lambda p: all(x in submittable for x in str(p).split("+") if x))]
    if not cand.empty:
        best = cand.sort_values(["r2_mean", "delta_r2"], ascending=False).iloc[0]
        return {"name": best["name"], "reason": "highest repeated-CV R² among GENUINE_GAIN rows with no fallback fold",
                "r2_mean": float(best["r2_mean"]), "delta_r2": float(best.get("delta_r2", np.nan)),
                "delta_ci_low": float(best.get("delta_ci_low", np.nan)), "delta_ci_high": float(best.get("delta_ci_high", np.nan)),
                "is_baseline": False}
    b = base.iloc[0] if not base.empty else None
    return {"name": ref_name, "reason": "no multimodal row is a validated GENUINE_GAIN with test coverage: the "
                                        "validated reference baseline is the champion (this is NOT a multimodal result)",
            "r2_mean": float(b["r2_mean"]) if b is not None else np.nan, "delta_r2": 0.0, "is_baseline": True}


def stability_table(oof: pd.DataFrame, names: list[str], scheme: str = "repeated_kfold") -> pd.DataFrame:
    """Per-fold R² distribution (fold stability) for the named rows."""
    from ..stats import r2

    rows = []
    d = oof[(oof["scheme"] == scheme) & oof["name"].isin(names)]
    for (n, rep, f), g in d.groupby(["name", "repeat", "fold"]):
        rows.append({"name": n, "repeat": rep, "fold": f, "r2": r2(g["y_true"], g["y_pred"]), "n": len(g)})
    return pd.DataFrame(rows)


def paired_fold_deltas(oof: pd.DataFrame, ref: str, names: list[str], scheme: str = "repeated_kfold") -> pd.DataFrame:
    """Per-fold paired ΔR² (same validation students) against the reference."""
    from ..stats import r2

    d = oof[oof["scheme"] == scheme]
    b = d[d["name"] == ref]
    rows = []
    for n in names:
        m = d[d["name"] == n].merge(b, on=[ID, "repeat", "fold"], suffixes=("", "_b"))
        for (rep, f), g in m.groupby(["repeat", "fold"]):
            rows.append({"name": n, "repeat": rep, "fold": f, "delta_r2": r2(g["y_true"], g["y_pred"]) - r2(g["y_true"], g["y_pred_b"])})
    return pd.DataFrame(rows)
