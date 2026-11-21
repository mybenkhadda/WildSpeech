"""Section 25 - scientific hypothesis registry (H01-H20).

Each hypothesis is evaluated programmatically from executed artifacts only.
A hypothesis is never marked SUPPORTED on correlation or nominal significance
alone: support requires out-of-fold incremental evidence with a confidence
interval excluding zero and, where available, school-grouped confirmation.
Missing artifacts yield NOT_TESTED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HYPOTHESES = {
    "H01": "Audio adds predictive information beyond conventional screening",
    "H02": "Correct responses contain heterogeneous phonetic information",
    "H03": "Complexity sensitivity predicts future reading performance",
    "H04": "Within-student acoustic consistency predicts outcomes",
    "H05": "Different literacy tasks provide complementary information",
    "H06": "Specific phonetic error patterns have incremental relevance",
    "H07": "Distinct literacy profiles exist among similar screening scores",
    "H08": "Multiscale acoustic features contain additional signal",
    "H09": "Pretrained representations capture complementary information",
    "H10": "Acoustic features explain screening prediction residuals",
    "H11": "Useful acoustic signals generalize across schools",
    "H12": "Speech-related information outperforms technical artifacts",
    "H13": "Task-specific processing improves upon global aggregation",
    "H14": "Response variability is more informative than averages alone",
    "H15": "Acoustic characteristics interact with initial literacy ability",
    "H16": "Item-conditioned features outperform unadjusted acoustic features",
    "H17": "Composite literacy indices improve interpretability or prediction",
    "H18": "Adaptive assessment behavior contains complementary information",
    "H19": "Multimodal features recover identifiable baseline failure patterns",
    "H20": "Residual learning improves upon direct multimodal regression",
}


class Evidence:
    def __init__(self, ctx):
        self.ctx = ctx
        self._cache: dict = {}

    def table(self, name: str, stage: str) -> pd.DataFrame | None:
        if name not in self._cache:
            self._cache[name] = self.ctx.load_df(name, required=False) if self.ctx.stage_available(stage) else None
        return self._cache[name]


def _delta_status(delta, lo, hi, grouped=None) -> tuple[str, str]:
    if delta is None or not np.isfinite(delta):
        return "INCONCLUSIVE", "no estimate"
    if np.isfinite(lo) and lo > 0:
        if grouped is None or not np.isfinite(grouped):
            return "PARTIALLY_SUPPORTED", "CI excludes 0 under student CV; school-grouped confirmation unavailable"
        return ("SUPPORTED", "CI excludes 0 and school-grouped gain is positive") if grouped > 0 else (
            "PARTIALLY_SUPPORTED", "student-CV gain does not hold under school-grouped CV")
    if np.isfinite(hi) and hi <= 0:
        return "NOT_SUPPORTED", "CI upper bound <= 0"
    return "INCONCLUSIVE", "CI includes 0"


def _ablation_row(ev: Evidence, exp: str, scheme: str = "CV_A_REPEATED_STUDENT_KFOLD"):
    abl = ev.table("ablation_results", "20")
    if abl is None:
        return None
    model = ev.ctx.config["ablation_primary_model"]
    r = abl[(abl["experiment"] == exp) & (abl["scheme"] == scheme) & (abl["model"].isin([model, f"TWO_STAGE_{model}"]))]
    return r.iloc[0] if len(r) else None


def _ablation_test(ev, exp, ref="full_tabular"):
    a = _ablation_row(ev, exp)
    if a is None:
        return None
    b = _ablation_row(ev, exp, "CV_B_SCHOOL_GROUP_KFOLD")
    grouped = b[f"delta_r2_vs_{ref}"] if b is not None else np.nan
    status, why = _delta_status(a[f"delta_r2_vs_{ref}"], a[f"delta_r2_vs_{ref}_ci_lower"], a[f"delta_r2_vs_{ref}_ci_upper"], grouped)
    return {"observed_effect": f"delta R2 vs {ref} = {a[f'delta_r2_vs_{ref}']:.4f}",
            "uncertainty": f"95% CI [{a[f'delta_r2_vs_{ref}_ci_lower']:.4f}, {a[f'delta_r2_vs_{ref}_ci_upper']:.4f}]",
            "robustness": f"school-grouped delta = {grouped:.4f}" if np.isfinite(grouped) else "school-grouped CV unavailable",
            "support_status": status, "limitations": why}


def _registry_family(ev, families, pattern=None):
    reg = ev.table("feature_registry", "24")
    if reg is None:
        return None
    r = reg[reg["feature_family"].isin(families)]
    if pattern:
        r = r[r["feature_name"].str.contains(pattern, regex=True)]
    if r.empty:
        return None
    counts = r["selection_status"].value_counts()
    robust, promising = int(counts.get("ROBUST_SIGNAL", 0)), int(counts.get("PROMISING_SIGNAL", 0))
    status = "PARTIALLY_SUPPORTED" if robust else ("INCONCLUSIVE" if promising else "NOT_SUPPORTED")
    return {"observed_effect": f"{robust} robust and {promising} promising of {len(r)} features; max |residual rho|={r['residual_spearman'].abs().max():.3f}",
            "uncertainty": "bootstrap CIs and FDR in feature registry", "robustness": f"median school sign consistency {r['cross_school_sign_consistency'].median():.2f}",
            "support_status": status, "limitations": "feature-level evidence; selection-induced optimism possible; needs CV confirmation"}


def evaluate(ev: Evidence, hid: str) -> dict:
    cfg = ev.ctx.config
    res = None
    if hid == "H01":
        res = _ablation_test(ev, "A13_TABULAR_MULTIMODAL")
    elif hid == "H02":
        res = _registry_family(ev, ["TEMPORAL", "SPECTRAL", "MFCC", "PHONETIC", "WAVELETS"], r"__corr=1")
    elif hid == "H03":
        res = _registry_family(ev, ["COMPLEXITY_SENSITIVITY"])
    elif hid == "H04":
        res = _registry_family(ev, ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "ITEM_ADJUSTED_AUDIO"], r"__(std|mad|cv|iqr)$")
    elif hid == "H05":
        rel = ev.table("task_relevance_matrix", "12")
        if rel is not None and "residual_signal" in rel:
            n_tasks = int((rel["residual_n_fdr"] > 0).sum()) if "residual_n_fdr" in rel else 0
            res = {"observed_effect": f"{n_tasks} task families with FDR-significant residual features",
                   "uncertainty": "FDR within task", "robustness": f"school stability {rel['school_stability'].round(2).tolist()}",
                   "support_status": "PARTIALLY_SUPPORTED" if n_tasks >= 2 else "INCONCLUSIVE",
                   "limitations": "association-level evidence; complementarity in prediction tested in H13/ablation"}
    elif hid == "H06":
        res = _ablation_test(ev, "A11_TABULAR_PHONETIC")
    elif hid == "H07":
        prof = ev.table("cluster_profiles", "15")
        if prof is not None and "residual_mean" in prof:
            sub = prof[prof["space"] == "C09_SCREENING_ADJUSTED_AUDIO"]
            spread = float(sub["residual_mean"].max() - sub["residual_mean"].min()) if len(sub) else np.nan
            info = ev.ctx.load_json("cluster_space_info", required=False) or {}
            stable = (info.get("C09_SCREENING_ADJUSTED_AUDIO") or {}).get("stable")
            res = {"observed_effect": f"residual-mean spread across C09 clusters = {spread:.3f}",
                   "uncertainty": "bootstrap ARI stability", "robustness": f"stable={stable}",
                   "support_status": "PARTIALLY_SUPPORTED" if stable and np.isfinite(spread) and spread > 0 else "INCONCLUSIVE",
                   "limitations": "descriptive clusters; residual differences not tested out of fold"}
    elif hid == "H08":
        res = _ablation_test(ev, "A09_TABULAR_WAVELETS")
    elif hid == "H09":
        reg = ev.table("feature_registry", "24")
        if ev.ctx.stage_available("23") and reg is not None:
            res = _registry_family(ev, ["EMBEDDINGS"])
    elif hid == "H10":
        fam = ev.table("residual_family_ranking", "16")
        if fam is not None:
            audio = fam[fam["family"].isin(["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO"])]
            sig = audio[(audio["n_residual_fdr"] > 0) & (audio["mean_replication_rate"].fillna(0) >= 0.5)]
            res = {"observed_effect": f"{len(sig)} acoustic families with FDR-significant and replicated residual associations",
                   "uncertainty": "held-out half replication", "robustness": "within-school permutation for top features",
                   "support_status": "PARTIALLY_SUPPORTED" if len(sig) else "NOT_SUPPORTED" if len(audio) else "INCONCLUSIVE",
                   "limitations": "associations; predictive value is H01"}
    elif hid == "H11":
        b = _ablation_row(ev, "A13_TABULAR_MULTIMODAL", "CV_B_SCHOOL_GROUP_KFOLD")
        if b is not None:
            status, why = _delta_status(b["delta_r2_vs_full_tabular"], b["delta_r2_vs_full_tabular_ci_lower"], b["delta_r2_vs_full_tabular_ci_upper"], b["delta_r2_vs_full_tabular"])
            res = {"observed_effect": f"school-grouped delta R2 = {b['delta_r2_vs_full_tabular']:.4f}",
                   "uncertainty": f"CI [{b['delta_r2_vs_full_tabular_ci_lower']:.4f}, {b['delta_r2_vs_full_tabular_ci_upper']:.4f}]",
                   "robustness": f"generalization gap {b.get('school_generalization_gap', np.nan):.4f}", "support_status": status, "limitations": why}
    elif hid == "H12":
        nc = ev.table("negative_controls", "17")
        if nc is not None and "delta_r2_vs_screening" in nc:
            sp = nc[(nc["control"] == "SPEECH_FEATURES_REFERENCE") & (nc["variant"] == "with_screening")]
            te = nc[(nc["control"] == "NC07_TECHNICAL_FEATURES_ONLY") & (nc["variant"] == "with_screening")]
            if len(sp) and len(te):
                s, t = sp.iloc[0], te.iloc[0]
                ok = np.isfinite(s["delta_ci_lower"]) and s["delta_ci_lower"] > 0 and s["delta_r2_vs_screening"] > t["delta_r2_vs_screening"]
                res = {"observed_effect": f"speech delta={s['delta_r2_vs_screening']:.4f} vs technical delta={t['delta_r2_vs_screening']:.4f}",
                       "uncertainty": f"speech CI [{s['delta_ci_lower']:.4f}, {s['delta_ci_upper']:.4f}]",
                       "robustness": "shuffled controls NC05/NC06 reported separately",
                       "support_status": "SUPPORTED" if ok and t["delta_ci_lower"] <= 0 else ("PARTIALLY_SUPPORTED" if ok else "INCONCLUSIVE"),
                       "limitations": "technical features may partially encode speech; speaker roles not identified"}
    elif hid in {"H13", "H15"}:
        inter = ev.table("interaction_results", "18")
        if inter is not None:
            label = "TASK x ACOUSTIC" if hid == "H13" else None
            rows = inter[inter["interaction"] == label] if label else inter[inter["interaction"].str.startswith("SCREENING x") & (inter["status"] == "OK")]
            rows = rows.dropna(subset=["delta_r2_interaction"]) if "delta_r2_interaction" in rows else rows.iloc[0:0]
            if len(rows):
                r = rows.sort_values("ci_lower", ascending=False).iloc[0]
                status, why = _delta_status(r["delta_r2_interaction"], r["ci_lower"], r["ci_upper"])
                res = {"observed_effect": f"{r['interaction']}: delta R2 = {r['delta_r2_interaction']:.4f}",
                       "uncertainty": f"CI [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
                       "robustness": f"fold sign consistency {r.get('fold_sign_consistency', np.nan)}",
                       "support_status": status if hid == "H13" else ("PARTIALLY_SUPPORTED" if status == "SUPPORTED" else status),
                       "limitations": "best of several predefined interactions (multiplicity)" if hid == "H15" else why}
    elif hid == "H14":
        reg = ev.table("feature_registry", "24")
        if reg is not None:
            audio = reg[reg["feature_family"].isin(["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY"])]
            var = audio[audio["feature_name"].str.contains(r"__(std|mad|cv|iqr)$")]["residual_spearman"].abs()
            avg = audio[audio["feature_name"].str.endswith("__mean")]["residual_spearman"].abs()
            if len(var) and len(avg):
                res = {"observed_effect": f"median |residual rho| variability={var.median():.3f} vs mean={avg.median():.3f}; max {var.max():.3f} vs {avg.max():.3f}",
                       "uncertainty": "descriptive comparison of distributions", "robustness": "see feature registry stability",
                       "support_status": "PARTIALLY_SUPPORTED" if var.max() > avg.max() and var.median() > avg.median() else "NOT_SUPPORTED",
                       "limitations": "no direct CV comparison of variability-only vs mean-only feature sets"}
    elif hid == "H16":
        a12, a06 = _ablation_row(ev, "A12_TABULAR_ITEM_ADJUSTED_AUDIO"), _ablation_row(ev, "A06_TABULAR_TEMPORAL")
        if a12 is not None and a06 is not None:
            d = a12["delta_r2_vs_full_tabular"] - a06["delta_r2_vs_full_tabular"]
            res = {"observed_effect": f"item-adjusted gain {a12['delta_r2_vs_full_tabular']:.4f} vs temporal gain {a06['delta_r2_vs_full_tabular']:.4f}",
                   "uncertainty": f"A12 CI [{a12['delta_r2_vs_full_tabular_ci_lower']:.4f}, {a12['delta_r2_vs_full_tabular_ci_upper']:.4f}]",
                   "robustness": "student CV", "support_status": "PARTIALLY_SUPPORTED" if d > 0 and a12["delta_r2_vs_full_tabular_ci_lower"] > 0 else "INCONCLUSIVE",
                   "limitations": "difference of deltas not directly bootstrapped"}
    elif hid == "H17":
        rep = ev.table("literacy_index_report", "19")
        if rep is not None and "retention" in rep:
            n = int((rep["retention"] == "RETAIN").sum())
            a14 = _ablation_test(ev, "A14_TABULAR_LITERACY_INDICES")
            res = a14 or {}
            res.update({"observed_effect": f"{n} indices retained; " + res.get("observed_effect", "A14 not run"),
                        "support_status": res.get("support_status", "PARTIALLY_SUPPORTED" if n else "NOT_SUPPORTED"),
                        "uncertainty": res.get("uncertainty", "bootstrap CIs per index"), "robustness": res.get("robustness", "cross-school sign consistency"),
                        "limitations": "experimental constructs, not established assessments"})
    elif hid == "H18":
        res = _ablation_test(ev, "A04_THETA_ADAPTIVE_BEHAVIOR", ref="screening")
    elif hid == "H19":
        reg = ev.table("failure_registry", "22")
        if reg is not None and reg["error_reduction"].notna().any():
            from .stats_utils import bootstrap_ci

            hard = reg[reg["baseline_error"].abs() >= reg["baseline_error"].abs().quantile(0.8)]["error_reduction"].dropna().to_numpy()
            ci = bootstrap_ci(np.mean, hard, n_boot=int(cfg["bootstrap_iterations"]), seed=int(cfg["seed"]))
            status, why = _delta_status(ci["estimate"], ci["lower"], ci["upper"])
            res = {"observed_effect": f"mean |error| reduction among hardest 20% = {ci['estimate']:.4f}",
                   "uncertainty": f"CI [{ci['lower']:.4f}, {ci['upper']:.4f}]", "robustness": "see failure phenotype summary",
                   "support_status": "PARTIALLY_SUPPORTED" if status == "SUPPORTED" else status,
                   "limitations": "regression to the mean inflates apparent recovery for extreme baseline errors"}
    elif hid == "H20":
        rl = ev.table("residual_learning_results", "21")
        if rl is not None:
            a = rl[(rl["approach"] == "TWO_STAGE_RESIDUAL") & (rl["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD")]
            b = rl[(rl["approach"] == "TWO_STAGE_RESIDUAL") & (rl["scheme"] == "CV_B_SCHOOL_GROUP_KFOLD")]
            if len(a):
                r = a.iloc[0]
                status, why = _delta_status(r["delta_r2_vs_direct_multimodal"], r["delta_r2_vs_direct_multimodal_ci_lower"],
                                            r["delta_r2_vs_direct_multimodal_ci_upper"],
                                            b.iloc[0]["delta_r2_vs_direct_multimodal"] if len(b) else None)
                res = {"observed_effect": f"delta R2 vs direct = {r['delta_r2_vs_direct_multimodal']:.4f}",
                       "uncertainty": f"CI [{r['delta_r2_vs_direct_multimodal_ci_lower']:.4f}, {r['delta_r2_vs_direct_multimodal_ci_upper']:.4f}]",
                       "robustness": "school-grouped run reported" if len(b) else "student CV only", "support_status": status, "limitations": why}
    return res


DESIGN = {
    "H01": ("Ablation A13 vs A05", "SCREENING+tabular", "paired bootstrap on pooled OOF R2"),
    "H02": ("Correct-only aggregates vs residual", "screening via residual", "Spearman+FDR+bootstrap"),
    "H03": ("Shrunk complexity slopes", "screening via residual", "Spearman+FDR; ablation A12"),
    "H04": ("Within-student dispersion features", "screening via residual", "Spearman+FDR"),
    "H05": ("Task relevance matrix", "screening via residual", "per-task FDR"),
    "H06": ("Ablation A11 vs A05", "SCREENING+tabular", "paired bootstrap"),
    "H07": ("Clustering in screening-adjusted audio space", "screening (residualised space)", "bootstrap ARI"),
    "H08": ("Ablation A09 vs A05", "SCREENING+tabular", "paired bootstrap"),
    "H09": ("Embedding features in registry/ablation", "SCREENING", "registry evidence"),
    "H10": ("Residual discovery with held-out replication", "OOF screening residual", "FDR+replication"),
    "H11": ("School-grouped ablation", "SCREENING+tabular", "CV_B paired bootstrap"),
    "H12": ("Negative controls NC01-NC07", "SCREENING", "paired bootstrap"),
    "H13": ("Task-specific vs pooled representations", "SCREENING", "paired bootstrap"),
    "H14": ("Dispersion vs mean aggregates", "screening via residual", "descriptive"),
    "H15": ("Screening x acoustic interactions", "main effects", "paired bootstrap"),
    "H16": ("Ablation A12 vs A06", "SCREENING+tabular", "paired bootstrap"),
    "H17": ("Index report and ablation A14", "SCREENING", "bootstrap"),
    "H18": ("Ablation A04 vs A01", "SCREENING", "paired bootstrap"),
    "H19": ("Failure registry error reduction", "baseline OOF", "bootstrap mean"),
    "H20": ("Nested two-stage vs direct", "SCREENING+tabular", "paired bootstrap"),
}
VALIDATION = {"H07": "unsupervised; descriptive", "H14": "descriptive", "H05": "association"}


def stage_hypothesis_registry(ctx) -> dict:
    ev = Evidence(ctx)
    rows = []
    for hid, question in HYPOTHESES.items():
        try:
            res = evaluate(ev, hid)
        except Exception as exc:
            res = {"support_status": "NOT_TESTED", "limitations": f"evaluation error: {type(exc).__name__}: {exc}"}
        design, controls, method = DESIGN[hid]
        row = {"hypothesis_id": hid, "research_question": question, "experimental_design": design,
               "required_features": design.split(" ")[0], "control_variables": controls, "statistical_method": method,
               "validation_scheme": VALIDATION.get(hid, "student-disjoint CV (+ school-grouped where available)"),
               "observed_effect": "", "uncertainty": "", "multiple_testing_adjustment": "Benjamini-Hochberg FDR where applicable",
               "robustness": "", "support_status": "NOT_TESTED", "limitations": "required artifacts unavailable",
               "recommended_next_step": ""}
        if res:
            row.update(res)
        row["recommended_next_step"] = {
            "SUPPORTED": "carry into P02 feature engineering and confirm on new folds",
            "PARTIALLY_SUPPORTED": "confirm with school-grouped/nested validation before relying on it",
            "INCONCLUSIVE": "increase power (more features/better extraction) or refine the measurement",
            "NOT_SUPPORTED": "deprioritise in P02", "NOT_TESTED": "enable the required stages/modalities"}[row["support_status"]]
        rows.append(row)
    reg = pd.DataFrame(rows)
    ctx.save_df(reg, "hypothesis_registry", also_csv=True)
    return {"outputs": ["hypothesis_registry"], "summary": reg["support_status"].value_counts().to_dict()}
