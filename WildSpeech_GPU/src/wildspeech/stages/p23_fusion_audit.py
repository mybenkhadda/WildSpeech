"""P23 — audit of the V2 multimodal evaluation (run BEFORE any V3 ASR experiment).

1. classify every V2 P21 row: identical-to-baseline r2, zero ΔR², zero-width CI, budget cap
2. lineage audit: stale stages under the current config, upstream fingerprints that changed after a stage ran
3. re-run the configured V2 specs with tracing (method='v2_family' reproduces V2 exactly) + diagnostic variants
   (forced alpha, continuous alpha grid, V3 budget-following reducer on inner-OOF residuals)
4. verdict per spec: genuine_negative | silent_fallback | positive_signal_discarded | unused_features |
   evaluation_error | fitted_positive
5. V1 TASK_ENSEMBLE re-run with prediction tracing vs the repaired ensemble
Outputs: outputs/V3/P23_FUSION_AUDIT/tables/*.csv + figures; hypotheses V3H01 / V3H02.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.fusion_audit import (TaskEnsembleAudit, budget_cap_table, classify_v2_results, lineage_audit,
                                       reaudit, verdicts)
from ..evaluation.nested_asr_cv import V3Evaluator, result_row
from ..v3 import viz as V
from ..v3.common import available_schemes, begin
from ..v3.registry import save_figure_v3

S = "P23"


def run(ctx) -> dict:
    reg, v3r = begin(ctx, S)
    a = ctx.cfg.get_path("v3.audit", {}) or {}
    tol = float(a.get("zero_delta_tol", 1e-12))
    v2_stage = a.get("stage_v2", "P21")
    summary: dict = {}

    # ---- 1. classification of the reported V2 table
    v2t = ctx.read_stage_table(v2_stage, "all_results")
    cls = classify_v2_results(v2t, tol) if v2t is not None else pd.DataFrame()
    if not cls.empty:
        ctx.save_table(cls, S, "v2_result_classification")
        cap = budget_cap_table(cls)
        ctx.save_table(cap, S, "v2_budget_cap")
        summary.update({"v2_rows": int(len(cls)), "v2_suspect_fallback_rows": int(cls["suspect_fallback"].sum())})
        bud = cls[cls["group"] == "budget"].copy()
        if not bud.empty:
            bud["family"] = bud["name"].str.split("_").str[2]
            save_figure_v3(ctx, S, "v2_budget_cap", V.lines(bud, "budget", "n_added", "family",
                           "V2 budget vs columns actually added (PCA capped per family)", xlabel="requested budget",
                           ylabel="columns added", logx=True), bud[["name", "family", "budget", "n_added", "delta_r2"]],
                           "Requested feature budget vs columns actually added in V2 P21")
    else:
        summary["v2_rows"] = 0

    # ---- 2. lineage
    lin = lineage_audit(ctx)
    ctx.save_table(lin, S, "lineage_audit")
    summary["stale_v1_v2_stages"] = lin.loc[lin.get("stale_under_current_config", pd.Series(dtype=object)) == True, "stage"].tolist() if not lin.empty else []  # noqa: E712

    # ---- 3. traced re-audit
    from ..v2.providers import build_providers

    provs = build_providers(ctx)
    ev = V3Evaluator(ctx, S, providers=provs)
    sch = available_schemes(ctx, list(a.get("schemes", ["repeated_kfold"])))
    base = {}
    rows = []
    for sc in sch:
        b = ev.evaluate(ev.baseline_v3("V2_BASELINE"), sc)
        base[sc] = b
        r = result_row(b)
        if v2t is not None:
            v = v2t[(v2t["name"] == "V2_BASELINE") & (v2t["scheme"] == sc)]
            r["v2_r2_mean"] = float(v.iloc[0]["r2_mean"]) if not v.empty else np.nan
            r["reproduced"] = bool(abs(r["v2_r2_mean"] - r["r2_mean"]) < 1e-6) if not v.empty else None
        rows.append(r)
    bver = pd.DataFrame(rows)
    ctx.save_table(bver, S, "baseline_reproduction")
    if not bver.empty:
        summary["baseline_r2"] = float(bver.iloc[0]["r2_mean"])
        summary["baseline_reproduced"] = bver.iloc[0].get("reproduced")
    specs = ctx.read_stage_json(v2_stage, "specs", {}) or {}
    names = [n for n in a.get("reaudit_specs", []) if n in specs]
    audit, traces = reaudit(ev, specs, names, sch, base, v2t, ctx.cfg) if names else (pd.DataFrame(), pd.DataFrame())
    ctx.save_table(audit, S, "fusion_audit")
    ctx.save_table(traces, S, "fusion_audit_folds", parquet=True)
    ver = verdicts(audit)
    ctx.save_table(ver, S, "fusion_verdicts")
    if not ver.empty:
        summary["verdicts"] = ver["verdict"].value_counts().to_dict()
        fb = ver["verdict"].isin(["silent_fallback", "positive_signal_discarded"])
        v3r.update_hypothesis("V3H01", "supported" if fb.any() else "not_supported", S,
                              f"{int(fb.sum())}/{len(ver)} audited V2 specs were baseline fallbacks: "
                              + ",".join(ver.loc[fb, "audited"]), n=int(len(ver)))
    if not traces.empty:
        t = traces[traces["variant"] == "reproduce"].copy()
        if "alpha" in t and t["alpha"].notna().any():
            t["fold_id"] = t["repeat"].astype(str) + "/" + t["fold"].astype(str)
            mat = t.pivot_table(index="audited", columns="fold_id", values="alpha", aggfunc="first")
            save_figure_v3(ctx, S, "residual_alpha_by_fold", V.heatmap(mat, "Residual shrinkage α chosen per fold (0 = baseline fallback)",
                           annotate=True, xlabel="repeat/fold"), mat, "Residual-fusion shrinkage α per fold")
        if "stage2_pred_sd" in t:
            save_figure_v3(ctx, S, "stage2_prediction_sd", V.bars(t.groupby("audited")["stage2_pred_sd"].mean().reset_index(),
                           "audited", "stage2_pred_sd", "Mean SD of the residual (stage-2) prediction on validation folds",
                           ylabel="SD (outcome units)"), t[["audited", "repeat", "fold", "stage2_pred_sd", "alpha", "n_added"]],
                           "Residual model prediction variance")
    if not audit.empty and "delta_r2" in audit:
        fd = audit.dropna(subset=["delta_r2"]).copy()
        fd["label"] = fd["audited"] + " · " + fd["variant"]
        save_figure_v3(ctx, S, "audit_forest", V.forest(fd, "delta_r2", "delta_ci_low", "delta_ci_high", "label",
                       "ΔR² vs GAM: reproduced V2 vs diagnostic variants", group="variant"),
                       fd[["audited", "variant", "scheme", "status", "delta_r2", "delta_ci_low", "delta_ci_high"]],
                       "Reproduced V2 results and diagnostic variants")

    # ---- 4. V1 task-ensemble audit
    if a.get("task_ensemble_audit", True):
        try:
            oof, fold_t, mem = TaskEnsembleAudit(ctx, ctx.cfg.get_path("v3.runtime.max_repeats")).run()
            ctx.save_table(fold_t, S, "task_ensemble_folds")
            ctx.save_table(mem, S, "task_ensemble_members")
            ctx.save_parquet(oof, S, "task_ensemble_oof")
            agg = fold_t.groupby("variant").agg(r2_mean=("r2", "mean"), r2_min=("r2", "min"), max_abs_z=("max_abs_z", "max"),
                                               extreme_frac=("extreme_frac", "mean"), weight_sum_mean=("weight_sum", "mean")).reset_index()
            ctx.save_table(agg, S, "task_ensemble_summary")
            save_figure_v3(ctx, S, "task_ensemble_extremes", V.bars(agg, "variant", "max_abs_z",
                           "Largest |prediction − mean| / SD per variant (V1 vs repaired task ensemble)", ylabel="max |z|"),
                           agg, "Task-ensemble failure and repair")
            v1 = agg[agg["variant"] == "v1_task_ensemble"]
            rp = agg[agg["variant"] == "repaired_robust"]
            if not v1.empty and not rp.empty:
                ok = float(v1["extreme_frac"].iloc[0]) > 0 and float(rp["extreme_frac"].iloc[0]) == 0
                v3r.update_hypothesis("V3H02", "supported" if ok else "not_supported", S,
                                      f"extreme_frac V1={float(v1['extreme_frac'].iloc[0]):.4f}, repaired={float(rp['extreme_frac'].iloc[0]):.4f}; "
                                      f"mean weight sum V1={float(v1['weight_sum_mean'].iloc[0]):.2f}",
                                      effect=float(v1["max_abs_z"].iloc[0]))
            summary["task_ensemble"] = agg.to_dict("records")
        except Exception as e:  # noqa: BLE001 - recorded, the rest of the audit stands
            ctx.save_table(pd.DataFrame([{"status": "failed", "error": f"{type(e).__name__}: {e}"}]), S, "task_ensemble_folds")
            summary["task_ensemble"] = f"failed: {e}"

    # ---- 5. what V3 changes (each item is evaluated above or in P28/P29)
    ctx.save_table(pd.DataFrame([
        {"issue": "residual fusion alpha=0 reported as ΔR²=0 'ok'", "v3_fix": "status fallback_baseline; never reported as a result; forced/continuous-alpha diagnostics"},
        {"issue": "budget capped by family_components=10", "v3_fix": "CompactReducer: components follow the budget"},
        {"issue": "budget selection on IN-SAMPLE ridge residual", "v3_fix": "selection target = inner-OOF residual of the training fold"},
        {"issue": "V1 task ensemble unbounded NNLS weights / extreme members", "v3_fix": "RobustTaskEnsemble (convex weights, winsorised inputs, missing-task gating, clipping) + extreme-prediction audit"},
        {"issue": "stale V2 fingerprints under a different profile", "v3_fix": "V3 profiles layer V2_FULL overrides; V3 never re-plans V1/V2; staleness recorded"},
        {"issue": "failed predictions could be silently replaced", "v3_fix": "V3Evaluator returns status failed + traceback; no baseline substitution"},
    ]), S, "v3_repairs")
    return summary
