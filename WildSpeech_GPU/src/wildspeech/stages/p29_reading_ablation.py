"""P29 — reading prediction and ablation ASR00-ASR12 against the tabular GAM.

1. ASR00 baseline on every scheme (repeated K-fold, school K-fold, LOSO, instrument / sample-rate transfer)
2. ASR01-ASR09 single families x fusions (early GAM, late, task-pooled, nested residual) — pre-specified
   reducer/budget; robustness schemes for early + residual
3. ASR10 (validated families, rule in evaluation.ablation), ASR11 nested residual, ASR12 regularised ensembles
4. model comparison on ASR10: Ridge, Elastic Net, GAM, PLS, CatBoost, XGBoost, LightGBM (early + residual)
5. negative controls: permuted features, shuffled outcome, recording-conditions-only
6. inference: paired bootstrap CI, BH-FDR; transfer table; selection stability; confounding; subgroups
7. champion comparison: a V3 model replaces the GAM only if reportable (no fallback), ΔR² CI > 0 with q < α on
   the primary scheme AND ΔR² > 0 under school-grouped CV; otherwise the GAM remains the champion
8. optional submission (v3.ablation.submission.enabled) — test features from P24/P25 on the test split
Registries: outputs/V3/ablation_results.csv, experiment_registry.csv, hypothesis_registry.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID, TARGET
from ..evaluation.ablation import (DESCRIPTIONS, FAMILY_EXPERIMENTS, SUPPLEMENTARY, AblationRunner, add_inference,
                                   combined_specs, control_specs, family_specs, model_specs, validated_families)
from ..evaluation.nested_asr_cv import V3Evaluator, V3Spec, result_row
from ..evaluation.robustness import confounding_table, selection_stability, subgroups, transfer_table
from ..io_utils import write_json
from ..v3 import viz as V
from ..v3.common import available_schemes, begin, not_executed, student_meta
from ..v3.providers import build_all
from ..v3.registry import save_figure_v3

S = "P29"


def run(ctx) -> dict:
    from ..asr.outputs import require_outputs

    reg, v3r = begin(ctx, S)
    ok, _acc, why = require_outputs(ctx, S, "any", 1, ("train",))
    if not ok:
        return not_executed(ctx, S, why)
    a = ctx.cfg.get_path("v3.ablation", {}) or {}
    ev = V3Evaluator(ctx, S, providers={})
    provs, avail, pm, models = build_all(ctx, ev)
    ctx.save_table(avail, S, "family_availability")
    for k, p in provs.items():
        ev.add_provider(k, p)
    fams_avail = [f for f in sorted(provs) if f != "REC"]
    if not fams_avail:
        return not_executed(ctx, S, "no V3 phonetic family available")
    sch = available_schemes(ctx, list(a.get("schemes", ["repeated_kfold"])))
    prim = a.get("primary_scheme", "repeated_kfold")
    if prim not in sch:
        return not_executed(ctx, S, f"primary scheme {prim} not in the validation registry")
    rob = [s for s in sch if s != prim]
    alphas = tuple(ctx.cfg.get_path("v3.audit.continuous_alphas", [0, .1, .25, .5, 1]))
    bud, meth = int(a.get("default_budget", 20)), str(a.get("default_method", "pca"))
    bm = ev.baseline["model"]
    base = {sc: ev.evaluate(ev.baseline_v3(), sc) for sc in sch}
    bt = pd.DataFrame([result_row(b, {"delta_r2": 0.0}) for b in base.values()])
    v2b = ctx.read_stage_table("P21", "all_results")
    if v2b is not None:
        bt = bt.merge(v2b[v2b["name"] == "V2_BASELINE"][["scheme", "r2_mean"]].rename(columns={"r2_mean": "v2_r2_mean"}), on="scheme", how="left")
        bt["matches_v2"] = (bt["r2_mean"] - bt["v2_r2_mean"]).abs() < 1e-6
    ctx.save_table(bt, S, "baseline_by_scheme")
    ctx.save_table(baseline_reference_check(ctx, bt, prim), S, "baseline_reference_check")
    fusions = list(a.get("fusions", ["early", "late", "task_pool", "residual"]))
    # ---- stage 1: single families (all fusions on the primary scheme; early+residual on robustness schemes)
    r1 = AblationRunner(ev, base, int(a.get("max_evals", 600)), float(a.get("eval_time_budget_s", 14400)), ctx.logger)
    for exp, fams in {**FAMILY_EXPERIMENTS, **SUPPLEMENTARY}.items():
        if not all(f in provs for f in fams):
            continue
        for sp in family_specs(exp, fams, fusions, bud, meth if fams != ["ASRE_MIL"] else "none", bm, alphas):
            if exp in SUPPLEMENTARY:
                sp.group = "supplementary"
            scs = [prim] + (rob if sp.fusion in ("early", "residual") else [])
            r1.add(sp, scs, 10 if sp.fusion in ("early", "residual") else 20)
    rows1, sched1 = r1.run()
    rows1 = add_inference(rows1, float(a.get("fdr_alpha", 0.05)))
    val, rule = validated_families(rows1, [f for f in fams_avail if f in {x for v in FAMILY_EXPERIMENTS.values() for x in v}], prim)
    ctx.save_table(pd.DataFrame([{"validated_families": "+".join(val), "rule": rule}]), S, "validated_families")
    # ---- stage 2: combined, models, controls
    r2 = AblationRunner(ev, base, int(a.get("max_evals", 600)), float(a.get("eval_time_budget_s", 14400)), ctx.logger)
    for sp in combined_specs(val, bud, meth, bm, alphas):
        r2.add(sp, [prim] + (rob if sp.experiment in ("ASR10", "ASR11") else []), 5)
    for sp in model_specs(val, list(a.get("models", ["RIDGE", "GAM"])), bud, meth, alphas):
        r2.add(sp, [prim], 30)
    ctrl = [c for c in a.get("negative_controls", []) if c != "recording_only" or "REC" in provs]
    for sp in control_specs(val, bud, meth, alphas, ctrl, bm):
        r2.add(sp, [prim], 15)
    rows2, sched2 = r2.run()
    rows2 = add_inference(rows2, float(a.get("fdr_alpha", 0.05)))
    # ---- stage 3: source-model overlap sensitivity. The Pasketti models list ReadNet (Reach Every Reader data) among
    # their training corpora, so every validated family is re-evaluated WITHOUT features derived from them.
    rows3, sens = pd.DataFrame(), []
    for tag, ms in dict(a.get("exclusion_sets", {"NO_PASKETTI": ["pasketti_phonetic", "pasketti_word"]}) or {}).items():
        ms = [m for m in ms if m in models]
        if not ms or not val:
            continue
        p3, _a3, _pm3, _m3 = build_all(ctx, ev, exclude_models=tuple(ms))
        keys3 = []
        for f in val:
            if f in p3:
                ev.add_provider(f"{f}__{tag}", p3[f])
                keys3.append(f"{f}__{tag}")
        if not keys3:
            sens.append({"exclusion": tag, "excluded_models": ",".join(ms), "status": "no family left without these models"})
            continue
        r3 = AblationRunner(ev, base, 40, float(a.get("eval_time_budget_s", 14400)), ctx.logger)
        for fu in ("early", "residual"):
            r3.add(V3Spec(f"ASR10_{tag}_{fu.upper()}", bm if fu == "early" else "RIDGE", keys3, bud, meth, fu,
                          alphas=alphas if fu == "residual" else None, group="sensitivity",
                          experiment=f"ASR10_{tag}"), [prim], 10)
        rr, _s3 = r3.run()
        rr = add_inference(rr, float(a.get("fdr_alpha", 0.05)))
        rows3 = pd.concat([rows3, rr], ignore_index=True)
        ref = rows2[(rows2["experiment"] == "ASR10") & (rows2["scheme"] == prim)] if not rows2.empty else rows2
        for x in rr.to_dict("records"):
            same = ref[ref["fusion"] == x.get("fusion")] if (not ref.empty and "fusion" in ref) else ref
            sens.append({"exclusion": tag, "excluded_models": ",".join(ms), "name": x.get("name"), "fusion": x.get("fusion"),
                         "delta_r2_without": x.get("delta_r2"), "ci_low_without": x.get("delta_ci_low"),
                         "ci_high_without": x.get("delta_ci_high"),
                         "delta_r2_with": float(same["delta_r2"].max()) if not same.empty else np.nan,
                         "status": x.get("status")})
    ctx.save_table(pd.DataFrame(sens), S, "overlap_sensitivity")
    rows = pd.concat([bt.assign(experiment="ASR00", group="baseline"), rows1, rows2, rows3], ignore_index=True)
    rows["description"] = rows["experiment"].map(DESCRIPTIONS).fillna(rows["experiment"])
    rows["reportable"] = rows["reportable"].fillna(rows["group"] == "baseline").astype(bool) if "reportable" in rows else rows["group"] == "baseline"
    for col in ("significant",):
        if col in rows:
            rows[col] = rows[col].fillna(False).astype(bool)
    from ..evaluation.ablation import classify_results

    rows = classify_results(rows, float(a.get("fdr_alpha", 0.05)))
    ctx.save_table(rows, S, "ablation_results")
    ctx.save_table(rows.groupby(["scheme", "result_class"]).size().rename("n").reset_index(), S, "result_classes")
    ctx.save_table(pd.concat([sched1.assign(phase=1), sched2.assign(phase=2)], ignore_index=True), S, "schedule")
    for x in rows.to_dict("records"):
        v3r.log_ablation(f"{S}:{x.get('name')}:{x.get('scheme')}", stage=S, kind=x.get("group"),
                         **{k: x.get(k) for k in ("experiment", "description", "name", "scheme", "status", "model", "fusion",
                                                  "method", "budget", "providers", "control", "r2_mean", "r2_sd", "r2_pooled",
                                                  "r2_fold_mean", "rmse", "mae", "delta_r2", "delta_ci_low", "delta_ci_high",
                                                  "p", "q", "significant", "reportable", "fallback_folds")})
    # ---- robustness / stability / confounding / subgroups
    tt = transfer_table(rows[rows["group"] != "baseline"])
    ctx.save_table(tt, S, "transfer_robustness")
    if not tt.empty:
        mat = tt.set_index("name")[[c for c in ("repeated_kfold", "school_kfold", "loso", "instrument_transfer", "sample_rate_transfer") if c in tt]]
        save_figure_v3(ctx, S, "robustness_heatmap", V.heatmap(mat, "ΔR² vs GAM across validation schemes", diverging=True,
                       annotate=True, fmt="{:.3f}"), mat, "Cross-school, instrument and sample-rate robustness")
    try:
        st_spec = V3Spec("ASR10_STABILITY", "RIDGE", val, bud, "select", "early")
        freq, js = selection_stability(ev, st_spec, prim, int(a.get("stability_top_k", 20)))
        ctx.save_table(freq, S, "feature_selection_stability")
        ctx.save_table(pd.DataFrame([js]), S, "feature_selection_jaccard")
        if not freq.empty:
            fam_stab = freq.groupby("family")["selection_freq"].agg(["mean", "max", "count"]).reset_index()
            ctx.save_table(fam_stab, S, "family_stability")
            save_figure_v3(ctx, S, "family_stability", V.bars(fam_stab, "family", "mean", "Mean fold selection frequency per family",
                           ylabel="selection frequency"), fam_stab, "Feature-family stability")
    except Exception as e:  # noqa: BLE001
        ctx.logger.warning("P29 stability failed: %s", e)
    from ..validation.metrics import mean_oof

    mo = mean_oof(base[prim].oof)
    mo.index = mo.index.astype(str)
    resid = (mo["y_true"] - mo["y_pred"]).rename("resid")
    meta = student_meta(ctx)
    try:
        conf = confounding_table(ev, {k: provs[k] for k in val}, resid, meta)
        ctx.save_table(conf, S, "confounding")
    except Exception as e:  # noqa: BLE001
        ctx.logger.warning("P29 confounding failed: %s", e)
    # ---- figures
    pr = rows[(rows["scheme"] == prim) & rows["group"].isin(["family", "combined", "supplementary"])].dropna(subset=["delta_r2"])
    if not pr.empty:
        save_figure_v3(ctx, S, "incremental_r2_forest", V.forest(pr.sort_values("delta_r2", ascending=False), "delta_r2",
                       "delta_ci_low", "delta_ci_high", "name", "Incremental R² over the tabular GAM (paired bootstrap 95% CI)",
                       group="fusion"), pr, "Incremental R² forest plot")
        best_by = pr[pr["reportable"]].sort_values("delta_r2", ascending=False).groupby("experiment").head(1)
        save_figure_v3(ctx, S, "family_ablation", V.bars(best_by.sort_values("experiment"), "experiment", "delta_r2",
                       "Best reportable fusion per feature family", "delta_ci_low", "delta_ci_high", ylabel="ΔR²", ref=0.0),
                       best_by, "Feature-family ablations")
    mdl = rows[(rows["group"] == "model") & (rows["scheme"] == prim)].dropna(subset=["delta_r2"])
    if not mdl.empty:
        save_figure_v3(ctx, S, "model_comparison", V.forest(mdl, "delta_r2", "delta_ci_low", "delta_ci_high", "name",
                       "Models on validated phonetic families", group="fusion"), mdl, "Model comparison")
    ctl = rows[rows["group"] == "control"].dropna(subset=["delta_r2"])
    if not ctl.empty:
        save_figure_v3(ctx, S, "negative_controls", V.forest(ctl, "delta_r2", "delta_ci_low", "delta_ci_high", "name",
                       "Negative controls (should show no gain)"), ctl, "Confounding and negative controls")
        bad = ctl[(ctl["delta_ci_low"] > 0)]
        v3r.update_hypothesis("V3H11", "supported" if bad.empty else "not_supported", S,
                              "controls with ΔR² CI > 0: " + (",".join(bad["name"]) or "none"))
    # ---- hypotheses
    fam_rows = rows1[(rows1["group"] == "family")] if not rows1.empty else rows1
    school = fam_rows[fam_rows["scheme"] == "school_kfold"].set_index("name")["delta_r2"] if not fam_rows.empty else pd.Series(dtype=float)
    pf = fam_rows[(fam_rows["scheme"] == prim) & fam_rows["reportable"]] if not fam_rows.empty else fam_rows
    alpha = float(a.get("fdr_alpha", 0.05))
    if not pf.empty:
        gen = pf[(pf["delta_ci_low"] > 0) & (pf["q"] < alpha) & pf["name"].map(lambda n: school.get(n, np.nan) > 0 if n in school else True)]
        v3r.update_hypothesis("V3H08", "supported" if not gen.empty else "not_supported", S,
                              "families with genuine incremental value: " + (",".join(gen["name"]) or "none"),
                              effect=float(pf["delta_r2"].max()))
        o6 = pf[pf["experiment"] == "ASR06"]
        if not o6.empty:
            b6 = o6.sort_values("delta_r2", ascending=False).iloc[0]
            v3r.update_hypothesis("V3H09", "supported" if b6["delta_ci_low"] > 0 else "not_supported", S,
                                  f"{b6['name']} ΔR²={b6['delta_r2']:.4f} [{b6['delta_ci_low']:.4f},{b6['delta_ci_high']:.4f}]",
                                  effect=float(b6["delta_r2"]), ci_low=float(b6["delta_ci_low"]), ci_high=float(b6["delta_ci_high"]))
    tr = rows[rows["scheme"].isin(["sample_rate_transfer", "instrument_transfer"]) & (rows["experiment"] == "ASR10")]
    if not tr.empty:
        ok = bool((tr.groupby("scheme")["delta_r2"].max() > 0).all())
        v3r.update_hypothesis("V3H10", "supported" if ok else "not_supported", S,
                              "; ".join(f"{r.scheme}: {r.name} ΔR²={r.delta_r2:.4f}" for r in tr.itertuples()))
    # ---- champion comparison
    cand = rows[(rows["scheme"] == prim) & rows["reportable"] & rows["group"].isin(["family", "combined", "model"])]
    cand = cand[(cand["delta_ci_low"] > 0) & (cand["q"] < alpha)]
    sch_d = rows[rows["scheme"] == "school_kfold"].set_index("name")["delta_r2"]
    cand = cand[cand["name"].map(lambda n: bool(sch_d.get(n, np.nan) > 0) if n in sch_d else False)]
    champion = {"champion": "ASR00_TABULAR_GAM", "reason": "no V3 model met the pre-specified replacement rule",
                "baseline_r2": float(base[prim].summary.get("r2_mean", np.nan)), "rule":
                "reportable, ΔR² CI>0, q<alpha (primary), ΔR²>0 school_kfold"}
    if not cand.empty:
        best = cand.sort_values("delta_r2", ascending=False).iloc[0]
        champion = {"champion": best["name"], "experiment": best["experiment"], "delta_r2": float(best["delta_r2"]),
                    "delta_ci": [float(best["delta_ci_low"]), float(best["delta_ci_high"])], "q": float(best["q"]),
                    "r2_mean": float(best["r2_mean"]), "baseline_r2": champion["baseline_r2"], "rule": champion["rule"],
                    "spec": next((r.spec.as_dict() for (n, s), r in {**r1.results, **r2.results}.items() if n == best["name"] and s == prim), None)}
    write_json(ctx.paths.v3_root / "v3_champion.json", champion)
    ctx.save_table(pd.DataFrame([{k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in champion.items()}]), S, "champion_comparison")
    cres = None
    if champion["champion"] != "ASR00_TABULAR_GAM":
        cres = {**r1.results, **r2.results}.get((champion["champion"], prim))
    sg = subgroups(cres.oof if cres is not None else base[prim].oof, meta)
    ctx.save_table(sg, S, "champion_subgroups")
    out = {"validated": val, "rule": rule, "champion": champion["champion"], "n_rows": int(len(rows))}
    if bool((a.get("submission", {}) or {}).get("enabled", False)):
        out["submission"] = _submission(ctx, ev, champion, cres)
    from ..v3.report import build_report, export_root_tables

    export_root_tables(ctx)
    out["report"] = build_report(ctx)
    return out


REFERENCE_GAM = "B_GAM__SCREENING+BEHAVIOR+ITEM"


def baseline_reference_check(ctx, bt: pd.DataFrame, prim: str, tol: float = 0.01) -> pd.DataFrame:
    """The V3 baseline must be the strongest tabular GAM (P10 B_GAM__SCREENING+BEHAVIOR+ITEM, R² ≈ 0.2153) on the
    same student-disjoint folds. P10 and the frozen V2 registry can differ in fold seeds, so a small gap is
    reported, a large one flagged (the ΔR² values are always paired against the V3 baseline row itself)."""
    row = bt[bt["scheme"] == prim] if "scheme" in bt else bt
    v3 = float(row["r2_mean"].iloc[0]) if (not row.empty and "r2_mean" in row) else np.nan
    p10 = ctx.read_stage_table("P10", "model_benchmark")
    ref = np.nan
    if p10 is not None and not p10.empty and "name" in p10 and "r2_mean" in p10:
        m = p10[p10["name"] == REFERENCE_GAM]
        ref = float(m["r2_mean"].iloc[0]) if not m.empty else np.nan
    v2 = float(row["v2_r2_mean"].iloc[0]) if (not row.empty and "v2_r2_mean" in row) else np.nan
    gap = v3 - ref if np.isfinite(v3) and np.isfinite(ref) else np.nan
    status = ("NO_REFERENCE" if not np.isfinite(gap) else ("MATCH" if abs(gap) <= tol else "MISMATCH"))
    if status == "MISMATCH":
        ctx.logger.warning("P29 baseline R² %.4f differs from P10 %s (%.4f) by %.4f", v3, REFERENCE_GAM, ref, gap)
    return pd.DataFrame([{"scheme": prim, "v3_baseline_r2": v3, "p10_reference": REFERENCE_GAM, "p10_r2": ref,
                          "v2_baseline_r2": v2, "gap_vs_p10": gap, "tolerance": tol, "status": status}])


def _submission(ctx, ev: V3Evaluator, champion: dict, cres) -> dict:
    """Champion predictions for the test students (requires P24/P25 test features for the champion's families)."""
    from ..submission import validate_submission

    b = ctx.bundle()
    if any("subsampled" in n for n in b.notes):
        return {"status": "skipped", "reason": "data subsampled by profile"}
    fmt = b.submission_format
    ids = fmt[ID].astype(str).tolist()
    spec = ev.baseline_v3() if champion["champion"] == "ASR00_TABULAR_GAM" else V3Spec(**champion["spec"])
    fmod, Xtr, Xte = ev.fit_full_v3(spec, ids)
    p = fmod.predict(Xte.reindex(ids))
    y = ev.y
    if bool(ctx.cfg.get_path("v3.ablation.submission.clip_to_train_range", True)):
        p = np.clip(p, float(y.min()), float(y.max()))
    sub = pd.DataFrame({ID: fmt[ID], TARGET: p})
    sub = sub[list(fmt.columns)] if set(fmt.columns) <= set(sub.columns) else sub
    chk = validate_submission(sub, fmt)
    d = ctx.paths.root / ctx.cfg.get_path("v3.ablation.submission.directory", "submissions/V3")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"submission_{champion['champion']}.csv"
    if path.exists():
        path = d / f"submission_{champion['champion']}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"   # never overwrite
    if (chk["status"] == "PASS").all():
        sub.to_csv(path, index=False)
    ctx.save_table(chk, S, "submission_checks")
    return {"status": "written" if (chk["status"] == "PASS").all() else "failed_checks", "path": str(path),
            "n_missing_test_features": int(Xte.isna().all(1).sum())}
