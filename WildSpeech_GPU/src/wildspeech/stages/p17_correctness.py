"""P17 — clip-level correctness intelligence.

Clip level (student-disjoint OOF): feature sets x classifiers -> AUROC, AUPRC (incorrect class), balanced accuracy,
Brier, log loss, ECE, per task / recording domain / instrument, cross-domain transfer AUROC, audio-specific gain
over metadata-only (paired student-block bootstrap), ROC / PR / reliability / confusion figures.
Student level: nested cross-fitted soft-correctness features per outer fold of the frozen registry, evaluated as
nested residual fusion over the strongest tabular comparator, and against metadata-only soft features (audio-
specific information). Residual analyses use the comparator's OOF residuals with a discovery / confirmation split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID
from ..correctness.calibration import IsotonicCalibrator, classification_metrics, reliability_table
from ..correctness.classifiers import FEATURE_SETS, ClipPipeline
from ..correctness.crossfit import clip_oof, student_groups_kfold
from ..diagnostics.scientific_validation import auc_fast, block_bootstrap
from ..models.compact_regression import CompactSpec, StudentLevelEvaluator
from ..models.zoo import ModelUnavailable
from ..v2 import visualization_v2 as V
from ..v2.providers import build_bank, soft_provider
from ..v2.registry import save_figure_v2
from ..v2.scheduler import ExperimentScheduler
from ..v2.validation_registry import discovery_split
from ._v2common import association_table, baseline_residuals, begin, evaluate_against, schemes

S = "P17"


def _make(bank, fs, model, cfg, seed):
    return lambda: ClipPipeline(bank, fs, model, cfg, seed=seed, device="cpu",
                                embedding_pca=int(cfg.get_path("v2.correctness.embedding_pca", 32)),
                                include_theta=bool(cfg.get_path("v2.correctness.include_screening_theta", False)))


def _crossfit_calibrate(p: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Isotonic calibration of OOF probabilities, itself cross-fitted over students (display only)."""
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p) & np.isfinite(y)
    for tr, va in student_groups_kfold(groups, 2, seed):
        t = tr[ok[tr]]
        if len(t) > 50:
            out[va] = IsotonicCalibrator().fit(p[t], y[t]).transform(np.nan_to_num(p[va], nan=0.5))
    return np.where(np.isfinite(p), out, np.nan)


def run(ctx) -> dict:
    reg, v2r = begin(ctx, S)
    cc = ctx.cfg.get_path("v2.correctness", {}) or {}
    nb = int(cc.get("bootstrap", 500))
    bank = build_bank(ctx, ("train",))
    ctx.save_table(bank.summary(), S, "feature_bank")
    c = bank.c
    rows = np.flatnonzero(((c["split"] == "train") & c["item_score"].notna()).to_numpy())
    cr = c.iloc[rows].reset_index(drop=True)
    y = cr["item_score"].to_numpy(float)
    g = cr[ID].astype(str).to_numpy()
    sets = [s for s in cc.get("feature_sets", []) if s in FEATURE_SETS and bank.set_available(s)]
    ctx.save_table(pd.DataFrame([{"feature_set": s, "available": bank.set_available(s), "blocks": "+".join(FEATURE_SETS[s]["blocks"])}
                                 for s in cc.get("feature_sets", []) if s in FEATURE_SETS]), S, "feature_set_availability")

    # ---------------------------------------------------------------- clip-level grid
    oofs: dict[tuple[str, str], np.ndarray] = {}
    sched = ExperimentScheduler(max_evals=int(cc.get("max_grid_evals", 60)), logger=ctx.logger)
    for i, s in enumerate(sets):
        for j, m in enumerate(cc.get("models", ["LOGISTIC"])):
            def task(s=s, m=m):
                try:
                    oofs[(s, m)] = clip_oof(c, rows, _make(bank, s, m, ctx.cfg, ctx.seed), int(cc.get("clip_cv_splits", 5)), ctx.seed)
                except ModelUnavailable as e:
                    return f"unavailable: {e}"
                return "ok"
            sched.add(f"{s}:{m}", task, priority=10 * i + j, group=s)
    ctx.save_table(sched.run(), S, "grid_schedule")
    met, by_group = [], []
    st = ctx.bundle().train_students.drop_duplicates(ID).set_index(ID)
    for (s, m), p in oofs.items():
        met.append({"feature_set": s, "model": m, **classification_metrics(y, p)})
        for col, vals in (("task", cr["task"].to_numpy()), ("orig_rate", np.where(cr["orig_sr_low"] == 1, "low_rate", "16k")),
                          ("instrument", cr["instrument"].to_numpy())):
            for gv in np.unique(vals):
                mm = vals == gv
                by_group.append({"feature_set": s, "model": m, "group_col": col, "group": gv, **classification_metrics(y[mm], p[mm])})
    met = pd.DataFrame(met)
    ctx.save_table(met, S, "clip_metrics")
    ctx.save_table(pd.DataFrame(by_group), S, "clip_metrics_by_group")
    if met.empty:
        return {"status": "no clip-level model could be fitted"}
    best = met.sort_values("logloss").drop_duplicates("feature_set").set_index("feature_set")["model"].to_dict()

    # ------------------------------------------------- audio-specific gain + transfer
    gains, tr_rows = [], []
    if "META" in best:
        pm = oofs[("META", best["META"])]
        df = pd.DataFrame({ID: g, "y": y, "pm": pm})
        for s, m in best.items():
            if s == "META":
                continue
            d = df.assign(pa=oofs[(s, m)]).dropna()
            est, lo, hi = block_bootstrap(lambda x: auc_fast(x["y"], x["pa"]) - auc_fast(x["y"], x["pm"]), d, ID, nb, ctx.seed)
            gains.append({"feature_set": s, "model": m, "auroc": auc_fast(d["y"], d["pa"]), "auroc_meta": auc_fast(d["y"], d["pm"]),
                          "delta_auroc": est, "ci_low": lo, "ci_high": hi})
    gains = pd.DataFrame(gains)
    ctx.save_table(gains, S, "audio_specific_gain")
    dom = cr["orig_sr_low"].to_numpy()
    for s, m in best.items():
        for src, tgt in ((0.0, 1.0), (1.0, 0.0)):
            fit = rows[(dom == src)]
            prd = rows[(dom == tgt)]
            if len(fit) < 100 or len(prd) < 50:
                continue
            try:
                p = _make(bank, s, m, ctx.cfg, ctx.seed)().fit(fit).predict_proba(prd)
            except ModelUnavailable:
                continue
            tr_rows.append({"feature_set": s, "model": m, "train_domain": "low_rate" if src else "16k",
                            "test_domain": "low_rate" if tgt else "16k", **classification_metrics(c.iloc[prd]["item_score"].to_numpy(float), p)})
    ctx.save_table(pd.DataFrame(tr_rows), S, "domain_transfer_auc")

    # ------------------------------------------------------------------ figures
    curves = {f"{s}/{m}": (y, oofs[(s, m)]) for s, m in best.items()}
    save_figure_v2(ctx, S, "roc_curves", V.roc_curves(curves, "Correctness ROC (student-disjoint OOF)"),
                   met, "ROC per feature set (best model each)")
    save_figure_v2(ctx, S, "pr_curves_incorrect", V.pr_curves(curves, "Precision-recall for INCORRECT responses"), met, "PR curves")
    rel = {k: reliability_table(yy, _crossfit_calibrate(pp, yy, g, ctx.seed)) for k, (yy, pp) in curves.items()}
    rel_raw = {k: reliability_table(yy, pp) for k, (yy, pp) in curves.items()}
    save_figure_v2(ctx, S, "calibration_raw", V.reliability(rel_raw, "Reliability (raw OOF)"),
                   pd.concat([t.assign(name=k) for k, t in rel_raw.items()]), "Calibration before calibration")
    save_figure_v2(ctx, S, "calibration_isotonic", V.reliability(rel, "Reliability (cross-fitted isotonic)"),
                   pd.concat([t.assign(name=k) for k, t in rel.items()]), "Calibration after cross-fitted isotonic")
    if not gains.empty:
        save_figure_v2(ctx, S, "audio_specific_gain", V.bar_ci(gains.sort_values("delta_auroc"), "feature_set", "delta_auroc", "ci_low", "ci_high",
                       "AUROC gain over metadata-only"), gains, "Audio-specific information (student-block bootstrap)")

    # ---------------------------------------------------------- set selection
    model = cc.get("crossfit_model", "LIGHTGBM")
    cand = met[met["feature_set"].isin([s for s in cc.get("crossfit_feature_sets", []) if s != "META"])]
    cand_m = cand[cand["model"] == model]
    if cand_m.empty:
        model = "LOGISTIC"
        cand_m = cand[cand["model"] == model]
    selected = str(cand_m.sort_values("logloss").iloc[0]["feature_set"]) if not cand_m.empty else ("COMBINED" if bank.set_available("COMBINED") else "META")
    ctx.save_json({"selected_set": selected, "model": model,
                   "rule": "lowest clip-level OOF log loss of the pre-specified crossfit model among crossfit_feature_sets (excluding META)"},
                  S, "selection")
    conf_mats = {}
    pa = oofs.get((selected, model), oofs.get((selected, best.get(selected))))
    if pa is not None:
        from sklearn.metrics import confusion_matrix

        for t in sorted(cr["task"].unique()):
            mm = (cr["task"] == t).to_numpy() & np.isfinite(pa)
            conf_mats[t] = confusion_matrix(y[mm].astype(int), (pa[mm] >= 0.5).astype(int), labels=[0, 1])
        save_figure_v2(ctx, S, "confusion_by_task", V.confusion_grid(conf_mats, f"Correctness confusion ({selected})"),
                       pd.DataFrame([{"task": t, "tn": m_[0, 0], "fp": m_[0, 1], "fn": m_[1, 0], "tp": m_[1, 1]} for t, m_ in conf_mats.items()]),
                       "Threshold 0.5")
        save_figure_v2(ctx, S, "soft_correctness_distributions",
                       V.distributions_by_group({"officially correct": pa[y == 1], "officially incorrect": pa[y == 0]},
                                                f"P(correct) by official score ({selected})", "P(correct)"),
                       pd.DataFrame({"y": y, "p": pa}), "Soft correctness distributions")

    # ----------------------------------------------------------- student level
    ev = StudentLevelEvaluator(ctx, S)
    sch = [s for s in cc.get("crossfit_schemes", ["repeated_kfold"]) if s in reg.schemes()]
    base = {sc: ev.evaluate(ev.baseline_spec(), sc) for sc in sch}
    ev.add_provider("SOFTMETA", soft_provider(ctx, bank, "META", "SOFTMETA", model))
    meta_spec = CompactSpec("P17_SOFTMETA", "RIDGE", providers=["SOFTMETA"], budget=int(ctx.cfg.get_path("v2.evaluation.default_budget", 20)), fusion="residual")
    stu_rows = []
    rws, meta_res = evaluate_against(ev, meta_spec, base, sch)
    stu_rows += rws
    for s in [x for x in cc.get("crossfit_feature_sets", []) if x != "META" and bank.set_available(x)]:
        key = f"SOFT_{s}"
        ev.add_provider(key, soft_provider(ctx, bank, s, "SOFT", model))
        spec = CompactSpec(f"P17_{key}", "RIDGE", providers=[key], budget=int(ctx.cfg.get_path("v2.evaluation.default_budget", 20)), fusion="residual")
        rws, res_by = evaluate_against(ev, spec, base, sch)
        for r in rws:
            if r["scheme"] in meta_res and meta_res[r["scheme"]].status == "ok" and res_by[r["scheme"]].status == "ok":
                d = ev.compare(meta_res[r["scheme"]], res_by[r["scheme"]])
                r.update({"delta_vs_meta": d.get("delta_r2"), "delta_vs_meta_lo": d.get("delta_ci_low"), "delta_vs_meta_hi": d.get("delta_ci_high")})
        stu_rows += rws
    stu = pd.DataFrame(stu_rows)
    ctx.save_table(stu, S, "student_soft_results")
    if not stu.empty and "delta_r2" in stu:
        d = stu[stu["scheme"] == "repeated_kfold"].dropna(subset=["delta_r2"])
        if not d.empty:
            save_figure_v2(ctx, S, "student_incremental_r2", V.forest(d.sort_values("delta_r2", ascending=False), "delta_r2", "delta_ci_low",
                           "delta_ci_high", "name", "Soft-correctness features: ΔR² over the comparator"), d, "Student-level incremental R²")

    # ------------------------------------------------- student residual analysis
    if "repeated_kfold" in base and base["repeated_kfold"].status == "ok":
        resid = baseline_residuals(ev, base["repeated_kfold"])
        key = f"SOFT_{selected}"
        if key not in ev.providers and bank.set_available(selected):
            ev.add_provider(key, soft_provider(ctx, bank, selected, "SOFT", model))
        if key in ev.providers:
            T = ev.providers[key].table(ev.all_train, ev.all_train)
            disc, conf = discovery_split(ev.all_train, float(ctx.cfg.get_path("v2.validation.discovery_fraction", 0.5)), ctx.seed)
            assoc = association_table(T, resid, disc, conf, int(ctx.cfg.get_path("analysis.permutations", 1000)), ctx.seed)
            ctx.save_table(assoc, S, "student_residual_associations")
            top = assoc[assoc["tier"] == "exploratory"].dropna(subset=["p"]).sort_values("p").head(1)
            if not top.empty:
                f = top.iloc[0]["feature"]
                save_figure_v2(ctx, S, "residual_vs_top_soft_feature", V.viz.scatter_fit(T[f].reindex(resid.index), resid, f,
                               "comparator OOF residual", "Top soft feature vs residual (exploratory)"),
                               pd.DataFrame({ID: resid.index, f: T[f].reindex(resid.index).to_numpy(), "resid": resid.to_numpy()}),
                               "Exploratory: discovery-half screen; see confirmatory rows in the table")
            v2r.register_features(list(T.columns), S, key, f"soft correctness ({selected})")
    if not gains.empty and (gains["feature_set"] == selected).any():
        gr = gains[gains["feature_set"] == selected].iloc[0]
        v2r.update_hypothesis("V2H03", "supported" if gr["ci_low"] > 0 else "not_supported", S,
                              f"{selected} ΔAUROC over metadata-only = {gr['delta_auroc']:.4f}", effect=float(gr["delta_auroc"]),
                              ci_low=float(gr["ci_low"]), ci_high=float(gr["ci_high"]))
    sel_row = stu[(stu["name"] == f"P17_SOFT_{selected}") & (stu["scheme"] == "repeated_kfold")] if not stu.empty else stu
    if not sel_row.empty and "delta_ci_low" in sel_row and pd.notna(sel_row.iloc[0].get("delta_ci_low")):
        r = sel_row.iloc[0]
        v2r.update_hypothesis("V2H04", "supported" if r["delta_ci_low"] > 0 else "not_supported", S,
                              f"soft features ({selected}) ΔR²={r['delta_r2']:.4f}", effect=float(r["delta_r2"]),
                              ci_low=float(r["delta_ci_low"]), ci_high=float(r["delta_ci_high"]))
    return {"selected_set": selected, "model": model, "n_clip_models": len(oofs), "sets": sets}
