"""Section 22 - failure phenotype atlas.

Scientific question: which student profiles are systematically difficult to
predict, and why? Outcome-based failure categories describe prediction
errors; they are not diagnoses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz

PHENOTYPES = ["F01_UNEXPECTED_HIGH_OUTCOME", "F02_UNEXPECTED_LOW_OUTCOME", "F03_SKILL_IMBALANCE", "F04_RESPONSE_INCONSISTENCY",
              "F05_SPARSE_OBSERVATIONS", "F06_RECORDING_QUALITY", "F07_SCHOOL_DOMAIN_SHIFT", "F08_EXTREME_TARGET",
              "F09_COMPLEXITY_SENSITIVITY", "F10_PHONETIC_DIFFICULTY", "F11_MODEL_DISAGREEMENT", "F12_MISSING_MODALITIES"]


def _multimodal_oof(ctx) -> tuple[pd.Series | None, str]:
    if ctx.stage_available("21"):
        r = ctx.load_df("residual_learning_oof", required=False)
        if r is not None and len(r):
            g = r[(r["approach"] == "TWO_STAGE_RESIDUAL") & (r["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD")]
            if len(g):
                return g.groupby("student_id")["pred"].mean(), "TWO_STAGE_RESIDUAL"
    if ctx.stage_available("20"):
        o = ctx.load_df("ablation_oof_predictions", required=False)
        if o is not None and len(o):
            for exp in ("A22_ALL_VALIDATED_FEATURES", "A13_TABULAR_MULTIMODAL", "A05_COMPLETE_TABULAR"):
                g = o[(o["experiment"] == exp) & (o["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD") & (o["model"] == ctx.config["ablation_primary_model"])]
                if len(g):
                    return g.groupby("student_id")["pred"].mean(), exp
    return None, "UNAVAILABLE"


def stage_failure_atlas(ctx) -> dict:
    from sklearn.mixture import GaussianMixture

    cfg = ctx.config
    base = ctx.load_df("baseline_residuals").set_index("student_id")
    multi, multi_name = _multimodal_oof(ctx)
    S = da.student_features(ctx, "train") if ctx.stage_available("13") else pd.DataFrame(index=base.index)
    meta = da.student_meta(ctx, "train")
    D = base[["y", "oof_prediction"]].rename(columns={"y": "observed_outcome", "oof_prediction": "baseline_prediction"})
    D["multimodal_prediction"] = multi.reindex(D.index) if multi is not None else np.nan
    D["baseline_error"] = D["baseline_prediction"] - D["observed_outcome"]
    D["multimodal_error"] = D["multimodal_prediction"] - D["observed_outcome"]
    D["error_reduction"] = D["baseline_error"].abs() - D["multimodal_error"].abs()
    oof_all = ctx.load_df("baseline_oof_predictions")
    prim = oof_all[(oof_all["model_id"] == cfg["primary_baseline"]) & (oof_all["scheme"] == cfg["primary_cv_scheme"])]
    persist = prim.assign(abs_err=(prim["pred"] - prim["y"]).abs())
    thr = persist["abs_err"].quantile(0.8)
    D["error_persistence"] = persist.groupby("student_id")["abs_err"].apply(lambda s: float((s >= thr).mean())).reindex(D.index)
    preds = prim.pivot_table(index="student_id", columns="repeat", values="pred")
    models = oof_all[oof_all["scheme"] == cfg["primary_cv_scheme"]].groupby(["student_id", "model_id"])["pred"].mean().unstack()
    D["model_disagreement"] = models.drop(columns=["M00_MEAN"], errors="ignore").std(axis=1).reindex(D.index)
    D["repeat_prediction_sd"] = preds.std(axis=1).reindex(D.index)
    J = D.join(S, how="left").join(meta, rsuffix="_meta")
    err = D["baseline_error"]
    q_hi, q_lo = err.quantile(0.9), err.quantile(0.1)
    flags = pd.DataFrame(index=D.index)
    flags["F01_UNEXPECTED_HIGH_OUTCOME"] = err <= q_lo
    flags["F02_UNEXPECTED_LOW_OUTCOME"] = err >= q_hi
    flags["F03_SKILL_IMBALANCE"] = J.get("theta_skill_imbalance", pd.Series(np.nan, index=D.index)) >= J.get("theta_skill_imbalance", pd.Series(np.nan, index=D.index)).quantile(0.9)
    inc = J[[c for c in J.columns if c.startswith("ia_item_adjusted_acoustic_variability")]].mean(axis=1) if any(c.startswith("ia_item_adjusted_acoustic_variability") for c in J.columns) else pd.Series(np.nan, index=D.index)
    flags["F04_RESPONSE_INCONSISTENCY"] = inc >= inc.quantile(0.9)
    flags["F05_SPARSE_OBSERVATIONS"] = J.get("meta_n_clips", pd.Series(np.nan, index=D.index)) <= J.get("meta_n_clips", pd.Series(np.nan, index=D.index)).quantile(0.1)
    flags["F06_RECORDING_QUALITY"] = J.get("meta_frac_quality_flagged", pd.Series(np.nan, index=D.index)) >= 0.5
    if "school_id" in J.columns and J["school_id"].notna().any():
        school_err = D["baseline_error"].abs().groupby(J["school_id"]).mean()
        flags["F07_SCHOOL_DOMAIN_SHIFT"] = J["school_id"].map(school_err) >= school_err.quantile(0.9)
    else:
        flags["F07_SCHOOL_DOMAIN_SHIFT"] = False
    flags["F08_EXTREME_TARGET"] = (D["observed_outcome"] <= D["observed_outcome"].quantile(0.05)) | (D["observed_outcome"] >= D["observed_outcome"].quantile(0.95))
    cx = J[[c for c in J.columns if c.startswith("cx_t_active_dur__vs_") and c.endswith("__all")]]
    flags["F09_COMPLEXITY_SENSITIVITY"] = (cx.iloc[:, 0] >= cx.iloc[:, 0].quantile(0.9)) if cx.shape[1] else False
    ph = J[[c for c in J.columns if c.startswith("ph_") and c.endswith("_phone_er__mean")]]
    flags["F10_PHONETIC_DIFFICULTY"] = (ph.iloc[:, 0] >= ph.iloc[:, 0].quantile(0.9)) if ph.shape[1] else False
    flags["F11_MODEL_DISAGREEMENT"] = D["model_disagreement"] >= D["model_disagreement"].quantile(0.9)
    miss_cols = [c for c in J.columns if c.startswith("meta_missing_")]
    flags["F12_MISSING_MODALITIES"] = J[miss_cols].sum(axis=1) >= 2 if miss_cols else False
    flags = flags.fillna(False).astype(bool)
    hard = D["baseline_error"].abs() >= D["baseline_error"].abs().quantile(0.8)
    feats = ["baseline_error", "error_persistence", "model_disagreement"] + [c for c in ("theta_mean", "theta_skill_imbalance", "acc_all",
                                                                                     "meta_n_clips", "q_snr_db_approx__mean", "t_active_dur__mean") if c in J.columns]
    X = J.loc[hard, feats].apply(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0).fillna(0)
    D["failure_cluster"] = "NOT_HARD"
    cluster_info = {}
    if len(X) >= 30:
        best, best_bic = None, np.inf
        for k in range(2, min(6, len(X) // 10) + 1):
            g = GaussianMixture(k, random_state=int(cfg["seed"]), n_init=3).fit(X)
            if g.bic(X) < best_bic:
                best, best_bic = g, g.bic(X)
        if best is not None:
            D.loc[X.index, "failure_cluster"] = [f"FC{l}" for l in best.predict(X)]
            cluster_info = {"k": int(best.n_components), "bic": float(best_bic), "features": feats}
    def mech(row):
        f = [p for p in PHENOTYPES if flags.loc[row.name, p]]
        return ";".join(f) if f else "NO_PREDEFINED_PATTERN"
    D["candidate_mechanism"] = D.apply(mech, axis=1)
    D["available_evidence"] = D.apply(lambda r: ";".join(c for c in ("baseline", "multimodal" if np.isfinite(r["multimodal_prediction"]) else None,
                                                                   "audio" if "t_active_dur__mean" in J.columns and np.isfinite(J.loc[r.name].get("t_active_dur__mean", np.nan)) else None,
                                                                   "phonetic" if ph.shape[1] and np.isfinite(ph.iloc[:, 0].get(r.name, np.nan)) else None) if c), axis=1)
    reg = D.reset_index().rename(columns={"index": "student_id"})
    reg.insert(0, "failure_id", [f"FAIL_{i:05d}" for i in range(len(reg))])
    cols = ["failure_id", "student_id", "baseline_prediction", "multimodal_prediction", "observed_outcome", "baseline_error",
            "multimodal_error", "error_reduction", "failure_cluster", "candidate_mechanism", "available_evidence",
            "error_persistence", "model_disagreement", "repeat_prediction_sd"]
    reg = reg[cols]
    ctx.save_df(reg, "failure_registry")
    summary = []
    for p in PHENOTYPES:
        m = flags[p]
        summary.append({"phenotype": p, "n": int(m.sum()), "mean_abs_baseline_error": float(D.loc[m, "baseline_error"].abs().mean()) if m.any() else np.nan,
                        "mean_error_reduction": float(D.loc[m, "error_reduction"].mean()) if m.any() else np.nan,
                        "overall_mean_abs_baseline_error": float(D["baseline_error"].abs().mean())})
    summary = pd.DataFrame(summary)
    ctx.save_df(summary, "failure_summary", fmt="csv")
    overlap = flags.T.astype(int) @ flags.astype(int)
    ctx.save_df(overlap.reset_index().rename(columns={"index": "phenotype"}), "failure_overlap", fmt="csv")
    ctx.save_json({"multimodal_source": multi_name, "clusters": cluster_info,
                   "note": "Outcome-based failure categories are descriptive, not diagnoses."}, "failure_info")
    _failure_figures(ctx, D, summary, X, flags)
    return {"outputs": ["failure_registry", "failure_summary", "failure_overlap", "failure_info"],
            "summary": {"multimodal_source": multi_name, "n_hard": int(hard.sum()),
                        "mean_error_reduction": float(D["error_reduction"].mean()) if D["error_reduction"].notna().any() else None}}


def _failure_figures(ctx, D, summary, X, flags) -> None:
    n = int(len(D))
    e = D["baseline_error"].abs().sort_values(ascending=False)
    fig, axes = viz.new_figure(1, 3, (5, 4))
    share = (e ** 2).cumsum() / (e ** 2).sum()
    axes[0, 0].plot(np.arange(1, len(e) + 1) / len(e), share.to_numpy())
    viz.finalize_axis(axes[0, 0], "Error concentration curve", "fraction of students (sorted by |error|)", "share of squared error", n)
    if D["multimodal_error"].notna().any():
        axes[0, 1].scatter(D["baseline_error"].abs(), D["multimodal_error"].abs(), s=6, alpha=0.5)
        lim = float(np.nanmax(D[["baseline_error", "multimodal_error"]].abs().to_numpy()))
        axes[0, 1].plot([0, lim], [0, lim], "--", color="grey")
        viz.finalize_axis(axes[0, 1], "Baseline vs multimodal |error|", "baseline |error|", "multimodal |error|", n)
    axes[0, 2].scatter(D["model_disagreement"], D["baseline_error"].abs(), s=6, alpha=0.5)
    viz.finalize_axis(axes[0, 2], "Model disagreement vs error", "SD across baseline models", "|baseline error|", n)
    ctx.figure(fig, "failure_overview", "Failure atlas overview", "Which students are hard to predict?", "failure_registry", n,
               f"Top 10% of students account for {share.iloc[max(0, int(0.1 * len(share)) - 1)]:.1%} of squared error.")
    if len(summary):
        fig, axes = viz.new_figure(1, 1, (8, 5))
        s = summary.set_index("phenotype")[["mean_abs_baseline_error", "mean_error_reduction"]]
        s.plot(kind="barh", ax=axes[0, 0])
        viz.finalize_axis(axes[0, 0], "Failure phenotypes: error and multimodal recovery", "outcome units", "", n)
        ctx.figure(fig, "failure_recovery", "Model recovery by failure phenotype", "Which modalities recover baseline failures?",
                   "failure_summary", n, "Positive error reduction means the multimodal model reduced |error|.")
        M = flags.groupby(D["failure_cluster"]).mean().T
        fig, axes = viz.new_figure(1, 1, (6, 6))
        viz.heatmap(axes[0, 0], M, "Phenotype prevalence by empirical failure cluster", cmap="viridis", vmin=0, vmax=1)
        ctx.figure(fig, "failure_heatmap", "Error heatmap by failure cluster", "Do empirical failure clusters match predefined phenotypes?",
                   "failure_registry", n, "Empirical clusters fitted on hard-to-predict students (top 20% |error|).")
    if len(X) >= 10:
        _, _, vt = np.linalg.svd(X.to_numpy() - X.to_numpy().mean(0), full_matrices=False)
        P = (X.to_numpy() - X.to_numpy().mean(0)) @ vt[:2].T
        fig, axes = viz.new_figure(1, 1)
        labs = D.loc[X.index, "failure_cluster"].astype("category")
        axes[0, 0].scatter(P[:, 0], P[:, 1], c=labs.cat.codes, cmap="tab10", s=8)
        viz.finalize_axis(axes[0, 0], "Failure embedding map (PCA of hard students)", "PC1", "PC2", len(X))
        ctx.figure(fig, "failure_embedding", "Failure embedding map", "How do hard-to-predict students group?", "failure_registry",
                   len(X), "Colours: empirical failure clusters.")
