"""Shared analysis engine for acoustic feature families (Sections 07-10).

For every feature of a family it reports distributions, clip-level
associations with correctness (within item), item and student ICCs,
relationships with expected-text complexity and item difficulty, and
student-level associations with the target and (when available) the
out-of-fold screening residual, including partial correlations controlling for
screening ability and recording quality, and item-adjusted variants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .stats_utils import bh_fdr, bootstrap_ci, hedges_g, icc1, partial_corr, safe_corr


def _family_features(clips: pd.DataFrame, prefixes: tuple) -> list:
    feats = []
    for c in clips.columns:
        if not c.startswith(prefixes) or not pd.api.types.is_numeric_dtype(clips[c]):
            continue
        s = clips[c]
        if s.notna().mean() < 0.05 or s.nunique(dropna=True) < 3:
            continue
        feats.append(c)
    return feats


def _complexity_column(clips: pd.DataFrame) -> str | None:
    for c in ("l_n_phonemes", "l_n_chars", "l_n_words"):
        if c in clips.columns and clips[c].notna().mean() > 0.5:
            return c
    return None


def _student_context(ctx) -> pd.DataFrame:
    y = da.target(ctx).rename("target")
    out = y.to_frame()
    scr = da.screening(ctx)
    if scr is not None:
        for c in ["theta_mean"] + da.raw_theta_columns(scr):
            if c in scr.columns:
                out[c] = scr[c]
    res = da.residuals(ctx)
    if res is not None:
        out["residual"] = res
    return out


def analyze_family(ctx, family: str, prefixes: tuple, split_by_correctness: bool = False,
                   control_quality: bool = False) -> dict:
    cfg = ctx.config
    min_n = int(cfg["min_n_correlation"])
    clips = da.clip_table(ctx, "train")
    feats = _family_features(clips, prefixes)
    tag = family.lower()
    if not feats:
        ctx.save_json({"status": "NO_FEATURES", "family": family}, f"{tag}_family_status")
        return {"outputs": [f"{tag}_family_status"], "summary": {"status": "NO_FEATURES"}}
    has_correct = "correct" in clips.columns and clips["correct"].notna().sum() >= min_n
    has_item = "item_key" in clips.columns and clips["item_key"].notna().any()
    cplx = _complexity_column(clips)
    stud_ctx = _student_context(ctx)
    theta_cols = [c for c in stud_ctx.columns if c.startswith("theta_") and c != "theta_mean"]
    quality_cols = [c for c in ("q_snr_db_approx", "q_duration_s", "q_noise_floor_db") if c in clips.columns]
    smeans = clips.groupby("student_id")[feats + quality_cols].mean()
    if has_item:
        item_means = clips.groupby("item_key")[feats].transform("mean")
        adj = clips[feats] - item_means
        adj["student_id"] = clips["student_id"]
        smeans_adj = adj.groupby("student_id")[feats].mean()
        if has_correct:
            corr_dm = clips["correct"] - clips.groupby("item_key")["correct"].transform("mean")
    else:
        smeans_adj = None
    S = stud_ctx.join(smeans, how="inner")
    S_adj = stud_ctx.join(smeans_adj, how="inner") if smeans_adj is not None else None
    split_means = {}
    if split_by_correctness and has_correct:
        for label, val in (("correct", 1.0), ("incorrect", 0.0)):
            split_means[label] = stud_ctx.join(clips[clips["correct"] == val].groupby("student_id")[feats].mean(), how="inner")
    rows = []
    for f in feats:
        x = clips[f]
        row = {"feature": f, "family": family, "n_clips": int(x.notna().sum()), "missing_rate": float(x.isna().mean()),
               "mean": float(x.mean()), "std": float(x.std()), "median": float(x.median()),
               "p05": float(x.quantile(0.05)), "p95": float(x.quantile(0.95)), "skew": float(x.skew())}
        row["student_icc"] = icc1(x, clips["student_id"])
        if has_item:
            row["item_icc"] = icc1(x, clips["item_key"])
            if has_correct:
                xd = x - item_means[f]
                row["within_item_correct_r"], row["within_item_correct_p"], _ = safe_corr(xd, corr_dm, "pearson", min_n)
                row["within_item_correct_g"] = hedges_g(xd[clips["correct"] == 1], xd[clips["correct"] == 0])
        elif has_correct:
            row["within_item_correct_r"], row["within_item_correct_p"], _ = safe_corr(x, clips["correct"], "pearson", min_n)
        if cplx:
            row[f"rho_{cplx}"] = safe_corr(x, clips[cplx], "spearman", min_n)[0]
        if "item_b" in clips.columns:
            row["rho_item_difficulty"] = safe_corr(x, clips["item_b"], "spearman", min_n)[0]
        for tf in sorted(clips["task_family"].dropna().unique()):
            m = clips["task_family"] == tf
            if has_correct and has_item and m.sum() >= min_n:
                row[f"within_item_correct_r__{tf}"] = safe_corr((x - item_means[f])[m], corr_dm[m], "pearson", min_n)[0]
        row["target_rho"], row["target_p"], row["n_students"] = safe_corr(S[f], S["target"], "spearman", min_n)
        if theta_cols:
            row["target_partial_r_theta"], row["target_partial_p_theta"], _ = partial_corr(S[f], S["target"], S[theta_cols], "spearman", min_n)
        if control_quality and quality_cols and theta_cols:
            ctrl = [c for c in theta_cols] + [q for q in quality_cols if q != f]
            row["target_partial_r_theta_quality"], row["target_partial_p_theta_quality"], _ = partial_corr(
                S[f], S["target"], S[ctrl], "spearman", min_n)
        if "theta_mean" in S.columns:
            row["theta_mean_rho"] = safe_corr(S[f], S["theta_mean"], "spearman", min_n)[0]
        if "residual" in S.columns:
            row["residual_rho"], row["residual_p"], _ = safe_corr(S[f], S["residual"], "spearman", min_n)
        if S_adj is not None:
            row["item_adjusted_target_rho"] = safe_corr(S_adj[f], S_adj["target"], "spearman", min_n)[0]
            if "residual" in S_adj.columns:
                row["item_adjusted_residual_rho"], row["item_adjusted_residual_p"], _ = safe_corr(
                    S_adj[f], S_adj["residual"], "spearman", min_n)
        for label, frame in split_means.items():
            row[f"target_rho_{label}_only"] = safe_corr(frame[f], frame["target"], "spearman", min_n)[0]
        rows.append(row)
    report = pd.DataFrame(rows)
    for pcol in [c for c in report.columns if c.endswith("_p") or c.endswith("_p_theta") or c.endswith("_p_theta_quality")]:
        report[pcol.replace("_p", "_q", 1) if pcol.endswith("_p") else pcol + "_fdr"] = bh_fdr(report[pcol])
    ctx.save_df(report, f"{tag}_family_report", also_csv=True)
    _family_figures(ctx, family, report, clips, S, feats)
    top = report.reindex(report["target_rho"].abs().sort_values(ascending=False).index).head(10)
    summary = {"n_features": len(feats), "top_target_features": top[["feature", "target_rho"]].to_dict("records")}
    if "residual_rho" in report.columns:
        summary["n_residual_fdr_significant"] = int((report.get("residual_q", pd.Series(dtype=float)) < cfg["fdr_alpha"]).sum())
    return {"outputs": [f"{tag}_family_report"], "summary": summary}


def _family_figures(ctx, family: str, report: pd.DataFrame, clips: pd.DataFrame, S: pd.DataFrame, feats: list) -> None:
    tag = family.lower()
    k = 25
    if "within_item_correct_r" in report.columns and report["within_item_correct_r"].notna().any():
        top = report.reindex(report["within_item_correct_r"].abs().sort_values(ascending=False).index).head(k)
        fig, axes = viz.new_figure(1, 1, (8, 0.25 * len(top) + 2))
        ax = axes[0, 0]
        ax.barh(top["feature"], top["within_item_correct_r"])
        ax.invert_yaxis()
        viz.finalize_axis(ax, f"{family}: within-item association with correctness", "Pearson r (item-demeaned)", "", int(top["n_clips"].max()))
        ctx.figure(fig, f"{tag}_correctness_association", f"{family} features vs correctness (within item)",
                   "Do acoustic properties differ between correct and incorrect responses to the same item?",
                   f"{tag}_family_report", int(report["n_clips"].max()),
                   f"Largest |r|={top['within_item_correct_r'].abs().max():.3f} ({top.iloc[0]['feature']}).")
    top_t = report.reindex(report["target_rho"].abs().sort_values(ascending=False).index).head(k)
    if top_t["target_rho"].notna().any():
        fig, axes = viz.new_figure(1, 1, (8, 0.25 * len(top_t) + 2))
        ax = axes[0, 0]
        ax.barh(top_t["feature"], top_t["target_rho"], label="raw")
        if "residual_rho" in top_t.columns:
            ax.barh(top_t["feature"], top_t["residual_rho"], alpha=0.6, label="OOF screening residual")
        ax.legend(fontsize=7)
        ax.invert_yaxis()
        viz.finalize_axis(ax, f"{family}: student-level association with outcome", "Spearman rho", "", int(top_t["n_students"].max()))
        ctx.figure(fig, f"{tag}_target_association", f"{family} features vs end-of-year outcome",
                   "Which features relate to the outcome and to the screening residual?", f"{tag}_family_report",
                   int(top_t["n_students"].max()),
                   f"Top raw rho {top_t['target_rho'].abs().max():.3f}; residual associations shown when available.")
    show = top_t["feature"].head(6).tolist()
    if show:
        fig, axes = viz.new_figure(2, 3, (4.5, 3.2))
        for ax, f in zip(axes.ravel(), show):
            viz.hist_kde(ax, clips[f], f)
        ctx.figure(fig, f"{tag}_distributions", f"{family}: clip-level distributions of top features",
                   "What do the most outcome-related features look like?", f"{tag}_family_report",
                   int(len(clips)), "Histograms with KDE; titles report missing values.")
    viz.top_correlation_heatmap(ctx, S, top_t["feature"].tolist() + ["target"], f"{tag}_top_correlations",
                                f"{family}: redundancy among outcome-related features",
                                "Are top features redundant?", f"{tag}_family_report")


def complexity_surface(ctx, feature: str, outputs: list) -> None:
    from .student_aggregation import shrunk_slopes

    clips = da.clip_table(ctx, "train")
    cplx = _complexity_column(clips) or ("item_b" if "item_b" in clips.columns else None)
    if cplx is None or feature not in clips.columns:
        ctx.save_json({"status": "UNAVAILABLE", "reason": "no expected-text complexity or difficulty"}, "complexity_surface_status")
        outputs.append("complexity_surface_status")
        return
    scr = da.screening(ctx)
    if scr is not None and "theta_mean" in scr.columns:
        clips = clips.merge(scr[["theta_mean"]], left_on="student_id", right_index=True, how="left")
        surf = viz.binned_surface(clips, cplx, "theta_mean", feature, bins=6, min_count=int(ctx.config["min_group_size"]))
        if not surf.empty:
            fig, axes = viz.new_figure(1, 1, (7, 5))
            viz.heatmap(axes[0, 0], surf, f"mean {feature} by {cplx} (x) and screening theta_mean (y)", cmap="viridis",
                        cbar_label=feature)
            ctx.figure(fig, f"surface_{feature}_{cplx}", "Complexity-duration response surface",
                       "How does response duration depend on item complexity and initial ability?", "clip_table",
                       int(clips[[cplx, "theta_mean", feature]].dropna().shape[0]),
                       f"Cells with fewer than {ctx.config['min_group_size']} clips are masked.")
    slopes = shrunk_slopes(clips, "student_id", cplx, feature, k=float(ctx.config["slope_shrinkage_k"]),
                           min_obs=int(ctx.config["aggregation_min_obs"]))
    y = da.target(ctx)
    slopes = slopes.join(y.rename("target"), how="left")
    from .stats_utils import safe_corr

    r, p, n = safe_corr(slopes["slope_shrunk"], slopes["target"], "spearman", int(ctx.config["min_n_correlation"]))
    ctx.save_df(slopes.reset_index(), "complexity_sensitivity_estimates", also_csv=True)
    outputs.append("complexity_sensitivity_estimates")
    fig, axes = viz.new_figure(1, 2)
    viz.hist_kde(axes[0, 0], slopes["slope_shrunk"], f"shrunk slope of {feature} on {cplx}")
    viz.scatter_fit(axes[0, 1], slopes["slope_shrunk"], slopes["target"], "complexity sensitivity", "target",
                    "complexity sensitivity vs outcome")
    ctx.figure(fig, f"complexity_sensitivity_{feature}", "Student-level complexity sensitivity",
               "Does sensitivity to item complexity relate to later reading?", "complexity_sensitivity_estimates", n,
               f"Empirical-Bayes shrunk slopes (k={ctx.config['slope_shrinkage_k']}); Spearman rho with target={r:.3f} (p={p:.3g}).")


def multiscale_comparisons(ctx, prefixes: tuple, outputs: list) -> None:
    from scipy.stats import kruskal

    cfg = ctx.config
    clips = da.clip_table(ctx, "train")
    feats = _family_features(clips, prefixes)
    if not feats:
        return
    stud = _student_context(ctx)
    means = stud.join(clips.groupby("student_id")[feats].mean(), how="inner")
    rows = []
    has_correct = "correct" in clips.columns and clips["correct"].notna().any() and "item_key" in clips.columns
    for f in feats:
        row = {"feature": f}
        if has_correct:
            xd = clips[f] - clips.groupby("item_key")[f].transform("mean")
            row["correct_vs_incorrect_g_within_item"] = hedges_g(xd[clips["correct"] == 1], xd[clips["correct"] == 0])
        if "item_key" in clips.columns:
            row["same_item_icc"] = icc1(clips[f], clips["item_key"])
        if "theta_mean" in means.columns:
            terc = pd.qcut(means["theta_mean"], 3, labels=["low", "mid", "high"], duplicates="drop")
            groups = [means.loc[terc == g, f].dropna() for g in terc.dropna().unique()]
            groups = [g for g in groups if len(g) >= 5]
            if len(groups) >= 2:
                h, p = kruskal(*groups)
                row["ability_tertile_kruskal_h"], row["ability_tertile_p"] = float(h), float(p)
        row["outcome_rho"] = safe_corr(means[f], means["target"], "spearman", int(cfg["min_n_correlation"]))[0]
        if "residual" in means.columns and "theta_mean" in means.columns:
            lo_q, hi_q = cfg["residual_group_quantiles"]
            lo, hi = means["residual"].quantile(lo_q), means["residual"].quantile(hi_q)
            bins = pd.qcut(means["theta_mean"], 5, duplicates="drop")
            diffs, weights = [], []
            for _, g in means.groupby(bins, observed=True):
                pos, neg = g.loc[g["residual"] >= hi, f], g.loc[g["residual"] <= lo, f]
                gg = hedges_g(pos, neg)
                if np.isfinite(gg):
                    diffs.append(gg)
                    weights.append(min(len(pos.dropna()), len(neg.dropna())))
            row["similar_screening_pos_vs_neg_residual_g"] = float(np.average(diffs, weights=weights)) if diffs else np.nan
            sub = means[["residual", f]].dropna()
            ci = bootstrap_ci(lambda a, b_: safe_corr(a, b_, "spearman", 10)[0], sub[f].to_numpy(), sub["residual"].to_numpy(),
                              n_boot=min(int(cfg["bootstrap_iterations"]), 300), seed=int(cfg["seed"]))
            row["residual_rho_ci_lower"], row["residual_rho_ci_upper"] = ci["lower"], ci["upper"]
        rows.append(row)
    table = pd.DataFrame(rows)
    ctx.save_df(table, "multiscale_comparisons", also_csv=True)
    outputs.append("multiscale_comparisons")
    if "correct_vs_incorrect_g_within_item" in table.columns:
        top = table.reindex(table["correct_vs_incorrect_g_within_item"].abs().sort_values(ascending=False).index).head(20)
        fig, axes = viz.new_figure(1, 1, (8, 0.25 * len(top) + 2))
        axes[0, 0].barh(top["feature"], top["correct_vs_incorrect_g_within_item"])
        axes[0, 0].invert_yaxis()
        viz.finalize_axis(axes[0, 0], "Wavelet features: correct vs incorrect (within item)", "Hedges g", "", int(len(clips)))
        ctx.figure(fig, "wavelet_correct_vs_incorrect", "Multiscale differences between correct and incorrect responses",
                   "Do multiscale transitions differ by correctness for the same item?", "multiscale_comparisons",
                   int(len(clips)), f"Max |g|={top['correct_vs_incorrect_g_within_item'].abs().max():.3f}.")
