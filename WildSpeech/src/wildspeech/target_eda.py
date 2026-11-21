"""Section 02 - target variable intelligence.

Scientific question: what determines the distribution and predictability of
end-of-year reading outcomes? Instruments are never assumed interchangeable:
their distributions and their relationships with screening are compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data_access as da
from . import visualization as viz
from .stats_utils import bimodality_coefficient, quantile_bins, variance_decomposition


def _describe(y: pd.Series) -> dict:
    y = y.dropna()
    if len(y) == 0:
        return {"n": 0}
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    mad = float((y - y.median()).abs().median())
    mn, mx = y.min(), y.max()
    return {
        "n": int(len(y)), "mean": float(y.mean()), "std": float(y.std()), "median": float(y.median()), "mad": mad,
        "min": float(mn), "max": float(mx), "iqr": float(iqr), "skewness": float(sps.skew(y)) if len(y) > 2 else np.nan,
        "excess_kurtosis": float(sps.kurtosis(y)) if len(y) > 3 else np.nan,
        "bimodality_coefficient": bimodality_coefficient(y),
        "n_outliers_iqr": int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum()),
        "n_outliers_mad": int(((y - y.median()).abs() > 3.5 * 1.4826 * mad).sum()) if mad > 0 else 0,
        "floor_fraction": float((y == mn).mean()), "ceiling_fraction": float((y == mx).mean()),
        "n_unique": int(y.nunique()),
    }


def gmm_modes(y: pd.Series, seed: int) -> dict:
    from sklearn.mixture import GaussianMixture

    x = y.dropna().to_numpy().reshape(-1, 1)
    if len(x) < 30:
        return {"status": "INSUFFICIENT_DATA"}
    bics = {}
    for k in range(1, 5):
        g = GaussianMixture(k, random_state=seed, n_init=3).fit(x)
        bics[k] = float(g.bic(x))
    best = min(bics, key=bics.get)
    return {"bic_by_components": bics, "bic_best_components": best,
            "note": "BIC preference for >1 component suggests, but does not prove, multimodality."}


def heteroscedasticity(y: pd.Series, x: pd.Series) -> dict:
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 30:
        return {"status": "INSUFFICIENT_DATA"}
    yy, xx = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    beta = np.polyfit(xx, yy, 1)
    r2 = (yy - np.polyval(beta, xx)) ** 2
    aux = np.polyfit(xx, r2, 1)
    fitted = np.polyval(aux, xx)
    ss_res = ((r2 - fitted) ** 2).sum()
    ss_tot = ((r2 - r2.mean()) ** 2).sum()
    rsq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    lm = len(d) * rsq
    return {"breusch_pagan_lm": float(lm), "p": float(sps.chi2.sf(lm, 1)), "n": int(len(d))}


def stage_target_intelligence(ctx) -> dict:
    cfg = ctx.config
    sm = da.students(ctx, "train")
    y = pd.to_numeric(sm["target"], errors="coerce")
    sm = sm.assign(target=y)
    thetas = da.raw_theta_columns(sm)
    if thetas:
        sm["theta_mean_raw"] = sm[thetas].mean(axis=1)
    summary = {"overall": _describe(y), "missing_outcomes": int(y.isna().sum()),
               "gmm": gmm_modes(y, int(cfg["seed"]))}
    quant = y.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).rename("value").reset_index().rename(columns={"index": "quantile"})
    ctx.save_df(quant, "target_quantiles", fmt="csv")
    groups = {}
    for col in ("instrument", "school_id"):
        if col in sm.columns and sm[col].notna().any():
            tab = sm.groupby(col)["target"].apply(lambda s: pd.Series(_describe(s))).unstack()
            groups[col] = tab
            ctx.save_df(tab.reset_index(), f"target_by_{col}", fmt="csv")
    decomp = {}
    if "school_id" in sm.columns and sm["school_id"].nunique() > 1:
        decomp["school"] = variance_decomposition(y, sm["school_id"])
        if "instrument" in sm.columns and sm["instrument"].nunique() > 1:
            decomp["school_within_instrument"] = {
                str(inst): variance_decomposition(g["target"], g["school_id"]) for inst, g in sm.groupby("instrument")}
    if "instrument" in sm.columns and sm["instrument"].nunique() > 1:
        decomp["instrument"] = variance_decomposition(y, sm["instrument"])
        valid = [g["target"].dropna() for _, g in sm.groupby("instrument") if g["target"].notna().sum() >= 5]
        if len(valid) >= 2:
            summary["instrument_kruskal"] = dict(zip(["h", "p"], map(float, sps.kruskal(*valid))))
            summary["instrument_levene"] = dict(zip(["w", "p"], map(float, sps.levene(*valid))))
            if len(valid) == 2:
                summary["instrument_ks"] = dict(zip(["d", "p"], map(float, sps.ks_2samp(*valid))))
        if thetas:
            rel = []
            for inst, g in sm.groupby("instrument"):
                d = g[["theta_mean_raw", "target"]].dropna()
                if len(d) >= 10:
                    slope = np.polyfit(d["theta_mean_raw"], d["target"], 1)[0]
                    rel.append({"instrument": inst, "n": len(d), "slope_target_on_theta_mean": float(slope),
                                "spearman": float(sps.spearmanr(d["theta_mean_raw"], d["target"])[0])})
            summary["instrument_screening_relationship"] = rel
    if thetas:
        summary["heteroscedasticity_vs_theta_mean"] = heteroscedasticity(sm["target"], sm["theta_mean_raw"])
        sm["screening_band"] = quantile_bins(sm["theta_mean_raw"], 5, int(cfg["min_group_size"]), "S")
        ctx.save_df(sm.groupby("screening_band")["target"].apply(lambda s: pd.Series(_describe(s))).unstack().reset_index(),
                    "target_by_screening_band", fmt="csv")
    cm = da.clips(ctx, "train")
    completion = cm.groupby("student_id").agg(n_clips=("clip_id", "size"), n_tasks=("task_family", "nunique"))
    sm = sm.merge(completion, left_on="student_id", right_index=True, how="left")
    sm["completion_band"] = quantile_bins(sm["n_clips"], 4, int(cfg["min_group_size"]), "C")
    summary["variance_decomposition"] = decomp
    ctx.save_json(summary, "target_summary")
    _target_figures(ctx, sm, summary)
    return {"outputs": ["target_summary", "target_quantiles"], "summary": {"n": summary["overall"].get("n"),
                                                                          "skewness": summary["overall"].get("skewness")}}


def _target_figures(ctx, sm: pd.DataFrame, summary: dict) -> None:
    y = sm["target"]
    n = int(y.notna().sum())
    fig, axes = viz.new_figure(1, 3, (5, 4))
    viz.hist_kde(axes[0, 0], y, "end-of-year outcome")
    viz.ecdf(axes[0, 1], y, "end-of-year outcome")
    viz.qq(axes[0, 2], y, "end-of-year outcome")
    o = summary["overall"]
    ctx.figure(fig, "target_distribution", "End-of-year outcome distribution",
               "What is the shape of the outcome distribution?", "student_master", n,
               f"mean={o.get('mean', np.nan):.3g}, sd={o.get('std', np.nan):.3g}, skew={o.get('skewness', np.nan):.2f}, "
               f"floor={o.get('floor_fraction', np.nan):.1%}, ceiling={o.get('ceiling_fraction', np.nan):.1%}.")
    for col in ("instrument", "school_id", "screening_band", "completion_band"):
        if col in sm.columns and sm[col].notna().any():
            fig, axes = viz.new_figure(1, 2, (max(6, sm[col].nunique() * 0.3 + 3), 4.5))
            viz.grouped_box(axes[0, 0], sm, "target", col, "outcome", violin=False)
            viz.grouped_box(axes[0, 1], sm, "target", col, "outcome", violin=True)
            ctx.figure(fig, f"target_by_{col}", f"Outcome by {col}", f"Does the outcome differ by {col}?",
                       "student_master", n, f"Groups with fewer than 3 students are omitted ({sm[col].nunique()} levels).")
    if "instrument" in sm.columns and sm["instrument"].nunique() > 1 and "theta_mean_raw" in sm.columns:
        fig, axes = viz.new_figure(1, 1)
        ax = axes[0, 0]
        for inst, g in sm.groupby("instrument"):
            ax.scatter(g["theta_mean_raw"], g["target"], s=8, alpha=0.5, label=f"{inst} (n={len(g)})")
        ax.legend(fontsize=7)
        viz.finalize_axis(ax, "Outcome vs mean screening theta by instrument", "mean screening theta", "outcome", n)
        ctx.figure(fig, "instrument_screening_relationship", "Instrument comparison",
                   "Are instruments interchangeable given screening?", "student_master", n,
                   "; ".join(f"{r['instrument']}: slope={r['slope_target_on_theta_mean']:.3g}"
                             for r in summary.get("instrument_screening_relationship", [])))
