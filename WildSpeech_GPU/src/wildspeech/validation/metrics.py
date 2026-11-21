"""Metrics: official R2 + RMSE/MAE, repeat-level summaries, incremental R2 and subgroup diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import delta_r2_bootstrap, r2


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


def metric_row(y, p) -> dict:
    return {"r2": r2(y, p), "rmse": rmse(y, p), "mae": mae(y, p), "n": int(len(y)),
            "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan}


def summarize_oof(oof: pd.DataFrame) -> dict:
    """R2 per repeat on the full OOF vector, then mean/sd across repeats; also the repeat-averaged OOF."""
    per = oof.groupby("repeat").apply(lambda d: pd.Series(metric_row(d["y_true"], d["y_pred"])))
    avg = oof.groupby("subject_id").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    m = metric_row(avg["y_true"], avg["y_pred"])
    return {"r2_mean": float(per["r2"].mean()), "r2_sd": float(per["r2"].std(ddof=0)) if len(per) > 1 else 0.0,
            "rmse_mean": float(per["rmse"].mean()), "mae_mean": float(per["mae"].mean()),
            "r2_repeat_avg_pred": m["r2"], "n_students": int(avg.shape[0]), "n_repeats": int(per.shape[0])}


def mean_oof(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.groupby("subject_id").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))


def incremental_r2(oof_base: pd.DataFrame, oof_new: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    a, b = mean_oof(oof_base), mean_oof(oof_new)
    j = a.join(b, lsuffix="_base", rsuffix="_new", how="inner")
    est, lo, hi = delta_r2_bootstrap(j["y_true_base"], j["y_pred_base"], j["y_pred_new"], n_boot, seed)
    # repeat-paired delta (same folds across models when the same seed is used)
    rep = None
    if {"repeat"} <= set(oof_base.columns) and {"repeat"} <= set(oof_new.columns):
        ra = oof_base.groupby("repeat").apply(lambda d: r2(d["y_true"], d["y_pred"]))
        rb = oof_new.groupby("repeat").apply(lambda d: r2(d["y_true"], d["y_pred"]))
        rep = (rb - ra).reindex(ra.index)
    return {"delta_r2": est, "delta_ci_low": lo, "delta_ci_high": hi,
            "delta_r2_repeat_mean": float(rep.mean()) if rep is not None else np.nan,
            "delta_r2_repeat_min": float(rep.min()) if rep is not None else np.nan}


def subgroup_metrics(oof: pd.DataFrame, meta: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = mean_oof(oof).join(meta.set_index("subject_id")[cols], how="left")
    rows = []
    for c in cols:
        for g, d in m.groupby(m[c].astype(str)):
            if len(d) < 5:
                continue
            rows.append({"group_col": c, "group": g, **metric_row(d["y_true"], d["y_pred"]),
                         "bias": float((d["y_pred"] - d["y_true"]).mean())})
    m["y_q"] = pd.qcut(m["y_true"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    for g, d in m.groupby("y_q", observed=True):
        rows.append({"group_col": "outcome_quintile", "group": str(g), **metric_row(d["y_true"], d["y_pred"]),
                     "bias": float((d["y_pred"] - d["y_true"]).mean())})
    return pd.DataFrame(rows)
