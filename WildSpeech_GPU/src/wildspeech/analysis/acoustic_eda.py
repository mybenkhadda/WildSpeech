"""P02 — acoustic EDA on clip-level features (train only).

Questions answered:
  * How do features distribute by task / instrument / original sample rate (artifact screen)?
  * Do correct and incorrect responses to the SAME item differ acoustically (item-centred contrasts)?
  * Which clip features, averaged per student and task, associate with the outcome?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID, TARGET
from ..stats import bh_fdr, cliffs_delta, eta_squared, spearman


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "::" in c and not c.startswith(("META::", "TXT::")) and pd.api.types.is_numeric_dtype(df[c])]


def merge_clip_table(train_clips: pd.DataFrame, students: pd.DataFrame, feats: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keep = [ID] + [c for c in ("instrument", "original_sample_rate", "school_id") if c in students]
    return (train_clips.merge(feats, on=FILE, how="inner").merge(students[keep], on=ID, how="left")
            .merge(labels, on=ID, how="left"))


def feature_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df[cols]
    s = pd.DataFrame({"n": d.notna().sum(), "missing_frac": d.isna().mean(), "mean": d.mean(), "sd": d.std(),
                      "p01": d.quantile(.01), "median": d.median(), "p99": d.quantile(.99)})
    s["family"] = [c.split("::")[0] for c in s.index]
    return s.rename_axis("feature").reset_index()


def group_effects(df: pd.DataFrame, cols: list[str], factors=("task", "instrument", "original_sample_rate", "school_id")) -> pd.DataFrame:
    rows = []
    for c in cols:
        for f in factors:
            if f in df:
                e, p = eta_squared(df[c], df[f])
                rows.append({"feature": c, "factor": f, "eta2": e, "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"])
    return out


def correctness_contrasts(df: pd.DataFrame, cols: list[str], min_per_group: int = 5) -> pd.DataFrame:
    """Item-centred correct-vs-incorrect differences: subtract the item mean, then compare groups per task."""
    d = df[df["item_score"].notna()].copy()
    rows = []
    for t, g in d.groupby("task"):
        cent = g[cols] - g.groupby("item_id")[cols].transform("mean")
        cor, inc = g["item_score"] == 1, g["item_score"] == 0
        if cor.sum() < min_per_group or inc.sum() < min_per_group:
            continue
        from scipy.stats import mannwhitneyu

        for c in cols:
            a, b = cent.loc[cor, c].dropna(), cent.loc[inc, c].dropna()
            if len(a) < min_per_group or len(b) < min_per_group:
                continue
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            sd = cent[c].std()
            rows.append({"task": t, "feature": c, "n_correct": len(a), "n_incorrect": len(b),
                         "diff_item_centred": a.mean() - b.mean(), "std_diff": (a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                         "cliffs_delta": cliffs_delta(a, b), "p": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"])
        out = out.sort_values("p")
    return out


def student_level_associations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Student x task means of clip features vs outcome (descriptive; P07 does the rigorous residual test)."""
    g = df.groupby([ID, "task"])[cols].mean()
    y = df.groupby(ID)[TARGET].first()
    rows = []
    for t in g.index.get_level_values("task").unique():
        gt = g.xs(t, level="task")
        yt = y.reindex(gt.index)
        for c in cols:
            r, p, n = spearman(gt[c], yt)
            rows.append({"task": t, "feature": c, "spearman": r, "p": p, "n": n})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"])
        out = out.sort_values("p")
    return out


def bandlimit_by_rate(df: pd.DataFrame) -> pd.DataFrame:
    c = "QUALITY::bandlimit_hz"
    if c not in df or "original_sample_rate" not in df:
        return pd.DataFrame()
    return df.groupby("original_sample_rate")[c].describe().reset_index()
