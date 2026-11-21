"""Fold-fitted item / ability / recording conditioning.

v3_clip_frame(ctx)            V2 clip frame (item, task, domain, log duration, linguistic complexity, item_score)
                              joined with screening theta per student (feature-safe, present in train and test)
ItemReference                 fitted on TRAINING-FOLD clips only:
    difficulty_               item p(correct) shrunk toward the task mean (k pseudo-counts); unseen -> task mean
    ability_edges_            quantile edges of the student screening ability (mean theta) -> strata
    residualizer_             ItemConditionedResidualizer (controls: item, domain, duration, complexity) per feature
  transform(clips, F)        -> item-conditioned z-scores, difficulty, ability stratum per clip
condition_mask(c, correct=1)  rows with official correctness == value (None = all scored rows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID


def v3_clip_frame(ctx) -> pd.DataFrame:
    key = "v3_clip_frame"
    if key in ctx._cache:
        return ctx._cache[key]
    from ..features.conditional_aggregation import clip_frame

    c = clip_frame(ctx).copy()
    b = ctx.bundle()
    st = b.all_students.drop_duplicates(ID).set_index(ID)
    th = [t for t in b.theta_cols if t in st.columns]
    if th:
        c["ability"] = c[ID].map(st[th].astype(float).mean(1))
    else:
        c["ability"] = np.nan
    if "original_sample_rate" in st:
        c["orig_sr"] = c[ID].map(st["original_sample_rate"]).astype(float)
    ctx._cache[key] = c
    return c


class ItemReference:
    def __init__(self, controls=("item", "domain", "duration", "complexity"), shrinkage_k: float = 10.0,
                 ability_bins: int = 5):
        self.controls, self.k, self.ability_bins = tuple(controls), float(shrinkage_k), int(ability_bins)

    def fit(self, c: pd.DataFrame, F: np.ndarray | None = None) -> "ItemReference":
        from ..features.conditional_aggregation import ItemConditionedResidualizer

        s = c[c["is_scored"] == 1]
        task_p = s.groupby("task")["item_score"].mean()
        g = s.groupby("item_id")["item_score"]
        n, m = g.count(), g.mean()
        tp = s.drop_duplicates("item_id").set_index("item_id")["task"].map(task_p).reindex(m.index)
        self.difficulty_ = (n * m + self.k * tp) / (n + self.k)
        self.task_p_ = task_p
        ab = c.drop_duplicates(ID)["ability"].astype(float).dropna()
        qs = np.linspace(0, 1, self.ability_bins + 1)[1:-1]
        self.ability_edges_ = np.unique(np.quantile(ab, qs)) if len(ab) > self.ability_bins else np.array([])
        self.residualizer_ = None
        if F is not None and F.shape[1]:
            self.residualizer_ = ItemConditionedResidualizer(self.controls, self.k).fit(c, F)
        return self

    def difficulty(self, c: pd.DataFrame) -> np.ndarray:
        d = c["item_id"].map(self.difficulty_).astype(float)
        return d.fillna(c["task"].map(self.task_p_)).fillna(float(self.task_p_.mean())).to_numpy()

    def ability_stratum(self, c: pd.DataFrame) -> np.ndarray:
        a = c["ability"].astype(float).to_numpy()
        out = np.digitize(a, self.ability_edges_) if len(self.ability_edges_) else np.zeros(len(a), int)
        return np.where(np.isfinite(a), out, -1)

    def transform(self, c: pd.DataFrame, F: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        Z = self.residualizer_.transform(c, F) if (self.residualizer_ is not None and F is not None) else None
        return Z, self.difficulty(c), self.ability_stratum(c)


def condition_mask(c: pd.DataFrame, correct: int | None = 1) -> np.ndarray:
    s = c["item_score"].astype(float).to_numpy()
    if correct is None:
        return np.isfinite(s)
    return s == float(correct)


def clip_design(ctx, wide: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Clip frame restricted to clips with V3 features -> (frame, feature matrix aligned to it)."""
    c = v3_clip_frame(ctx)
    m = c.merge(wide[[FILE] + cols], on=FILE, how="inner").reset_index(drop=True)
    return m, m[cols].to_numpy(float)
