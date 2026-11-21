"""Validation schemes. Every fold is a partition of STUDENTS (never recordings).

repeated_kfold        stratified (outcome quantile bins) K-fold, repeated with different seeds
school_kfold          K-fold over schools (GroupKFold balanced by student count)
loso                  leave-one-school-out (schools smaller than min_school_size pooled into one fold)
instrument_transfer   train on one instrument, validate on the other(s)
sample_rate_transfer  train on 16 kHz-origin students, validate on low-rate-origin students and vice versa
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    scheme: str
    repeat: int
    fold: int
    train_ids: tuple
    valid_ids: tuple
    label: str = ""

    def check(self) -> "Fold":
        if set(self.train_ids) & set(self.valid_ids):
            raise AssertionError(f"{self.scheme} r{self.repeat} f{self.fold}: student overlap")
        return self


def stratified_folds(y: pd.Series, n_splits: int, seed: int, n_bins: int = 5) -> list[tuple[list[str], list[str]]]:
    from sklearn.model_selection import StratifiedKFold

    ids = np.asarray(y.index.astype(str))
    vals = y.to_numpy(float)
    bins = pd.qcut(pd.Series(vals).rank(method="first"), q=min(n_bins, max(2, len(vals) // n_splits)), labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(list(ids[tr]), list(ids[va])) for tr, va in skf.split(ids, bins)]


def repeated_kfold(y: pd.Series, n_splits: int, n_repeats: int, seed: int, n_bins: int = 5) -> list[Fold]:
    out = []
    for r in range(n_repeats):
        for k, (tr, va) in enumerate(stratified_folds(y, n_splits, seed + 1000 * r, n_bins)):
            out.append(Fold("repeated_kfold", r, k, tuple(tr), tuple(va)).check())
    return out


def school_kfold(groups: pd.Series, n_splits: int, seed: int) -> list[Fold]:
    """Assign whole schools to folds, largest first, always to the currently smallest fold."""
    g = groups.astype(str)
    sizes = g.value_counts()
    rng = np.random.default_rng(seed)
    order = sizes.sample(frac=1.0, random_state=int(rng.integers(1e9))).sort_values(ascending=False, kind="mergesort")
    load = np.zeros(n_splits)
    assign = {}
    for school, n in order.items():
        f = int(np.argmin(load))
        assign[school] = f
        load[f] += n
    fold_of = g.map(assign)
    ids = g.index.astype(str)
    return [Fold("school_kfold", 0, k, tuple(ids[fold_of != k]), tuple(ids[fold_of == k])).check() for k in range(n_splits)]


def loso(groups: pd.Series, min_size: int) -> list[Fold]:
    g = groups.astype(str)
    sizes = g.value_counts()
    small = set(sizes[sizes < min_size].index)
    gg = g.where(~g.isin(small), "__pooled_small__")
    ids = g.index.astype(str)
    return [Fold("loso", 0, k, tuple(ids[gg != s]), tuple(ids[gg == s]), label=str(s)).check()
            for k, s in enumerate(sorted(gg.unique()))]


def attribute_transfer(attr: pd.Series, scheme: str) -> list[Fold]:
    a = attr.astype(str)
    ids = a.index.astype(str)
    vals = sorted(a.unique())
    if len(vals) < 2:
        return []
    return [Fold(scheme, 0, k, tuple(ids[a != v]), tuple(ids[a == v]), label=f"valid={v}").check()
            for k, v in enumerate(vals)]


def build_folds(scheme: str, train: pd.DataFrame, y: pd.Series, cfg, seed: int) -> list[Fold]:
    v = cfg.section("validation")
    tr = train.set_index("subject_id")
    if scheme == "repeated_kfold":
        return repeated_kfold(y, int(v.get("n_splits", 5)), int(v.get("n_repeats", 3)), seed, int(v.get("stratify_bins", 5)))
    if scheme == "school_kfold":
        if "school_id" not in tr:
            return []
        return school_kfold(tr["school_id"].reindex(y.index), int(v.get("n_splits", 5)), seed)
    if scheme == "loso":
        if "school_id" not in tr:
            return []
        return loso(tr["school_id"].reindex(y.index), int(v.get("min_school_size", 8)))
    if scheme == "instrument_transfer":
        return attribute_transfer(tr["instrument"].reindex(y.index), scheme) if "instrument" in tr else []
    if scheme == "sample_rate_transfer":
        if "original_sample_rate" not in tr:
            return []
        lab = np.where(tr["original_sample_rate"].reindex(y.index) < 16000, "low_rate", "16k")
        return attribute_transfer(pd.Series(lab, index=y.index), scheme)
    raise ValueError(f"unknown scheme {scheme}")
