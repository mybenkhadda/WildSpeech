"""Linguistic-complexity sensitivity: how strongly a student's clip-level measurements change with the
complexity of the administered items (per task, shrunk toward zero for few items).

slope_{s,t}(v) = cov(v, x) / var(x) * n/(n+k), x = log1p(n_phonemes) of the item, computed over the student's
clips of task t. Label-free (item_score is official metadata in both splits), so no nesting is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE, ID
from .builder import sanitize
from .conditional_aggregation import FoldFeatureProvider, clip_frame


def complexity_slopes(c: pd.DataFrame, V: np.ndarray, names: list[str], prefix: str, complexity_col: str = "n_phonemes",
                      shrink_k: float = 5.0, min_n: int = 3) -> pd.DataFrame:
    x = np.log1p(c[complexity_col].astype(float).to_numpy())
    X = np.asarray(V, float)
    M = np.isfinite(X) & np.isfinite(x)[:, None]
    X0 = np.where(M, X, 0.0)
    xx = np.where(M, x[:, None], 0.0)
    keys = pd.MultiIndex.from_arrays([c[ID].astype(str).to_numpy(), c["task"].to_numpy()], names=[ID, "task"])
    names = [sanitize(n) for n in names]
    parts = {k: pd.DataFrame(v, index=keys, columns=names).groupby(level=[0, 1]).sum()
             for k, v in {"n": M.astype(float), "sv": X0, "sx": xx, "svx": X0 * xx, "sxx": xx * xx}.items()}
    n = parts["n"]
    cov = parts["svx"] - parts["sv"] * parts["sx"] / n.replace(0, np.nan)
    var = parts["sxx"] - parts["sx"] ** 2 / n.replace(0, np.nan)
    slope = (cov / var.where(var > 1e-9)) * (n / (n + shrink_k))
    slope = slope.where(n >= min_n)
    w = slope.unstack("task")
    w.columns = [f"{prefix}__{t}__{f}__cpx_slope" for f, t in w.columns]
    w.index = w.index.astype(str)
    w.index.name = ID
    return w


class ComplexitySensitivityProvider(FoldFeatureProvider):
    """Complexity slopes of correctness, duration and any supplied clip metrics (e.g. PER, GOP, DTW deviation)."""

    family = "CPX"

    def __init__(self, ctx, name: str, clip_metrics: pd.DataFrame | None = None, metric_cols: list[str] | None = None,
                 shrink_k: float = 5.0):
        super().__init__(ctx, name, "complexity sensitivity slopes")
        self.cm = clip_metrics
        self.cols = [c for c in (metric_cols or []) if clip_metrics is not None and c in clip_metrics.columns]
        self.k = float(shrink_k)

    def fingerprint_payload(self) -> dict:
        cfr = clip_frame(self.ctx)
        n = 0 if self.cm is None else int(self.cm[FILE].isin(set(cfr.loc[cfr["split"] == "train", FILE])).sum())
        return {"cols": self.cols, "k": self.k, "n_train": n}

    def fit_table(self, train_ids, target_ids) -> pd.DataFrame:
        c = clip_frame(self.ctx)
        ids = set(train_ids) | set(target_ids)
        c = c[c[ID].isin(ids)].reset_index(drop=True)
        if self.cm is not None and self.cols:
            c = c.merge(self.cm[[FILE] + self.cols].drop_duplicates(FILE), on=FILE, how="left")
        cols = ["item_score", "log_dur"] + self.cols
        names = ["accuracy", "log_duration"] + [x.replace("::", "_").lower() for x in self.cols]
        out = complexity_slopes(c, c[cols].to_numpy(float), names, self.family, shrink_k=self.k)
        return out.reindex(sorted(ids))
