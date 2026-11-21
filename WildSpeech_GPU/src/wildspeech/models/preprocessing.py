"""Fold-fitted tabular preprocessing and supervised selection (fit on training-fold students only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class TabularPreprocessor:
    """Drop near-empty/constant columns, median-impute, add missing indicators, standardise."""

    def __init__(self, max_missing_frac: float = 0.6, add_indicators: bool = True, max_indicators: int = 50, scale: bool = True):
        self.max_missing_frac, self.add_indicators, self.max_indicators, self.scale = max_missing_frac, add_indicators, max_indicators, scale

    def fit(self, X: pd.DataFrame, y=None) -> "TabularPreprocessor":
        miss = X.isna().mean()
        keep = miss[miss <= self.max_missing_frac].index
        Xk = X[keep]
        std = Xk.std(ddof=0)
        keep = std[std > 1e-12].index
        self.cols_ = list(keep)
        self.median_ = X[self.cols_].median()
        m = miss.reindex(self.cols_)
        self.ind_cols_ = list(m[(m > 0.02)].sort_values(ascending=False).index[: self.max_indicators]) if self.add_indicators else []
        Xi = X[self.cols_].fillna(self.median_)
        self.mean_, self.std_ = Xi.mean(), Xi.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xi = X.reindex(columns=self.cols_).fillna(self.median_)
        if self.scale:
            Xi = (Xi - self.mean_) / self.std_
        if self.ind_cols_:
            ind = X.reindex(columns=self.ind_cols_).isna().astype(float)
            ind.columns = [f"{c}__isna" for c in self.ind_cols_]
            Xi = pd.concat([Xi, ind], axis=1)
        return Xi.astype(np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class SupervisedSelector:
    """Keep the top-k columns by |Spearman| with the training target; always keep protected prefixes."""

    def __init__(self, k: int | None, keep_prefixes=("SCREENING__", "RECORDING__")):
        self.k, self.keep_prefixes = k, tuple(keep_prefixes)

    def fit(self, X: pd.DataFrame, y) -> "SupervisedSelector":
        cols = list(X.columns)
        if not self.k or len(cols) <= self.k:
            self.cols_ = cols
            return self
        yr = rankdata(np.asarray(y, float))
        yr = (yr - yr.mean()) / (yr.std() + 1e-12)
        Xr = np.apply_along_axis(rankdata, 0, X.to_numpy(float))
        Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
        score = np.abs((Xr * yr[:, None]).mean(0))
        protected = [c for c in cols if c.startswith(self.keep_prefixes)]
        order = [cols[i] for i in np.argsort(-score) if cols[i] not in protected]
        self.cols_ = protected + order[: max(0, self.k - len(protected))]
        self.scores_ = pd.Series(score, index=cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.cols_]


class FoldModel:
    """preprocessor -> selector -> estimator, all fitted on the training fold only."""

    def __init__(self, estimator, selector_k: int | None, max_missing_frac: float = 0.6, scale: bool = True):
        self.estimator = estimator
        self.prep = TabularPreprocessor(max_missing_frac=max_missing_frac, scale=scale)
        self.sel = SupervisedSelector(selector_k)

    def fit(self, X: pd.DataFrame, y) -> "FoldModel":
        Xp = self.prep.fit_transform(X)
        Xs = self.sel.fit(Xp, y).transform(Xp)
        self.estimator.fit(Xs, np.asarray(y, float))
        self.columns_ = list(Xs.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.sel.transform(self.prep.transform(X))
        return np.asarray(self.estimator.predict(Xs), float).ravel()
