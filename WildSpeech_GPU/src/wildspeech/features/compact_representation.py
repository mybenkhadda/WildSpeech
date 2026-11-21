"""Compact student-level representations, fitted inside each training fold.

FamilyReducer
    for each added family: impute + standardise (train fold) -> PCA to `family_components` (train fold)
    -> union of reduced columns -> supervised top-`budget` selection by |Spearman| with the selection target
    (by default the residual of the baseline model on the training fold). Never sees validation rows.
IndexTransformer  (redesigned literacy indices)
    component columns matched by recipe patterns; each component's DIRECTION is validated on the training fold
    (sign of Spearman with the outcome, kept only if |rho| >= min_abs_rho); index = mean of signed z-scores of
    >= min_components components; Cronbach's alpha of the signed components is recorded per fold.
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..models.preprocessing import TabularPreprocessor
from ..stats import cronbach_alpha


def spearman_vector(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Spearman| of every column of X with y (NaN-safe via median fill), vectorised."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.size == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0)
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med[None, :])
    yr = rankdata(y)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    Xr = np.apply_along_axis(rankdata, 0, X)
    Xr = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
    return (Xr * yr[:, None]).mean(0)


class FamilyReducer:
    def __init__(self, groups: dict[str, list[str]], budget: int | None, family_components: int = 10,
                 reducer: str = "pca_then_supervised", seed: int = 0, max_missing_frac: float = 0.6):
        self.groups = {k: list(v) for k, v in groups.items() if v}
        self.budget, self.k, self.reducer, self.seed, self.mmf = budget, int(family_components), reducer, seed, max_missing_frac

    def fit(self, X: pd.DataFrame, y_sel) -> "FamilyReducer":
        from sklearn.decomposition import PCA

        self.preps_, self.pcas_, parts = {}, {}, []
        for fam, cols in self.groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            prep = TabularPreprocessor(max_missing_frac=self.mmf, add_indicators=False, scale=True).fit(X[cols])
            if not prep.cols_:
                continue
            Z = prep.transform(X[cols])
            self.preps_[fam] = (cols, prep)
            if self.reducer in ("pca_family", "pca_then_supervised") and Z.shape[1] > self.k and len(Z) > self.k + 2:
                pca = PCA(n_components=self.k, random_state=self.seed).fit(Z.to_numpy(float))
                self.pcas_[fam] = pca
                Z = pd.DataFrame(pca.transform(Z.to_numpy(float)), index=Z.index,
                                 columns=[f"{fam}__all__pc{j}__value" for j in range(self.k)])
            parts.append(Z)
        A = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
        self.cols_ = list(A.columns)
        if self.budget is not None and len(self.cols_) > int(self.budget) and self.reducer in ("supervised", "pca_then_supervised"):
            score = np.abs(spearman_vector(A.to_numpy(float), np.asarray(y_sel, float)))
            order = np.argsort(-score, kind="mergesort")[: int(self.budget)]
            self.cols_ = [self.cols_[i] for i in order]
            self.scores_ = pd.Series(score, index=A.columns)
        elif self.budget is not None and len(self.cols_) > int(self.budget):
            self.cols_ = self.cols_[: int(self.budget)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for fam, (cols, prep) in self.preps_.items():
            Z = prep.transform(X.reindex(columns=cols))
            if fam in self.pcas_:
                pca = self.pcas_[fam]
                Z = pd.DataFrame(pca.transform(Z.to_numpy(float)), index=Z.index,
                                 columns=[f"{fam}__all__pc{j}__value" for j in range(pca.n_components_)])
            parts.append(Z)
        A = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
        return A.reindex(columns=self.cols_).astype(np.float32)


def match_columns(columns, patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        out += [c for c in columns if fnmatch.fnmatchcase(c, p) and c not in out]
    return out


class IndexTransformer:
    def __init__(self, recipes: dict, min_components: int = 2, min_abs_rho: float = 0.0):
        self.recipes, self.min_components, self.min_abs_rho = recipes, int(min_components), float(min_abs_rho)

    def fit(self, X: pd.DataFrame, y) -> "IndexTransformer":
        y = np.asarray(y, float)
        self.indices_, rows = {}, []
        for name, spec in (self.recipes or {}).items():
            cols = match_columns(X.columns, list(spec.get("include", [])))
            if len(cols) < self.min_components:
                rows.append({"index": name, "n_components": len(cols), "status": "insufficient_components"})
                continue
            sub = X[cols].astype(float)
            mu, sd = sub.mean(), sub.std(ddof=0).replace(0, np.nan)
            Z = ((sub - mu) / sd).fillna(0.0)
            rho = pd.Series(spearman_vector(Z.to_numpy(float), y), index=cols)
            keep = rho[rho.abs() >= self.min_abs_rho].index.tolist()
            if len(keep) < self.min_components:
                rows.append({"index": name, "n_components": len(keep), "status": "insufficient_validated_components"})
                continue
            signs = np.sign(rho[keep]).replace(0, 1.0)
            alpha = cronbach_alpha((Z[keep] * signs).to_numpy(float))
            self.indices_[name] = (keep, mu[keep], sd[keep], signs)
            rows.append({"index": name, "n_components": len(keep), "status": "ok", "cronbach_alpha": alpha,
                         "components": ";".join(f"{c}:{int(signs[c])}" for c in keep)})
        self.report_ = pd.DataFrame(rows)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for name, (keep, mu, sd, signs) in self.indices_.items():
            Z = ((X.reindex(columns=keep).astype(float) - mu) / sd).fillna(0.0)
            out[f"IDX__all__{name}__value"] = (Z * signs).mean(axis=1)
        return out.astype(np.float32)
