"""Fusion estimators. Each is a self-contained sklearn-style regressor whose learned combination is fitted by
INNER cross-validation on the training fold only, so placing it inside the outer CV loop is fully nested.

LateFusionRegressor   one base model per column group (feature family or task) -> non-negative stacking
ResidualStackRegressor stage 1 on screening columns; stage 2 on the rest fitted to inner-OOF residuals;
                       shrinkage alpha chosen by inner CV
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from ..features.builder import family_of

ALPHAS = np.logspace(-2, 4, 31)


def _groups_by_family(columns, base_prefixes=("SCREENING", "RECORDING")) -> dict[str, list[str]]:
    base = [c for c in columns if family_of(c) in base_prefixes]
    groups = {"BASE": base}
    for c in columns:
        f = family_of(c)
        if f not in base_prefixes:
            groups.setdefault(f, list(base)).append(c)
    return groups


def _groups_by_task(columns, tasks, base_prefixes=("SCREENING", "RECORDING")) -> dict[str, list[str]]:
    base = [c for c in columns if family_of(c) in base_prefixes]
    groups = {"BASE": base}
    for t in tasks:
        cols = [c for c in columns if f"__{t}__" in c]
        if cols:
            groups[t] = base + cols
    return groups


class LateFusionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base=None, grouping: str = "family", tasks=("blending", "deletion", "sentence_repetition"),
                 inner_splits: int = 4, seed: int = 0):
        self.base, self.grouping, self.tasks, self.inner_splits, self.seed = base, grouping, tasks, inner_splits, seed

    def _base(self):
        return clone(self.base) if self.base is not None else RidgeCV(alphas=ALPHAS)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.groups_ = _groups_by_family(X.columns) if self.grouping == "family" else _groups_by_task(X.columns, self.tasks)
        self.groups_ = {k: v for k, v in self.groups_.items() if v}
        names = list(self.groups_)
        Z = np.zeros((len(y), len(names)))
        kf = KFold(self.inner_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            for j, g in enumerate(names):
                m = self._base().fit(X.iloc[tr][self.groups_[g]], y[tr])
                Z[va, j] = m.predict(X.iloc[va][self.groups_[g]])
        # non-negative weights + intercept on inner-OOF predictions
        Zc = Z - Z.mean(0)
        w, _ = nnls(Zc, y - y.mean())
        if w.sum() <= 0:
            w = np.zeros(len(names))
            w[0] = 1.0
        self.weights_ = pd.Series(w, index=names)
        self.intercept_ = float(y.mean() - Z.mean(0) @ w)
        self.models_ = {g: self._base().fit(X[self.groups_[g]], y) for g in names if self.weights_[g] > 0}
        return self

    def predict(self, X: pd.DataFrame):
        out = np.full(len(X), self.intercept_)
        for g, m in self.models_.items():
            out += self.weights_[g] * m.predict(X[self.groups_[g]])
        return out


class ResidualStackRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, stage2=None, base_prefixes=("SCREENING", "RECORDING"), alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 inner_splits: int = 4, seed: int = 0):
        self.stage2, self.base_prefixes, self.alphas, self.inner_splits, self.seed = stage2, base_prefixes, alphas, inner_splits, seed

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.c1_ = [c for c in X.columns if family_of(c) in self.base_prefixes]
        self.c2_ = [c for c in X.columns if c not in self.c1_]
        kf = KFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof1 = np.zeros(len(y))
        for tr, va in kf.split(X):
            oof1[va] = RidgeCV(alphas=ALPHAS).fit(X.iloc[tr][self.c1_], y[tr]).predict(X.iloc[va][self.c1_])
        resid = y - oof1
        # choose alpha by a second inner CV on the residual model
        s2 = np.zeros(len(y))
        if self.c2_:
            for tr, va in kf.split(X):
                m2 = (clone(self.stage2) if self.stage2 is not None else RidgeCV(alphas=ALPHAS)).fit(X.iloc[tr][self.c2_], resid[tr])
                s2[va] = m2.predict(X.iloc[va][self.c2_])
        errs = {a: np.mean((y - oof1 - a * s2) ** 2) for a in self.alphas}
        self.alpha_ = float(min(errs, key=errs.get))
        self.m1_ = RidgeCV(alphas=ALPHAS).fit(X[self.c1_], y)
        self.m2_ = None
        if self.c2_ and self.alpha_ > 0:
            self.m2_ = (clone(self.stage2) if self.stage2 is not None else RidgeCV(alphas=ALPHAS)).fit(X[self.c2_], resid)
        return self

    def predict(self, X: pd.DataFrame):
        p = self.m1_.predict(X[self.c1_])
        if self.m2_ is not None:
            p = p + self.alpha_ * self.m2_.predict(X[self.c2_])
        return p


def make_fusion(name: str, cfg, seed: int, params: dict):
    inner = int(cfg.get_path("validation.inner_splits", 4))
    base_name = params.pop("base", None)
    base = None
    if base_name:
        from .zoo import make_model

        base = make_model(base_name, cfg, seed)
    if name == "LATE_FUSION":
        return LateFusionRegressor(base=base, grouping="family", inner_splits=inner, seed=seed)
    if name == "TASK_ENSEMBLE":
        return LateFusionRegressor(base=base, grouping="task", inner_splits=inner, seed=seed)
    if name == "RESIDUAL_STACK":
        return ResidualStackRegressor(stage2=base, alphas=tuple(cfg.get_path("multimodal.residual_alphas", [0, .25, .5, .75, 1])),
                                      inner_splits=inner, seed=seed)
    raise ValueError(name)
