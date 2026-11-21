"""V2 fusion estimators — every learned combination is fitted by INNER cross-validation on the training fold.

convex_weights          non-negative weights summing to one (+ intercept) on inner-OOF predictions (SLSQP);
                        bounded weights cannot amplify an unstable member (V1 NNLS weights were unbounded).
NestedResidualRegressor stage 1 = baseline model on the baseline columns; stage 2 = model on the new columns
                        fitted to stage-1 inner-OOF residuals; shrinkage alpha chosen by inner CV.
ConvexLateFusion        one model per feature family (each on baseline + that family) -> convex stack.
RobustTaskEnsemble      repaired task ensemble: per-task models on baseline + task columns, inputs winsorised to
                        training-fold quantiles, ridge alphas up to 1e6, MISSING-TASK GATING (a student without a
                        task uses the baseline prediction in that slot), convex weights, output clipped to the
                        training target range +- margin.
RegularizedStack        several model types on the same columns -> convex stack.
DomainPartialPooling    lightweight mixture-of-experts: global ridge + per-recording-domain ridge on the global
                        residual, partially pooled with a shrinkage chosen by inner CV; domains below a minimum size
                        fall back to the global expert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from ..features.builder import family_of

ALPHAS = np.logspace(-2, 4, 31)
WIDE_ALPHAS = np.logspace(-2, 6, 41)


def _ridge(alphas=ALPHAS):
    return RidgeCV(alphas=alphas)


def convex_weights(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """argmin ||y - b - Z w||^2  s.t. w >= 0, sum(w) = 1."""
    from scipy.optimize import minimize

    Z, y = np.asarray(Z, float), np.asarray(y, float)
    k = Z.shape[1]
    if k == 1:
        return np.ones(1), float(np.mean(y - Z[:, 0]))
    Zc, yc = Z - Z.mean(0), y - y.mean()

    def f(w):
        r = yc - Zc @ w
        return float(r @ r)

    def g(w):
        return -2.0 * Zc.T @ (yc - Zc @ w)

    w0 = np.full(k, 1.0 / k)
    res = minimize(f, w0, jac=g, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success else w0, 0, None)
    w = w / w.sum() if w.sum() > 0 else w0
    return w, float(y.mean() - Z.mean(0) @ w)


def _kfold(n_splits: int, seed: int):
    return KFold(n_splits, shuffle=True, random_state=seed)


# ============================================================================ residual
class NestedResidualRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, stage1=None, stage2=None, stage1_cols=None, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 inner_splits: int = 4, seed: int = 0):
        self.stage1, self.stage2, self.stage1_cols, self.alphas, self.inner_splits, self.seed = stage1, stage2, stage1_cols, alphas, inner_splits, seed

    def _m1(self):
        return clone(self.stage1) if self.stage1 is not None else _ridge()

    def _m2(self):
        return clone(self.stage2) if self.stage2 is not None else _ridge()

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.c1_ = [c for c in (self.stage1_cols or []) if c in X.columns]
        self.c2_ = [c for c in X.columns if c not in set(self.c1_)]
        kf = _kfold(self.inner_splits, self.seed)
        oof1, s2 = np.zeros(len(y)), np.zeros(len(y))
        for tr, va in kf.split(X):
            oof1[va] = self._m1().fit(X.iloc[tr][self.c1_], y[tr]).predict(X.iloc[va][self.c1_])
        resid = y - oof1
        if self.c2_:
            for tr, va in kf.split(X):
                s2[va] = self._m2().fit(X.iloc[tr][self.c2_], resid[tr]).predict(X.iloc[va][self.c2_])
        errs = {a: float(np.mean((y - oof1 - a * s2) ** 2)) for a in self.alphas}
        self.alpha_ = float(min(errs, key=errs.get))
        self.inner_mse_ = errs
        self.m1_ = self._m1().fit(X[self.c1_], y)
        self.m2_ = self._m2().fit(X[self.c2_], resid) if (self.c2_ and self.alpha_ > 0) else None
        return self

    def predict(self, X: pd.DataFrame):
        p = np.asarray(self.m1_.predict(X[self.c1_]), float).ravel()
        if self.m2_ is not None:
            p = p + self.alpha_ * np.asarray(self.m2_.predict(X[self.c2_]), float).ravel()
        return p


# ========================================================================== late fusion
class ConvexLateFusion(BaseEstimator, RegressorMixin):
    def __init__(self, base_cols=None, base=None, inner_splits: int = 4, seed: int = 0):
        self.base_cols, self.base, self.inner_splits, self.seed = base_cols, base, inner_splits, seed

    def _m(self):
        return clone(self.base) if self.base is not None else _ridge()

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        base = [c for c in (self.base_cols or []) if c in X.columns]
        groups = {"BASE": base}
        for c in X.columns:
            if c not in set(base):
                groups.setdefault(family_of(c), list(base)).append(c)
        self.groups_ = {k: v for k, v in groups.items() if v}
        names = list(self.groups_)
        Z = np.zeros((len(y), len(names)))
        for tr, va in _kfold(self.inner_splits, self.seed).split(X):
            for j, g in enumerate(names):
                Z[va, j] = self._m().fit(X.iloc[tr][self.groups_[g]], y[tr]).predict(X.iloc[va][self.groups_[g]])
        w, b = convex_weights(Z, y)
        self.weights_, self.intercept_ = pd.Series(w, index=names), b
        self.models_ = {g: self._m().fit(X[self.groups_[g]], y) for g in names if self.weights_[g] > 1e-6}
        return self

    def predict(self, X: pd.DataFrame):
        out = np.full(len(X), self.intercept_)
        for g, m in self.models_.items():
            out += self.weights_[g] * np.asarray(m.predict(X[self.groups_[g]]), float).ravel()
        return out


# ===================================================================== task ensemble (fixed)
class RobustTaskEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_cols=None, tasks=("blending", "deletion", "sentence_repetition"), base=None,
                 inner_splits: int = 4, seed: int = 0, winsor_q: float = 0.01, clip_margin_sd: float = 1.0):
        self.base_cols, self.tasks, self.base, self.inner_splits, self.seed = base_cols, tasks, base, inner_splits, seed
        self.winsor_q, self.clip_margin_sd = winsor_q, clip_margin_sd

    def _m(self):
        return clone(self.base) if self.base is not None else _ridge(WIDE_ALPHAS)

    def _wins(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.lo_, upper=self.hi_, axis=1)

    def _avail(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {t: f"__avail__{t}" for t in self.tasks}
        return pd.DataFrame({t: (X[c].to_numpy(float) > 0.5) if c in X else np.ones(len(X), bool) for t, c in cols.items()},
                            index=X.index)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        feat = [c for c in X.columns if not c.startswith("__avail__")]
        self.lo_ = X[feat].quantile(self.winsor_q)
        self.hi_ = X[feat].quantile(1 - self.winsor_q)
        A = self._avail(X)
        Xw = self._wins(X[feat])
        base = [c for c in (self.base_cols or []) if c in feat]
        self.groups_ = {"BASE": base}
        for t in self.tasks:
            cols = [c for c in feat if f"__{t}__" in c and c not in set(base)]
            if cols:
                self.groups_[t] = base + cols
        names = list(self.groups_)
        Z = np.zeros((len(y), len(names)))
        for tr, va in _kfold(self.inner_splits, self.seed).split(Xw):
            for j, g in enumerate(names):
                tr_g = tr if g == "BASE" else tr[A.iloc[tr][g].to_numpy()]
                if len(tr_g) < 10:
                    tr_g = tr
                Z[va, j] = self._m().fit(Xw.iloc[tr_g][self.groups_[g]], y[tr_g]).predict(Xw.iloc[va][self.groups_[g]])
        Z = self._gate(Z, A, names)
        w, b = convex_weights(Z, y)
        self.weights_, self.intercept_ = pd.Series(w, index=names), b
        self.models_ = {}
        for g in names:
            rows = np.ones(len(y), bool) if g == "BASE" else A[g].to_numpy()
            if rows.sum() < 10:
                rows = np.ones(len(y), bool)
            self.models_[g] = self._m().fit(Xw[rows][self.groups_[g]], y[rows])
        self.y_lo_ = float(y.min() - self.clip_margin_sd * y.std())
        self.y_hi_ = float(y.max() + self.clip_margin_sd * y.std())
        return self

    @staticmethod
    def _gate(Z: np.ndarray, A: pd.DataFrame, names: list[str]) -> np.ndarray:
        Z = Z.copy()
        for j, g in enumerate(names):
            if g != "BASE" and g in A:
                miss = ~A[g].to_numpy()
                Z[miss, j] = Z[miss, 0]
        return Z

    def predict(self, X: pd.DataFrame):
        feat = list(self.lo_.index)
        Xw = self._wins(X.reindex(columns=feat))
        A = self._avail(X)
        names = list(self.groups_)
        Z = np.column_stack([np.asarray(self.models_[g].predict(Xw[self.groups_[g]]), float).ravel() for g in names])
        Z = self._gate(Z, A, names)
        return np.clip(self.intercept_ + Z @ self.weights_.to_numpy(), self.y_lo_, self.y_hi_)


# ============================================================================== stacking
class RegularizedStack(BaseEstimator, RegressorMixin):
    def __init__(self, members=None, inner_splits: int = 4, seed: int = 0):
        self.members, self.inner_splits, self.seed = members, inner_splits, seed

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        mem = list(self.members or [_ridge()])
        Z = np.zeros((len(y), len(mem)))
        for tr, va in _kfold(self.inner_splits, self.seed).split(X):
            for j, m in enumerate(mem):
                Z[va, j] = np.asarray(clone(m).fit(X.iloc[tr], y[tr]).predict(X.iloc[va]), float).ravel()
        self.weights_, self.intercept_ = convex_weights(Z, y)
        self.models_ = [clone(m).fit(X, y) if w > 1e-6 else None for m, w in zip(mem, self.weights_)]
        return self

    def predict(self, X: pd.DataFrame):
        out = np.full(len(X), self.intercept_)
        for m, w in zip(self.models_, self.weights_):
            if m is not None:
                out += w * np.asarray(m.predict(X), float).ravel()
        return out


# ===================================================================== domain MoE (pooled)
class DomainPartialPooling(BaseEstimator, RegressorMixin):
    def __init__(self, domain_cols=None, min_domain_size: int = 40, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 inner_splits: int = 4, seed: int = 0):
        self.domain_cols, self.min_domain_size, self.lambdas, self.inner_splits, self.seed = domain_cols, min_domain_size, lambdas, inner_splits, seed

    def _domain(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in (self.domain_cols or []) if c in X.columns]
        if not cols:
            return np.zeros(len(X), dtype=object)
        bits = [(X[c].to_numpy(float) > self.mid_[c]).astype(int).astype(str) for c in cols]
        return np.array(["".join(t) for t in zip(*bits)], dtype=object)

    def _fit_once(self, X, y):
        g = _ridge().fit(X, y)
        r = y - g.predict(X)
        dom = self._domain(X)
        experts = {}
        for d in np.unique(dom):
            m = dom == d
            if m.sum() >= self.min_domain_size:
                experts[d] = _ridge().fit(X[m], r[m])
        return g, experts

    def _pred(self, g, experts, X, lam):
        p = np.asarray(g.predict(X), float)
        dom = self._domain(X)
        for d, e in experts.items():
            m = dom == d
            if m.any():
                p[m] += lam * np.asarray(e.predict(X[m]), float)
        return p

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        cols = [c for c in (self.domain_cols or []) if c in X.columns]
        self.mid_ = {c: float((X[c].min() + X[c].max()) / 2) for c in cols}
        P = {lam: np.zeros(len(y)) for lam in self.lambdas}
        for tr, va in _kfold(self.inner_splits, self.seed).split(X):
            g, ex = self._fit_once(X.iloc[tr], y[tr])
            for lam in self.lambdas:
                P[lam][va] = self._pred(g, ex, X.iloc[va], lam)
        errs = {lam: float(np.mean((y - p) ** 2)) for lam, p in P.items()}
        self.lambda_ = float(min(errs, key=errs.get))
        self.g_, self.experts_ = self._fit_once(X, y)
        self.n_experts_ = len(self.experts_)
        return self

    def predict(self, X: pd.DataFrame):
        return self._pred(self.g_, self.experts_, X, self.lambda_)
