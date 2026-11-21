"""Statistics: bootstrap CIs, FDR, effect sizes, partial/adjusted associations, ICC, permutation tests."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as ss


def bh_fdr(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def bootstrap_ci(fn: Callable[..., float], *arrays: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over rows (arrays resampled jointly). Returns (estimate, low, high)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    est = float(fn(*arrays))
    if n < 3 or n_boot <= 0:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[b] = fn(*[a[idx] for a in arrays])
        except Exception:
            vals[b] = np.nan
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return est, np.nan, np.nan
    return est, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def spearman(x, y) -> tuple[float, float, int]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 5 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan, np.nan, n
    r, p = ss.spearmanr(x[ok], y[ok])
    return float(r), float(p), n


def _residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Zc = np.column_stack([np.ones(len(v)), Z])
    beta, *_ = np.linalg.lstsq(Zc, v, rcond=None)
    return v - Zc @ beta


def partial_corr(x, y, Z, rank: bool = True) -> tuple[float, float, int]:
    """(Rank-)partial correlation of x and y controlling for columns of Z; p from t with n-2-k df."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1) if Z is not None and np.size(Z) else np.empty((len(x), 0))
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    n, k = int(ok.sum()), Z.shape[1]
    if n < k + 6:
        return np.nan, np.nan, n
    xs, ys, Zs = x[ok], y[ok], Z[ok]
    if rank:
        xs, ys = ss.rankdata(xs), ss.rankdata(ys)
        Zs = np.column_stack([ss.rankdata(c) for c in Zs.T]) if k else Zs
    rx, ry = (_residualize(xs, Zs), _residualize(ys, Zs)) if k else (xs - xs.mean(), ys - ys.mean())
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = n - 2 - k
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    return r, float(2 * ss.t.sf(abs(t), df)), n


def ols_hc3(y: np.ndarray, X: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    """OLS with HC3 heteroskedasticity-robust standard errors (intercept added)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = (resid / np.clip(1 - h, 1e-8, None)) ** 2
    cov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * ss.t.sf(np.abs(t), max(n - k, 1))
    names = ["intercept"] + (names or [f"x{i}" for i in range(k - 1)])
    return pd.DataFrame({"term": names, "coef": beta, "se_hc3": se, "t": t, "p": p, "n": n})


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def eta_squared(values, groups) -> tuple[float, float]:
    """One-way eta^2 and Kruskal-Wallis p-value."""
    df = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    if df["g"].nunique() < 2 or len(df) < 5:
        return np.nan, np.nan
    grand = df["v"].mean()
    ss_b = df.groupby("g")["v"].agg(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    ss_t = ((df["v"] - grand) ** 2).sum()
    try:
        p = ss.kruskal(*[g["v"].values for _, g in df.groupby("g") if len(g) > 0]).pvalue
    except ValueError:
        p = np.nan
    return float(ss_b / ss_t) if ss_t > 0 else np.nan, float(p)


def icc1(values, groups) -> float:
    """ICC(1) from one-way ANOVA with unbalanced group sizes."""
    df = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    g = df.groupby("g")["v"]
    k = g.size()
    if len(k) < 2:
        return np.nan
    n, a = len(df), len(k)
    grand = df["v"].mean()
    msb = (k * (g.mean() - grand) ** 2).sum() / (a - 1)
    msw = ((df["v"] - df["g"].map(g.mean())) ** 2).sum() / max(n - a, 1)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    den = msb + (k0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def permutation_pvalue_corr(x, y, n_perm: int = 1000, seed: int = 0) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = ss.rankdata(x[ok]), ss.rankdata(y[ok])
    if len(x) < 5:
        return np.nan
    obs = abs(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    cnt = sum(abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= obs for _ in range(n_perm))
    return float((cnt + 1) / (n_perm + 1))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_t = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss_t) if ss_t > 0 else np.nan


def delta_r2_bootstrap(y, p_base, p_new, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired (same students) bootstrap of R2(new) - R2(base)."""
    return bootstrap_ci(lambda a, b, c: r2(a, c) - r2(a, b), np.asarray(y), np.asarray(p_base), np.asarray(p_new),
                        n_boot=n_boot, seed=seed)


def spearman_brown(r: float) -> float:
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan


def cronbach_alpha(X: np.ndarray) -> float:
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), 1)]
    k = X.shape[1] if X.ndim == 2 else 0
    if k < 2 or len(X) < 5:
        return np.nan
    tot = X.sum(1).var(ddof=1)
    return float(k / (k - 1) * (1 - X.var(0, ddof=1).sum() / tot)) if tot > 0 else np.nan
