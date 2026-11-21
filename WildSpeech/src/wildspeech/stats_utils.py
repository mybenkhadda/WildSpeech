"""Statistical helpers that return explicit NaN/status values on insufficient data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def safe_corr(x, y, method: str = "spearman", min_n: int = 10) -> tuple[float, float, int]:
    x, y = _clean_xy(x, y)
    n = int(len(x))
    if n < min_n or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, n
    if method == "pearson":
        r, p = sps.pearsonr(x, y)
    elif method == "spearman":
        r, p = sps.spearmanr(x, y)
    elif method == "kendall":
        r, p = sps.kendalltau(x, y)
    else:
        raise ValueError(method)
    return float(r), float(p), n


def association_table(X: pd.DataFrame, y, methods=("pearson", "spearman"), min_n: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    rows = []
    for col in X.columns:
        row = {"feature": col}
        values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        for m in methods:
            r, p, n = safe_corr(values, y, m, min_n)
            row[f"{m}_r"], row[f"{m}_p"], row["n"] = r, p, n
        rows.append(row)
    return pd.DataFrame(rows)


def residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Z1 = np.column_stack([np.ones(len(v)), Z]) if Z.size else np.ones((len(v), 1))
    beta, *_ = np.linalg.lstsq(Z1, v, rcond=None)
    return v - Z1 @ beta


def partial_corr(x, y, Z, method: str = "pearson", min_n: int = 15) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    n = int(mask.sum())
    k = Z.shape[1]
    if n < max(min_n, k + 5):
        return np.nan, np.nan, n
    xs, ys, Zs = x[mask], y[mask], Z[mask]
    if method == "spearman":
        xs, ys = sps.rankdata(xs), sps.rankdata(ys)
        Zs = np.column_stack([sps.rankdata(Zs[:, j]) for j in range(k)])
    rx, ry = residualize(xs, Zs), residualize(ys, Zs)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - 2 - k
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan, n
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = float(2 * sps.t.sf(abs(t), dof))
    return r, p, n


def bh_fdr(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(pv)
    adj[order] = np.clip(ranked, 0, 1)
    out[mask] = adj
    return out


def bootstrap_ci(func, *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                 groups=None) -> dict:
    """Percentile bootstrap; resamples whole groups (clusters) when given."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    try:
        estimate = float(func(*arrays))
    except Exception:
        estimate = np.nan
    if n < 5:
        return {"estimate": estimate, "lower": np.nan, "upper": np.nan, "se": np.nan, "n_boot": 0}
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(int(n_boot)):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([index_by_group[g] for g in chosen])
        try:
            value = func(*[a[idx] for a in arrays])
        except Exception:
            continue
        if np.isfinite(value):
            stats.append(value)
    if len(stats) < max(10, n_boot // 10):
        return {"estimate": estimate, "lower": np.nan, "upper": np.nan, "se": np.nan, "n_boot": len(stats)}
    arr = np.asarray(stats)
    return {"estimate": estimate, "lower": float(np.quantile(arr, alpha / 2)),
            "upper": float(np.quantile(arr, 1 - alpha / 2)), "se": float(arr.std(ddof=1)), "n_boot": len(arr)}


def spearman_stat(x, y) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(sps.spearmanr(x, y)[0])


def distance_correlation(x, y, max_n: int = 2000, seed: int = 0) -> float:
    x, y = _clean_xy(x, y)
    n = len(x)
    if n < 10:
        return np.nan
    if n > max_n:
        idx = np.random.default_rng(seed).choice(n, max_n, replace=False)
        x, y = x[idx], y[idx]
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov = (A * B).mean()
    dvar = np.sqrt((A * A).mean() * (B * B).mean())
    return float(np.sqrt(max(dcov, 0) / dvar)) if dvar > 0 else np.nan


def mutual_information(X: pd.DataFrame, y, seed: int = 0, min_n: int = 20) -> pd.Series:
    from sklearn.feature_selection import mutual_info_regression

    y = np.asarray(y, dtype=float)
    out = {}
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_n or np.std(x[mask]) == 0:
            out[col] = np.nan
            continue
        out[col] = float(mutual_info_regression(x[mask, None], y[mask], random_state=seed, n_neighbors=3)[0])
    return pd.Series(out)


def icc1(values, groups, min_groups: int = 3) -> float:
    """One-way random-effects ICC(1) with unequal group sizes."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    if df.empty:
        return np.nan
    stats = df.groupby("g")["v"].agg(["count", "mean", "var"])
    stats = stats[stats["count"] >= 2]
    k = len(stats)
    if k < min_groups:
        return np.nan
    N = stats["count"].sum()
    grand = (stats["count"] * stats["mean"]).sum() / N
    ss_between = (stats["count"] * (stats["mean"] - grand) ** 2).sum()
    ss_within = ((stats["count"] - 1) * stats["var"]).sum()
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k) if N > k else np.nan
    n0 = (N - (stats["count"] ** 2).sum() / N) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float((ms_between - ms_within) / denom)


def variance_decomposition(values, groups) -> dict:
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    if df.empty or df["g"].nunique() < 2:
        return {"n": len(df), "n_groups": int(df["g"].nunique()) if not df.empty else 0,
                "between_var": np.nan, "within_var": np.nan, "icc1": np.nan}
    total = df["v"].var(ddof=1)
    group_means = df.groupby("g")["v"].transform("mean")
    within = (df["v"] - group_means).var(ddof=1)
    return {"n": int(len(df)), "n_groups": int(df["g"].nunique()), "total_var": float(total),
            "between_var": float(group_means.var(ddof=1)), "within_var": float(within),
            "icc1": icc1(df["v"], df["g"])}


def smd(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def hedges_g(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (n1 + n2) - 9)))


def cluster_robust_ols(y, X: pd.DataFrame, groups=None) -> pd.DataFrame:
    """OLS with HC1 (no groups) or CR1 cluster-robust standard errors."""
    y = np.asarray(y, dtype=float)
    Xv = np.column_stack([np.ones(len(y)), X.to_numpy(dtype=float)])
    names = ["intercept"] + list(X.columns)
    mask = np.isfinite(y) & np.all(np.isfinite(Xv), axis=1)
    if groups is not None:
        groups = np.asarray(groups)
        mask &= pd.notna(groups)
    y, Xv = y[mask], Xv[mask]
    n, k = Xv.shape
    if n <= k + 2:
        return pd.DataFrame({"term": names, "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": n})
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ y
    resid = y - Xv @ beta
    if groups is None:
        meat = (Xv * resid[:, None] ** 2).T @ Xv
        V = XtX_inv @ meat @ XtX_inv * n / (n - k)
        dof = n - k
    else:
        g = groups[mask]
        uniq = pd.unique(g)
        G = len(uniq)
        meat = np.zeros((k, k))
        for level in uniq:
            idx = g == level
            s = Xv[idx].T @ resid[idx]
            meat += np.outer(s, s)
        V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
        dof = max(G - 1, 1)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p = 2 * sps.t.sf(np.abs(t), dof)
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "t": t, "p": p, "n": n})


def permute_within_groups(n: int, groups, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n)
    if groups is None:
        return rng.permutation(n)
    groups = np.asarray(groups)
    out = idx.copy()
    for g in pd.unique(groups):
        members = np.flatnonzero(groups == g)
        out[members] = rng.permutation(members)
    return out


def permutation_pvalue(x, y, n_perm: int = 1000, groups=None, seed: int = 0, stat=spearman_stat) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if groups is not None:
        groups = np.asarray(groups)[mask]
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"stat": np.nan, "p_perm": np.nan, "n": len(x)}
    rng = np.random.default_rng(seed)
    observed = stat(x, y)
    exceed = 0
    valid = 0
    for _ in range(int(n_perm)):
        perm = permute_within_groups(len(y), groups, rng)
        s = stat(x, y[perm])
        if np.isfinite(s):
            valid += 1
            exceed += abs(s) >= abs(observed)
    p = (exceed + 1) / (valid + 1) if valid else np.nan
    return {"stat": observed, "p_perm": p, "n": len(x), "n_perm": valid}


def bimodality_coefficient(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    g = sps.skew(x, bias=False)
    k = sps.kurtosis(x, bias=False)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        sd = x.std()
        return (x - x.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else x * np.nan
    return (x - med) / mad


def quantile_bins(x: pd.Series, q: int, min_count: int = 10, labels_prefix: str = "Q") -> pd.Series:
    """Quantile bins; reduces the number of bins until each has min_count members."""
    x = pd.to_numeric(x, errors="coerce")
    for k in range(q, 1, -1):
        try:
            bins = pd.qcut(x, k, duplicates="drop")
        except ValueError:
            continue
        counts = bins.value_counts()
        if len(counts) >= 2 and counts.min() >= min_count:
            codes = bins.cat.codes.astype(float).where(bins.notna())
            return codes.map(lambda c: f"{labels_prefix}{int(c) + 1}" if np.isfinite(c) else np.nan)
    return pd.Series(np.nan, index=x.index, dtype=object)


def cramers_v(a, b) -> float:
    tab = pd.crosstab(pd.Series(a), pd.Series(b))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = sps.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r, k = tab.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def robust_stats(values, prefix: str, stats: list, min_n: int = 5) -> dict:
    """Robust summary statistics with NaN for insufficient observations."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out: dict = {}
    for s in stats:
        key = f"{prefix}_{s}"
        if n == 0:
            out[key] = np.nan
            continue
        if s == "mean":
            out[key] = float(x.mean())
        elif s == "std":
            out[key] = float(x.std(ddof=1)) if n >= 2 else np.nan
        elif s == "median":
            out[key] = float(np.median(x))
        elif s == "mad":
            out[key] = float(np.median(np.abs(x - np.median(x)))) if n >= 2 else np.nan
        elif s == "min":
            out[key] = float(x.min())
        elif s == "max":
            out[key] = float(x.max())
        elif s.startswith("p") and s[1:].isdigit():
            out[key] = float(np.percentile(x, int(s[1:]))) if n >= 2 else np.nan
        elif s == "iqr":
            out[key] = float(np.percentile(x, 75) - np.percentile(x, 25)) if n >= 4 else np.nan
        elif s == "skewness":
            out[key] = float(sps.skew(x)) if n >= max(min_n, 3) and np.std(x) > 0 else np.nan
        elif s == "kurtosis":
            out[key] = float(sps.kurtosis(x)) if n >= max(min_n, 4) and np.std(x) > 0 else np.nan
        elif s == "cv":
            m = x.mean()
            out[key] = float(x.std(ddof=1) / abs(m)) if n >= 2 and abs(m) > 1e-9 else np.nan
        elif s == "count":
            out[key] = float(n)
        else:
            raise ValueError(f"unknown statistic {s}")
    return out


def linear_slope(x, y) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 3 or np.var(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def shrink_towards(estimates: pd.Series, counts: pd.Series, prior: float, k: float) -> pd.Series:
    w = counts / (counts + k)
    return w * estimates.fillna(prior) + (1 - w) * prior
