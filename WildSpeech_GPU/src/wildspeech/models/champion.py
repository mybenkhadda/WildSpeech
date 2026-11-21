"""P13 — robust champion selection, regularised stacking, ensemble weights and uncertainty.

Candidates are OOF prediction sets on the SAME repeated folds. Selection rules:
  robust score = mean R2 across repeats - lambda * sd across repeats; a candidate replaces the screening
  baseline only if its paired delta-R2 bootstrap CI excludes 0 (stable incremental improvements first).
Stacking: non-negative weights on candidate OOF predictions, evaluated by cross-validating the weights over
students (weights fitted on other folds' students). Uncertainty: bootstrap CI of R2 and split-conformal
prediction intervals from cross-fitted absolute residuals (with per-instrument coverage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from ..stats import bootstrap_ci, r2
from ..validation.metrics import incremental_r2, mean_oof, metric_row


def candidate_matrix(oofs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    P = pd.concat({k: mean_oof(v)["y_pred"] for k, v in oofs.items()}, axis=1).dropna()
    y = mean_oof(next(iter(oofs.values())))["y_true"].reindex(P.index)
    return P, y


def robust_ranking(summaries: pd.DataFrame, oofs: dict[str, pd.DataFrame], baseline: str, lam: float, n_boot: int) -> pd.DataFrame:
    s = summaries.copy()
    s["robust_score"] = s["r2_mean"] - lam * s["r2_sd"].fillna(0)
    rows = []
    for name in s["name"]:
        if name == baseline or baseline not in oofs:
            rows.append({"name": name, "delta_r2": 0.0, "delta_ci_low": np.nan, "delta_ci_high": np.nan})
            continue
        rows.append({"name": name, **incremental_r2(oofs[baseline], oofs[name], n_boot, 0)})
    s = s.merge(pd.DataFrame(rows), on="name", how="left")
    s["beats_baseline"] = (s["delta_ci_low"] > 0) | (s["name"] == baseline)
    return s.sort_values(["beats_baseline", "robust_score"], ascending=False).reset_index(drop=True)


def nnls_weights(P: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> tuple[np.ndarray, float]:
    """Non-negative, sum-to-one-ish weights via NNLS on centred predictions with ridge augmentation."""
    mu = P.mean(0)
    Pc, yc = P - mu, y - y.mean()
    k = P.shape[1]
    A = np.vstack([Pc, np.sqrt(ridge * len(y)) * np.eye(k)])
    b = np.concatenate([yc, np.full(k, 0.0)])
    w, _ = nnls(A, b)
    if w.sum() <= 0:
        w = np.ones(k) / k
    return w, float(y.mean() - mu @ w)


def simplex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    k = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(lambda w: np.mean((y - P @ w) ** 2), np.ones(k) / k, bounds=[(0, 1)] * k, constraints=cons, method="SLSQP")
    return res.x if res.success else np.ones(k) / k


def cv_stack(P: pd.DataFrame, y: pd.Series, folds_of: pd.Series, method: str = "nnls") -> tuple[pd.Series, pd.DataFrame]:
    """Cross-validated stacking: weights for fold k are fitted on students outside fold k."""
    pred = pd.Series(np.nan, index=P.index)
    wrows = []
    for k in sorted(folds_of.unique()):
        te = folds_of.reindex(P.index) == k
        A, b = P[~te].to_numpy(), y[~te].to_numpy()
        if method == "nnls":
            w, c = nnls_weights(A, b)
        else:
            w, c = simplex_weights(A, b), 0.0
        pred[te] = P[te].to_numpy() @ w + c
        wrows.append({"fold": int(k), "intercept": c, **dict(zip(P.columns, w))})
    return pred, pd.DataFrame(wrows)


def conformal_intervals(y: pd.Series, pred: pd.Series, groups: pd.Series | None, alphas=(0.1, 0.2), folds_of: pd.Series | None = None) -> pd.DataFrame:
    """Cross-conformal: the quantile for fold k comes from |residuals| of the other folds."""
    res = (y - pred).abs()
    rows = []
    folds = folds_of.reindex(y.index) if folds_of is not None else pd.Series(0, index=y.index)
    for a in alphas:
        covered = pd.Series(False, index=y.index)
        width = pd.Series(np.nan, index=y.index)
        for k in folds.unique():
            te = folds == k
            cal = res[~te] if (~te).any() else res
            q = np.quantile(cal, min(1.0, (1 - a) * (1 + 1 / max(len(cal), 1))))
            covered[te] = res[te] <= q
            width[te] = 2 * q
        rows.append({"alpha": a, "group": "all", "coverage": float(covered.mean()), "mean_width": float(width.mean())})
        if groups is not None:
            for g, idx in groups.reindex(y.index).groupby(groups.reindex(y.index)).groups.items():
                rows.append({"alpha": a, "group": str(g), "coverage": float(covered.loc[idx].mean()), "mean_width": float(width.loc[idx].mean())})
    return pd.DataFrame(rows)


def final_q(y: pd.Series, pred: pd.Series, alpha: float) -> float:
    res = (y - pred).abs()
    return float(np.quantile(res, min(1.0, (1 - alpha) * (1 + 1 / len(res)))))


def r2_bootstrap(y, p, n_boot: int) -> dict:
    est, lo, hi = bootstrap_ci(r2, np.asarray(y), np.asarray(p), n_boot=n_boot, seed=0)
    return {"r2": est, "r2_ci_low": lo, "r2_ci_high": hi, **{k: v for k, v in metric_row(y, p).items() if k != "r2"}}
