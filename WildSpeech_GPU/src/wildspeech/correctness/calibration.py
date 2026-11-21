"""Probability calibration (fitted on inner-OOF predictions of training students only) and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def transform(self, p):
        return np.asarray(p, float)


class PlattCalibrator:
    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression

        from .classifiers import logit

        self.m_ = LogisticRegression(C=1e3, max_iter=2000).fit(logit(p).reshape(-1, 1), np.asarray(y, int))
        return self

    def transform(self, p):
        from .classifiers import logit

        return self.m_.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression

        self.m_ = IsotonicRegression(y_min=1e-3, y_max=1 - 1e-3, out_of_bounds="clip").fit(np.asarray(p, float), np.asarray(y, float))
        return self

    def transform(self, p):
        return self.m_.predict(np.asarray(p, float))


def calibrator_factory(kind: str):
    kind = (kind or "none").lower()
    if kind == "platt":
        return PlattCalibrator
    if kind == "isotonic":
        return IsotonicCalibrator
    return None


def ece(y, p, n_bins: int = 15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if not len(y):
        return np.nan
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        s = bins == b
        if s.any():
            tot += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(tot)


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        s = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        rows.append({"bin": i, "p_low": edges[i], "p_high": edges[i + 1], "n": int(s.sum()),
                     "mean_pred": float(p[s].mean()) if s.any() else np.nan,
                     "frac_correct": float(y[s].mean()) if s.any() else np.nan})
    return pd.DataFrame(rows)


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(int), np.clip(p[m], 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auroc": np.nan, "auprc": np.nan, "bacc": np.nan, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
    return {"n": int(len(y)), "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "auprc_incorrect": float(average_precision_score(1 - y, 1 - p)),
            "bacc": float(balanced_accuracy_score(y, (p >= threshold).astype(int))), "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p)), "ece": ece(y, p)}
