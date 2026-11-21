"""P12 — failure analysis: clusters of large baseline errors and which models/modalities recover them."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..stats import bootstrap_ci
from ..validation.metrics import mean_oof
from .phenotypes import prepare_matrix


def error_table(oofs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, oof in oofs.items():
        m = mean_oof(oof)
        parts.append(m.rename(columns={"y_pred": f"pred__{name}"})[[f"pred__{name}"]])
        y = m["y_true"]
    T = pd.concat(parts, axis=1)
    T.insert(0, "y_true", y)
    for name in oofs:
        T[f"abserr__{name}"] = (T[f"pred__{name}"] - T["y_true"]).abs()
        T[f"resid__{name}"] = T["y_true"] - T[f"pred__{name}"]
    return T


def failure_clusters(X: pd.DataFrame, err: pd.DataFrame, baseline: str, quantile: float, k_range: tuple[int, int], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ae = err[f"abserr__{baseline}"]
    thr = ae.quantile(quantile)
    fail_ids = ae[ae >= thr].index
    Z = prepare_matrix(X.reindex(fail_ids))
    lab = pd.Series(-1, index=err.index, name="failure_cluster")
    sel = []
    if len(fail_ids) >= 12 and Z.shape[1] >= 2:
        best_k, best_s = 2, -1
        for k in range(k_range[0], min(k_range[1], len(fail_ids) // 4) + 1):
            km = KMeans(k, n_init=10, random_state=seed).fit(Z.values)
            s = silhouette_score(Z.values, km.labels_)
            sel.append({"k": k, "silhouette": s})
            if s > best_s:
                best_k, best_s = k, s
        km = KMeans(best_k, n_init=10, random_state=seed).fit(Z.values)
        lab.loc[Z.index] = km.labels_
    else:
        lab.loc[fail_ids] = 0
    sign = np.sign(err[f"resid__{baseline}"])
    out = err.join(lab)
    out["direction"] = np.where(sign > 0, "underpredicted", "overpredicted")
    out["is_failure"] = out.index.isin(fail_ids)
    return out, pd.DataFrame(sel)


def recovery_table(fc: pd.DataFrame, models: list[str], baseline: str, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for c, d in fc[fc["is_failure"]].groupby("failure_cluster"):
        for m in models:
            if m == baseline:
                continue
            red = (d[f"abserr__{baseline}"] - d[f"abserr__{m}"]).to_numpy()
            est, lo, hi = bootstrap_ci(np.mean, red, n_boot=n_boot, seed=seed)
            rows.append({"failure_cluster": int(c), "n": len(d), "model": m, "mae_baseline": d[f"abserr__{baseline}"].mean(),
                         "mae_model": d[f"abserr__{m}"].mean(), "mae_reduction": est, "ci_low": lo, "ci_high": hi,
                         "recovers": bool(np.isfinite(lo) and lo > 0)})
    return pd.DataFrame(rows)


def describe_clusters(fc: pd.DataFrame, X: pd.DataFrame, meta: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    Zall = prepare_matrix(X)
    rows = []
    for c, d in fc[fc["is_failure"]].groupby("failure_cluster"):
        ids = d.index
        z = Zall.reindex(ids).mean().sort_values(key=lambda s: -s.abs())
        r = {"failure_cluster": int(c), "n": len(ids), "frac_underpredicted": float((d["direction"] == "underpredicted").mean()),
             "outcome_mean": float(d["y_true"].mean()),
             "top_features": "; ".join(f"{k}={v:+.2f}" for k, v in z.head(top).items())}
        for col in [x for x in ("instrument", "original_sample_rate", "school_id") if x in meta]:
            vc = meta.reindex(ids)[col].astype(str).value_counts(normalize=True)
            r[f"{col}_mix"] = ", ".join(f"{a}:{b:.0%}" for a, b in vc.head(3).items())
        rows.append(r)
    return pd.DataFrame(rows)
