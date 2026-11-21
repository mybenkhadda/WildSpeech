"""P06 — student phenotypes: PCA, optional UMAP, KMeans/GMM model selection, bootstrap stability, profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from ..features.builder import family_of
from ..stats import bootstrap_ci, eta_squared


def prepare_matrix(X: pd.DataFrame, max_missing: float = 0.3, exclude_families=("SCREENING", "RECORDING")) -> pd.DataFrame:
    cols = [c for c in X.columns if family_of(c) not in exclude_families]
    Z = X[cols]
    Z = Z.loc[:, Z.isna().mean() <= max_missing]
    Z = Z.loc[:, Z.std() > 1e-9]
    Z = Z.fillna(Z.median())
    return (Z - Z.mean()) / Z.std(ddof=0)


def pca_embedding(Z: pd.DataFrame, n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    k = min(n, Z.shape[1], Z.shape[0] - 1)
    pca = PCA(n_components=k, random_state=seed).fit(Z.values)
    scores = pca.transform(Z.values)
    ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(k)], "explained": pca.explained_variance_ratio_,
                       "cumulative": np.cumsum(pca.explained_variance_ratio_)})
    load = pd.DataFrame(pca.components_.T, index=Z.columns, columns=[f"PC{i + 1}" for i in range(k)])
    return scores, ev, load


def umap_embedding(scores: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        import umap

        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(scores)
    except Exception:
        return None


def model_selection(S: np.ndarray, k_range: tuple[int, int], seed: int) -> pd.DataFrame:
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(k, n_init=10, random_state=seed).fit(S)
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=3).fit(S)
        lab_g = gm.predict(S)
        rows.append({"k": k, "kmeans_silhouette": silhouette_score(S, km.labels_), "kmeans_ch": calinski_harabasz_score(S, km.labels_),
                     "kmeans_db": davies_bouldin_score(S, km.labels_), "kmeans_inertia": km.inertia_,
                     "gmm_bic": gm.bic(S), "gmm_silhouette": silhouette_score(S, lab_g) if len(set(lab_g)) > 1 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_stability(S: np.ndarray, k: int, n_boot: int, seed: int) -> dict:
    """Refit on bootstrap samples; ARI against the full-data solution on the full data (predict)."""
    base = KMeans(k, n_init=10, random_state=seed).fit(S)
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, len(S), len(S))
        km = KMeans(k, n_init=5, random_state=seed + b + 1).fit(S[idx])
        aris.append(adjusted_rand_score(base.labels_, km.predict(S)))
    aris = np.array(aris)
    return {"k": k, "ari_mean": float(aris.mean()), "ari_p05": float(np.quantile(aris, .05)), "ari_p95": float(np.quantile(aris, .95)),
            "n_boot": n_boot}


def cluster_profiles(Z: pd.DataFrame, labels: np.ndarray, meta: pd.DataFrame, y: pd.Series, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = pd.Series(labels, index=Z.index, name="cluster")
    prof = Z.groupby(lab).mean().T
    prof["family"] = [family_of(c) for c in prof.index]
    rows = []
    for c, ids in lab.groupby(lab).groups.items():
        yy = y.reindex(ids).dropna().to_numpy()
        est, lo, hi = bootstrap_ci(np.mean, yy, n_boot=n_boot, seed=seed) if len(yy) > 3 else (np.nan, np.nan, np.nan)
        r = {"cluster": c, "n": len(ids), "outcome_mean": est, "outcome_ci_low": lo, "outcome_ci_high": hi}
        for col in [x for x in ("instrument", "original_sample_rate", "school_id") if x in meta]:
            vc = meta.reindex(ids)[col].astype(str).value_counts(normalize=True)
            r[f"{col}_top"] = f"{vc.index[0]} ({vc.iloc[0]:.0%})" if len(vc) else ""
        rows.append(r)
    summary = pd.DataFrame(rows)
    e, p = eta_squared(y.reindex(lab.index), lab)
    summary["outcome_eta2"], summary["outcome_p"] = e, p
    return prof, summary


def family_profiles(prof: pd.DataFrame) -> pd.DataFrame:
    """Mean |z| per family per cluster -> which modalities define each phenotype."""
    num = prof.drop(columns="family")
    return num.abs().groupby(prof["family"]).mean()
