"""Section 15 - student phenotype and clustering atlas.

Scientific question: are there distinct multimodal student profiles that
conventional screening does not describe adequately?

Clusters are fitted in PCA space (never only in a 2-D visualisation
embedding) and never use the target. Discovered clusters are characterised
afterwards; no predefined educational or cognitive labels are imposed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .families import columns_by_family, modeling_columns
from .stats_utils import cramers_v, smd

SPACES = {
    "C01_SCREENING": ["SCREENING"],
    "C02_ITEM_BEHAVIOR": ["CORRECTNESS", "ITEM_DIFFICULTY", "ADAPTIVE"],
    "C03_TEMPORAL": ["TEMPORAL"],
    "C04_SPECTRAL": ["SPECTRAL", "MFCC"],
    "C05_WAVELETS": ["WAVELETS"],
    "C06_PHONETIC": ["PHONETIC"],
    "C07_PROSODY": ["PROSODY"],
    "C08_MULTIMODAL": ["SCREENING", "CORRECTNESS", "TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC"],
    "C09_SCREENING_ADJUSTED_AUDIO": ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY"],
    "C10_TASK_SPECIFIC": ["TEMPORAL", "SPECTRAL", "MFCC", "CORRECTNESS"],
}
AUDIO_CLUSTER_FAMILIES = ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY", "PHONETIC"]


def prepare_space(S: pd.DataFrame, space: str, fit_index=None, max_components: int = 20, seed: int = 0):
    """Median-impute, robust-scale, optionally residualise on screening, then PCA (fit on fit_index)."""
    from sklearn.decomposition import PCA

    fams = SPACES[space]
    cols = [c for c in S.columns if c in set(modeling_columns(S.columns, fams))
            and pd.api.types.is_numeric_dtype(S[c]) and not c.startswith("meta_")]
    if space == "C10_TASK_SPECIFIC":
        cols = [c for c in cols if "__task=" in c]
    X = S[cols]
    X = X.loc[:, X.notna().mean() >= 0.5]
    fit_index = X.index if fit_index is None else fit_index
    Xf = X.loc[fit_index]
    med = Xf.median()
    iqr = (Xf.quantile(0.75) - Xf.quantile(0.25)).replace(0, np.nan)
    keep = iqr.notna()
    X = ((X.loc[:, keep] - med[keep]) / iqr[keep]).fillna(0.0).clip(-10, 10)
    if X.shape[1] < 2:
        return None, {"status": "INSUFFICIENT_FEATURES", "n_features": int(X.shape[1])}
    if space == "C09_SCREENING_ADJUSTED_AUDIO":
        theta = [c for c in da.raw_theta_columns(S)]
        if theta:
            T = S[theta].fillna(S.loc[fit_index, theta].median())
            T1 = np.column_stack([np.ones(len(T)), T.reindex(X.index).to_numpy()])
            mask = X.index.isin(fit_index)
            beta, *_ = np.linalg.lstsq(T1[mask], X.to_numpy()[mask], rcond=None)
            X = pd.DataFrame(X.to_numpy() - T1 @ beta, index=X.index, columns=X.columns)
    n_comp = int(min(max_components, X.shape[1], len(fit_index) - 1))
    pca = PCA(n_components=n_comp, random_state=seed).fit(X.loc[fit_index])
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(min(n_comp, np.searchsorted(cum, 0.9) + 1))
    Z = pd.DataFrame(pca.transform(X)[:, :k], index=X.index, columns=[f"PC{i + 1}" for i in range(k)])
    return Z, {"status": "OK", "n_features": int(X.shape[1]), "n_components": k, "explained": float(cum[k - 1])}


def _fit_method(method: str, Z: np.ndarray, k: int, seed: int, cfg: dict):
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture

    if method == "kmeans":
        m = KMeans(k, random_state=seed, n_init=10).fit(Z)
        return m.labels_, None
    if method == "gmm":
        m = GaussianMixture(k, random_state=seed, n_init=3, covariance_type="diag").fit(Z)
        post = m.predict_proba(Z)
        return post.argmax(1), post
    if method == "hierarchical":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z), None
    if method == "spectral":
        return SpectralClustering(n_clusters=k, random_state=seed, affinity="nearest_neighbors",
                                  n_neighbors=min(15, len(Z) - 1)).fit_predict(Z), None
    raise ValueError(method)


def _hdbscan(Z: np.ndarray, cfg: dict):
    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(min_cluster_size=int(cfg["hdbscan_min_cluster_size"])).fit_predict(Z)
    except ImportError:
        try:
            import hdbscan  # type: ignore

            return hdbscan.HDBSCAN(min_cluster_size=int(cfg["hdbscan_min_cluster_size"])).fit_predict(Z)
        except ImportError:
            return None


def bootstrap_stability(Z: np.ndarray, labels: np.ndarray, method: str, k: int, n_boot: int, seed: int, cfg: dict) -> float:
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    aris = []
    n = len(Z)
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        try:
            lb, _ = _fit_method(method, Z[idx], k, seed + b + 1, cfg)
        except Exception:
            continue
        aris.append(adjusted_rand_score(labels[idx], lb))
    return float(np.mean(aris)) if aris else np.nan


def evaluate_space(Z: pd.DataFrame, cfg: dict, meta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    seed = int(cfg["seed"])
    lo, hi = cfg["cluster_k_range"]
    hi = int(min(hi, max(2, len(Z) // 20)))
    methods = ["kmeans", "gmm", "hierarchical"] + (["spectral"] if len(Z) <= int(cfg["spectral_clustering_max_n"]) else [])
    rows, labels_store = [], {}
    A = Z.to_numpy()
    for method in methods:
        for k in range(int(lo), hi + 1):
            try:
                lab, post = _fit_method(method, A, k, seed, cfg)
            except Exception as exc:
                rows.append({"method": method, "k": k, "status": f"FAILED: {exc}"})
                continue
            if len(np.unique(lab)) < 2:
                continue
            sizes = np.bincount(lab)
            row = {"method": method, "k": k, "status": "OK", "silhouette": float(silhouette_score(A, lab)),
                   "davies_bouldin": float(davies_bouldin_score(A, lab)), "calinski_harabasz": float(calinski_harabasz_score(A, lab)),
                   "min_size": int(sizes.min()), "size_balance": float(sizes.min() / sizes.max()),
                   "assignment_uncertainty": float(1 - post.max(1).mean()) if post is not None else np.nan}
            if method != "spectral":
                row["bootstrap_ari"] = bootstrap_stability(A, lab, method, k, int(cfg["cluster_bootstrap"]), seed, cfg)
            for col in ("school_id", "instrument"):
                if col in meta.columns and meta[col].nunique() > 1:
                    row[f"cramers_v_{col}"] = cramers_v(lab, meta.reindex(Z.index)[col].astype(str).to_numpy())
            rows.append(row)
            labels_store[(method, k)] = lab
    hdb = _hdbscan(A, cfg)
    if hdb is not None and len(set(hdb) - {-1}) >= 2:
        mask = hdb >= 0
        rows.append({"method": "hdbscan", "k": int(len(set(hdb) - {-1})), "status": "OK",
                     "silhouette": float(silhouette_score(A[mask], hdb[mask])), "noise_fraction": float((~mask).mean())})
        labels_store[("hdbscan", int(len(set(hdb) - {-1})))] = hdb
    ev = pd.DataFrame(rows)
    ok = ev[(ev.get("status") == "OK") & (ev["method"] != "hdbscan")] if len(ev) else ev
    if len(ok):
        stable = ok[(ok.get("bootstrap_ari", pd.Series(1.0, index=ok.index)).fillna(0) >= 0.6) & (ok["min_size"] >= int(cfg["min_group_size"]))]
        cand = stable if len(stable) else ok
        best = cand.sort_values("silhouette", ascending=False).iloc[0]
        chosen = {"method": best["method"], "k": int(best["k"]), "silhouette": float(best["silhouette"]),
                  "stable": bool(len(stable) > 0), "labels": labels_store[(best["method"], int(best["k"]))]}
    else:
        chosen = {"method": None}
    return ev, chosen


def stage_student_phenotypes(ctx) -> dict:
    from sklearn.metrics import adjusted_rand_score

    cfg = ctx.config
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    res = da.residuals(ctx)
    evals, assignments, profiles, space_info = [], {}, [], {}
    for space in SPACES:
        Z, info = prepare_space(S, space, max_components=int(cfg["cluster_max_pca_components"]), seed=int(cfg["seed"]))
        space_info[space] = info
        if Z is None or len(Z) < 30:
            space_info[space]["status"] = info.get("status", "INSUFFICIENT_STUDENTS")
            continue
        ev, chosen = evaluate_space(Z, cfg, meta)
        evals.append(ev.assign(space=space))
        if chosen.get("method") is None:
            continue
        lab = pd.Series(chosen["labels"], index=Z.index, name=space)
        assignments[space] = lab
        space_info[space].update({k: v for k, v in chosen.items() if k != "labels"})
        _characterise(ctx, S, meta, res, lab, space, Z, profiles)
    if not assignments:
        ctx.save_json(space_info, "cluster_space_info")
        return {"outputs": ["cluster_space_info"], "summary": {"status": "NO_CLUSTERABLE_SPACE"}}
    A = pd.DataFrame(assignments)
    ctx.save_df(A.reset_index(), "cluster_assignments")
    ctx.save_df(pd.concat(evals, ignore_index=True), "cluster_evaluation", also_csv=True)
    ctx.save_df(pd.DataFrame(profiles), "cluster_profiles", also_csv=True)
    sens = pd.DataFrame(index=A.columns, columns=A.columns, dtype=float)
    for a in A.columns:
        for b in A.columns:
            m = A[a].notna() & A[b].notna()
            sens.loc[a, b] = adjusted_rand_score(A.loc[m, a], A.loc[m, b]) if m.sum() > 10 else np.nan
    ctx.save_df(sens.reset_index().rename(columns={"index": "space"}), "cluster_space_sensitivity", fmt="csv")
    ctx.save_json(space_info, "cluster_space_info")
    fig, axes = viz.new_figure(1, 1, (7, 6))
    viz.heatmap(axes[0, 0], sens, "Feature-space sensitivity (ARI between chosen clusterings)", cmap="viridis", vmin=0, vmax=1, annotate=True)
    ctx.figure(fig, "cluster_space_sensitivity", "Cluster stability across feature spaces",
               "Do different modalities produce the same phenotypes?", "cluster_space_sensitivity", int(len(A)),
               "ARI near 0 means the spaces induce unrelated partitions.")
    return {"outputs": ["cluster_assignments", "cluster_evaluation", "cluster_profiles", "cluster_space_sensitivity",
                        "cluster_space_info"],
            "summary": {s: {k: v for k, v in i.items() if k in ("method", "k", "silhouette", "stable", "status")}
                        for s, i in space_info.items()}}


def _characterise(ctx, S, meta, res, lab: pd.Series, space: str, Z: pd.DataFrame, profiles: list) -> None:
    cfg = ctx.config
    d = pd.DataFrame({"cluster": lab}).join(meta).join(S)
    if res is not None:
        d["residual"] = res
    feat_cols = [c for c in modeling_columns(S.columns, SPACES[space]) if c in d.columns][:2000]
    for c, g in d.groupby("cluster"):
        rest = d[d["cluster"] != c]
        diffs = {f: smd(g[f], rest[f]) for f in feat_cols}
        top = sorted(((f, v) for f, v in diffs.items() if np.isfinite(v)), key=lambda t: -abs(t[1]))[:10]
        prof = {"space": space, "cluster": int(c), "n": int(len(g)), "target_mean": float(g["target"].mean()),
                "target_sd": float(g["target"].std()), "top_distinguishing_features": "; ".join(f"{f}:{v:+.2f}" for f, v in top)}
        for col in ("theta_mean", "acc_all", "q_snr_db_approx__mean", "residual"):
            if col in g.columns:
                prof[f"{col}_mean"] = float(g[col].mean())
        profiles.append(prof)
    tag = space.lower()
    fig, axes = viz.new_figure(1, 3, (5, 4.3))
    ax = axes[0, 0]
    ax.scatter(Z.iloc[:, 0], Z.iloc[:, 1] if Z.shape[1] > 1 else np.zeros(len(Z)), c=lab.reindex(Z.index), s=6, cmap="tab10")
    viz.finalize_axis(ax, f"{space}: PCA map (clusters fitted in {Z.shape[1]}-D PCA)", "PC1", "PC2", len(Z))
    emb = None
    if cfg.get("umap_enabled"):
        try:
            import umap  # type: ignore

            emb = umap.UMAP(random_state=int(cfg["seed"])).fit_transform(Z.to_numpy())
            title = "UMAP (visualisation only)"
        except Exception:
            emb = None
    if emb is None and cfg.get("tsne_enabled") and len(Z) >= 10:
        from sklearn.manifold import TSNE

        emb = TSNE(n_components=2, random_state=int(cfg["seed"]), perplexity=min(30, max(5, len(Z) // 10))).fit_transform(Z.to_numpy())
        title = "t-SNE (visualisation only)"
    if emb is not None:
        axes[0, 1].scatter(emb[:, 0], emb[:, 1], c=lab.reindex(Z.index), s=6, cmap="tab10")
        viz.finalize_axis(axes[0, 1], title, "dim 1", "dim 2", len(Z))
    viz.grouped_box(axes[0, 2], d.assign(cluster=d["cluster"].astype(str)), "target", "cluster", "outcome")
    ctx.figure(fig, f"{tag}_maps", f"{space}: phenotype maps and outcome by cluster",
               "Do multimodal profiles differ in outcome?", "cluster_assignments", int(len(lab)),
               f"{lab.nunique()} clusters; target used only for characterisation.")
    top_feats = []
    for p in profiles:
        if p["space"] == space:
            top_feats += [t.split(":")[0] for t in p["top_distinguishing_features"].split("; ") if t]
    top_feats = list(dict.fromkeys(top_feats))[:40]
    if top_feats:
        zf = d[top_feats].apply(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0)
        M = zf.groupby(d["cluster"]).mean().T
        fig, axes = viz.new_figure(1, 1, (max(5, M.shape[1] * 0.8 + 3), 0.22 * len(M) + 2))
        viz.heatmap(axes[0, 0], M, f"{space}: standardised cluster means of distinguishing features", cbar_label="z")
        ctx.figure(fig, f"{tag}_feature_heatmap", f"{space}: cluster feature profiles", "What characterises each cluster?",
                   "cluster_profiles", int(len(lab)), "Features ranked by |SMD| versus all other students.")
        radar_cols = top_feats[:8]
        if len(radar_cols) >= 3:
            fig = viz._plt().figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            viz.radar(ax, radar_cols, {f"cluster {c}": M.loc[radar_cols, c].tolist() for c in M.columns}, f"{space}: radar profiles")
            ctx.figure(fig, f"{tag}_radar", f"{space}: radar profiles", "How do clusters differ?", "cluster_profiles",
                       int(len(lab)), "Standardised means.")
    if len(Z) >= 10:
        from scipy.cluster.hierarchy import dendrogram, linkage

        sub = Z.iloc[np.linspace(0, len(Z) - 1, min(len(Z), 400)).astype(int)]
        fig, axes = viz.new_figure(1, 1, (9, 4))
        dendrogram(linkage(sub.to_numpy(), "ward"), ax=axes[0, 0], no_labels=True, color_threshold=None)
        viz.finalize_axis(axes[0, 0], f"{space}: Ward dendrogram (subsample)", "students", "distance", len(sub))
        ctx.figure(fig, f"{tag}_dendrogram", f"{space}: dendrogram", "Is there hierarchical structure?", "cluster_assignments",
                   len(sub), "Evenly spaced subsample of at most 400 students.")
    for col in ("school_id", "instrument"):
        if col in d.columns and d[col].nunique() > 1:
            comp = pd.crosstab(d["cluster"], d[col], normalize="index")
            comp = comp.loc[:, comp.sum().sort_values(ascending=False).index[:15]]
            fig, axes = viz.new_figure(1, 1)
            comp.plot(kind="bar", stacked=True, ax=axes[0, 0], legend=False)
            viz.finalize_axis(axes[0, 0], f"{space}: {col} composition per cluster", "cluster", "proportion", int(len(d)))
            ctx.figure(fig, f"{tag}_{col}_composition", f"{space}: {col} composition", f"Are clusters confounded with {col}?",
                       "cluster_assignments", int(len(d)), f"Cramer's V={cramers_v(d['cluster'], d[col].astype(str)):.3f}.")


def fold_cluster_features(base: pd.DataFrame, train_ids, cfg: dict, k: int = 4) -> pd.DataFrame:
    """Fold-fitted cluster distances on the multimodal audio space (for predictive use)."""
    from sklearn.cluster import KMeans

    S = base.copy()
    Z, info = prepare_space(S, "C09_SCREENING_ADJUSTED_AUDIO", fit_index=pd.Index(train_ids),
                            max_components=int(cfg["cluster_max_pca_components"]), seed=int(cfg["seed"]))
    if Z is None:
        return pd.DataFrame(index=base.index)
    km = KMeans(k, random_state=int(cfg["seed"]), n_init=10).fit(Z.loc[train_ids])
    dist = km.transform(Z)
    out = pd.DataFrame(dist, index=Z.index, columns=[f"clu_dist_{i}" for i in range(k)])
    out["clu_min_dist"] = dist.min(axis=1)
    return out
