"""P06 — reproducible student phenotypes from cross-fitted multimodal features (screening excluded)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..analysis import phenotypes as Ph
from ._common import load_crossfit, save_fig, student_meta

S = "P06"


def run(ctx) -> dict:
    a = ctx.cfg.section("analysis")
    X = load_crossfit(ctx)
    y = ctx.bundle().y.reindex(X.index)
    meta = student_meta(ctx).set_index("subject_id")
    Z = Ph.prepare_matrix(X)
    if Z.shape[1] < 3:
        return {"status": "too few non-screening features"}
    scores, ev, load = Ph.pca_embedding(Z, int(a.get("pca_components", 20)), ctx.seed)
    ctx.save_table(ev, S, "pca_explained_variance")
    ctx.save_table(load.rename_axis("feature").reset_index(), S, "pca_loadings")
    k_lo, k_hi = a.get("cluster_k_range", [2, 7])
    sel = Ph.model_selection(scores, (int(k_lo), int(k_hi)), ctx.seed)
    ctx.save_table(sel, S, "cluster_model_selection")
    stab = pd.DataFrame([Ph.bootstrap_stability(scores, int(k), int(a.get("cluster_bootstrap", 50)), ctx.seed) for k in sel["k"]])
    ctx.save_table(stab, S, "cluster_bootstrap_stability")
    sel = sel.merge(stab, on="k")
    # prefer stable solutions: highest silhouette among k with ARI >= 0.6, else most stable
    stable = sel[sel["ari_mean"] >= 0.6]
    best_k = int((stable if not stable.empty else sel).sort_values("kmeans_silhouette" if not stable.empty else "ari_mean",
                                                                   ascending=False).iloc[0]["k"])
    from sklearn.cluster import KMeans

    labels = KMeans(best_k, n_init=20, random_state=ctx.seed).fit_predict(scores)
    prof, summ = Ph.cluster_profiles(Z, labels, meta, y, int(a.get("bootstrap", 1000)), ctx.seed)
    ctx.save_table(prof.rename_axis("feature").reset_index(), S, "cluster_feature_profiles")
    ctx.save_table(summ, S, "cluster_summary")
    fp = Ph.family_profiles(prof)
    ctx.save_table(fp.reset_index(), S, "cluster_family_profiles")
    save_fig(ctx, S, "cluster_family_profiles", viz.heatmap(fp, f"Mean |z| per feature family, k={best_k}", diverging=False), fp.reset_index())
    assign = pd.DataFrame({"subject_id": Z.index, "cluster": labels, "pc1": scores[:, 0], "pc2": scores[:, 1]})
    ctx.save_table(assign, S, "student_phenotype_assignments")
    save_fig(ctx, S, "pca_clusters", viz.embedding_scatter(scores[:, :2], [f"C{l}" for l in labels], "Phenotypes in PC1/PC2"), assign)
    if a.get("umap_enabled", True):
        U = Ph.umap_embedding(scores, ctx.seed)
        if U is not None:
            save_fig(ctx, S, "umap_clusters", viz.embedding_scatter(U, [f"C{l}" for l in labels], "Phenotypes (UMAP of PCA scores)"),
                     pd.DataFrame({"subject_id": Z.index, "u1": U[:, 0], "u2": U[:, 1], "cluster": labels}))
    save_fig(ctx, S, "outcome_by_cluster", viz.bar_ci(summ.assign(cluster=lambda d: "C" + d["cluster"].astype(str)), "cluster",
                                                     "outcome_mean", "outcome_ci_low", "outcome_ci_high",
                                                     "Outcome by phenotype (bootstrap 95% CI)", horizontal=False), summ)
    ari = float(sel.loc[sel["k"] == best_k, "ari_mean"].iloc[0])
    ctx.registries.update_hypothesis("H05", "supported" if ari >= 0.6 else "not_supported", S,
                                     f"best k={best_k}, bootstrap ARI={ari:.2f}, outcome eta2={summ['outcome_eta2'].iloc[0]:.3f}",
                                     effect=ari)
    return {"best_k": best_k, "ari": ari, "n_features_used": Z.shape[1]}
