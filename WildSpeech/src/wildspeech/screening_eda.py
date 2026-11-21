"""Section 03 - beginning-of-year literacy (screening) intelligence.

Scientific question: how much of the future reading outcome can be explained
by conventional literacy screening?

``theta_common_factor`` (first principal component of the standardised
skill estimates) is fitted on training students for EDA; predictive stages
refit it inside each training fold.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import data_access as da
from . import visualization as viz
from .stats_utils import (association_table, distance_correlation, mutual_information, partial_corr,
                          quantile_bins, safe_corr)

SKILLS = ("blending", "deletion", "repetition")


def derive_screening_features(df: pd.DataFrame, pca_params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Row-wise screening features plus the (optionally pre-fitted) common factor."""
    cols = da.raw_theta_columns(df)
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c.startswith("theta_se_"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
    if not cols:
        return out, pca_params or {}
    T = out[cols]
    out["theta_mean"] = T.mean(axis=1)
    out["theta_min"] = T.min(axis=1)
    out["theta_max"] = T.max(axis=1)
    out["theta_std"] = T.std(axis=1, ddof=0)
    out["theta_range"] = out["theta_max"] - out["theta_min"]
    out["theta_median"] = T.median(axis=1)
    out["theta_skill_imbalance"] = (T.sub(out["theta_mean"], axis=0).abs()).mean(axis=1)
    out["theta_n_available"] = T.notna().sum(axis=1)
    present = {s: f"theta_{s}" for s in SKILLS if f"theta_{s}" in out.columns}
    for a, b in itertools.combinations(SKILLS, 2):
        if a in present and b in present:
            out[f"theta_{a}_minus_{b}"] = out[present[a]] - out[present[b]]
            out[f"theta_{a}_x_{b}"] = out[present[a]] * out[present[b]]
    for c in cols:
        out[f"theta_missing_{c[6:]}"] = out[c].isna().astype(float)
    if pca_params is None:
        Z = T.copy()
        mu, sd = Z.mean(), Z.std(ddof=0).replace(0, 1)
        Zs = ((Z - mu) / sd).fillna(0.0)
        if Zs.shape[1] >= 2 and len(Zs) >= 5:
            _, _, vt = np.linalg.svd(Zs.to_numpy() - Zs.to_numpy().mean(0), full_matrices=False)
            loadings = vt[0]
            if loadings.sum() < 0:
                loadings = -loadings
        else:
            loadings = np.ones(Zs.shape[1])
        pca_params = {"columns": cols, "mean": mu.to_dict(), "sd": sd.to_dict(), "loadings": loadings.tolist()}
    Zs = ((T[pca_params["columns"]] - pd.Series(pca_params["mean"])) / pd.Series(pca_params["sd"])).fillna(0.0)
    out["theta_common_factor"] = Zs.to_numpy() @ np.asarray(pca_params["loadings"])
    return out, pca_params


def stage_screening_intelligence(ctx) -> dict:
    cfg = ctx.config
    min_n = int(cfg["min_n_correlation"])
    sm = da.students(ctx, "train").set_index("student_id")
    feats, pca = derive_screening_features(sm)
    feats_out = feats.reset_index()
    ctx.save_df(feats_out, "screening_features")
    outputs = ["screening_features"]
    st = da.students(ctx, "test").set_index("student_id")
    if len(st):
        tfeat, _ = derive_screening_features(st, pca)
        ctx.save_df(tfeat.reset_index(), "screening_features_test")
        outputs.append("screening_features_test")
    ctx.save_json(pca, "screening_common_factor_params")
    y = pd.to_numeric(sm["target"], errors="coerce")
    raw = da.raw_theta_columns(feats)
    if not raw:
        ctx.save_json({"status": "NO_SCREENING_VARIABLES"}, "screening_status")
        return {"outputs": outputs + ["screening_status"], "summary": {"status": "NO_SCREENING_VARIABLES"}}
    fcols = [c for c in feats.columns if c.startswith("theta_") and not c.startswith("theta_missing_")]
    assoc = association_table(feats[fcols], y, methods=("pearson", "spearman", "kendall"), min_n=min_n)
    assoc["mutual_information"] = assoc["feature"].map(mutual_information(feats[fcols], y, int(cfg["seed"]), min_n))
    assoc["distance_correlation"] = [distance_correlation(feats[c], y, seed=int(cfg["seed"])) for c in fcols]
    lin_r2, spl_r2 = [], []
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from sklearn.linear_model import RidgeCV

    for c in fcols:
        d = pd.concat([feats[c], y], axis=1).dropna()
        if len(d) < 50 or d[c].nunique() < 6:
            lin_r2.append(np.nan)
            spl_r2.append(np.nan)
            continue
        X, yy = d[[c]].to_numpy(), d.iloc[:, 1].to_numpy()
        lin_r2.append(float(cross_val_score(make_pipeline(StandardScaler(), RidgeCV()), X, yy, cv=5, scoring="r2").mean()))
        spl_r2.append(float(cross_val_score(make_pipeline(SplineTransformer(n_knots=5), RidgeCV()), X, yy, cv=5, scoring="r2").mean()))
    assoc["cv_r2_linear"] = lin_r2
    assoc["cv_r2_spline"] = spl_r2
    assoc["nonlinearity_gain"] = assoc["cv_r2_spline"] - assoc["cv_r2_linear"]
    ctx.save_df(assoc, "screening_associations", also_csv=True)
    outputs.append("screening_associations")
    partial = []
    for c in raw:
        others = [o for o in raw if o != c]
        if others:
            r, p, n = partial_corr(feats[c], y, feats[others], "pearson", min_n)
            rs, ps, _ = partial_corr(feats[c], y, feats[others], "spearman", min_n)
            partial.append({"feature": c, "controls": ",".join(others), "partial_pearson": r, "p_pearson": p,
                            "partial_spearman": rs, "p_spearman": ps, "n": n})
    ctx.save_df(pd.DataFrame(partial), "screening_partial_correlations", fmt="csv")
    outputs.append("screening_partial_correlations")
    sm_full = sm.join(feats[[c for c in feats.columns if c not in sm.columns]])
    cond_rows = []
    for c in raw + ["theta_mean", "theta_common_factor"]:
        bins = quantile_bins(sm_full[c], 5, int(cfg["min_group_size"]), "Q")
        for b, g in sm_full.groupby(bins):
            t = pd.to_numeric(g["target"], errors="coerce").dropna()
            cond_rows.append({"feature": c, "bin": b, "n": len(t), "target_mean": t.mean(), "target_sd": t.std(),
                              "target_q25": t.quantile(.25), "target_q75": t.quantile(.75),
                              "feature_mean": g[c].mean()})
    ctx.save_df(pd.DataFrame(cond_rows), "screening_conditional_outcomes", fmt="csv")
    outputs.append("screening_conditional_outcomes")
    conditioned = []
    for col in ("instrument", "school_id"):
        if col in sm_full.columns and sm_full[col].nunique() > 1:
            for level, g in sm_full.groupby(col):
                for c in raw + ["theta_mean"]:
                    r, p, n = safe_corr(g[c], g["target"], "spearman", int(cfg["min_group_size"]))
                    conditioned.append({"by": col, "level": level, "feature": c, "spearman": r, "p": p, "n": n})
    if conditioned:
        cond = pd.DataFrame(conditioned)
        # shrink school-level correlations towards the pooled correlation (Fisher-z, weight n-3)
        pooled = {c: safe_corr(sm_full[c], sm_full["target"], "spearman", min_n)[0] for c in raw + ["theta_mean"]}
        z0 = cond["feature"].map(lambda f: np.arctanh(np.clip(pooled.get(f, np.nan), -0.999, 0.999)))
        z = np.arctanh(cond["spearman"].clip(-0.999, 0.999))
        w = (cond["n"] - 3).clip(lower=0) / ((cond["n"] - 3).clip(lower=0) + 20)
        cond["spearman_shrunk"] = np.tanh(w * z.fillna(z0) + (1 - w) * z0)
        ctx.save_df(cond, "screening_group_conditioned", fmt="csv")
        outputs.append("screening_group_conditioned")
    Z = feats[raw].apply(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0)
    Zc = Z.dropna()
    pca_rep: dict = {}
    if len(Zc) >= 10 and Zc.shape[1] >= 2:
        _, svals, vt = np.linalg.svd(Zc.to_numpy() - Zc.to_numpy().mean(0), full_matrices=False)
        ev = svals ** 2 / (svals ** 2).sum()
        pca_rep = {"explained_variance_ratio": ev.tolist(), "loadings": {f"PC{i + 1}": dict(zip(raw, vt[i].tolist())) for i in range(len(ev))}}
    ctx.save_json(pca_rep, "screening_pca")
    outputs.append("screening_pca")
    miss = feats[raw].isna()
    miss_rep = {"missing_rate": miss.mean().to_dict(),
                "target_mean_missing_vs_present": {c: {"missing": float(y[miss[c]].mean()) if miss[c].any() else None,
                                                      "present": float(y[~miss[c]].mean())} for c in raw}}
    ctx.save_json(miss_rep, "screening_missingness")
    outputs.append("screening_missingness")
    _screening_figures(ctx, sm_full, raw, assoc, pca_rep)
    best = assoc.reindex(assoc["spearman_r"].abs().sort_values(ascending=False).index).head(5)
    return {"outputs": outputs, "summary": {"theta_columns": raw, "top": best[["feature", "spearman_r"]].to_dict("records"),
                                            "pca_explained": pca_rep.get("explained_variance_ratio")}}


def _screening_figures(ctx, sm: pd.DataFrame, raw: list, assoc: pd.DataFrame, pca_rep: dict) -> None:
    n = int(sm["target"].notna().sum())
    cols = raw + ["target"]
    k = len(cols)
    fig, axes = viz.new_figure(k, k, (2.4, 2.2))
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(pd.to_numeric(sm[a], errors="coerce").dropna(), bins=30)
            else:
                ax.scatter(sm[b], sm[a], s=3, alpha=0.4)
            if i == k - 1:
                ax.set_xlabel(b, fontsize=7)
            if j == 0:
                ax.set_ylabel(a, fontsize=7)
    ctx.figure(fig, "screening_pairplot", "Screening skills and outcome pair plot",
               "How do screening skills relate to each other and to the outcome?", "screening_features", n,
               "Diagonal: histograms; off-diagonal: scatter plots.")
    corr = sm[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    fig, axes = viz.new_figure(1, 1, (6, 5))
    viz.heatmap(axes[0, 0], corr, "Spearman correlation matrix", vmin=-1, vmax=1, annotate=True, cbar_label="rho")
    ctx.figure(fig, "screening_correlation_matrix", "Screening correlation matrix", "How correlated are the skills?",
               "screening_features", n, f"Skill-target rho: " + ", ".join(f"{c}={corr.loc[c, 'target']:.3f}" for c in raw))
    if len(raw) >= 3:
        fig = viz._plt().figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        d = sm[raw[:3] + ["target"]].dropna()
        sc = ax.scatter(d[raw[0]], d[raw[1]], d[raw[2]], c=d["target"], s=6, cmap="viridis")
        ax.set_xlabel(raw[0])
        ax.set_ylabel(raw[1])
        ax.set_zlabel(raw[2])
        fig.colorbar(sc, label="outcome", fraction=0.03)
        ctx.figure(fig, "ability_map_3d", "Three-dimensional ability map", "Where are students in skill space?",
                   "screening_features", len(d), "Colour encodes end-of-year outcome.")
    pairs = list(itertools.combinations(raw, 2))
    if pairs:
        fig, axes = viz.new_figure(1, len(pairs), (5, 4.2))
        for ax, (a, b) in zip(axes.ravel(), pairs):
            surf = viz.binned_surface(sm, a, b, "target", bins=5, min_count=int(ctx.config["min_group_size"]))
            if not surf.empty:
                viz.heatmap(ax, surf, f"mean outcome: {a} (x) x {b} (y)", cmap="viridis", cbar_label="outcome")
        ctx.figure(fig, "screening_response_surfaces", "Pairwise outcome response surfaces",
                   "Do skills combine non-additively?", "screening_features", n,
                   f"Quantile-binned means; cells with < {ctx.config['min_group_size']} students masked.")
    from sklearn.cluster import KMeans

    Z = sm[raw].apply(lambda s: (s - s.mean()) / s.std()).dropna()
    if len(Z) >= 30:
        km = KMeans(4, random_state=int(ctx.config["seed"]), n_init=10).fit(Z)
        prof = pd.DataFrame(km.cluster_centers_, columns=raw)
        fig = viz._plt().figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        viz.radar(ax, raw, {f"profile {i} (n={int((km.labels_ == i).sum())})": prof.loc[i].tolist() for i in prof.index},
                  "Screening profiles (z-scores, k-means k=4, descriptive)")
        ctx.figure(fig, "screening_profile_radar", "Radar charts of screening profiles",
                   "What screening profiles exist?", "screening_features", len(Z),
                   "Descriptive k-means profiles on standardised skills; not used for prediction.")
        dom = sm.loc[Z.index, raw].apply(lambda s: (s - s.mean()) / s.std()).idxmax(axis=1)
        fig, axes = viz.new_figure(1, 2)
        dom.value_counts().plot(kind="bar", ax=axes[0, 0])
        viz.finalize_axis(axes[0, 0], "Dominant (relatively strongest) skill", "skill", "students", len(dom))
        tmp = pd.DataFrame({"dominant": dom, "target": sm.loc[Z.index, "target"]})
        viz.grouped_box(axes[0, 1], tmp, "target", "dominant", "outcome")
        ctx.figure(fig, "skill_dominance", "Skill dominance", "Does relative skill strength relate to outcome?",
                   "screening_features", len(dom), "Dominant skill = largest within-student z-score.")
    fig, axes = viz.new_figure(1, len(raw) + 1, (4, 3.6))
    for ax, c in zip(axes.ravel(), raw + ["theta_mean"]):
        bins = quantile_bins(sm[c], 8, int(ctx.config["min_group_size"]), "Q")
        g = sm.groupby(bins)
        means = g["target"].mean()
        se = g["target"].std() / np.sqrt(g["target"].count())
        x = g[c].mean()
        ax.errorbar(x, means, yerr=1.96 * se, fmt="o-")
        viz.finalize_axis(ax, f"outcome | {c}", c, "mean outcome (95% CI)", int(sm[c].notna().sum()))
    ctx.figure(fig, "conditional_target", "Conditional outcome by screening bins",
               "Is the screening-outcome relation linear?", "screening_conditional_outcomes", n,
               "Quantile bins with at least min_group_size students; error bars are 95% normal CIs.")
