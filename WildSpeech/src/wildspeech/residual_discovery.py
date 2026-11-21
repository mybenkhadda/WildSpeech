"""Section 16 - residual literacy signal discovery (central scientific module).

residual = observed_outcome - oof_screening_prediction (Section 14).

All associations are observational; nothing here is a causal effect. Primary
analyses use continuous residuals. Exploratory discoveries are confirmed on
held-out student halves (features selected on one half, evaluated on the
other) rather than on the observations used for selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .families import feature_family, modeling_columns
from .stats_utils import bh_fdr, bootstrap_ci, cluster_robust_ols, partial_corr, permutation_pvalue, safe_corr, smd, spearman_stat

DISCOVERY_FAMILIES = ["CORRECTNESS", "ITEM_DIFFICULTY", "ADAPTIVE", "LINGUISTIC", "TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS",
                      "PROSODY", "PHONETIC", "ITEM_ADJUSTED_AUDIO", "COMPLEXITY_SENSITIVITY", "CORRECTNESS_CONTRAST",
                      "CROSS_TASK", "EMBEDDINGS", "QUALITY", "BACKGROUND", "RECORDING_METADATA"]


def candidate_features(S: pd.DataFrame, max_missing: float = 0.7) -> list:
    cols = [c for c in S.columns if feature_family(c) in DISCOVERY_FAMILIES and pd.api.types.is_numeric_dtype(S[c])]
    cols = [c for c in cols if not c.endswith(("_reliable", "_n", "_n_min", "_se")) or c.startswith("meta_")]
    keep = []
    for c in cols:
        s = S[c]
        if s.isna().mean() <= max_missing and s.nunique(dropna=True) > 3:
            keep.append(c)
    return keep


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * np.nan


def association_screen(D: pd.DataFrame, feats: list, theta: list, cfg: dict) -> pd.DataFrame:
    min_n = int(cfg["min_n_correlation"])
    schools = D["school_id"] if "school_id" in D.columns and D["school_id"].notna().all() else None
    inst = pd.get_dummies(D["instrument"].astype(str), prefix="inst", drop_first=True, dtype=float) \
        if "instrument" in D.columns and D["instrument"].nunique() > 1 else pd.DataFrame(index=D.index)
    sch_fe = pd.get_dummies(D["school_id"].astype(str), prefix="sch", drop_first=True, dtype=float) \
        if schools is not None and 2 <= D["school_id"].nunique() <= int(cfg["max_schools_for_loso"]) else None
    exposure = [c for c in ("l_n_phonemes__mean", "l_n_chars__mean", "meta_n_clips") if c in D.columns]
    rows = []
    for f in feats:
        x = _z(D[f])
        r = {"feature": f, "family": feature_family(f), "n": int(x.notna().sum())}
        r["target_spearman"], r["target_p"], _ = safe_corr(x, D["target"], "spearman", min_n)
        r["target_pearson"] = safe_corr(x, D["target"], "pearson", min_n)[0]
        r["residual_spearman"], r["residual_p"], _ = safe_corr(x, D["residual"], "spearman", min_n)
        r["residual_pearson"], r["residual_pearson_p"], _ = safe_corr(x, D["residual"], "pearson", min_n)
        if theta:
            r["partial_r_target_given_theta"], r["partial_p"], _ = partial_corr(x, D["target"], D[theta], "pearson", min_n)
            reg = cluster_robust_ols(D["target"], pd.concat([x.rename("x"), D[theta]], axis=1), schools)
            row = reg[reg["term"] == "x"].iloc[0]
            r["screening_adjusted_coef"], r["screening_adjusted_se"], r["screening_adjusted_p"] = row["coef"], row["se"], row["p"]
        for label, ctrl in (("item_adjusted", D[exposure] if exposure else None),
                            ("correctness_adjusted", D[["acc_all"]] if "acc_all" in D.columns else None),
                            ("instrument_adjusted", inst if inst.shape[1] else None),
                            ("school_fe_adjusted", sch_fe)):
            if ctrl is None:
                continue
            reg = cluster_robust_ols(D["residual"], pd.concat([x.rename("x"), ctrl], axis=1), schools)
            row = reg[reg["term"] == "x"].iloc[0]
            r[f"{label}_coef"], r[f"{label}_p"] = row["coef"], row["p"]
        rows.append(r)
    out = pd.DataFrame(rows)
    for p in [c for c in out.columns if c.endswith("_p")]:
        out[p[:-2] + "_q"] = bh_fdr(out[p])
    return out


def held_out_replication(D: pd.DataFrame, feats: list, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]))
    top_k = int(cfg["discovery_top_k"])
    groups = D["school_id"] if "school_id" in D.columns and D["school_id"].nunique() >= 4 else None
    tallies = {f: {"selected": 0, "replicated": 0, "rho_heldout": []} for f in feats}
    for rep in range(int(cfg["discovery_split_repeats"])):
        if groups is not None:
            schools = groups.unique()
            half = set(rng.choice(schools, len(schools) // 2, replace=False))
            a = D.index[groups.isin(half)]
        else:
            a = D.index[rng.permutation(len(D))[: len(D) // 2]]
        b = D.index.difference(a)
        rho_a = {f: spearman_stat(D.loc[a, f], D.loc[a, "residual"]) for f in feats}
        ranked = sorted((f for f in feats if np.isfinite(rho_a[f])), key=lambda f: -abs(rho_a[f]))[:top_k]
        for f in ranked:
            rb, pb, _ = safe_corr(D.loc[b, f], D.loc[b, "residual"], "spearman", 10)
            tallies[f]["selected"] += 1
            tallies[f]["rho_heldout"].append(rb)
            if np.isfinite(rb) and np.sign(rb) == np.sign(rho_a[f]) and pb < 0.05:
                tallies[f]["replicated"] += 1
    rows = []
    for f, t in tallies.items():
        if t["selected"]:
            rows.append({"feature": f, "family": feature_family(f), "times_selected": t["selected"],
                         "times_replicated": t["replicated"], "replication_rate": t["replicated"] / t["selected"],
                         "median_heldout_rho": float(np.nanmedian(t["rho_heldout"]))})
    return pd.DataFrame(rows).sort_values("replication_rate", ascending=False) if rows else pd.DataFrame(
        columns=["feature", "family", "times_selected", "times_replicated", "replication_rate", "median_heldout_rho"])


def greedy_match(treated: pd.DataFrame, control: pd.DataFrame, caliper: float, exact: str | None) -> list:
    pairs = []
    used: set = set()
    for tid, trow in treated.iterrows():
        cand = control if exact is None else control[control[exact] == trow[exact]]
        cand = cand.drop(index=[c for c in cand.index if c in used], errors="ignore")
        if cand.empty:
            continue
        num = cand.drop(columns=[exact] if exact else []).to_numpy(dtype=float)
        t = trow.drop(labels=[exact] if exact else []).to_numpy(dtype=float)
        d = np.sqrt(np.nanmean((num - t) ** 2, axis=1))
        j = int(np.nanargmin(d))
        if d[j] <= caliper:
            pairs.append((tid, cand.index[j], float(d[j])))
            used.add(cand.index[j])
    return pairs


def matched_analysis(D: pd.DataFrame, theta: list, feats: list, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    covs = theta + [c for c in ("meta_n_clips", "acc_all", "q_snr_db_approx__mean", "q_duration_s__mean") if c in D.columns]
    exact = "instrument" if "instrument" in D.columns and D["instrument"].nunique() > 1 else None
    Z = D[covs].apply(_z).fillna(0.0)
    if exact:
        Z[exact] = D[exact].astype(str)
    treated = Z[D["residual_group"] == "POSITIVE_RESIDUAL"]
    control = Z[D["residual_group"] == "NEGATIVE_RESIDUAL"]
    pairs_all, balance, comparisons = [], [], []
    rng = np.random.default_rng(int(cfg["seed"]))
    for cal in cfg["matching_calipers"]:
        pairs = greedy_match(treated, control, float(cal), exact)
        pairs_all += [{"caliper": cal, "treated": a, "control": b, "distance": d} for a, b, d in pairs]
        if not pairs:
            continue
        t_ids = [p[0] for p in pairs]
        c_ids = [p[1] for p in pairs]
        for c in covs:
            balance.append({"caliper": cal, "covariate": c, "smd_before": smd(D.loc[treated.index, c], D.loc[control.index, c]),
                            "smd_after": smd(D.loc[t_ids, c], D.loc[c_ids, c]), "n_pairs": len(pairs)})
        for f in feats:
            diff = D.loc[t_ids, f].to_numpy() - D.loc[c_ids, f].to_numpy()
            diff = diff[np.isfinite(diff)]
            if len(diff) < 10:
                continue
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(min(int(cfg["bootstrap_iterations"]), 500))]
            sd = D[f].std()
            comparisons.append({"caliper": cal, "feature": f, "n_pairs": len(diff), "mean_paired_diff": float(diff.mean()),
                                "std_paired_diff_units": float(diff.mean() / sd) if sd > 0 else np.nan,
                                "ci_lower": float(np.quantile(boots, 0.025)), "ci_upper": float(np.quantile(boots, 0.975))})
    return pd.DataFrame(pairs_all), pd.DataFrame(balance), pd.DataFrame(comparisons)


def stage_residual_discovery(ctx) -> dict:
    cfg = ctx.config
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    resid = ctx.load_df("baseline_residuals").set_index("student_id")
    D = S.join(meta, rsuffix="_meta").join(resid[["residual", "oof_prediction", "residual_group"]], how="inner")
    theta = [c for c in da.raw_theta_columns(D)]
    feats = candidate_features(D)
    ctx.logger.info("Residual discovery over %d candidate features and %d students", len(feats), len(D))
    assoc = association_screen(D, feats, theta, cfg)
    top_boot = assoc.reindex(assoc["residual_spearman"].abs().sort_values(ascending=False).index).head(int(cfg["feature_registry_bootstrap_top_k"]))
    groups = D["school_id"] if D["school_id"].notna().all() and D["school_id"].nunique() > 1 else None
    ci_rows = []
    for f in top_boot["feature"]:
        sub = D[[f, "residual"]].dropna()
        ci = bootstrap_ci(spearman_stat, sub[f].to_numpy(), sub["residual"].to_numpy(),
                          n_boot=int(cfg["bootstrap_iterations"]), seed=int(cfg["seed"]),
                          groups=groups.reindex(sub.index).to_numpy() if groups is not None else None)
        ci_rows.append({"feature": f, "residual_rho_ci_lower": ci["lower"], "residual_rho_ci_upper": ci["upper"]})
    assoc = assoc.merge(pd.DataFrame(ci_rows, columns=["feature", "residual_rho_ci_lower", "residual_rho_ci_upper"]),
                        on="feature", how="left")
    perm_rows = []
    for f in top_boot["feature"].head(int(cfg["permutation_top_k"])):
        pr = permutation_pvalue(D[f], D["residual"], n_perm=int(cfg["permutation_iterations"]),
                                groups=groups.to_numpy() if groups is not None else None, seed=int(cfg["seed"]))
        perm_rows.append({"feature": f, "residual_perm_p_within_school": pr["p_perm"]})
    assoc = assoc.merge(pd.DataFrame(perm_rows, columns=["feature", "residual_perm_p_within_school"]), on="feature", how="left")
    repl = held_out_replication(D, feats, cfg)
    assoc = assoc.merge(repl[["feature", "replication_rate", "median_heldout_rho"]], on="feature", how="left")
    ctx.save_df(assoc, "residual_associations", also_csv=True)
    ctx.save_df(repl, "residual_replication", fmt="csv")
    fam = assoc.groupby("family").agg(n_features=("feature", "size"),
                                      n_residual_fdr=("residual_q", lambda q: int((q < cfg["fdr_alpha"]).sum())),
                                      median_abs_residual_rho=("residual_spearman", lambda s: float(s.abs().median())),
                                      max_abs_residual_rho=("residual_spearman", lambda s: float(s.abs().max())),
                                      mean_replication_rate=("replication_rate", "mean")).reset_index()
    fam = fam.sort_values(["n_residual_fdr", "max_abs_residual_rho"], ascending=False)
    ctx.save_df(fam, "residual_family_ranking", fmt="csv")
    top_feats = assoc.reindex(assoc["residual_spearman"].abs().sort_values(ascending=False).index)["feature"].head(30).tolist()
    pairs, balance, comps = matched_analysis(D, theta, top_feats, cfg)
    ctx.save_df(pairs, "matched_pairs", fmt="csv")
    ctx.save_df(balance, "matched_balance", fmt="csv")
    ctx.save_df(comps, "matched_feature_comparison", fmt="csv")
    _residual_figures(ctx, D, assoc, top_feats, comps, balance)
    return {"outputs": ["residual_associations", "residual_replication", "residual_family_ranking", "matched_pairs",
                        "matched_balance", "matched_feature_comparison"],
            "summary": {"n_features": len(feats), "n_residual_fdr": int((assoc["residual_q"] < cfg["fdr_alpha"]).sum()),
                        "top_families": fam.head(5).to_dict("records"),
                        "note": "Observational associations; not causal effects."}}


def _residual_figures(ctx, D, assoc, top_feats, comps, balance) -> None:
    cfg = ctx.config
    n = int(len(D))
    fig, axes = viz.new_figure(1, 2)
    viz.scatter_fit(axes[0, 0], D["oof_prediction"], D["target"], "OOF screening prediction", "observed", "actual vs predicted")
    viz.hist_kde(axes[0, 1], D["residual"], "OOF screening residual")
    ctx.figure(fig, "residual_overview", "Out-of-fold screening residuals", "What remains unexplained by screening?",
               "baseline_residuals", n, f"Residual SD={D['residual'].std():.3g}.")
    top = assoc[assoc["feature"].isin(top_feats)].copy()
    if "screening_adjusted_coef" in top.columns:
        top["lo"] = top["screening_adjusted_coef"] - 1.96 * top["screening_adjusted_se"]
        top["hi"] = top["screening_adjusted_coef"] + 1.96 * top["screening_adjusted_se"]
        top = top.sort_values("screening_adjusted_coef")
        fig, axes = viz.new_figure(1, 1, (8, 0.25 * len(top) + 2))
        viz.forest(axes[0, 0], top, "feature", "screening_adjusted_coef", "lo", "hi",
                   "Screening-adjusted effect per SD (cluster-robust 95% CI)", "outcome units per feature SD")
        ctx.figure(fig, "residual_forest", "Feature effect-size forest plot", "Which features predict outcome after screening adjustment?",
                   "residual_associations", n, "Top 30 features by |residual rho|; selection and estimation use the same data (exploratory).")
    viz.top_correlation_heatmap(ctx, D, top_feats + ["residual", "target"], "residual_feature_heatmap",
                                "Residual-feature correlation heatmap", "Are residual-related features redundant?", "residual_associations")
    if len(comps):
        c = comps[comps["caliper"] == comps["caliper"].median()] if comps["caliper"].nunique() > 1 else comps
        c = c.sort_values("std_paired_diff_units")
        c = c.assign(lo=c["ci_lower"] / D[c["feature"]].std().to_numpy(), hi=c["ci_upper"] / D[c["feature"]].std().to_numpy())
        fig, axes = viz.new_figure(1, 1, (8, 0.25 * len(c) + 2))
        viz.forest(axes[0, 0], c, "feature", "std_paired_diff_units", "lo", "hi",
                   "Matched positive vs negative residual students (paired diff, SD units)", "SD units")
        ctx.figure(fig, "matched_comparison", "Matched-student comparison", "Do similar-screening students with different outcomes differ acoustically?",
                   "matched_feature_comparison", int(c["n_pairs"].max()), "Greedy nearest-neighbour matching with caliper; descriptive only.")
    if len(balance):
        b = balance.groupby("caliper")[["smd_before", "smd_after"]].apply(lambda g: g.abs().mean())
        fig, axes = viz.new_figure(1, 1)
        b.plot(kind="bar", ax=axes[0, 0])
        viz.finalize_axis(axes[0, 0], "Covariate balance (mean |SMD|) by caliper", "caliper", "|SMD|", n)
        ctx.figure(fig, "matching_balance", "Matching covariate balance", "Were matched groups comparable?", "matched_balance", n,
                   "Lower |SMD| after matching indicates better balance; sensitivity across calipers.")
    if top_feats and "theta_mean" in D.columns:
        f = top_feats[0]
        surf = viz.binned_surface(D, f, "theta_mean", "residual", bins=5, min_count=int(cfg["min_group_size"]))
        if not surf.empty:
            fig, axes = viz.new_figure(1, 1, (7, 5))
            viz.heatmap(axes[0, 0], surf, f"mean residual by {f} (x) and theta_mean (y)", cbar_label="residual")
            ctx.figure(fig, "residual_response_surface", "Residual response surface", "Does the residual association depend on ability?",
                       "residual_associations", n, f"Feature {f}; masked cells have < {cfg['min_group_size']} students.")
        rows = {}
        terc = pd.qcut(D["theta_mean"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for f in top_feats[:25]:
            rows[f] = {f"theta_{lvl}": safe_corr(D.loc[terc == lvl, f], D.loc[terc == lvl, "residual"], "spearman", 15)[0]
                       for lvl in terc.dropna().unique()}
            if "instrument" in D.columns and D["instrument"].nunique() > 1:
                for inst, g in D.groupby("instrument"):
                    rows[f][f"inst_{inst}"] = safe_corr(g[f], g["residual"], "spearman", 15)[0]
        M = pd.DataFrame(rows).T
        fig, axes = viz.new_figure(1, 1, (6, 0.25 * len(M) + 2))
        viz.heatmap(axes[0, 0], M, "Conditional residual associations (Spearman rho)", vmin=-1, vmax=1, cbar_label="rho")
        ctx.figure(fig, "conditional_association_matrix", "Conditional association matrix",
                   "Are residual associations stable across ability and instrument?", "residual_associations", n,
                   "Within-subgroup Spearman correlations (min 15 students).")
