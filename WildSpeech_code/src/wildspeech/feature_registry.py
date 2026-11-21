"""Section 24 - feature discovery registry.

Tracks every generated student-level feature with provenance, availability,
association evidence (raw and residual, adjusted, FDR, bootstrap), CV
incremental value (student-level and school-grouped), cross-school sign
consistency, redundancy, confounding risk and voice-conversion sensitivity.
Evidence is kept in separate rankings rather than collapsed into a single
arbitrary importance score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from .families import COMPUTATION_COST, FAMILY_HYPOTHESIS, INTERPRETABILITY, TECHNICAL_FAMILIES, feature_family, feature_subfamily

STAT_WORDS = {"mean": "mean", "std": "standard deviation", "median": "median", "mad": "median absolute deviation",
              "min": "minimum", "max": "maximum", "iqr": "interquartile range", "cv": "coefficient of variation",
              "count": "observation count", "p10": "10th percentile", "p25": "25th percentile", "p75": "75th percentile",
              "p90": "90th percentile"}


def describe(col: str) -> tuple[str, str, str]:
    parts = col.split("__")
    base = parts[0]
    stat = parts[-1] if len(parts) > 1 and parts[-1] in STAT_WORDS else ""
    conds = [p for p in parts[1:] if "=" in p]
    level = "student" if not stat and not conds else "clip->student"
    desc = f"{STAT_WORDS.get(stat, stat)} of {base}" if stat else base
    if conds:
        desc += " restricted to " + ", ".join(conds)
    return desc, ("conditional:" + ",".join(conds) if conds else (stat or "direct")), level


def _source(col: str) -> str:
    fam = feature_family(col)
    return {"SCREENING": "train/test_student_metadata.csv", "CORRECTNESS": "clip metadata (correctness)",
            "ITEM_DIFFICULTY": "IRT on clip correctness", "ADAPTIVE": "ordered clip correctness",
            "LINGUISTIC": "expected text (local G2P)", "PHONETIC": "local ASR vs expected text",
            "EMBEDDINGS": "frozen foundation model"}.get(fam, "anonymised audio (unified extraction)")


def _dependencies(fam: str) -> str:
    return {"WAVELETS": "PyWavelets", "PROSODY": "librosa.pyin", "PHONETIC": "transformers+torch (ASR), G2P",
            "EMBEDDINGS": "transformers+torch", "ITEM_DIFFICULTY": "scipy (IRT)", "LINGUISTIC": "G2P backend"}.get(fam, "numpy/scipy/librosa")


def redundancy_groups(S: pd.DataFrame, cols: list, threshold: float, max_cols: int = 1500) -> pd.Series:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    cols = cols[:max_cols]
    if len(cols) < 2:
        return pd.Series(dtype=object)
    R = S[cols].rank().corr(min_periods=20).abs().fillna(0.0).to_numpy()
    np.fill_diagonal(R, 1.0)
    D = np.clip(1 - R, 0, 1)
    Z = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Z, t=1 - threshold, criterion="distance")
    return pd.Series([f"RG{l:05d}" for l in lab], index=cols)


def classify(r: pd.Series, alpha: float, min_n: int) -> tuple[str, str]:
    if r.get("valid_observation_count", 0) < min_n or not np.isfinite(r.get("variance", np.nan)) or r.get("variance", 0) == 0:
        return "UNAVAILABLE", "too few valid observations or constant"
    q = r.get("fdr_adjusted_pvalue", np.nan)
    if not np.isfinite(q):
        return "INSUFFICIENT_EVIDENCE", "residual association not testable"
    ci_excl = np.isfinite(r.get("bootstrap_ci_lower", np.nan)) and (r["bootstrap_ci_lower"] > 0 or r["bootstrap_ci_upper"] < 0)
    cons = r.get("cross_school_sign_consistency", np.nan)
    cons_ok = (not np.isfinite(cons)) or cons >= 0.75
    sig = q < alpha
    if (sig or ci_excl) and r.get("confound_risk") == "HIGH":
        return "CONFOUNDED_SIGNAL", "residual association but high technical confounding risk"
    if sig and np.isfinite(cons) and cons < 0.6:
        return "UNSTABLE_SIGNAL", f"significant but school sign consistency {cons:.2f}"
    cv_ok = (not np.isfinite(r.get("cv_delta_r2", np.nan))) or r["cv_delta_r2"] > 0
    sg_ok = (not np.isfinite(r.get("school_grouped_delta_r2", np.nan))) or r["school_grouped_delta_r2"] > 0
    if sig and ci_excl and cons_ok and cv_ok and sg_ok:
        if r.get("is_redundancy_leader") is False:
            return "REDUNDANT_SIGNAL", f"robust but redundant with {r.get('redundancy_leader')}"
        return "ROBUST_SIGNAL", "FDR-significant, bootstrap CI excludes 0, stable across schools, positive CV gain"
    if (q < 0.2 or ci_excl) and (not np.isfinite(cons) or cons >= 0.6):
        if r.get("is_redundancy_leader") is False:
            return "REDUNDANT_SIGNAL", f"promising but redundant with {r.get('redundancy_leader')}"
        return "PROMISING_SIGNAL", "suggestive residual association; needs confirmation"
    return "WEAK_SIGNAL", "no reliable residual association"


def stage_feature_registry(ctx) -> dict:
    from .ablation import ColumnSetBuilder, cv_evaluate
    from .residual_discovery import association_screen, candidate_features
    from .stats_utils import bootstrap_ci, mutual_information, safe_corr, spearman_stat
    from .validation import available_schemes, paired_bootstrap_delta, pooled_predictions

    cfg = ctx.config
    alpha = float(cfg["fdr_alpha"])
    min_n = int(cfg["min_n_correlation"])
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    S = S.reindex(y.index)
    resid = ctx.load_df("baseline_residuals").set_index("student_id")
    D = S.join(meta, rsuffix="_meta").join(resid[["residual", "residual_group", "oof_prediction"]])
    all_cols = [c for c in S.columns if pd.api.types.is_numeric_dtype(S[c]) and feature_family(c) != "OTHER"]
    if ctx.stage_available("16"):
        assoc = ctx.load_df("residual_associations")
    else:
        assoc = association_screen(D, candidate_features(D), da.raw_theta_columns(D), cfg)
    assoc = assoc.set_index("feature")
    sens = ctx.load_df("anonymization_sensitivity").set_index("feature")["sensitivity"] if ctx.stage_available("10") else pd.Series(dtype=object)
    sep = ctx.load_df("school_separability") if ctx.stage_available("17") else pd.DataFrame()
    sep_map = sep.set_index("family")["balanced_accuracy"] / sep.set_index("family")["chance"] if "balanced_accuracy" in sep else pd.Series(dtype=float)
    abl = ctx.load_df("ablation_results") if ctx.stage_available("20") else pd.DataFrame()
    artifacts = [c for c in ("q_snr_db_approx__mean", "q_noise_floor_db__mean", "q_eff_bandwidth_hz__mean", "s_bg_level_db__mean",
                             "meta_orig_sr_mode") if c in S.columns]
    schools = meta["school_id"].reindex(S.index)
    mi = mutual_information(S[all_cols[:3000]], y, int(cfg["seed"]), min_n) if all_cols else pd.Series(dtype=float)
    rows = []
    for c in all_cols:
        fam = feature_family(c)
        x = S[c]
        desc, agg, level = describe(c)
        a = assoc.loc[c] if c in assoc.index else None
        art = [abs(safe_corr(x, S[q], "spearman", min_n)[0]) for q in artifacts if q != c]
        art_max = float(np.nanmax(art)) if art and np.isfinite(art).any() else np.nan
        sep_ratio = float(sep_map.get(fam, np.nan)) if len(sep_map) else np.nan
        risk = "HIGH" if fam in TECHNICAL_FAMILIES or (np.isfinite(art_max) and art_max >= 0.5) or \
            sens.get(c.split("__")[0]) == "RECORDING_ARTIFACT_SENSITIVE" else (
            "MEDIUM" if (np.isfinite(art_max) and art_max >= 0.3) or (np.isfinite(sep_ratio) and sep_ratio >= 2) else "LOW")
        base_name = c.split("__")[0].replace("ia_", "", 1)
        vc = "POTENTIALLY_VOICE_CONVERSION_SENSITIVE" if sens.get(base_name) == "POTENTIALLY_VOICE_CONVERSION_SENSITIVE" else (
            str(sens.get(base_name)) if base_name in sens.index else ("NOT_APPLICABLE" if fam in {"SCREENING", "CORRECTNESS", "ITEM_DIFFICULTY", "ADAPTIVE", "LINGUISTIC"} else "UNDETERMINED"))
        signs = []
        r_all = a["residual_spearman"] if a is not None else safe_corr(x, D["residual"], "spearman", min_n)[0]
        for s, ids in schools.dropna().groupby(schools.dropna()).groups.items():
            if len(ids) >= int(cfg["min_group_size"]) and np.isfinite(r_all):
                rs = safe_corr(x.loc[ids], D.loc[ids, "residual"], "spearman", int(cfg["min_group_size"]))[0]
                if np.isfinite(rs):
                    signs.append(np.sign(rs) == np.sign(r_all))
        row = {
            "feature_id": f"F{len(rows):06d}", "feature_name": c, "feature_family": fam, "feature_subfamily": feature_subfamily(c),
            "description": desc, "scientific_hypothesis": FAMILY_HYPOTHESIS.get(fam, ""), "source": _source(c),
            "source_columns": c.split("__")[0], "extraction_version": cfg["extraction_version"], "aggregation_method": agg,
            "hierarchical_level": level, "task": next((p.split("=")[1] for p in c.split("__") if p.startswith("task=")), "all"),
            "required_dependencies": _dependencies(fam), "availability": "AVAILABLE" if x.notna().any() else "UNAVAILABLE",
            "missing_rate": float(x.isna().mean()), "valid_observation_count": int(x.notna().sum()),
            "variance": float(x.var()) if x.notna().sum() > 1 else np.nan,
            "target_pearson": a["target_pearson"] if a is not None else safe_corr(x, y, "pearson", min_n)[0],
            "target_spearman": a["target_spearman"] if a is not None else safe_corr(x, y, "spearman", min_n)[0],
            "target_mutual_information": float(mi.get(c, np.nan)),
            "residual_pearson": a["residual_pearson"] if a is not None else safe_corr(x, D["residual"], "pearson", min_n)[0],
            "residual_spearman": r_all,
            "adjusted_regression_coefficient": a.get("screening_adjusted_coef", np.nan) if a is not None else np.nan,
            "effect_size": a.get("screening_adjusted_coef", np.nan) / y.std() if a is not None else np.nan,
            "pvalue": a["residual_p"] if a is not None else np.nan,
            "fdr_adjusted_pvalue": a["residual_q"] if a is not None else np.nan,
            "bootstrap_ci_lower": a.get("residual_rho_ci_lower", np.nan) if a is not None else np.nan,
            "bootstrap_ci_upper": a.get("residual_rho_ci_upper", np.nan) if a is not None else np.nan,
            "cv_delta_r2": np.nan, "school_grouped_delta_r2": np.nan,
            "cross_school_sign_consistency": float(np.mean(signs)) if signs else np.nan,
            "confound_risk": risk, "artifact_max_abs_rho": art_max, "family_school_separability_ratio": sep_ratio,
            "voice_conversion_sensitivity": vc, "computation_cost": COMPUTATION_COST.get(fam, "UNKNOWN"),
            "interpretability": INTERPRETABILITY.get(fam, "UNKNOWN"),
        }
        rows.append(row)
    reg = pd.DataFrame(rows)
    # bootstrap CIs for top features not covered by Section 16
    need = reg[reg["bootstrap_ci_lower"].isna()].reindex(reg["residual_spearman"].abs().sort_values(ascending=False).index).dropna(subset=["residual_spearman"])
    need = need.head(int(cfg["feature_registry_bootstrap_top_k"]))
    for i, r in need.iterrows():
        sub = D[[r["feature_name"], "residual"]].dropna()
        ci = bootstrap_ci(spearman_stat, sub.iloc[:, 0].to_numpy(), sub["residual"].to_numpy(), n_boot=min(int(cfg["bootstrap_iterations"]), 500),
                          seed=int(cfg["seed"]))
        reg.loc[i, "bootstrap_ci_lower"], reg.loc[i, "bootstrap_ci_upper"] = ci["lower"], ci["upper"]
    # per-feature CV incremental value (top-K by |residual rho|)
    assign = ctx.load_df("cv_assignments")
    num = S[[c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]]
    screening = da.raw_theta_columns(num)
    top = reg.reindex(reg["residual_spearman"].abs().sort_values(ascending=False).index).dropna(subset=["residual_spearman"])
    top = top[~top["feature_family"].isin(["SCREENING"])].head(int(cfg["feature_registry_cv_top_k"]))
    refs = {}
    for scheme in [s for s in ("CV_A_REPEATED_STUDENT_KFOLD", "CV_B_SCHOOL_GROUP_KFOLD") if s in available_schemes(assign)]:
        rep = [0] if scheme.startswith("CV_A") else None
        refs[scheme] = pooled_predictions(cv_evaluate(ColumnSetBuilder(num, screening), None, y, assign, scheme, cfg, repeats=rep).oof)
        col = "cv_delta_r2" if scheme.startswith("CV_A") else "school_grouped_delta_r2"
        for i, r in top.iterrows():
            p = pooled_predictions(cv_evaluate(ColumnSetBuilder(num, screening + [r["feature_name"]]), None, y, assign, scheme, cfg, repeats=rep).oof)
            common = p.index.intersection(refs[scheme].index)
            reg.loc[i, col] = paired_bootstrap_delta(p.loc[common, "y"], p.loc[common, "pred"], refs[scheme].loc[common, "pred"],
                                                     {**cfg, "bootstrap_iterations": 50}, "r2")["delta"]
    groups = redundancy_groups(S, reg.reindex(reg["residual_spearman"].abs().sort_values(ascending=False).index)["feature_name"].tolist(),
                               float(cfg["redundancy_threshold"]))
    reg["redundancy_group"] = reg["feature_name"].map(groups)
    strength = reg.set_index("feature_name")["residual_spearman"].abs()
    leaders = reg.dropna(subset=["redundancy_group"]).assign(s=lambda d: d["feature_name"].map(strength)).sort_values("s", ascending=False) \
        .groupby("redundancy_group")["feature_name"].first()
    reg["redundancy_leader"] = reg["redundancy_group"].map(leaders)
    reg["is_redundancy_leader"] = reg["redundancy_leader"].isna() | (reg["redundancy_leader"] == reg["feature_name"])
    cls = reg.apply(lambda r: classify(r, alpha, min_n), axis=1)
    reg["selection_status"] = [c[0] for c in cls]
    reg["selection_reason"] = [c[1] for c in cls]
    if len(abl):
        fam_gain = abl[(abl["scheme"] == "CV_A_REPEATED_STUDENT_KFOLD")].groupby("families")["delta_r2_vs_full_tabular"].max()
        reg["family_ablation_context"] = reg["feature_family"].map(
            lambda f: float(fam_gain[[f in k.split("+") for k in fam_gain.index]].max()) if any(f in k.split("+") for k in fam_gain.index) else np.nan)
    ctx.save_df(reg, "feature_registry", also_csv=True)
    rankings = {
        "predictive": reg.assign(score=reg["target_spearman"].abs()).sort_values("score", ascending=False),
        "incremental": reg.dropna(subset=["cv_delta_r2"]).sort_values("cv_delta_r2", ascending=False),
        "interpretability": reg.assign(i=reg["interpretability"].map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}).fillna(3),
                                       s=-reg["residual_spearman"].abs()).sort_values(["i", "s"]),
        "cross_school_robustness": reg.dropna(subset=["cross_school_sign_consistency"]).sort_values(
            ["cross_school_sign_consistency", "residual_spearman"], ascending=False),
        "residual": reg.assign(score=reg["residual_spearman"].abs()).sort_values("score", ascending=False),
        "computational_efficiency": reg.assign(c=reg["computation_cost"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2, "VERY_HIGH": 3}).fillna(4),
                                               s=-reg["residual_spearman"].abs()).sort_values(["c", "s"]),
        "confounding_risk": reg.assign(c=reg["confound_risk"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})).sort_values(["c", "artifact_max_abs_rho"]),
    }
    outputs = ["feature_registry"]
    for name, df in rankings.items():
        ctx.save_df(df[["feature_name", "feature_family", "selection_status", "target_spearman", "residual_spearman", "cv_delta_r2",
                        "cross_school_sign_consistency", "confound_risk", "computation_cost", "interpretability"]].head(500),
                    f"feature_ranking_{name}", fmt="csv")
        outputs.append(f"feature_ranking_{name}")
    evidence = reg.set_index("feature_name")[["selection_status", "selection_reason", "fdr_adjusted_pvalue", "bootstrap_ci_lower",
                                              "bootstrap_ci_upper", "cross_school_sign_consistency", "cv_delta_r2",
                                              "school_grouped_delta_r2", "confound_risk", "redundancy_group", "redundancy_leader"]]
    ctx.save_df(evidence.reset_index(), "feature_retention_evidence", fmt="csv")
    outputs.append("feature_retention_evidence")
    from . import visualization as viz

    counts = reg.groupby(["feature_family", "selection_status"]).size().unstack(fill_value=0)
    fig, axes = viz.new_figure(1, 1, (9, 0.35 * len(counts) + 2))
    counts.plot(kind="barh", stacked=True, ax=axes[0, 0])
    viz.finalize_axis(axes[0, 0], "Feature registry: selection status by family", "features", "", int(len(y)))
    ctx.figure(fig, "feature_registry_status", "Feature discovery registry", "Which features carry robust signal?", "feature_registry",
               int(len(y)), "; ".join(f"{k}={v}" for k, v in reg["selection_status"].value_counts().items()))
    return {"outputs": outputs, "summary": reg["selection_status"].value_counts().to_dict()}
