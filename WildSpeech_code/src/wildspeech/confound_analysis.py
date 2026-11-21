"""Section 17 - confounding, artifact detection and robustness.

Scientific question: are discovered acoustic signals genuine speech-related
predictive information or technical shortcuts?

Negative controls preserve grouping structure: student-level shuffles are
performed within school, clip-level shuffles within item. Speech-active
frames are never assumed to contain only the child; the multi-speaker
indicator below is a crude proxy, not speaker identification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .ablation import ColumnSetBuilder, cv_evaluate
from .families import columns_by_family, feature_family, modeling_columns
from .stats_utils import permute_within_groups, safe_corr
from .validation import available_schemes, group_performance, paired_bootstrap_delta, pooled_predictions

SPEECH_FAMILIES = ["TEMPORAL", "SPECTRAL", "MFCC", "WAVELETS", "PROSODY"]


def _cols(S: pd.DataFrame, families) -> list:
    return [c for c in modeling_columns(S.columns, families) if pd.api.types.is_numeric_dtype(S[c])]


def shuffled_within_groups(S: pd.DataFrame, cols: list, groups: pd.Series | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    perm = permute_within_groups(len(S), groups.reindex(S.index).fillna("NA").to_numpy() if groups is not None else None, rng)
    out = S.copy()
    out[cols] = S[cols].to_numpy()[perm]
    return out


def shuffled_within_item_aggregates(ctx, seed: int) -> pd.DataFrame:
    from .student_aggregation import core_features, numeric_clip_features

    clips = da.clip_table(ctx, "train")
    feats = core_features(numeric_clip_features(clips), ctx.config)
    feats = [f for f in feats if f.startswith(("t_", "s_", "m_", "w_", "p_"))]
    if not feats or "item_key" not in clips.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    shuffled = clips[["student_id", "item_key"] + feats].copy()
    for item, idx in shuffled.groupby("item_key").groups.items():
        idx = np.asarray(idx)
        shuffled.loc[idx, feats] = shuffled.loc[rng.permutation(idx), feats].to_numpy()
    out = shuffled.groupby("student_id")[feats].agg(["mean", "std"])
    out.columns = [f"{a}__{b}" for a, b in out.columns]
    return out


def school_separability(S: pd.DataFrame, families: dict, schools: pd.Series, cfg: dict) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    sch = schools.reindex(S.index)
    counts = sch.value_counts()
    keep = counts[counts >= int(cfg["min_group_size"])].index
    mask = sch.isin(keep)
    if len(keep) < 2:
        return pd.DataFrame([{"status": "INSUFFICIENT_SCHOOLS"}])
    yv = sch[mask].astype(str).to_numpy()
    cv = StratifiedKFold(min(5, int(counts[keep].min())), shuffle=True, random_state=int(cfg["seed"]))
    rows = []
    for fam, cols in families.items():
        if not cols:
            continue
        X = S.loc[mask, cols[:400]]
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        try:
            pred = cross_val_predict(model, X, yv, cv=cv)
        except Exception as exc:
            rows.append({"family": fam, "status": f"FAILED: {exc}"})
            continue
        rows.append({"family": fam, "status": "OK", "n_students": int(mask.sum()), "n_schools": int(len(keep)),
                     "balanced_accuracy": float(balanced_accuracy_score(yv, pred)), "chance": 1.0 / len(keep),
                     "n_features": len(cols[:400])})
    return pd.DataFrame(rows)


def confound_matrix(S: pd.DataFrame, meta: pd.DataFrame, families: dict, cfg: dict) -> pd.DataFrame:
    confounders = {}
    for c in ("q_snr_db_approx__mean", "q_duration_s__mean", "meta_n_clips", "meta_orig_sr_mode", "q_noise_floor_db__mean",
              "q_eff_bandwidth_hz__mean", "meta_frac_quality_flagged"):
        if c in S.columns:
            confounders[c] = S[c]
    rows = []
    for fam, cols in families.items():
        sub = cols[:300]
        row = {"family": fam, "n_features": len(cols)}
        for name, conf in confounders.items():
            rs = [abs(safe_corr(S[c], conf, "spearman", 20)[0]) for c in sub if c != name]
            row[name] = float(np.nanmedian(rs)) if rs and np.isfinite(rs).any() else np.nan
        for cat in ("school_id", "instrument"):
            if cat in meta.columns and meta[cat].nunique() > 1:
                g = meta[cat].reindex(S.index)
                etas = []
                for c in sub:
                    x = S[c]
                    m = x.notna() & g.notna()
                    if m.sum() < 20:
                        continue
                    grand = x[m].mean()
                    ss_t = ((x[m] - grand) ** 2).sum()
                    ss_b = x[m].groupby(g[m]).apply(lambda v: len(v) * (v.mean() - grand) ** 2).sum()
                    etas.append(np.sqrt(ss_b / ss_t) if ss_t > 0 else np.nan)
                row[f"eta_{cat}"] = float(np.nanmedian(etas)) if etas else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def sign_consistency(S: pd.DataFrame, residual: pd.Series, feats: list, assign: pd.DataFrame, meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    from .validation import iter_folds

    rows = []
    sch = meta["school_id"].reindex(S.index)
    for f in feats:
        overall = safe_corr(S[f], residual, "spearman", 20)[0]
        signs = []
        for _, _, tr, _ in iter_folds(assign, "CV_A_REPEATED_STUDENT_KFOLD", repeats=[0]):
            tr = [t for t in tr if t in S.index]
            r = safe_corr(S.loc[tr, f], residual.reindex(tr), "spearman", 20)[0]
            if np.isfinite(r) and np.isfinite(overall):
                signs.append(np.sign(r) == np.sign(overall))
        school_rs = {}
        for s, idx in sch.dropna().groupby(sch.dropna()).groups.items():
            if len(idx) >= int(cfg["min_group_size"]):
                school_rs[s] = safe_corr(S.loc[idx, f], residual.reindex(idx), "spearman", int(cfg["min_group_size"]))[0]
        vals = np.array([v for v in school_rs.values() if np.isfinite(v)])
        rows.append({"feature": f, "residual_rho": overall, "fold_sign_consistency": float(np.mean(signs)) if signs else np.nan,
                     "school_sign_consistency": float(np.mean(np.sign(vals) == np.sign(overall))) if len(vals) and np.isfinite(overall) else np.nan,
                     "n_schools": len(vals), "school_rho_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                     **{f"school_{k}": v for k, v in school_rs.items()}})
    return pd.DataFrame(rows)


def stage_confound_analysis(ctx) -> dict:
    from .fold_features import FoldFeatureBuilder

    cfg = ctx.config
    S = da.student_features(ctx, "train")
    meta = da.student_meta(ctx, "train")
    y = meta["target"].dropna().astype(float)
    S = S.reindex(y.index)
    assign = ctx.load_df("cv_assignments")
    residual = da.residuals(ctx).reindex(y.index)
    schools = meta["school_id"] if meta["school_id"].notna().any() else None
    scheme = "CV_A_REPEATED_STUDENT_KFOLD"
    builder = FoldFeatureBuilder(ctx)
    builder.base = builder.base.reindex(y.index)
    screening_cols = builder.columns_for({"SCREENING"})
    duration_cols = [c for c in S.columns if c.startswith(("q_duration_s__", "t_total_dur__", "t_active_dur__"))]
    speech_cols = _cols(S, SPEECH_FAMILIES)
    controls = {
        "NC01_BACKGROUND_ONLY": _cols(S, ["BACKGROUND"]),
        "NC02_RECORDING_METADATA_ONLY": _cols(S, ["RECORDING_METADATA"]),
        "NC03_AUDIO_QUALITY_ONLY": _cols(S, ["QUALITY"]),
        "NC04_DURATION_ONLY": duration_cols,
        "NC07_TECHNICAL_FEATURES_ONLY": _cols(S, ["QUALITY", "BACKGROUND", "RECORDING_METADATA"]),
        "SPEECH_FEATURES_REFERENCE": speech_cols,
    }
    rows, oofs = [], {}
    ref = cv_evaluate(ColumnSetBuilder(builder.base, screening_cols), None, y, assign, scheme, cfg, repeats=[0], groups=schools)
    oofs["SCREENING_REFERENCE"] = pooled_predictions(ref.oof)
    rows.append({"control": "SCREENING_REFERENCE", "variant": "screening", "n_features": len(screening_cols), **ref.summary})

    def run(name, base, cols):
        if not cols:
            rows.append({"control": name, "status": "UNAVAILABLE"})
            return
        for variant, cset in (("alone", cols), ("with_screening", screening_cols + cols)):
            bd = ColumnSetBuilder(base.reindex(y.index), cset)
            res = cv_evaluate(bd, None, y, assign, scheme, cfg, repeats=[0], groups=schools)
            p = pooled_predictions(res.oof)
            oofs[f"{name}|{variant}"] = p
            row = {"control": name, "variant": variant, "n_features": len(cset), **res.summary}
            if variant == "with_screening":
                d = paired_bootstrap_delta(p["y"], p["pred"], oofs["SCREENING_REFERENCE"].reindex(p.index)["pred"], cfg, "r2")
                row.update({"delta_r2_vs_screening": d["delta"], "delta_ci_lower": d["ci_lower"], "delta_ci_upper": d["ci_upper"]})
            rows.append(row)

    base_full = builder.base.join(S[[c for c in S.columns if c not in builder.base.columns]])
    for name, cols in controls.items():
        run(name, base_full, cols)
    run("NC05_SHUFFLED_STUDENT_AUDIO", shuffled_within_groups(base_full, speech_cols, schools, int(cfg["seed"])), speech_cols)
    nc06 = shuffled_within_item_aggregates(ctx, int(cfg["seed"]))
    if not nc06.empty:
        nc06_base = base_full[screening_cols].join(nc06.add_prefix("nc06_"))
        nc06_cols = [c for c in nc06_base.columns if c.startswith("nc06_")]
        run("NC06_SHUFFLED_AUDIO_WITHIN_ITEM", nc06_base, nc06_cols)
    else:
        rows.append({"control": "NC06_SHUFFLED_AUDIO_WITHIN_ITEM", "status": "UNAVAILABLE"})
    regions = {
        "full_recording_features": [c for c in speech_cols if c.startswith(("s_full_", "m_full_"))],
        "speech_active_features": [c for c in speech_cols if c.startswith(("s_sp_", "m_sp_"))],
        "background_region_features": _cols(S, ["BACKGROUND"]),
        "content_conditioned_features": [c for c in speech_cols if "__task=" in c or "__corr=" in c or "__cplx=" in c],
    }
    for name, cols in regions.items():
        run(f"REGION_{name}", base_full, cols)
    ia = cv_evaluate(builder, ["SCREENING", "ITEM_ADJUSTED_AUDIO"], y, assign, scheme, cfg, repeats=[0], groups=schools)
    rows.append({"control": "REGION_item_adjusted_features", "variant": "with_screening", **ia.summary})
    nc = pd.DataFrame(rows)
    ctx.save_df(nc, "negative_controls", also_csv=True)
    fams = {f: c for f, c in columns_by_family([c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]).items()
            if f not in {"OTHER"}}
    sep = school_separability(S, fams, schools, cfg) if schools is not None else pd.DataFrame([{"status": "NO_SCHOOL_INFO"}])
    ctx.save_df(sep, "school_separability", fmt="csv")
    cmat = confound_matrix(S, meta, fams, cfg)
    ctx.save_df(cmat, "confound_matrix", fmt="csv")
    rank = pd.Series({c: abs(safe_corr(S[c], residual, "spearman", 20)[0]) for c in speech_cols}).dropna().sort_values(ascending=False)
    sc = sign_consistency(S, residual, rank.index[:50].tolist(), assign, meta, cfg)
    ctx.save_df(sc, "sign_consistency", fmt="csv")
    multi = {}
    for c in ("p_f0_jump_frac__mean", "t_n_segments__mean", "t_active_ratio__mean"):
        if c in S.columns:
            multi[c] = {"target_rho": safe_corr(S[c], y, "spearman", 20)[0], "residual_rho": safe_corr(S[c], residual, "spearman", 20)[0]}
    sr = {}
    if "meta_orig_sr_mode" in S.columns and S["meta_orig_sr_mode"].nunique() > 1:
        sr = S.groupby("meta_orig_sr_mode").apply(lambda g: pd.Series({"n": len(g), "target_mean": y.reindex(g.index).mean(),
                                                                       "residual_mean": residual.reindex(g.index).mean()})).reset_index().to_dict("records")
    miss = {c: safe_corr(S[c], residual, "spearman", 20)[0] for c in S.columns if c.startswith("meta_missing_") and S[c].nunique() > 1}
    ctx.save_json({"multi_speaker_proxy": multi, "sampling_rate_groups": sr, "missingness_residual_rho": miss,
                   "caveat": "Proxies only: child-only analysis requires a validated speaker-role identification method."},
                  "confound_diagnostics")
    subgroup = []
    multi_res = cv_evaluate(builder, ["SCREENING"] + SPEECH_FAMILIES, y, assign, scheme, cfg, repeats=[0], groups=schools)
    pm = pooled_predictions(multi_res.oof)
    for col in ("school_id", "instrument"):
        if col in meta.columns and meta[col].notna().any():
            for label, p in (("screening", oofs["SCREENING_REFERENCE"]), ("screening_plus_speech", pm)):
                gp = group_performance(p, meta[col], int(cfg["min_group_size"]))
                subgroup.append(gp.assign(by=col, model=label))
    if subgroup:
        ctx.save_df(pd.concat(subgroup, ignore_index=True), "subgroup_performance", fmt="csv")
    gap_rows = []
    if "CV_B_SCHOOL_GROUP_KFOLD" in available_schemes(assign):
        for label, fams_ in (("screening", ["SCREENING"]), ("screening_plus_speech", ["SCREENING"] + SPEECH_FAMILIES)):
            a = cv_evaluate(builder, fams_, y, assign, scheme, cfg, repeats=[0]).summary.get("pooled_r2")
            b = cv_evaluate(builder, fams_, y, assign, "CV_B_SCHOOL_GROUP_KFOLD", cfg).summary.get("pooled_r2")
            gap_rows.append({"model": label, "r2_student_cv": a, "r2_school_grouped_cv": b, "generalization_gap": a - b if a is not None and b is not None else np.nan})
    ctx.save_df(pd.DataFrame(gap_rows), "domain_generalization_gap", fmt="csv")
    _confound_figures(ctx, nc, sep, cmat, sc, meta, S, y)
    return {"outputs": ["negative_controls", "school_separability", "confound_matrix", "sign_consistency",
                        "confound_diagnostics", "domain_generalization_gap"] + (["subgroup_performance"] if subgroup else []),
            "summary": {"negative_controls": nc[["control", "variant", "pooled_r2"]].dropna().to_dict("records") if "pooled_r2" in nc else []}}


def _confound_figures(ctx, nc, sep, cmat, sc, meta, S, y) -> None:
    if "pooled_r2" in nc.columns:
        d = nc.dropna(subset=["pooled_r2"]).copy()
        d["label"] = d["control"] + " | " + d["variant"].fillna("")
        d = d.sort_values("pooled_r2")
        fig, axes = viz.new_figure(1, 1, (8, 0.3 * len(d) + 2))
        axes[0, 0].barh(d["label"], d["pooled_r2"])
        viz.finalize_axis(axes[0, 0], "Negative controls vs reference models (pooled OOF R²)", "R²", "", int(len(y)))
        ctx.figure(fig, "negative_controls", "Negative-control comparison", "Do technical features predict literacy as well as speech features?",
                   "negative_controls", int(len(y)), "Shuffles preserve school (NC05) and item (NC06) grouping.")
    if "balanced_accuracy" in sep.columns:
        s = sep.dropna(subset=["balanced_accuracy"]).sort_values("balanced_accuracy")
        fig, axes = viz.new_figure(1, 1, (7, 0.3 * len(s) + 2))
        axes[0, 0].barh(s["family"], s["balanced_accuracy"])
        axes[0, 0].axvline(s["chance"].iloc[0], color="red", ls="--", label="chance")
        axes[0, 0].legend(fontsize=7)
        viz.finalize_axis(axes[0, 0], "School separability by feature family", "balanced accuracy (CV)", "", int(s["n_students"].max()))
        ctx.figure(fig, "school_separability", "School-prediction diagnostic", "Do features encode recording environment?",
                   "school_separability", int(s["n_students"].max()), "Logistic regression; well above chance suggests environment encoding.")
    if len(cmat) and cmat.shape[1] > 2:
        M = cmat.set_index("family").drop(columns=["n_features"]).astype(float)
        fig, axes = viz.new_figure(1, 1, (8, 0.35 * len(M) + 2))
        viz.heatmap(axes[0, 0], M, "Feature-family confounding (median |rho| or eta)", cmap="viridis", vmin=0, vmax=1, annotate=True)
        ctx.figure(fig, "confound_matrix", "Feature-family confounding matrix", "Which families track technical confounders?",
                   "confound_matrix", int(len(S)), "Median association between family features and each potential confounder.")
    school_cols = [c for c in sc.columns if c.startswith("school_") and c not in ("school_sign_consistency", "school_rho_sd")]
    if school_cols:
        M = sc.set_index("feature")[school_cols].head(30)
        fig, axes = viz.new_figure(1, 1, (max(6, len(school_cols) * 0.3 + 3), 0.25 * len(M) + 2))
        viz.heatmap(axes[0, 0], M, "Per-school residual correlation of top speech features", vmin=-1, vmax=1, cbar_label="rho")
        ctx.figure(fig, "school_stability_heatmap", "School-level feature stability", "Do associations generalise across schools?",
                   "sign_consistency", int(len(S)), f"Median school sign consistency {sc['school_sign_consistency'].median():.2f}.")
    if "meta_orig_sr_mode" in S.columns and S["meta_orig_sr_mode"].nunique() > 1:
        fig, axes = viz.new_figure(1, 1)
        viz.grouped_box(axes[0, 0], pd.DataFrame({"sr": S["meta_orig_sr_mode"].astype(str), "target": y}), "target", "sr", "outcome")
        ctx.figure(fig, "sampling_rate_comparison", "Outcome by original sampling rate", "Could sampling rate act as a shortcut?",
                   "student_features", int(len(S)), "Modal original sampling rate per student.")
