"""Section 13 - intelligent student-level aggregation.

The target is student-level: clips are never treated as independent labelled
regression examples. This engine converts clip-level measurements into
student-level representations (global, conditional, cross-task,
item-conditioned, complexity-sensitivity and correctness-contrast features)
with observation counts, reliability flags and missingness indicators.

Item reference statistics used for item adjustment are fitted on training
students only; for predictive evaluation they are refitted within each fold
(see ``fold_features.FoldFeatureBuilder``). Difficulty-conditioned aggregates
(``__diff=``) use all-training item parameters and are EDA-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .families import columns_by_family, sanitize_token

DEFAULT_CORE = [
    "q_duration_s", "q_snr_db_approx", "t_active_dur", "t_active_ratio", "t_initial_sil", "t_internal_sil",
    "t_n_pauses", "t_pause_speech_ratio", "t_pause_dur_mean", "t_n_segments", "t_rhythm_regularity",
    "t_temporal_entropy", "t_mod_freq_hz", "t_onset_sharpness", "s_sp_centroid_mean", "s_sp_bandwidth_mean",
    "s_sp_flatness_mean", "s_sp_entropy_mean", "s_sp_flux_mean", "s_sp_zcr_mean", "s_sp_slope_mean",
    "s_sp_tf_entropy", "m_sp_cepstral_variability", "m_sp_transition_rate", "m_sp_cepstral_traj_length_per_s",
    "w_db4_entropy", "w_db4_transient_density", "w_wp_entropy", "p_f0_std_st", "p_voiced_frac", "p_hnr_db_median",
]
CLIP_PREFIXES = ("q_", "t_", "s_", "m_", "w_", "p_", "ph_", "meta_", "e_")
QUANTILE_STATS = {"p05": 0.05, "p10": 0.10, "p25": 0.25, "p75": 0.75, "p90": 0.90, "p95": 0.95}


def numeric_clip_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(CLIP_PREFIXES) and pd.api.types.is_numeric_dtype(df[c])
            and not c.startswith("meta_decode")]


def core_features(available: list, cfg: dict) -> list:
    core = cfg.get("aggregation_core_features") or DEFAULT_CORE
    core = [c for c in core if c in available]
    ph_conf = [c for c in available if c.startswith("ph_") and c.endswith(("_seq_confidence", "_phone_er", "_word_er"))]
    return core + [c for c in ph_conf if c not in core]


def group_stats(df: pd.DataFrame, key: str, feats: list, stats: list, min_obs: int) -> pd.DataFrame:
    if not feats:
        return pd.DataFrame(index=pd.Index(df[key].unique(), name=key))
    g = df.groupby(key, sort=False)[feats]
    parts = []
    count = g.count()
    if "mean" in stats or "cv" in stats:
        mean = g.mean()
    for s in stats:
        if s == "mean":
            frame = mean
        elif s == "median":
            frame = g.median()
        elif s == "std":
            frame = g.std(ddof=1)
        elif s == "min":
            frame = g.min()
        elif s == "max":
            frame = g.max()
        elif s in QUANTILE_STATS:
            frame = g.quantile(QUANTILE_STATS[s])
        elif s == "iqr":
            frame = g.quantile(0.75) - g.quantile(0.25)
        elif s == "mad":
            med = g.transform("median")
            frame = (df[feats] - med).abs().groupby(df[key], sort=False).median()
        elif s == "cv":
            sd = g.std(ddof=1)
            frame = (sd / mean.abs().where(mean.abs() > 1e-9)).where(count >= min_obs)
        elif s == "count":
            frame = count.astype(float)
        else:
            raise ValueError(s)
        if s in {"std", "mad", "iqr"} or s in QUANTILE_STATS:
            frame = frame.where(count >= (2 if s == "std" else min_obs))
        frame = frame.copy()
        frame.columns = [f"{c}__{s}" for c in feats]
        parts.append(frame)
    return pd.concat(parts, axis=1)


def conditional_stats(df: pd.DataFrame, cond: str, label: str, feats: list, stats: list, min_obs: int) -> pd.DataFrame:
    d = df.dropna(subset=[cond])
    if d.empty or not feats:
        return pd.DataFrame()
    pieces = []
    for level, g in d.groupby(cond):
        st = group_stats(g, "student_id", feats, stats + ["count"] if "count" not in stats else stats, min_obs)
        tag = f"{label}={sanitize_token(level)}"
        st.columns = [c.replace("__", f"__{tag}__", 1) for c in st.columns]
        pieces.append(st)
    return pd.concat(pieces, axis=1)


def shrunk_slopes(df: pd.DataFrame, group: str, x: str, y: str, k: float, min_obs: int) -> pd.DataFrame:
    """Per-group OLS slopes shrunk towards the pooled within-group slope (empirical Bayes style)."""
    d = df[[group, x, y]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["slope_raw", "slope_shrunk", "n", "reliable"])
    g = d.groupby(group)
    dx = d[x] - g[x].transform("mean")
    dy = d[y] - g[y].transform("mean")
    n = g.size()
    sxx = (dx ** 2).groupby(d[group]).sum()
    sxy = (dx * dy).groupby(d[group]).sum()
    raw = sxy / sxx.where(sxx > 1e-12)
    pooled = float(sxy.sum() / sxx.sum()) if sxx.sum() > 0 else np.nan
    w = n / (n + k)
    reliable = (n >= min_obs) & (sxx > 1e-12)
    shrunk = (w * raw + (1 - w) * pooled).where(reliable, pooled)
    return pd.DataFrame({"slope_raw": raw, "slope_shrunk": shrunk, "n": n, "reliable": reliable.astype(float)})


def item_reference(fit: pd.DataFrame, feats: list, k: float) -> dict:
    task_mean = fit.groupby("task_family")[feats].mean()
    task_sd = fit.groupby("task_family")[feats].std(ddof=1)
    g = fit.groupby("item_key")
    n = g[feats].count()
    m = g[feats].mean()
    task_of_item = g["task_family"].first()
    prior = task_mean.reindex(task_of_item.values)
    prior.index = task_of_item.index
    shrunk = (n * m.fillna(0) + k * prior) / (n + k)
    return {"item_mean": shrunk, "task_mean": task_mean, "task_sd": task_sd.replace(0, np.nan)}


def apply_item_reference(clips: pd.DataFrame, feats: list, ref: dict) -> pd.DataFrame:
    mu = ref["item_mean"].reindex(clips["item_key"]).to_numpy()
    fallback = ref["task_mean"].reindex(clips["task_family"]).to_numpy()
    mu = np.where(np.isnan(mu), fallback, mu)
    sd = ref["task_sd"].reindex(clips["task_family"]).to_numpy()
    z = (clips[feats].to_numpy(dtype=float) - mu) / sd
    return pd.DataFrame(z, columns=feats, index=clips.index)


def item_adjusted_features(clips: pd.DataFrame, feats: list, fit_students, k: float, prefix: str = "ia_") -> pd.DataFrame:
    feats = [f for f in feats if f in clips.columns]
    if not feats or "item_key" not in clips.columns or clips["item_key"].isna().all():
        return pd.DataFrame()
    fit = clips[clips["student_id"].isin(set(fit_students))]
    ref = item_reference(fit, feats, k)
    z = apply_item_reference(clips, feats, ref)
    z["student_id"] = clips["student_id"].to_numpy()
    z["task_family"] = clips["task_family"].to_numpy()
    g = z.groupby("student_id")[feats]
    out = pd.concat([g.mean().add_prefix(prefix).add_suffix("__mean"), g.std(ddof=1).add_prefix(prefix).add_suffix("__std")], axis=1)
    for task, gt in z.groupby("task_family"):
        out = out.join(gt.groupby("student_id")[feats].mean().add_prefix(prefix).add_suffix(f"__task={sanitize_token(task)}__mean"))
    alias = {"item_adjusted_duration": ["t_active_dur"],
             "item_adjusted_spectral_features": [f for f in feats if f.startswith("s_sp_")],
             "item_adjusted_phonetic_confidence": [f for f in feats if f.startswith("ph_") and f.endswith("seq_confidence")],
             "item_adjusted_fluency": ["t_pause_speech_ratio", "t_n_pauses", "t_internal_sil"]}
    means = g.mean()
    for name, cols in alias.items():
        cols = [c for c in cols if c in means.columns]
        if cols:
            sign = -1.0 if name == "item_adjusted_fluency" else 1.0
            out[f"{prefix}{name}"] = sign * means[cols].mean(axis=1)
    out[f"{prefix}item_adjusted_acoustic_variability"] = g.std(ddof=1).mean(axis=1)
    out.index.name = "student_id"
    return out


def accuracy_features(clips: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.Index(clips["student_id"].unique(), name="student_id"))
    out["meta_n_clips"] = clips.groupby("student_id").size()
    out["meta_n_tasks"] = clips.groupby("student_id")["task_family"].nunique()
    for task, g in clips.groupby("task_family"):
        out[f"acc_n_attempted_{task}"] = g.groupby("student_id").size()
    out = out.fillna({c: 0 for c in out.columns if c.startswith("acc_n_attempted_")})
    if "correct" not in clips.columns or clips["correct"].notna().sum() == 0:
        return out
    out["acc_all"] = clips.groupby("student_id")["correct"].mean()
    out["acc_n_scored_all"] = clips.groupby("student_id")["correct"].count()
    for task, g in clips.groupby("task_family"):
        gs = g.groupby("student_id")["correct"]
        mean, n = gs.mean(), gs.count()
        prior = g["correct"].mean()
        out[f"acc_{task}"] = mean
        out[f"acc_{task}_n"] = n
        out[f"acc_{task}_shrunk"] = (mean * n + prior * k) / (n + k)
    return out


def correctness_contrasts(clips: pd.DataFrame, feats: dict, min_obs: int) -> pd.DataFrame:
    if "correct" not in clips.columns:
        return pd.DataFrame()
    frames = []
    tasks = [("all", clips)] + list(clips.groupby("task_family"))
    for task, g in tasks:
        for label, col in feats.items():
            if col not in g.columns:
                continue
            c = g[g["correct"] == 1].groupby("student_id")[col].agg(["mean", "var", "count"])
            i = g[g["correct"] == 0].groupby("student_id")[col].agg(["mean", "var", "count"])
            j = c.join(i, lsuffix="_c", rsuffix="_i", how="outer")
            name = f"cc_correct_minus_incorrect_{label}__{sanitize_token(task)}"
            ok = (j["count_c"] >= min_obs) & (j["count_i"] >= min_obs)
            diff = (j["mean_c"] - j["mean_i"]).where(ok)
            se = np.sqrt(j["var_c"] / j["count_c"] + j["var_i"] / j["count_i"]).where(ok)
            frames.append(pd.DataFrame({name: diff, f"{name}_se": se,
                                        f"{name}_n_min": j[["count_c", "count_i"]].min(axis=1).fillna(0)}))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def complexity_sensitivity(clips: pd.DataFrame, feats: list, cplx: str, k: float, min_obs: int) -> pd.DataFrame:
    frames = []
    targets = [f for f in feats if f in clips.columns] + (["correct"] if "correct" in clips.columns else [])
    for task, g in [("all", clips)] + list(clips.groupby("task_family")):
        for f in targets:
            sl = shrunk_slopes(g, "student_id", cplx, f, k, min_obs)
            if sl.empty:
                continue
            name = f"cx_{f}__vs_{cplx}__{sanitize_token(task)}"
            frames.append(pd.DataFrame({name: sl["slope_shrunk"], f"{name}_reliable": sl["reliable"], f"{name}_n": sl["n"]}))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def cross_task(clips: pd.DataFrame, core: list, acc: pd.DataFrame) -> pd.DataFrame:
    tasks = sorted(clips["task_family"].dropna().unique())
    out = pd.DataFrame(index=pd.Index(clips["student_id"].unique(), name="student_id"))
    if len(tasks) < 2:
        return out
    means = clips.groupby(["student_id", "task_family"])[core].mean()
    sds = clips.groupby("student_id")[core].std(ddof=1)
    asym = []
    for f in core:
        wide = means[f].unstack()
        for i, a in enumerate(tasks):
            for b in tasks[i + 1:]:
                if a in wide.columns and b in wide.columns:
                    out[f"xt_{f}__{a}_minus_{b}"] = wide[a] - wide[b]
                    asym.append(((wide[a] - wide[b]).abs() / sds[f].where(sds[f] > 0)).rename(f))
        out[f"xt_{f}__between_task_std"] = wide.std(axis=1, ddof=1)
    if asym:
        out["xt_acoustic_asymmetry"] = pd.concat(asym, axis=1).mean(axis=1)
    acc_cols = [f"acc_{t}" for t in tasks if f"acc_{t}" in acc.columns]
    if len(acc_cols) >= 2:
        out["xt_skill_asymmetry_accuracy_range"] = acc[acc_cols].max(axis=1) - acc[acc_cols].min(axis=1)
        out["xt_accuracy_between_task_std"] = acc[acc_cols].std(axis=1, ddof=1)
        out["xt_strongest_task_accuracy_gap"] = acc[acc_cols].max(axis=1) - acc[acc_cols].median(axis=1)
    return out


def build_student_features(ctx, split: str) -> tuple[pd.DataFrame, dict]:
    cfg = ctx.config
    min_obs = int(cfg["aggregation_min_obs"])
    k_slope = float(cfg["slope_shrinkage_k"])
    sm = da.students(ctx, split).set_index("student_id")
    clips = da.clip_table(ctx, split, with_audio=ctx.stage_available("06"))
    report: dict = {"split": split}
    feats = numeric_clip_features(clips)
    core = core_features(feats, cfg)
    report["n_clip_features"] = len(feats)
    report["core_features"] = core
    out = pd.DataFrame(index=sm.index)
    acc = accuracy_features(clips)
    out = out.join(acc)
    glob = group_stats(clips, "student_id", [f for f in feats if f not in core], list(cfg["aggregation_global_stats"]), min_obs)
    core_stats = group_stats(clips, "student_id", core, list(cfg["aggregation_core_stats"]), min_obs)
    out = out.join(glob).join(core_stats)
    ling = [c for c in clips.columns if c.startswith("l_") and pd.api.types.is_numeric_dtype(clips[c])]
    if ling:
        out = out.join(group_stats(clips, "student_id", ling, ["mean", "max"], min_obs))
        out = out.join(conditional_stats(clips, "task_family", "task", ling, ["mean"], min_obs))
    if core:
        cond_stats = ["mean", "std"]
        out = out.join(conditional_stats(clips, "task_family", "task", core, cond_stats, min_obs))
        if "correct" in clips.columns:
            out = out.join(conditional_stats(clips, "correct", "corr", core, cond_stats, min_obs))
        if "item_b" in clips.columns and clips["item_b"].notna().any():
            items = clips.drop_duplicates("item_key")["item_b"]
            lo, hi = items.quantile([1 / 3, 2 / 3])
            clips["diff_level"] = pd.cut(clips["item_b"], [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
            out = out.join(conditional_stats(clips, "diff_level", "diff", core, cond_stats, min_obs))
        cplx = next((c for c in ("l_n_phonemes", "l_n_chars") if c in clips.columns and clips[c].notna().any()), None)
        if cplx:
            item_cplx = ctx.load_df("item_linguistic_features")[cplx]
            lo, hi = item_cplx.quantile([1 / 3, 2 / 3])
            clips["cplx_level"] = pd.cut(clips[cplx], [-np.inf, lo, hi, np.inf], labels=["low", "mid", "high"])
            out = out.join(conditional_stats(clips, "cplx_level", "cplx", core, cond_stats, min_obs))
        if "q_snr_db_approx" in clips.columns:
            qual = pd.Series(np.where(clips["q_snr_db_approx"] >= float(cfg["quality_good_snr_db"]), "good", "poor"),
                             index=clips.index, dtype=object)
            clips["qual_level"] = qual.where(clips["q_snr_db_approx"].notna())
            out = out.join(conditional_stats(clips, "qual_level", "qual", core, cond_stats, min_obs))
        if "q_duration_s" in clips.columns:
            clips["len_level"] = pd.cut(clips["q_duration_s"], cfg["length_bins_s"], labels=False, include_lowest=True)
            out = out.join(conditional_stats(clips, "len_level", "len", core, cond_stats, min_obs))
        if "t_active_ratio" in clips.columns:
            clips["act_level"] = pd.cut(clips["t_active_ratio"], cfg["activity_bins"], labels=False, include_lowest=True)
            out = out.join(conditional_stats(clips, "act_level", "act", core, cond_stats, min_obs))
        order_ok = ctx.stage_available("05") and (ctx.load_json("order_validity", required=False) or {}).get("valid")
        if order_ok:
            key = ctx.load_json("order_validity")["source"]
            rank = clips.groupby(["student_id", "task_family"])[key].rank(pct=True)
            stage = pd.Series(np.where(rank <= 0.5, "early", "late"), index=clips.index, dtype=object)
            clips["stage_level"] = stage.where(rank.notna())
            out = out.join(conditional_stats(clips, "stage_level", "stage", core, cond_stats, min_obs))
        phc = next((c for c in clips.columns if c.startswith("ph_") and c.endswith("_seq_confidence")), None)
        if phc:
            clips["phconf_level"] = pd.cut(clips[phc], cfg["phonetic_confidence_bins"], labels=False, include_lowest=True)
            out = out.join(conditional_stats(clips, "phconf_level", "phconf", core, cond_stats, min_obs))
        out = out.join(cross_task(clips, core, acc))
        contrast_map = {"duration": "t_active_dur", "pause_ratio": "t_pause_speech_ratio",
                        "spectral_entropy": "s_sp_entropy_mean",
                        "wavelet_entropy": f"w_{cfg['wavelet_primary_family']}_entropy"}
        if phc:
            contrast_map["phoneme_confidence"] = phc
        out = out.join(correctness_contrasts(clips, contrast_map, max(2, min_obs - 1)))
        if cplx:
            cx_feats = [f for f in ("t_active_dur", "t_n_pauses", "t_pause_speech_ratio", "s_sp_entropy_std", "m_sp_cepstral_variability") if f in clips.columns]
            if phc:
                cx_feats.append(phc)
            out = out.join(complexity_sensitivity(clips, cx_feats, cplx, k_slope, min_obs))
        train_students = da.students(ctx, "train")["student_id"]
        ref_clips = da.clip_table(ctx, "train", with_audio=True) if split != "train" else clips
        both = pd.concat([ref_clips, clips], ignore_index=True) if split != "train" else clips
        ia = item_adjusted_features(both, core, train_students, float(cfg["item_shrinkage_k"]))
        if not ia.empty:
            out = out.join(ia.reindex(out.index))
    tab_sources = [("03", "screening_features" if split == "train" else "screening_features_test"),
                   ("04", "irt_student_features"), ("05", "trajectory_features"),
                   ("23", "embedding_student_features" if split == "train" else "embedding_student_features_test")]
    for stage, name in tab_sources:
        if ctx.stage_available(stage) and ctx.has(name):
            tab = ctx.load_df(name).set_index("student_id")
            tab = tab[[c for c in tab.columns if c not in out.columns]]
            out = out.join(tab)
            report[f"merged_{name}"] = int(tab.shape[1])
    if not any(c.startswith("theta_") for c in out.columns):
        raw = [c for c in sm.columns if c.startswith("theta_")]
        out = out.join(sm[raw])
    fams = columns_by_family(out.columns)
    for fam, cols in fams.items():
        out[f"meta_missing_{fam.lower()}"] = out[cols].isna().all(axis=1).astype(float)
    if "proc_status" in clips.columns:
        out["meta_frac_failed_audio"] = clips.groupby("student_id")["proc_status"].apply(lambda s: float((s == "FAILED").mean()))
    if "q_n_flags" in clips.columns:
        out["meta_frac_quality_flagged"] = clips.groupby("student_id")["q_n_flags"].apply(lambda s: float((s > 0).mean()))
    if "audio_sample_rate" in clips.columns:
        out["meta_orig_sr_mode"] = clips.groupby("student_id")["audio_sample_rate"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[:, ~out.columns.duplicated()]
    report["n_students"] = int(len(out))
    report["n_features"] = int(out.shape[1])
    report["family_counts"] = {k: len(v) for k, v in columns_by_family(out.columns).items()}
    return out, report


def stage_student_aggregation(ctx) -> dict:
    feats, report = build_student_features(ctx, "train")
    if feats.index.duplicated().any():
        raise RuntimeError("student feature table has duplicated students")
    ctx.save_df(feats.reset_index(), "student_features")
    outputs = ["student_features"]
    reports = {"train": report}
    if ctx.config.get("allow_test_inference"):
        tfeats, trep = build_student_features(ctx, "test")
        missing_cols = [c for c in feats.columns if c not in tfeats.columns]
        tfeats = tfeats.reindex(columns=feats.columns)
        trep["columns_missing_in_test"] = len(missing_cols)
        ctx.save_df(tfeats.reset_index(), "student_features_test")
        outputs.append("student_features_test")
        reports["test"] = trep
    ctx.save_json(reports, "aggregation_report")
    outputs.append("aggregation_report")
    fam = pd.Series(report["family_counts"]).sort_values()
    fig, axes = viz.new_figure(1, 2, (6, 4.5))
    fam.plot(kind="barh", ax=axes[0, 0])
    viz.finalize_axis(axes[0, 0], "Student-level features per family", "features", "family", report["n_students"])
    miss = feats.isna().mean()
    viz.hist_kde(axes[0, 1], miss.to_numpy(), "per-feature missing rate")
    ctx.figure(fig, "aggregation_overview", "Student-level aggregation overview",
               "What student representation was built?", "student_features", report["n_students"],
               f"{report['n_features']} features; median missing rate {miss.median():.1%}.")
    return {"outputs": outputs, "summary": {"n_students": report["n_students"], "n_features": report["n_features"]}}
