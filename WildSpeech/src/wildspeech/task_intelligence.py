"""Section 12 - task-specific cognitive intelligence.

The three task families (blending, deletion, sentence repetition) are analysed
independently after their dataset names have been resolved in Section 01.
Deletion-specific error categories are only computed when the original item
and the intended deletion operation are both known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .families import feature_family
from .linguistic_features import G2P, tokenize_words
from .stats_utils import bh_fdr, safe_corr

SIGNAL_FAMILIES = {"temporal_signal": ["TEMPORAL"], "spectral_signal": ["SPECTRAL", "MFCC"],
                   "wavelet_signal": ["WAVELETS"], "phonetic_signal": ["PHONETIC"]}


def _assoc(df: pd.DataFrame, cols: list, y: pd.Series, min_n: int) -> pd.DataFrame:
    rows = []
    for c in cols:
        r, p, n = safe_corr(df[c], y.reindex(df.index), "spearman", min_n)
        rows.append({"feature": c, "rho": r, "p": p, "n": n})
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = bh_fdr(out["p"])
    return out


def _school_stability(df: pd.DataFrame, col: str, y: pd.Series, schools: pd.Series, min_n: int) -> float:
    overall = safe_corr(df[col], y.reindex(df.index), "spearman", min_n)[0]
    if not np.isfinite(overall):
        return np.nan
    signs = []
    for _, idx in schools.reindex(df.index).dropna().groupby(schools.reindex(df.index).dropna()).groups.items():
        r = safe_corr(df.loc[idx, col], y.reindex(idx), "spearman", 10)[0]
        if np.isfinite(r):
            signs.append(np.sign(r) == np.sign(overall))
    return float(np.mean(signs)) if signs else np.nan


def relevance_matrix(ctx) -> pd.DataFrame:
    cfg = ctx.config
    min_n = int(cfg["min_n_correlation"])
    S = da.student_features(ctx, "train")
    y = da.target(ctx)
    res = da.residuals(ctx)
    schools = da.student_meta(ctx, "train")["school_id"]
    rows = []
    for task in da.task_families(ctx):
        tag = f"task={task}"
        row = {"task_family": task}
        theta = f"theta_{task}"
        row["screening_signal"] = abs(safe_corr(S[theta], y, "spearman", min_n)[0]) if theta in S.columns else np.nan
        acc = f"acc_{task}"
        row["correctness_signal"] = abs(safe_corr(S[acc], y, "spearman", min_n)[0]) if acc in S.columns else np.nan
        task_cols = [c for c in S.columns if f"__{tag}" in c or c.startswith(f"traj_{task}_") or c.startswith(f"irt_{task}_")]
        best_res, best_col = 0.0, None
        for key, fams in SIGNAL_FAMILIES.items():
            cols = [c for c in task_cols if feature_family(c) in fams]
            a = _assoc(S, cols, y, min_n)
            row[key] = float(a["rho"].abs().max()) if len(a) and a["rho"].notna().any() else np.nan
            row[f"{key}_n_fdr"] = int((a["q"] < cfg["fdr_alpha"]).sum()) if len(a) else 0
        if res is not None:
            a = _assoc(S, task_cols, res, min_n)
            if len(a) and a["rho"].notna().any():
                i = a["rho"].abs().idxmax()
                best_res, best_col = float(abs(a.loc[i, "rho"])), a.loc[i, "feature"]
            row["residual_signal"] = best_res if best_col else np.nan
            row["residual_best_feature"] = best_col
            row["residual_n_fdr"] = int((a["q"] < cfg["fdr_alpha"]).sum()) if len(a) else 0
        else:
            row["residual_signal"] = np.nan
        stab_col = best_col or (acc if acc in S.columns else None)
        row["school_stability"] = _school_stability(S, stab_col, res if best_col else y, schools, min_n) if stab_col else np.nan
        row["school_stability_feature"] = stab_col
        rows.append(row)
    return pd.DataFrame(rows)


def task_specific_analyses(ctx) -> pd.DataFrame:
    cfg = ctx.config
    min_n = int(cfg["min_n_correlation"])
    clips = da.clip_table(ctx, "train")
    y = da.target(ctx)
    res = da.residuals(ctx)
    rows = []

    def add(task, analysis, series, status="OK", note=""):
        if series is None:
            rows.append({"task_family": task, "analysis": analysis, "status": status, "note": note})
            return
        s = series.dropna()
        r, p, n = safe_corr(s, y.reindex(s.index), "spearman", min_n)
        rr = safe_corr(s, res.reindex(s.index), "spearman", min_n)[0] if res is not None else np.nan
        rows.append({"task_family": task, "analysis": analysis, "status": status if n >= min_n else "INSUFFICIENT_DATA",
                     "n_students": n, "target_rho": r, "target_p": p, "residual_rho": rr, "note": note})

    ph = [c for c in clips.columns if c.startswith("ph_")]
    alias = sorted({c.split("_")[1] for c in ph}) if ph else []
    pfx = f"ph_{alias[0]}" if alias else None
    for task in sorted(clips["task_family"].dropna().unique()):
        g = clips[clips["task_family"] == task]
        by = g.groupby("student_id")
        if task == "blending":
            add(task, "segmented_speech_fraction", by["t_n_segments"].apply(lambda s: (s > 1).mean()) if "t_n_segments" in g else None,
                "UNAVAILABLE", "requires acoustic features")
            if pfx:
                add(task, "partial_word_rate", by[f"{pfx}_word_completeness"].apply(lambda s: (s < 1).mean()), note="ASR proxy")
                add(task, "sound_repetition_rate", by[f"{pfx}_self_corrections"].mean(), note="ASR proxy")
            else:
                add(task, "partial_word_rate", None, "UNAVAILABLE", "requires ASR")
            add(task, "phonetic_transition_rate", by["m_sp_transition_rate"].mean() if "m_sp_transition_rate" in g else None,
                "UNAVAILABLE", "requires acoustic features")
        elif task == "deletion":
            if {"deletion_source_text", "deletion_target_sound"} <= set(g.columns) and pfx:
                g2p = G2P(cfg.get("g2p_backend", "auto"))
                cats = []
                for _, r in g.iterrows():
                    src = [p for w in tokenize_words(r["deletion_source_text"]) for p in g2p.word(w)]
                    target = [p for w in tokenize_words(r["deletion_target_sound"]) for p in g2p.word(w)]
                    exp = [p for w in tokenize_words(r.get("expected_text", "")) for p in g2p.word(w)]
                    hyp = [p for w in tokenize_words(r.get(f"{pfx}_text", "")) for p in g2p.word(w)] if f"{pfx}_text" in g else []
                    if not hyp:
                        cats.append(np.nan)
                    elif hyp == src:
                        cats.append("no_deletion")
                    elif hyp == exp:
                        cats.append("correct_deletion")
                    elif any(t in hyp for t in target) and len(hyp) < len(src):
                        cats.append("incorrect_sound_removed")
                    elif len(hyp) < len(exp):
                        cats.append("additional_omission")
                    else:
                        cats.append("substitution_or_other")
                g = g.assign(deletion_category=cats)
                for cat in ("no_deletion", "incorrect_sound_removed", "additional_omission", "substitution_or_other"):
                    add(task, f"rate_{cat}", g.groupby("student_id")["deletion_category"].apply(lambda s: (s == cat).mean()), note="ASR proxy")
            else:
                add(task, "deletion_error_categories", None, "UNAVAILABLE",
                    "original item and intended deletion operation not both known, or ASR disabled")
            if pfx:
                add(task, "response_completeness", by[f"{pfx}_phone_completeness"].mean(), note="ASR proxy")
        elif task == "repetition":
            if pfx:
                add(task, "word_preservation", by[f"{pfx}_word_er"].apply(lambda s: 1 - s.clip(upper=1).mean()), note="ASR proxy")
                add(task, "word_order_preservation", by[f"{pfx}_word_order_preservation"].mean(), note="ASR proxy")
                add(task, "sentence_completeness", by[f"{pfx}_word_completeness"].mean(), note="ASR proxy")
                add(task, "omission_rate", by[f"{pfx}_word_del_rate"].mean(), note="ASR proxy")
                add(task, "repetition_rate", by[f"{pfx}_self_corrections"].mean(), note="ASR proxy")
            else:
                add(task, "word_preservation", None, "UNAVAILABLE", "requires ASR")
            if "t_n_long_pauses" in g:
                add(task, "phrase_level_long_pauses", by["t_n_long_pauses"].mean())
            if "l_n_words" in g and "t_active_dur" in g:
                from .student_aggregation import shrunk_slopes

                sl = shrunk_slopes(g, "student_id", "l_n_words", "t_active_dur", k=float(cfg["slope_shrinkage_k"]),
                                   min_obs=int(cfg["aggregation_min_obs"]))
                add(task, "sentence_length_sensitivity", sl["slope_shrunk"], note="shrunk duration-per-word slope")
            if "p_f0_std_st" in g:
                add(task, "prosodic_organisation_f0_variability", by["p_f0_std_st"].mean(), note="voice-conversion sensitive")
    return pd.DataFrame(rows)


def stage_task_intelligence(ctx) -> dict:
    rel = relevance_matrix(ctx)
    ctx.save_df(rel, "task_relevance_matrix", also_csv=True)
    spec = task_specific_analyses(ctx)
    ctx.save_df(spec, "task_specific_report", also_csv=True)
    cols = ["screening_signal", "correctness_signal", "temporal_signal", "spectral_signal", "wavelet_signal",
            "phonetic_signal", "residual_signal", "school_stability"]
    M = rel.set_index("task_family")[[c for c in cols if c in rel.columns]].astype(float)
    if len(M):
        fig, axes = viz.new_figure(1, 1, (8, 0.6 * len(M) + 2.5))
        viz.heatmap(axes[0, 0], M, "Task-level scientific relevance (|Spearman rho|; stability = sign agreement)",
                    cmap="viridis", vmin=0, vmax=1, annotate=True)
        ctx.figure(fig, "task_relevance_matrix", "Task-level scientific relevance matrix",
                   "Which tasks carry the strongest information?", "task_relevance_matrix", int(len(da.target(ctx))),
                   "Max |rho| per family among task-conditioned features; residual uses OOF screening residuals.")
    return {"outputs": ["task_relevance_matrix", "task_specific_report"],
            "summary": rel[["task_family"] + [c for c in cols if c in rel.columns]].to_dict("records")}
