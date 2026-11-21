"""Section 05 - adaptive assessment and behavioural trajectories.

Scientific question: does the pattern of item responses provide information
beyond the final screening score?

Sequential analysis requires a reliable presentation order (a resolved
order/position column or a timestamp). Dataframe row order is never used. If
no valid order exists the stage reports why and produces no trajectory
features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_access as da
from . import visualization as viz
from .stats_utils import linear_slope, safe_corr

TRAJECTORY_FEATURES = ["n_attempted", "longest_correct_streak", "longest_error_streak", "error_recovery_rate",
                       "accuracy_early", "accuracy_late", "accuracy_decay", "difficulty_progression_slope",
                       "duration_progression_slope", "duration_difficulty_slope", "response_sequence_entropy",
                       "response_volatility", "response_autocorr_lag1", "difficulty_at_stop", "accuracy_last3"]


def order_validity(clips: pd.DataFrame) -> dict:
    report: dict = {"valid": False, "source": None, "reason": ""}
    if "order" in clips.columns and clips["order"].notna().mean() > 0.9:
        dup = clips.dropna(subset=["order"]).duplicated(["student_id", "task_family", "order"]).mean()
        report.update({"source": "order", "duplicate_rate_within_student_task": float(dup)})
        if dup <= 0.05:
            report.update(valid=True, reason="resolved order column with <=5% within-student-task duplicates")
            return report
        report["reason"] = f"order column has {dup:.1%} duplicated positions within student-task"
    if "timestamp" in clips.columns and clips["timestamp"].notna().mean() > 0.9:
        dup = clips.dropna(subset=["timestamp"]).duplicated(["student_id", "timestamp"]).mean()
        report.update({"source": "timestamp", "duplicate_rate_within_student": float(dup)})
        if dup <= 0.05:
            report.update(valid=True, reason="timestamp column with <=5% duplicates")
            return report
        report["reason"] = f"timestamps have {dup:.1%} duplicates"
    if not report["reason"]:
        report["reason"] = "no resolved order or timestamp column; dataframe order is not a valid substitute"
    return report


def _longest_run(seq: np.ndarray, value: float) -> int:
    best = cur = 0
    for v in seq:
        cur = cur + 1 if v == value else 0
        best = max(best, cur)
    return best


def sequence_features(seq: np.ndarray, diff: np.ndarray | None, dur: np.ndarray | None) -> dict:
    n = len(seq)
    out = {k: np.nan for k in TRAJECTORY_FEATURES}
    out["n_attempted"] = float(n)
    if n == 0:
        return out
    out["longest_correct_streak"] = float(_longest_run(seq, 1.0))
    out["longest_error_streak"] = float(_longest_run(seq, 0.0))
    if n >= 2:
        prev, nxt = seq[:-1], seq[1:]
        err = prev == 0
        out["error_recovery_rate"] = float(nxt[err].mean()) if err.any() else np.nan
        out["response_volatility"] = float((prev != nxt).mean())
        trans = pd.crosstab(prev, nxt, normalize="index")
        p_prev = pd.Series(prev).value_counts(normalize=True)
        h = 0.0
        for a in trans.index:
            row = trans.loc[a].to_numpy()
            row = row[row > 0]
            h += p_prev.get(a, 0) * -(row * np.log2(row)).sum()
        out["response_sequence_entropy"] = float(h)
        if np.std(seq) > 0 and n >= 4:
            out["response_autocorr_lag1"] = float(np.corrcoef(prev, nxt)[0, 1])
    half = n // 2
    if n >= 4:
        out["accuracy_early"] = float(seq[:half].mean())
        out["accuracy_late"] = float(seq[half:].mean())
        out["accuracy_decay"] = out["accuracy_early"] - out["accuracy_late"]
    out["accuracy_last3"] = float(seq[-3:].mean())
    pos = np.arange(n, dtype=float)
    if diff is not None:
        out["difficulty_progression_slope"] = linear_slope(pos, diff)
        out["difficulty_at_stop"] = float(diff[-1]) if np.isfinite(diff[-1]) else np.nan
    if dur is not None:
        out["duration_progression_slope"] = linear_slope(pos, dur)
        if diff is not None:
            out["duration_difficulty_slope"] = linear_slope(diff, dur)
    return out


def stage_adaptive_testing(ctx) -> dict:
    cfg = ctx.config
    clips = da.clips(ctx, "train")
    validity = order_validity(clips)
    ctx.save_json(validity, "order_validity")
    if not validity["valid"]:
        ctx.logger.warning("Sequential analysis disabled: %s", validity["reason"])
        return {"outputs": ["order_validity"], "summary": validity}
    if "correct" not in clips.columns or clips["correct"].notna().sum() == 0:
        validity["reason"] += "; no binary correctness"
        ctx.save_json(validity, "order_validity")
        return {"outputs": ["order_validity"], "summary": validity}
    key = validity["source"]
    d = clips.dropna(subset=[key, "correct"]).copy()
    diff = da.item_difficulty(ctx)
    if diff is not None:
        d = d.merge(diff[["item_key", "b"]], on="item_key", how="left")
    else:
        p = d.groupby("item_key")["correct"].transform("mean").clip(0.02, 0.98)
        d["b"] = -np.log(p / (1 - p))
    dur_col = None
    if ctx.stage_available("06"):
        cf = da.clip_features(ctx, "train")
        if cf is not None and "t_active_dur" in cf.columns:
            d = d.merge(cf[["clip_id", "t_active_dur"]], on="clip_id", how="left")
            dur_col = "t_active_dur"
    if dur_col is None and "response_duration" in d.columns:
        dur_col = "response_duration"
    rows = []
    seq_rows = []
    for (sid, task), g in d.sort_values(key).groupby(["student_id", "task_family"], sort=False):
        seq = g["correct"].to_numpy(dtype=float)
        feats = sequence_features(seq, g["b"].to_numpy(dtype=float), g[dur_col].to_numpy(dtype=float) if dur_col else None)
        feats.update({"student_id": sid, "task_family": task})
        rows.append(feats)
        for pos, (c, b) in enumerate(zip(seq, g["b"].to_numpy(dtype=float))):
            seq_rows.append({"student_id": sid, "task_family": task, "position": pos, "correct": c, "b": b})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="student_id", columns="task_family", values=TRAJECTORY_FEATURES)
    wide.columns = [f"traj_{task}_{feat}" for feat, task in wide.columns]
    for feat in TRAJECTORY_FEATURES:
        cols = [c for c in wide.columns if c.endswith(f"_{feat}")]
        wide[f"traj_all_{feat}"] = wide[cols].mean(axis=1)
    ctx.save_df(wide.reset_index(), "trajectory_features")
    seqs = pd.DataFrame(seq_rows)
    ctx.save_df(seqs, "response_sequences")
    trans = []
    for task, g in seqs.sort_values(["student_id", "position"]).groupby("task_family"):
        prev = g.groupby("student_id")["correct"].shift(1)
        t = pd.crosstab(prev.dropna(), g.loc[prev.notna(), "correct"], normalize="index")
        for a in t.index:
            for b in t.columns:
                trans.append({"task_family": task, "from": a, "to": b, "probability": float(t.loc[a, b])})
    ctx.save_df(pd.DataFrame(trans), "transition_matrix", fmt="csv")
    y = da.target(ctx)
    assoc = []
    for c in wide.columns:
        r, p, n = safe_corr(wide[c], y.reindex(wide.index), "spearman", int(cfg["min_n_correlation"]))
        assoc.append({"feature": c, "target_rho": r, "p": p, "n": n})
    ctx.save_df(pd.DataFrame(assoc), "trajectory_target_associations", fmt="csv")
    _trajectory_figures(ctx, seqs, pd.DataFrame(trans), wide, y)
    return {"outputs": ["order_validity", "trajectory_features", "response_sequences", "transition_matrix",
                        "trajectory_target_associations"],
            "summary": {"order_source": key, "n_students": int(len(wide)), "duration_source": dur_col}}


def _trajectory_figures(ctx, seqs: pd.DataFrame, trans: pd.DataFrame, wide: pd.DataFrame, y: pd.Series) -> None:
    scr = da.screening(ctx)
    for task, g in seqs.groupby("task_family"):
        M = g.pivot_table(index="student_id", columns="position", values="correct")
        if scr is not None and "theta_mean" in scr.columns:
            M = M.loc[scr["theta_mean"].reindex(M.index).sort_values().index]
        M = M.iloc[:, :60]
        sample = M if len(M) <= 400 else M.iloc[np.linspace(0, len(M) - 1, 400).astype(int)]
        fig, axes = viz.new_figure(1, 2, (6, 5))
        axes[0, 0].imshow(sample.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        viz.finalize_axis(axes[0, 0], f"{task}: student x position correctness (white = not attempted)", "position",
                          "students (sorted by screening)", len(M))
        prof = g.groupby("position")["b"].agg(["mean", "std", "count"])
        prof = prof[prof["count"] >= 5]
        axes[0, 1].errorbar(prof.index, prof["mean"], yerr=1.96 * prof["std"] / np.sqrt(prof["count"]), fmt="-o", ms=2)
        viz.finalize_axis(axes[0, 1], f"{task}: difficulty trajectory", "position", "mean item difficulty (95% CI)", len(M))
        ctx.figure(fig, f"sequence_heatmap_{task}", f"Response sequences and difficulty trajectory: {task}",
                   "How do students progress through adaptive items?", "response_sequences", len(M),
                   f"Median sequence length {g.groupby('student_id').size().median():.0f}.")
    if len(trans):
        tasks = trans["task_family"].unique()
        fig, axes = viz.new_figure(1, len(tasks), (3.5, 3.2))
        for ax, task in zip(axes.ravel(), tasks):
            t = trans[trans["task_family"] == task].pivot(index="from", columns="to", values="probability")
            viz.heatmap(ax, t, f"{task}: P(next | previous)", cmap="viridis", vmin=0, vmax=1, annotate=True)
        ctx.figure(fig, "correctness_transitions", "Correctness transition matrices", "Do errors cluster?",
                   "transition_matrix", int(seqs["student_id"].nunique()), "Rows: previous response; columns: next response.")
