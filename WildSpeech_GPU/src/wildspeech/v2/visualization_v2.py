"""V2 figure helpers, sharing the V1 style (wildspeech.viz): fixed categorical order, single-hue sequential and
blue-red diverging maps, ink-coloured text. Each helper returns a Figure; callers save it with its data table
through registry.save_figure_v2 (figure + CSV + figure_index)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import viz
from ..viz import AXIS, INK2, MUTED, SERIES, SURFACE, colors_for, heatmap  # noqa: F401 - re-exported

forest = viz.forest
bar_ci = viz.bar_ci
lines = viz.lines
embedding_scatter = viz.embedding_scatter


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str):
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmap = colors_for(list(curves), cap=len(SERIES))
    for name in sorted(curves):
        y, p = curves[name]
        m = np.isfinite(y) & np.isfinite(p)
        if len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], p[m])
        ax.plot(fpr, tpr, color=cmap[name], label=f"{name} ({roc_auc_score(y[m], p[m]):.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str, positive: int = 0):
    """Precision-recall for the minority 'incorrect' class by default (positive=0)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    cmap = colors_for(list(curves), cap=len(SERIES))
    base = None
    for name in sorted(curves):
        y, p = curves[name]
        m = np.isfinite(y) & np.isfinite(p)
        yy = (y[m] == positive).astype(int)
        s = p[m] if positive == 1 else 1 - p[m]
        if len(np.unique(yy)) < 2:
            continue
        pr, rc, _ = precision_recall_curve(yy, s)
        ax.plot(rc, pr, color=cmap[name], label=f"{name} ({average_precision_score(yy, s):.3f})", linewidth=1.6)
        base = yy.mean()
    if base is not None:
        ax.axhline(base, color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def reliability(tables: dict[str, pd.DataFrame], title: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cmap = colors_for(list(tables), cap=len(SERIES))
    for name in sorted(tables):
        t = tables[name].dropna(subset=["mean_pred", "frac_correct"])
        ax.plot(t["mean_pred"], t["frac_correct"], marker="o", markersize=3, color=cmap[name], label=name, linewidth=1.4)
    ax.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1)
    ax.set_xlabel("mean predicted P(correct)")
    ax.set_ylabel("observed fraction correct")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def confusion_grid(mats: dict[str, np.ndarray], title: str):
    n = len(mats)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(3.0 * max(n, 1), 3.0), squeeze=False)
    for ax, (name, M) in zip(axes[0], sorted(mats.items())):
        M = np.asarray(M, float)
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(Mn, cmap=viz.SEQ, vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{int(M[i, j])}\n{Mn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.6 else viz.INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["incorrect", "correct"])
        ax.set_title(name, fontsize=9)
        ax.grid(False)
    fig.suptitle(title)
    return fig


def line_ci(df: pd.DataFrame, x: str, y: str, lo: str | None, hi: str | None, group: str, title: str,
            ref: float | None = None, xlabel: str | None = None):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    cmap = colors_for(df[group], cap=len(SERIES))
    for g, d in df.groupby(df[group].astype(str)):
        d = d.sort_values(x)
        ax.plot(d[x], d[y], marker="o", markersize=4, color=cmap[g], label=g)
        if lo and hi and d[lo].notna().any():
            ax.fill_between(d[x], d[lo], d[hi], color=cmap[g], alpha=0.12, linewidth=0)
    if ref is not None:
        ax.axhline(ref, color=AXIS, linewidth=1)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def spectra_small_multiples(df: pd.DataFrame, title: str):
    """df: variant, mel_band, 16k, low_rate (mean band level dB)."""
    variants = sorted(df["variant"].unique())
    cols = min(3, len(variants))
    rows = int(np.ceil(len(variants) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False, sharey=True)
    for k, v in enumerate(variants):
        ax = axes[k // cols][k % cols]
        d = df[df["variant"] == v].sort_values("mel_band")
        for gi, g in enumerate(("16k", "low_rate")):
            if g in d:
                ax.plot(d["mel_band"], d[g], color=SERIES[gi], label=g)
        ax.set_title(v, fontsize=9)
        ax.set_xlabel("mel band")
        if k % cols == 0:
            ax.set_ylabel("mean level (dB)")
    for k in range(len(variants), rows * cols):
        axes[k // cols][k % cols].axis("off")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    return fig


def distributions_by_group(values: dict[str, np.ndarray], title: str, xlabel: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cmap = colors_for(list(values), cap=4)
    for name in sorted(values):
        v = np.asarray(values[name], float)
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=bins, alpha=0.5, density=True, color=cmap[name], label=f"{name} (n={len(v)})", edgecolor=SURFACE, linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=7)
    return fig


def alignment_segments(examples: list[dict], title: str):
    """examples: {label, spans: 'TOK:start:end:gop|...'} -> horizontal token bars coloured by GOP."""
    n = max(len(examples), 1)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 0.9 * n + 1.0), squeeze=False)
    for ax, ex in zip(axes[:, 0], examples):
        toks = [s.split(":") for s in str(ex.get("spans", "")).split("|") if s.count(":") == 3]
        for tok, s, e, g in toks:
            s, e, g = int(s), int(e), float(g)
            col = viz.DIV(0.5 - 0.5 * max(-1.0, min(0.0, g / 4.0)))     # grey = confident, red = low GOP
            ax.barh(0, (e - s) * 0.02, left=s * 0.02, color=col, edgecolor=SURFACE, height=0.8)
            ax.text((s + e) / 2 * 0.02, 0, tok, ha="center", va="center", fontsize=7, color=viz.INK)
        ax.set_yticks([])
        ax.set_title(str(ex.get("label", ""))[:90], fontsize=8, loc="left")
        ax.set_xlabel("time (s)", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def dtw_alignment(panels: list[dict], title: str):
    """panels: {label, C (local cost matrix), path [(i, j)]} -> cost matrices with the optimal warping path."""
    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for ax, pnl in zip(axes[0], panels):
        C = np.asarray(pnl["C"], float)
        C = np.where(C > 1e17, np.nan, C)
        ax.imshow(C, origin="lower", aspect="auto", cmap=viz.SEQ)
        pth = np.asarray(pnl["path"])
        if len(pth):
            ax.plot(pth[:, 1], pth[:, 0], color=SERIES[1], linewidth=1.5)
        ax.set_xlabel("exemplar frame")
        ax.set_ylabel("response frame")
        ax.set_title(str(pnl.get("label", ""))[:60], fontsize=8)
        ax.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def recovery_heatmap(rec: pd.DataFrame, title: str):
    piv = rec.pivot_table(index="population", columns="model", values="mae_reduction")
    return heatmap(piv, title, diverging=True, annotate=True)
