"""Matplotlib figure helpers with one consistent, colour-blind-aware style.

Categorical slots follow a fixed, pre-validated order (never cycled); a series beyond the cap folds into
'Other'. Scatter-type plots (all pairs adjacent) are capped at 3 colours. Sequential = one hue (blue),
diverging = blue <-> red with a grey midpoint. Text always uses ink colours, never series colours.
Every figure helper returns the Figure; the caller saves it AND the underlying table.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#898781"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("ws_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("ws_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#a32727"])
SCATTER_CAP = 3


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "lines.linewidth": 2.0,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES)})


apply_style()


def colors_for(categories, cap: int = len(SERIES)) -> dict:
    """Stable entity -> colour map (sorted order), folding categories beyond the cap into grey 'Other'."""
    cats = [c for c in pd.unique(pd.Series(list(categories)).astype(str))]
    cats = sorted(cats)
    return {c: (SERIES[i] if i < cap else OTHER) for i, c in enumerate(cats)}


def hist_by_group(df: pd.DataFrame, value: str, group: str | None, title: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = df[[value] + ([group] if group else [])].dropna()
    if group:
        cmap = colors_for(data[group], cap=4)
        for g, sub in data.groupby(data[group].astype(str)):
            ax.hist(sub[value], bins=bins, alpha=0.55, color=cmap[g], label=f"{g} (n={len(sub)})", edgecolor=SURFACE, linewidth=0.5)
        ax.legend()
    else:
        ax.hist(data[value], bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(value)
    ax.set_ylabel("count")
    return fig


def box_by_group(df: pd.DataFrame, value: str, group: str, title: str, max_groups: int = 30):
    data = df[[value, group]].dropna()
    order = data.groupby(group)[value].median().sort_values().index[:max_groups]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(order) + 2), 3.8))
    ax.boxplot([data.loc[data[group] == g, value] for g in order], showfliers=False,
               medianprops={"color": SERIES[1], "linewidth": 2}, boxprops={"color": INK2}, whiskerprops={"color": MUTED},
               capprops={"color": MUTED})
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([str(g) for g in order])
    ax.set_title(title)
    ax.set_ylabel(value)
    ax.set_xlabel(group)
    ax.tick_params(axis="x", rotation=60)
    return fig


def scatter_fit(x, y, xlabel: str, ylabel: str, title: str, hue=None):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if hue is not None:
        hue = np.asarray(hue).astype(str)
        cmap = colors_for(hue[ok], cap=SCATTER_CAP)
        for h in sorted(set(hue[ok])):
            m = ok & (hue == h)
            ax.scatter(x[m], y[m], s=14, alpha=0.7, color=cmap[h], label=h, edgecolors=SURFACE, linewidths=0.4)
        ax.legend()
    else:
        ax.scatter(x[ok], y[ok], s=14, alpha=0.6, color=SERIES[0], edgecolors=SURFACE, linewidths=0.4)
    if ok.sum() > 3 and np.ptp(x[ok]) > 0:
        b = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=INK2, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def heatmap(mat: pd.DataFrame, title: str, diverging: bool = True, vmax: float | None = None, annotate: bool = False):
    h, w = mat.shape
    fig, ax = plt.subplots(figsize=(min(0.28 * w + 3, 22), min(0.28 * h + 2, 22)))
    vals = mat.to_numpy(float)
    vm = vmax or (np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0)
    im = ax.imshow(vals, cmap=DIV if diverging else SEQ, vmin=-vm if diverging else np.nanmin(vals), vmax=vm, aspect="auto")
    ax.set_xticks(range(w))
    ax.set_xticklabels([str(c)[:40] for c in mat.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(h))
    ax.set_yticklabels([str(i)[:50] for i in mat.index], fontsize=6)
    ax.grid(False)
    if annotate and h * w <= 400:
        for i in range(h):
            for j in range(w):
                if np.isfinite(vals[i, j]):
                    ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(title)
    return fig


def forest(df: pd.DataFrame, est: str, lo: str, hi: str, label: str, title: str, ref: float = 0.0, max_rows: int = 40):
    d = df.dropna(subset=[est]).head(max_rows).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.24 * len(d) + 1)))
    y = np.arange(len(d))
    sig = (d[lo] > ref) | (d[hi] < ref)
    ax.hlines(y, d[lo], d[hi], color=np.where(sig, SERIES[0], MUTED), linewidth=2)
    ax.scatter(d[est], y, color=np.where(sig, SERIES[0], MUTED), s=28, zorder=3, edgecolors=SURFACE, linewidths=1)
    ax.axvline(ref, color=AXIS, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([str(s)[:60] for s in d[label]], fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(est)
    return fig


def bar_ci(df: pd.DataFrame, x: str, y: str, lo: str | None, hi: str | None, title: str, horizontal: bool = True):
    d = df.copy()
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.28 * len(d) + 1)) if horizontal else (max(5, 0.4 * len(d) + 2), 4))
    pos = np.arange(len(d))
    err = None
    if lo and hi:
        err = np.vstack([(d[y] - d[lo]).clip(lower=0), (d[hi] - d[y]).clip(lower=0)])
    if horizontal:
        ax.barh(pos, d[y], color=SERIES[0], height=0.7, xerr=err, ecolor=INK2, capsize=2, edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(pos)
        ax.set_yticklabels([str(s)[:55] for s in d[x]], fontsize=7)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlabel(y)
    else:
        ax.bar(pos, d[y], color=SERIES[0], width=0.7, yerr=err, ecolor=INK2, capsize=2, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(s)[:30] for s in d[x]], rotation=60, ha="right", fontsize=7)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_ylabel(y)
    ax.set_title(title)
    return fig


def lines(df: pd.DataFrame, x: str, y: str, group: str, title: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cmap = colors_for(df[group], cap=len(SERIES))
    for g, sub in df.groupby(df[group].astype(str)):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], color=cmap[g], label=g, marker="o", markersize=4)
    ax.legend()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    return fig


def embedding_scatter(coords: np.ndarray, labels, title: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    labels = np.asarray(labels).astype(str)
    cats = sorted(set(labels))
    if len(cats) > SCATTER_CAP:
        # small multiples beyond the scatter colour cap
        plt.close(fig)
        n = len(cats)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.8 * rows), squeeze=False)
        for k, c in enumerate(cats):
            a = axes[k // cols][k % cols]
            a.scatter(coords[:, 0], coords[:, 1], s=5, color=GRID)
            m = labels == c
            a.scatter(coords[m, 0], coords[m, 1], s=8, color=SERIES[0])
            a.set_title(f"{c} (n={m.sum()})", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(title)
        return fig
    cmap = colors_for(cats, cap=SCATTER_CAP)
    for c in cats:
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.75, color=cmap[c], label=f"{c} (n={m.sum()})",
                   edgecolors=SURFACE, linewidths=0.4)
    ax.legend()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
