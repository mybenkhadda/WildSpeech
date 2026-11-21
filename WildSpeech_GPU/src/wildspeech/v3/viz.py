"""V3 figure helpers (matplotlib, static PNG). One consistent style:

* categorical hues in fixed slot order (validated reference palette, light surface), never cycled; > 8 series
  fold into small multiples or "Other" at the call site
* sequential = single blue ramp; diverging = blue <-> red with a neutral gray midpoint
* one y-axis per panel, thin marks, recessive grid, text in ink colours (never the series colour)
* every figure is saved by save_figure_v3 together with the data table it was drawn from (the table view)
Every function returns a matplotlib Figure and never touches disk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "sans-serif", "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
    return plt


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("v3seq", SEQ)


def div_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("v3div", [DIV_NEG, DIV_MID, DIV_POS])


def color_for(keys) -> dict:
    """Stable entity -> colour (sorted keys, fixed slot order). Callers fold > 8 entities beforehand."""
    ks = sorted(map(str, keys))
    return {k: SERIES[i % len(SERIES)] for i, k in enumerate(ks[: len(SERIES)])}


def empty(title: str, note: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", fontweight="bold")
    ax.text(0.5, 0.3, note, ha="center", color=INK2, wrap=True)
    return fig


# ------------------------------------------------------------------------------------------- forest
def forest(df: pd.DataFrame, est: str, lo: str, hi: str, label: str, title: str, ref: float = 0.0,
           group: str | None = None, xlabel: str = "ΔR² vs tabular GAM"):
    plt = _plt()
    d = df.dropna(subset=[est]).reset_index(drop=True)
    if d.empty:
        return empty(title, "no finite estimates")
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.32 * len(d) + 1.0)))
    cmap = color_for(d[group].astype(str).unique()) if group else {}
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        c = cmap.get(str(r[group]), SERIES[0]) if group else SERIES[0]
        if np.isfinite(r.get(lo, np.nan)) and np.isfinite(r.get(hi, np.nan)):
            ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot([r[est]], [yi], "o", ms=6, color=c, mec=SURFACE, mew=1.5)
    ax.axvline(ref, color=AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label].astype(str), color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    if group:
        for k, c in cmap.items():
            ax.plot([], [], "o", color=c, label=k)
        ax.legend(loc="lower right", fontsize=8)
    return fig


# ------------------------------------------------------------------------------------------- heatmap
def heatmap(mat: pd.DataFrame, title: str, diverging: bool = False, annotate: bool = False, fmt: str = "{:.2f}",
            xlabel: str = "", ylabel: str = "", center: float = 0.0):
    plt = _plt()
    if mat is None or mat.empty:
        return empty(title, "no data")
    M = mat.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(min(14, 1.2 + 0.42 * M.shape[1]), min(12, 1.2 + 0.36 * M.shape[0])))
    if diverging:
        m = np.nanmax(np.abs(M - center)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap=div_cmap(), vmin=center - m, vmax=center + m, aspect="auto")
    else:
        im = ax.imshow(M, cmap=seq_cmap(), aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([str(i) for i in mat.index], fontsize=7)
    ax.grid(False)
    if annotate and M.size <= 400:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


# --------------------------------------------------------------------------------------- bars / lines
def bars(df: pd.DataFrame, x: str, y: str, title: str, lo: str | None = None, hi: str | None = None,
         ylabel: str = "", ref: float | None = None, horizontal: bool = True):
    plt = _plt()
    d = df.dropna(subset=[y]).reset_index(drop=True)
    if d.empty:
        return empty(title, "no data")
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.3 * len(d) + 1)) if horizontal else (max(5, 0.45 * len(d) + 2), 3.8))
    pos = np.arange(len(d))
    err = None
    if lo and hi:
        err = np.vstack([(d[y] - d[lo]).clip(lower=0), (d[hi] - d[y]).clip(lower=0)])
    if horizontal:
        ax.barh(pos, d[y], color=SERIES[0], height=0.6, xerr=err, ecolor=INK2, error_kw={"lw": 1})
        ax.set_yticks(pos)
        ax.set_yticklabels(d[x].astype(str), color=INK2)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        if ref is not None:
            ax.axvline(ref, color=AXIS, lw=1)
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(pos, d[y], color=SERIES[0], width=0.6, yerr=err, ecolor=INK2, error_kw={"lw": 1})
        ax.set_xticks(pos)
        ax.set_xticklabels(d[x].astype(str), rotation=45, ha="right", color=INK2)
        ax.set_ylabel(ylabel)
        if ref is not None:
            ax.axhline(ref, color=AXIS, lw=1)
        ax.grid(axis="x", visible=False)
    ax.set_title(title)
    return fig


def lines(df: pd.DataFrame, x: str, y: str, group: str, title: str, lo: str | None = None, hi: str | None = None,
          xlabel: str = "", ylabel: str = "", ref: float | None = None, logx: bool = False):
    plt = _plt()
    d = df.dropna(subset=[y])
    if d.empty:
        return empty(title, "no data")
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = color_for(d[group].astype(str).unique())
    for g, dd in d.groupby(d[group].astype(str)):
        if g not in cmap:
            continue
        dd = dd.sort_values(x)
        ax.plot(dd[x], dd[y], "-o", color=cmap[g], lw=2, ms=4, label=g)
        if lo and hi:
            ax.fill_between(dd[x].astype(float), dd[lo].astype(float), dd[hi].astype(float), color=cmap[g], alpha=0.12, lw=0)
    if ref is not None:
        ax.axhline(ref, color=AXIS, lw=1)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


# --------------------------------------------------------------------------------- distributions
def distributions(values: dict[str, np.ndarray], title: str, xlabel: str, bins: int = 40):
    plt = _plt()
    vals = {k: np.asarray(v, float)[np.isfinite(np.asarray(v, float))] for k, v in values.items()}
    vals = {k: v for k, v in vals.items() if len(v)}
    if not vals:
        return empty(title, "no finite values")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    cmap = color_for(vals)
    lo = min(np.percentile(v, 1) for v in vals.values())
    hi = max(np.percentile(v, 99) for v in vals.values())
    edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
    for k in sorted(vals):
        if k in cmap:
            ax.hist(vals[k], bins=edges, density=True, histtype="step", lw=2, color=cmap[k], label=f"{k} (n={len(vals[k])})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def small_multiple_distributions(df: pd.DataFrame, value: str, panel: str, hue: str, title: str, bins: int = 30):
    plt = _plt()
    d = df.dropna(subset=[value])
    panels = sorted(d[panel].astype(str).unique())[:12]
    if not panels:
        return empty(title, "no data")
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)
    cmap = color_for(d[hue].astype(str).unique())
    for ax, p in zip(axes.ravel(), panels):
        dd = d[d[panel].astype(str) == p]
        for h, v in dd.groupby(dd[hue].astype(str))[value]:
            if h in cmap:
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=cmap[h], label=h)
        ax.set_title(p, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, title: str, hue: str | None = None, xlabel: str = "", ylabel: str = "",
            trend: bool = True, max_points: int = 6000, seed: int = 0):
    plt = _plt()
    d = df.dropna(subset=[x, y])
    if d.empty:
        return empty(title, "no data")
    if len(d) > max_points:
        d = d.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    if hue:
        keys = d[hue].astype(str).unique()
        cmap = color_for(keys[:3]) if len(keys) > 3 else color_for(keys)   # scatter: <= 3 hues (all-pairs safe)
        for k, dd in d.groupby(d[hue].astype(str)):
            ax.scatter(dd[x], dd[y], s=10, alpha=0.5, color=cmap.get(k, MUTED), label=k if k in cmap else None, lw=0)
        ax.legend(fontsize=8)
    else:
        ax.scatter(d[x], d[y], s=10, alpha=0.5, color=SERIES[0], lw=0)
    if trend and len(d) > 3 and d[x].std() > 0:
        b = np.polyfit(d[x].astype(float), d[y].astype(float), 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xs, b[0] * xs + b[1], color=INK2, lw=1.5)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    return fig


def binned_profile(df: pd.DataFrame, x: str, y: str, group: str, title: str, bins: int = 8, xlabel: str = "",
                   ylabel: str = ""):
    """Mean of y (± 95% normal CI) in quantile bins of x, per group (e.g. confidence vs item difficulty)."""
    d = df.dropna(subset=[x, y]).copy()
    if d.empty:
        return empty(title, "no data"), pd.DataFrame()
    d["bin"] = pd.qcut(d[x].rank(method="first"), bins, labels=False)
    t = d.groupby([group, "bin"]).agg(x=(x, "mean"), m=(y, "mean"), s=(y, "std"), n=(y, "size")).reset_index()
    t["lo"] = t["m"] - 1.96 * t["s"] / np.sqrt(t["n"].clip(lower=1))
    t["hi"] = t["m"] + 1.96 * t["s"] / np.sqrt(t["n"].clip(lower=1))
    return lines(t, "x", "m", group, title, "lo", "hi", xlabel or x, ylabel or y), t


def alignment_panel(examples: list[dict], title: str):
    """Expected vs recognised phones per example: one row per example, M/S/D/I coloured tokens with text labels
    (identity never colour-only: the op letter is printed under each token)."""
    plt = _plt()
    if not examples:
        return empty(title, "no alignment examples")
    colors = {"M": SERIES[2], "S": SERIES[1], "D": SERIES[7], "I": SERIES[6]}
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(examples) + 1))
    ax.axis("off")
    for r, ex in enumerate(examples):
        yb = len(examples) - r
        ax.text(-0.5, yb + 0.25, ex.get("label", ""), fontsize=7, color=INK2, ha="right")
        for k, (op, ref, hyp) in enumerate(ex["ops"][:40]):
            c = colors.get(op, MUTED)
            ax.add_patch(plt.Rectangle((k, yb - 0.3), 0.92, 0.6, color=c, alpha=0.25, lw=0))
            ax.text(k + 0.46, yb + 0.12, ref or "∅", ha="center", fontsize=6.5)
            ax.text(k + 0.46, yb - 0.12, hyp or "∅", ha="center", fontsize=6.5, color=INK2)
            ax.text(k + 0.46, yb - 0.42, op, ha="center", fontsize=5.5, color=MUTED)
    ax.set_xlim(-6, 41)
    ax.set_ylim(0.3, len(examples) + 0.9)
    for op, c in colors.items():
        ax.plot([], [], "s", color=c, alpha=0.5, label={"M": "match", "S": "substitution", "D": "deletion", "I": "insertion"}[op])
    ax.legend(loc="upper right", ncol=4, fontsize=7)
    ax.set_title(title + "  (top: expected, bottom: recognised)")
    return fig


def throughput(df: pd.DataFrame, title: str):
    """Clips/s and peak VRAM per model: two panels, one axis each (never a dual axis)."""
    plt = _plt()
    if df is None or df.empty:
        return empty(title, "no inference runs")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    d = df.sort_values("model_key")
    pos = np.arange(len(d))
    axes[0].barh(pos, d["clips_per_s"], color=SERIES[0], height=0.6)
    axes[0].set_yticks(pos)
    axes[0].set_yticklabels(d["model_key"])
    axes[0].set_xlabel("clips / s")
    axes[0].set_title("throughput")
    axes[1].barh(pos, d["peak_vram_gb"], color=SERIES[0], height=0.6)
    axes[1].set_yticks(pos)
    axes[1].set_yticklabels([""] * len(d))
    axes[1].set_xlabel("peak reserved VRAM (GB)")
    axes[1].axvline(8.0, color=AXIS, lw=1)
    axes[1].set_title("memory (RTX 3070 = 8 GB)")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------- categorical status grid
STATUS_COLORS = {"PASS": "#1baf7a", "WARN": "#eda100", "PENDING": "#f3c26b", "NOT_PROBED": "#d9d8d2", "FAIL": "#e34948",
                 "NOT_APPLICABLE": "#f0efec"}
STATUS_ORDER = ("PASS", "WARN", "PENDING", "NOT_PROBED", "FAIL", "NOT_APPLICABLE")
# lifecycle stages in pipeline order; colour = how far the model got (review -> files -> runtime -> done)
LIFECYCLE_ORDER = ("INELIGIBLE", "LEGAL_REVIEW_PENDING", "PROVENANCE_REVIEW_PENDING", "READY_FOR_DOWNLOAD", "DOWNLOADING",
                   "CHECKPOINT_INCOMPLETE", "FRAMEWORK_INCOMPATIBLE", "FAILED", "READY_FOR_INFERENCE", "INFERENCE_RUNNING",
                   "INFERENCE_COMPLETE")
LIFECYCLE_COLORS = {"INELIGIBLE": "#e34948", "LEGAL_REVIEW_PENDING": "#f3c26b", "PROVENANCE_REVIEW_PENDING": "#f3c26b",
                    "READY_FOR_DOWNLOAD": "#9ec5f4", "DOWNLOADING": "#6da7ec", "CHECKPOINT_INCOMPLETE": "#eda100",
                    "FRAMEWORK_INCOMPATIBLE": "#eb6834", "FAILED": "#e34948", "READY_FOR_INFERENCE": "#3987e5",
                    "INFERENCE_RUNNING": "#256abf", "INFERENCE_COMPLETE": "#1baf7a"}


def lifecycle(df: pd.DataFrame, title: str):
    """One row per model: lifecycle stage reached (x position = pipeline order), labelled in text so the chart reads
    without colour. df needs model_key + lifecycle (+ optional priority)."""
    plt = _plt()
    if df is None or df.empty:
        return empty(title, "no models")
    d = df.copy()
    d["pos"] = d["lifecycle"].map({s: i for i, s in enumerate(LIFECYCLE_ORDER)}).fillna(0).astype(int)
    d = d.sort_values([c for c in ("priority", "model_key") if c in d]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.45 * len(d)))
    for i, r in d.iterrows():
        ax.barh(i, r["pos"] + 1, left=-0.5, color=LIFECYCLE_COLORS.get(r["lifecycle"], MUTED), height=0.6)
        ax.text(r["pos"] + 0.6, i, r["lifecycle"], va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["model_key"], fontsize=8)
    ax.set_xticks(range(len(LIFECYCLE_ORDER)))
    ax.set_xticklabels([s.replace("_", " ").lower() for s in LIFECYCLE_ORDER], rotation=45, ha="right", fontsize=7)
    ax.set_xlim(-0.5, len(LIFECYCLE_ORDER) + 3.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def status_matrix(mat: pd.DataFrame, title: str):
    """models x checks grid of status strings (PASS / WARN / PENDING / NOT_PROBED / FAIL). Each status is shown
    as a coloured cell AND a short text label, so the grid can be read without colour."""
    plt = _plt()
    if mat is None or mat.empty:
        return empty(title, "no models")
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    idx = {s: i for i, s in enumerate(STATUS_ORDER)}
    M = mat.apply(lambda c: c.map(lambda v: idx.get(str(v), idx["NOT_PROBED"]))).to_numpy(dtype=float)
    cmap = ListedColormap([STATUS_COLORS[s] for s in STATUS_ORDER])
    fig, ax = plt.subplots(figsize=(min(16, 2.5 + 0.55 * M.shape[1]), 1.6 + 0.5 * M.shape[0]))
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=len(STATUS_ORDER) - 0.5, aspect="auto")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([str(i) for i in mat.index], fontsize=8)
    ax.grid(False)
    short = {"PASS": "ok", "WARN": "warn", "PENDING": "rev", "NOT_PROBED": "–", "FAIL": "X", "NOT_APPLICABLE": "n/a"}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, short[STATUS_ORDER[int(M[i, j])]], ha="center", va="center", fontsize=6, color=INK)
    ax.legend(handles=[Patch(color=STATUS_COLORS[s], label=s) for s in STATUS_ORDER], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    return fig
