"""Plotting helpers, the figure registry and Section 26 (visual atlas).

Every figure is saved with a reproducible filename, a descriptive title, axis
labels, sample size and a description computed from actual results, and is
registered in ``figure_index.csv``. No figure is ever drawn from fabricated
values: helpers receive computed data only.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_text

FIGURE_COLUMNS = ["figure_id", "section", "title", "scientific_question", "filename", "source_table",
                  "sample_size", "description"]


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() in {"", "agg"} or not _in_notebook():
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


class FigureRegistry:
    def __init__(self, index_path: Path, dpi: int = 160, enabled: bool = True):
        self.index_path = Path(index_path)
        self.dpi = dpi
        self.enabled = enabled
        if self.index_path.exists():
            self.index = pd.read_csv(self.index_path)
        else:
            self.index = pd.DataFrame(columns=FIGURE_COLUMNS)

    def save(self, fig, section: str, fig_id: str, title: str, question: str, source_table: str,
             sample_size, description: str, directory: Path) -> Path | None:
        plt = _plt()
        if not self.enabled:
            plt.close(fig)
            return None
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in fig_id)
        filename = directory / f"S{section}_{safe_id}.png"
        if not fig._suptitle and title:
            fig.suptitle(title, fontsize=11)
        fig.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        row = {"figure_id": f"S{section}_{safe_id}", "section": section, "title": title,
               "scientific_question": question, "filename": str(filename), "source_table": source_table,
               "sample_size": sample_size, "description": description}
        self.index = self.index[self.index["figure_id"] != row["figure_id"]]
        self.index = pd.concat([self.index, pd.DataFrame([row])], ignore_index=True)
        atomic_write_csv(self.index, self.index_path)
        return filename


def new_figure(nrows: int = 1, ncols: int = 1, size=(7, 4.5)):
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    return fig, axes


def finalize_axis(ax, title: str, xlabel: str, ylabel: str, n=None) -> None:
    ax.set_title(title + (f" (n={n})" if n is not None else ""), fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def hist_kde(ax, values, label: str, unit: str = "", bins: int = 40) -> int:
    from scipy.stats import gaussian_kde

    x = np.asarray(values, dtype=float)
    n_missing = int((~np.isfinite(x)).sum())
    x = x[np.isfinite(x)]
    if len(x) == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", transform=ax.transAxes)
        return 0
    ax.hist(x, bins=bins, density=True, alpha=0.55, color="#4C72B0")
    if len(x) >= 10 and np.std(x) > 0:
        grid = np.linspace(x.min(), x.max(), 256)
        ax.plot(grid, gaussian_kde(x)(grid), color="#C44E52", lw=1.5, label="KDE")
        ax.legend(fontsize=8)
    finalize_axis(ax, f"{label} (missing={n_missing})", f"{label}{' [' + unit + ']' if unit else ''}", "density", len(x))
    return len(x)


def ecdf(ax, values, label: str, unit: str = "") -> None:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post")
    finalize_axis(ax, f"ECDF of {label}", f"{label}{' [' + unit + ']' if unit else ''}", "cumulative proportion", len(x))


def qq(ax, values, label: str) -> None:
    from scipy import stats

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return
    (osm, osr), (slope, intercept, r) = stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, s=6, alpha=0.6)
    ax.plot(osm, slope * osm + intercept, color="#C44E52")
    finalize_axis(ax, f"Normal Q-Q of {label} (r={r:.3f})", "theoretical quantile", "sample quantile", len(x))


def grouped_box(ax, df: pd.DataFrame, value: str, group: str, label: str, violin: bool = False,
                min_count: int = 3, max_groups: int = 30) -> None:
    sub = df[[value, group]].dropna()
    counts = sub[group].value_counts()
    keep = counts[counts >= min_count].index[:max_groups]
    sub = sub[sub[group].isin(keep)]
    order = sub.groupby(group)[value].median().sort_values().index.tolist()
    data = [sub.loc[sub[group] == g, value].to_numpy() for g in order]
    if not data:
        ax.text(0.5, 0.5, "insufficient groups", ha="center", transform=ax.transAxes)
        return
    if violin:
        ax.violinplot(data, showmedians=True)
    else:
        ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{g}\n(n={counts[g]})" for g in order], rotation=60, fontsize=7)
    finalize_axis(ax, f"{label} by {group}", group, label, len(sub))


def heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str = "RdBu_r", vmin=None, vmax=None,
            annotate: bool = False, cbar_label: str = "") -> None:
    data = matrix.to_numpy(dtype=float)
    if vmin is None and vmax is None and cmap == "RdBu_r":
        lim = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        vmin, vmax = -lim, lim
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([str(c)[:40] for c in matrix.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(r)[:40] for r in matrix.index], fontsize=6)
    if annotate and matrix.size <= 400:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    ax.set_title(title + " (grey = missing)", fontsize=10)


def scatter_fit(ax, x, y, xlabel: str, ylabel: str, title: str) -> dict:
    from .stats_utils import safe_corr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=8, alpha=0.5)
    r, p, n = safe_corr(x, y, "pearson", 3)
    if n >= 3 and np.std(x[mask]) > 0:
        coef = np.polyfit(x[mask], y[mask], 1)
        grid = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 50)
        ax.plot(grid, np.polyval(coef, grid), color="#C44E52")
    finalize_axis(ax, f"{title} (r={r:.3f})" if np.isfinite(r) else title, xlabel, ylabel, n)
    return {"pearson_r": r, "p": p, "n": n}


def forest(ax, df: pd.DataFrame, label_col: str, est_col: str, lo_col: str, hi_col: str, title: str,
           xlabel: str) -> None:
    d = df.dropna(subset=[est_col]).reset_index(drop=True)
    y = np.arange(len(d))
    err_lo = (d[est_col] - d[lo_col]).clip(lower=0).fillna(0)
    err_hi = (d[hi_col] - d[est_col]).clip(lower=0).fillna(0)
    ax.errorbar(d[est_col], y, xerr=[err_lo, err_hi], fmt="o", ms=3, capsize=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v)[:50] for v in d[label_col]], fontsize=6)
    finalize_axis(ax, title, xlabel, "", len(d))


def radar(ax, labels: list, series: dict, title: str) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for name, values in series.items():
        vals = list(np.nan_to_num(np.asarray(values, dtype=float))) + [np.nan_to_num(values[0])]
        ax.plot(angles, vals, lw=1.2, label=str(name))
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([str(l)[:22] for l in labels], fontsize=6)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.3, 1.1))


def binned_surface(df: pd.DataFrame, x: str, y: str, z: str, bins: int = 6, min_count: int = 10) -> pd.DataFrame:
    """Mean of z over quantile bins of x and y; cells with < min_count are NaN."""
    d = df[[x, y, z]].dropna()
    if len(d) < min_count * 2:
        return pd.DataFrame()
    xb = pd.qcut(d[x], bins, duplicates="drop")
    yb = pd.qcut(d[y], bins, duplicates="drop")
    grouped = d.groupby([yb, xb], observed=True)[z]
    mean = grouped.mean().unstack()
    count = grouped.count().unstack()
    mean = mean.where(count >= min_count)
    mean.index = [f"{iv.left:.2g}..{iv.right:.2g}" for iv in mean.index]
    mean.columns = [f"{iv.left:.2g}..{iv.right:.2g}" for iv in mean.columns]
    return mean


def top_correlation_heatmap(ctx, df: pd.DataFrame, cols: list, fig_id: str, title: str, question: str,
                            source: str) -> None:
    cols = [c for c in cols if c in df.columns][: int(ctx.config["max_features_for_heatmap"])]
    if len(cols) < 2:
        return
    corr = df[cols].corr(method="spearman", min_periods=20)
    fig, axes = new_figure(1, 1, (max(6, len(cols) * 0.18 + 3), max(5, len(cols) * 0.18 + 2)))
    heatmap(axes[0, 0], corr, f"Spearman correlation among {len(cols)} features", vmin=-1, vmax=1,
            cbar_label="Spearman rho")
    ctx.figure(fig, fig_id, title, question, source, len(df),
               f"Pairwise Spearman correlations (min 20 pairs) of {len(cols)} features; "
               f"median |rho|={np.nanmedian(np.abs(corr.to_numpy()[np.triu_indices(len(cols), 1)])):.3f}.")


def missingness_matrix(ctx, df: pd.DataFrame, fig_id: str, title: str, source: str, max_cols: int = 80) -> None:
    cols = list(df.columns)[:max_cols]
    if not cols:
        return
    miss = df[cols].isna().to_numpy()
    fig, axes = new_figure(1, 1, (max(6, len(cols) * 0.15 + 2), 5))
    ax = axes[0, 0]
    sample = miss if len(miss) <= 2000 else miss[np.linspace(0, len(miss) - 1, 2000).astype(int)]
    ax.imshow(sample, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[:30] for c in cols], rotation=90, fontsize=6)
    finalize_axis(ax, "Missingness (black = missing)", "column", "row (sampled)", len(df))
    rate = float(miss.mean())
    ctx.figure(fig, fig_id, title, "Where are values missing?", source, len(df),
               f"Overall missing rate {rate:.3%} across {len(cols)} columns.")


def stage_visualization_atlas(ctx) -> dict:
    """Section 26: collate every registered figure into a local HTML atlas."""
    reg = ctx.figures.index.copy()
    groups = {
        "Dataset atlas": ["01", "02"],
        "Literacy atlas": ["03", "04", "05"],
        "Acoustic atlas": ["06", "07", "08", "09", "10", "11"],
        "Scientific discovery atlas": ["12", "13", "15", "16", "17", "18", "19"],
        "Predictive atlas": ["14", "20", "21", "22", "23", "24", "25"],
    }
    # Dataset-level figures produced here from actual tables.
    sm = ctx.load_df("student_master", required=False)
    cm = ctx.load_df("clip_master", required=False)
    if sm is not None:
        missingness_matrix(ctx, sm, "dataset_student_missingness", "Student master missingness", "student_master")
        for col in ("school_id", "instrument"):
            if col in sm.columns and sm[col].notna().any():
                counts = sm.groupby(["split", col]).size().unstack(0).fillna(0)
                fig, axes = new_figure(1, 1, (max(6, len(counts) * 0.25 + 2), 4))
                counts.head(60).plot(kind="bar", ax=axes[0, 0])
                finalize_axis(axes[0, 0], f"Students per {col} and split", col, "students", int(len(sm)))
                ctx.figure(fig, f"dataset_{col}_distribution", f"Students per {col}",
                           f"How are students distributed across {col}?", "student_master", int(len(sm)),
                           f"{counts.shape[0]} distinct {col} values; largest has {int(counts.sum(1).max())} students.")
    if cm is not None and "task_family" in cm.columns:
        cov = cm.groupby(["split", "task_family"])["student_id"].nunique().unstack(0).fillna(0)
        fig, axes = new_figure(1, 1)
        cov.plot(kind="bar", ax=axes[0, 0])
        finalize_axis(axes[0, 0], "Students with at least one clip per task", "task family", "students", int(len(cm)))
        ctx.figure(fig, "dataset_task_coverage", "Student coverage per task", "Which tasks cover which students?",
                   "clip_master", int(cm["student_id"].nunique()), "Counts of distinct students per task family and split.")
        fig, axes = new_figure(1, 1, (8, 3.5))
        ax = axes[0, 0]
        levels = [("students", sm["student_id"].nunique() if sm is not None else np.nan),
                  ("tasks", cm["task_family"].nunique()), ("items", cm["item_key"].nunique()),
                  ("clips", len(cm))]
        from matplotlib.patches import FancyBboxPatch

        for i, (name, value) in enumerate(levels):
            ax.add_patch(FancyBboxPatch((i * 2.2, 0.3), 1.8, 1.0, boxstyle="round,pad=0.05",
                                        fc="#DDE6F2", ec="#4C72B0"))
            ax.text(i * 2.2 + 0.9, 0.8, f"{name}\n{value}", ha="center", va="center", fontsize=9)
            if i:
                ax.annotate("", xy=(i * 2.2, 0.8), xytext=(i * 2.2 - 0.4, 0.8), arrowprops={"arrowstyle": "->"})
        ax.set_xlim(-0.2, len(levels) * 2.2)
        ax.set_ylim(0, 1.6)
        ax.axis("off")
        ax.set_title("Hierarchical data structure: student > task > item > clip > acoustic frames", fontsize=10)
        ctx.figure(fig, "dataset_hierarchy", "Hierarchical data diagram", "How is the dataset nested?",
                   "clip_master", int(len(cm)), "Counts at each hierarchical level computed from the master tables.")
    reg = ctx.figures.index.copy()
    parts = ["<html><head><meta charset='utf-8'><title>WildSpeech P01 visual atlas</title>",
             "<style>body{font-family:sans-serif;max-width:1200px;margin:auto}img{max-width:100%;border:1px solid #ccc}"
             ".fig{margin:18px 0}</style></head><body><h1>WildSpeech P01 visual atlas</h1>",
             "<p>Local file. Contains derived figures only; do not publish recordings or identifiable material.</p>"]
    for group, sections in groups.items():
        sub = reg[reg["section"].astype(str).str.zfill(2).isin(sections)]
        parts.append(f"<h2>{html.escape(group)} ({len(sub)} figures)</h2>")
        for _, r in sub.iterrows():
            try:
                rel = Path(os.path.relpath(str(r["filename"]), str(ctx.out_dir()))).as_posix()
            except ValueError:  # different drive on Windows
                rel = Path(str(r["filename"])).as_posix()
            parts.append(f"<div class='fig'><h3>{html.escape(str(r['title']))}</h3>"
                         f"<p><b>Question:</b> {html.escape(str(r['scientific_question']))}<br>"
                         f"<b>n:</b> {html.escape(str(r['sample_size']))} &middot; <b>Source:</b> {html.escape(str(r['source_table']))}<br>"
                         f"{html.escape(str(r['description']))}</p><img src='{html.escape(str(rel))}'></div>")
    parts.append("</body></html>")
    out = ctx.out_dir() / "P01_VISUAL_ATLAS.html"
    atomic_write_text(out, "\n".join(parts))
    ctx.store.register("visual_atlas_html", out, ctx.current_stage, "html")
    ctx.save_df(reg, "figure_index_snapshot", fmt="csv")
    return {"outputs": ["visual_atlas_html", "figure_index_snapshot"], "summary": {"n_figures": int(len(reg))}}
