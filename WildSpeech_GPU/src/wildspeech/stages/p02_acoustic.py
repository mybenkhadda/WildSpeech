"""P02 — GPU acoustic extraction (resumable, per-family caches) + problem-oriented acoustic EDA (train only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import viz
from ..acoustic.extractor import AcousticExtractionRunner, active_families
from ..acoustic.store import FeatureStore
from ..analysis import acoustic_eda as A
from ..features.assemble import load_clip_features
from ._common import save_fig

S = "P02"


def run(ctx) -> dict:
    runner = AcousticExtractionRunner(ctx)
    fams = active_families(ctx.cfg)
    summaries = [runner.run("train", fams)]
    if ctx.cfg.get_path("audio.extract_test", False):
        summaries.append(runner.run("test", fams))
    ctx.save_table(pd.DataFrame(summaries).astype({"families": str}), S, "extraction_summary")
    store = FeatureStore(ctx.paths.cache / "features")
    fail = []
    for f in fams:
        p = store.dir(f, "train") / "_failures.parquet"
        if p.exists():
            fail.append(pd.read_parquet(p).assign(family=f))
    if fail:
        ctx.save_table(pd.concat(fail, ignore_index=True), S, "extraction_failures")
    ctx.save_table(store.status(), S, "feature_cache_status")

    # ------------------------------------------------------------------ EDA (train only)
    b = ctx.bundle()
    feats = load_clip_features(ctx, fams, splits=("train",))
    df = A.merge_clip_table(b.train_clips, b.train_students, feats, b.labels)
    cols = A.feature_columns(df)
    if not cols:
        return {"extracted": summaries, "eda": "no features"}
    ctx.save_table(A.feature_summary(df, cols), S, "clip_feature_summary")
    ge = A.group_effects(df, cols)
    ctx.save_table(ge, S, "clip_feature_group_effects")
    if not ge.empty:
        piv = ge.pivot_table(index="feature", columns="factor", values="eta2")
        top = piv.max(axis=1).sort_values(ascending=False).index[: int(ctx.cfg.get_path("analysis.heatmap_max_features", 60))]
        save_fig(ctx, S, "eta2_by_factor", viz.heatmap(piv.loc[top], "Variance explained (eta²) by task / instrument / rate / school",
                                                     diverging=False), piv.loc[top].reset_index())
    con = A.correctness_contrasts(df, cols)
    ctx.save_table(con, S, "correctness_contrasts_item_centred")
    alpha = float(ctx.cfg.get_path("validation.fdr_alpha", 0.05))
    if not con.empty:
        top = con.head(30).assign(label=lambda d: d["task"] + " · " + d["feature"], lo=lambda d: d["cliffs_delta"], hi=lambda d: d["cliffs_delta"])
        save_fig(ctx, S, "top_correctness_contrasts", viz.bar_ci(top, "label", "cliffs_delta", None, None,
                                                                 "Correct vs incorrect (same item): Cliff's delta, top 30 by p"), top)
        n_sig = int((con["q"] < alpha).sum())
        best = con.iloc[0]
        ctx.registries.update_hypothesis("H04", "supported" if n_sig > 0 else "not_supported", S,
                                         f"{n_sig} item-centred contrasts with q<{alpha}; strongest {best['task']}:{best['feature']}",
                                         effect=float(best["cliffs_delta"]), q_value=float(best["q"]))
    sa = A.student_level_associations(df, cols)
    ctx.save_table(sa, S, "student_task_mean_associations")
    bl = A.bandlimit_by_rate(df)
    if not bl.empty:
        ctx.save_table(bl, S, "bandlimit_by_original_rate")
        save_fig(ctx, S, "bandlimit_by_rate", viz.box_by_group(df.assign(original_sample_rate=df["original_sample_rate"].astype(str)),
                                                               "QUALITY::bandlimit_hz", "original_sample_rate",
                                                               "Effective bandwidth by original sample rate (artifact screen)"))
    if "QUALITY::snr_db" in df:
        save_fig(ctx, S, "snr_by_task", viz.hist_by_group(df, "QUALITY::snr_db", "task", "Estimated SNR by task"))
    # feature correlation structure (most variable features)
    var_top = df[cols].std().div(df[cols].mean().abs() + 1e-9).sort_values(ascending=False).index[:50]
    corr = df[var_top].corr(method="spearman")
    save_fig(ctx, S, "feature_correlation", viz.heatmap(corr, "Clip feature Spearman correlations (top-50 CV features)"), corr.reset_index())
    return {"families": fams, "n_clip_features": len(cols), "n_sig_contrasts": int((con["q"] < alpha).sum()) if not con.empty else 0,
            "train_processed": summaries[0].get("processed"), "clips_per_s": summaries[0].get("clips_per_s", np.nan)}
