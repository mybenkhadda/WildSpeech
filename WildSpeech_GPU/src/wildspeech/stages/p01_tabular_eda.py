"""P01 — tabular EDA and screening baselines (no audio)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import ID, TARGET, viz
from ..analysis import tabular_eda as T
from ..phonetic.text_features import item_text_table
from ..validation.metrics import subgroup_metrics
from ..validation.runner import ModelSpec
from ._common import get_runner, save_fig, student_meta

S = "P01"


def run(ctx) -> dict:
    b = ctx.bundle()
    a = ctx.cfg.section("analysis")
    nb, seed = int(a.get("bootstrap", 1000)), ctx.seed
    train = b.train
    th = b.theta_cols

    ctx.save_table(T.target_summary(train), S, "target_summary")
    ctx.save_table(T.target_group_effects(train), S, "target_group_effects")
    save_fig(ctx, S, "target_by_instrument", viz.hist_by_group(train, TARGET, "instrument" if "instrument" in train else None,
                                                               "End-of-year outcome by instrument"), train[[ID, TARGET] + (["instrument"] if "instrument" in train else [])])
    if "school_id" in train:
        save_fig(ctx, S, "target_by_school", viz.box_by_group(train, TARGET, "school_id", "Outcome by school (train only)"))

    assoc = T.screening_associations(train, th, nb, seed)
    ctx.save_table(assoc, S, "screening_associations")
    for c in th:
        save_fig(ctx, S, f"scatter_{c}", viz.scatter_fit(train[c], train[TARGET], c, TARGET, f"{c} vs outcome",
                                                         hue=train["instrument"] if "instrument" in train else None))
    ctx.save_table(T.missingness_patterns(b.all_students, th), S, "theta_missingness_patterns")

    beh = T.clip_behaviour(b.train_clips)
    ctx.save_table(beh, S, "clip_behaviour_per_student_task", parquet=True)
    ctx.save_table(T.score_status_table(b.train_clips, train), S, "score_status_vs_outcome")
    text = item_text_table(b.all_clips, ctx.cfg.get_path("phonetic.g2p_backend", "auto"))
    items = T.item_table(b.train_clips, train, text)
    ctx.save_table(items, S, "item_table")
    if "n_syllables" in items:
        save_fig(ctx, S, "item_difficulty_vs_syllables", viz.scatter_fit(items["n_syllables"], items["p_correct"], "syllables",
                                                                         "p(correct)", "Item difficulty vs length", hue=items["task"]),
                 items[["item_id", "task", "n_syllables", "p_correct"]])
    save_fig(ctx, S, "item_exposure", viz.hist_by_group(items, "n_students", "task", "Item exposure (students per item)"))

    irt_items, irt_persons = T.irt_analysis(b.train_clips, train, th, ctx.cfg)
    ctx.save_table(irt_items, S, "irt_item_parameters")
    ctx.save_table(irt_persons, S, "irt_person_thetas", parquet=True)
    ctx.save_table(T.irt_vs_provided(irt_persons, th), S, "irt_vs_provided_thetas")
    adap = T.adaptive_behaviour(beh, train, th)
    ctx.save_table(adap, S, "adaptive_behaviour")

    # ---- tabular baselines (student-disjoint repeated CV; item statistics fitted in-fold)
    runner = get_runner(ctx, with_audio=False)
    specs = [ModelSpec("B00_MEAN", "MEAN", ["SCREENING"]),
             ModelSpec("B01_RIDGE_SCREENING", "RIDGE", ["SCREENING"]),
             ModelSpec("B02_RIDGE_SCREENING_RECORDING", "RIDGE", ["SCREENING", "RECORDING"]),
             ModelSpec("B03_RIDGE_SCR_BEH_ITEM", "RIDGE", ["SCREENING", "RECORDING", "BEHAVIOR", "ITEM"]),
             ModelSpec("B04_RIDGE_TABULAR_ALL", "RIDGE", ["SCREENING", "RECORDING", "BEHAVIOR", "ITEM", "LINGUISTIC"]),
             ModelSpec("B05_LGBM_TABULAR_ALL", "LIGHTGBM", ["SCREENING", "RECORDING", "BEHAVIOR", "ITEM", "LINGUISTIC"])]
    rows, best = [], None
    for sp in specs:
        res = runner.evaluate(sp, "repeated_kfold", stage=S)
        runner.save_oof(res, S)
        rows.append({"name": sp.name, "model": sp.model, "families": "+".join(sp.families), "status": res.status, **res.summary})
        if res.status == "ok" and (best is None or res.summary["r2_mean"] > best.summary["r2_mean"]):
            best = res
    base = pd.DataFrame(rows)
    ctx.save_table(base, S, "tabular_baselines")
    if best is not None:
        ctx.save_table(subgroup_metrics(best.oof, student_meta(ctx), [c for c in ("instrument", "original_sample_rate", "school_id")
                                                                     if c in b.train_students]), S, "best_baseline_subgroups")
        ok = base[base.status == "ok"]
        save_fig(ctx, S, "tabular_baselines_r2", viz.bar_ci(ok.assign(lo=ok.r2_mean - ok.r2_sd, hi=ok.r2_mean + ok.r2_sd),
                                                             "name", "r2_mean", "lo", "hi", "Tabular baselines (CV R2 ± sd over repeats)"), ok)
    return {"n_train": len(train), "best_baseline": best.spec.name if best else None,
            "best_r2": float(best.summary["r2_mean"]) if best else np.nan}
