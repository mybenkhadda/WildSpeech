"""P27 — ASR disagreement and model consensus.

Pairwise clip-level disagreement between every two recognisers with outputs, consensus + high-uncertainty
indicator, task-level summaries, model x model agreement matrix, variance decomposition (item / recording /
correctness / student) and split-half consistency of student-level disagreement (V3H06). Disagreement is
described, never relabelled as a child's pronunciation difficulty.
Writes cache/v3/disagreement/consensus.parquet for P28/P29.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE
from ..asr.model_registry import registry_for
from ..discovery.asr_disagreement import (DIS_COLS, consensus, pairwise_disagreement, split_half_consistency,
                                          variance_decomposition)
from ..discovery.item_conditioning import v3_clip_frame
from ..io_utils import atomic_write_parquet
from ..phonetics.builder import load_phonetic_tables
from ..phonetics.g2p import ExpectedPhones
from ..v3 import viz as V
from ..v3.common import begin, not_executed
from ..v3.registry import save_figure_v3

S = "P27"


def consensus_path(ctx):
    d = ctx.paths.v3_cache / "disagreement"
    d.mkdir(parents=True, exist_ok=True)
    return d / "consensus.parquet"


def run(ctx) -> dict:
    from ..asr.outputs import require_outputs

    reg, v3r = begin(ctx, S)
    dcfg = ctx.cfg.get_path("v3.disagreement", {}) or {}
    ok, _acc, why = require_outputs(ctx, S, "text", int(dcfg.get("min_models", 2)), ("train",))
    if not ok:
        return not_executed(ctx, S, why)
    wide, det = load_phonetic_tables(ctx, ("train",))
    specs, _ = registry_for(ctx)
    models = sorted(det["model_key"].unique()) if not det.empty else []
    if len(models) < int(dcfg.get("min_models", 2)):
        return not_executed(ctx, S, f"needs >= {dcfg.get('min_models', 2)} ASR models with outputs; have {models}")
    outputs = {k: specs[k].output for k in models if k in specs}
    child = {k for k in models if k in specs and specs[k].is_child_model}
    clipf = v3_clip_frame(ctx)
    train_files = set(clipf.loc[clipf["split"] == "train", FILE].astype(str))
    g2p = ExpectedPhones(ctx.cfg.get_path("v3.phonetics.g2p_backend", "auto"), bool(ctx.cfg.get_path("v3.phonetics.allow_approximate_g2p", False)))
    pairs = pairwise_disagreement(det, wide, outputs, child, g2p, train_files)
    ctx.save_table(pairs, S, "pairwise_disagreement", parquet=True)
    cons = consensus(pairs, wide, models, float(dcfg.get("high_uncertainty_quantile", 0.85)), train_files)
    atomic_write_parquet(cons, consensus_path(ctx))
    ctx.save_table(cons, S, "consensus", parquet=True)
    # ---- summaries
    pc = pairs.merge(clipf[[FILE, "task", "item_score"]], on=FILE, how="left")
    task_t = pc.groupby(["pair", "task"])[[c for c in DIS_COLS if c in pc]].mean().reset_index()
    ctx.save_table(task_t, S, "disagreement_by_task")
    hm_col = next((c for c in ("dis_phones", "dis_transcript", "dis_word_vs_phone") if c in task_t and task_t[c].notna().any()), None)
    if hm_col:
        mat = task_t.pivot_table(index="pair", columns="task", values=hm_col)
        save_figure_v3(ctx, S, "disagreement_heatmap", V.heatmap(mat, f"Mean {hm_col} by model pair and task", annotate=True),
                       mat, "ASR disagreement heatmap")
    agree = pd.DataFrame(index=models, columns=models, dtype=float)
    for (a, b), g in pairs.groupby(["model_a", "model_b"]):
        v = g[[c for c in ("dis_phones", "dis_transcript", "dis_word_vs_phone") if c in g]].mean(1).mean()
        agree.loc[a, b] = agree.loc[b, a] = 1 - v
    for m in models:
        agree.loc[m, m] = 1.0
    save_figure_v3(ctx, S, "model_agreement_matrix", V.heatmap(agree, "Model x model agreement (1 − normalized edit distance)",
                   annotate=True), agree, "Model-consensus matrix")
    cc = cons.merge(clipf[[FILE, "task", "item_score"]], on=FILE, how="left")
    cc["official"] = cc["item_score"].map({1.0: "correct", 0.0: "incorrect"})
    save_figure_v3(ctx, S, "consensus_vs_disagreement", V.scatter(cc.dropna(subset=["official"]), "dis_max", "consensus_near_correct",
                   "Consensus near-correctness vs maximum model disagreement", hue="official",
                   xlabel="max pairwise disagreement", ylabel="consensus near-correctness", trend=False),
                   cc[[FILE, "task", "official", "dis_max", "consensus_near_correct", "high_uncertainty"]], "Model consensus")
    # ---- what explains disagreement
    vd = pd.concat([variance_decomposition(cons, clipf, t) for t in ("dis_mean", "dis_max") if t in cons], ignore_index=True)
    ctx.save_table(vd, S, "variance_decomposition")
    if not vd.empty:
        save_figure_v3(ctx, S, "disagreement_variance_decomposition", V.bars(vd[vd["target"] == "dis_mean"], "block", "r2_increment",
                       "Sequential R² of mean disagreement: item → recording → correctness → student", ylabel="ΔR²"),
                       vd, "What explains recogniser disagreement")
        v = vd[vd["target"] == "dis_mean"].set_index("block")["r2_increment"]
        share_ir = float(v.get("item", 0) + v.get("recording", 0))
        share_st = float(v.get("student", 0))
        v3r.update_hypothesis("V3H06", "supported" if share_ir > share_st else "not_supported", S,
                              f"item+recording ΔR²={share_ir:.3f}, student ΔR² (after item/recording/correctness)={share_st:.3f}",
                              tier="exploratory", effect=share_ir - share_st)
    sh = pd.DataFrame([split_half_consistency(cons, clipf, c) for c in ("dis_mean", "dis_max", "high_uncertainty") if c in cons])
    ctx.save_table(sh, S, "student_disagreement_consistency")
    hu = cc.groupby(["task", "official"])["high_uncertainty"].mean().reset_index() if "high_uncertainty" in cc else pd.DataFrame()
    ctx.save_table(hu, S, "high_uncertainty_by_task")
    return {"models": models, "n_pairs": int(pairs["pair"].nunique()) if not pairs.empty else 0,
            "threshold": cons.attrs.get("threshold", np.nan) if hasattr(cons, "attrs") else None}
