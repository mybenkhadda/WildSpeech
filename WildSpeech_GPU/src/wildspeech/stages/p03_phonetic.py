"""P03 — phonetic intelligence: local ASR + CTC alignment confidence + expected-text deviation (opt-in)."""
from __future__ import annotations

import pandas as pd

from ..analysis import acoustic_eda as A
from ..features.assemble import load_clip_features
from ..model_registry import eligibility_table, load_registry
from ..phonetic.runner import PhoneticRunner
from ..phonetic.text_features import item_text_table

S = "P03"


def run(ctx) -> dict:
    ctx.save_table(eligibility_table(load_registry(ctx.paths.configs)), S, "model_eligibility")
    b = ctx.bundle()
    ctx.save_table(item_text_table(b.all_clips, ctx.cfg.get_path("phonetic.g2p_backend", "auto")), S, "item_text_complexity")
    runner = PhoneticRunner(ctx)
    out = [runner.run("train")]
    if ctx.cfg.get_path("audio.extract_test", False):
        out.append(runner.run("test"))
    ctx.save_json(out, S, "phonetic_run_summary")
    feats = load_clip_features(ctx, ["asr", "align"], splits=("train",))
    if feats.shape[1] <= 1:
        return {"status": "no phonetic features (models not eligible or not available locally)"}
    df = A.merge_clip_table(b.train_clips, b.train_students, feats, b.labels)
    cols = A.feature_columns(df)
    ctx.save_table(A.feature_summary(df, cols), S, "phonetic_feature_summary")
    con = A.correctness_contrasts(df, cols)
    ctx.save_table(con, S, "phonetic_correctness_contrasts")
    ctx.save_table(A.student_level_associations(df, cols), S, "phonetic_student_associations")
    if "ASR::asr_text" in feats:
        sample = df[["filename", "task", "expected_text", "item_score"]].merge(feats[["filename", "ASR::asr_text"]], on="filename").head(200)
        ctx.save_table(sample, S, "asr_examples_local_only")
    return {"n_features": len(cols), "families": sorted({c.split("::")[0] for c in cols})}
