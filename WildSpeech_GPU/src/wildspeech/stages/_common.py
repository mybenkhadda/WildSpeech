"""Shared stage helpers: builders/runners, cross-fitted features, screening-only OOF residuals."""
from __future__ import annotations

import pandas as pd

from .. import ID
from ..features.assemble import builder_tag, make_builder
from ..validation.metrics import mean_oof
from ..validation.runner import CVRunner, ModelSpec

SCREENING_SPEC = ModelSpec("SCREENING_RIDGE", "RIDGE", ["SCREENING", "RECORDING"])


def get_runner(ctx, with_audio: bool = True, families: list[str] | None = None) -> CVRunner:
    key = ("runner", with_audio, tuple(families) if families else None)
    if key not in ctx._cache:
        tag = ("audio_" if with_audio else "tab_") + builder_tag(ctx, families, with_audio)
        ctx._cache[key] = CVRunner(ctx, make_builder(ctx, families, with_audio), tag=tag)
    return ctx._cache[key]


def load_crossfit(ctx) -> pd.DataFrame:
    X = ctx.read_stage_table("P05", "student_features_crossfit")
    if X is None:
        raise FileNotFoundError("P05 cross-fitted features missing — run P05")
    return X.set_index(ID) if ID in X.columns else X


def screening_residuals(ctx, refresh: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Repeated-CV screening-only OOF predictions -> mean residual per student (cached in P07 tables)."""
    cached = None if refresh else ctx.read_stage_table("P07", "screening_oof")
    if cached is None:
        runner = get_runner(ctx, with_audio=False)
        res = runner.evaluate(SCREENING_SPEC, "repeated_kfold", stage="P07")
        oof = res.oof
        ctx.save_parquet(oof, "P07", "screening_oof")
    else:
        oof = cached
    m = mean_oof(oof)
    m.index = m.index.astype(str)
    return (m["y_true"] - m["y_pred"]).rename("resid"), oof


def student_meta(ctx) -> pd.DataFrame:
    b = ctx.bundle()
    cols = [ID] + [c for c in ("instrument", "original_sample_rate", "school_id") if c in b.train_students]
    return b.train_students[cols].assign(**{ID: lambda d: d[ID].astype(str)})


def save_fig(ctx, stage: str, name: str, fig, table: pd.DataFrame | None = None) -> None:
    """Every figure is paired with the CSV of the data it shows."""
    ctx.save_figure(fig, stage, name)
    if table is not None:
        ctx.save_table(table, stage, f"fig_{name}")
