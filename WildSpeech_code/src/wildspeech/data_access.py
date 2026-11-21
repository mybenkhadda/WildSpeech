"""Convenience loaders shared by stages (training-only by default)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def students(ctx, split: str = "train") -> pd.DataFrame:
    sm = ctx.load_df("student_master")
    return sm[sm["split"] == split].reset_index(drop=True) if split != "all" else sm


def target(ctx) -> pd.Series:
    sm = students(ctx, "train")
    y = pd.to_numeric(sm.set_index("student_id")["target"], errors="coerce")
    return y[np.isfinite(y)]


def clips(ctx, split: str = "train") -> pd.DataFrame:
    cm = ctx.load_df("clip_master")
    return cm[cm["split"] == split].reset_index(drop=True) if split != "all" else cm


def screening(ctx, split: str = "train") -> pd.DataFrame | None:
    name = "screening_features" if split == "train" else "screening_features_test"
    df = ctx.load_df(name, required=False)
    return None if df is None else df.set_index("student_id")


def theta_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("theta_") and not c.startswith("theta_se_")]


def raw_theta_columns(df: pd.DataFrame) -> list:
    return [c for c in ("theta_blending", "theta_deletion", "theta_repetition") if c in df.columns] + \
        sorted(c for c in df.columns if c.startswith("theta_other_"))


def clip_features(ctx, split: str = "train", required: bool = False) -> pd.DataFrame | None:
    name = "clip_features" if split == "train" else "clip_features_test"
    return ctx.load_df(name, required=required)


def item_difficulty(ctx) -> pd.DataFrame | None:
    """All-training IRT item parameters (EDA only; prediction refits within folds)."""
    if not ctx.stage_available("04"):
        return None
    items = ctx.load_df("irt_item_params", required=False)
    if items is None or items.empty:
        return None
    prim = items[items["model"] == items["selected_model"]] if "selected_model" in items.columns else items
    return prim[["item_key", "b", "a"]].drop_duplicates("item_key")


def item_linguistics(ctx) -> pd.DataFrame | None:
    if not ctx.stage_available("11"):
        return None
    return ctx.load_df("item_linguistic_features", required=False)


def clip_phonetics(ctx, split: str = "train") -> pd.DataFrame | None:
    if not ctx.stage_available("11"):
        return None
    name = "clip_phonetic_features" if split == "train" else "clip_phonetic_features_test"
    return ctx.load_df(name, required=False)


def clip_table(ctx, split: str = "train", with_audio: bool = True) -> pd.DataFrame:
    """Clip master joined with audio, item difficulty, linguistic and phonetic features."""
    cm = clips(ctx, split)
    if with_audio:
        cf = clip_features(ctx, split)
        if cf is not None:
            cm = cm.merge(cf, on="clip_id", how="left")
    diff = item_difficulty(ctx)
    if diff is not None:
        cm = cm.merge(diff.rename(columns={"b": "item_b", "a": "item_a"}), on="item_key", how="left")
    ling = item_linguistics(ctx)
    if ling is not None:
        cm = cm.merge(ling, on="item_key", how="left")
    ph = clip_phonetics(ctx, split)
    if ph is not None:
        cm = cm.merge(ph, on="clip_id", how="left")
    return cm


def residuals(ctx) -> pd.Series | None:
    if not ctx.stage_available("14"):
        return None
    res = ctx.load_df("baseline_residuals", required=False)
    if res is None:
        return None
    return res.set_index("student_id")["residual"]


def oof_baseline(ctx) -> pd.Series | None:
    if not ctx.stage_available("14"):
        return None
    res = ctx.load_df("baseline_residuals", required=False)
    return None if res is None else res.set_index("student_id")["oof_prediction"]


def student_features(ctx, split: str = "train") -> pd.DataFrame:
    name = "student_features" if split == "train" else "student_features_test"
    return ctx.load_df(name).set_index("student_id")


def student_meta(ctx, split: str = "train") -> pd.DataFrame:
    sm = students(ctx, split).set_index("student_id")
    cols = [c for c in ("school_id", "instrument", "target") if c in sm.columns]
    return sm[cols]


def numeric_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def task_families(ctx) -> list:
    cm = clips(ctx, "train")
    return sorted(cm["task_family"].dropna().unique().tolist()) if "task_family" in cm.columns else []
