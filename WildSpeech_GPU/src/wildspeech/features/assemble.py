"""Assemble clip-level feature tables from the family caches and construct feature builders."""
from __future__ import annotations

import pandas as pd

from .. import FILE
from ..acoustic.store import FeatureStore
from ..phonetic.text_features import item_text_table
from .builder import StudentFeatureBuilder

META_KEYS = ("true_duration_s", "decoded_duration_s", "cropped", "file_sr")


def cached_families(ctx) -> list[str]:
    root = ctx.paths.cache / "features"
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any((p / "train").glob("part-*.parquet")))


def load_clip_features(ctx, families: list[str] | None = None, splits=("train", "test")) -> pd.DataFrame:
    store = FeatureStore(ctx.paths.cache / "features")
    fams = families if families is not None else cached_families(ctx)
    out = None
    for fam in fams:
        parts = [store.load(fam, s) for s in splits]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
        if df.empty:
            continue
        df = df.drop_duplicates(FILE, keep="last")
        meta = [c for c in df.columns if c.split("::")[-1] in META_KEYS]
        if out is None:
            out = df.rename(columns={c: "META::" + c.split("::")[-1] for c in meta})
        else:
            out = out.merge(df.drop(columns=meta), on=FILE, how="outer")
    return out if out is not None else pd.DataFrame(columns=[FILE])


def builder_tag(ctx, audio_families: list[str] | None = None, with_audio: bool = True) -> str:
    """Cache tag for fold feature matrices: changes whenever the set or fingerprint of cached families changes."""
    from ..io_utils import stable_hash

    store = FeatureStore(ctx.paths.cache / "features")
    fams = (audio_families if audio_families is not None else cached_families(ctx)) if with_audio else []
    fps = {f: [(store.fingerprint(f, s) or {}).get("fingerprint") for s in ("train", "test")] +
           [len(store.parts(f, "train")), len(store.parts(f, "test"))] for f in fams}
    b = ctx.bundle()
    return stable_hash({"fams": fps, "n_train": len(b.train_students), "n_test": len(b.test_students)})


def make_builder(ctx, audio_families: list[str] | None = None, with_audio: bool = True) -> StudentFeatureBuilder:
    """audio_families=None -> every cached family; with_audio=False -> tabular-only builder."""
    b = ctx.bundle()
    key = ("clipfeat", tuple(audio_families) if audio_families else None, with_audio)
    if key not in ctx._cache:
        ctx._cache[key] = load_clip_features(ctx, audio_families) if with_audio else pd.DataFrame(columns=[FILE])
    if "text_table" not in ctx._cache:
        ctx._cache["text_table"] = item_text_table(b.all_clips, ctx.cfg.get_path("phonetic.g2p_backend", "auto"))
    return StudentFeatureBuilder(b.all_clips, b.all_students, ctx.cfg, b.theta_cols, ctx._cache[key],
                                 ctx._cache["text_table"], instruments=sorted(b.all_students["instrument"].dropna().unique())
                                 if "instrument" in b.all_students else None)
