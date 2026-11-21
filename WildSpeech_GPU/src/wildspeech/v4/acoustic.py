"""Acoustic features for V4: reuse compatible caches, extract only what is missing, optional harmonized variants.

reuse
    cache/features/<family>/<split> under output_root/cache is the V4 store (FeatureStore, unchanged format).
    v4.acoustic.search_dirs may list other cache/features roots (e.g. a Drive copy of the local Windows cache).
    A family/split is imported from there ONLY if its _fingerprint.json equals the fingerprint the existing
    AcousticExtractionRunner computes for the current audio settings — incompatible caches are never mixed in.
extract
    AcousticExtractionRunner.run(split, families) is resumable and skips clips already cached, so calling it after
    the reuse step computes exactly the missing clips. TARs are staged to local disk first (config.stage_archive).
harmonize
    student-level features are item-conditioned and domain-controlled by the existing
    ItemConditionedResidualizer (controls item / domain = original sample rate + instrument / duration /
    complexity); v4.acoustic.variant selects a V2 bandwidth-harmonized variant cache (P16) when it exists.
QUALITY features describe recording conditions: they are the REC control, never ACOU speech features.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .. import FILE


def _runner(ctx):
    from ..acoustic.extractor import AcousticExtractionRunner

    key = "v4_acoustic_runner"
    if key not in ctx._cache:
        ctx._cache[key] = AcousticExtractionRunner(ctx)
    return ctx._cache[key]


def family_fingerprints(ctx, families: list[str]) -> dict[str, str]:
    r = _runner(ctx)
    out = {}
    for f in families:
        try:
            out[f] = r.family_fingerprint(f)
        except KeyError:
            out[f] = None
    return out


def import_cached_families(ctx, families: list[str], search_dirs: list[str], splits=("train", "test")) -> pd.DataFrame:
    from ..acoustic.store import FeatureStore
    from ..io_utils import read_json

    dst = FeatureStore(ctx.paths.cache / "features")
    fps = family_fingerprints(ctx, families)
    rows = []
    for root in search_dirs or []:
        root = Path(root)
        for fam in families:
            for s in splits:
                src = root / fam / s
                if not src.is_dir():
                    continue
                meta = read_json(src / "_fingerprint.json", {}) or {}
                parts = sorted(src.glob("part-*.parquet"))
                ok = bool(parts) and fps.get(fam) is not None and meta.get("fingerprint") == fps[fam]
                have = dst.parts(fam, s)
                action = "skipped_incompatible" if not ok else ("kept_existing" if have else "imported")
                if ok and not have:
                    d = dst.dir(fam, s)
                    for p in parts + [src / "_fingerprint.json"] + list(src.glob("_failures.parquet")):
                        shutil.copy2(p, d / p.name)
                rows.append({"search_dir": str(root), "family": fam, "split": s, "parts": len(parts),
                             "source_fingerprint": meta.get("fingerprint"), "expected_fingerprint": fps.get(fam),
                             "action": action})
    return pd.DataFrame(rows)


def inventory(ctx, families: list[str], splits=("train", "test")) -> pd.DataFrame:
    from ..acoustic.store import FeatureStore

    store = FeatureStore(ctx.paths.cache / "features")
    fps = family_fingerprints(ctx, families)
    b = ctx.bundle()
    rows = []
    for fam in families:
        for s in splits:
            clips = set((b.train_clips if s == "train" else b.test_clips)[FILE].astype(str))
            done = store.done(fam, s) if store.parts(fam, s) else set()
            cur = (store.fingerprint(fam, s) or {}).get("fingerprint")
            rows.append({"family": fam, "split": s, "n_metadata_clips": len(clips), "n_cached": len(done & clips),
                         "coverage": len(done & clips) / max(len(clips), 1), "n_failed": len(store.failed(fam, s)),
                         "fingerprint": cur, "expected_fingerprint": fps.get(fam),
                         "compatible": (cur == fps.get(fam)) if cur else None, "n_missing": len(clips - done)})
    return pd.DataFrame(rows)


def extract_missing(ctx, families: list[str], splits, local_root: Path | None = None) -> list[dict]:
    """Stage the TARs locally, then let the resumable extractor compute only the missing clips."""
    from .config import stage_archive

    runner = _runner(ctx)
    out = []
    for s in splits:
        key = f"{s}_audio"
        if key not in ctx.paths.files:
            out.append({"split": s, "status": "NOT_EXECUTED", "reason": f"{s}_audio.tar not found under data_root"})
            continue
        if local_root is not None:
            ctx.paths.files[key] = stage_archive(ctx.paths.files[key], Path(local_root), ctx.logger)
        res = runner.run(s, list(families))
        out.append({"split": s, "status": "ok", **{k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                                    for k, v in (res or {}).items()}})
    return out


def load_acoustic_clip_table(ctx, families: list[str], variant: str | None = None, splits=("train", "test")) -> pd.DataFrame:
    if variant:
        from ..harmonization.bandwidth import load_variant_features

        t = load_variant_features(ctx, variant, list(families), splits)
        if not t.empty:
            t.attrs["source"] = f"harmonized_variant:{variant}"
            return t
    from ..features.assemble import load_clip_features

    t = load_clip_features(ctx, list(families), splits)
    t.attrs["source"] = "original"
    return t


def acoustic_presence(ctx, families: list[str]) -> pd.DataFrame:
    from ..acoustic.store import FeatureStore

    store = FeatureStore(ctx.paths.cache / "features")
    out = None
    for fam in families:
        done = set()
        for s in ("train", "test"):
            if store.parts(fam, s):
                done |= store.done(fam, s)
        t = pd.DataFrame({FILE: sorted(done), f"acoustic_present__{fam}": 1.0})
        out = t if out is None else out.merge(t, on=FILE, how="outer")
    if out is None:
        return pd.DataFrame(columns=[FILE])
    return out.fillna(0.0)


def file_sr_series(clip_table: pd.DataFrame) -> pd.Series | None:
    col = next((c for c in clip_table.columns if c.endswith("::file_sr")), None)
    if col is None:
        return None
    return clip_table.drop_duplicates(FILE).set_index(FILE)[col]


def speech_columns(clip_table: pd.DataFrame, core: dict[str, list[str]]) -> list[str]:
    """Configured core acoustic speech columns present in the clip table (QUALITY excluded unless listed)."""
    cols = []
    for fam, feats in (core or {}).items():
        for f in feats or []:
            c = f"{fam.upper()}::{f}"
            if c in clip_table.columns:
                cols.append(c)
    mf = [c for c in clip_table.columns if c.startswith("MFCC::") and c.endswith("_mean")][:6] if not (core or {}).get("MFCC") else []
    return cols + [c for c in mf if c not in cols]
