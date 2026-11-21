"""Incremental Parquet checkpoints + provenance for ASR outputs (resumable).

cache/v3/asr/<model_key>/<split>/
    part-*.parquet          clip rows (FILE + ASR_COLUMNS + model_key / model_version), atomically written
    _failures.parquet       clips that failed (decode / model / OOM-after-CPU-fallback), skipped on resume
    _fingerprint.json       fingerprint of (model source + revision + decoding settings + code version)
    _provenance.json        eligibility record, model version info, dtype, parameter count, run history
cache/v3/asr_emb/<model_key>/<split>/part-*.parquet   pooled embeddings (FILE + e000..eNNN, float16)

A changed fingerprint invalidates ONLY that model's cache. Resume = skip clips present in parts or failures.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import FILE
from ..acoustic.store import FeatureStore
from ..io_utils import read_json, stable_hash, write_json

CODE_VERSION = "v3-asr-1.0.0"


def asr_fingerprint(model_key: str, source: str, revision: str | None, settings: dict) -> str:
    return stable_hash({"model": model_key, "source": str(source), "revision": revision, "settings": settings,
                        "code": CODE_VERSION})


class ASRCheckpointStore:
    def __init__(self, root: Path, model_key: str):
        self.root = Path(root)
        self.model_key = model_key
        self.store = FeatureStore(self.root / "asr")
        self.emb_store = FeatureStore(self.root / "asr_emb")

    # ------------------------------------------------------------------ identity
    def prepare(self, split: str, fp: str, provenance: dict, logger=None) -> str:
        state = self.store.validate(self.model_key, split, fp, True, logger)
        self.emb_store.validate(self.model_key, split, fp, True, logger)
        self.store.set_fingerprint(self.model_key, split, fp, {"model": self.model_key})
        self.emb_store.set_fingerprint(self.model_key, split, fp, {"model": self.model_key})
        p = self.provenance_path(split)
        prov = read_json(p, {}) or {}
        hist = list(prov.get("runs", []))
        hist.append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "fingerprint": fp, "cache_state": state})
        write_json(p, {**provenance, "fingerprint": fp, "code_version": CODE_VERSION, "runs": hist[-50:]})
        return state

    def provenance_path(self, split: str) -> Path:
        return self.store.dir(self.model_key, split) / "_provenance.json"

    def provenance(self, split: str) -> dict:
        return read_json(self.provenance_path(split), {}) or {}

    # ------------------------------------------------------------------ resume
    def done(self, split: str) -> set[str]:
        return self.store.done(self.model_key, split)

    def failed(self, split: str) -> set[str]:
        return self.store.failed(self.model_key, split)

    def todo(self, split: str, candidates: list[str], retry_failures: bool = False) -> list[str]:
        skip = self.done(split) | (set() if retry_failures else self.failed(split))
        if retry_failures:
            self.store.clear_failures(self.model_key, split)
        return sorted(set(map(str, candidates)) - skip)

    # ------------------------------------------------------------------ writes
    def append(self, split: str, rows: list[dict], model_version: str) -> str:
        if not rows:
            return "-"
        recs, embs = [], []
        for r in rows:
            e = r.pop("_embedding", None)
            recs.append({**r, "model_key": self.model_key, "model_version": model_version})
            if e is not None:
                embs.append((r[FILE], np.asarray(e, np.float32)))
        name = self.store.append(self.model_key, split, pd.DataFrame(recs))
        if embs:
            dim = max(len(v) for _, v in embs)
            mat = np.full((len(embs), dim), np.nan, np.float32)
            for i, (_, v) in enumerate(embs):
                mat[i, : len(v)] = v
            E = pd.DataFrame(mat.astype(np.float16), columns=[f"e{j:04d}" for j in range(dim)])
            E.insert(0, FILE, [f for f, _ in embs])
            self.emb_store.append(self.model_key, split, E)
        return name

    def add_failures(self, split: str, rows: list[dict]) -> None:
        self.store.add_failures(self.model_key, split, rows)

    # ------------------------------------------------------------------ reads
    def load(self, split: str) -> pd.DataFrame:
        return self.store.load(self.model_key, split, prefix=False)

    def load_embeddings(self, split: str) -> pd.DataFrame:
        return self.emb_store.load(self.model_key, split, prefix=False)

    def failures(self, split: str) -> pd.DataFrame:
        p = self.store.dir(self.model_key, split) / "_failures.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=[FILE, "stage", "error"])


def load_all_asr(root: Path, splits=("train",), models: list[str] | None = None) -> pd.DataFrame:
    """Long table of every cached ASR output: FILE, model_key, ASR columns (one row per clip x model)."""
    base = Path(root) / "asr"
    keys = sorted(p.name for p in base.glob("*") if p.is_dir()) if base.exists() else []
    if models:
        keys = [k for k in keys if k in set(m.lower() for m in models)]
    parts = []
    for k in keys:
        st = ASRCheckpointStore(root, k)
        for s in splits:
            df = st.load(s)
            if not df.empty:
                parts.append(df.assign(model_key=k, split=s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[FILE, "model_key"])
