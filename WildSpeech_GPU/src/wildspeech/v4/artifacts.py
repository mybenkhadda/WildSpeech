"""V4 artefact store: output tree, atomic writes, schema versions, fingerprints and resume compatibility.

Every table is written atomically (io_utils) together with a sidecar `<name>.meta.json`:
    {"schema_version", "fingerprint", "inputs", "rows", "columns", "created", "code_version"}
A later run with the same fingerprint and schema version reuses the artefact (resume); a different fingerprint
recomputes it and archives the previous meta under _history/ (artefacts are overwritten only by a newer result of
the same step, never deleted). Existing V1-V3 outputs are never touched: V4 writes only under output_root.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..io_utils import (atomic_write_csv, atomic_write_parquet, atomic_write_text, read_json, stable_hash,
                        write_json)
from . import SCHEMA_VERSION
from .config import OUTPUT_DIRS


class ArtifactStore:
    def __init__(self, root: Path | str, code_version: str = "v4.0.0", resume: bool = True):
        self.root = Path(root)
        self.code_version, self.resume = code_version, bool(resume)
        for d in OUTPUT_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path(self, sub: str, name: str) -> Path:
        p = self.root / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def meta_path(self, sub: str, name: str) -> Path:
        return self.path(sub, f"{Path(name).stem}.meta.json")

    # ------------------------------------------------------------------ meta / resume
    def meta(self, sub: str, name: str) -> dict:
        return read_json(self.meta_path(sub, name), {}) or {}

    def is_fresh(self, sub: str, name: str, fingerprint: str) -> bool:
        if not self.resume:
            return False
        m = self.meta(sub, name)
        return bool(m) and m.get("fingerprint") == fingerprint and m.get("schema_version") == SCHEMA_VERSION \
            and self.path(sub, name).exists()

    def _write_meta(self, sub: str, name: str, fingerprint: str | None, inputs: dict | None, extra: dict) -> None:
        mp = self.meta_path(sub, name)
        old = read_json(mp, None)
        if old and old.get("fingerprint") != fingerprint:
            h = mp.parent / "_history"
            h.mkdir(parents=True, exist_ok=True)
            write_json(h / f"{mp.stem}.{time.strftime('%Y%m%dT%H%M%S')}.json", old)
        write_json(mp, {"schema_version": SCHEMA_VERSION, "code_version": self.code_version,
                        "fingerprint": fingerprint, "inputs": inputs or {},
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra})

    # ------------------------------------------------------------------ writers
    def save_table(self, df: pd.DataFrame, sub: str, name: str, fingerprint: str | None = None,
                   inputs: dict | None = None, csv: bool | None = None) -> Path:
        """name with extension: .parquet (optionally also .csv mirror) or .csv."""
        p = self.path(sub, name)
        df = df if df is not None else pd.DataFrame()
        if p.suffix == ".parquet":
            atomic_write_parquet(_parquet_safe(df), p)
            if csv:
                atomic_write_csv(df, p.with_suffix(".csv"))
        else:
            atomic_write_csv(df, p)
        self._write_meta(sub, name, fingerprint, inputs, {"rows": int(len(df)), "columns": [str(c) for c in df.columns][:400]})
        return p

    def save_json(self, obj: Any, sub: str, name: str, fingerprint: str | None = None) -> Path:
        p = write_json(self.path(sub, name), obj)
        self._write_meta(sub, name, fingerprint, None, {})
        return p

    def save_text(self, text: str, sub: str, name: str) -> Path:
        return atomic_write_text(self.path(sub, name), text)

    # ------------------------------------------------------------------ readers
    def load_table(self, sub: str, name: str) -> pd.DataFrame | None:
        p = self.path(sub, name)
        if not p.exists():
            return None
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    def load_json(self, sub: str, name: str, default=None):
        return read_json(self.path(sub, name), default)

    def inventory(self) -> pd.DataFrame:
        rows = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and "_history" not in p.parts and not p.name.endswith(".meta.json") and "cache" not in p.relative_to(self.root).parts[:1]:
                m = read_json(p.with_name(f"{p.stem}.meta.json"), {}) or {}
                rows.append({"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size,
                             "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)),
                             "schema_version": m.get("schema_version"), "fingerprint": m.get("fingerprint")})
        return pd.DataFrame(rows)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns with mixed types -> str (pyarrow refuses mixed object columns)."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            vals = out[c].dropna()
            if len(vals) and not vals.map(lambda v: isinstance(v, str)).all():
                out[c] = out[c].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    out.columns = [str(c) for c in out.columns]
    return out


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fingerprint(*parts: Any) -> str:
    return stable_hash(list(parts))


def env_report() -> dict:
    import platform
    import sys

    out = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "pyarrow", "torch", "lightgbm", "catboost", "xgboost", "pygam",
                "shap", "g2p_en", "nltk", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "installed")
        except Exception:  # noqa: BLE001 - optional package
            out[mod] = None
    try:
        import torch

        out["cuda"] = bool(torch.cuda.is_available())
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["cuda"] = False
    return out
