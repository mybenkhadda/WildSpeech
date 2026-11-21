"""Atomic file IO, stable hashing and file fingerprints."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic hash of any JSON-like object (dict keys sorted)."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default, ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: Path, head_bytes: int = 1 << 20) -> dict:
    """Cheap fingerprint: size, mtime and hash of the first MiB (never reads the full TAR)."""
    path = Path(path)
    st = path.stat()
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(head_bytes))
    return {"name": path.name, "size": int(st.st_size), "mtime": int(st.st_mtime), "head_sha1": h.hexdigest()[:16]}


def _atomic_target(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> Path:
    return atomic_write_text(Path(path), json.dumps(obj, indent=2, sort_keys=False, default=_json_default))


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=index)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


# A DataFrame without columns used to be written as a header-less CSV ("\n"), which pandas cannot parse
# (EmptyDataError: No columns to parse from file). Such frames are now written with this single marker column, and
# every reader below turns a marker file, a zero-byte file or a header-less file into an empty frame flagged
# NO_RESULTS (df.attrs["status"]). An empty result is therefore always explicit, never a crash.
NO_RESULTS = "NO_RESULTS"
NO_RESULTS_COLUMN = "_no_results"


def atomic_write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    if df is None or (len(df.columns) == 0 and not index):
        df = pd.DataFrame(columns=[NO_RESULTS_COLUMN])
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        df.to_csv(tmp, index=index, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def no_results_frame(reason: str = "empty file") -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["status"] = NO_RESULTS
    df.attrs["reason"] = reason
    return df


def is_no_results(df: pd.DataFrame | None) -> bool:
    return df is not None and (df.attrs.get("status") == NO_RESULTS or df.empty)


def read_csv_safe(path: Path, **kw) -> pd.DataFrame:
    """pd.read_csv that never raises on an empty result file: zero-byte, header-less or marker-only files return an
    empty DataFrame with attrs status=NO_RESULTS. Genuine parse errors of non-empty files still raise."""
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return no_results_frame("zero-byte file")
    except OSError:
        return no_results_frame("unreadable file")
    try:
        df = pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return no_results_frame("no columns (empty result written by an earlier version)")
    if NO_RESULTS_COLUMN in df.columns:
        df = df.drop(columns=[NO_RESULTS_COLUMN])
        if df.empty:
            out = no_results_frame("stage wrote an empty table")
            return out
    if df.empty:
        df.attrs["status"] = NO_RESULTS
        df.attrs["reason"] = "header only (zero rows)"
    return df


def read_parquet_safe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if NO_RESULTS_COLUMN in df.columns:
        df = df.drop(columns=[NO_RESULTS_COLUMN])
    if df.empty:
        df.attrs["status"] = NO_RESULTS
        df.attrs["reason"] = "zero rows"
    return df


def read_table(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if path.with_suffix(".parquet").exists():
        return read_parquet_safe(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return read_csv_safe(path.with_suffix(".csv"))
    return None
