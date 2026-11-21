"""Atomic file writes, JSON serialisation helpers and the artifact store."""
from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    return obj


def _tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    return Path(tmp)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    text = json.dumps(_clean_json(json.loads(json.dumps(obj, default=json_default))), indent=2, sort_keys=False)
    return atomic_write_text(path, text)


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type object columns to strings so Parquet writes never fail."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            non_null = out[col].dropna()
            kinds = {type(v) for v in non_null.head(1000)}
            if len(kinds) > 1 or (kinds and not kinds <= {str}):
                out[col] = out[col].map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


def atomic_write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        parquet_safe(df).to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    tmp = _tmp_path(path)
    try:
        df.to_csv(tmp, index=index)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_fingerprint(path: str | Path) -> dict:
    path = Path(path)
    st = path.stat()
    return {"path": str(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


class ArtifactMissing(FileNotFoundError):
    """Raised when a required artifact has not been produced yet."""


class ArtifactStore:
    """Registry of named artifacts (tables, JSON documents, figures).

    Every artifact has a globally unique name. The index is persisted so that
    later sessions (RESUME profile) can locate earlier outputs.
    """

    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.index: dict = read_json(self.index_path, default={}) or {}
        self._memory: dict = {}

    def _persist(self) -> None:
        atomic_write_json(self.index_path, self.index)

    def register(self, name: str, path: Path, stage: str, kind: str, extra: dict | None = None) -> None:
        self.index[name] = {
            "path": str(path),
            "stage": stage,
            "kind": kind,
            "written_at": _dt.datetime.now().isoformat(timespec="seconds"),
            **(extra or {}),
        }
        self._memory.pop(name, None)
        self._persist()

    def path(self, name: str) -> Path:
        if name not in self.index:
            raise ArtifactMissing(f"artifact {name!r} has not been produced")
        return Path(self.index[name]["path"])

    def has(self, name: str) -> bool:
        return name in self.index and Path(self.index[name]["path"]).exists()

    def stage_of(self, name: str) -> str | None:
        return self.index.get(name, {}).get("stage")

    def save_df(self, df: pd.DataFrame, name: str, directory: Path, stage: str, fmt: str = "parquet") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            path = atomic_write_parquet(df, directory / f"{name}.parquet")
        elif fmt == "csv":
            path = atomic_write_csv(df, directory / f"{name}.csv")
        else:
            raise ValueError(f"unsupported table format {fmt}")
        self.register(name, path, stage, f"table/{fmt}", {"rows": int(len(df)), "cols": int(df.shape[1])})
        return path

    def load_df(self, name: str, use_memory: bool = True) -> pd.DataFrame:
        if use_memory and name in self._memory:
            return self._memory[name].copy()
        path = self.path(name)
        if not path.exists():
            raise ArtifactMissing(f"artifact {name!r} is registered but {path} is missing")
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        if use_memory:
            self._memory[name] = df
        return df.copy()

    def save_json(self, obj: Any, name: str, directory: Path, stage: str) -> Path:
        path = atomic_write_json(directory / f"{name}.json", obj)
        self.register(name, path, stage, "json")
        return path

    def load_json(self, name: str) -> Any:
        path = self.path(name)
        if not path.exists():
            raise ArtifactMissing(f"artifact {name!r} is registered but {path} is missing")
        return read_json(path)

    def outputs_exist(self, names: list[str]) -> bool:
        return all(self.has(n) for n in names)

    def clear_memory(self) -> None:
        self._memory.clear()
