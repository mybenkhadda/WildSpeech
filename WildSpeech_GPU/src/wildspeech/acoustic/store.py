"""Independent per-family feature caches: cache/features/<family>/<split>/part-*.parquet + _fingerprint.json.

* Each family is fingerprinted separately (params + extractor version), so changing e.g. pitch settings
  invalidates only the pitch cache.
* Parts are written atomically; an interrupted run resumes from the filenames already present.
* Failures are recorded per family in _failures.parquet and skipped on resume unless retried.
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import pandas as pd

from .. import FILE
from ..io_utils import atomic_write_parquet, read_json, write_json


class FeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def dir(self, family: str, split: str) -> Path:
        d = self.root / family.lower() / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fingerprint(self, family: str, split: str) -> dict | None:
        return read_json(self.dir(family, split) / "_fingerprint.json")

    def set_fingerprint(self, family: str, split: str, fp: str, params: dict) -> None:
        write_json(self.dir(family, split) / "_fingerprint.json", {"fingerprint": fp, "params": params,
                                                                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")})

    def parts(self, family: str, split: str) -> list[Path]:
        return sorted(Path(p) for p in glob.glob(str(self.dir(family, split) / "part-*.parquet")))

    def validate(self, family: str, split: str, fp: str, auto_invalidate: bool, logger=None) -> str:
        """Return 'empty' | 'valid' | 'invalidated'. Raises if stale and auto_invalidate is False."""
        cur = self.fingerprint(family, split)
        if not self.parts(family, split):
            self.set_fingerprint(family, split, fp, {})
            return "empty"
        if cur and cur.get("fingerprint") == fp:
            return "valid"
        if not auto_invalidate:
            raise RuntimeError(f"stale cache for {family}/{split}; set runtime.auto_invalidate_stale_cache or clear it")
        self.invalidate(family, split)
        self.set_fingerprint(family, split, fp, {})
        if logger:
            logger.warning("cache %s/%s invalidated (fingerprint changed)", family, split)
        return "invalidated"

    def invalidate(self, family: str, split: str) -> None:
        d = self.dir(family, split)
        for p in list(d.glob("part-*.parquet")) + list(d.glob("_failures*.parquet")) + list(d.glob("_fingerprint.json")):
            p.unlink(missing_ok=True)

    def done(self, family: str, split: str) -> set[str]:
        keys: set[str] = set()
        for p in self.parts(family, split):
            keys.update(pd.read_parquet(p, columns=[FILE])[FILE].astype(str))
        return keys

    def failed(self, family: str, split: str) -> set[str]:
        p = self.dir(family, split) / "_failures.parquet"
        return set(pd.read_parquet(p)[FILE].astype(str)) if p.exists() else set()

    def append(self, family: str, split: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "-"
        name = f"part-{time.strftime('%Y%m%d-%H%M%S')}-{len(self.parts(family, split)):05d}.parquet"
        atomic_write_parquet(df.reset_index(drop=True), self.dir(family, split) / name)
        return name

    def add_failures(self, family: str, split: str, rows: list[dict]) -> None:
        if not rows:
            return
        p = self.dir(family, split) / "_failures.parquet"
        new = pd.DataFrame(rows)
        if p.exists():
            new = pd.concat([pd.read_parquet(p), new], ignore_index=True).drop_duplicates(FILE, keep="last")
        atomic_write_parquet(new, p)

    def clear_failures(self, family: str, split: str) -> None:
        (self.dir(family, split) / "_failures.parquet").unlink(missing_ok=True)

    def load(self, family: str, split: str, prefix: bool = True) -> pd.DataFrame:
        parts = self.parts(family, split)
        if not parts:
            return pd.DataFrame(columns=[FILE])
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates(FILE, keep="last")
        if prefix:
            df = df.rename(columns={c: f"{family.upper()}::{c}" for c in df.columns if c != FILE})
        return df

    def status(self) -> pd.DataFrame:
        rows = []
        for fam_dir in sorted(p for p in self.root.glob("*") if p.is_dir()):
            for split_dir in sorted(p for p in fam_dir.glob("*") if p.is_dir()):
                parts = sorted(split_dir.glob("part-*.parquet"))
                fp = read_json(split_dir / "_fingerprint.json", {}) or {}
                n = sum(len(pd.read_parquet(p, columns=[FILE])) for p in parts)
                fail = split_dir / "_failures.parquet"
                rows.append({"family": fam_dir.name, "split": split_dir.name, "parts": len(parts), "rows": n,
                             "failures": len(pd.read_parquet(fail)) if fail.exists() else 0,
                             "fingerprint": fp.get("fingerprint"), "bytes": sum(p.stat().st_size for p in parts)})
        return pd.DataFrame(rows)
