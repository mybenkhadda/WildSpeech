"""Centralised feature, hypothesis, experiment and failure registries (Parquet + CSV mirror)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_parquet


class Registry:
    """Keyed table with upsert semantics. Later records overwrite earlier ones column-wise."""

    def __init__(self, path: Path, key: list[str]):
        self.path = Path(path)
        self.key = list(key)
        self._lock = threading.Lock()
        self._df = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(columns=self.key)

    def upsert(self, records: Iterable[dict] | pd.DataFrame, save: bool = True) -> None:
        new = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        if new.empty:
            return
        missing = [k for k in self.key if k not in new.columns]
        if missing:
            raise KeyError(f"registry {self.path.name}: records lack key columns {missing}")
        new["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            old = self._df
            if old.empty:
                merged = new
            else:
                old_i = old.set_index(self.key)
                new_i = new.drop_duplicates(self.key, keep="last").set_index(self.key)
                for col in new_i.columns:
                    if col not in old_i.columns:
                        old_i[col] = np.nan
                both = old_i.index.intersection(new_i.index)
                if len(both):
                    upd = new_i.loc[both]
                    for col in upd.columns:
                        vals = upd[col]
                        mask = vals.notna()
                        if mask.any():
                            if old_i[col].dtype != object and vals.dtype == object:
                                old_i[col] = old_i[col].astype(object)
                            old_i.loc[vals.index[mask], col] = vals[mask]
                fresh = new_i.loc[new_i.index.difference(old_i.index)]
                merged = pd.concat([old_i, fresh], axis=0).reset_index()
            self._df = merged.reset_index(drop=True)
            if save:
                self.save()

    def frame(self) -> pd.DataFrame:
        return self._df.copy()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = self._df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str) or (isinstance(v, float) and np.isnan(v)) else str(v))
        atomic_write_parquet(df, self.path)
        atomic_write_csv(df, self.path.with_suffix(".csv"))


class Registries:
    def __init__(self, root: Path):
        root = Path(root) / "registries"
        root.mkdir(parents=True, exist_ok=True)
        self.features = Registry(root / "feature_registry.parquet", ["feature"])
        self.hypotheses = Registry(root / "hypothesis_registry.parquet", ["hypothesis_id"])
        self.experiments = Registry(root / "experiment_registry.parquet", ["experiment_id"])
        self.failures = Registry(root / "failure_registry.parquet", ["failure_id"])

    def seed_hypotheses(self, items: list[dict]) -> None:
        existing = set(self.hypotheses.frame().get("hypothesis_id", pd.Series(dtype=str)).astype(str))
        rows = [{"hypothesis_id": h["id"], "statement": h["statement"], "family": h.get("family"),
                 "tested_in": h.get("tested_in"), "rule": h.get("rule"), "status": "proposed"}
                for h in items if h["id"] not in existing]
        if rows:
            self.hypotheses.upsert(rows)

    def update_hypothesis(self, hid: str, status: str, stage: str, evidence: str, effect: float | None = None,
                          ci_low: float | None = None, ci_high: float | None = None, q_value: float | None = None) -> None:
        if status not in {"supported", "not_supported", "inconclusive", "proposed", "not_tested"}:
            raise ValueError(status)
        self.hypotheses.upsert([{"hypothesis_id": hid, "status": status, "evidence_stage": stage, "evidence": evidence,
                                 "effect": effect, "ci_low": ci_low, "ci_high": ci_high, "q_value": q_value}])

    def log_experiment(self, experiment_id: str, **fields) -> None:
        self.experiments.upsert([{"experiment_id": experiment_id, **fields}])
