"""Append-only preregistration and experiment ledger stored on private Drive.

No silent overwrites: comparison plans are immutable, run status is event-sourced
and all files are committed atomically. One writer at a time is required.
"""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .data import atomic_json, atomic_text, sha_json


PLANNED_COMPARISONS = [
    {"comparison_id":"PARITY_REF","baseline":None,"challenger":"REF_RIDGE14",
     "scope":"reference parity [0.188,0.198]"},
    {"comparison_id":"M73_VS_REF","baseline":"REF_RIDGE14","challenger":"M73_GLOBAL",
     "scope":"historical adapter missing until repository source restored"},
    {"comparison_id":"M83_ALL_VS_REF","baseline":"REF_RIDGE14","challenger":"M83_ALL",
     "scope":"exploratory legacy feature-bank parity, not promotion-eligible pending audit"},
    {"comparison_id":"M83_NO_AUDIO_VS_REF","baseline":"REF_RIDGE14",
     "challenger":"M83_WITHOUT_AUDIO_NAMING",
     "scope":"legacy taxonomy impure; exploratory-only"},
    {"comparison_id":"M83_ALL_VS_NO_AUDIO","baseline":"M83_WITHOUT_AUDIO_NAMING",
     "challenger":"M83_ALL","scope":"diagnostic of legacy naming, not causal audio ablation"},
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def preregister(root: Path, comparisons=None):
    root=Path(root)
    path=root/"ledger"/"planned_comparisons.json"
    comparisons=PLANNED_COMPARISONS if comparisons is None else comparisons
    ids=[p["comparison_id"] for p in comparisons]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate preregistered comparison ID")
    obj={"version":1,"comparisons":comparisons,"plan_hash":sha_json(comparisons)}
    if path.is_file():
        prior=json.loads(path.read_text(encoding="utf-8"))
        if prior["plan_hash"]!=obj["plan_hash"]:
            raise RuntimeError("PRE-REGISTERED COMPARISONS CHANGED: start a new experiment directory")
        return prior
    atomic_json(path,obj)
    return obj


class Ledger:
    def __init__(self,root):
        self.root=Path(root)
        self.path=self.root/"ledger"/"events.jsonl"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if not (self.root/"ledger"/"planned_comparisons.json").exists():
            raise RuntimeError("Create preregistered comparison plan BEFORE running models")

    def _events(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def latest(self,run_id):
        rows=[e for e in self._events() if e["run_id"]==run_id]
        return rows[-1] if rows else None

    def transition(self,run_id,status,**fields):
        current=self.latest(run_id)
        allowed={
            None:{"PLANNED","HISTORICAL_UNALIGNED","NOT_REPRODUCED"},
            "PLANNED":{"RUNNING","NOT_REPRODUCED","FAILED"},
            "RUNNING":{"COMPLETE","FAILED"},
            "COMPLETE":set(),"FAILED":set(),"HISTORICAL_UNALIGNED":set(),
            "NOT_REPRODUCED":set()
        }
        before=None if current is None else current["status"]
        if status not in allowed[before]:
            raise RuntimeError(f"Invalid immutable status transition {before}->{status} for {run_id}")
        event = dict(current or {})
        event.update(fields)
        event.update(run_id=run_id,status=status,event_utc=utc_now(),
                     prev_status=before)
        # Concurrent writers are forbidden; fail immediately instead of corrupting events.
        lock=self.path.with_suffix(".lock")
        try:
            fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as exc:
            raise RuntimeError("Ledger already locked; inspect stale lock and competing process") from exc
        try:
            with os.fdopen(fd,"w") as f:
                f.write(str(os.getpid()))
            # Recheck under lock to avoid interleaving invalid transitions.
            existing=[e for e in self._events() if e["run_id"]==run_id]
            latest=existing[-1]["status"] if existing else None
            if latest!=before:
                raise RuntimeError("Run state changed concurrently")
            with self.path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(event,sort_keys=True,default=str)+"\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            lock.unlink(missing_ok=True)
        self.snapshot()
        return event

    def snapshot(self):
        events=self._events()
        by_run={}
        for e in events:
            by_run[e["run_id"]]=e
        df=pd.DataFrame(list(by_run.values()))
        target=self.root/"ledger"/"experiments.parquet"
        temp=target.with_name("experiments.tmp.parquet")
        df.to_parquet(temp,index=False)
        os.replace(temp,target)
        return df


def make_run_id(model_id,git_sha,feature_hash,fold_hash,config_hash):
    return model_id.lower()+"__"+sha_json([model_id,git_sha,feature_hash,fold_hash,config_hash])[:14]
