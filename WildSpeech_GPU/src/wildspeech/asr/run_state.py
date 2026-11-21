"""Runtime state that the lifecycle status needs but the registry cannot know: locks, last runs, smoke tests, hashes.

  models/_locks/<model>.download.json      live while asr.model_manager downloads / converts (heartbeat thread)
  cache/v3/asr/<model>/_running.json       live while P24 inference runs for the model (heartbeat thread)
  cache/v3/asr/<model>/_last_run.json      {"status": "ok" | "failed", "error", "finished", "revision", "splits"}
  cache/v3/asr_smoke/<model>/_smoke.json   last smoke decode {"ok", "revision", "n_clips", "clips_per_s", ...}
  models/_sha256_cache.json                verified LFS sha256 keyed by file:size:mtime (multi-GB files hashed once)

A lock is LIVE when its heartbeat is younger than `stale_after_s` (default 180 s) — a crashed process therefore never
leaves a model stuck in DOWNLOADING / INFERENCE_RUNNING. Everything here is plain JSON on the local disk.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from ..io_utils import read_json, write_json

STALE_AFTER_S = 180.0
HEARTBEAT_S = 30.0


def models_root(ctx) -> Path:
    from .model_registry import document_for

    doc = document_for(ctx)
    return Path(ctx.paths.root) / (doc.checkpoint_root or "models")


def download_lock_path(ctx, key: str) -> Path:
    return models_root(ctx) / "_locks" / f"{key}.download.json"


def run_lock_path(ctx, key: str) -> Path:
    return Path(ctx.paths.v3_cache) / "asr" / key.lower() / "_running.json"


def last_run_path(ctx, key: str) -> Path:
    return Path(ctx.paths.v3_cache) / "asr" / key.lower() / "_last_run.json"


def smoke_path(ctx, key: str) -> Path:
    return Path(ctx.paths.v3_cache) / "asr_smoke" / key.lower() / "_smoke.json"


def hash_cache_path(ctx) -> Path:
    return models_root(ctx) / "_sha256_cache.json"


def _live(p: Path, stale_after_s: float = STALE_AFTER_S) -> bool:
    d = read_json(p, {}) if p.exists() else {}
    if not d:
        return False
    return (time.time() - float(d.get("heartbeat", 0))) < stale_after_s


class HeartbeatLock:
    """Context manager: writes the lock file, refreshes its heartbeat from a daemon thread, removes it on exit."""

    def __init__(self, path: Path, info: dict | None = None, every_s: float = HEARTBEAT_S):
        self.path, self.info, self.every = Path(path), dict(info or {}), float(every_s)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _write(self) -> None:
        write_json(self.path, {**self.info, "pid": os.getpid(), "heartbeat": time.time(),
                               "started": self.info.get("started")})

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.info.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._write()

        def beat():
            while not self._stop.wait(self.every):
                try:
                    self._write()
                except Exception:  # noqa: BLE001
                    pass
        self._t = threading.Thread(target=beat, daemon=True)
        self._t.start()
        return self

    def update(self, **kw) -> None:
        self.info.update(kw)
        self._write()

    def __exit__(self, *exc):
        self._stop.set()
        try:
            self.path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return False


def record_last_run(ctx, key: str, status: str, **kw) -> None:
    write_json(last_run_path(ctx, key), {"status": status, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw})


def record_smoke(ctx, key: str, result: dict) -> None:
    write_json(smoke_path(ctx, key), {**result, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})


def load_hash_cache(ctx) -> dict:
    p = hash_cache_path(ctx)
    return dict(read_json(p, {}) or {}) if p.exists() else {}


def save_hash_cache(ctx, cache: dict) -> None:
    if cache:
        write_json(hash_cache_path(ctx), dict(sorted(cache.items())))


def runtime_states(ctx, keys: list[str], splits=None) -> dict:
    """{model_key: {downloading, running, last_failed, complete, smoke}} from local files only."""
    out = {k: {"downloading": _live(download_lock_path(ctx, k)), "running": _live(run_lock_path(ctx, k)),
               "last_failed": (read_json(last_run_path(ctx, k), {}) or {}).get("status") == "failed",
               "complete": False, "smoke": read_json(smoke_path(ctx, k), {}) or {}} for k in keys}
    try:
        from .outputs import execution_state

        st = execution_state(ctx, keys, splits)
        if not st.empty:
            comp = st.groupby("model_key")["state"].apply(lambda s: bool((s == "complete").all()))
            for k, v in comp.items():
                if k in out:
                    out[k]["complete"] = bool(v)
                    if v:
                        out[k]["last_failed"] = False
    except Exception:  # noqa: BLE001 - data bundle unavailable: completeness unknown
        pass
    return out


def lock_table(ctx, keys: list[str]) -> list[dict]:
    rows = []
    for k in keys:
        for kind, p in (("download", download_lock_path(ctx, k)), ("inference", run_lock_path(ctx, k))):
            if p.exists():
                d = read_json(p, {}) or {}
                rows.append({"model_key": k, "lock": kind, "live": _live(p), "pid": d.get("pid"),
                             "started": d.get("started"), "age_s": round(time.time() - float(d.get("heartbeat", 0)), 1),
                             "info": json.dumps({x: y for x, y in d.items() if x not in ("pid", "heartbeat", "started")})[:200]})
    return rows
