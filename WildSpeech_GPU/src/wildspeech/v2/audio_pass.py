"""One reusable bounded-memory audio pass for every V2 extraction (harmonized acoustics, frame sequences,
phonetic models, foundation-model embeddings).

TAR byte-offset reads (one handle per thread) -> threaded FLAC decoding with at most (prefetch+1) batches in RAM
-> duration-sorted batches under a padded-seconds budget -> `process(waves, filenames)` on the GPU with
recursive CUDA-OOM splitting (single clip -> optional CPU fallback) -> buffered rows flushed as atomic Parquet
checkpoints -> resume = skip filenames already stored or recorded as failures. tqdm shows %, ETA, clips/s,
process RAM, VRAM reserved/peak, failures and the last checkpoint.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .. import FILE
from ..acoustic.batching import VRAMMonitor, is_oom, plan_duration_batches
from ..data.audio_io import PrefetchingBatchLoader, StreamingTarBatchLoader, make_reader
from ..data.tar_manifest import build_tar_manifest
from ..logging_utils import ResourceProgress

ProcessFn = Callable[[list, list], list]


class AudioPass:
    def __init__(self, ctx, split: str, desc: str, max_seconds: float, max_batch_size: int | None = None,
                 batch_seconds: float | None = None, sample_rate: int | None = None):
        self.ctx, self.split, self.desc = ctx, split, desc
        rt = ctx.cfg.get_path("v2.runtime", {}) or {}
        self.max_seconds = float(max_seconds)
        self.max_batch_size = int(max_batch_size or rt.get("max_batch_size", 48))
        self.batch_seconds = float(batch_seconds or rt.get("batch_seconds", 360.0))
        self.sr = int(sample_rate or ctx.cfg.get_path("audio.sample_rate", 16000))
        self.threads = int(rt.get("audio_threads", ctx.cfg.get_path("runtime.decode_threads", 6)))
        self.prefetch = int(rt.get("prefetch_batches", 2))
        self.ram_limit = ctx.cfg.get_path("runtime.ram_soft_limit_gb")
        self.checkpoint_every = int(rt.get("checkpoint_every", 1000))
        self.log = ctx.logger
        self.ooms = 0

    # ----------------------------------------------------------- candidates
    def clips(self, max_clips: int | None = None) -> pd.DataFrame:
        b = self.ctx.bundle()
        c = (b.train_clips if self.split == "train" else b.test_clips)[[FILE, "duration_sec", "expected_text", "task"]]
        c = c.drop_duplicates(FILE)
        if max_clips:
            c = c.sort_values(FILE).head(int(max_clips))
        return c.reset_index(drop=True)

    def manifest(self) -> pd.DataFrame | None:
        key = f"{self.split}_audio"
        if key not in self.ctx.paths.files:
            return None
        return build_tar_manifest(self.ctx.paths.files[key], self.ctx.paths.cache / "manifests", logger=self.log)

    # ------------------------------------------------------------ safe call
    def safe_process(self, process: ProcessFn, waves: list, names: list, cpu_fallback: ProcessFn | None = None) -> list:
        try:
            out = process(waves, names)
            if len(out) != len(waves):
                raise RuntimeError(f"process returned {len(out)} rows for {len(waves)} clips")
            return out
        except Exception as e:  # noqa: BLE001 - recovery path; every unrecoverable clip is recorded as a failure
            oom = is_oom(e)
            if oom:
                self.ooms += 1
                VRAMMonitor.release()
            if len(waves) == 1:
                if oom and cpu_fallback is not None:
                    try:
                        return cpu_fallback(waves, names)
                    except Exception as e2:  # noqa: BLE001
                        self.log.warning("%s: CPU fallback failed for %s: %s", self.desc, names[0], e2)
                else:
                    self.log.warning("%s: failure on %s: %s: %s", self.desc, names[0], type(e).__name__, e)
                return [None]
            mid = len(waves) // 2
            return (self.safe_process(process, waves[:mid], names[:mid], cpu_fallback)
                    + self.safe_process(process, waves[mid:], names[mid:], cpu_fallback))

    # ------------------------------------------------------------------ run
    def run(self, todo: list[str], process: ProcessFn, flush: Callable[[list[dict]], str],
            record_failures: Callable[[list[dict]], None], cpu_fallback: ProcessFn | None = None,
            durations: pd.Series | None = None) -> dict:
        t0 = time.perf_counter()
        summary = {"split": self.split, "requested": len(todo), "processed": 0, "failures": 0, "seconds": 0.0}
        if not todo:
            return summary
        man = self.manifest()
        if man is None:
            raise FileNotFoundError(f"{self.split}_audio TAR not found")
        in_tar = set(man.loc[man["safe"], FILE]) if "safe" in man else set(man[FILE])
        missing = sorted(set(todo) - in_tar)
        if missing:
            record_failures([{FILE: m, "stage": "manifest", "error": "missing_in_tar"} for m in missing])
        todo = sorted(set(todo) & in_tar)
        if durations is None:
            durations = self.clips().set_index(FILE)["duration_sec"]
        durs = durations.reindex(todo)
        batches = plan_duration_batches(durs, self.max_batch_size, self.batch_seconds, self.max_seconds)
        reader = make_reader(self.ctx.paths.files[f"{self.split}_audio"], man)
        if reader is not None:
            loader = PrefetchingBatchLoader(reader, batches, self.sr, self.max_seconds, self.threads, self.prefetch, self.ram_limit)
        else:
            loader = StreamingTarBatchLoader(self.ctx.paths.files[f"{self.split}_audio"], set(todo), self.max_batch_size,
                                             self.sr, self.max_seconds)
        prog = ResourceProgress(total=len(todo), desc=self.desc)
        buf: list[dict] = []
        fails: list[dict] = []
        n_done = 0
        try:
            for batch in loader:
                fails += [{FILE: f, "stage": "decode", "error": err} for f, err in batch.failures]
                n_fail = len(batch.failures)
                if batch.waves:
                    rows = self.safe_process(process, batch.waves, batch.filenames, cpu_fallback)
                    for f, r in zip(batch.filenames, rows):
                        if r is None:
                            fails.append({FILE: f, "stage": "process", "error": "model_or_feature_failure"})
                            n_fail += 1
                        else:
                            buf.append({FILE: f, **r})
                ck = None
                if len(buf) >= self.checkpoint_every:
                    ck = flush(buf)
                    buf = []
                if fails and len(fails) >= 200:
                    record_failures(fails)
                    fails = []
                n_done += len(batch.filenames) + len(batch.failures)
                prog.update(len(batch.filenames) + len(batch.failures), failures=n_fail, checkpoint=ck, ooms=self.ooms)
            ck = flush(buf) if buf else "-"
            buf = []
            if fails:
                record_failures(fails)
                fails = []
            prog.update(0, checkpoint=ck)
        except KeyboardInterrupt:
            # interruption: persist every finished clip so resume continues exactly after them
            if buf:
                try:
                    flush(buf)
                except Exception:  # noqa: BLE001
                    self.log.exception("%s: flush on interrupt failed", self.desc)
            if fails:
                try:
                    record_failures(fails)
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            prog.close()
            if reader is not None:
                reader.close()
        dt = time.perf_counter() - t0
        summary.update({"processed": n_done, "failures": prog.failures, "seconds": dt,
                        "clips_per_s": n_done / max(dt, 1e-6), "ooms": self.ooms})
        return summary


def to_device_batch(waves: list[np.ndarray], device: str):
    """Pad a list of waveforms to (B, T) float32 on `device` + int64 lengths."""
    import torch

    lengths = np.array([len(w) for w in waves], dtype=np.int64)
    T = int(max(lengths.max(), 1))
    buf = np.zeros((len(waves), T), dtype=np.float32)
    for i, w in enumerate(waves):
        buf[i, :len(w)] = w
    x = torch.from_numpy(buf)
    if str(device).startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
    return x, torch.from_numpy(lengths).to(device)
