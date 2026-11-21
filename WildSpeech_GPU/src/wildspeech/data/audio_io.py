"""Bounded-memory audio access.

* TarRandomAccessReader: reads a member's bytes by (offset, size) with one file handle per thread.
* decode_audio_bytes: FLAC/WAV -> mono float32 at the target rate (soundfile; libsndfile releases the GIL).
* PrefetchingBatchLoader: decodes the NEXT `prefetch_batches` batches on a small thread pool while the GPU
  works on the current one. At most (prefetch_batches + 1) batches of waveforms live in RAM.
* StreamingTarBatchLoader: sequential fallback for compressed archives.
No multiprocessing is used anywhere (Windows spawn + large buffers caused TerminatedWorkerError before).
"""
from __future__ import annotations

import io
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class TarRandomAccessReader:
    def __init__(self, tar_path: Path, manifest: pd.DataFrame):
        self.tar_path = Path(tar_path)
        m = manifest[manifest["safe"]] if "safe" in manifest else manifest
        if bool(m["compressed"].any()):
            raise ValueError("compressed archive: use StreamingTarBatchLoader")
        self._index = {r.filename: (int(r.offset), int(r.size)) for r in m.itertuples(index=False)}
        self._local = threading.local()
        self._handles: list = []
        self._lock = threading.Lock()

    def __contains__(self, filename: str) -> bool:
        return filename in self._index

    def _fh(self):
        fh = getattr(self._local, "fh", None)
        if fh is None:
            fh = open(self.tar_path, "rb")
            self._local.fh = fh
            with self._lock:
                self._handles.append(fh)
        return fh

    def read(self, filename: str) -> bytes:
        offset, size = self._index[filename]
        fh = self._fh()
        fh.seek(offset)
        data = fh.read(size)
        if len(data) != size:
            raise IOError(f"short read for {filename}: {len(data)}/{size}")
        return data

    def close(self) -> None:
        with self._lock:
            for fh in self._handles:
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()


def _resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or len(y) == 0:
        return y
    try:
        import soxr

        return soxr.resample(y, sr, target_sr, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy.signal import resample_poly

        g = int(np.gcd(int(sr), int(target_sr)))
        return resample_poly(y, int(target_sr) // g, int(sr) // g).astype(np.float32)


def decode_audio_bytes(data: bytes, target_sr: int, max_seconds: float | None = None) -> tuple[np.ndarray, dict]:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(data)) as f:
        sr = int(f.samplerate)
        n_total = int(f.frames)
        n_read = n_total if not max_seconds else min(n_total, int(np.ceil(max_seconds * sr)))
        x = f.read(frames=n_read, dtype="float32", always_2d=True)
        channels = int(f.channels)
    y = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    bad = ~np.isfinite(y)
    if bad.any():
        y = np.where(bad, 0.0, y).astype(np.float32)
    y = _resample(np.ascontiguousarray(y, dtype=np.float32), sr, int(target_sr))
    meta = {"file_sr": sr, "channels": channels, "true_duration_s": n_total / sr if sr else np.nan,
            "decoded_duration_s": len(y) / target_sr, "cropped": n_read < n_total, "n_invalid": int(bad.sum())}
    return y, meta


@dataclass
class LoadedBatch:
    filenames: list[str]
    waves: list[np.ndarray]
    metas: list[dict]
    failures: list[tuple[str, str]] = field(default_factory=list)
    decode_s: float = 0.0


class PrefetchingBatchLoader:
    def __init__(self, reader: TarRandomAccessReader, batches: list[list[str]], target_sr: int,
                 max_seconds: float | None, n_threads: int = 6, prefetch_batches: int = 2,
                 ram_soft_limit_gb: float | None = None):
        self.reader = reader
        self.batches = batches
        self.target_sr = int(target_sr)
        self.max_seconds = max_seconds
        self.n_threads = max(1, int(n_threads))
        self.prefetch = max(0, int(prefetch_batches))
        self.ram_soft_limit_gb = ram_soft_limit_gb

    def _load_one(self, filename: str):
        t0 = time.perf_counter()
        try:
            y, meta = decode_audio_bytes(self.reader.read(filename), self.target_sr, self.max_seconds)
            return filename, y, meta, None, time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - recorded as an explicit failure row
            return filename, None, None, f"{type(e).__name__}: {e}", time.perf_counter() - t0

    def _ram_ok(self) -> bool:
        if not self.ram_soft_limit_gb:
            return True
        from ..logging_utils import process_ram_gb

        r = process_ram_gb()
        return not np.isfinite(r) or r < self.ram_soft_limit_gb

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        pool = ThreadPoolExecutor(max_workers=self.n_threads, thread_name_prefix="decode")
        pending: deque[list[Future]] = deque()
        it = iter(self.batches)
        try:
            def submit_next() -> bool:
                b = next(it, None)
                if b is None:
                    return False
                pending.append([pool.submit(self._load_one, f) for f in b])
                return True

            for _ in range(self.prefetch + 1):
                if not submit_next():
                    break
            while pending:
                futs = pending.popleft()
                results = [f.result() for f in futs]
                waits = 0
                while not self._ram_ok() and waits < 50:
                    time.sleep(0.1)
                    waits += 1
                submit_next()
                batch = LoadedBatch([], [], [])
                for fname, y, meta, err, dt in results:
                    batch.decode_s += dt
                    if err is not None:
                        batch.failures.append((fname, err))
                    else:
                        batch.filenames.append(fname)
                        batch.waves.append(y)
                        batch.metas.append(meta)
                yield batch
        finally:
            for futs in pending:
                for f in futs:
                    f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)


class StreamingTarBatchLoader:
    """Sequential reader for compressed TARs: yields batches in archive order for wanted files only."""

    def __init__(self, tar_path: Path, wanted: set[str], batch_size: int, target_sr: int, max_seconds: float | None):
        self.tar_path, self.wanted = Path(tar_path), set(wanted)
        self.batch_size, self.target_sr, self.max_seconds = int(batch_size), int(target_sr), max_seconds

    def __len__(self) -> int:
        return int(np.ceil(len(self.wanted) / max(self.batch_size, 1)))

    def __iter__(self):
        batch = LoadedBatch([], [], [])
        with tarfile.open(self.tar_path, mode="r|*") as tf:
            for m in tf:
                base = m.name.replace("\\", "/").split("/")[-1]
                if not m.isfile() or base not in self.wanted:
                    continue
                t0 = time.perf_counter()
                try:
                    data = tf.extractfile(m).read()
                    y, meta = decode_audio_bytes(data, self.target_sr, self.max_seconds)
                    batch.filenames.append(base)
                    batch.waves.append(y)
                    batch.metas.append(meta)
                except Exception as e:  # noqa: BLE001
                    batch.failures.append((base, f"{type(e).__name__}: {e}"))
                batch.decode_s += time.perf_counter() - t0
                if len(batch.filenames) + len(batch.failures) >= self.batch_size:
                    yield batch
                    batch = LoadedBatch([], [], [])
        if batch.filenames or batch.failures:
            yield batch


def make_reader(tar_path: Path, manifest: pd.DataFrame) -> TarRandomAccessReader | None:
    if manifest.empty or bool(manifest["compressed"].any()):
        return None
    return TarRandomAccessReader(tar_path, manifest)
