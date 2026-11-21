"""Duration-aware batch planning, adaptive batch size control and CUDA OOM recovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def plan_duration_batches(durations: pd.Series, max_batch_size: int, max_batch_seconds: float,
                          clip_cap_s: float | None = None) -> list[list[str]]:
    """Sort clips by duration and greedily pack so that n * max_duration_in_batch <= max_batch_seconds.

    Padding waste is minimal because neighbours have similar length. `durations` is indexed by filename.
    """
    d = durations.astype(float).fillna(durations.median() if durations.notna().any() else 5.0)
    if clip_cap_s:
        d = d.clip(upper=clip_cap_s)
    d = d.clip(lower=0.05).sort_values(kind="mergesort")
    batches, cur, cur_max = [], [], 0.0
    for fname, dur in d.items():
        new_max = max(cur_max, dur)
        if cur and (len(cur) + 1 > max_batch_size or new_max * (len(cur) + 1) > max_batch_seconds):
            batches.append(cur)
            cur, new_max = [], dur
        cur.append(fname)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass
class AdaptiveBatchController:
    batch_size: int = 24
    max_batch_size: int = 64
    min_batch_size: int = 1
    max_batch_seconds: float = 480.0
    safety_fraction: float = 0.80
    grow_after: int = 40
    successes: int = 0
    ooms: int = 0

    def split(self, waves: list[np.ndarray], sr: int) -> list[slice]:
        """Split a decoded batch into sub-batches that respect the current size and seconds budget."""
        out, start, cur_max = [], 0, 0.0
        for i, w in enumerate(waves):
            dur = len(w) / sr
            n = i - start + 1
            m = max(cur_max, dur)
            if n > 1 and (n > self.batch_size or m * n > self.max_batch_seconds):
                out.append(slice(start, i))
                start, m = i, dur
            cur_max = m
        out.append(slice(start, len(waves)))
        return out

    def on_success(self, peak_fraction: float) -> None:
        self.successes += 1
        if self.successes >= self.grow_after and peak_fraction < self.safety_fraction * 0.75:
            self.batch_size = min(self.max_batch_size, int(np.ceil(self.batch_size * 1.25)))
            self.max_batch_seconds *= 1.15
            self.successes = 0

    def on_oom(self) -> None:
        self.ooms += 1
        self.batch_size = max(self.min_batch_size, self.batch_size // 2)
        self.max_batch_seconds = max(10.0, self.max_batch_seconds / 2)
        self.successes = 0

    def state(self) -> dict:
        return {"batch_size": self.batch_size, "max_batch_seconds": round(self.max_batch_seconds, 1), "ooms": self.ooms}


class VRAMMonitor:
    def __init__(self, device: str):
        self.cuda = device == "cuda"
        self.total = 0.0
        if self.cuda:
            import torch

            self.total = torch.cuda.get_device_properties(0).total_memory

    def reset(self) -> None:
        if self.cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def peak_fraction(self) -> float:
        if not self.cuda or not self.total:
            return 0.0
        import torch

        return torch.cuda.max_memory_reserved() / self.total

    def snapshot(self) -> dict:
        if not self.cuda:
            return {"device": "cpu"}
        import torch

        free, total = torch.cuda.mem_get_info()
        return {"allocated_gb": torch.cuda.memory_allocated() / 1e9, "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "free_gb": free / 1e9, "total_gb": total / 1e9}

    @staticmethod
    def release() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
