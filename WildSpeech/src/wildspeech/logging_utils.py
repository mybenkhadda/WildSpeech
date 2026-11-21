"""Structured logging, timers, memory monitoring and progress bars."""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

_LOGGERS: dict = {}


def get_logger(name: str = "wildspeech", log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    key = (name, str(log_dir))
    if key in _LOGGERS:
        return _LOGGERS[key]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_ws", False) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh._ws = True  # type: ignore[attr-defined]
        logger.addHandler(sh)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGERS[key] = logger
    return logger


def memory_usage_mb() -> float:
    """Resident set size of the current process in MB (NaN if unavailable)."""
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return usage / 1e3 if sys.platform != "darwin" else usage / 1e6
        except Exception:
            return float("nan")


def available_memory_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / 1e6
    except Exception:
        return float("nan")


class StageTimer:
    """Context manager recording elapsed time and memory deltas."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.logger = logger
        self.elapsed = float("nan")
        self.mem_start = float("nan")
        self.mem_end = float("nan")

    def __enter__(self) -> "StageTimer":
        self.t0 = time.perf_counter()
        self.mem_start = memory_usage_mb()
        if self.logger:
            self.logger.info("START %s (rss=%.0f MB)", self.label, self.mem_start)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.t0
        gc.collect()
        self.mem_end = memory_usage_mb()
        if self.logger:
            state = "FAILED" if exc_type else "END"
            self.logger.info("%s %s in %.1fs (rss=%.0f MB, delta=%.0f MB)", state, self.label,
                             self.elapsed, self.mem_end, self.mem_end - self.mem_start)

    def as_dict(self) -> dict:
        return {"elapsed_s": self.elapsed, "rss_start_mb": self.mem_start, "rss_end_mb": self.mem_end}


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[StageTimer]:
    timer = StageTimer(label, logger)
    with timer:
        yield timer


def progress(iterable: Iterable, total: int | None = None, desc: str = "", disable: bool = False):
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, disable=disable, leave=False)
    except Exception:
        return iterable


def release_memory() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
