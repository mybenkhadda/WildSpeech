"""Logging to console + rotating file, plus a resource-aware tqdm progress wrapper."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "wildspeech", log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        from logging.handlers import RotatingFileHandler

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(Path(log_dir) / f"{name}.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGERS[name] = logger
    return logger


def process_ram_gb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float("nan")


def system_ram_available_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("nan")


def vram_gb() -> tuple[float, float]:
    """(currently reserved GB, peak reserved GB) for the current CUDA device."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1e9, torch.cuda.max_memory_reserved() / 1e9
    except Exception:
        pass
    return 0.0, 0.0


class ResourceProgress:
    """tqdm bar showing %, ETA, clips/s, RAM, VRAM, failures and checkpoint status."""

    def __init__(self, total: int, desc: str, unit: str = "clip"):
        from tqdm.auto import tqdm

        self.bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, smoothing=0.05)
        self.failures = 0
        self.checkpoint = "-"
        self._t0 = time.perf_counter()
        self._done = 0

    def update(self, n: int, failures: int = 0, checkpoint: str | None = None, **extra) -> None:
        self._done += n
        self.failures += failures
        if checkpoint is not None:
            self.checkpoint = checkpoint
        elapsed = max(time.perf_counter() - self._t0, 1e-6)
        vram, peak = vram_gb()
        post = {"clips/s": f"{self._done / elapsed:.1f}", "RAM": f"{process_ram_gb():.1f}G",
                "VRAM": f"{vram:.1f}/{peak:.1f}G", "fail": self.failures, "ckpt": self.checkpoint}
        post.update({k: v for k, v in extra.items()})
        self.bar.set_postfix(post, refresh=False)
        self.bar.update(n)

    def close(self) -> None:
        self.bar.close()
