"""Shared DSP constants (numpy) used by both the torch GPU extractor and the numpy reference."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

EPS = 1e-10


@dataclass(frozen=True)
class AcousticParams:
    sr: int = 16000
    n_fft: int = 512
    hop: int = 160
    win: int = 400
    n_mels: int = 64
    n_mfcc: int = 13
    fmin: float = 20.0
    fmax: float = 8000.0
    mel_summary_bands: int = 8
    vad_noise_percentile: float = 10.0
    vad_peak_percentile: float = 99.0
    vad_offset_db: float = 9.0
    vad_max_below_peak_db: float = 40.0
    vad_min_dynamic_range_db: float = 6.0
    min_speech_s: float = 0.06
    min_silence_s: float = 0.12
    long_pause_s: float = 0.5
    nucleus_prominence_db: float = 4.0
    nucleus_min_gap_s: float = 0.10
    f0_min_hz: float = 80.0
    f0_max_hz: float = 600.0
    yin_frame_length: int = 1024
    yin_threshold: float = 0.15
    yin_voicing_max: float = 0.35
    spectral_min_frames: int = 5
    bandlimit_drop_db: float = 45.0
    hf_split_hz: float = 4000.0

    @property
    def hop_s(self) -> float:
        return self.hop / self.sr

    @classmethod
    def from_cfg(cls, cfg) -> "AcousticParams":
        a = cfg.section("audio") if hasattr(cfg, "section") else dict(cfg.get("audio", {}))
        mapping = {"sr": a.get("sample_rate"), "hop": a.get("hop_length"), "win": a.get("win_length")}
        kwargs = {}
        for f in fields(cls):
            v = mapping.get(f.name, a.get(f.name))
            if v is not None:
                kwargs[f.name] = type(f.default)(v)
        return cls(**kwargs)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """HTK-style triangular filters with Slaney area normalisation. Shape (n_fft//2+1, n_mels)."""
    fmax = min(fmax, sr / 2)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mpts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hpts = mel_to_hz(mpts)
    fb = np.zeros((len(freqs), n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, ce, hi = hpts[m], hpts[m + 1], hpts[m + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - freqs) / max(hi - ce, 1e-9)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down))
        fb[:, m] *= 2.0 / max(hi - lo, 1e-9)
    return fb.astype(np.float32)


def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape (n_in, n_out): coeffs = x @ D."""
    n = np.arange(n_in)[:, None]
    k = np.arange(n_out)[None, :]
    d = np.cos(np.pi / n_in * (n + 0.5) * k)
    d[:, 0] *= 1.0 / np.sqrt(2.0)
    return (d * np.sqrt(2.0 / n_in)).astype(np.float32)


def hann_window(win: int) -> np.ndarray:
    """Periodic Hann, identical to torch.hann_window(win, periodic=True)."""
    n = np.arange(win)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / win)).astype(np.float32)


def padded_window(win: int, n_fft: int) -> np.ndarray:
    w = hann_window(win)
    left = (n_fft - win) // 2
    out = np.zeros(n_fft, dtype=np.float32)
    out[left:left + win] = w
    return out


def n_frames(n_samples: int, hop: int) -> int:
    return 1 + n_samples // hop
