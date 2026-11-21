"""Controlled audio perturbations for anonymisation / recording-condition sensitivity (P08)."""
from __future__ import annotations

import numpy as np


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)


def add_noise(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    scale = _rms(y) / (10 ** (snr_db / 20.0)) / _rms(noise)
    return (y + scale * noise).astype(np.float32)


def lowpass_fft(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Brick-wall low-pass (simulates narrow-band capture, e.g. upsampled 8-11 kHz recordings)."""
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    Y[f > cutoff_hz] = 0
    return np.fft.irfft(Y, n=len(y)).astype(np.float32)


def gain(y: np.ndarray, db: float) -> np.ndarray:
    return (y * 10 ** (db / 20.0)).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, str]:
    """Phase-vocoder pitch shift (torchaudio) preserving duration; falls back to resampling (speed change)."""
    try:
        import torch
        import torchaudio.functional as AF

        out = AF.pitch_shift(torch.from_numpy(y).float().unsqueeze(0), sr, n_steps=semitones).squeeze(0).numpy()
        return out.astype(np.float32), "torchaudio_phase_vocoder"
    except Exception:
        from scipy.signal import resample

        factor = 2 ** (semitones / 12.0)
        sped = resample(y, max(16, int(len(y) / factor)))
        return sped.astype(np.float32), "resample_speed_change"


def apply_perturbation(y: np.ndarray, sr: int, name: str, seed: int = 0) -> tuple[np.ndarray, str]:
    if name.startswith("pitch_up_"):
        return pitch_shift(y, sr, float(name.split("_")[-1].replace("st", "")))
    if name.startswith("pitch_down_"):
        return pitch_shift(y, sr, -float(name.split("_")[-1].replace("st", "")))
    if name.startswith("noise_"):
        return add_noise(y, float(name.split("_")[1].replace("db", "")), seed), "white_noise"
    if name.startswith("lowpass_"):
        return lowpass_fft(y, sr, float(name.split("_")[1].replace("k", "")) * 1000.0), "fft_lowpass"
    if name.startswith("gain_"):
        v = name.split("_", 1)[1].replace("db", "").replace("minus_", "-").replace("plus_", "")
        return gain(y, float(v)), "gain"
    raise ValueError(f"unknown perturbation {name}")
