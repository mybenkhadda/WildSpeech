"""Independent numpy reference implementation for CPU/GPU numerical parity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dsp import EPS, AcousticParams, dct_matrix, mel_filterbank, padded_window


def numpy_power_spectrogram(y: np.ndarray, p: AcousticParams) -> np.ndarray:
    """Equivalent of torch.stft(center=True, pad_mode='constant', periodic Hann). Returns (N, F)."""
    pad = p.n_fft // 2
    y = np.asarray(y, dtype=np.float64)
    if len(y) < p.n_fft // 2 + 1:
        y = np.pad(y, (0, p.n_fft // 2 + 1 - len(y)))
    yp = np.pad(y, (pad, pad))
    n = 1 + (len(yp) - p.n_fft) // p.hop
    idx = np.arange(p.n_fft)[None, :] + p.hop * np.arange(n)[:, None]
    frames = yp[idx] * padded_window(p.win, p.n_fft)[None, :]
    S = np.fft.rfft(frames, n=p.n_fft, axis=1)
    return (S.real ** 2 + S.imag ** 2)


def numpy_parity_stats(y: np.ndarray, p: AcousticParams) -> dict:
    P = numpy_power_spectrogram(y, p)
    n_valid = min(1 + len(y) // p.hop, P.shape[0])
    P = P[:n_valid]
    freqs = np.linspace(0, p.sr / 2, p.n_fft // 2 + 1)
    pn = P / (P.sum(1, keepdims=True) + EPS)
    cen = (pn * freqs).sum(1)
    ridx = np.minimum((np.cumsum(pn, 1) < 0.85).sum(1), P.shape[1] - 1)
    flat = np.exp(np.log(P + EPS).mean(1)) / (P.mean(1) + EPS)
    win_energy = float((padded_window(p.win, p.n_fft) ** 2).sum())
    e_db = 10 * np.log10(2 * P.sum(1) / (p.n_fft * win_energy) + EPS)
    mf = (10 * np.log10(P @ mel_filterbank(p.sr, p.n_fft, p.n_mels, p.fmin, p.fmax).astype(np.float64) + EPS)) @ dct_matrix(p.n_mels, p.n_mfcc).astype(np.float64)
    out = {"centroid_mean_all": cen.mean(), "rolloff_mean_all": freqs[ridx].mean(), "flatness_mean_all": flat.mean(),
           "energy_db_mean_all": e_db.mean()}
    for c, v in enumerate(mf.mean(0)):
        out[f"mfcc{c}_mean_all"] = v
    return {k: float(v) for k, v in out.items()}


def compare_parity(a: list[dict], b: list[dict], rtol: float, atol: float, label_a: str = "gpu", label_b: str = "cpu") -> pd.DataFrame:
    """Per-feature max abs / relative difference and pass flag (|a-b| <= atol + rtol*|b|)."""
    da, db = pd.DataFrame(a), pd.DataFrame(b)
    rows = []
    for col in sorted(set(da.columns) & set(db.columns)):
        x, y = da[col].to_numpy(float), db[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        both_nan = (~np.isfinite(x)) & (~np.isfinite(y))
        diff = np.abs(x[ok] - y[ok])
        tol = atol + rtol * np.abs(y[ok])
        rows.append({"feature": col, "n": int(ok.sum()), "nan_mismatch": int((~ok & ~both_nan).sum()),
                     "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
                     "max_rel_diff": float((diff / (np.abs(y[ok]) + 1e-12)).max()) if len(diff) else np.nan,
                     "pass": bool((diff <= tol).all() and (~ok & ~both_nan).sum() == 0),
                     "compared": f"{label_a} vs {label_b}"})
    return pd.DataFrame(rows)
