"""CPU discrete-wavelet features (pywt is C-backed; GPU gives no benefit at these sizes)."""
from __future__ import annotations

import numpy as np


def wavelet_features(y: np.ndarray, sr: int, wavelet: str = "db4", level: int = 6, max_seconds: float = 10.0) -> dict:
    import pywt
    from scipy.stats import kurtosis

    y = np.asarray(y[: int(max_seconds * sr)], dtype=np.float64)
    out: dict = {}
    keys_level = range(1, level + 1)
    if len(y) < 64 or not np.any(y):
        out.update({f"wl_rel_d{l}": np.nan for l in keys_level})
        out.update({f"wl_kurt_d{l}": np.nan for l in keys_level})
        out.update({"wl_rel_a": np.nan, "wl_entropy": np.nan, "wl_log_energy": np.nan, "wl_levels": 0.0})
        return out
    w = pywt.Wavelet(wavelet)
    L = max(1, min(level, pywt.dwt_max_level(len(y), w.dec_len)))
    coeffs = pywt.wavedec(y, w, level=L)
    energies = np.array([np.sum(c ** 2) for c in coeffs])
    total = energies.sum() + 1e-12
    rel = energies / total
    out["wl_rel_a"] = float(rel[0])
    for l in keys_level:
        # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]; detail level l (1 = finest) sits at index L - l + 1
        if l <= L:
            c = coeffs[L - l + 1]
            out[f"wl_rel_d{l}"] = float(rel[L - l + 1])
            out[f"wl_kurt_d{l}"] = float(kurtosis(c)) if len(c) > 8 else np.nan
        else:
            out[f"wl_rel_d{l}"] = np.nan
            out[f"wl_kurt_d{l}"] = np.nan
    out["wl_entropy"] = float(-(rel * np.log(rel + 1e-12)).sum() / np.log(len(rel))) if len(rel) > 1 else np.nan
    out["wl_log_energy"] = float(np.log10(total / len(y)))
    out["wl_levels"] = float(L)
    return out
