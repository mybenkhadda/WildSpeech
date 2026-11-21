"""Section 07 - temporal, rhythm and fluency intelligence.

Scientific question: does response organisation predict later reading
performance?

Note: ``t_initial_sil`` is the silence before the first detected speech
segment in the recording. It is NOT a verified reaction time: the recording
start is not known to coincide with prompt offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_utils import robust_stats

TEMPORAL_NAN_KEYS = [
    "t_total_dur", "t_active_dur", "t_active_ratio", "t_n_segments", "t_initial_sil", "t_final_sil",
    "t_internal_sil", "t_n_pauses", "t_n_short_pauses", "t_n_long_pauses", "t_pause_speech_ratio",
    "t_pause_cv", "t_peak_rate_hz", "t_peak_spacing_mean", "t_peak_spacing_std", "t_rhythm_regularity",
    "t_npvi", "t_mod_freq_hz", "t_mod_energy_ratio_2_8hz", "t_temporal_entropy", "t_energy_transitions",
    "t_onset_sharpness", "t_offset_sharpness",
]


def _speech_span(b) -> tuple[int, int]:
    if not b.segments:
        return 0, len(b.rms)
    hop_s = b.hop / b.sr
    start = int(np.floor(b.segments[0][0] / hop_s))
    end = int(np.ceil(b.segments[-1][1] / hop_s))
    return max(0, start), min(len(b.rms), max(end, start + 1))


def temporal_features(b, cfg: dict) -> dict:
    from scipy.signal import find_peaks
    from scipy.stats import kurtosis, skew

    out: dict = {k: np.nan for k in TEMPORAL_NAN_KEYS}
    dur = float(b.duration)
    segs = list(b.segments)
    hop_s = b.hop / b.sr
    fs_env = 1.0 / hop_s
    out["t_total_dur"] = dur
    seg_durs = np.array([e - s for s, e in segs], dtype=float)
    active = float(seg_durs.sum()) if len(seg_durs) else 0.0
    out["t_active_dur"] = active
    out["t_active_ratio"] = active / dur if dur > 0 else np.nan
    out["t_n_segments"] = float(len(segs))
    out.update(robust_stats(seg_durs, "t_seg_dur", ["mean", "std", "median", "max"], min_n=2))
    if segs:
        gaps = np.array([segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)], dtype=float)
        out["t_initial_sil"] = float(segs[0][0])
        out["t_final_sil"] = float(max(dur - segs[-1][1], 0.0))
        out["t_internal_sil"] = float(gaps.sum()) if len(gaps) else 0.0
        out["t_n_pauses"] = float(len(gaps))
        out["t_n_short_pauses"] = float((gaps < cfg["long_pause_s"]).sum()) if len(gaps) else 0.0
        out["t_n_long_pauses"] = float((gaps >= cfg["long_pause_s"]).sum()) if len(gaps) else 0.0
        out["t_pause_speech_ratio"] = float(gaps.sum() / active) if active > 0 else np.nan
        out.update(robust_stats(gaps, "t_pause_dur", ["mean", "std", "median", "max"], min_n=2))
        if len(gaps) >= 2 and gaps.mean() > 0:
            out["t_pause_cv"] = float(gaps.std(ddof=1) / gaps.mean())
    else:
        out.update({f"t_pause_dur_{s}": np.nan for s in ["mean", "std", "median", "max"]})
    lo, hi = _speech_span(b)
    env = b.rms[lo:hi].astype(float)
    env_db = b.rms_db[lo:hi].astype(float)
    if len(env_db) >= 3:
        out["t_env_mean"] = float(env_db.mean())
        out["t_env_std"] = float(env_db.std(ddof=1))
        out["t_env_skew"] = float(skew(env_db)) if env_db.std() > 0 else np.nan
        out["t_env_kurt"] = float(kurtosis(env_db)) if env_db.std() > 0 else np.nan
    else:
        out.update({"t_env_mean": np.nan, "t_env_std": np.nan, "t_env_skew": np.nan, "t_env_kurt": np.nan})
    if len(env_db) >= 5 and active > 0:
        distance = max(1, int(round(0.08 / hop_s)))
        peaks, _ = find_peaks(env_db, prominence=3.0, distance=distance)
        out["t_peak_rate_hz"] = float(len(peaks) / active)
        if len(peaks) >= 3:
            intervals = np.diff(peaks) * hop_s
            out["t_peak_spacing_mean"] = float(intervals.mean())
            out["t_peak_spacing_std"] = float(intervals.std(ddof=1))
            cv = intervals.std(ddof=1) / intervals.mean() if intervals.mean() > 0 else np.nan
            out["t_rhythm_regularity"] = float(1.0 / (1.0 + cv)) if np.isfinite(cv) else np.nan
            pair = np.abs(np.diff(intervals)) / ((intervals[1:] + intervals[:-1]) / 2)
            out["t_npvi"] = float(100 * pair.mean())
        crossings = np.diff((env_db > np.median(env_db)).astype(int))
        out["t_energy_transitions"] = float(np.abs(crossings).sum() / max(len(env_db) * hop_s, 1e-6))
    centered = env - env.mean() if len(env) else env
    denom = float((centered ** 2).sum()) if len(env) else 0.0
    for lag_s in (0.1, 0.2, 0.3, 0.5):
        lag = int(round(lag_s / hop_s))
        key = f"t_env_autocorr_{int(lag_s * 1000)}ms"
        if denom > 0 and len(env) > lag + 5:
            out[key] = float((centered[:-lag] * centered[lag:]).sum() / denom)
        else:
            out[key] = np.nan
    if len(env) >= int(fs_env):  # at least one second for modulation analysis
        spec = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=hop_s)
        band = (freqs >= 1) & (freqs <= 16)
        if band.any() and spec[band].sum() > 0:
            out["t_mod_freq_hz"] = float(freqs[band][np.argmax(spec[band])])
            mid = (freqs >= 2) & (freqs <= 8)
            out["t_mod_energy_ratio_2_8hz"] = float(spec[mid].sum() / spec[band].sum())
    if len(env) >= 5 and env.sum() > 0:
        p = env / env.sum()
        p = p[p > 0]
        out["t_temporal_entropy"] = float(-(p * np.log(p)).sum() / np.log(len(env)))
    if segs and len(b.rms_db) >= 5:
        slope = np.diff(b.rms_db) / hop_s
        w = max(1, int(round(0.05 / hop_s)))
        s0 = int(round(segs[0][0] / hop_s))
        e0 = int(round(segs[-1][1] / hop_s))
        on = slope[max(0, s0 - w):min(len(slope), s0 + w + 1)]
        off = slope[max(0, e0 - w - 1):min(len(slope), e0 + w)]
        out["t_onset_sharpness"] = float(on.max()) if len(on) else np.nan
        out["t_offset_sharpness"] = float(-off.min()) if len(off) else np.nan
    return out


def stage_temporal_features(ctx) -> dict:
    from .family_analysis import analyze_family, complexity_surface

    result = analyze_family(ctx, family="TEMPORAL", prefixes=("t_",), split_by_correctness=True)
    complexity_surface(ctx, feature="t_active_dur", outputs=result["outputs"])
    return result
