"""Section 08 - spectral and spectrotemporal intelligence.

Computes frame-level spectral descriptors once from the shared STFT and
summarises them separately for speech-active frames (``s_sp_``), all frames
(``s_full_``) and background frames (``s_bg_``). Cepstral features use the
``m_`` prefix.
"""
from __future__ import annotations

import numpy as np

from .stats_utils import robust_stats

EPS = 1e-12
FRAME_FEATURES = ["centroid", "bandwidth", "rolloff", "flatness", "entropy", "flux", "zcr", "slope",
                  "skew", "kurt", "dominant"]
PSD_BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def frame_descriptors(b) -> dict:
    import librosa

    P = b.power + EPS
    f = b.freqs[:, None]
    total = P.sum(axis=0)
    pn = P / total
    centroid = (f * pn).sum(axis=0)
    dev = f - centroid
    bandwidth = np.sqrt((dev ** 2 * pn).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        skewness = (dev ** 3 * pn).sum(axis=0) / np.maximum(bandwidth ** 3, EPS)
        kurt = (dev ** 4 * pn).sum(axis=0) / np.maximum(bandwidth ** 4, EPS)
    cum = np.cumsum(pn, axis=0)
    rolloff = b.freqs[np.argmax(cum >= 0.85, axis=0)]
    flatness = np.exp(np.log(P).mean(axis=0)) / P.mean(axis=0)
    entropy = -(pn * np.log(pn)).sum(axis=0) / np.log(P.shape[0])
    logS = np.log1p(b.S)
    flux = np.concatenate([[0.0], np.sqrt((np.clip(np.diff(logS, axis=1), 0, None) ** 2).sum(axis=0))])
    db = 10 * np.log10(P)
    fc = b.freqs - b.freqs.mean()
    slope = (fc[:, None] * (db - db.mean(axis=0))).sum(axis=0) / max((fc ** 2).sum(), EPS)
    zcr = librosa.feature.zero_crossing_rate(b.y, frame_length=b.n_fft, hop_length=b.hop, center=True)[0]
    T = P.shape[1]
    zcr = np.resize(zcr, T) if len(zcr) != T else zcr
    dominant = b.freqs[np.argmax(P, axis=0)]
    return {"centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff, "flatness": flatness,
            "entropy": entropy, "flux": flux, "zcr": zcr, "slope": slope * 1000.0, "skew": skewness,
            "kurt": kurt, "dominant": dominant}


def _region_masks(b) -> dict:
    T = b.power.shape[1]
    energetic = b.rms_db[:T] > (np.nanmin(b.rms_db) + 1.0) if T else np.zeros(0, bool)
    sp = b.vad_mask[:T]
    return {"full": energetic if energetic.any() else np.ones(T, bool), "sp": sp, "bg": ~sp}


def _psd_bands(x: np.ndarray, sr: int, prefix: str) -> dict:
    from scipy.signal import welch

    out = {}
    names = [f"{prefix}_band{lo}_{hi}_rel" for lo, hi in PSD_BANDS]
    if len(x) < 256:
        out.update({n: np.nan for n in names})
        out.update({f"{prefix}_ratio_low_high": np.nan, f"{prefix}_ratio_mid_high": np.nan,
                    f"{prefix}_psd_dominant_hz": np.nan, f"{prefix}_psd_slope": np.nan})
        return out
    freqs, pxx = welch(x, fs=sr, nperseg=min(1024, len(x)))
    total = pxx.sum() + EPS
    for (lo, hi), name in zip(PSD_BANDS, names):
        out[name] = float(pxx[(freqs >= lo) & (freqs < hi)].sum() / total)
    low = pxx[freqs < 1000].sum()
    mid = pxx[(freqs >= 1000) & (freqs < 3000)].sum()
    high = pxx[freqs >= 3000].sum() + EPS
    out[f"{prefix}_ratio_low_high"] = float(np.log10((low + EPS) / high))
    out[f"{prefix}_ratio_mid_high"] = float(np.log10((mid + EPS) / high))
    out[f"{prefix}_psd_dominant_hz"] = float(freqs[np.argmax(pxx)])
    valid = (freqs > 50) & (pxx > 0)
    out[f"{prefix}_psd_slope"] = float(np.polyfit(np.log10(freqs[valid]), np.log10(pxx[valid]), 1)[0]) if valid.sum() > 5 else np.nan
    return out


def _active_signal(b) -> np.ndarray:
    if not b.segments:
        return np.zeros(0, dtype=np.float32)
    parts = [b.y[int(s * b.sr):int(e * b.sr)] for s, e in b.segments]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _background_signal(b) -> np.ndarray:
    keep = np.ones(len(b.y), dtype=bool)
    for s, e in b.segments:
        keep[int(s * b.sr):int(e * b.sr)] = False
    return b.y[keep]


def spectral_features(b, cfg: dict) -> dict:
    import librosa

    out: dict = {}
    frames = frame_descriptors(b)
    masks = _region_masks(b)
    min_frames = int(cfg["spectral_min_frames"])
    full_stats = list(cfg["spectral_robust_stats"])
    for region, mask in masks.items():
        stats = full_stats if region == "sp" else ["mean", "std", "median"]
        n = int(mask.sum())
        out[f"s_{region}_n_frames"] = float(n)
        for name in FRAME_FEATURES:
            values = frames[name][mask] if n >= min_frames else np.array([])
            out.update(robust_stats(values, f"s_{region}_{name}", stats, min_n=min_frames))
    n_bands = int(min(6, np.floor(np.log2((b.sr / 2.0) / 100.0) - 1e-9)))
    contrast = librosa.feature.spectral_contrast(S=b.S, sr=b.sr, n_fft=b.n_fft, n_bands=n_bands, fmin=100.0)
    T = contrast.shape[1]
    sp = masks["sp"][:T]
    for k in range(contrast.shape[0]):
        vals = contrast[k, sp] if sp.sum() >= min_frames else np.array([])
        out[f"s_sp_contrast{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
        out[f"s_sp_contrast{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    out.update(_psd_bands(_active_signal(b), b.sr, "s_sp"))
    bg = _background_signal(b)
    out.update(_psd_bands(bg, b.sr, "s_bg"))
    out["s_bg_level_db"] = float(np.median(b.rms_db[masks["bg"][:len(b.rms_db)]])) if masks["bg"].any() else np.nan
    out["s_bg_duration_s"] = float(len(bg) / b.sr)
    # Time-frequency entropy, stationarity and spectrotemporal modulation (speech region)
    lm = b.logmel[:, masks["sp"][:b.logmel.shape[1]]] if masks["sp"].any() else np.zeros((b.logmel.shape[0], 0))
    if lm.shape[1] >= min_frames:
        pw = np.power(10.0, lm / 10.0)
        pn = pw / (pw.sum() + EPS)
        out["s_sp_tf_entropy"] = float(-(pn * np.log(pn + EPS)).sum() / np.log(pn.size))
        chunks = np.array_split(lm, 4, axis=1)
        means = np.stack([c.mean(axis=1) for c in chunks if c.shape[1] > 0])
        within = np.mean([c.std(axis=1).mean() for c in chunks if c.shape[1] > 1]) if lm.shape[1] >= 8 else np.nan
        between = np.mean([np.linalg.norm(means[i] - means[j]) / np.sqrt(means.shape[1])
                           for i in range(len(means)) for j in range(i + 1, len(means))]) if len(means) > 1 else np.nan
        out["s_sp_stationarity"] = float(between / within) if within and np.isfinite(within) and within > 0 else np.nan
        mod = np.abs(np.fft.rfft2(lm - lm.mean())) ** 2
        tf = np.fft.rfftfreq(lm.shape[1], d=b.hop / b.sr)
        total = mod.sum() + EPS
        for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32)):
            out[f"s_sp_stmod_t{lo}_{hi}hz_rel"] = float(mod[:, (tf >= lo) & (tf < hi)].sum() / total)
        spec_axis = mod.sum(axis=1)
        half = max(1, len(spec_axis) // 8)
        out["s_sp_stmod_spectral_low_rel"] = float((spec_axis[:half].sum() + spec_axis[-half:].sum()) / (spec_axis.sum() + EPS))
    else:
        out.update({"s_sp_tf_entropy": np.nan, "s_sp_stationarity": np.nan, "s_sp_stmod_spectral_low_rel": np.nan})
        out.update({f"s_sp_stmod_t{lo}_{hi}hz_rel": np.nan for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32))})
    if cfg.get("enable_chroma"):
        chroma = librosa.feature.chroma_stft(S=b.power, sr=b.sr, n_fft=b.n_fft)
        c = chroma[:, masks["sp"][:chroma.shape[1]]]
        if c.shape[1] >= min_frames:
            cn = c / (c.sum(axis=0, keepdims=True) + EPS)
            ent = -(cn * np.log(cn + EPS)).sum(axis=0) / np.log(12)
            out["s_sp_chroma_entropy_mean"] = float(ent.mean())
            out["s_sp_chroma_peak_ratio_mean"] = float(cn.max(axis=0).mean())
        else:
            out["s_sp_chroma_entropy_mean"] = out["s_sp_chroma_peak_ratio_mean"] = np.nan
    if cfg.get("enable_hpss"):
        H, Pm = librosa.decompose.hpss(b.S)
        sp_mask = masks["sp"][:H.shape[1]]
        if sp_mask.sum() >= min_frames:
            h, p = (H[:, sp_mask] ** 2).sum(), (Pm[:, sp_mask] ** 2).sum()
            out["s_sp_harmonic_ratio"] = float(h / (h + p + EPS))
            out["s_sp_percussive_ratio"] = float(p / (h + p + EPS))
        else:
            out["s_sp_harmonic_ratio"] = out["s_sp_percussive_ratio"] = np.nan
    return out


def mfcc_features(b, cfg: dict) -> dict:
    import librosa

    out: dict = {}
    min_frames = int(cfg["spectral_min_frames"])
    M = b.mfcc
    T = M.shape[1]
    width = min(9, T if T % 2 == 1 else T - 1)
    if width >= 3:
        d1 = librosa.feature.delta(M, width=width, order=1)
        d2 = librosa.feature.delta(M, width=width, order=2)
    else:
        d1 = d2 = np.full_like(M, np.nan)
    regions = {"sp": b.vad_mask[:T], "full": np.ones(T, dtype=bool)}
    n_mfcc = M.shape[0]
    for region, mask in regions.items():
        ok = mask.sum() >= min_frames
        Mr, D1, D2 = M[:, mask], d1[:, mask], d2[:, mask]
        for k in range(n_mfcc):
            out[f"m_{region}_mfcc{k}_mean"] = float(Mr[k].mean()) if ok else np.nan
            out[f"m_{region}_mfcc{k}_std"] = float(Mr[k].std(ddof=1)) if ok else np.nan
            if region == "sp":
                out[f"m_sp_dmfcc{k}_std"] = float(np.nanstd(D1[k], ddof=1)) if ok else np.nan
                out[f"m_sp_ddmfcc{k}_std"] = float(np.nanstd(D2[k], ddof=1)) if ok else np.nan
        if region != "sp":
            continue
        C = Mr[1:] if ok else np.zeros((n_mfcc - 1, 0))
        if ok and C.shape[1] >= max(min_frames, 3):
            out["m_sp_cepstral_variability"] = float(C.std(axis=1, ddof=1).mean())
            steps = np.linalg.norm(np.diff(C, axis=1), axis=0)
            dur = C.shape[1] * b.hop / b.sr
            out["m_sp_cepstral_traj_length_per_s"] = float(steps.sum() / dur)
            net = np.linalg.norm(C[:, -1] - C[:, 0])
            out["m_sp_cepstral_traj_tortuosity"] = float(steps.sum() / (net + 1e-6))
            cov = np.cov(C) + 1e-6 * np.eye(C.shape[0])
            sign, logdet = np.linalg.slogdet(cov)
            out["m_sp_cov_logdet"] = float(logdet) if sign > 0 else np.nan
            out["m_sp_cov_trace"] = float(np.trace(cov))
            corr = np.corrcoef(C)
            off = corr[np.triu_indices_from(corr, 1)]
            out["m_sp_corr_mean_abs_offdiag"] = float(np.nanmean(np.abs(off)))
            lm = b.logmel[:, mask]
            a, bb = lm[:, :-1], lm[:, 1:]
            cos = (a * bb).sum(axis=0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(bb, axis=0) + EPS)
            out["m_sp_transition_rate"] = float(np.mean(1 - cos))
            out["m_sp_transition_rate_std"] = float(np.std(1 - cos, ddof=1)) if len(cos) > 1 else np.nan
        else:
            for key in ("m_sp_cepstral_variability", "m_sp_cepstral_traj_length_per_s", "m_sp_cepstral_traj_tortuosity",
                        "m_sp_cov_logdet", "m_sp_cov_trace", "m_sp_corr_mean_abs_offdiag", "m_sp_transition_rate",
                        "m_sp_transition_rate_std"):
                out[key] = np.nan
    return out


def stage_spectral_features(ctx) -> dict:
    from .family_analysis import analyze_family

    return analyze_family(ctx, family="SPECTRAL", prefixes=("s_full_", "s_sp_", "s_bg_", "m_"),
                          split_by_correctness=False, control_quality=True)
