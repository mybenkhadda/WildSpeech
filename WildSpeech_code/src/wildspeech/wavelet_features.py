"""Section 09 - wavelet and multiscale signal discovery.

Discrete families (DWT, wavelet packets) and continuous families (CWT) are
kept strictly separate: a continuous wavelet (morl, mexh, cmor) is never
passed to a discrete transform and vice versa. Decomposition depth adapts to
signal length; CWT is limited by downsampling, a configurable number of
scales and a maximum analysed duration. Only interpretable summary features
are retained - no scalograms are stored.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
CWT_BANDS_HZ = [(0, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000)]
ENV_LEVELS = 5


def compatible_families(families: list) -> dict:
    import pywt

    discrete = set(pywt.wavelist(kind="discrete"))
    continuous = set(pywt.wavelist(kind="continuous"))
    out = {"discrete": [], "continuous": [], "rejected": []}
    for fam in families:
        base = fam.split("-")[0].rstrip("0123456789.")
        if fam in discrete:
            out["discrete"].append(fam)
        elif fam in continuous or (base in {"cmor", "shan", "fbsp"} and base in continuous):
            out["continuous"].append(fam)
        else:
            out["rejected"].append(fam)
    return out


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else np.nan


def _analysis_signal(b) -> np.ndarray:
    if b.segments:
        parts = [b.y[int(s * b.sr):int(e * b.sr)] for s, e in b.segments]
        x = np.concatenate(parts) if parts else b.y
        if len(x) >= int(0.2 * b.sr):
            return x.astype(float)
    return b.y.astype(float)


def dwt_features(x: np.ndarray, sr: int, family: str, max_level: int, detailed: bool) -> dict:
    import pywt
    from scipy.stats import kurtosis

    prefix = f"w_{family}"
    names = [f"{prefix}_D{j}_rel" for j in range(1, max_level + 1)] + [f"{prefix}_A_rel", f"{prefix}_entropy",
                                                                       f"{prefix}_transient_density", f"{prefix}_levels"]
    out = {n: np.nan for n in names}
    if detailed:
        for j in range(1, min(max_level, 6) + 1):
            out[f"{prefix}_D{j}_kurt"] = np.nan
            out[f"{prefix}_D{j}_logstd"] = np.nan
        for k in ("ratio_D1_D3", "ratio_D2_D4", "ratio_hi_lo", "xscale_consistency", "ms_variability", "hf_bursts_per_s"):
            out[f"{prefix}_{k}"] = np.nan
    w = pywt.Wavelet(family)
    level = min(max_level, pywt.dwt_max_level(len(x), w.dec_len))
    if level < 2:
        return out
    coeffs = pywt.wavedec(x, w, level=level)
    approx, details = coeffs[0], coeffs[1:][::-1]  # details[0] is D1 (finest)
    energies = np.array([np.sum(d ** 2) for d in details] + [np.sum(approx ** 2)])
    total = energies.sum() + EPS
    rel = energies / total
    for j in range(level):
        out[f"{prefix}_D{j + 1}_rel"] = float(rel[j])
    out[f"{prefix}_A_rel"] = float(rel[-1])
    out[f"{prefix}_entropy"] = _entropy(rel)
    out[f"{prefix}_levels"] = float(level)
    dur = len(x) / sr
    count = 0
    for d in details[:3]:
        mad = np.median(np.abs(d - np.median(d))) / 0.6745
        if mad > 0:
            count += int((np.abs(d) > 4 * mad).sum())
    out[f"{prefix}_transient_density"] = float(count / dur) if dur > 0 else np.nan
    if detailed:
        for j, d in enumerate(details[:6]):
            if len(d) >= 8 and np.std(d) > 0:
                out[f"{prefix}_D{j + 1}_kurt"] = float(kurtosis(d))
                out[f"{prefix}_D{j + 1}_logstd"] = float(np.log10(np.std(d) + EPS))
        e = energies
        if level >= 3:
            out[f"{prefix}_ratio_D1_D3"] = float(np.log10((e[0] + EPS) / (e[2] + EPS)))
        if level >= 4:
            out[f"{prefix}_ratio_D2_D4"] = float(np.log10((e[1] + EPS) / (e[3] + EPS)))
            out[f"{prefix}_ratio_hi_lo"] = float(np.log10((e[:2].sum() + EPS) / (e[3:level].sum() + EPS)))
        # cross-scale consistency: correlation of adjacent detail envelopes on a common grid
        n_grid = max(16, min(512, len(details[min(level - 1, 3)])))
        envs = []
        for d in details[:min(level, 5)]:
            env = np.abs(d)
            idx = np.linspace(0, len(env) - 1, n_grid)
            envs.append(np.interp(idx, np.arange(len(env)), env))
        cors = [np.corrcoef(envs[i], envs[i + 1])[0, 1] for i in range(len(envs) - 1)
                if np.std(envs[i]) > 0 and np.std(envs[i + 1]) > 0]
        out[f"{prefix}_xscale_consistency"] = float(np.mean(cors)) if cors else np.nan
        cvs = []
        for j, d in enumerate(details[:min(level, 5)]):
            win = max(4, int(0.05 * sr / (2 ** (j + 1))))
            nwin = len(d) // win
            if nwin >= 3:
                we = (d[:nwin * win].reshape(nwin, win) ** 2).sum(axis=1)
                if we.mean() > 0:
                    cvs.append(we.std(ddof=1) / we.mean())
        out[f"{prefix}_ms_variability"] = float(np.mean(cvs)) if cvs else np.nan
        d1 = np.abs(details[0])
        win = max(4, int(0.01 * sr / 2))
        nwin = len(d1) // win
        if nwin >= 5:
            we = d1[:nwin * win].reshape(nwin, win).mean(axis=1)
            thr = np.median(we) + 3 * (np.median(np.abs(we - np.median(we))) / 0.6745 + EPS)
            bursts = np.flatnonzero(np.diff((we > thr).astype(int)) == 1)
            out[f"{prefix}_hf_bursts_per_s"] = float(len(bursts) / dur) if dur > 0 else np.nan
    return out


def packet_features(x: np.ndarray, family: str, level_cfg: int) -> dict:
    import pywt

    n_nodes = 2 ** level_cfg
    out = {f"w_wp_node{k}_rel": np.nan for k in range(n_nodes)}
    out["w_wp_entropy"] = np.nan
    out["w_wp_level"] = np.nan
    w = pywt.Wavelet(family)
    level = min(level_cfg, pywt.dwt_max_level(len(x), w.dec_len))
    if level < 2:
        return out
    wp = pywt.WaveletPacket(data=x, wavelet=w, mode="symmetric", maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    e = np.array([np.sum(np.asarray(n.data) ** 2) for n in nodes])
    rel = e / (e.sum() + EPS)
    out["w_wp_entropy"] = _entropy(rel)
    out["w_wp_level"] = float(level)
    if level == level_cfg:
        for k, v in enumerate(rel):
            out[f"w_wp_node{k}_rel"] = float(v)
    return out


def cwt_features(x: np.ndarray, sr: int, wavelet: str, cwt_sr: int, n_scales: int, max_seconds: float) -> dict:
    import pywt
    from scipy.signal import resample_poly

    tag = wavelet.replace(".", "p").replace("-", "_")
    prefix = f"w_cwt_{tag}"
    out = {f"{prefix}_band{k}_rel": np.nan for k in range(len(CWT_BANDS_HZ))}
    out[f"{prefix}_entropy"] = np.nan
    out[f"{prefix}_temporal_cv"] = np.nan
    if len(x) < int(0.1 * sr):
        return out
    g = np.gcd(int(sr), int(cwt_sr))
    xd = resample_poly(x, int(cwt_sr) // g, int(sr) // g) if cwt_sr < sr else x
    xd = xd[: int(max_seconds * cwt_sr)]
    if len(xd) < 64:
        return out
    fc = pywt.central_frequency(wavelet)
    freqs = np.geomspace(40, min(2000, cwt_sr / 2 * 0.95), n_scales)
    scales = fc * cwt_sr / freqs
    coefs, pf = pywt.cwt(xd, scales, wavelet, sampling_period=1.0 / cwt_sr)
    power = np.abs(coefs) ** 2
    scale_energy = power.sum(axis=1)
    total = scale_energy.sum() + EPS
    for k, (lo, hi) in enumerate(CWT_BANDS_HZ):
        out[f"{prefix}_band{k}_rel"] = float(scale_energy[(pf >= lo) & (pf < hi)].sum() / total)
    out[f"{prefix}_entropy"] = _entropy(scale_energy / total)
    tp = power.sum(axis=0)
    out[f"{prefix}_temporal_cv"] = float(tp.std() / tp.mean()) if tp.mean() > 0 else np.nan
    return out


def envelope_wavelet_features(rms: np.ndarray) -> dict:
    import pywt

    out = {f"w_env_D{j}_rel": np.nan for j in range(1, ENV_LEVELS + 1)}
    out["w_env_entropy"] = np.nan
    if len(rms) < 16:
        return out
    w = pywt.Wavelet("db4")
    level = min(ENV_LEVELS, pywt.dwt_max_level(len(rms), w.dec_len))
    if level < 2:
        return out
    coeffs = pywt.wavedec(rms - rms.mean(), w, level=level)
    details = coeffs[1:][::-1]
    e = np.array([np.sum(d ** 2) for d in details])
    rel = e / (e.sum() + EPS)
    for j, v in enumerate(rel):
        out[f"w_env_D{j + 1}_rel"] = float(v)
    out["w_env_entropy"] = _entropy(rel)
    return out


def wavelet_features(b, cfg: dict) -> dict:
    x = _analysis_signal(b)
    fams = compatible_families(list(cfg["wavelet_families_dwt"]) + list(cfg.get("cwt_wavelets") or []))
    out: dict = {}
    max_level = int(cfg["wavelet_max_level"])
    primary = cfg["wavelet_primary_family"]
    for fam in fams["discrete"]:
        if fam in cfg["wavelet_families_dwt"]:
            out.update(dwt_features(x, b.sr, fam, max_level, detailed=(fam == primary)))
    if primary not in fams["discrete"] or primary not in cfg["wavelet_families_dwt"]:
        out.update(dwt_features(x, b.sr, primary, max_level, detailed=True))
    out.update(packet_features(x, cfg["wavelet_packet_family"], int(cfg["wavelet_packet_level"])))
    if cfg.get("enable_cwt"):
        for wav in fams["continuous"]:
            if wav in (cfg.get("cwt_wavelets") or []):
                out.update(cwt_features(x, b.sr, wav, int(cfg["cwt_sr"]), int(cfg["cwt_n_scales"]),
                                        float(cfg["cwt_max_seconds"])))
    out.update(envelope_wavelet_features(b.rms.astype(float)))
    return out


def stage_wavelet_features(ctx) -> dict:
    from .family_analysis import analyze_family, multiscale_comparisons

    result = analyze_family(ctx, family="WAVELETS", prefixes=("w_",), split_by_correctness=False, control_quality=True)
    multiscale_comparisons(ctx, prefixes=("w_",), outputs=result["outputs"])
    compat = compatible_families(list(ctx.config["wavelet_families_all"]))
    ctx.save_json(compat, "wavelet_family_compatibility")
    result["outputs"].append("wavelet_family_compatibility")
    return result
