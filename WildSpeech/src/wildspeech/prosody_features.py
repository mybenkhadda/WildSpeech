"""Section 10 - prosody, articulation and voice-conversion robustness.

All ``p_`` features describe the anonymised (voice-converted) signal. Pitch,
formant, jitter, shimmer and harmonicity measures are flagged a priori as
POTENTIALLY_VOICE_CONVERSION_SENSITIVE. Nothing here infers speaker identity,
age, sex or medical conditions, and no attempt is made to reverse the
anonymisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats_utils import robust_stats

EPS = 1e-12
PROSODY_KEYS = [
    "p_voiced_frac", "p_f0_median", "p_f0_mean", "p_f0_std_st", "p_f0_range_st", "p_f0_slope_st_per_s",
    "p_f0_transition_var_st", "p_f0_jump_frac", "p_jitter_local", "p_shimmer_local", "p_shimmer_db",
    "p_hnr_db_median", "p_f1_median", "p_f2_median", "p_f3_median", "p_f1_std", "p_f2_std",
    "p_f2_transition_rate", "p_f2_f1_ratio_median", "p_int_mean", "p_int_std", "p_int_range", "p_int_slope",
    "p_prosody_cv",
]


def f0_track(b, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    fmin, fmax = float(cfg["f0_min_hz"]), float(cfg["f0_max_hz"])
    frame_length = 1024 if b.sr >= 16000 else 512
    if len(b.y) < frame_length:
        return np.zeros(0), np.zeros(0, bool)
    if cfg.get("prosody_f0_method", "pyin") == "pyin":
        f0, voiced, _ = librosa.pyin(b.y, fmin=fmin, fmax=fmax, sr=b.sr, frame_length=frame_length,
                                     hop_length=b.hop, center=True)
        voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    else:
        f0 = librosa.yin(b.y, fmin=fmin, fmax=fmax, sr=b.sr, frame_length=frame_length, hop_length=b.hop, center=True)
        T = min(len(f0), len(b.vad_mask))
        voiced = np.zeros(len(f0), bool)
        voiced[:T] = b.vad_mask[:T]
    T = min(len(f0), len(b.vad_mask))
    voiced = voiced[:T] & b.vad_mask[:T]
    return np.asarray(f0[:T], dtype=float), voiced


def _formants(frame: np.ndarray, sr: int) -> tuple:
    import librosa

    if np.std(frame) == 0:
        return (np.nan, np.nan, np.nan)
    x = np.append(frame[0], frame[1:] - 0.97 * frame[:-1]) * np.hamming(len(frame))
    order = int(2 + sr / 1000)
    try:
        a = librosa.lpc(x, order=order)
    except Exception:
        return (np.nan, np.nan, np.nan)
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.01]
    if len(roots) == 0:
        return (np.nan, np.nan, np.nan)
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bws = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
    good = np.sort(freqs[(freqs > 90) & (bws < 400)])
    good = list(good[:3]) + [np.nan] * (3 - min(3, len(good)))
    return tuple(good[:3])


def prosody_features(b, cfg: dict) -> dict:
    out: dict = {k: np.nan for k in PROSODY_KEYS}
    f0, voiced = f0_track(b, cfg)
    n_speech = int(b.vad_mask.sum())
    hop_s = b.hop / b.sr
    if n_speech > 0 and len(voiced):
        out["p_voiced_frac"] = float(voiced.sum() / n_speech)
    sp_db = b.rms_db[b.vad_mask[:len(b.rms_db)]]
    if len(sp_db) >= 3:
        out["p_int_mean"] = float(sp_db.mean())
        out["p_int_std"] = float(sp_db.std(ddof=1))
        out["p_int_range"] = float(np.percentile(sp_db, 95) - np.percentile(sp_db, 5))
        t = np.flatnonzero(b.vad_mask[:len(b.rms_db)]) * hop_s
        out["p_int_slope"] = float(np.polyfit(t, sp_db, 1)[0]) if np.ptp(t) > 0 else np.nan
    if voiced.sum() >= 5:
        fv = f0[voiced]
        st = 12 * np.log2(fv / np.median(fv))
        out["p_f0_median"] = float(np.median(fv))
        out["p_f0_mean"] = float(fv.mean())
        out["p_f0_std_st"] = float(st.std(ddof=1))
        out["p_f0_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        tv = np.flatnonzero(voiced) * hop_s
        out["p_f0_slope_st_per_s"] = float(np.polyfit(tv, st, 1)[0]) if np.ptp(tv) > 0 else np.nan
        idx = np.flatnonzero(voiced)
        consecutive = np.diff(idx) == 1
        if consecutive.sum() >= 3:
            d = np.diff(st)[consecutive]
            out["p_f0_transition_var_st"] = float(d.std(ddof=1))
            out["p_f0_jump_frac"] = float((np.abs(d) > 2.0).mean())
            periods = 1.0 / fv
            dp = np.abs(np.diff(periods))[consecutive]
            out["p_jitter_local"] = float(dp.mean() / periods.mean())
            amps = np.array([np.max(np.abs(b.y[int(i * b.hop):int(i * b.hop + b.win)])) if int(i * b.hop) < len(b.y) else np.nan
                             for i in idx])
            amps = np.where(amps > 0, amps, np.nan)
            da = np.abs(np.diff(amps))[consecutive]
            ratio = np.abs(20 * np.log10(amps[1:] / amps[:-1]))[consecutive]
            out["p_shimmer_local"] = float(np.nanmean(da) / np.nanmean(amps)) if np.isfinite(amps).sum() > 3 else np.nan
            out["p_shimmer_db"] = float(np.nanmean(ratio)) if np.isfinite(ratio).any() else np.nan
        hnr = []
        rng = np.random.default_rng(0)
        sel = idx if len(idx) <= 100 else np.sort(rng.choice(idx, 100, replace=False))
        for i in sel:
            start = int(i * b.hop)
            seg = b.y[start:start + 2 * b.win].astype(float)
            lag = int(round(b.sr / f0[i])) if np.isfinite(f0[i]) and f0[i] > 0 else 0
            if lag <= 0 or len(seg) <= lag + 8 or np.std(seg) == 0:
                continue
            seg = seg - seg.mean()
            r = float((seg[:-lag] * seg[lag:]).sum() / np.sqrt((seg[:-lag] ** 2).sum() * (seg[lag:] ** 2).sum() + EPS))
            if 0 < r < 1:
                hnr.append(10 * np.log10(r / (1 - r)))
        out["p_hnr_db_median"] = float(np.median(hnr)) if hnr else np.nan
        max_frames = int(cfg["formant_max_frames"])
        fsel = idx if len(idx) <= max_frames else idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
        forms = np.array([_formants(b.y[int(i * b.hop):int(i * b.hop) + b.win].astype(float), b.sr)
                          for i in fsel if int(i * b.hop) + b.win <= len(b.y)], dtype=float)
        if len(forms) >= 3:
            out["p_f1_median"], out["p_f2_median"], out["p_f3_median"] = [float(np.nanmedian(forms[:, k])) if np.isfinite(forms[:, k]).any() else np.nan for k in range(3)]
            out["p_f1_std"] = float(np.nanstd(forms[:, 0], ddof=1)) if np.isfinite(forms[:, 0]).sum() > 2 else np.nan
            out["p_f2_std"] = float(np.nanstd(forms[:, 1], ddof=1)) if np.isfinite(forms[:, 1]).sum() > 2 else np.nan
            out["p_f2_transition_rate"] = float(np.nanmean(np.abs(np.diff(forms[:, 1])))) if np.isfinite(forms[:, 1]).sum() > 2 else np.nan
            ratio = forms[:, 1] / forms[:, 0]
            out["p_f2_f1_ratio_median"] = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else np.nan
        lin = 10 ** (sp_db / 20) if len(sp_db) else np.array([])
        cv_f0 = fv.std(ddof=1) / fv.mean() if fv.mean() > 0 else np.nan
        cv_int = lin.std(ddof=1) / lin.mean() if len(lin) > 2 and lin.mean() > 0 else np.nan
        out["p_prosody_cv"] = float(np.nanmean([cv_f0, cv_int]))
    return out


def anonymization_sensitivity(ctx, clip_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Classify features as content-related, voice-conversion- or artifact-sensitive."""
    from .families import feature_family, is_voice_prior
    from .stats_utils import icc1, safe_corr

    cfg = ctx.config
    artifact_cols = [c for c in ("q_snr_db_approx", "q_noise_floor_db", "q_eff_bandwidth_hz", "s_bg_level_db",
                                 "q_clip_ratio") if c in clip_df.columns]
    has_correct = "correct" in clip_df.columns and clip_df["correct"].notna().sum() > 50
    if has_correct:
        demeaned_correct = clip_df["correct"] - clip_df.groupby("item_key")["correct"].transform("mean")
    rows = []
    rng = np.random.default_rng(int(cfg["seed"]))
    sample = clip_df if len(clip_df) <= 50000 else clip_df.iloc[rng.choice(len(clip_df), 50000, replace=False)]
    for col in feature_cols:
        x = pd.to_numeric(sample[col], errors="coerce")
        fam = feature_family(col)
        item_icc = icc1(x, sample["item_key"]) if "item_key" in sample.columns else np.nan
        student_icc = icc1(x, sample["student_id"])
        art = [abs(safe_corr(x, sample[a], "spearman", 30)[0]) for a in artifact_cols if a != col]
        art_max = float(np.nanmax(art)) if art and np.isfinite(art).any() else np.nan
        corr_r = np.nan
        if has_correct:
            xd = x - x.groupby(sample["item_key"]).transform("mean")
            corr_r = safe_corr(xd, demeaned_correct.loc[sample.index], "pearson", 30)[0]
        if is_voice_prior(col):
            label, reason = "POTENTIALLY_VOICE_CONVERSION_SENSITIVE", "physiological voice measure (a priori)"
        elif fam in {"QUALITY", "BACKGROUND", "RECORDING_METADATA"} or (np.isfinite(art_max) and art_max >= cfg["sensitivity_artifact_corr"]):
            label, reason = "RECORDING_ARTIFACT_SENSITIVE", f"max |rho| with artifact indicators={art_max:.2f}"
        elif np.isfinite(student_icc) and student_icc > 0.5 and (not np.isfinite(item_icc) or item_icc < 0.05):
            label, reason = "POTENTIALLY_VOICE_CONVERSION_SENSITIVE", "high speaker consistency, little item dependence"
        elif (np.isfinite(item_icc) and item_icc >= cfg["sensitivity_item_icc"]) or (
                np.isfinite(corr_r) and abs(corr_r) >= cfg["sensitivity_correctness_r"]):
            label, reason = "LIKELY_CONTENT_RELATED", f"item ICC={item_icc:.2f}, within-item correctness r={corr_r:.2f}"
        else:
            label, reason = "UNDETERMINED", "no decisive evidence"
        rows.append({"feature": col, "family": fam, "item_icc": item_icc, "student_icc": student_icc,
                     "artifact_max_abs_rho": art_max, "within_item_correctness_r": corr_r,
                     "sensitivity": label, "reason": reason})
    return pd.DataFrame(rows)


def stage_prosody(ctx) -> dict:
    from . import data_access as da
    from . import visualization as viz
    from .family_analysis import analyze_family

    outputs: list = []
    summary: dict = {}
    if ctx.config.get("enable_prosody"):
        result = analyze_family(ctx, family="PROSODY", prefixes=("p_",), split_by_correctness=False, control_quality=True)
        outputs += result["outputs"]
        summary.update(result["summary"])
    clips = da.clip_table(ctx, "train")
    feats = [c for c in clips.columns if c.startswith(("q_", "t_", "s_", "m_", "w_", "p_", "ph_"))
             and pd.api.types.is_numeric_dtype(clips[c])]
    report = anonymization_sensitivity(ctx, clips, feats)
    ctx.save_df(report, "anonymization_sensitivity", also_csv=True)
    outputs.append("anonymization_sensitivity")
    counts = report.groupby(["family", "sensitivity"]).size().unstack(fill_value=0)
    fig, axes = viz.new_figure(1, 1, (9, 4.5))
    counts.plot(kind="bar", stacked=True, ax=axes[0, 0])
    viz.finalize_axis(axes[0, 0], "Anonymisation sensitivity by feature family", "family", "features", int(len(report)))
    ctx.figure(fig, "anonymization_sensitivity", "Anonymisation sensitivity report",
               "Which features remain useful after voice conversion?", "anonymization_sensitivity", int(len(report)),
               "; ".join(f"{k}={v}" for k, v in report["sensitivity"].value_counts().items()))
    summary["sensitivity_counts"] = report["sensitivity"].value_counts().to_dict()
    return {"outputs": outputs, "summary": summary}
