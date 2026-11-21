"""Section 06 - acoustic quality, signal atlas and the unified extraction pipeline.

Each recording is decoded exactly once. A shared ``SignalBundle`` (STFT,
power, mel, log-mel, MFCC, RMS envelope and voice-activity mask) feeds the
quality, temporal, spectral, cepstral, wavelet and prosody extractors.
Extraction runs in parallel chunks, writes each chunk atomically and resumes
after interruption. Failed recordings keep their row with an explicit status.

Voice activity detection marks speech-like regions only. It is not speaker
identification: speech-active frames may contain other voices.
"""
from __future__ import annotations

import gc
import hashlib
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import extraction_config, subset_hash, EXTRACTION_KEYS
from .io_utils import atomic_write_parquet

EPS = 1e-10


@dataclass
class SignalBundle:
    y: np.ndarray
    sr: int
    n_fft: int
    hop: int
    win: int
    duration: float
    S: np.ndarray
    power: np.ndarray
    freqs: np.ndarray
    mel: np.ndarray
    logmel: np.ndarray
    mfcc: np.ndarray
    rms: np.ndarray
    rms_db: np.ndarray
    vad_mask: np.ndarray
    segments: list = field(default_factory=list)
    vad_info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- decode
def decode_audio(source, target_sr: int) -> tuple[np.ndarray, dict]:
    """Decode a path or raw bytes to mono float32 at ``target_sr``."""
    meta: dict = {"decode_backend": None}
    data = None
    try:
        import soundfile as sf

        handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
        data, sr = sf.read(handle, dtype="float32", always_2d=True)
        meta["decode_backend"] = "soundfile"
    except Exception as first:
        import librosa

        path, tmp = source, None
        if isinstance(source, (bytes, bytearray)):
            fd, tmp = tempfile.mkstemp(suffix=".audio")
            with os.fdopen(fd, "wb") as fh:
                fh.write(source)
            path = tmp
        try:
            y, sr = librosa.load(str(path), sr=None, mono=False)
            data = (y[:, None] if y.ndim == 1 else y.T).astype(np.float32)
            meta["decode_backend"] = "librosa"
            meta["soundfile_error"] = f"{type(first).__name__}"
        finally:
            if tmp:
                os.unlink(tmp)
    meta["orig_sr"] = int(sr)
    meta["channels"] = int(data.shape[1])
    meta["n_samples_orig"] = int(data.shape[0])
    y = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    bad = ~np.isfinite(y)
    meta["n_invalid_values"] = int(bad.sum())
    if bad.any():
        y = np.where(bad, 0.0, y).astype(np.float32)
    if int(sr) != int(target_sr) and len(y) > 0:
        try:
            import librosa

            y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr), res_type="soxr_hq")
        except Exception:
            from scipy.signal import resample_poly

            g = np.gcd(int(sr), int(target_sr))
            y = resample_poly(y, int(target_sr) // g, int(sr) // g).astype(np.float32)
    meta["resampled"] = int(sr) != int(target_sr)
    return np.ascontiguousarray(y, dtype=np.float32), meta


# --------------------------------------------------------------------------------- VAD
def _hysteresis(values: np.ndarray, on: float, off: float) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    state = False
    for i, v in enumerate(values):
        if not state and v >= on:
            state = True
        elif state and v < off:
            state = False
        mask[i] = state
    return mask


def _clean_mask(mask: np.ndarray, hop_s: float, min_speech_s: float, min_silence_s: float) -> np.ndarray:
    mask = mask.copy()
    def runs(m, value):
        out, start = [], None
        for i, v in enumerate(np.append(m, not value)):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                out.append((start, i))
                start = None
        return out
    for s, e in runs(mask, False):
        if s > 0 and e < len(mask) and (e - s) * hop_s < min_silence_s:
            mask[s:e] = True
    for s, e in runs(mask, True):
        if (e - s) * hop_s < min_speech_s:
            mask[s:e] = False
    return mask


def energy_vad(rms_db: np.ndarray, hop_s: float, cfg: dict) -> tuple[np.ndarray, dict]:
    finite = rms_db[np.isfinite(rms_db)]
    if len(finite) == 0:
        return np.zeros(len(rms_db), bool), {"vad_status": "NO_FRAMES"}
    floor = float(np.percentile(finite, 10))
    peak = float(np.percentile(finite, 99))
    info = {"vad_backend": "energy", "vad_noise_floor_db": floor, "vad_peak_db": peak}
    if peak - floor < float(cfg["vad_min_dynamic_range_db"]):
        info["vad_status"] = "LOW_DYNAMIC_RANGE"
        return np.zeros(len(rms_db), bool), info
    on = max(floor + float(cfg["vad_energy_offset_db"]), peak - float(cfg["vad_max_below_peak_db"]))
    mask = _hysteresis(rms_db, on, on - 3.0)
    mask = _clean_mask(mask, hop_s, float(cfg["vad_min_speech_s"]), float(cfg["vad_min_silence_s"]))
    info.update({"vad_status": "OK", "vad_threshold_db": on})
    return mask, info


def webrtc_vad(y: np.ndarray, sr: int, n_frames: int, hop: int, cfg: dict) -> tuple[np.ndarray, dict]:
    import webrtcvad  # type: ignore

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError("webrtcvad requires 8/16/32/48 kHz audio")
    vad = webrtcvad.Vad(int(cfg["vad_webrtc_aggressiveness"]))
    frame = int(0.03 * sr)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    flags = []
    for start in range(0, len(pcm) - frame + 1, frame):
        flags.append(vad.is_speech(pcm[start:start + frame].tobytes(), sr))
    flags = np.array(flags, dtype=bool)
    times = (np.arange(n_frames) * hop) // frame
    mask = flags[np.clip(times, 0, max(len(flags) - 1, 0))] if len(flags) else np.zeros(n_frames, bool)
    mask = _clean_mask(mask, hop / sr, float(cfg["vad_min_speech_s"]), float(cfg["vad_min_silence_s"]))
    return mask, {"vad_backend": "webrtc", "vad_status": "OK"}


_SILERO = {}


def silero_local_vad(y: np.ndarray, sr: int, n_frames: int, hop: int, cfg: dict) -> tuple[np.ndarray, dict]:
    """Silero VAD from a user-supplied local TorchScript file (license verified by the user)."""
    import torch  # type: ignore

    if not cfg.get("vad_model_path"):
        raise ValueError("vad_model_path required for silero_local")
    path = str(cfg["vad_model_path"])
    if path not in _SILERO:
        _SILERO[path] = torch.jit.load(path, map_location="cpu").eval()
    model = _SILERO[path]
    window = 512 if sr == 16000 else 256
    probs = []
    with torch.no_grad():
        if hasattr(model, "reset_states"):
            model.reset_states()
        for start in range(0, len(y) - window + 1, window):
            chunk = torch.from_numpy(y[start:start + window].copy())
            probs.append(float(model(chunk, sr).item()))
    probs = np.array(probs)
    times = (np.arange(n_frames) * hop) // window
    mask = probs[np.clip(times, 0, max(len(probs) - 1, 0))] >= 0.5 if len(probs) else np.zeros(n_frames, bool)
    mask = _clean_mask(mask, hop / sr, float(cfg["vad_min_speech_s"]), float(cfg["vad_min_silence_s"]))
    return mask, {"vad_backend": "silero_local", "vad_status": "OK"}


def mask_to_segments(mask: np.ndarray, hop_s: float, duration: float) -> list:
    segs, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * hop_s, min(i * hop_s, duration)))
            start = None
    return [(float(s), float(e)) for s, e in segs if e > s]


# -------------------------------------------------------------------------- bundle
def compute_bundle(y: np.ndarray, sr: int, cfg: dict) -> SignalBundle:
    import librosa

    n_fft, hop, win = int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg["win_length"])
    duration = len(y) / sr
    yy = y if len(y) >= n_fft else np.pad(y, (0, n_fft - len(y)))
    S = np.abs(librosa.stft(yy, n_fft=n_fft, hop_length=hop, win_length=win, center=True))
    power = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=int(cfg["n_mels"]), fmax=sr / 2)
    logmel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=int(cfg["n_mfcc"]))
    rms = librosa.feature.rms(y=yy, frame_length=n_fft, hop_length=hop, center=True)[0]
    T = S.shape[1]
    rms = np.resize(rms, T) if len(rms) != T else rms
    rms_db = 20 * np.log10(np.maximum(rms, EPS))
    hop_s = hop / sr
    if cfg.get("enable_vad", True):
        backend = cfg.get("vad_backend", "energy")
        try:
            if backend == "webrtc":
                mask, info = webrtc_vad(yy, sr, T, hop, cfg)
            elif backend == "silero_local":
                mask, info = silero_local_vad(yy, sr, T, hop, cfg)
            else:
                mask, info = energy_vad(rms_db, hop_s, cfg)
        except Exception as exc:
            mask, info = energy_vad(rms_db, hop_s, cfg)
            info["vad_fallback_reason"] = f"{backend}: {type(exc).__name__}: {exc}"
    else:
        mask, info = np.ones(T, bool), {"vad_backend": "disabled", "vad_status": "DISABLED"}
    segments = mask_to_segments(mask, hop_s, duration)
    return SignalBundle(y=yy.astype(np.float32), sr=sr, n_fft=n_fft, hop=hop, win=win, duration=duration, S=S,
                        power=power, freqs=freqs, mel=mel, logmel=logmel, mfcc=mfcc, rms=rms, rms_db=rms_db,
                        vad_mask=mask, segments=segments, vad_info=info)


def quality_features(y: np.ndarray, b: SignalBundle, cfg: dict) -> dict:
    """Signal-quality descriptors. SNR and noise floor are explicitly approximate."""
    out: dict = {}
    dur = len(y) / b.sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    out["q_duration_s"] = dur
    out["q_peak"] = peak
    out["q_rms"] = rms
    out["q_rms_db"] = float(20 * np.log10(max(rms, EPS)))
    out["q_dc_offset"] = float(np.mean(y)) if len(y) else np.nan
    out["q_crest_factor"] = float(peak / rms) if rms > 0 else np.nan
    out["q_clip_ratio"] = float(np.mean(np.abs(y) >= float(cfg["clip_level_threshold"]))) if len(y) else np.nan
    out["q_near_zero_ratio"] = float(np.mean(np.abs(y) < 1e-4)) if len(y) else np.nan
    finite = b.rms_db[np.isfinite(b.rms_db)]
    out["q_dynamic_range_db"] = float(np.percentile(finite, 99) - np.percentile(finite, 5)) if len(finite) else np.nan
    out["q_noise_floor_db_approx"] = float(np.percentile(finite, 10)) if len(finite) else np.nan
    out["q_noise_floor_db"] = out["q_noise_floor_db_approx"]
    speech = b.rms_db[b.vad_mask[:len(b.rms_db)]]
    out["q_speech_level_db_approx"] = float(np.median(speech)) if len(speech) else np.nan
    out["q_snr_db_approx"] = (out["q_speech_level_db_approx"] - out["q_noise_floor_db_approx"]) if len(speech) else np.nan
    out["q_silence_ratio"] = float(1 - b.vad_mask.mean()) if len(b.vad_mask) else np.nan
    mean_spec = b.power.mean(axis=1) + EPS
    total = mean_spec.sum()
    for name, lo, hi in (("low", 0, 300), ("mid", 300, 3400), ("high", 3400, b.sr / 2 + 1)):
        out[f"q_band_energy_{name}_rel"] = float(mean_spec[(b.freqs >= lo) & (b.freqs < hi)].sum() / total)
    out["q_spectral_flatness"] = float(np.exp(np.mean(np.log(mean_spec))) / np.mean(mean_spec))
    cum = np.cumsum(mean_spec) / total
    out["q_eff_bandwidth_hz"] = float(b.freqs[np.argmax(cum >= 0.99)])
    out["q_flag_short"] = float(dur < 0.3)
    out["q_flag_silent"] = float(peak < 1e-4)
    out["q_flag_clipping"] = float(out["q_clip_ratio"] > 0.001) if np.isfinite(out["q_clip_ratio"]) else np.nan
    out["q_flag_low_snr"] = float(out["q_snr_db_approx"] < 10) if np.isfinite(out["q_snr_db_approx"]) else 1.0
    out["q_flag_dc"] = float(abs(out["q_dc_offset"]) > 0.01) if np.isfinite(out["q_dc_offset"]) else np.nan
    out["q_flag_narrowband"] = float(out["q_eff_bandwidth_hz"] < 0.45 * b.sr / 2)
    out["q_flag_no_speech"] = float(len(b.segments) == 0)
    flags = [v for k, v in out.items() if k.startswith("q_flag_") and np.isfinite(v)]
    out["q_n_flags"] = float(np.sum(flags))
    return out


def process_clip(task: dict, cfg: dict) -> tuple[dict, list]:
    """Decode one recording and compute all enabled feature families."""
    from .prosody_features import prosody_features
    from .spectral_features import mfcc_features, spectral_features
    from .temporal_features import temporal_features
    from .wavelet_features import wavelet_features

    t0 = time.perf_counter()
    rec: dict = {"clip_id": task["clip_id"], "proc_status": "OK", "proc_error": ""}
    segs: list = []
    try:
        source = task.get("bytes") if task.get("bytes") is not None else task["path"]
        y, meta = decode_audio(source, int(cfg["sample_rate"]))
        rec.update({f"meta_{k}": v for k, v in meta.items() if isinstance(v, (int, float, bool))})
        rec["meta_decode_backend"] = meta.get("decode_backend")
        if meta["n_invalid_values"]:
            rec["proc_status"] = "INVALID_VALUES_REPLACED"
        if len(y) == 0:
            rec["proc_status"] = "EMPTY"
            return rec, segs
        dur = len(y) / int(cfg["sample_rate"])
        if dur < float(cfg["min_audio_duration_s"]):
            rec["proc_status"] = "TOO_SHORT"
            rec["q_duration_s"] = dur
            return rec, segs
        if np.ptp(y) == 0:
            rec["proc_status"] = "CONSTANT"
            rec["q_duration_s"] = dur
            return rec, segs
        if dur > float(cfg["max_audio_duration_s"]):
            y = y[: int(float(cfg["max_audio_duration_s"]) * int(cfg["sample_rate"]))]
            rec["proc_status"] = "TRUNCATED_LONG"
        with np.errstate(all="ignore"):
            b = compute_bundle(y, int(cfg["sample_rate"]), cfg)
            segs = b.segments
            rec.update({f"vad_{k}" if not k.startswith("vad_") else k: v for k, v in b.vad_info.items()})
            rec.update(quality_features(y, b, cfg))
            if np.max(np.abs(y)) < 1e-5:
                rec["proc_status"] = "SILENT"
            rec.update(temporal_features(b, cfg))
            rec.update(spectral_features(b, cfg))
            rec.update(mfcc_features(b, cfg))
            if cfg.get("enable_wavelets"):
                rec.update(wavelet_features(b, cfg))
            if cfg.get("enable_prosody"):
                rec.update(prosody_features(b, cfg))
    except Exception as exc:
        rec["proc_status"] = "FAILED"
        rec["proc_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    rec["proc_seconds"] = time.perf_counter() - t0
    for k, v in list(rec.items()):
        if isinstance(v, (np.floating, np.integer)):
            rec[k] = v.item()
    return rec, segs


# --------------------------------------------------------------- unified extraction
def feature_version_hash(cfg: dict) -> str:
    return subset_hash(cfg, list(EXTRACTION_KEYS))


def _audio_sources(ctx, clips: pd.DataFrame, split: str) -> tuple[pd.DataFrame, str]:
    """Return clip rows with a file path (extracted) or the archive for streaming."""
    from . import audio_manifest as am
    from .environment import disk_free_bytes

    cfg = ctx.config
    archive = ctx.data_root() / ("train_audio.tar" if split == "train" else "test_audio.tar")
    dest = ctx.paths.extracted_train if split == "train" else ctx.paths.extracted_test
    extract = cfg["extract_train_audio"] if split == "train" else cfg["extract_test_audio"]
    members = clips["audio_member"].dropna().unique().tolist()
    if extract:
        manifest = ctx.load_df("audio_manifest")
        sizes = manifest[(manifest["archive"] == archive.name) & manifest["member"].isin(members)]
        report = am.safe_extract(archive, dest, members, disk_free_bytes(dest), float(cfg["disk_safety_factor"]),
                                 float(cfg["min_free_disk_gb"]), ctx.logger)
        ctx.logger.info("Extraction %s: %s (members %d, %.2f GB)", split, report["status"], len(members),
                        sizes["size_bytes"].sum() / 1e9)
        clips = clips.assign(path=clips["audio_member"].map(lambda m: str(dest / Path(*str(m).split("/"))) if pd.notna(m) else None))
        return clips, "file"
    return clips.assign(path=None), "stream"


def run_unified_extraction(ctx, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    from joblib import Parallel, delayed

    from . import audio_manifest as am
    from .environment import resolve_n_jobs
    from .logging_utils import progress

    cfg = ctx.config
    clips = ctx.load_df("clip_master")
    clips = clips[(clips["split"] == split) & clips["audio_member"].notna()].copy()
    fhash = feature_version_hash(cfg)
    cache_dir = ctx.paths.cache_dir("acoustic") / f"{split}_{fhash}"
    seg_dir = ctx.paths.cache_dir("vad") / f"{split}_{fhash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("overwrite_cache"):
        for p in list(cache_dir.glob("chunk_*.parquet")) + list(seg_dir.glob("chunk_*.parquet")):
            p.unlink()
    done: set = set()
    for p in sorted(cache_dir.glob("chunk_*.parquet")):
        try:
            done |= set(pd.read_parquet(p, columns=["clip_id"])["clip_id"])
        except Exception:
            ctx.logger.warning("Removing unreadable cache chunk %s", p)
            p.unlink()
    todo = clips[~clips["clip_id"].isin(done)]
    ctx.logger.info("Unified extraction %s: %d clips cached, %d to process", split, len(done), len(todo))
    if len(todo):
        todo, mode = _audio_sources(ctx, todo, split)
        # duration-aware scheduling: interleave long and short recordings across chunks
        if "audio_duration_s" in todo.columns:
            todo = todo.sort_values("audio_duration_s", ascending=False, na_position="last")
            order = np.argsort(np.arange(len(todo)) % max(1, int(np.ceil(len(todo) / int(cfg["extraction_chunk_size"])))), kind="stable")
            todo = todo.iloc[order]
        archive = ctx.data_root() / ("train_audio.tar" if split == "train" else "test_audio.tar")
        fcfg = extraction_config(cfg)
        n_jobs = resolve_n_jobs(int(cfg["n_jobs"]))
        size = int(cfg["extraction_chunk_size"])
        start_idx = len(list(cache_dir.glob("chunk_*.parquet")))
        for ci, start in enumerate(progress(range(0, len(todo), size), total=int(np.ceil(len(todo) / size)),
                                            desc=f"extract {split}")):
            chunk = todo.iloc[start:start + size]
            tasks = []
            if mode == "stream":
                blobs = am.read_member_bytes(archive, chunk["audio_member"].tolist())
                for _, r in chunk.iterrows():
                    tasks.append({"clip_id": r["clip_id"], "bytes": blobs.get(r["audio_member"]), "path": None})
            else:
                tasks = [{"clip_id": r["clip_id"], "path": r["path"], "bytes": None} for _, r in chunk.iterrows()]
            tasks = [t for t in tasks if t["bytes"] is not None or t["path"]]
            results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
                delayed(process_clip)(t, fcfg) for t in tasks)
            feats = pd.DataFrame([r for r, _ in results])
            seg_rows = [{"clip_id": r["clip_id"], "seg_index": i, "start_s": s, "end_s": e}
                        for (r, segs) in results for i, (s, e) in enumerate(segs)]
            idx = start_idx + ci
            atomic_write_parquet(pd.DataFrame(seg_rows, columns=["clip_id", "seg_index", "start_s", "end_s"]),
                                 seg_dir / f"chunk_{idx:05d}.parquet")
            atomic_write_parquet(feats, cache_dir / f"chunk_{idx:05d}.parquet")
            del results, feats, tasks
            gc.collect()
    parts = [pd.read_parquet(p) for p in sorted(cache_dir.glob("chunk_*.parquet"))]
    features = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=["clip_id"])
    features = features.drop_duplicates("clip_id", keep="last")
    features = features[features["clip_id"].isin(set(clips["clip_id"]))]
    id_cols = ["clip_id", "proc_status", "proc_error", "proc_seconds"]
    other = sorted(c for c in features.columns if c not in id_cols)
    features = features[[c for c in id_cols if c in features.columns] + other]
    sparts = [pd.read_parquet(p) for p in sorted(seg_dir.glob("chunk_*.parquet"))]
    segments = pd.concat(sparts, ignore_index=True) if sparts else pd.DataFrame(columns=["clip_id", "seg_index", "start_s", "end_s"])
    segments = segments[segments["clip_id"].isin(set(features["clip_id"]))].drop_duplicates(["clip_id", "seg_index"], keep="last")
    return features, segments


# ------------------------------------------------------------------------- atlas
def _anon(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:10]


def select_atlas_examples(ctx, clips: pd.DataFrame) -> pd.DataFrame:
    """Systematic selection across task, correctness, ability, difficulty, target, quality, residual."""
    from . import data_access as da
    from .stats_utils import quantile_bins

    cfg = ctx.config
    rng = np.random.default_rng(int(cfg["seed"]))
    df = clips[clips["proc_status"].isin(["OK", "INVALID_VALUES_REPLACED", "TRUNCATED_LONG"])].copy()
    if df.empty:
        return df
    sm = da.student_meta(ctx, "train")
    df = df.merge(sm[["target"]], left_on="student_id", right_index=True, how="left")
    scr = da.screening(ctx) if ctx.stage_available("03") else None
    if scr is not None and "theta_mean" in scr.columns:
        df = df.merge(scr[["theta_mean"]], left_on="student_id", right_index=True, how="left")
        df["stratum_ability"] = quantile_bins(df["theta_mean"], 3, 1, "A")
    if "item_b" in df.columns:
        df["stratum_difficulty"] = quantile_bins(df["item_b"], 3, 1, "D")
    df["stratum_target"] = quantile_bins(df["target"], 5, 1, "T")
    if "q_snr_db_approx" in df.columns:
        df["stratum_quality"] = quantile_bins(df["q_snr_db_approx"], 3, 1, "SNR")
    res = da.residuals(ctx)
    if res is not None:
        df["stratum_residual"] = quantile_bins(df["student_id"].map(res), 5, 1, "R")
    dims = ["task_family", "correct"] + [c for c in df.columns if c.startswith("stratum_")]
    chosen = []
    per = int(cfg["atlas_examples_per_stratum"])
    for dim in dims:
        if dim not in df.columns:
            continue
        for level, g in df.dropna(subset=[dim]).groupby(dim):
            pick = g.iloc[rng.choice(len(g), min(per, len(g)), replace=False)]
            chosen.append(pick.assign(atlas_dimension=dim, atlas_level=str(level)))
    if not chosen:
        return df.head(0)
    sel = pd.concat(chosen).drop_duplicates("clip_id")
    return sel.head(int(cfg["atlas_max_examples"]))


def render_atlas_example(ctx, row: pd.Series, source) -> None:
    import librosa
    import librosa.display  # noqa: F401

    from . import visualization as viz

    cfg = ctx.config
    y, _ = decode_audio(source, int(cfg["sample_rate"]))
    b = compute_bundle(y, int(cfg["sample_rate"]), cfg)
    sr, hop = b.sr, b.hop
    t = np.arange(len(b.y)) / sr
    tf = np.arange(b.S.shape[1]) * hop / sr
    fig, axes = viz.new_figure(5, 2, (7, 2.4))
    ax = axes.ravel()
    ax[0].plot(t, b.y, lw=0.4)
    viz.finalize_axis(ax[0], "Waveform", "time [s]", "amplitude")
    ax[1].plot(tf, b.rms_db, lw=0.8)
    ax[1].fill_between(tf, b.rms_db.min(), b.rms_db.max(), where=b.vad_mask, alpha=0.2, color="green", label="speech-active (VAD)")
    ax[1].legend(fontsize=6)
    viz.finalize_axis(ax[1], "Energy envelope and VAD mask", "time [s]", "RMS [dB]")
    for a, data, title, unit in ((ax[2], 20 * np.log10(b.S + EPS), "STFT magnitude", "dB"),
                                 (ax[3], np.sqrt(b.mel), "Mel spectrogram (sqrt power)", "a.u."),
                                 (ax[4], b.logmel, "Log-Mel spectrogram", "dB"),
                                 (ax[5], b.mfcc, "MFCC", "coef")):
        im = a.imshow(data, origin="lower", aspect="auto", extent=[0, tf[-1] if len(tf) else 1, 0, data.shape[0]])
        fig.colorbar(im, ax=a, fraction=0.04).set_label(unit)
        viz.finalize_axis(a, title, "time [s]", "bin")
    try:
        f0, voiced, _ = librosa.pyin(b.y, fmin=float(cfg["f0_min_hz"]), fmax=float(cfg["f0_max_hz"]), sr=sr,
                                     frame_length=1024, hop_length=hop)
        ax[6].plot(np.arange(len(f0)) * hop / sr, f0, ".", ms=2)
    except Exception:
        ax[6].text(0.5, 0.5, "pitch unavailable", ha="center", transform=ax[6].transAxes)
    viz.finalize_axis(ax[6], "Pitch contour (anonymised voice)", "time [s]", "F0 [Hz]")
    flux = np.concatenate([[0], np.sqrt((np.clip(np.diff(np.log1p(b.S), axis=1), 0, None) ** 2).sum(axis=0))])
    ax[7].plot(tf, flux, lw=0.7)
    viz.finalize_axis(ax[7], "Spectral flux", "time [s]", "flux")
    try:
        import pywt
        from scipy.signal import resample_poly

        xd = resample_poly(b.y, 1, max(1, sr // int(cfg["cwt_sr"])))[: int(float(cfg["cwt_max_seconds"]) * int(cfg["cwt_sr"]))]
        freqs = np.geomspace(40, int(cfg["cwt_sr"]) / 2 * 0.95, 48)
        scales = pywt.central_frequency("morl") * int(cfg["cwt_sr"]) / freqs
        coefs, pf = pywt.cwt(xd, scales, "morl", sampling_period=1 / int(cfg["cwt_sr"]))
        ax[8].imshow(np.log10(np.abs(coefs) ** 2 + EPS), aspect="auto", origin="upper",
                     extent=[0, len(xd) / int(cfg["cwt_sr"]), pf[-1], pf[0]])
    except Exception:
        ax[8].text(0.5, 0.5, "scalogram unavailable", ha="center", transform=ax[8].transAxes)
    viz.finalize_axis(ax[8], "Morlet wavelet scalogram", "time [s]", "frequency [Hz]")
    ax[9].axis("off")
    ax[9].text(0.0, 0.9, "\n".join([
        f"anonymous id: {_anon(row['clip_id'])}", f"task: {row.get('task_family')}",
        f"correct: {row.get('correct')}", f"stratum: {row.get('atlas_dimension')}={row.get('atlas_level')}",
        f"duration: {b.duration:.2f}s  segments: {len(b.segments)}",
        f"SNR(approx): {row.get('q_snr_db_approx', np.nan):.1f} dB"]), va="top", fontsize=8)
    ctx.figure(fig, f"atlas_{row['atlas_dimension']}_{row['atlas_level']}_{_anon(row['clip_id'])}",
               "Acoustic atlas example (training data, anonymous id)",
               "What information is available in an anonymised recording?", "clip_features", 1,
               f"Representative example for {row['atlas_dimension']}={row['atlas_level']}. Local use only.", sub="atlas")


def stage_acoustic_atlas(ctx) -> dict:
    from . import audio_manifest as am
    from . import visualization as viz

    cfg = ctx.config
    outputs = []
    feats, segs = run_unified_extraction(ctx, "train")
    ctx.save_df(feats, "clip_features")
    ctx.save_df(segs, "vad_segments")
    outputs += ["clip_features", "vad_segments"]
    if cfg.get("allow_test_inference") and cfg.get("extract_test_audio"):
        tf, tsegs = run_unified_extraction(ctx, "test")
        ctx.save_df(tf, "clip_features_test")
        ctx.save_df(tsegs, "vad_segments_test")
        outputs += ["clip_features_test", "vad_segments_test"]
    status = feats["proc_status"].value_counts().rename_axis("proc_status").reset_index(name="n")
    qcols = [c for c in feats.columns if c.startswith("q_")]
    summary = feats[qcols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index().rename(columns={"index": "feature"})
    flag_rates = feats[[c for c in qcols if c.startswith("q_flag_")]].mean().rename("rate").reset_index().rename(columns={"index": "flag"})
    ctx.save_df(status, "audio_processing_status", fmt="csv")
    ctx.save_df(summary, "audio_quality_summary", fmt="csv")
    ctx.save_df(flag_rates, "audio_quality_flag_rates", fmt="csv")
    outputs += ["audio_processing_status", "audio_quality_summary", "audio_quality_flag_rates"]
    show = [c for c in ("q_duration_s", "q_snr_db_approx", "q_noise_floor_db_approx", "q_clip_ratio",
                        "q_silence_ratio", "q_eff_bandwidth_hz") if c in feats.columns]
    if show:
        fig, axes = viz.new_figure(2, 3, (4.5, 3.2))
        for ax, c in zip(axes.ravel(), show):
            viz.hist_kde(ax, feats[c], c)
        ctx.figure(fig, "audio_quality_distributions", "Audio quality distributions (training)",
                   "How reliable is the acoustic information?", "clip_features", int(len(feats)),
                   "; ".join(f"{r.flag}={r.rate:.1%}" for r in flag_rates.itertuples()))
    if cfg.get("atlas_enabled"):
        cm = ctx.load_df("clip_master")
        cm = cm[cm["split"] == "train"]
        merged = cm.merge(feats, on="clip_id", how="inner")
        from . import data_access as da

        diff = da.item_difficulty(ctx)
        if diff is not None:
            merged = merged.merge(diff.rename(columns={"b": "item_b"})[["item_key", "item_b"]], on="item_key", how="left")
        examples = select_atlas_examples(ctx, merged)
        ctx.save_df(examples.assign(anonymous_id=examples["clip_id"].map(_anon))[
            ["anonymous_id", "atlas_dimension", "atlas_level", "task_family"]], "atlas_examples", fmt="csv")
        outputs.append("atlas_examples")
        archive = ctx.data_root() / "train_audio.tar"
        blobs = am.read_member_bytes(archive, examples["audio_member"].tolist()) if not cfg["extract_train_audio"] else {}
        for _, row in examples.iterrows():
            src = blobs.get(row["audio_member"]) if blobs else str(ctx.paths.extracted_train / Path(*str(row["audio_member"]).split("/")))
            try:
                render_atlas_example(ctx, row, src)
            except Exception as exc:
                ctx.logger.warning("Atlas example failed: %s", exc)
    return {"outputs": outputs, "summary": {"n_clips": int(len(feats)), "status": status.set_index("proc_status")["n"].to_dict(),
                                            "feature_version": feature_version_hash(cfg), "n_features": int(feats.shape[1])}}
