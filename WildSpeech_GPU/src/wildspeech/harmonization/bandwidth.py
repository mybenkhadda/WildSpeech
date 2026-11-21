"""Bandwidth / amplitude harmonization variants and their resumable GPU extraction.

A variant = (low-pass cutoff, optional high-pass, optional active-speech-level normalisation). The transform is a
zero-phase FFT-domain raised-cosine filter applied to the padded batch on the GPU (extra zero padding prevents
circular wrap; samples beyond each clip's true length are re-zeroed). The transformed waveforms are then passed
through the *existing* V1 GPUAcousticExtractor with variant-specific parameters (mel fmax = cutoff, HF split =
cutoff/2), plus the robust family (normalization.RobustFeatureExtractor). Every (variant, family) has its own
fingerprinted FeatureStore under cache/v2/features/h_<variant>_<family>/<split>; variant 'orig' reuses the P02
caches in cache/features when `reuse_original_cache` is true.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .. import FILE
from ..acoustic.dsp import AcousticParams
from ..acoustic.extractor import FAMILY_PARAM_KEYS
from ..acoustic.store import FeatureStore
from ..io_utils import stable_hash

META_KEYS = ("true_duration_s", "decoded_duration_s", "cropped", "file_sr")
VERSION = "harm-2.0.0"


@dataclass(frozen=True)
class Variant:
    name: str
    lowpass_hz: float | None = None
    highpass_hz: float | None = None
    amp_norm: bool = False

    @property
    def is_identity(self) -> bool:
        return not self.lowpass_hz and not self.highpass_hz and not self.amp_norm

    def as_dict(self) -> dict:
        return {"name": self.name, "lowpass_hz": self.lowpass_hz, "highpass_hz": self.highpass_hz, "amp_norm": self.amp_norm}


def configured_variants(cfg) -> list[Variant]:
    h = cfg.get_path("v2.harmonization", {}) or {}
    allv = h.get("variants", {}) or {}
    active = h.get("active_variants") or list(allv)
    out = []
    for name in active:
        v = allv.get(name)
        if v is None:
            raise KeyError(f"active variant {name} not defined in v2.harmonization.variants")
        out.append(Variant(name, v.get("lowpass_hz"), v.get("highpass_hz"), bool(v.get("amp_norm", False))))
    return out


def variant_params(base: AcousticParams, v: Variant) -> AcousticParams:
    p = base
    if v.lowpass_hz:
        p = replace(p, fmax=float(min(p.fmax, v.lowpass_hz)), hf_split_hz=float(v.lowpass_hz) / 2.0)
    if v.highpass_hz:
        p = replace(p, fmin=float(max(p.fmin, v.highpass_hz)))
    return p


# =========================================================================== transforms
def raised_cosine_gain(freqs: np.ndarray, lowpass: float | None, highpass: float | None, transition: float) -> np.ndarray:
    g = np.ones_like(freqs, dtype=np.float64)
    t = max(float(transition), 1.0)
    if lowpass:
        lo, hi = lowpass - t / 2, lowpass + t / 2
        g = np.where(freqs >= hi, 0.0, g)
        m = (freqs > lo) & (freqs < hi)
        g[m] *= 0.5 * (1 + np.cos(np.pi * (freqs[m] - lo) / t))
    if highpass:
        lo, hi = highpass - t / 2, highpass + t / 2
        g = np.where(freqs <= lo, 0.0, g)
        m = (freqs > lo) & (freqs < hi)
        g[m] *= 0.5 * (1 - np.cos(np.pi * (freqs[m] - lo) / t))
    return g


def _fft_len(n: int) -> int:
    from scipy.fft import next_fast_len

    return int(next_fast_len(int(n)))


def bandlimit_numpy(y: np.ndarray, sr: int, lowpass: float | None, highpass: float | None = None,
                    transition: float = 250.0, guard: int = 2048) -> np.ndarray:
    """CPU reference of the GPU transform (used for parity checks)."""
    if not lowpass and not highpass:
        return np.asarray(y, np.float32)
    n = len(y)
    L = _fft_len(n + guard)
    Y = np.fft.rfft(np.asarray(y, np.float64), n=L)
    f = np.fft.rfftfreq(L, 1.0 / sr)
    out = np.fft.irfft(Y * raised_cosine_gain(f, lowpass, highpass, transition), n=L)[:n]
    return out.astype(np.float32)


def active_level_db(y: np.ndarray, sr: int, frame_s: float = 0.02) -> float:
    """Active speech level: RMS over the loudest half of 20 ms frames (robust to leading/trailing silence)."""
    hop = max(1, int(sr * frame_s))
    n = len(y) // hop
    if n < 1:
        return float(10 * np.log10(np.mean(np.asarray(y, np.float64) ** 2) + 1e-12))
    e = (np.asarray(y[: n * hop], np.float64).reshape(n, hop) ** 2).mean(1)
    top = np.sort(e)[n // 2:]
    return float(10 * np.log10(top.mean() + 1e-12))


def amp_normalize_numpy(y: np.ndarray, sr: int, target_dbfs: float = -26.0, peak_cap: float = 0.99) -> np.ndarray:
    g = 10 ** ((target_dbfs - active_level_db(y, sr)) / 20.0)
    out = np.asarray(y, np.float64) * g
    pk = np.abs(out).max() if len(out) else 0.0
    if pk > peak_cap:
        out *= peak_cap / pk
    return out.astype(np.float32)


class BandwidthHarmonizer:
    """Batched GPU version of bandlimit_numpy + amp_normalize_numpy."""

    def __init__(self, sr: int, device: str, transition: float = 250.0, target_dbfs: float = -26.0, guard: int = 2048):
        self.sr, self.device, self.transition, self.target_dbfs, self.guard = int(sr), device, float(transition), float(target_dbfs), int(guard)

    def apply(self, waves: list[np.ndarray], v: Variant) -> list[np.ndarray]:
        if v.is_identity:
            return [np.asarray(w, np.float32) for w in waves]
        import torch

        from ..v2.audio_pass import to_device_batch

        x, lengths = to_device_batch(waves, self.device)
        B, T = x.shape
        valid = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
        if v.lowpass_hz or v.highpass_hz:
            L = _fft_len(T + self.guard)
            X = torch.fft.rfft(x.float(), n=L)              # float32: consumer GPUs have very low FP64 throughput
            f = np.fft.rfftfreq(L, 1.0 / self.sr)
            g = torch.from_numpy(raised_cosine_gain(f, v.lowpass_hz, v.highpass_hz, self.transition).astype(np.float32)).to(x.device)
            x = torch.fft.irfft(X * g[None, :], n=L)[:, :T].float()
            x = x * valid
        if v.amp_norm:
            out = []
            xn = x.detach().cpu().numpy()
            for i, n in enumerate(lengths.cpu().numpy()):
                out.append(amp_normalize_numpy(xn[i, :n], self.sr, self.target_dbfs))
            return out
        xn = x.detach().cpu().numpy()
        return [xn[i, :n].astype(np.float32) for i, n in enumerate(lengths.cpu().numpy())]


# ================================================================== extraction runner
def family_store(ctx, original: bool = False) -> FeatureStore:
    return FeatureStore(ctx.paths.cache / "features") if original else FeatureStore(ctx.paths.v2_cache / "features")


def h_family(variant: str, family: str) -> str:
    return f"h_{variant}_{family}"


class HarmonizedExtractionRunner:
    def __init__(self, ctx):
        self.ctx, self.cfg, self.log = ctx, ctx.cfg, ctx.logger
        self.h = ctx.cfg.get_path("v2.harmonization", {}) or {}
        self.base = AcousticParams.from_cfg(ctx.cfg)
        self.store = family_store(ctx)
        self.orig_store = family_store(ctx, original=True)
        self.device = ctx.device
        self.harm = BandwidthHarmonizer(self.base.sr, self.device, float(self.h.get("transition_hz", 250)),
                                        float(self.h.get("amp_target_dbfs", -26.0)))
        self._ext: dict[str, object] = {}

    # ----------------------------------------------------------- bookkeeping
    def families(self) -> list[str]:
        fams = [f for f in self.h.get("families", ["quality", "temporal", "spectral", "mfcc", "mel"]) if f in FAMILY_PARAM_KEYS]
        return fams + (["robust"] if self.h.get("robust_family", True) else [])

    def reuse_orig(self, family: str, split: str) -> bool:
        return (bool(self.h.get("reuse_original_cache", True)) and family != "robust"
                and bool(self.orig_store.parts(family, split)))

    def fingerprint(self, v: Variant, family: str) -> str:
        p = variant_params(self.base, v).as_dict()
        keys = FAMILY_PARAM_KEYS.get(family, ["sr", "n_fft", "hop", "win", "n_mels", "n_mfcc", "fmin", "fmax",
                                              "vad_offset_db", "min_speech_s", "min_silence_s", "spectral_min_frames"])
        return stable_hash({"family": family, "variant": v.as_dict(), "params": {k: p[k] for k in keys},
                            "transition": self.h.get("transition_hz"), "amp": self.h.get("amp_target_dbfs"),
                            "max_s": self.h.get("max_clip_seconds"), "v": VERSION})

    def extractor(self, v: Variant, kind: str):
        key = f"{v.name}:{kind}"
        if key not in self._ext:
            p = variant_params(self.base, v)
            if kind == "gpu":
                from ..acoustic.gpu_features import GPUAcousticExtractor

                self._ext[key] = GPUAcousticExtractor(p, self.device)
            else:
                from .normalization import RobustFeatureExtractor

                self._ext[key] = RobustFeatureExtractor(p, self.device)
        return self._ext[key]

    # ------------------------------------------------------------------ run
    def plan(self, split: str, variants: list[Variant], max_clips: int | None) -> tuple[dict, pd.DataFrame]:
        from ..v2.audio_pass import AudioPass

        ap = AudioPass(self.ctx, split, f"P16 {split} harmonized", float(self.h.get("max_clip_seconds", 30.0)))
        cand = ap.clips(max_clips)
        todo: dict[tuple[str, str], set[str]] = {}
        for v in variants:
            for fam in self.families():
                if v.is_identity and self.reuse_orig(fam, split):
                    continue
                name = h_family(v.name, fam)
                fp = self.fingerprint(v, fam)
                self.store.validate(name, split, fp, True, self.log)
                self.store.set_fingerprint(name, split, fp, {"variant": v.as_dict(), "family": fam})
                left = set(cand[FILE]) - self.store.done(name, split) - self.store.failed(name, split)
                if left:
                    todo[(v.name, fam)] = left
        return todo, cand

    def run(self, split: str, variants: list[Variant] | None = None, max_clips: int | None = None) -> dict:
        from ..v2.audio_pass import AudioPass

        variants = variants or configured_variants(self.cfg)
        todo, cand = self.plan(split, variants, max_clips)
        need = sorted(set().union(*todo.values())) if todo else []
        summary = {"split": split, "jobs": {f"{v}/{f}": len(s) for (v, f), s in todo.items()}, "clips": len(need)}
        if not need:
            return summary
        vmap = {v.name: v for v in variants}
        jobs = sorted(todo)

        def process(waves, names):
            rows = [dict() for _ in waves]
            for vname in sorted({v for v, _ in jobs}):
                v = vmap[vname]
                fams = [f for (vv, f) in jobs if vv == vname]
                want = [i for i, n in enumerate(names) if any(n in todo[(vname, f)] for f in fams)]
                if not want:
                    continue
                tw = self.harm.apply([waves[i] for i in want], v)
                gpu_f = [f for f in fams if f != "robust"]
                res = self.extractor(v, "gpu").extract_batch(tw, gpu_f) if gpu_f else {}
                rob = self.extractor(v, "robust").extract_batch(tw) if "robust" in fams else None
                for k, i in enumerate(want):
                    for f in gpu_f:
                        rows[i][h_family(vname, f)] = res[f][k]
                    if rob is not None:
                        rows[i][h_family(vname, "robust")] = rob[k]
            return rows

        def flush(buf):
            name = "-"
            for (vname, fam) in jobs:
                fam_name = h_family(vname, fam)
                recs = [{FILE: r[FILE], **r[fam_name]} for r in buf if fam_name in r and r[FILE] in todo[(vname, fam)]]
                if recs:
                    name = self.store.append(fam_name, split, pd.DataFrame(recs))
            return name

        def record_failures(rows):
            for (vname, fam) in jobs:
                self.store.add_failures(h_family(vname, fam), split, [r for r in rows if r[FILE] in todo[(vname, fam)]])

        ap = AudioPass(self.ctx, split, f"P16 {split} harmonized ({len(jobs)} family jobs)",
                       float(self.h.get("max_clip_seconds", 30.0)))
        out = ap.run(need, process, flush, record_failures, durations=cand.set_index(FILE)["duration_sec"])
        summary.update(out)
        return summary


def load_variant_features(ctx, variant: str, families: list[str], splits=("train",)) -> pd.DataFrame:
    """Wide clip table FILE + 'FAMILY::feature' for one variant (variant prefix removed for comparability)."""
    h = ctx.cfg.get_path("v2.harmonization", {}) or {}
    v2_store, orig_store = family_store(ctx), family_store(ctx, original=True)
    out = None
    for fam in families:
        parts = []
        for s in splits:
            use_orig = variant == "orig" and fam != "robust" and bool(h.get("reuse_original_cache", True)) and orig_store.parts(fam, s)
            df = orig_store.load(fam, s, prefix=False) if use_orig else v2_store.load(h_family(variant, fam), s, prefix=False)
            if not df.empty:
                parts.append(df)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates(FILE, keep="last")
        df = df.drop(columns=[c for c in df.columns if c in META_KEYS])
        df = df.rename(columns={c: f"{fam.upper()}::{c}" for c in df.columns if c != FILE})
        out = df if out is None else out.merge(df, on=FILE, how="outer")
    return out if out is not None else pd.DataFrame(columns=[FILE])
