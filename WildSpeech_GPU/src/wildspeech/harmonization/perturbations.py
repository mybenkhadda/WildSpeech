"""Perturbation-stability with reuse (also used by V1's P08).

Expensive parts are computed once and cached:
  * the deterministic clip sample and its decoded waveforms (in memory for one run)
  * base features per (variant, extractor fingerprint)          -> cache/v2/perturbation/<key>.parquet
  * perturbed features per (variant, perturbation, fingerprint)  -> cache/v2/perturbation/<key>.parquet
Perturbations are applied in batches on the GPU (torchaudio phase-vocoder pitch shift, white noise, gain,
raised-cosine low-pass); the harmonization variant is applied AFTER the perturbation (the perturbation simulates
a recording condition, the variant is the analysis representation). Stability = Spearman(original, perturbed)
across clips for every feature (model-independent, computed once per cache entry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE
from ..io_utils import atomic_write_parquet, stable_hash
from .bandwidth import BandwidthHarmonizer, Variant


def pick_clips(names: pd.Series, n: int) -> list[str]:
    names = names.astype(str).sort_values().reset_index(drop=True)
    order = pd.util.hash_pandas_object(names, index=False).sort_values().index[:n]
    return list(names.iloc[order])


def perturb_batch(waves: list[np.ndarray], sr: int, name: str, device: str, seed: int = 0) -> tuple[list[np.ndarray], str]:
    """Vectorised perturbations; falls back to the V1 per-clip implementation for unknown names."""
    import torch

    from ..acoustic.perturb import apply_perturbation
    from ..v2.audio_pass import to_device_batch

    if name.startswith(("pitch_up_", "pitch_down_")):
        st = float(name.split("_")[-1].replace("st", "")) * (1 if name.startswith("pitch_up_") else -1)
        try:
            import torchaudio.functional as AF

            x, lengths = to_device_batch(waves, device)
            y = AF.pitch_shift(x, sr, n_steps=st)
            y = y[:, : x.shape[1]]
            yn = y.detach().float().cpu().numpy()
            return [yn[i, :n].astype(np.float32) for i, n in enumerate(lengths.cpu().numpy())], "torchaudio_phase_vocoder_gpu"
        except Exception:  # noqa: BLE001 - CPU fallback (resampling)
            out = [apply_perturbation(w, sr, name, seed + i) for i, w in enumerate(waves)]
            return [o[0] for o in out], out[0][1] if out else "resample_speed_change"
    if name.startswith("noise_"):
        snr = float(name.split("_")[1].replace("db", ""))
        rng = np.random.default_rng(seed)
        out = []
        for w in waves:
            noise = rng.standard_normal(len(w)).astype(np.float32)
            scale = np.sqrt(np.mean(w.astype(np.float64) ** 2) + 1e-12) / (10 ** (snr / 20.0)) / np.sqrt(np.mean(noise ** 2) + 1e-12)
            out.append((w + scale * noise).astype(np.float32))
        return out, "white_noise"
    if name.startswith("lowpass_"):
        cut = float(name.split("_")[1].replace("k", "")) * 1000.0
        return BandwidthHarmonizer(sr, device).apply(waves, Variant("p", cut)), "raised_cosine_lowpass"
    if name.startswith("gain_"):
        v = name.split("_", 1)[1].replace("db", "").replace("minus_", "-").replace("plus_", "")
        g = 10 ** (float(v) / 20.0)
        return [(w * g).astype(np.float32) for w in waves], "gain"
    out = [apply_perturbation(w, sr, name, seed + i) for i, w in enumerate(waves)]
    return [o[0] for o in out], out[0][1] if out else name


def stability(base: pd.DataFrame, pert: pd.DataFrame) -> pd.Series:
    """Spearman per column between aligned base / perturbed clip tables (vectorised)."""
    from scipy.stats import rankdata

    cols = [c for c in base.columns if c != FILE and c in pert.columns]
    a = base.set_index(FILE)[cols]
    b = pert.set_index(FILE).reindex(a.index)[cols]
    A, B = a.to_numpy(float), b.to_numpy(float)
    out = {}
    for j, c in enumerate(cols):
        m = np.isfinite(A[:, j]) & np.isfinite(B[:, j])
        if m.sum() < 8 or np.nanstd(A[m, j]) == 0 or np.nanstd(B[m, j]) == 0:
            out[c] = np.nan
            continue
        ra, rb = rankdata(A[m, j]), rankdata(B[m, j])
        out[c] = float(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(out)


class PerturbationFeatureCache:
    def __init__(self, ctx, feature_fn, feature_fp: str, tag: str):
        """feature_fn(waves) -> list[dict] (clip features); feature_fp identifies the extractor configuration."""
        self.ctx, self.fn, self.fp, self.tag = ctx, feature_fn, feature_fp, tag
        self.root = ctx.paths.v2_cache / "perturbation"
        self.root.mkdir(parents=True, exist_ok=True)
        self._waves: dict[str, np.ndarray] | None = None

    def _decode(self, clips: list[str], seconds: float) -> dict[str, np.ndarray]:
        if self._waves is None:
            from ..data.audio_io import decode_audio_bytes, make_reader
            from ..data.tar_manifest import build_tar_manifest

            man = build_tar_manifest(self.ctx.paths.files["train_audio"], self.ctx.paths.cache / "manifests", logger=self.ctx.logger)
            reader = make_reader(self.ctx.paths.files["train_audio"], man)
            if reader is None:
                raise RuntimeError("perturbation analysis needs an uncompressed TAR (random access)")
            waves = {}
            try:
                for f in clips:
                    if f in reader:
                        try:
                            waves[f] = decode_audio_bytes(reader.read(f), int(self.ctx.cfg.get_path("audio.sample_rate", 16000)), seconds)[0]
                        except Exception:  # noqa: BLE001 - undecodable clip is skipped
                            continue
            finally:
                reader.close()
            self._waves = waves
        return self._waves

    def _features(self, waves: dict[str, np.ndarray], transform=None, batch: int = 16) -> pd.DataFrame:
        names = sorted(waves)
        rows = []
        for i in range(0, len(names), batch):
            chunk = names[i:i + batch]
            ws = [waves[n] for n in chunk]
            if transform is not None:
                ws = transform(ws)
            for n, r in zip(chunk, self.fn(ws)):
                rows.append({FILE: n, **r})
        return pd.DataFrame(rows)

    def get(self, clips: list[str], perturbation: str | None, variant_apply=None, seconds: float = 10.0) -> tuple[pd.DataFrame, str]:
        key = stable_hash({"clips": clips, "p": perturbation, "fp": self.fp, "tag": self.tag, "s": seconds})
        path = self.root / f"{key}.parquet"
        meta_path = self.root / f"{key}.method"
        if path.exists():
            return pd.read_parquet(path), (meta_path.read_text(encoding="utf-8") if meta_path.exists() else "")
        waves = self._decode(clips, seconds)
        sr = int(self.ctx.cfg.get_path("audio.sample_rate", 16000))
        method = "none"
        if perturbation is not None:
            names = sorted(waves)
            pw, method = [], ""
            for i in range(0, len(names), 16):
                out, method = perturb_batch([waves[n] for n in names[i:i + 16]], sr, perturbation, self.ctx.device, seed=i)
                pw += out
            waves = dict(zip(names, pw))
        df = self._features(waves, variant_apply)
        atomic_write_parquet(df, path)
        meta_path.write_text(method, encoding="utf-8")
        return df, method


def stability_experiment(ctx, feature_fn, feature_fp: str, tag: str, clips: list[str], perturbations: list[str],
                         variant_apply=None, seconds: float = 10.0) -> pd.DataFrame:
    """Long table: feature, perturbation, method, stability_rho."""
    cache = PerturbationFeatureCache(ctx, feature_fn, feature_fp, tag)
    base, _ = cache.get(clips, None, variant_apply, seconds)
    rows = []
    for name in perturbations:
        pert, method = cache.get(clips, name, variant_apply, seconds)
        s = stability(base, pert)
        rows += [{"feature": f, "perturbation": name, "method": method, "stability_rho": v} for f, v in s.items()]
    return pd.DataFrame(rows)
