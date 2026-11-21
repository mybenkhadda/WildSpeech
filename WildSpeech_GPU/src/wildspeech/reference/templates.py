"""Frame sequences and training-fold exemplar templates for reference-conditioned modelling.

SequenceExtractor   GPU: STFT at a 20 ms hop -> VAD (same algorithm as V1) -> utterance-CMVN MFCCs and
                    mean-normalised 24-band log-mel, trimmed to the speech span (+ margin), capped at max_frames.
                    Computed on the harmonized waveform variant chosen by P16 (default: band-limited).
SequenceStore       ragged float16 shards: cache/v2/sequences/<fingerprint>/<split>/shard-*.npz (resumable).
ExemplarBank        fitted on TRAINING-FOLD clips only: per item, up to K correct responses (deterministic hash
                    order, so different folds share most exemplars and pairwise DTW results are reused);
                    items with fewer than `min_correct` correct responses get no template.
SyntheticReferences optional, eligible locally generated reference speech (<dir>/<task>__<text>.wav), used as
                    extra exemplars when v2.reference.synthetic.enabled and license_verified are true.
"""
from __future__ import annotations

import glob
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import FILE, ID
from ..acoustic.dsp import EPS, AcousticParams, mel_filterbank
from ..harmonization.normalization import RobustFeatureExtractor
from ..io_utils import read_json, stable_hash, write_json

VERSION = "seq-2.0.0"


class SequenceExtractor(RobustFeatureExtractor):
    def __init__(self, params: AcousticParams, device: str = "cuda", n_mels_seq: int = 24, max_frames: int = 200,
                 margin: int = 5):
        super().__init__(params, device)
        self.melfb_seq = torch.from_numpy(mel_filterbank(params.sr, params.n_fft, n_mels_seq, params.fmin, params.fmax)).to(self.device, self.dtype)
        self.max_frames, self.margin = int(max_frames), int(margin)

    @torch.no_grad()
    def sequences(self, waves: list[np.ndarray]) -> list[dict]:
        from ..acoustic.gpu_features import masked_mean, masked_std

        x, lengths = self._pad(waves)
        P = self._stft_power(x)
        valid, speech, sel = self._masks(P, lengths)
        w = sel.to(self.dtype)
        mf = (10 * torch.log10(P @ self.melfb + EPS)) @ self.dct
        cm = (mf - masked_mean(mf, w).unsqueeze(1)) / masked_std(mf, w).clamp_min(1e-6).unsqueeze(1)
        lm = 10 * torch.log10(P @ self.melfb_seq + EPS)
        lm = lm - masked_mean(lm, w).unsqueeze(1)
        cm_n, lm_n = cm.float().cpu().numpy(), lm.float().cpu().numpy()
        sp, va = speech.cpu().numpy(), valid.cpu().numpy()
        out = []
        for i in range(len(waves)):
            idx = np.flatnonzero(sp[i]) if sp[i].any() else np.flatnonzero(va[i])
            if len(idx) == 0:
                out.append(None)
                continue
            s = max(0, idx[0] - self.margin)
            e = min(int(va[i].sum()), idx[-1] + 1 + self.margin)
            e = min(e, s + self.max_frames)
            out.append({"mfcc": cm_n[i, s:e].astype(np.float16), "mel": lm_n[i, s:e].astype(np.float16),
                        "n_speech": int(sp[i].sum()), "start": int(s)})
        return out


class SequenceStore:
    KINDS = ("mfcc", "mel")

    def __init__(self, root: Path, fingerprint: str):
        self.root = Path(root) / fingerprint
        self.fp = fingerprint

    def dir(self, split: str) -> Path:
        d = self.root / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shards(self, split: str) -> list[Path]:
        return sorted(Path(p) for p in glob.glob(str(self.dir(split) / "shard-*.npz")))

    def done(self, split: str) -> set[str]:
        out: set[str] = set()
        for p in self.shards(split):
            with np.load(p, allow_pickle=False) as z:
                out.update(z["files"].astype(str).tolist())
        return out

    def failed(self, split: str) -> set[str]:
        return set(read_json(self.dir(split) / "_failures.json", []) or [])

    def add_failures(self, split: str, rows: list[dict]) -> None:
        cur = self.failed(split) | {r[FILE] for r in rows}
        write_json(self.dir(split) / "_failures.json", sorted(cur))

    def append(self, split: str, rows: list[dict]) -> str:
        rows = [r for r in rows if r.get("mfcc") is not None]
        if not rows:
            return "-"
        arrays = {"files": np.array([r[FILE] for r in rows])}
        for k in self.KINDS:
            mats = [r[k] for r in rows]
            arrays[f"{k}_data"] = np.concatenate(mats, 0).astype(np.float16)
            arrays[f"{k}_off"] = np.cumsum([0] + [len(m) for m in mats]).astype(np.int64)
        arrays["n_speech"] = np.array([r["n_speech"] for r in rows], dtype=np.int32)
        name = f"shard-{time.strftime('%Y%m%d-%H%M%S')}-{len(self.shards(split)):05d}.npz"
        path = self.dir(split) / name
        tmp = path.with_name(path.name + ".partial")          # not matched by the shard glob until complete
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        tmp.replace(path)
        return name

    def load(self, splits=("train",)) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {k: {} for k in self.KINDS}
        for s in splits:
            for p in self.shards(s):
                with np.load(p, allow_pickle=False) as z:
                    files = z["files"].astype(str)
                    for k in self.KINDS:
                        data, off = z[f"{k}_data"], z[f"{k}_off"]
                        for i, f in enumerate(files):
                            out[k][f] = data[off[i]:off[i + 1]]
        return out


def sequence_fingerprint(cfg, variant) -> str:
    r = cfg.get_path("v2.reference", {}) or {}
    return stable_hash({"variant": variant.as_dict(), "hop": r.get("frame_hop"), "n_mfcc": r.get("n_mfcc"),
                        "n_mels": r.get("n_mels"), "max_frames": r.get("max_frames"), "margin": r.get("trim_margin_frames"),
                        "audio": {k: cfg.get_path(f"audio.{k}") for k in ("n_fft", "win_length", "n_mels", "fmin", "fmax",
                                                                          "vad_offset_db", "min_speech_s", "min_silence_s")},
                        "v": VERSION})


class SequenceExtractionRunner:
    def __init__(self, ctx, variant):
        from ..harmonization.bandwidth import BandwidthHarmonizer, variant_params

        self.ctx, self.variant = ctx, variant
        self.r = ctx.cfg.get_path("v2.reference", {}) or {}
        base = AcousticParams.from_cfg(ctx.cfg)
        p = replace(variant_params(base, variant), hop=int(self.r.get("frame_hop", 320)), n_mfcc=int(self.r.get("n_mfcc", 13)))
        self.params = p
        self.fp = sequence_fingerprint(ctx.cfg, variant)
        self.store = SequenceStore(ctx.paths.v2_cache / "sequences", self.fp)
        h = ctx.cfg.get_path("v2.harmonization", {}) or {}
        self.harm = BandwidthHarmonizer(base.sr, ctx.device, float(h.get("transition_hz", 250)), float(h.get("amp_target_dbfs", -26)))
        self._ext = None

    @property
    def ext(self) -> SequenceExtractor:
        if self._ext is None:
            self._ext = SequenceExtractor(self.params, self.ctx.device, int(self.r.get("n_mels", 24)),
                                          int(self.r.get("max_frames", 200)), int(self.r.get("trim_margin_frames", 5)))
        return self._ext

    def run(self, split: str) -> dict:
        from ..v2.audio_pass import AudioPass

        ap = AudioPass(self.ctx, split, f"P19 sequences {split}", float(self.ctx.cfg.get_path("v2.harmonization.max_clip_seconds", 30.0)))
        cand = ap.clips(self.r.get("max_clips"))
        todo = sorted(set(cand[FILE]) - self.store.done(split) - self.store.failed(split))

        def process(waves, names):
            return self.ext.sequences(self.harm.apply(waves, self.variant))

        res = ap.run(todo, process, lambda buf: self.store.append(split, buf),
                     lambda rows: self.store.add_failures(split, rows), durations=cand.set_index(FILE)["duration_sec"])
        return {"fingerprint": self.fp, **res}


class ExemplarBank:
    def __init__(self, k: int = 4, min_correct: int = 5, seed: int = 0):
        self.k, self.min_correct, self.seed = int(k), int(min_correct), int(seed)

    def fit(self, c_train: pd.DataFrame, available: set[str], synthetic: dict[str, list[str]] | None = None) -> "ExemplarBank":
        d = c_train[(c_train["item_score"] == 1) & c_train[FILE].isin(available)][[FILE, "item_id", ID]]
        h = pd.util.hash_pandas_object(d[FILE].astype(str) + f"#{self.seed}", index=False).to_numpy()
        d = d.assign(_h=h).sort_values(["item_id", "_h"], kind="mergesort")
        cnt = d.groupby("item_id")[FILE].count()
        ok = cnt[cnt >= self.min_correct].index
        self.ex_ = {it: list(g[FILE].head(self.k)) for it, g in d[d["item_id"].isin(ok)].groupby("item_id")}
        self.n_correct_ = cnt.to_dict()
        self.owner_ = dict(zip(d[FILE], d[ID].astype(str)))
        for it, files in (synthetic or {}).items():
            self.ex_.setdefault(it, [])
            self.ex_[it] = self.ex_[it] + [f for f in files if f not in self.ex_[it]]
        return self

    def exemplars(self, item_id: str, exclude_student: str | None = None) -> list[str]:
        ex = self.ex_.get(item_id, [])
        if exclude_student is None:
            return ex
        return [f for f in ex if self.owner_.get(f) != exclude_student]


def item_slug(item_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(item_id).lower().replace("::", "__")).strip("_")


class SyntheticReferences:
    """Pre-rendered, eligible local reference speech: <dir>/<item slug>.(wav|flac). Returns sequences keyed by a
    pseudo filename 'synthetic::<slug>' so they can be used as exemplars."""

    def __init__(self, ctx, extractor_runner: SequenceExtractionRunner):
        self.ctx, self.runner = ctx, extractor_runner
        s = ctx.cfg.get_path("v2.reference.synthetic", {}) or {}
        self.enabled = bool(s.get("enabled")) and bool(s.get("license_verified")) and bool(s.get("directory"))
        self.dir = Path(s.get("directory") or ".")

    def load(self, items: list[str]) -> tuple[dict[str, list[str]], dict[str, dict[str, np.ndarray]]]:
        if not self.enabled or not self.dir.exists():
            return {}, {k: {} for k in SequenceStore.KINDS}
        import soundfile as sf

        from ..data.audio_io import _resample

        sr_t = int(self.ctx.cfg.get_path("audio.sample_rate", 16000))
        mapping, seqs = {}, {k: {} for k in SequenceStore.KINDS}
        for it in items:
            slug = item_slug(it)
            path = next((p for p in (self.dir / f"{slug}.wav", self.dir / f"{slug}.flac") if p.exists()), None)
            if path is None:
                continue
            y, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = _resample(y.mean(1), sr, sr_t)
            s = self.runner.ext.sequences(self.runner.harm.apply([y], self.runner.variant))[0]
            if s is None:
                continue
            key = f"synthetic::{slug}"
            mapping[it] = [key]
            for k in SequenceStore.KINDS:
                seqs[k][key] = s[k]
        return mapping, seqs
