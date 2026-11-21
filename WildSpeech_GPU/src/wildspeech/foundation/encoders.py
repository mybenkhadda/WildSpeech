"""Frozen encoders, compact embedding shards and the resumable GPU extraction runner.

FrozenSpeechEncoder  local HF checkpoint (whisper encoder or SSL encoder), fp16 on CUDA, all (or chosen) hidden
                     layers, valid-frame mask from true lengths, speech mask from energy at 50 frames/s.
EmbeddingShardStore  cache/v2/embeddings/<model>/<fingerprint>/<split>/shard-*.npy  (N, L, P, D) float16
                     + shard-*.files.json; incremental, resumable; slices one (layer, pooling) by memory-mapping.
EmbeddingV2Runner    AudioPass (duration-aware batches, OOM split, CPU fallback for single clips) per model.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import FILE
from ..io_utils import read_json, stable_hash, write_json
from ..model_registry import ModelNotEligible, load_registry, resolve_model_source
from .pooling import POOLINGS, lengths_to_mask, pool_layers, speech_mask_from_waves

VERSION = "emb-2.0.0"


class FrozenSpeechEncoder:
    def __init__(self, kind: str, source: str, load_kwargs: dict, device: str, fp16: bool = True, layers="all",
                 poolings=POOLINGS):
        import torch

        self.torch, self.kind, self.device = torch, kind, device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.poolings = tuple(poolings)
        if kind.startswith("whisper"):
            from transformers import WhisperFeatureExtractor, WhisperModel

            self.fe = WhisperFeatureExtractor.from_pretrained(source, **load_kwargs)
            self.model = WhisperModel.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).get_encoder().to(device).eval()
            n = int(self.model.config.encoder_layers)
        else:
            from transformers import AutoFeatureExtractor, AutoModel

            self.fe = AutoFeatureExtractor.from_pretrained(source, **load_kwargs)
            self.model = AutoModel.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).to(device).eval()
            n = int(self.model.config.num_hidden_layers)
        self.n_layers = n
        self.layers = list(range(n + 1)) if layers in ("all", None) else ([n // 2, n] if layers == "auto" else [int(x) for x in layers])

    def embed(self, waves: list[np.ndarray], sr: int = 16000) -> np.ndarray:
        """-> (B, L, P, D) float16."""
        torch = self.torch
        lens = np.array([len(w) for w in waves])
        with torch.no_grad():
            if self.kind.startswith("whisper"):
                feats = self.fe(waves, sampling_rate=sr, return_tensors="pt").input_features.to(self.device, self.dtype)
                hs = self.model(feats, output_hidden_states=True).hidden_states
                T = hs[0].shape[1]
                n_frames = np.minimum(np.ceil(lens / sr * 50).astype(int), T)
            else:
                inp = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
                vals = inp.input_values.to(self.device, self.dtype)
                use_mask = getattr(self.model.config, "feat_extract_norm", "group") == "layer"
                am = inp.attention_mask.to(self.device) if use_mask else None
                hs = self.model(vals, attention_mask=am, output_hidden_states=True).hidden_states
                T = hs[0].shape[1]
                n_frames = np.minimum(self.model._get_feat_extract_output_lengths(torch.tensor(lens)).cpu().numpy(), T)
            valid = lengths_to_mask(n_frames, T, self.device)
            speech = speech_mask_from_waves(waves, n_frames, T, sr, self.device) if "speech_mean" in self.poolings else None
            pooled = pool_layers(hs, self.layers, valid, speech, self.poolings)
        return pooled.cpu().numpy().astype(np.float16)


class EmbeddingShardStore:
    def __init__(self, root: Path, model: str, fingerprint: str):
        self.root = Path(root) / model / fingerprint
        self.model, self.fp = model, fingerprint

    def dir(self, split: str) -> Path:
        d = self.root / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shards(self, split: str) -> list[Path]:
        return sorted(Path(p) for p in glob.glob(str(self.dir(split) / "shard-*.npy")))

    def files_of(self, shard: Path) -> list[str]:
        return json.loads(shard.with_suffix(".files.json").read_text(encoding="utf-8"))

    def done(self, split: str) -> set[str]:
        out: set[str] = set()
        for s in self.shards(split):
            out.update(self.files_of(s))
        return out

    def failed(self, split: str) -> set[str]:
        return set(read_json(self.dir(split) / "_failures.json", []) or [])

    def add_failures(self, split: str, rows: list[dict]) -> None:
        write_json(self.dir(split) / "_failures.json", sorted(self.failed(split) | {r[FILE] for r in rows}))

    def append(self, split: str, rows: list[dict]) -> str:
        rows = [r for r in rows if r.get("emb") is not None]
        if not rows:
            return "-"
        arr = np.stack([r["emb"] for r in rows]).astype(np.float16)
        name = f"shard-{time.strftime('%Y%m%d-%H%M%S')}-{len(self.shards(split)):05d}"
        d = self.dir(split)
        tmp = d / f"{name}.partial"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        (d / f"{name}.files.json").write_text(json.dumps([r[FILE] for r in rows]), encoding="utf-8")
        tmp.replace(d / f"{name}.npy")
        return name

    def meta(self) -> dict:
        return read_json(self.root / "meta.json", {}) or {}

    def set_meta(self, meta: dict) -> None:
        write_json(self.root / "meta.json", meta)

    def load(self, layer_index: int, pooling_index: int, splits=("train",)) -> pd.DataFrame:
        """FILE + e0..e{D-1} for one (layer position, pooling position)."""
        files, mats = [], []
        for s in splits:
            for sh in self.shards(s):
                a = np.load(sh, mmap_mode="r")
                mats.append(np.asarray(a[:, layer_index, pooling_index, :], dtype=np.float32))
                files += self.files_of(sh)
        if not mats:
            return pd.DataFrame(columns=[FILE])
        X = np.concatenate(mats, 0)
        df = pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])
        df.insert(0, FILE, files)
        return df.drop_duplicates(FILE, keep="last").reset_index(drop=True)


def embedding_jobs(ctx) -> tuple[list[dict], pd.DataFrame]:
    f = ctx.cfg.get_path("v2.foundation", {}) or {}
    reg = load_registry(ctx.paths.configs)
    rows, jobs = [], []
    for key in f.get("models", []):
        m = reg.get(key, {})
        row = {"model": key, "hf_id": m.get("hf_id"), "kind": m.get("kind"), "license": m.get("license"),
               "child_speech": m.get("child_speech"), "license_verified": bool(m.get("license_verified")),
               "overlap_checked": bool(m.get("overlap_checked"))}
        try:
            src, kw = resolve_model_source(key, reg, ctx.cfg)
            rows.append({**row, "eligible": True, "reason": "verified"})
            jobs.append({"key": key, "kind": m.get("kind", "ssl_encoder"), "source": src, "kwargs": kw})
        except ModelNotEligible as e:
            rows.append({**row, "eligible": False, "reason": str(e)})
    return jobs, pd.DataFrame(rows)


def embedding_fingerprint(cfg, job: dict) -> str:
    f = cfg.get_path("v2.foundation", {}) or {}
    return stable_hash({"key": job["key"], "src": job["source"], "layers": f.get("layers"), "pool": f.get("poolings"),
                        "max_s": f.get("max_seconds"), "v": VERSION})


def open_store(ctx, job: dict) -> EmbeddingShardStore:
    return EmbeddingShardStore(ctx.paths.v2_cache / "embeddings", job["key"], embedding_fingerprint(ctx.cfg, job))


class EmbeddingV2Runner:
    def __init__(self, ctx):
        self.ctx, self.cfg, self.log = ctx, ctx.cfg, ctx.logger
        self.f = ctx.cfg.get_path("v2.foundation", {}) or {}
        self.sr = int(ctx.cfg.get_path("audio.sample_rate", 16000))

    def run(self, split: str) -> dict:
        from ..acoustic.batching import VRAMMonitor
        from ..v2.audio_pass import AudioPass

        jobs, elig = embedding_jobs(self.ctx)
        out = {"split": split, "models": {}, "eligibility": elig}
        cap = float(self.f.get("max_seconds", 20.0))
        bs = int(self.f.get("batch_size", 8))
        for job in jobs:
            store = open_store(self.ctx, job)
            ap = AudioPass(self.ctx, split, f"P20 {job['key']} {split}", cap, max_batch_size=bs, batch_seconds=bs * cap)
            cand = ap.clips(self.f.get("max_clips"))
            todo = sorted(set(cand[FILE]) - store.done(split) - store.failed(split))
            if not todo:
                out["models"][job["key"]] = {"status": "cached", "fingerprint": store.fp}
                continue
            enc = FrozenSpeechEncoder(job["kind"], job["source"], job["kwargs"], self.ctx.device,
                                      bool(self.f.get("fp16", True)), self.f.get("layers", "all"), tuple(self.f.get("poolings", POOLINGS)))
            store.set_meta({"layers": enc.layers, "poolings": list(enc.poolings), "n_layers": enc.n_layers,
                            "kind": job["kind"], "source": job["source"]})
            cpu_enc = {}

            def process(waves, names, enc=enc):
                E = enc.embed(waves, self.sr)
                return [{"emb": E[i]} for i in range(len(waves))]

            def cpu_fallback(waves, names, job=job):
                if "enc" not in cpu_enc:
                    cpu_enc["enc"] = FrozenSpeechEncoder(job["kind"], job["source"], job["kwargs"], "cpu", False,
                                                         self.f.get("layers", "all"), tuple(self.f.get("poolings", POOLINGS)))
                E = cpu_enc["enc"].embed(waves, self.sr)
                return [{"emb": E[i]} for i in range(len(waves))]

            res = ap.run(todo, process, lambda buf, store=store: store.append(split, buf),
                         lambda rows, store=store: store.add_failures(split, rows), cpu_fallback=cpu_fallback,
                         durations=cand.set_index(FILE)["duration_sec"])
            out["models"][job["key"]] = {"status": "done", "fingerprint": store.fp, "layers": enc.layers, **res}
            del enc
            cpu_enc.clear()
            VRAMMonitor.release()
        return out
