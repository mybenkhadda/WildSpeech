"""P04: speech foundation-model embeddings (wav2vec2 / HuBERT / WavLM / Whisper encoder), fully local.

Per clip: mask-aware mean pooling of selected hidden layers -> float32 columns e{layer}_{dim}.
Stored per model as feature family EMB_<model>. Dimensionality reduction (PCA) happens later INSIDE folds.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import FILE
from .acoustic.batching import VRAMMonitor, is_oom, plan_duration_batches
from .acoustic.store import FeatureStore
from .data.audio_io import PrefetchingBatchLoader, make_reader
from .data.tar_manifest import build_tar_manifest
from .io_utils import stable_hash
from .logging_utils import ResourceProgress
from .model_registry import ModelNotEligible, load_registry, resolve_model_source


class SpeechEncoder:
    def __init__(self, kind: str, source: str, load_kwargs: dict, device: str, fp16: bool, layers="auto"):
        import torch

        self.torch, self.kind, self.device = torch, kind, device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        if kind.startswith("whisper"):
            from transformers import WhisperFeatureExtractor, WhisperModel

            self.fe = WhisperFeatureExtractor.from_pretrained(source, **load_kwargs)
            self.model = WhisperModel.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).get_encoder().to(device).eval()
            n_layers = self.model.config.encoder_layers
        else:
            from transformers import AutoFeatureExtractor, AutoModel

            self.fe = AutoFeatureExtractor.from_pretrained(source, **load_kwargs)
            self.model = AutoModel.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).to(device).eval()
            n_layers = self.model.config.num_hidden_layers
        self.layers = [n_layers // 2, n_layers] if layers in (None, "auto") else [int(l) for l in layers]

    def embed(self, waves: list[np.ndarray], sr: int = 16000) -> np.ndarray:
        torch = self.torch
        lens = np.array([len(w) for w in waves])
        with torch.no_grad():
            if self.kind.startswith("whisper"):
                feats = self.fe(waves, sampling_rate=sr, return_tensors="pt").input_features.to(self.device, self.dtype)
                hs = self.model(feats, output_hidden_states=True).hidden_states
                n_frames = np.minimum(np.ceil(lens / sr * 50).astype(int), hs[0].shape[1])   # 50 encoder frames/s
            else:
                inp = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True,
                              return_attention_mask=True)
                vals = inp.input_values.to(self.device, self.dtype)
                use_mask = getattr(self.model.config, "feat_extract_norm", "group") == "layer"
                am = inp.attention_mask.to(self.device) if use_mask else None
                hs = self.model(vals, attention_mask=am, output_hidden_states=True).hidden_states
                n_frames = self.model._get_feat_extract_output_lengths(torch.tensor(lens)).cpu().numpy()
                n_frames = np.minimum(n_frames, hs[0].shape[1])
            T = hs[0].shape[1]
            m = (torch.arange(T, device=self.device)[None, :] < torch.tensor(n_frames, device=self.device)[:, None]).float()
            pooled = []
            for l in self.layers:
                h = hs[l].float()
                pooled.append((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1))
            return torch.cat(pooled, -1).cpu().numpy().astype(np.float32)


class EmbeddingRunner:
    def __init__(self, ctx):
        self.ctx, self.cfg, self.log = ctx, ctx.cfg, ctx.logger
        self.store = FeatureStore(ctx.paths.cache / "features")
        self.reg = load_registry(ctx.paths.configs)
        self.sr = int(self.cfg.get_path("audio.sample_rate", 16000))

    def run(self, split: str, model_keys: list[str] | None = None) -> dict:
        e = self.cfg.section("embeddings")
        keys = model_keys or list(e.get("models", []))
        bundle = self.ctx.bundle()
        clips = (bundle.train_clips if split == "train" else bundle.test_clips).drop_duplicates(FILE)
        if self.cfg.get_path("audio.max_clips"):
            clips = clips.sort_values(FILE).head(int(self.cfg.get_path("audio.max_clips")))
        man = build_tar_manifest(self.ctx.paths.files[f"{split}_audio"], self.ctx.paths.cache / "manifests", logger=self.log)
        reader = make_reader(self.ctx.paths.files[f"{split}_audio"], man)
        summary = {"split": split, "models": {}}
        cap = float(e.get("max_seconds", 20.0))
        try:
            for key in keys:
                fam = f"emb_{key}"
                try:
                    src, kw = resolve_model_source(key, self.reg, self.cfg)
                except ModelNotEligible as err:
                    self.log.warning("P04 %s skipped: %s", key, err)
                    summary["models"][key] = {"status": "not_eligible", "reason": str(err)}
                    continue
                fp = stable_hash({"model": key, "src": src, "layers": e.get("layers"), "max_s": cap, "v": "emb-1.0"})
                self.store.validate(fam, split, fp, True, self.log)
                todo = sorted((set(clips[FILE]) - self.store.done(fam, split) - self.store.failed(fam, split)) & set(man[FILE]))
                if not todo:
                    summary["models"][key] = {"status": "cached"}
                    continue
                enc = SpeechEncoder(self.reg[key].get("kind", "ssl_encoder"), src, kw, self.ctx.device,
                                    bool(self.cfg.get_path("gpu.use_fp16_embeddings", True)), e.get("layers", "auto"))
                bs = int(e.get("batch_size", 8))
                batches = plan_duration_batches(clips.set_index(FILE).loc[todo, "duration_sec"], bs, bs * cap, cap)
                loader = PrefetchingBatchLoader(reader, batches, self.sr, cap, int(self.cfg.get_path("runtime.decode_threads", 6)), 2)
                prog = ResourceProgress(len(todo), f"P04 {key} {split}")
                buf, fails, t0 = [], [], time.perf_counter()
                for b in loader:
                    fails += [{FILE: f, "stage": "decode", "error": er} for f, er in b.failures]
                    if b.waves:
                        embs = self._safe(enc, b.waves)
                        for f, v in zip(b.filenames, embs):
                            if v is None:
                                fails.append({FILE: f, "stage": "embed", "error": "model_failure"})
                            else:
                                buf.append({FILE: f, **{f"e{i}": float(x) for i, x in enumerate(v)}})
                    ck = None
                    if len(buf) >= int(self.cfg.get_path("audio.checkpoint_every", 1000)):
                        ck = self.store.append(fam, split, pd.DataFrame(buf))
                        buf = []
                    prog.update(len(b.filenames) + len(b.failures), failures=len(b.failures), checkpoint=ck)
                self.store.append(fam, split, pd.DataFrame(buf))
                self.store.add_failures(fam, split, fails)
                prog.close()
                summary["models"][key] = {"status": "done", "clips": len(todo), "failures": len(fails),
                                          "seconds": time.perf_counter() - t0, "layers": enc.layers}
                del enc
                VRAMMonitor.release()
        finally:
            if reader is not None:
                reader.close()
        return summary

    def _safe(self, enc: SpeechEncoder, waves: list[np.ndarray]) -> list:
        try:
            return list(enc.embed(waves, self.sr))
        except Exception as err:  # noqa: BLE001
            if is_oom(err):
                VRAMMonitor.release()
            if len(waves) == 1:
                self.log.warning("P04 embed failure: %s", err)
                return [None]
            mid = len(waves) // 2
            return self._safe(enc, waves[:mid]) + self._safe(enc, waves[mid:])
