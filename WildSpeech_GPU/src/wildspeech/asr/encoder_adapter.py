"""Frozen self-supervised speech encoders (wav2vec2-base, HuBERT-base, WavLM-base-plus): pooled embeddings only.

The model is loaded with transformers AutoModel (Wav2Vec2Model / HubertModel / WavLMModel; pre-training heads such
as the wav2vec2 quantizer are dropped) from the LOCAL snapshot, in eval mode under torch.no_grad — never trained.

Per clip:
  embedding     concatenation, over the registry `encoder_layers` (hidden_states index; 0 = CNN/projection
                output, L = last transformer layer), of the MEAN hidden state over VALID frames only. Valid frames
                come from `_get_feat_extract_output_lengths(waveform lengths)`; padding frames never enter a mean.
                Encoders whose config says feat_extract_norm="group" ignore attention masks, so their clips are
                run one by one (v3.asr.unbatched_group_norm); feat_extract_norm="layer" encoders are batched with
                the attention mask. The mode is read from each checkpoint's config at load time.
                Long clips are chunked with overlap; chunk means are combined weighted by valid frames.
  transcript / phones / token probabilities: none (encoders do not recognise speech; nothing is fabricated)
WavLM runs in fp32 (relative position bias overflows in half precision). Embedding layout is recorded per row
(`embedding_layers`, `embedding_dim`) and in the cache provenance.
"""
from __future__ import annotations

import time

import numpy as np

from .base import ASRResult, BaseASRAdapter, count_parameters, select_dtype, split_waveform


class FrozenEncoderAdapter(BaseASRAdapter):
    output = "embeddings"
    exposes_token_probs = False
    exposes_frame_posteriors = False

    def __init__(self, spec, source, load_kwargs, device, cfg):
        super().__init__(spec, source, load_kwargs, device, cfg)
        a = cfg.get_path("v3.asr", {}) or {}
        self.unbatched_gn = bool(a.get("unbatched_group_norm", True))
        self.layers_req = [int(x) for x in (getattr(spec, "encoder_layers", None) or [])]
        self.save_embeddings = True

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

        from .environment import HALF_UNSAFE_MODEL_TYPES

        self.torch = torch
        mcfg = AutoConfig.from_pretrained(self.source, **self.load_kwargs)
        prec = "fp32" if mcfg.model_type in HALF_UNSAFE_MODEL_TYPES else self.precision
        self.dtype, self.dtype_name = select_dtype(self.device, prec)
        self.fe = AutoFeatureExtractor.from_pretrained(self.source, **self.load_kwargs)
        self.model = AutoModel.from_pretrained(self.source, torch_dtype=self.dtype, **self.load_kwargs).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_params = count_parameters(self.model)
        nl = int(getattr(self.model.config, "num_hidden_layers", 12) or 12)
        self.layers = [x for x in (self.layers_req or [nl]) if 0 <= x <= nl] or [nl]
        self.group_norm = getattr(self.model.config, "feat_extract_norm", "group") == "group"
        self.hidden = int(getattr(self.model.config, "hidden_size", 768) or 768)

    def version_info(self) -> dict:
        return {**super().version_info(), "embedding_layers": self.layers, "embedding_dim": self.hidden * len(self.layers),
                "group_norm_unbatched": bool(self.group_norm and self.unbatched_gn)}

    # ------------------------------------------------------------------ forward
    def _pooled(self, waves: list[np.ndarray], sr: int) -> list[tuple[np.ndarray | None, int]]:
        torch = self.torch
        inp = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
        vals = inp.input_values.to(self.model.device, self.model.dtype)
        mask = inp.get("attention_mask")
        use_mask = getattr(self.model.config, "feat_extract_norm", "group") == "layer"
        with torch.no_grad():
            out = self.model(vals, attention_mask=mask.to(self.model.device) if (mask is not None and use_mask) else None,
                             output_hidden_states=True)
        hs = out.hidden_states
        lens = self.model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves])).tolist()
        res = []
        for i, L in enumerate(lens):
            n = int(min(L, hs[0].shape[1]))
            if n <= 0:
                res.append((None, 0))
                continue
            v = torch.cat([hs[k][i, :n].float().mean(0) for k in self.layers]).cpu().numpy().astype(np.float32)
            res.append((v, n))
        return res

    def _pooled_many(self, waves: list[np.ndarray], sr: int):
        if self.group_norm and self.unbatched_gn and len(waves) > 1:
            out = []
            for w in waves:
                out += self._pooled([w], sr)
            return out
        return self._pooled(waves, sr)

    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        t0 = time.perf_counter()
        pieces, owner, nch = [], [], []
        for i, w in enumerate(waves):
            ch, plan = split_waveform(np.asarray(w, np.float32), sr, self.chunk_s, self.overlap_s)
            pieces += ch
            owner += [i] * len(ch)
            nch.append(len(plan))
        pooled = self._pooled_many(pieces, sr)
        acc: dict[int, list] = {}
        for (v, n), i in zip(pooled, owner):
            if v is not None:
                acc.setdefault(i, []).append((v, n))
        dt = (time.perf_counter() - t0) / max(len(waves), 1)
        out = []
        for i, w in enumerate(waves):
            parts = acc.get(i, [])
            emb, err = None, ""
            if parts:
                V = np.stack([v for v, _ in parts])
                W = np.array([n for _, n in parts], np.float64)
                emb = (V * W[:, None]).sum(0) / max(W.sum(), 1e-9)
                if not np.all(np.isfinite(emb)):
                    err, emb = "embedding_nonfinite", None
            else:
                err = "no_valid_frames"
            r = ASRResult(n_chunks=nch[i], audio_seconds=len(w) / sr, decode_seconds=dt, embedding=emb, error=err)
            r.extra.update({"embedding_dim": int(len(emb)) if emb is not None else 0,
                            "embedding_layers": ",".join(map(str, self.layers)),
                            "n_frames": int(sum(n for _, n in parts))})
            out.append(r)
        return out
