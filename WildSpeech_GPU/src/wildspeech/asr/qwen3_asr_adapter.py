"""drivendata/pasketti-word = Qwen3-ASR-1.7B fine-tune (Qwen3ASRForConditionalGeneration, model_type qwen3_asr).

It is NOT a Whisper / CTC checkpoint, and the card states that `pip install transformers` alone cannot load it.
The supported loader is the `qwen_asr` package with the vLLM backend (card: qwen_asr[vllm]==0.0.6, vllm==0.14.0):

    from qwen_asr import Qwen3ASRModel
    asr = Qwen3ASRModel.LLM(<local snapshot>, ...)
    results = asr.transcribe(audio=[(wave, 16000), ...], language="English")

vLLM has no native Windows build, so on the MITSUI Windows env this model is FRAMEWORK_INCOMPATIBLE and P24 skips
it with that reason. Run it from WSL2 / Linux with scripts/v3_pasketti_word_wsl.sh: same repository, same cache/v3/asr
output (the Parquet schema is identical), so P25-P29 on Windows consume it afterwards.

An EXPERIMENTAL transformers backend (Qwen3ASRModel.from_pretrained) exists in qwen_asr but the card says it is not
supported for this checkpoint; it is used only when v3.asr.qwen3_asr.allow_transformers_backend is true, and its rows
are tagged backend=qwen_asr_transformers.

Outputs: transcript (lower-cased words). The transcribe API exposes no token probabilities, so has_token_probs is
False and every probability-derived column is NaN — nothing is fabricated. Audio is passed in memory (never written
outside the machine). Long clips are chunked (<= 30 s, overlap) and merged.
"""
from __future__ import annotations

import gc
import inspect
import os
import time

import numpy as np

from .base import ASRResult, BaseASRAdapter, merge_transcripts, split_waveform

MAX_WINDOW_S = 30.0


def _filter_kwargs(fn, kw: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kw)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in sig.parameters}


def result_text(r) -> str:
    """qwen_asr result objects differ between versions: object with .text, dict with 'text', or a plain string."""
    if r is None:
        return ""
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        return str(r.get("text") or r.get("transcript") or "")
    for attr in ("text", "transcript"):
        v = getattr(r, attr, None)
        if v is not None:
            return str(v)
    return str(r)


class Qwen3ASRAdapter(BaseASRAdapter):
    output = "words"
    exposes_token_probs = False
    exposes_frame_posteriors = False

    def __init__(self, spec, source, load_kwargs, device, cfg):
        super().__init__(spec, source, load_kwargs, device, cfg)
        q = cfg.get_path("v3.asr.qwen3_asr", {}) or {}
        self.backend = "qwen_asr_transformers" if q.get("allow_transformers_backend") else "qwen_asr_vllm"
        self.llm_kwargs = dict(q.get("llm_kwargs") or {"gpu_memory_utilization": 0.80, "max_model_len": 4096})
        self.language = str(q.get("language", "English"))
        self.win_s = min(self.chunk_s, MAX_WINDOW_S)
        self.save_embeddings = False

    def load(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from qwen_asr import Qwen3ASRModel

        if self.backend == "qwen_asr_vllm":
            fn = Qwen3ASRModel.LLM
            self.model = fn(self.source, **_filter_kwargs(fn, self.llm_kwargs))
            self.dtype_name = "bf16"
        else:
            import torch

            fn = Qwen3ASRModel.from_pretrained
            kw = {"dtype": torch.bfloat16, "device_map": self.device if str(self.device).startswith("cuda") else "cpu"}
            self.model = fn(self.source, **_filter_kwargs(fn, kw))
            self.dtype_name = "bf16"
        self.n_params = int(float(self.spec.expected_params_millions or 0) * 1e6)   # vLLM hides the module tree

    def unload(self) -> None:
        m, self.model = self.model, None
        try:
            if m is not None and hasattr(m, "shutdown"):
                m.shutdown()
        except Exception:  # noqa: BLE001
            pass
        del m
        gc.collect()
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001 - not vLLM / already torn down
            pass
        super().unload()

    def set_fp32(self) -> None:          # the precision guard does not apply: no scores are exposed
        return None

    def to_cpu(self) -> None:
        raise RuntimeError("qwen3_asr: no CPU fallback (vLLM engine is GPU-only)")

    def version_info(self) -> dict:
        return {**super().version_info(), "backend": self.backend}

    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        t0 = time.perf_counter()
        pieces, owner, nch = [], [], []
        for i, w in enumerate(waves):
            ch, plan = split_waveform(np.asarray(w, np.float32), sr, self.win_s, self.overlap_s)
            pieces += ch
            owner += [i] * len(ch)
            nch.append(len(plan))
        fn = self.model.transcribe
        res = fn(**_filter_kwargs(fn, {"audio": [(p, sr) for p in pieces], "language": self.language}))
        res = list(res or [])
        if len(res) != len(pieces):
            raise RuntimeError(f"qwen3_asr returned {len(res)} results for {len(pieces)} audio pieces")
        texts: dict[int, list[str]] = {i: [] for i in range(len(waves))}
        for r, i in zip(res, owner):
            texts[i].append(result_text(r).strip())
        dt = (time.perf_counter() - t0) / max(len(waves), 1)
        out = []
        for i, w in enumerate(waves):
            out.append(ASRResult(transcript=merge_transcripts(texts[i]).lower(), raw_tokens="", token_logprobs=None,
                                 n_chunks=nch[i], audio_seconds=len(w) / sr, decode_seconds=dt,
                                 extra={"confidence_source": "unavailable", "backend": self.backend}))
        return out
