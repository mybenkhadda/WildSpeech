"""ASR output schema, long-recording chunking + stitching, precision selection and the one-model GPU slot.

Output row (one per clip and model) — `ASR_COLUMNS`:
  transcript            recognised words (word models) or "" (phone-only models)
  phones                recognised ARPAbet phones, space separated ("" when the model has no phone output)
  raw_tokens            the model's own token strings (IPA / characters / BPE), space separated
  token_logprobs        JSON list of per-token log-probabilities — ONLY if the model exposes them, else ""
  has_token_probs       bool; False means every probability-derived column is NaN (never fabricated)
  mean_token_logprob, min_token_logprob, token_entropy_mean, n_tokens
  frame_entropy_mean, blank_frac, greedy_ll          CTC models (NaN otherwise)
  expected_ll, expected_alignment, has_expected_alignment
                        likelihood of the EXPECTED response (teacher forcing / CTC forced alignment). This is a
                        conditional likelihood, NOT evidence that the expected sounds were produced.
  n_chunks, audio_seconds, truncated, decode_seconds, error
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

import numpy as np

ASR_COLUMNS = ["transcript", "phones", "raw_tokens", "token_logprobs", "has_token_probs", "mean_token_logprob",
               "min_token_logprob", "token_entropy_mean", "n_tokens", "frame_entropy_mean", "blank_frac", "greedy_ll",
               "expected_ll", "expected_alignment", "has_expected_alignment", "n_chunks", "audio_seconds", "truncated",
               "decode_seconds", "error"]


@dataclass
class ASRResult:
    transcript: str = ""
    phones: str = ""
    raw_tokens: str = ""
    token_logprobs: list[float] | None = None
    token_entropies: list[float] | None = None
    frame_entropy_mean: float = np.nan
    blank_frac: float = np.nan
    greedy_ll: float = np.nan
    expected_ll: float = np.nan
    expected_alignment: list[dict] | None = None
    n_chunks: int = 1
    audio_seconds: float = np.nan
    truncated: bool = False
    decode_seconds: float = np.nan
    error: str = ""
    embedding: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def row(self, save_token_probs: bool = True) -> dict:
        lp = [float(x) for x in (self.token_logprobs or []) if np.isfinite(x)]
        has = self.token_logprobs is not None and len(lp) > 0
        ent = [float(x) for x in (self.token_entropies or []) if np.isfinite(x)]
        out = {"transcript": self.transcript or "", "phones": self.phones or "", "raw_tokens": self.raw_tokens or "",
               "token_logprobs": json.dumps([round(x, 5) for x in lp]) if (has and save_token_probs) else "",
               "has_token_probs": bool(has),
               "mean_token_logprob": float(np.mean(lp)) if has else np.nan,
               "min_token_logprob": float(np.min(lp)) if has else np.nan,
               "token_entropy_mean": float(np.mean(ent)) if ent else np.nan,
               "n_tokens": float(len(lp)) if has else float(len((self.raw_tokens or "").split())),
               "frame_entropy_mean": float(self.frame_entropy_mean), "blank_frac": float(self.blank_frac),
               "greedy_ll": float(self.greedy_ll), "expected_ll": float(self.expected_ll),
               "expected_alignment": json.dumps(self.expected_alignment) if self.expected_alignment else "",
               "has_expected_alignment": bool(self.expected_alignment),
               "n_chunks": int(self.n_chunks), "audio_seconds": float(self.audio_seconds), "truncated": bool(self.truncated),
               "decode_seconds": float(self.decode_seconds), "error": self.error or ""}
        out.update({k: v for k, v in self.extra.items() if k not in out})
        return out


# ===================================================================================== chunking
@dataclass(frozen=True)
class Chunk:
    index: int
    start: int          # first sample of the chunk in the original waveform
    end: int            # one past the last sample
    core_start: int     # region of the chunk that "owns" these samples after stitching (midpoint rule)
    core_end: int


def plan_chunks(n_samples: int, sr: int, chunk_s: float, overlap_s: float) -> list[Chunk]:
    """Overlapping windows covering [0, n). Each sample belongs to exactly one chunk core; the boundary between
    consecutive cores is the midpoint of their overlap, so stitched outputs never duplicate or lose content."""
    n = int(n_samples)
    L = max(1, int(round(chunk_s * sr)))
    ov = int(round(overlap_s * sr))
    if ov >= L:
        raise ValueError("chunk overlap must be shorter than the chunk")
    if n <= L:
        return [Chunk(0, 0, n, 0, n)]
    step = L - ov
    starts = list(range(0, max(n - L, 0) + 1, step))
    if starts[-1] + L < n:
        starts.append(n - L)
    chunks = []
    for i, s in enumerate(starts):
        e = min(s + L, n)
        cs = 0 if i == 0 else (s + chunks[-1].end) // 2
        chunks.append(Chunk(i, s, e, cs, n))
        if i > 0:
            p = chunks[i - 1]
            chunks[i - 1] = Chunk(p.index, p.start, p.end, p.core_start, cs)
    return chunks


def split_waveform(wave: np.ndarray, sr: int, chunk_s: float, overlap_s: float) -> tuple[list[np.ndarray], list[Chunk]]:
    plan = plan_chunks(len(wave), sr, chunk_s, overlap_s)
    return [np.asarray(wave[c.start:c.end]) for c in plan], plan


def reconstruct_waveform(pieces: list[np.ndarray], plan: list[Chunk]) -> np.ndarray:
    """Inverse of split_waveform (keeps each chunk's core) — used to test that chunking is lossless."""
    parts = [p[c.core_start - c.start: c.core_end - c.start] for p, c in zip(pieces, plan)]
    return np.concatenate(parts) if parts else np.zeros(0)


def stitch_frames(frame_arrays: list[np.ndarray], plan: list[Chunk], samples_per_frame: int) -> np.ndarray:
    """Concatenate per-chunk frame-level outputs (e.g. CTC log-posteriors, (T_i, V)) keeping each chunk's core
    frames, so the stitched sequence is aligned to the original audio at the model frame rate."""
    out = []
    for a, c in zip(frame_arrays, plan):
        f0 = int(round((c.core_start - c.start) / samples_per_frame))
        f1 = int(round((c.core_end - c.start) / samples_per_frame))
        f1 = min(max(f1, f0), len(a))
        out.append(a[f0:f1])
    return np.concatenate(out, axis=0) if out else np.zeros((0,))


def merge_transcripts(texts: list[str], max_overlap_words: int = 8) -> str:
    """Join chunk transcripts, removing the longest duplicated word run at each boundary (overlap audio is
    transcribed twice by seq2seq models)."""
    words: list[str] = []
    for t in texts:
        w = str(t or "").split()
        if not words:
            words = w
            continue
        best = 0
        for k in range(min(max_overlap_words, len(words), len(w)), 0, -1):
            if [x.lower().strip(".,!?") for x in words[-k:]] == [x.lower().strip(".,!?") for x in w[:k]]:
                best = k
                break
        words += w[best:]
    return " ".join(words)


# ==================================================================================== precision
def select_dtype(device: str, precision: str = "auto"):
    """bf16 when the GPU supports it, else fp16 on CUDA; fp32 on CPU. Explicit 'fp32' / 'fp16' / 'bf16' honoured
    only when supported (unsupported requests fall back to fp32 and are reported)."""
    import torch

    if not str(device).startswith("cuda"):
        return torch.float32, "fp32"
    p = str(precision or "auto").lower()
    bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if p == "fp32":
        return torch.float32, "fp32"
    if p == "bf16":
        return (torch.bfloat16, "bf16") if bf16_ok else (torch.float32, "fp32(bf16 unsupported)")
    if p == "fp16":
        return torch.float16, "fp16"
    return (torch.bfloat16, "bf16") if bf16_ok else (torch.float16, "fp16")


def count_parameters(model) -> int:
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001 - non-torch model
        return 0


# ================================================================================ single GPU slot
class ModelSlot:
    """Process-wide guard: at most ONE ASR model is resident on the GPU at a time (8 GB RTX 3070)."""

    _lock = threading.Lock()
    _owner: str | None = None

    def __init__(self, key: str):
        self.key = key

    def __enter__(self):
        with ModelSlot._lock:
            if ModelSlot._owner is not None and ModelSlot._owner != self.key:
                raise RuntimeError(f"GPU slot busy: {ModelSlot._owner} is still loaded; unload it before {self.key}")
            ModelSlot._owner = self.key
        return self

    def __exit__(self, *exc):
        with ModelSlot._lock:
            ModelSlot._owner = None
        from ..acoustic.batching import VRAMMonitor

        VRAMMonitor.release()
        return False

    @classmethod
    def owner(cls) -> str | None:
        return cls._owner


class BaseASRAdapter:
    """Adapter protocol. Subclasses implement load(), unload(), transcribe_batch()."""

    output = "words"            # words | phones | chars
    exposes_token_probs = False
    exposes_frame_posteriors = False

    def __init__(self, spec, source: str, load_kwargs: dict, device: str, cfg):
        self.spec, self.source, self.load_kwargs, self.device, self.cfg = spec, source, dict(load_kwargs), device, cfg
        a = cfg.get_path("v3.asr", {}) or {}
        self.precision = getattr(spec, "precision", None) or a.get("precision", "auto")   # registry override wins
        self.chunk_s = float(a.get("chunk_seconds", 25.0))
        self.overlap_s = float(a.get("chunk_overlap_seconds", 2.0))
        self.save_embeddings = bool(a.get("save_embeddings", True))
        self.model = None
        self.dtype_name = "fp32"
        self.n_params = 0

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        from ..acoustic.batching import VRAMMonitor

        VRAMMonitor.release()

    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        raise NotImplementedError

    def version_info(self) -> dict:
        return {"source": self.source, "dtype": self.dtype_name, "n_params": self.n_params, "output": self.output,
                "token_probs": self.exposes_token_probs, "frame_posteriors": self.exposes_frame_posteriors}

    def set_fp32(self) -> None:
        """Precision guard: keep the device, switch weights + activations to fp32 for the rest of the run."""
        import torch

        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(dtype=torch.float32)
        self.dtype = torch.float32
        self.dtype_name = "fp32"

    def to_cpu(self) -> None:
        """Move the loaded model to CPU fp32 (single-clip OOM fallback); to_gpu() restores it."""
        import torch

        self._gpu_state = (self.device, getattr(self, "dtype", None), self.dtype_name)
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to("cpu", torch.float32)
        self.device = "cpu"
        self.dtype_name = "fp32"
        from ..acoustic.batching import VRAMMonitor

        VRAMMonitor.release()

    def to_gpu(self) -> None:
        st = getattr(self, "_gpu_state", None)
        if not st:
            return
        dev, dtype, name = st
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(dev, dtype) if dtype is not None else self.model.to(dev)
        self.device, self.dtype_name = dev, name
        self._gpu_state = None
