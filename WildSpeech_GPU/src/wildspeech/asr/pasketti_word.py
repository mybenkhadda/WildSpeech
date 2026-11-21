"""drivendata/pasketti-word — child-specific word recogniser.

The published checkpoint (revision 7756647, 2026-10-23) is a Qwen3-ASR-1.7B fine-tune
(config: Qwen3ASRForConditionalGeneration, model_type qwen3_asr). It is NOT loadable with the ordinary Whisper
adapter. The architecture is still read from the local snapshot rather than assumed:
  qwen3_asr   -> Qwen3ASRAdapter (qwen_asr + vLLM; transcripts, no token probabilities)
  hf_whisper  -> WhisperAdapter(output="words")   (only if DrivenData ever republishes a Whisper variant)
  hf_ctc      -> CTCAdapter(output="chars")
  nemo        -> NemoASRAdapter                    (transcripts; token confidences only if exposed)
  other       -> UnsupportedArchitecture -> NOT_EXECUTED
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import ASRResult, BaseASRAdapter, count_parameters, merge_transcripts, split_waveform
from .pasketti_phonetic import UnsupportedArchitecture


class NemoASRAdapter(BaseASRAdapter):
    """NVIDIA NeMo ASR checkpoint (.nemo), local restore only. Audio is passed as in-memory arrays; when the
    installed NeMo version only accepts file paths, clips are written as temporary 16-bit WAV files in the
    project cache (never outside the machine) and deleted after the batch."""

    output = "words"
    exposes_token_probs = False

    def __init__(self, spec, source, load_kwargs, device, cfg):
        super().__init__(spec, source, load_kwargs, device, cfg)
        self.tmp_dir = None

    def load(self) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        p = Path(self.source)
        nemo_file = p if p.suffix == ".nemo" else sorted(p.glob("*.nemo"))[0]
        self.model = ASRModel.restore_from(str(nemo_file), map_location=torch.device(self.device)).eval()
        self.n_params = count_parameters(self.model)
        self.dtype_name = "fp32"
        try:                                  # request per-token confidence where the decoder supports it
            from omegaconf import open_dict

            dc = self.model.cfg.decoding
            with open_dict(dc):
                if "confidence_cfg" in dc:
                    dc.confidence_cfg.preserve_token_confidence = True
            self.model.change_decoding_strategy(dc)
            self._conf = True
        except Exception:  # noqa: BLE001 - decoder without confidence support
            self._conf = False

    def _transcribe(self, pieces: list[np.ndarray], sr: int):
        try:
            return self.model.transcribe(audio=pieces, batch_size=len(pieces), return_hypotheses=True, verbose=False)
        except TypeError:
            import soundfile as sf

            d = Path(self.cfg.get_path("paths.cache_dir", "cache")) / "v3" / "nemo_tmp"
            d.mkdir(parents=True, exist_ok=True)
            paths = []
            for k, w in enumerate(pieces):
                fp = d / f"clip_{k:04d}.wav"
                sf.write(str(fp), np.asarray(w, np.float32), sr, subtype="PCM_16")
                paths.append(str(fp))
            try:
                return self.model.transcribe(paths, batch_size=len(paths), return_hypotheses=True)
            finally:
                for fp in paths:
                    Path(fp).unlink(missing_ok=True)

    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        t0 = time.perf_counter()
        pieces, owner, nch = [], [], []
        for i, w in enumerate(waves):
            ch, plan = split_waveform(np.asarray(w, np.float32), sr, self.chunk_s, self.overlap_s)
            pieces += ch
            owner += [i] * len(ch)
            nch.append(len(plan))
        hyps = self._transcribe(pieces, sr)
        if isinstance(hyps, tuple):
            hyps = hyps[0]
        texts = {i: [] for i in range(len(waves))}
        confs = {i: [] for i in range(len(waves))}
        for h, i in zip(hyps, owner):
            texts[i].append(getattr(h, "text", h if isinstance(h, str) else ""))
            tc = getattr(h, "token_confidence", None)
            if self._conf and tc is not None and len(tc):
                # NeMo confidences are probabilities in (0, 1]; stored as log-probabilities
                confs[i] += [float(np.log(max(float(c), 1e-12))) for c in tc]
        dt = (time.perf_counter() - t0) / max(len(waves), 1)
        out = []
        for i, w in enumerate(waves):
            out.append(ASRResult(transcript=merge_transcripts(texts[i]).lower(), raw_tokens="",
                                 token_logprobs=confs[i] if confs[i] else None, n_chunks=nch[i],
                                 audio_seconds=len(w) / sr, decode_seconds=dt,
                                 extra={"confidence_source": "nemo_token_confidence" if confs[i] else "unavailable"}))
        return out


def build_pasketti_word(spec, source, load_kwargs, device, cfg, architecture: str):
    if architecture == "qwen3_asr":
        from .qwen3_asr_adapter import Qwen3ASRAdapter

        return Qwen3ASRAdapter(spec, source, load_kwargs, device, cfg)
    if architecture == "hf_whisper":
        from .whisper_adapter import WhisperAdapter

        return WhisperAdapter(spec, source, load_kwargs, device, cfg, output="words")
    if architecture == "hf_ctc":
        from .ctc_adapter import CTCAdapter

        return CTCAdapter(spec, source, load_kwargs, device, cfg, output="chars")
    if architecture == "nemo":
        return NemoASRAdapter(spec, source, {}, device, cfg)
    raise UnsupportedArchitecture(f"pasketti_word: architecture '{architecture}' is not supported locally")
