"""drivendata/pasketti-phonetic — child-specific phonetic recogniser.

The published checkpoint (revision 675dfe3, 2026-10-23) is a standard transformers WavLMForCTC (wavlm-large
fine-tune) emitting one CTC token per IPA character; it is run in fp32 (card) with greedy decoding (as evaluated on
the card). The architecture is still read from the local snapshot rather than assumed:
  hf_ctc      -> CTCAdapter(output="phones"): frame posteriors, greedy IPA -> ARPAbet (character merges such as
                 t+ʃ -> CH), raw IPA kept in `ipa`, expected-phoneme forced alignment
  hf_whisper  -> WhisperAdapter(output="phones"): decoded phone string parsed to ARPAbet, real token log-probs
  other       -> UnsupportedArchitecture (P24 records the model as NOT_EXECUTED with this reason)
The phone inventory of the checkpoint is mapped to ARPAbet through wildspeech.phonetic.g2p; symbols that cannot
be mapped are dropped and their share is reported as `unmapped_token_frac` so mapping loss is visible.
"""
from __future__ import annotations

import numpy as np

from .base import ASRResult


class UnsupportedArchitecture(RuntimeError):
    pass


def _unmapped_fraction(raw_tokens: str, phones: str = "") -> float:
    """Share of emitted phone tokens that map to NO ARPAbet phone on their own (glottal stops, bare diacritics,
    unknown symbols). Character-level IPA merges (t + ʃ -> CH) are not counted as loss."""
    from .ctc_adapter import ipa_sequence_to_arpa

    raw = [t for t in str(raw_tokens or "").split() if t not in ("|", "<pad>", "<s>", "</s>", "<unk>")]
    if not raw:
        return np.nan
    lost = sum(1 for t in raw if not ipa_sequence_to_arpa([t], merge=False))
    return float(lost / len(raw))


class PaskettiPhoneticAdapter:
    """Factory-wrapper exposing the BaseASRAdapter interface of the concrete adapter it builds."""

    def __init__(self, spec, source: str, load_kwargs: dict, device: str, cfg, architecture: str):
        self.spec, self.architecture = spec, architecture
        if architecture == "hf_ctc":
            from .ctc_adapter import CTCAdapter

            self.impl = CTCAdapter(spec, source, load_kwargs, device, cfg, output="phones")
        elif architecture == "hf_whisper":
            from .whisper_adapter import WhisperAdapter

            self.impl = WhisperAdapter(spec, source, load_kwargs, device, cfg, output="phones")
        else:
            raise UnsupportedArchitecture(f"pasketti_phonetic: architecture '{architecture}' is not supported locally")
        self.output = "phones"

    @property
    def exposes_token_probs(self) -> bool:
        return self.impl.exposes_token_probs

    @property
    def exposes_frame_posteriors(self) -> bool:
        return self.impl.exposes_frame_posteriors

    @property
    def device(self) -> str:
        return self.impl.device

    def load(self) -> None:
        self.impl.load()

    def unload(self) -> None:
        self.impl.unload()

    def to_cpu(self) -> None:
        self.impl.to_cpu()

    def to_gpu(self) -> None:
        self.impl.to_gpu()

    @property
    def n_params(self) -> int:
        return self.impl.n_params

    @property
    def dtype_name(self) -> str:
        return self.impl.dtype_name

    def set_fp32(self) -> None:
        self.impl.set_fp32()

    def version_info(self) -> dict:
        return {**self.impl.version_info(), "wrapper": "pasketti_phonetic", "architecture": self.architecture}

    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        res = self.impl.transcribe_batch(waves, expected, sr)
        for r in res:
            r.extra["unmapped_token_frac"] = _unmapped_fraction(r.raw_tokens, r.phones)
        return res
