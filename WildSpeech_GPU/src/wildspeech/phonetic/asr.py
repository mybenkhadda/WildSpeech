"""Local ASR (Whisper via transformers) and CTC character models with forced-alignment confidence.

Both run fully offline from local checkpoints (local_files_only=True). Audio never leaves the machine.
"""
from __future__ import annotations

import numpy as np

from .text_features import deviation_metrics, normalized_text


class WhisperTranscriber:
    def __init__(self, source: str, load_kwargs: dict, device: str = "cuda", fp16: bool = True):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(source, **load_kwargs)
        self.model = WhisperForConditionalGeneration.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).to(device).eval()
        self.multilingual = not source.endswith(".en")

    def transcribe(self, waves: list[np.ndarray], sr: int = 16000) -> list[dict]:
        torch = self.torch
        feats = self.processor(waves, sampling_rate=sr, return_tensors="pt").input_features.to(self.device, self.dtype)
        gen_kwargs = {"max_new_tokens": 96, "return_dict_in_generate": True, "output_scores": True}
        if self.multilingual:
            gen_kwargs.update({"language": "en", "task": "transcribe"})
        with torch.no_grad():
            out = self.model.generate(feats, **gen_kwargs)
            scores = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        res = []
        for i, t in enumerate(texts):
            s = scores[i].float().cpu().numpy()
            s = s[np.isfinite(s)]
            res.append({"text": t.strip(), "avg_logprob": float(s.mean()) if len(s) else np.nan,
                        "min_logprob": float(s.min()) if len(s) else np.nan, "n_tokens": float(len(s))})
        return res


class WhisperExpectedScorer:
    """Teacher-forced log-likelihood of the EXPECTED response under a local Whisper model (V2, P18).

    expected_ll_mean / _min / _sum : per-token log-probabilities of " <expected text><|endoftext|>" after the
                                     standard transcription prompt; n_expected_tokens.
    Shares the loaded model with WhisperTranscriber when given one.
    """

    def __init__(self, transcriber: "WhisperTranscriber"):
        self.t = transcriber
        tok = transcriber.processor.tokenizer
        if transcriber.multilingual:
            tok.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
        self.prefix = list(tok.prefix_tokens)
        self.eos = int(tok.eos_token_id)
        self.tok = tok

    def score(self, waves: list[np.ndarray], texts: list[str], sr: int = 16000) -> list[dict]:
        torch = self.t.torch
        feats = self.t.processor(waves, sampling_rate=sr, return_tensors="pt").input_features.to(self.t.device, self.t.dtype)
        seqs = [self.prefix + self.tok.encode(" " + str(t).strip(), add_special_tokens=False) + [self.eos] for t in texts]
        L = max(len(s) for s in seqs)
        dec = torch.full((len(seqs), L - 1), self.eos, dtype=torch.long)
        lab = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            dec[i, : len(s) - 1] = torch.tensor(s[:-1])
            lab[i, len(self.prefix) - 1: len(s) - 1] = torch.tensor(s[len(self.prefix):])
        with torch.no_grad():
            logits = self.t.model(input_features=feats, decoder_input_ids=dec.to(self.t.device)).logits.float()
            lp = torch.log_softmax(logits, -1).cpu()
        out = []
        for i in range(len(seqs)):
            m = lab[i] >= 0
            tl = lp[i][m].gather(1, lab[i][m].unsqueeze(1)).squeeze(1).numpy()
            out.append({"expected_ll_mean": float(tl.mean()) if len(tl) else np.nan,
                        "expected_ll_min": float(tl.min()) if len(tl) else np.nan,
                        "expected_ll_sum": float(tl.sum()) if len(tl) else np.nan, "n_expected_tokens": float(len(tl))})
        return out


class CTCAligner:
    """Character CTC model (e.g. wav2vec2-base-960h). Goodness-of-pronunciation via forced alignment of the
    expected text: per-token posterior of the aligned path vs the free (greedy) best path."""

    def __init__(self, source: str, load_kwargs: dict, device: str = "cuda", fp16: bool = True):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(source, **load_kwargs)
        self.model = AutoModelForCTC.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).to(device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        self.vocab = vocab
        self.blank = int(self.processor.tokenizer.pad_token_id or 0)
        self.delim = vocab.get("|", None)
        self.upper = any(k.isupper() for k in vocab if len(k) == 1 and k.isalpha())

    def _targets(self, text: str) -> list[int]:
        t = normalized_text(text)
        t = t.upper() if self.upper else t
        ids = []
        for ch in t:
            if ch == " ":
                if self.delim is not None:
                    ids.append(self.delim)
            elif ch in self.vocab:
                ids.append(self.vocab[ch])
        return ids

    def score(self, waves: list[np.ndarray], texts: list[str], sr: int = 16000, low_conf: float = -2.0) -> list[dict]:
        torch = self.torch
        import torchaudio.functional as AF

        inp = self.processor(waves, sampling_rate=sr, return_tensors="pt", padding=True)
        values = inp.input_values.to(self.device, self.dtype)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            logits = self.model(values, attention_mask=mask.to(self.device) if mask is not None else None).logits.float()
        logp = torch.log_softmax(logits, -1)
        lengths = self.model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves])).tolist()
        out = []
        for i, (text, L) in enumerate(zip(texts, lengths)):
            L = int(min(L, logp.shape[1]))
            lp = logp[i, :L]
            greedy = lp.argmax(-1)
            collapsed = torch.unique_consecutive(greedy)
            collapsed = collapsed[collapsed != self.blank].tolist()
            inv = {v: k for k, v in self.vocab.items()}
            hyp = "".join(" " if t == self.delim else inv.get(t, "") for t in collapsed)
            r = {"greedy_text": hyp.strip().lower(), **{f"greedy_{k}": v for k, v in deviation_metrics(hyp, text).items()
                                                        if k in ("cer", "wer")}}
            tgt = self._targets(text)
            if not tgt or L < len(tgt):
                r.update({"gop_mean": np.nan, "gop_min": np.nan, "low_conf_frac": np.nan, "gop_gap": np.nan,
                          "align_frames": float(L)})
                out.append(r)
                continue
            try:
                ali, scores = AF.forced_align(lp.unsqueeze(0).cpu(), torch.tensor([tgt], dtype=torch.int32), blank=self.blank)
                ali, scores = ali[0], scores[0]
                nonblank = ali != self.blank
                tok_lp = scores[nonblank].numpy()
                free = lp.max(-1).values.cpu().numpy()[nonblank.numpy()]
                r.update({"gop_mean": float(tok_lp.mean()), "gop_min": float(tok_lp.min()),
                          "low_conf_frac": float((tok_lp < low_conf).mean()),
                          "gop_gap": float((free - tok_lp).mean()), "align_frames": float(L)})
            except Exception:
                r.update({"gop_mean": np.nan, "gop_min": np.nan, "low_conf_frac": np.nan, "gop_gap": np.nan,
                          "align_frames": float(L)})
            r["greedy_cer"] = r.pop("greedy_cer", np.nan)
            out.append(r)
        return out
