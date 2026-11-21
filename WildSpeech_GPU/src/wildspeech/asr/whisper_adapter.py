"""Whisper (transformers) adapter — general-purpose ASR benchmark, also used for seq2seq child checkpoints.

Per clip:
  transcript                free decoding (greedy or beam), chunks of <= min(chunk_seconds, 30 s) with overlap,
                            stitched by merge_transcripts
  token_logprobs            REAL per-token log-probabilities from generate(output_scores=True) +
                            compute_transition_scores (special tokens removed)
  token_entropies           entropy of the model's next-token distribution at every emitted token
  expected_ll               teacher-forced mean log-probability of " <expected text><|endoftext|>" (clips that fit
                            one 30 s window; NaN + flag for longer clips — never approximated)
  embedding                 mean of the final encoder layer over VALID frames only (Whisper pads to 30 s)
output="phones" (seq2seq phone recognisers): the decoded string is parsed into phone tokens (ARPAbet or IPA ->
ARPAbet); transcript stays "".
"""
from __future__ import annotations

import time

import numpy as np

from .base import ASRResult, BaseASRAdapter, count_parameters, merge_transcripts, select_dtype, split_waveform

WHISPER_WINDOW_S = 30.0
ENCODER_FRAME_S = 0.02


def parse_phone_string(text: str) -> list[str]:
    """'k ae t' / 'K AE1 T' / 'kæt' (IPA) -> ARPAbet phones. Unmappable symbols are dropped."""
    from ..phonetic.g2p import ARPA_CONSONANTS, ARPA_VOWELS, _norm_ipa, ipa_to_arpa_table, strip_stress

    tab = ipa_to_arpa_table()
    toks = str(text or "").replace(",", " ").split()
    out: list[str] = []
    arpa = ARPA_VOWELS | ARPA_CONSONANTS
    for t in toks:
        s = strip_stress(t)
        if s in arpa:
            out.append(s)
            continue
        n = _norm_ipa(t)
        i = 0
        while i < len(n):                      # greedy longest EXACT match over IPA symbols (2-char first)
            two = n[i:i + 2]
            if len(two) == 2 and two in tab:
                if tab[two]:
                    out.append(tab[two])
                i += 2
                continue
            a1 = tab.get(n[i])
            if a1:
                out.append(a1)
            i += 1
    return out


class WhisperAdapter(BaseASRAdapter):
    output = "words"
    exposes_token_probs = True

    def __init__(self, spec, source, load_kwargs, device, cfg, output: str | None = None):
        super().__init__(spec, source, load_kwargs, device, cfg)
        self.output = output or getattr(spec, "output", "words") or "words"
        a = cfg.get_path("v3.asr", {}) or {}
        self.max_new_tokens = int(a.get("whisper_max_new_tokens", 96))
        self.num_beams = int(a.get("whisper_beam_size", 1))
        self.score_expected = bool(a.get("expected_text_scoring", True))
        self.win_s = min(self.chunk_s, WHISPER_WINDOW_S)

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.dtype, self.dtype_name = select_dtype(self.device, self.precision)
        self.processor = WhisperProcessor.from_pretrained(self.source, **self.load_kwargs)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.source, torch_dtype=self.dtype,
                                                                     **self.load_kwargs).to(self.device).eval()
        self.n_params = count_parameters(self.model)
        tok = self.processor.tokenizer
        gc = getattr(self.model, "generation_config", None)
        self.multilingual = bool(getattr(gc, "is_multilingual", False)) or ("<|en|>" in tok.get_vocab() and len(tok) > 51864)
        if self.multilingual:
            tok.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
        self.prefix = list(tok.prefix_tokens)
        self.eos = int(tok.eos_token_id)
        self.special = set(int(i) for i in tok.all_special_ids)

    def unload(self) -> None:
        self.processor = None
        super().unload()

    # ------------------------------------------------------------------ helpers
    def _features(self, waves: list[np.ndarray], sr: int):
        inp = self.processor(waves, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        return inp.input_features.to(self.model.device, self.model.dtype), inp.get("attention_mask")

    def _generate(self, feats, mask):
        torch = self.torch
        kw = {"max_new_tokens": self.max_new_tokens, "return_dict_in_generate": True, "output_scores": True,
              "num_beams": self.num_beams}
        if self.multilingual:
            kw.update({"language": "en", "task": "transcribe"})
        if mask is not None:
            kw["attention_mask"] = mask.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(feats, **kw)
            trans = self.model.compute_transition_scores(out.sequences, out.scores,
                                                         getattr(out, "beam_indices", None), normalize_logits=True)
        seqs = out.sequences.cpu().numpy()
        lp = trans.float().cpu().numpy()
        # next-token distribution entropy at each generated position (scores are processed logits per step)
        ent = None
        if self.num_beams == 1:
            steps = [torch.log_softmax(s.float(), -1) for s in out.scores]
            ent = torch.stack([-(st.exp() * st).nan_to_num(0.0).sum(-1) for st in steps], 1).cpu().numpy()
        return seqs, lp, ent

    def _token_rows(self, seq: np.ndarray, lp: np.ndarray, ent: np.ndarray | None):
        gen = seq[-lp.shape[0]:] if lp.shape[0] <= len(seq) else seq
        toks, lps, ents = [], [], []
        for k, t in enumerate(gen):
            if int(t) in self.special or not np.isfinite(lp[k]):
                continue
            toks.append(int(t))
            lps.append(float(lp[k]))
            if ent is not None and k < ent.shape[0]:
                ents.append(float(ent[k]))
        return toks, lps, ents

    def _encoder_embedding(self, feats, durations_s: list[float]) -> list[np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            h = self.model.get_encoder()(feats).last_hidden_state.float()
        out = []
        for i, d in enumerate(durations_s):
            n = int(np.clip(np.ceil(d / ENCODER_FRAME_S), 1, h.shape[1]))     # valid frames only
            out.append(h[i, :n].mean(0).cpu().numpy().astype(np.float32))
        return out

    def _expected_ll(self, feats, texts: list[str]) -> list[float]:
        torch = self.torch
        tok = self.processor.tokenizer
        seqs = [self.prefix + tok.encode(" " + str(t).strip(), add_special_tokens=False) + [self.eos] for t in texts]
        L = max(len(s) for s in seqs)
        dec = torch.full((len(seqs), L - 1), self.eos, dtype=torch.long)
        lab = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            dec[i, : len(s) - 1] = torch.tensor(s[:-1])
            lab[i, len(self.prefix) - 1: len(s) - 1] = torch.tensor(s[len(self.prefix):])
        with torch.no_grad():
            logits = self.model(input_features=feats, decoder_input_ids=dec.to(self.model.device)).logits.float()
            lp = torch.log_softmax(logits, -1).cpu()
        out = []
        for i in range(len(seqs)):
            m = lab[i] >= 0
            tl = lp[i][m].gather(1, lab[i][m].unsqueeze(1)).squeeze(1).numpy()
            out.append(float(tl.mean()) if len(tl) else np.nan)
        return out

    # ------------------------------------------------------------------ main
    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        t0 = time.perf_counter()
        tok = self.processor.tokenizer
        pieces, owner, plans = [], [], []
        for i, w in enumerate(waves):
            ch, plan = split_waveform(np.asarray(w, np.float32), sr, self.win_s, self.overlap_s)
            plans.append(plan)
            pieces += ch
            owner += [i] * len(ch)
        feats, mask = self._features(pieces, sr)
        seqs, lp, ent = self._generate(feats, mask)
        per_clip = {i: {"texts": [], "lps": [], "ents": [], "toks": []} for i in range(len(waves))}
        for j, i in enumerate(owner):
            toks, lps, ents = self._token_rows(seqs[j], lp[j], ent[j] if ent is not None else None)
            per_clip[i]["texts"].append(tok.decode(toks, skip_special_tokens=True).strip())
            per_clip[i]["lps"] += lps
            per_clip[i]["ents"] += ents
            per_clip[i]["toks"] += toks
        embs = None
        if self.save_embeddings:
            durs = [len(p) / sr for p in pieces]
            pe = self._encoder_embedding(feats, durs)
            embs = {}
            for j, i in enumerate(owner):
                embs.setdefault(i, []).append((pe[j], durs[j]))
        exp_ll = [np.nan] * len(waves)
        short = [i for i in range(len(waves)) if len(plans[i]) == 1 and expected[i]]
        if self.score_expected and short:
            idx = [owner.index(i) for i in short]
            vals = self._expected_ll(feats[idx], [expected[i] for i in short])
            for i, v in zip(short, vals):
                exp_ll[i] = v
        dt = (time.perf_counter() - t0) / max(len(waves), 1)
        res = []
        for i, w in enumerate(waves):
            text = merge_transcripts(per_clip[i]["texts"])
            raw = " ".join(tok.convert_ids_to_tokens(per_clip[i]["toks"]))
            r = ASRResult(transcript="" if self.output == "phones" else text, raw_tokens=raw,
                          token_logprobs=per_clip[i]["lps"], token_entropies=per_clip[i]["ents"],
                          expected_ll=exp_ll[i], n_chunks=len(plans[i]), audio_seconds=len(w) / sr, decode_seconds=dt)
            if self.output == "phones":
                r.phones = " ".join(parse_phone_string(text))
            else:
                r.extra["phones_source"] = "none"
            if len(plans[i]) > 1:
                r.extra["expected_scoring"] = "skipped_long_clip"
            if embs is not None:
                v = np.array([e for e, _ in embs[i]])
                wts = np.array([d for _, d in embs[i]])
                r.embedding = (v * wts[:, None]).sum(0) / max(wts.sum(), 1e-9)
            res.append(r)
        return res
