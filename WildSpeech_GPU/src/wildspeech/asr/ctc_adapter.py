"""CTC adapter (transformers AutoModelForCTC): character or phoneme vocabularies, fully local.

Supported checkpoints: Wav2Vec2ForCTC (LibriSpeech characters; eSpeak phoneme tokens) and WavLMForCTC (Pasketti
phonetic: one CTC token per IPA CHARACTER). The tokenizer class is read from the snapshot's tokenizer_config
(Wav2Vec2PhonemeCTCTokenizer with do_phonemize=False, or the plain CTC tokenizer) — never guessed.

Per clip:
  log-posteriors        computed on VALID frames only: output lengths from the waveform lengths
                        (`_get_feat_extract_output_lengths`), padding frames are discarded before any statistic.
                        Group-norm feature encoders (wav2vec2-base family, feat_extract_norm="group") ignore the
                        attention mask, so padded batching would change their outputs: those clips are run one by
                        one (v3.asr.unbatched_group_norm). Long clips are chunked with overlap and stitched.
  greedy decoding       collapsed best path -> characters (transcript) or phones. Phone tokens are converted to
                        ARPAbet by greedy longest-match over the IPA string of each word, so character-level IPA
                        ("t" "ʃ") becomes CH, "a" "ɪ" becomes AY; the raw IPA string is kept in `ipa`.
  token_logprobs        for every emitted greedy token: mean log-posterior of that token over its frames
  frame_entropy_mean    mean posterior entropy over speech (non-blank-dominated) frames; blank_frac
  expected_ll           CTC forward log-likelihood of the expected target sequence per target token
  expected_alignment    Viterbi forced alignment (torchaudio.functional.forced_align, CPU) per EXPECTED PHONE
                        (multi-token phones are merged): {p, s, e, lp, gop, post}; gop = mean target log-posterior -
                        mean best log-posterior (<= 0). The alignment is CONDITIONED on the expected text: a path
                        always exists, so it is never treated as evidence that the expected sounds were produced.
  embedding             mean of the final encoder hidden state over valid frames (forward hook, frozen)
WavLM checkpoints are always run in fp32 (relative position bias overflows in half precision; Pasketti card).
"""
from __future__ import annotations

import re
import time

import numpy as np

from .base import ASRResult, BaseASRAdapter, count_parameters, select_dtype, split_waveform, stitch_frames

FRAME_S = 0.02
DELIMS = ("|", " ")
RHOTIC_COMPOSITES = {"ɑɹ": ["AA", "R"], "ɔɹ": ["AO", "R"], "ɪɹ": ["IH", "R"], "ɛɹ": ["EH", "R"], "ʊɹ": ["UH", "R"]}


def ipa_sequence_to_arpa(tokens: list[str], merge: bool = True) -> list[str]:
    """IPA tokens -> ARPAbet by greedy longest match (3 -> 1 symbols) over normalised IPA.
    merge=True concatenates consecutive tokens within a word first (character-level inventories); merge=False
    parses each token on its own (multi-character phoneme tokens, e.g. eSpeak 'tʃ'). Word delimiters split words.
    ARPAbet tokens (e.g. 'AH0') pass through. Unmappable symbols (stress, glottal stop) are dropped."""
    from ..phonetic.g2p import ARPA_CONSONANTS, ARPA_VOWELS, _norm_ipa, ipa_to_arpa_table, strip_stress

    tab = dict(ipa_to_arpa_table())
    for k, v in RHOTIC_COMPOSITES.items():         # 'ɑɹ' is AA + R, not AA (keeps the rhotic consonant)
        tab[k] = v
    arpa = ARPA_VOWELS | ARPA_CONSONANTS
    maxlen = max((len(k) for k in tab), default=1)
    groups: list[list[str]] = [[]]
    for t in tokens:
        t = str(t)
        if t in DELIMS or not t.strip():
            groups.append([])
        elif strip_stress(t) in arpa:
            groups.append([t])
            groups.append([])
        else:
            groups[-1].append(t)
    out: list[str] = []
    for g in groups:
        if not g:
            continue
        if len(g) == 1 and strip_stress(g[0]) in arpa:
            out.append(strip_stress(g[0]))
            continue
        units = ["".join(_norm_ipa(x) for x in g)] if merge else [_norm_ipa(x) for x in g]
        for n in units:
            i = 0
            while i < len(n):
                for L in range(min(maxlen, len(n) - i), 0, -1):
                    seg = n[i:i + L]
                    if seg in tab:
                        v = tab[seg]
                        if isinstance(v, list):
                            out += v
                        elif v:
                            out.append(v)
                        i += L
                        break
                else:
                    i += 1
    return out


class CTCAdapter(BaseASRAdapter):
    exposes_token_probs = True
    exposes_frame_posteriors = True

    def __init__(self, spec, source, load_kwargs, device, cfg, output: str | None = None):
        super().__init__(spec, source, load_kwargs, device, cfg)
        self.output = output or getattr(spec, "output", "chars") or "chars"
        self.kind = "ctc_phoneme" if self.output == "phones" else "ctc_char"
        a = cfg.get_path("v3.asr", {}) or {}
        self.score_expected = bool(a.get("expected_text_scoring", True))
        self.unbatched_gn = bool(a.get("unbatched_group_norm", True))
        self.merge_ipa = str(getattr(spec, "phone_inventory", "") or "") == "ipa_characters"
        ph = cfg.get_path("v3.phonetics", {}) or {}
        from ..phonetic.g2p import G2P

        self.g2p = G2P(ph.get("g2p_backend", "auto"))
        self.allow_approx = bool(ph.get("allow_approximate_g2p", False))

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCTC

        from .environment import HALF_UNSAFE_MODEL_TYPES, load_ctc_tokenizer

        self.torch = torch
        mcfg = AutoConfig.from_pretrained(self.source, **self.load_kwargs)
        prec = "fp32" if mcfg.model_type in HALF_UNSAFE_MODEL_TYPES else self.precision
        self.dtype, self.dtype_name = select_dtype(self.device, prec)
        self.fe = AutoFeatureExtractor.from_pretrained(self.source, **self.load_kwargs)
        tok = load_ctc_tokenizer(self.source, self.output, self.load_kwargs)
        self.tokenizer = tok
        self.model = AutoModelForCTC.from_pretrained(self.source, torch_dtype=self.dtype, **self.load_kwargs).to(self.device).eval()
        self.n_params = count_parameters(self.model)
        self.group_norm = getattr(self.model.config, "feat_extract_norm", "group") == "group"
        self.vocab: dict[str, int] = tok.get_vocab()
        self.inv = {v: k for k, v in self.vocab.items()}
        self.blank = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
        self.delim = self.vocab.get("|", self.vocab.get(" ", None))
        self.upper = any(k.isupper() for k in self.vocab if len(k) == 1 and k.isalpha())
        self.special = {self.blank} | {i for k, i in self.vocab.items() if k.startswith("<") and k.endswith(">")}
        self.samples_per_frame = int(np.prod([int(k) for k in getattr(self.model.config, "conv_stride", [320])])) or 320
        if self.kind == "ctc_phoneme":
            self._build_phone_maps()
        self._hook_out = None
        if self.save_embeddings:
            base = getattr(self.model, "base_model", None) or self.model

            def hook(_m, _i, o):
                self._hook_out = (o[0] if isinstance(o, (tuple, list)) else getattr(o, "last_hidden_state", None))
            self._hook = base.register_forward_hook(hook)

    def _build_phone_maps(self) -> None:
        """ARPAbet phone -> token-id sequence for expected-phone targets (single IPA token preferred, else the
        IPA characters of the phone when every character is in the vocabulary)."""
        from ..phonetic.g2p import ARPA_TO_IPA, _norm_ipa

        self.arpa_tok: dict[str, list[int]] = {}
        norm_vocab = {}
        for k, i in self.vocab.items():
            if i in self.special or k in DELIMS:
                continue
            norm_vocab.setdefault(_norm_ipa(k), i)
        for a, cands in ARPA_TO_IPA.items():
            if a in self.vocab:
                self.arpa_tok[a] = [self.vocab[a]]
                continue
            seq = None
            for c in cands:
                if c in self.vocab:
                    seq = [self.vocab[c]]
                    break
                n = _norm_ipa(c)
                if n in norm_vocab:
                    seq = [norm_vocab[n]]
                    break
            if seq is None:
                for c in cands:
                    chars = [ch for ch in c if ch not in "ː"]
                    if chars and all(ch in self.vocab for ch in chars):
                        seq = [self.vocab[ch] for ch in chars]
                        break
            if seq:
                self.arpa_tok[a] = seq

    def unload(self) -> None:
        h = getattr(self, "_hook", None)
        if h is not None:
            h.remove()
        self._hook_out = None
        super().unload()

    # ------------------------------------------------------------------ posteriors
    def _forward(self, waves: list[np.ndarray], sr: int) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        torch = self.torch
        inp = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
        vals = inp.input_values.to(self.model.device, self.model.dtype)
        mask = inp.get("attention_mask")
        use_mask = getattr(self.model.config, "feat_extract_norm", "group") == "layer"
        with torch.no_grad():
            logits = self.model(vals, attention_mask=mask.to(self.model.device) if (mask is not None and use_mask) else None).logits.float()
            lp = torch.log_softmax(logits, -1).cpu()
        lens = self.model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves])).tolist()
        lps = [lp[i, : int(min(L, lp.shape[1]))].numpy() for i, L in enumerate(lens)]
        embs: list[np.ndarray | None] = [None] * len(waves)
        if self._hook_out is not None:
            h = self._hook_out.float()
            for i, L in enumerate(lens):
                n = int(min(L, h.shape[1]))
                embs[i] = h[i, :n].mean(0).cpu().numpy().astype(np.float32) if n > 0 else None
            self._hook_out = None
        return lps, embs

    def _forward_many(self, waves: list[np.ndarray], sr: int):
        """Padded batch for layer-norm encoders (masked); one clip at a time for group-norm encoders."""
        if self.group_norm and self.unbatched_gn and len(waves) > 1:
            lps, embs = [], []
            for w in waves:
                a, b = self._forward([w], sr)
                lps += a
                embs += b
            return lps, embs
        return self._forward(waves, sr)

    def log_posteriors(self, wave: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Valid-frame log posteriors (T, V) for one clip, chunked + stitched if longer than chunk_seconds."""
        pieces, plan = split_waveform(np.asarray(wave, np.float32), sr, self.chunk_s, self.overlap_s)
        lps, embs = self._forward_many(pieces, sr)
        lp = stitch_frames(lps, plan, self.samples_per_frame) if len(plan) > 1 else lps[0]
        ev = [e for e in embs if e is not None]
        emb = np.average(np.stack(ev), axis=0, weights=[len(p) for p, e in zip(pieces, embs) if e is not None]) if ev else None
        return lp, emb, len(plan)

    # ------------------------------------------------------------------ targets
    def char_targets(self, text: str) -> tuple[list[int], list[str], list[int]]:
        t = re.sub(r"[^a-z' ]+", " ", str(text or "").lower()).strip()
        t = re.sub(r"\s+", " ", t)
        t = t.upper() if self.upper else t
        ids, labs, owner = [], [], []
        for ch in t:
            if ch == " ":
                if self.delim is not None:
                    ids.append(self.delim)
                    labs.append("|")
                    owner.append(len(labs) - 1)
            elif ch in self.vocab:
                ids.append(self.vocab[ch])
                labs.append(ch.lower())
                owner.append(len(labs) - 1)
        return ids, labs, owner

    def phone_targets(self, text: str) -> tuple[list[int], list[str], list[int], float]:
        """(token ids, phone labels, owner phone index per token, coverage of the expected phones)."""
        if not (self.g2p.exact or self.allow_approx):
            return [], [], [], np.nan
        ph = self.g2p.phonemes(text)
        ids, labs, owner = [], [], []
        for p in ph:
            seq = self.arpa_tok.get(p)
            if not seq:
                continue
            labs.append(p)
            for t in seq:
                ids.append(t)
                owner.append(len(labs) - 1)
        return ids, labs, owner, len(labs) / max(len(ph), 1)

    # ------------------------------------------------------------------ decoding
    def greedy(self, lp: np.ndarray) -> tuple[list[int], list[float]]:
        """Collapsed best path + per-emitted-token mean log posterior over its frames."""
        best = lp.argmax(-1)
        toks, scores, prev, run = [], [], -1, []
        for t, b in enumerate(best):
            if b != prev:
                if prev not in self.special and prev >= 0 and run:
                    toks.append(int(prev))
                    scores.append(float(np.mean(run)))
                run = []
            run.append(float(lp[t, b]))
            prev = b
        if prev not in self.special and prev >= 0 and run:
            toks.append(int(prev))
            scores.append(float(np.mean(run)))
        return toks, scores

    def decode(self, toks: list[int]) -> tuple[str, str, str, str]:
        """-> (transcript, ARPAbet phones, raw tokens, ipa)."""
        raw = " ".join(self.inv.get(i, "?") for i in toks)
        if self.kind == "ctc_phoneme":
            syms = ["|" if i == self.delim else self.inv.get(i, "") for i in toks]
            phones = ipa_sequence_to_arpa(syms, merge=self.merge_ipa)
            ipa = ("".join(" " if s == "|" else s for s in syms) if self.merge_ipa
                   else " ".join(s for s in syms)).strip()
            return "", " ".join(phones), raw, re.sub(r"\s+", " ", ipa)
        text = "".join(" " if i == self.delim else self.inv.get(i, "") for i in toks).strip().lower()
        return re.sub(r"\s+", " ", text), "", raw, ""

    def frame_stats(self, lp: np.ndarray) -> dict:
        P = np.exp(lp)
        ent = -(P * lp).sum(-1)
        speech = P[:, self.blank] < 0.5
        return {"frame_entropy_mean": float(ent[speech].mean()) if speech.any() else np.nan,
                "blank_frac": float(1 - speech.mean()) if len(speech) else np.nan,
                "greedy_ll": float(lp.max(-1).mean()) if len(lp) else np.nan}

    def expected_alignment(self, lp: np.ndarray, target: list[int], labels: list[str],
                           owner: list[int] | None = None) -> tuple[float, list[dict] | None]:
        torch = self.torch
        import torchaudio.functional as AF

        T = lp.shape[0]
        if not target or T < len(target) + 1:
            return np.nan, None
        owner = owner if owner is not None else list(range(len(target)))
        lpt = torch.from_numpy(lp).float()
        tgt = torch.tensor([target], dtype=torch.int32)
        nll = torch.nn.functional.ctc_loss(lpt.unsqueeze(1), tgt, torch.tensor([T]), torch.tensor([len(target)]),
                                           blank=self.blank, reduction="sum", zero_infinity=True)
        ell = float(-nll.item() / max(len(labels), 1))
        try:
            ali, _ = AF.forced_align(lpt.unsqueeze(0), tgt, blank=self.blank)
            spans = AF.merge_tokens(ali[0], torch.exp(lpt.gather(1, ali[0].long().unsqueeze(1)).squeeze(1)), blank=self.blank)
        except Exception:  # noqa: BLE001 - alignment impossible
            return ell, None
        if len(spans) != len(target):
            return ell, None
        best = lp.max(-1)
        by_phone: dict[int, list] = {}
        for k, s in enumerate(spans):
            by_phone.setdefault(owner[k], []).append((s, target[k]))
        rows = []
        for j in sorted(by_phone):
            parts = by_phone[j]
            fr = np.concatenate([np.arange(s.start, s.end) for s, _ in parts])
            tl = np.concatenate([lp[np.arange(s.start, s.end), t] for s, t in parts])
            rows.append({"p": labels[j] if j < len(labels) else "?", "s": int(parts[0][0].start),
                         "e": int(parts[-1][0].end), "lp": round(float(tl.mean()), 4),
                         "gop": round(float(tl.mean() - best[fr].mean()), 4), "post": round(float(np.exp(tl).mean()), 4)})
        return ell, rows

    # ------------------------------------------------------------------ main
    def transcribe_batch(self, waves: list[np.ndarray], expected: list[str], sr: int) -> list[ASRResult]:
        t0 = time.perf_counter()
        long = [i for i, w in enumerate(waves) if len(w) > self.chunk_s * sr]
        short = [i for i in range(len(waves)) if i not in set(long)]
        lp_by, emb_by, nch = {}, {}, {}
        if short:
            lps, embs = self._forward_many([np.asarray(waves[i], np.float32) for i in short], sr)
            for i, lp, e in zip(short, lps, embs):
                lp_by[i], emb_by[i], nch[i] = lp, e, 1
        for i in long:
            lp_by[i], emb_by[i], nch[i] = self.log_posteriors(waves[i], sr)
        dt = (time.perf_counter() - t0) / max(len(waves), 1)
        out = []
        for i, w in enumerate(waves):
            lp = lp_by[i]
            toks, scores = self.greedy(lp)
            text, phones, raw, ipa = self.decode(toks)
            r = ASRResult(transcript=text, phones=phones, raw_tokens=raw, token_logprobs=scores, n_chunks=nch[i],
                          audio_seconds=len(w) / sr, decode_seconds=dt, embedding=emb_by[i], **self.frame_stats(lp))
            r.extra["n_frames"] = int(lp.shape[0])
            r.extra["ipa"] = ipa
            r.extra["unbatched"] = bool(self.group_norm and self.unbatched_gn)
            if self.score_expected and expected[i]:
                if self.kind == "ctc_phoneme":
                    tgt, labs, owner, cov = self.phone_targets(expected[i])
                    r.extra["target_coverage"] = cov
                else:
                    tgt, labs, owner = self.char_targets(expected[i])
                r.expected_ll, r.expected_alignment = self.expected_alignment(lp, tgt, labs, owner)
            out.append(r)
        return out
