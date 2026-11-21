"""CTC models (character or phoneme vocabularies) with expected-text forced alignment — fully local.

Per clip (only what the model + metadata support):
  expected_ll            log P(expected | audio) per target token (CTC forward, -ctc_loss / L)
  greedy_ll              mean per-frame log-prob of the greedy path;  ll_ratio = greedy_ll*T/L - expected_ll
  gop_mean/min/p10       goodness-of-pronunciation proxy per aligned token: mean log-posterior of the target
                         token over its frames minus the mean best log-posterior over the same frames (<= 0)
  low_conf_frac          share of tokens with mean target log-posterior < low_confidence_logprob
  align_entropy          mean posterior entropy over aligned (non-blank) frames
  tok_dur_mean/cv        aligned token durations (s);  vowel_/consonant_gop_mean (phoneme models)
  gop_initial/medial/final, worst_pos   GOP by relative position;  greedy text / phones
  post_<PHONE>           (phoneme models) mean posterior mass per ARPAbet class over speech frames -> P19
Forced alignment uses torchaudio.functional.forced_align on CPU (exact Viterbi); logits are computed on the GPU.
"""
from __future__ import annotations

import numpy as np

from .g2p import ARPABET, ARPA_TO_IPA, ipa_to_arpa, is_vowel

FRAME_S = 0.02


class CTCModel:
    def __init__(self, source: str, load_kwargs: dict, kind: str, device: str = "cuda", fp16: bool = True):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoProcessor

        self.torch, self.kind, self.device = torch, kind, device
        self.dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32
        if kind == "ctc_phoneme":
            # decoding / vocabulary only: never phonemize text here (no phonemizer / espeak dependency)
            from transformers import Wav2Vec2PhonemeCTCTokenizer

            self.fe = AutoFeatureExtractor.from_pretrained(source, **load_kwargs)
            tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(source, do_phonemize=False, **load_kwargs)
        else:
            proc = AutoProcessor.from_pretrained(source, **load_kwargs)
            self.fe, tok = proc.feature_extractor, proc.tokenizer
        self.tokenizer = tok
        self.model = AutoModelForCTC.from_pretrained(source, torch_dtype=self.dtype, **load_kwargs).to(device).eval()
        self.vocab: dict[str, int] = tok.get_vocab()
        self.inv = {v: k for k, v in self.vocab.items()}
        self.blank = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
        self.delim = self.vocab.get("|", self.vocab.get(" ", None))
        self.upper = any(k.isupper() for k in self.vocab if len(k) == 1 and k.isalpha())
        self.special = {self.blank} | {i for k, i in self.vocab.items() if k.startswith("<") and k.endswith(">")}
        if kind == "ctc_phoneme":
            self.tok_arpa = {i: ipa_to_arpa(k) for k, i in self.vocab.items() if i not in self.special}
            self.arpa_tok = {}
            for a, cands in ARPA_TO_IPA.items():
                for cnd in cands:
                    if cnd in self.vocab:
                        self.arpa_tok[a] = self.vocab[cnd]
                        break

    # ------------------------------------------------------------------ logits
    def log_posteriors(self, waves: list[np.ndarray], sr: int = 16000) -> list[np.ndarray]:
        torch = self.torch
        inp = self.fe(waves, sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
        vals = inp.input_values.to(self.device, self.dtype)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            use_mask = getattr(self.model.config, "feat_extract_norm", "group") == "layer"
            logits = self.model(vals, attention_mask=mask.to(self.device) if (mask is not None and use_mask) else None).logits.float()
            lp = torch.log_softmax(logits, -1).cpu()
        lens = self.model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves])).tolist()
        return [lp[i, : int(min(L, lp.shape[1]))].numpy() for i, L in enumerate(lens)]

    # ----------------------------------------------------------------- targets
    def char_targets(self, text: str) -> list[int]:
        import re

        t = re.sub(r"[^a-z' ]+", " ", str(text or "").lower()).strip()
        t = re.sub(r"\s+", " ", t)
        t = t.upper() if self.upper else t
        ids = []
        for ch in t:
            if ch == " ":
                if self.delim is not None:
                    ids.append(self.delim)
            elif ch in self.vocab:
                ids.append(self.vocab[ch])
        return ids

    def phone_targets(self, phones: list[str]) -> tuple[list[int], float]:
        ids = [self.arpa_tok[p] for p in phones if p in self.arpa_tok]
        return ids, (len(ids) / max(len(phones), 1))

    def greedy(self, lp: np.ndarray) -> list[int]:
        best = lp.argmax(-1)
        out, prev = [], -1
        for t in best:
            if t != prev and t not in self.special:
                out.append(int(t))
            prev = t
        return out

    def decode_text(self, ids: list[int]) -> str:
        return "".join(" " if i == self.delim else self.inv.get(i, "") for i in ids).strip().lower()

    def decode_phones(self, ids: list[int]) -> list[str]:
        return [a for a in (self.tok_arpa.get(i) for i in ids) if a]

    # --------------------------------------------------------------- features
    def align_features(self, lp: np.ndarray, target: list[int], token_labels: list[str] | None = None,
                       low_conf: float = -2.0, bins: int = 3) -> tuple[dict, str]:
        torch = self.torch
        import torchaudio.functional as AF

        T = lp.shape[0]
        r = {"n_frames": float(T)}
        if not target or T < len(target) + 1:
            return r, ""
        lpt = torch.from_numpy(lp).float()
        tgt = torch.tensor([target], dtype=torch.int32)
        nll = torch.nn.functional.ctc_loss(lpt.unsqueeze(1), tgt, torch.tensor([T]), torch.tensor([len(target)]),
                                           blank=self.blank, reduction="sum", zero_infinity=True)
        r["expected_ll"] = float(-nll.item() / len(target))
        r["greedy_ll"] = float(lp.max(-1).mean())
        r["ll_ratio"] = float(r["greedy_ll"] * T / len(target) - r["expected_ll"])
        try:
            ali, _ = AF.forced_align(lpt.unsqueeze(0), tgt, blank=self.blank)
            spans = AF.merge_tokens(ali[0], torch.exp(lpt.gather(1, ali[0].long().unsqueeze(1)).squeeze(1)), blank=self.blank)
        except Exception:  # noqa: BLE001 - alignment impossible (e.g. too few frames)
            return r, ""
        if len(spans) != len(target):
            return r, ""
        best = lp.max(-1)
        P = np.exp(lp)
        ent = -(P * lp).sum(-1)
        gops, durs, ents, conf, rows = [], [], [], [], []
        for k, s in enumerate(spans):
            fr = np.arange(s.start, s.end)
            tl = lp[fr, target[k]]
            gops.append(float(tl.mean() - best[fr].mean()))
            conf.append(float(tl.mean()))
            durs.append(len(fr) * FRAME_S)
            ents.append(float(ent[fr].mean()))
            lab = token_labels[k] if token_labels and k < len(token_labels) else self.inv.get(target[k], "?")
            rows.append(f"{lab}:{s.start}:{s.end}:{gops[-1]:.3f}")
        g = np.array(gops)
        r.update({"gop_mean": float(g.mean()), "gop_min": float(g.min()), "gop_p10": float(np.percentile(g, 10)),
                  "low_conf_frac": float((np.array(conf) < low_conf).mean()), "align_entropy": float(np.mean(ents)),
                  "tok_dur_mean": float(np.mean(durs)), "tok_dur_cv": float(np.std(durs) / max(np.mean(durs), 1e-6)),
                  "worst_pos": float(np.argmin(g) / max(len(g) - 1, 1))})
        names = ["initial", "medial", "final"] if bins == 3 else [f"bin{k}" for k in range(bins)]
        for b, nm in enumerate(names):
            sel = [i for i in range(len(g)) if min(bins - 1, int(bins * i / len(g))) == b]
            r[f"gop_{nm}"] = float(g[sel].mean()) if sel else np.nan
        if token_labels is not None and self.kind == "ctc_phoneme":
            v = [gops[i] for i, lab in enumerate(token_labels[:len(gops)]) if is_vowel(lab)]
            c = [gops[i] for i, lab in enumerate(token_labels[:len(gops)]) if not is_vowel(lab)]
            r["vowel_gop_mean"] = float(np.mean(v)) if v else np.nan
            r["consonant_gop_mean"] = float(np.mean(c)) if c else np.nan
        return r, "|".join(rows)

    def posterior_summary(self, lp: np.ndarray) -> dict:
        """Mean posterior mass per ARPAbet class over non-blank-dominated frames (phoneme models only)."""
        if self.kind != "ctc_phoneme":
            return {}
        P = np.exp(lp)
        speech = P[:, self.blank] < 0.5
        if not speech.any():
            return {f"post_{a}": np.nan for a in ARPABET}
        Ps = P[speech]
        out = {a: 0.0 for a in ARPABET}
        for i, a in self.tok_arpa.items():
            if a in out and i < Ps.shape[1]:
                out[a] += float(Ps[:, i].mean())
        tot = sum(out.values()) or 1.0
        return {f"post_{a}": v / tot for a, v in out.items()}
