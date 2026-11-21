"""Local grapheme-to-phoneme processing and expected-text complexity features.

G2P backends (all local, tried in order for ``g2p_backend='auto'``):
``g2p_en`` (neural G2P; needs its bundled model and NLTK tagger data available
locally), ``pronouncing`` (CMU Pronouncing Dictionary lookup), and a
deterministic rule-based grapheme approximation (flagged as approximate).
No text is sent to remote services. Word-frequency and phonetic-neighbourhood
features are only computed when a licensed local lexicon is supplied.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
RULE_DIGRAPHS = {"ch": "CH", "sh": "SH", "th": "TH", "ph": "F", "ng": "NG", "ck": "K", "wh": "W", "qu": "K W",
                 "ee": "IY", "ea": "IY", "oo": "UW", "ai": "EY", "ay": "EY", "oa": "OW", "ou": "AW", "ow": "OW",
                 "oi": "OY", "oy": "OY", "igh": "AY"}
RULE_LETTERS = {"a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G", "h": "HH", "i": "IH",
                "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R", "s": "S",
                "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K S", "y": "Y", "z": "Z"}


def tokenize_words(text: str) -> list:
    return re.findall(r"[a-z']+", str(text).lower())


def strip_stress(ph: str) -> str:
    return re.sub(r"\d", "", ph)


def rule_g2p(word: str) -> list:
    word = word.lower().replace("'", "")
    out, i = [], 0
    while i < len(word):
        for size in (3, 2):
            chunk = word[i:i + size]
            if chunk in RULE_DIGRAPHS:
                out.extend(RULE_DIGRAPHS[chunk].split())
                i += size
                break
        else:
            ch = word[i]
            if ch in RULE_LETTERS and not (ch == "e" and i == len(word) - 1 and len(word) > 2):
                out.extend(RULE_LETTERS[ch].split())
            i += 1
    return out


class G2P:
    def __init__(self, backend: str = "auto"):
        self.backend = None
        self._g2p = None
        self._pron = None
        order = ["g2p_en", "pronouncing", "rules"] if backend == "auto" else [backend]
        for name in order:
            try:
                if name == "g2p_en":
                    from g2p_en import G2p  # type: ignore

                    self._g2p = G2p()
                    self._g2p("test")
                elif name == "pronouncing":
                    import pronouncing  # type: ignore

                    pronouncing.phones_for_word("test")
                    self._pron = pronouncing
                self.backend = name
                break
            except Exception:
                continue
        if self.backend is None:
            self.backend = "rules"

    @lru_cache(maxsize=200000)
    def word(self, w: str) -> tuple:
        if self.backend == "g2p_en":
            return tuple(strip_stress(p) for p in self._g2p(w) if p.strip() and p not in {" ", "'"})
        if self.backend == "pronouncing":
            phones = self._pron.phones_for_word(w)
            if phones:
                return tuple(strip_stress(p) for p in phones[0].split())
        return tuple(rule_g2p(w))

    def text(self, text: str) -> list:
        return [list(self.word(w)) for w in tokenize_words(text)]

    def oov_rate(self, text: str) -> float:
        if self.backend != "pronouncing":
            return np.nan
        words = tokenize_words(text)
        if not words:
            return np.nan
        return float(np.mean([not self._pron.phones_for_word(w) for w in words]))


def lz_complexity(seq: list) -> float:
    """Lempel-Ziv (1976) phrase count normalised by n / log2(n)."""
    n = len(seq)
    if n < 2:
        return np.nan
    s = [str(x) for x in seq]
    i, c = 0, 0
    while i < n:
        k = 1
        while i + k <= n and _contains(s[:i + k - 1], s[i:i + k]):
            k += 1
        c += 1
        i += k
    return float(c / (n / np.log2(n)))


def _contains(hay: list, needle: list) -> bool:
    m = len(needle)
    return any(hay[j:j + m] == needle for j in range(len(hay) - m + 1))


def text_features(text: str, g2p: G2P) -> dict:
    words = tokenize_words(text)
    phon_words = g2p.text(text)
    phones = [p for w in phon_words for p in w]
    vowels = [p for p in phones if p in ARPABET_VOWELS]
    cons_flags = [p not in ARPABET_VOWELS for p in phones]
    clusters, run = [], 0
    for w in phon_words:
        for p in w:
            if p not in ARPABET_VOWELS:
                run += 1
            else:
                if run >= 2:
                    clusters.append(run)
                run = 0
        if run >= 2:
            clusters.append(run)
        run = 0
    bigrams = list(zip(phones[:-1], phones[1:]))
    out = {
        "l_n_chars": float(len(re.sub(r"[^a-z]", "", str(text).lower()))),
        "l_n_words": float(len(words)),
        "l_n_phonemes": float(len(phones)),
        "l_n_syllables": float(max(len(vowels), 1)) if phones else np.nan,
        "l_mean_word_len": float(np.mean([len(w) for w in words])) if words else np.nan,
        "l_max_word_len": float(max((len(w) for w in words), default=np.nan)) if words else np.nan,
        "l_n_consonants": float(sum(cons_flags)),
        "l_n_vowels": float(len(vowels)),
        "l_n_consonant_clusters": float(len(clusters)),
        "l_max_cluster_len": float(max(clusters)) if clusters else 0.0,
        "l_phonetic_diversity": float(len(set(phones)) / len(phones)) if phones else np.nan,
        "l_transition_complexity": float(len(set(bigrams)) / len(bigrams)) if bigrams else np.nan,
        "l_repeated_bigrams": float(len(bigrams) - len(set(bigrams))) if bigrams else 0.0,
        "l_sentence_len_words": float(len(words)),
        "l_lz_complexity": lz_complexity(phones),
        "l_g2p_oov_rate": g2p.oov_rate(text),
    }
    return out


def item_linguistic_table(items: pd.DataFrame, cfg: dict, lexicon: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    g2p = G2P(cfg.get("g2p_backend", "auto"))
    rows = []
    text_col = "expected_text" if "expected_text" in items.columns else None
    if text_col is None:
        return pd.DataFrame(columns=["item_key"]), {"status": "NO_EXPECTED_TEXT", "g2p_backend": g2p.backend}
    for _, r in items.iterrows():
        feats = text_features(r[text_col], g2p) if pd.notna(r[text_col]) else {}
        feats["item_key"] = r["item_key"]
        feats["l_phonemes"] = " ".join(p for w in g2p.text(r[text_col]) for p in w) if pd.notna(r[text_col]) else ""
        if lexicon is not None and pd.notna(r[text_col]):
            freq = lexicon.set_index("word")["log_frequency"]
            ws = tokenize_words(r[text_col])
            vals = [freq.get(w, np.nan) for w in ws]
            feats["l_mean_log_word_frequency"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        rows.append(feats)
    table = pd.DataFrame(rows)
    return table, {"status": "OK", "g2p_backend": g2p.backend,
                   "approximate": g2p.backend == "rules", "n_items": int(len(table))}
