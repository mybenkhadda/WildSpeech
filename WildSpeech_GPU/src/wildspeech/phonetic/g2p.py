"""Grapheme-to-phoneme conversion and the ARPAbet <-> IPA phone inventory.

Backends (first available, unless forced): g2p_en (neural + CMUdict), nltk CMUdict, rule-based letter proxy.
Only the first two produce real phonemes (`exact=True`); phoneme-level metrics are skipped for the rule-based
proxy unless v2.phonetic.allow_approximate_g2p is true. Stress digits are removed (39-phone ARPAbet set).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ARPA_CONSONANTS = {"B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T",
                   "TH", "V", "W", "Y", "Z", "ZH"}
ARPABET = sorted(ARPA_VOWELS | ARPA_CONSONANTS)

ARPA_TO_IPA = {"AA": ["ɑː", "ɑ", "a"], "AE": ["æ"], "AH": ["ʌ", "ə", "ɐ"], "AO": ["ɔː", "ɔ"], "AW": ["aʊ"],
               "AY": ["aɪ"], "B": ["b"], "CH": ["tʃ"], "D": ["d"], "DH": ["ð"], "EH": ["ɛ"], "ER": ["ɚ", "ɝ", "ɜː", "ɜ"],
               "EY": ["eɪ", "e"], "F": ["f"], "G": ["ɡ", "g"], "HH": ["h"], "IH": ["ɪ"], "IY": ["iː", "i"], "JH": ["dʒ"],
               "K": ["k"], "L": ["l", "ɫ"], "M": ["m"], "N": ["n"], "NG": ["ŋ"], "OW": ["oʊ", "əʊ", "o"], "OY": ["ɔɪ"],
               "P": ["p"], "R": ["ɹ", "r"], "S": ["s"], "SH": ["ʃ"], "T": ["t"], "TH": ["θ"], "UH": ["ʊ"],
               "UW": ["uː", "u"], "V": ["v"], "W": ["w"], "Y": ["j"], "Z": ["z"], "ZH": ["ʒ"]}
_EXTRA_IPA = {"ɾ": "T", "ʔ": None, "x": "HH", "l̩": "L", "n̩": "N", "m̩": "M", "ɑɹ": "AA", "ɔɹ": "AO", "ɪɹ": "IH",
              "ɛɹ": "EH", "ʊɹ": "UH", "a": "AA", "ɒ": "AA", "ɜ": "ER", "ᵻ": "IH", "ɨ": "IH", "ʉ": "UW", "ɐ": "AH"}


def _norm_ipa(tok: str) -> str:
    t = unicodedata.normalize("NFD", str(tok))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("ˈ", "").replace("ˌ", "").replace("ː", "").replace(":", "").strip()


@lru_cache(maxsize=1)
def ipa_to_arpa_table() -> dict[str, str | None]:
    tab: dict[str, str | None] = {}
    for a, ipas in ARPA_TO_IPA.items():
        for i in ipas:
            tab.setdefault(_norm_ipa(i), a)
    for i, a in _EXTRA_IPA.items():
        tab.setdefault(_norm_ipa(i), a)
    return tab


def ipa_to_arpa(tok: str) -> str | None:
    """Map one IPA token (possibly with length/stress marks) to an ARPAbet phone, None if unmappable."""
    n = _norm_ipa(tok)
    if not n:
        return None
    tab = ipa_to_arpa_table()
    if n in tab:
        return tab[n]
    if n[:-1] in tab and len(n) > 1:
        return tab[n[:-1]]
    return tab.get(n[0])


def strip_stress(p: str) -> str:
    return re.sub(r"\d", "", str(p)).upper()


def is_vowel(p: str) -> bool:
    return strip_stress(p) in ARPA_VOWELS


class G2P:
    def __init__(self, backend: str = "auto"):
        self.backend, self.exact, self._impl = "rule", False, None
        order = ["g2p_en", "cmudict"] if backend == "auto" else [backend]
        for b in order:
            if b == "g2p_en":
                try:
                    from g2p_en import G2p

                    self._impl, self.backend, self.exact = G2p(), "g2p_en", True
                    break
                except Exception:  # noqa: BLE001 - backend unavailable
                    continue
            if b == "cmudict":
                try:
                    from nltk.corpus import cmudict

                    self._impl, self.backend, self.exact = cmudict.dict(), "cmudict", True
                    break
                except Exception:  # noqa: BLE001
                    continue
        self._cache: dict[str, tuple[str, ...]] = {}

    def word(self, w: str) -> tuple[str, ...]:
        w = re.sub(r"[^a-z']", "", w.lower())
        if not w:
            return ()
        if w in self._cache:
            return self._cache[w]
        if self.backend == "g2p_en":
            out = tuple(strip_stress(p) for p in self._impl(w) if strip_stress(p) in ARPA_VOWELS | ARPA_CONSONANTS)
        elif self.backend == "cmudict":
            prons = self._impl.get(w)
            out = tuple(strip_stress(p) for p in prons[0]) if prons else self._rule(w)
        else:
            out = self._rule(w)
        self._cache[w] = out
        return out

    @staticmethod
    def _rule(w: str) -> tuple[str, ...]:
        """Letter-to-phone proxy (approximate): digraphs merged, letters mapped to the closest ARPAbet phone."""
        dig = {"sh": "SH", "ch": "CH", "th": "TH", "ph": "F", "wh": "W", "ck": "K", "ng": "NG", "qu": "K"}
        let = {"a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G", "h": "HH", "i": "IH", "j": "JH",
               "k": "K", "l": "L", "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T",
               "u": "AH", "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z"}
        out, i = [], 0
        while i < len(w):
            if w[i:i + 2] in dig:
                out.append(dig[w[i:i + 2]])
                i += 2
                continue
            if w[i] in let:
                out.append(let[w[i]])
            i += 1
        return tuple(out)

    def phonemes(self, text: str) -> list[str]:
        out: list[str] = []
        for w in re.findall(r"[A-Za-z']+", str(text or "")):
            out += list(self.word(w))
        return out

    def words(self, text: str) -> list[list[str]]:
        return [list(self.word(w)) for w in re.findall(r"[A-Za-z']+", str(text or ""))]
