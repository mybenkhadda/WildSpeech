"""G2P (reuses the V2 backend chain g2p_en -> cmudict -> rule proxy) + an articulatory feature table for the
39-phone ARPAbet set.

PhoneFeatures(p)   class (vowel/consonant), place, manner, voicing; vowels: height, backness, diphthong
phone_distance     0 for identical phones, otherwise the share of differing articulatory features in (0, 1]
                   (used as a substitution cost so 'B'->'P' counts less than 'B'->'IY')
consonant_clusters runs of >= 2 consecutive consonants inside a word (phone indices)
syllable_count     number of vowel nuclei
ExpectedPhones     cached expected-text phones per word with cluster / position annotations
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..phonetic.g2p import ARPA_CONSONANTS, ARPA_VOWELS, ARPABET, G2P, is_vowel, strip_stress  # noqa: F401

CONS = {
    "P": ("bilabial", "stop", "voiceless"), "B": ("bilabial", "stop", "voiced"), "M": ("bilabial", "nasal", "voiced"),
    "W": ("labiovelar", "approximant", "voiced"), "F": ("labiodental", "fricative", "voiceless"),
    "V": ("labiodental", "fricative", "voiced"), "TH": ("dental", "fricative", "voiceless"),
    "DH": ("dental", "fricative", "voiced"), "T": ("alveolar", "stop", "voiceless"), "D": ("alveolar", "stop", "voiced"),
    "N": ("alveolar", "nasal", "voiced"), "S": ("alveolar", "fricative", "voiceless"), "Z": ("alveolar", "fricative", "voiced"),
    "L": ("alveolar", "lateral", "voiced"), "R": ("alveolar", "approximant", "voiced"),
    "SH": ("postalveolar", "fricative", "voiceless"), "ZH": ("postalveolar", "fricative", "voiced"),
    "CH": ("postalveolar", "affricate", "voiceless"), "JH": ("postalveolar", "affricate", "voiced"),
    "Y": ("palatal", "approximant", "voiced"), "K": ("velar", "stop", "voiceless"), "G": ("velar", "stop", "voiced"),
    "NG": ("velar", "nasal", "voiced"), "HH": ("glottal", "fricative", "voiceless"),
}
VOWELS = {
    "IY": ("high", "front", False), "IH": ("high", "front", False), "EY": ("mid", "front", True),
    "EH": ("mid", "front", False), "AE": ("low", "front", False), "AA": ("low", "back", False),
    "AO": ("mid", "back", False), "OW": ("mid", "back", True), "UH": ("high", "back", False),
    "UW": ("high", "back", False), "AH": ("mid", "central", False), "ER": ("mid", "central", False),
    "AY": ("low", "central", True), "AW": ("low", "central", True), "OY": ("mid", "back", True),
}
SONORITY = {"stop": 1, "affricate": 2, "fricative": 3, "nasal": 4, "lateral": 5, "approximant": 6, "vowel": 7}


@dataclass(frozen=True)
class PhoneFeatures:
    phone: str
    cls: str
    a: str
    b: str
    c: str

    @property
    def sonority(self) -> int:
        return SONORITY["vowel"] if self.cls == "vowel" else SONORITY.get(self.b, 0)


@lru_cache(maxsize=None)
def features(p: str) -> PhoneFeatures | None:
    p = strip_stress(p)
    if p in CONS:
        return PhoneFeatures(p, "consonant", *CONS[p])
    if p in VOWELS:
        h, bk, d = VOWELS[p]
        return PhoneFeatures(p, "vowel", h, bk, "diphthong" if d else "monophthong")
    return None


@lru_cache(maxsize=None)
def phone_distance(a: str | None, b: str | None) -> float:
    if a is None or b is None:
        return 1.0
    if strip_stress(a) == strip_stress(b):
        return 0.0
    fa, fb = features(a), features(b)
    if fa is None or fb is None:
        return 1.0
    if fa.cls != fb.cls:
        return 1.0
    diff = sum(x != y for x, y in ((fa.a, fb.a), (fa.b, fb.b), (fa.c, fb.c)))
    return max(0.25, diff / 3.0)


def substitution_type(ref: str, hyp: str) -> str:
    """voicing | place | manner | vowel_height | vowel_backness | vowel_quality | cv_swap | other."""
    fa, fb = features(ref), features(hyp)
    if fa is None or fb is None:
        return "other"
    if fa.cls != fb.cls:
        return "cv_swap"
    if fa.cls == "consonant":
        d = [n for n, x, y in (("place", fa.a, fb.a), ("manner", fa.b, fb.b), ("voicing", fa.c, fb.c)) if x != y]
    else:
        d = [n for n, x, y in (("vowel_height", fa.a, fb.a), ("vowel_backness", fa.b, fb.b), ("vowel_quality", fa.c, fb.c)) if x != y]
    return d[0] if len(d) == 1 else ("multi_feature" if d else "identical")


def consonant_clusters(phones: list[str]) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] of runs of >= 2 consonants."""
    out, i = [], 0
    while i < len(phones):
        if not is_vowel(phones[i]):
            j = i
            while j < len(phones) and not is_vowel(phones[j]):
                j += 1
            if j - i >= 2:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def syllable_count(phones: list[str]) -> int:
    return int(sum(is_vowel(p) for p in phones))


class ExpectedPhones:
    """Expected-text phones per item with annotations, cached by text."""

    def __init__(self, backend: str = "auto", allow_approx: bool = False):
        self.g2p = G2P(backend)
        self.allow_approx = bool(allow_approx)
        self._cache: dict[str, dict] = {}

    @property
    def exact(self) -> bool:
        return self.g2p.exact

    @property
    def usable(self) -> bool:
        return self.g2p.exact or self.allow_approx

    def __call__(self, text: str) -> dict:
        t = str(text or "")
        if t in self._cache:
            return self._cache[t]
        words = self.g2p.words(t) if self.usable else []
        phones, word_of, in_cluster = [], [], []
        for wi, w in enumerate(words):
            cl = consonant_clusters(w)
            mark = [False] * len(w)
            for s, e in cl:
                for k in range(s, e):
                    mark[k] = True
            phones += w
            word_of += [wi] * len(w)
            in_cluster += mark
        d = {"phones": phones, "word_of": word_of, "in_cluster": in_cluster, "n_words": len(words),
             "n_syllables": syllable_count(phones), "n_clusters": sum(len(consonant_clusters(w)) for w in words),
             "backend": self.g2p.backend, "exact": self.g2p.exact}
        self._cache[t] = d
        return d

    def phonemes(self, text: str) -> list[str]:
        return list(self(text)["phones"])
