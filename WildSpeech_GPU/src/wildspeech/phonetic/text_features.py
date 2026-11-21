"""Linguistic complexity of the expected text (item-level, label-free, deterministic)."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

VOWELS = set("aeiouy")
DIGRAPHS = ("sh", "ch", "th", "ph", "wh", "ck", "ng", "qu")
_g2p = None


def _get_g2p(backend: str = "auto"):
    global _g2p
    if _g2p is not None or backend == "rule":
        return _g2p
    try:
        from g2p_en import G2p

        _g2p = G2p()
    except Exception:
        _g2p = False
    return _g2p


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def consonant_clusters(word: str) -> list[int]:
    w = re.sub(r"[^a-z]", "", word.lower())
    return [len(c) for c in re.findall(r"[^aeiouy]{2,}", w)]


@lru_cache(maxsize=20000)
def phonemes(text: str, backend: str = "auto") -> tuple[str, ...]:
    g = _get_g2p(backend)
    if g:
        return tuple(p for p in g(text) if p.strip() and p not in {",", ".", "'", "?", "!"})
    # rule-based proxy: letters with digraphs merged
    out = []
    for word in text.lower().split():
        w = re.sub(r"[^a-z]", "", word)
        i = 0
        while i < len(w):
            if w[i:i + 2] in DIGRAPHS:
                out.append(w[i:i + 2])
                i += 2
            else:
                out.append(w[i])
                i += 1
    return tuple(out)


def text_complexity(text: str, backend: str = "auto") -> dict:
    t = str(text or "").strip()
    words = re.findall(r"[A-Za-z']+", t)
    if not words:
        return {k: np.nan for k in ("n_words", "n_chars", "n_syllables", "mean_word_len", "max_word_len", "n_clusters",
                                    "max_cluster", "n_phonemes", "vowel_ratio", "n_digraphs", "multisyllabic_frac")}
    syl = [syllables(w) for w in words]
    clusters = [c for w in words for c in consonant_clusters(w)]
    letters = re.sub(r"[^a-z]", "", t.lower())
    return {"n_words": float(len(words)), "n_chars": float(len(letters)), "n_syllables": float(sum(syl)),
            "mean_word_len": float(np.mean([len(w) for w in words])), "max_word_len": float(max(len(w) for w in words)),
            "n_clusters": float(len(clusters)), "max_cluster": float(max(clusters) if clusters else 0),
            "n_phonemes": float(len(phonemes(t.lower(), backend))),
            "vowel_ratio": float(sum(ch in VOWELS for ch in letters) / max(len(letters), 1)),
            "n_digraphs": float(sum(letters.count(d) for d in DIGRAPHS)),
            "multisyllabic_frac": float(np.mean([s >= 2 for s in syl]))}


def item_text_table(clips: pd.DataFrame, backend: str = "auto") -> pd.DataFrame:
    items = clips[["item_id", "task", "expected_text"]].drop_duplicates("item_id")
    rows = [{"item_id": r.item_id, **text_complexity(r.expected_text, backend)} for r in items.itertuples(index=False)]
    return pd.DataFrame(rows)


def edit_distance(a: list | str, b: list | str) -> int:
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def normalized_text(s: str) -> str:
    s = re.sub(r"[^a-z' ]+", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def deviation_metrics(hyp: str, ref: str, backend: str = "auto") -> dict:
    h, r = normalized_text(hyp), normalized_text(ref)
    hw, rw = h.split(), r.split()
    cer = edit_distance(h.replace(" ", ""), r.replace(" ", "")) / max(len(r.replace(" ", "")), 1)
    wer = edit_distance(hw, rw) / max(len(rw), 1)
    ph_h, ph_r = phonemes(h, backend), phonemes(r, backend)
    per = edit_distance(list(ph_h), list(ph_r)) / max(len(ph_r), 1)
    return {"cer": float(cer), "wer": float(wer), "per": float(per), "len_ratio": float(len(hw) / max(len(rw), 1)),
            "empty": float(len(hw) == 0), "exact_match": float(h == r)}
