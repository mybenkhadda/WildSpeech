"""Expected-text-conditioned error analysis of a recognised response.

align_ops              Levenshtein alignment with backtrace -> ops (M|S|I|D, ref token, hyp token, ref position)
error_summary          S / I / D counts and rates, error rate, LCS-based sequence preservation, partial correctness
position_profile       error rate in relative ref-position bins (initial / medial / final ...)
cv_profile             consonant vs vowel substitution/deletion rates (ARPAbet phones)
repetitions / restarts disfluency indicators from the hypothesis word sequence
clip_error_features    everything above for one (hypothesis text, expected text) pair, at word and phoneme level
"""
from __future__ import annotations

import re

import numpy as np

from .g2p import G2P, is_vowel


def align_ops(ref: list, hyp: list) -> list[tuple[str, object, object, int]]:
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("M" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1], max(i - 1, 0)))
            j -= 1
    return ops[::-1]


def lcs_len(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def error_summary(ref: list, hyp: list, prefix: str) -> dict:
    ops = align_ops(ref, hyp)
    n = max(len(ref), 1)
    s = sum(o[0] == "S" for o in ops)
    i = sum(o[0] == "I" for o in ops)
    d = sum(o[0] == "D" for o in ops)
    er = (s + i + d) / n
    return {f"{prefix}_sub": float(s), f"{prefix}_ins": float(i), f"{prefix}_del": float(d), f"{prefix}_er": float(er),
            f"{prefix}_sub_rate": s / n, f"{prefix}_ins_rate": i / n, f"{prefix}_del_rate": d / n,
            f"{prefix}_seq_preservation": lcs_len(ref, hyp) / n, f"{prefix}_partial_correct": float(max(0.0, 1.0 - min(er, 1.0))),
            f"{prefix}_exact": float(ref == hyp)}


def position_profile(ops, n_ref: int, bins: int = 3, prefix: str = "ph") -> dict:
    err = np.zeros(bins)
    tot = np.zeros(bins)
    for kind, _, _, pos in ops:
        if n_ref <= 0:
            break
        b = min(bins - 1, int(bins * pos / max(n_ref, 1)))
        if kind in ("M", "S", "D"):
            tot[b] += 1
        if kind != "M":
            err[b] += 1
    names = ["initial", "medial", "final"] if bins == 3 else [f"bin{k}" for k in range(bins)]
    return {f"{prefix}_err_{nm}": (err[k] / tot[k] if tot[k] else np.nan) for k, nm in enumerate(names)}


def cv_profile(ops, prefix: str = "ph") -> dict:
    out = {}
    for cls, test in (("vowel", True), ("consonant", False)):
        rel = [o for o in ops if o[1] is not None and is_vowel(o[1]) == test]
        n = len(rel)
        out[f"{prefix}_{cls}_err"] = (sum(o[0] in ("S", "D") for o in rel) / n) if n else np.nan
    return out


def repetitions(words: list[str]) -> float:
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    bigrams = list(zip(words, words[1:]))
    reps += sum(1 for k in range(len(bigrams) - 2) if bigrams[k] == bigrams[k + 2])
    return float(reps)


def restarts(hyp_words: list[str], ref_words: list[str]) -> float:
    """Fragments: a hyp token that is a strict prefix (>= 1 letter) of a ref word and is followed by a token that
    starts the same way (e.g. 'p- pot', 'swee sweet'), or an immediate repetition of a ref word."""
    refs = set(ref_words)
    cnt = 0
    for a, b in zip(hyp_words, hyp_words[1:]):
        if a != b and any(r.startswith(a) and r != a for r in refs) and b.startswith(a[:1]):
            cnt += 1
        elif a == b and a in refs:
            cnt += 1
    return float(cnt)


def _words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z' ]+", " ", str(text or "").lower()).split() if w]


def clip_error_features(hyp_text: str, ref_text: str, g2p: G2P | None, bins: int = 3, allow_approx: bool = False) -> tuple[dict, dict]:
    """Returns (numeric features, detail dict with ops strings for confusion / position analyses)."""
    hw, rw = _words(hyp_text), _words(ref_text)
    feats = {**error_summary(rw, hw, "w"), "n_words_hyp": float(len(hw)), "n_words_ref": float(len(rw)),
             "len_ratio": len(hw) / max(len(rw), 1), "empty": float(len(hw) == 0),
             "repetitions": repetitions(hw), "restarts": restarts(hw, rw)}
    hc, rc = list("".join(hw)), list("".join(rw))
    feats.update(error_summary(rc, hc, "c"))
    detail = {"ref_phones": "", "hyp_phones": "", "ops": ""}
    if g2p is not None and (g2p.exact or allow_approx):
        rp, hp = g2p.phonemes(" ".join(rw)), g2p.phonemes(" ".join(hw))
        ops = align_ops(rp, hp)
        feats.update(error_summary(rp, hp, "ph"))
        feats.update(position_profile(ops, len(rp), bins, "ph"))
        feats.update(cv_profile(ops, "ph"))
        feats["ph_per"] = feats["ph_er"]
        detail = {"ref_phones": " ".join(rp), "hyp_phones": " ".join(hp), "ops": ops_to_string(ops)}
    return feats, detail


def ops_to_string(ops) -> str:
    return "|".join(f"{k}:{r or '-'}:{h or '-'}:{p}" for k, r, h, p in ops)


def ops_from_string(s: str) -> list[tuple[str, str | None, str | None, int]]:
    out = []
    for part in str(s or "").split("|"):
        if not part:
            continue
        k, r, h, p = part.split(":")
        out.append((k, None if r == "-" else r, None if h == "-" else h, int(p)))
    return out


def confusion_counts(ops_strings, top: int | None = None):
    """(ref phone, hyp phone) counts including deletions ('∅' hyp) and insertions ('∅' ref)."""
    import pandas as pd

    pairs = []
    for s in ops_strings:
        for k, r, h, _ in ops_from_string(s):
            if k == "M":
                pairs.append((r, r))
            elif k == "S":
                pairs.append((r, h))
            elif k == "D":
                pairs.append((r, "∅"))
            else:
                pairs.append(("∅", h))
    if not pairs:
        return pd.DataFrame()
    df = pd.DataFrame(pairs, columns=["ref", "hyp"])
    mat = pd.crosstab(df["ref"], df["hyp"])
    if top:
        keep = df["ref"].value_counts().index[:top]
        mat = mat.reindex(index=[k for k in keep if k in mat.index])
        cols = [c for c in mat.columns if c in set(keep) or c == "∅"]
        mat = mat[cols]
    return mat
