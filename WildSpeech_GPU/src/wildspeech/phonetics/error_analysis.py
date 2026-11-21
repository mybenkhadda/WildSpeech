"""Expected-text comparison of a recognised response (label-free).

word_features(hyp_text, ref_text)
    w_wer, w_sub/del/ins_rate, w_completeness (share of expected words recognised), w_order (Kendall tau of
    matched words), w_exact, w_empty, w_len_ratio, w_repetitions, w_self_corrections, c_cer
phoneme_features(expected, hyp_phones)
    ph_per, ph_sub/del/ins_rate, ph_consonant_err, ph_vowel_err, ph_cluster_acc, ph_cluster_reduction,
    ph_initial_omission, ph_final_omission, ph_err_initial/medial/final, ph_err_pos_mean, ph_order_lcs,
    ph_sub_<type>_rate (voicing / place / manner / vowel_*), ph_sub_distance_mean (phonological closeness of
    substitutions), ph_near_correct (1 - weighted phonological edit distance / len), ph_exact
clip_details(...)    alignment strings for confusion matrices / heatmaps / alignment figures
confusion_table(ops) long (ref, hyp, op) counts
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .alignment import Op, align, counts, lcs_ratio, ops_from_string, ops_to_string, order_preservation, phonological_align
from .g2p import is_vowel, phone_distance, substitution_type

SUB_TYPES = ("voicing", "place", "manner", "vowel_height", "vowel_backness", "vowel_quality", "cv_swap", "multi_feature")


def words_of(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z' ]+", " ", str(text or "").lower()).split() if w]


def repetitions(words: list[str]) -> int:
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b)
    bi = list(zip(words, words[1:]))
    reps += sum(1 for k in range(len(bi) - 2) if bi[k] == bi[k + 2])
    return int(reps)


def self_corrections(hyp: list[str], ref: list[str]) -> int:
    """A fragment / wrong attempt immediately followed by an expected word that starts the same way
    ('ca- cat', 'bat cat' when only 'cat' is expected), or an immediate repeat of an expected word."""
    refs = set(ref)
    n = 0
    for a, b in zip(hyp, hyp[1:]):
        if a == b and a in refs:
            n += 1
        elif a != b and b in refs and a not in refs and (b.startswith(a) or a[:1] == b[:1]):
            n += 1
    return int(n)


def word_features(hyp_text: str, ref_text: str) -> dict:
    hw, rw = words_of(hyp_text), words_of(ref_text)
    ops = align(rw, hw)
    c = counts(ops)
    n = max(len(rw), 1)
    hc, rc = list("".join(hw)), list("".join(rw))
    cops = align(rc, hc)
    cc = counts(cops)
    return {"w_wer": (c["S"] + c["D"] + c["I"]) / n, "w_sub_rate": c["S"] / n, "w_del_rate": c["D"] / n,
            "w_ins_rate": c["I"] / n, "w_completeness": c["M"] / n, "w_order": order_preservation(rw, hw),
            "w_exact": float(hw == rw), "w_empty": float(len(hw) == 0), "w_len_ratio": len(hw) / n,
            "w_n_hyp": float(len(hw)), "w_n_ref": float(len(rw)), "w_repetitions": float(repetitions(hw)),
            "w_self_corrections": float(self_corrections(hw, rw)),
            "c_cer": (cc["S"] + cc["D"] + cc["I"]) / max(len(rc), 1)}


def _bins(i: int, n: int, k: int = 3) -> int:
    return min(k - 1, int(k * i / max(n, 1)))


def phoneme_features(expected: dict, hyp_phones: list[str], bins: int = 3) -> tuple[dict, list[Op]]:
    rp = list(expected.get("phones", []))
    if not rp:
        return {}, []
    ops = phonological_align(rp, hyp_phones)
    c = counts(ops)
    n = len(rp)
    f = {"ph_per": (c["S"] + c["D"] + c["I"]) / n, "ph_sub_rate": c["S"] / n, "ph_del_rate": c["D"] / n,
         "ph_ins_rate": c["I"] / n, "ph_exact": float(rp == list(hyp_phones)), "ph_n_ref": float(n),
         "ph_n_hyp": float(len(hyp_phones)), "ph_order_lcs": lcs_ratio(rp, hyp_phones)}
    ref_ops = [o for o in ops if o.op != "I"]
    for cls, test in (("vowel", True), ("consonant", False)):
        rel = [o for o in ref_ops if is_vowel(o.ref) == test]
        f[f"ph_{cls}_err"] = (sum(o.op != "M" for o in rel) / len(rel)) if rel else np.nan
    cl = expected.get("in_cluster", [])
    cl_ops = [o for o in ref_ops if o.i < len(cl) and cl[o.i]]
    f["ph_cluster_acc"] = (sum(o.op == "M" for o in cl_ops) / len(cl_ops)) if cl_ops else np.nan
    f["ph_cluster_reduction"] = (sum(o.op == "D" for o in cl_ops) / len(cl_ops)) if cl_ops else np.nan
    first = next((o for o in ref_ops if o.i == 0), None)
    last = next((o for o in ref_ops if o.i == n - 1), None)
    f["ph_initial_omission"] = float(first is not None and first.op == "D")
    f["ph_final_omission"] = float(last is not None and last.op == "D")
    err, tot = np.zeros(bins), np.zeros(bins)
    pos_err = []
    for o in ops:
        b = _bins(min(o.i, n - 1), n, bins)
        if o.op != "I":
            tot[b] += 1
        if o.op != "M":
            err[b] += 1
            pos_err.append(min(o.i, n - 1) / max(n - 1, 1))
    names = ["initial", "medial", "final"] if bins == 3 else [f"bin{k}" for k in range(bins)]
    for k, nm in enumerate(names):
        f[f"ph_err_{nm}"] = err[k] / tot[k] if tot[k] else np.nan
    f["ph_err_pos_mean"] = float(np.mean(pos_err)) if pos_err else np.nan
    subs = [o for o in ops if o.op == "S"]
    st = Counter(substitution_type(o.ref, o.hyp) for o in subs)
    for t in SUB_TYPES:
        f[f"ph_sub_{t}_rate"] = st.get(t, 0) / n
    f["ph_sub_distance_mean"] = float(np.mean([phone_distance(o.ref, o.hyp) for o in subs])) if subs else np.nan
    wdist = sum(phone_distance(o.ref, o.hyp) for o in subs) + c["D"] + c["I"]
    f["ph_near_correct"] = float(max(0.0, 1.0 - wdist / n))
    return f, ops


def clip_details(ops: list[Op], expected: dict, hyp_phones: list[str]) -> dict:
    return {"ref_phones": " ".join(expected.get("phones", [])), "hyp_phones": " ".join(hyp_phones),
            "ops": ops_to_string(ops)}


def confusion_table(ops_strings, model_key: str | None = None) -> pd.DataFrame:
    """Long counts (ref, hyp, op); deletions hyp='∅', insertions ref='∅'."""
    cnt: Counter = Counter()
    for s in ops_strings:
        for o in ops_from_string(s):
            r = o.ref if o.ref is not None else "∅"
            h = o.hyp if o.hyp is not None else "∅"
            cnt[(r, h, o.op)] += 1
    df = pd.DataFrame([{"ref": r, "hyp": h, "op": op, "count": n} for (r, h, op), n in cnt.items()])
    if model_key is not None and not df.empty:
        df["model_key"] = model_key
    return df


def confusion_matrix(long: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Row-normalised ref x hyp matrix over the `top` most frequent reference phones (+ ∅ column)."""
    if long is None or long.empty:
        return pd.DataFrame()
    m = long.pivot_table(index="ref", columns="hyp", values="count", aggfunc="sum", fill_value=0)
    keep = long[long["ref"] != "∅"].groupby("ref")["count"].sum().sort_values(ascending=False).index[:top]
    m = m.reindex(index=list(keep))
    cols = [c for c in list(keep) + ["∅"] if c in m.columns]
    m = m[cols]
    return m.div(m.sum(1).replace(0, np.nan), axis=0)
