"""Measured phonetic error features — ONLY from phone strings produced by a phone recognizer in the export.

Hypothesis : the recognizer's phone sequence (ARPAbet or IPA; IPA is mapped with phonetic.g2p.ipa_to_arpa)
Reference  : the expected pronunciation of the EXPECTED text (phonetic.g2p.G2P, g2p_en / CMUdict; the rule-based
             letter proxy is refused unless v4.phonetic.allow_approximate_g2p is true)
A G2P conversion of a recognized WORD transcript is not a measured phonetic transcription and is never used here.
When no exported model carries enough phone strings, every phonetic experiment is reported UNAVAILABLE.

Per clip and phone model (existing phonetic.error_features): ph_sub/ins/del (+ _rate), ph_er (= PER),
ph_seq_preservation, ph_partial_correct, ph_exact, ph_err_initial/medial/final, ph_vowel_err, ph_consonant_err,
ph_map_rate (share of exported phone tokens mapped to ARPAbet) and the alignment ops string (confusions).
Within-student phonetic consistency is the `sd` statistic of the student aggregation and the split-half
reliability computed for H5.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..phonetic.error_features import align_ops, cv_profile, error_summary, ops_to_string, position_profile
from ..phonetic.g2p import ARPABET, ipa_to_arpa, ipa_to_arpa_table, strip_stress

_ARPA = set(ARPABET)


def parse_phones(s: str) -> tuple[list[str], float]:
    """Exported phone string -> (ARPAbet list, share of tokens mapped)."""
    s = str(s or "").strip()
    if not s:
        return [], np.nan
    toks = [t for t in re.split(r"[\s|,;/]+", s) if t]
    if len(toks) == 1 and len(toks[0]) > 3 and strip_stress(toks[0]) not in _ARPA:
        toks = _split_ipa(toks[0])                       # contiguous IPA string
    out, mapped = [], 0
    for t in toks:
        a = strip_stress(t)
        if a in _ARPA:
            out.append(a)
            mapped += 1
            continue
        m = ipa_to_arpa(t)
        if m:
            out.append(m)
            mapped += 1
    return out, (mapped / len(toks)) if toks else np.nan


def _split_ipa(s: str) -> list[str]:
    """Greedy longest-match segmentation of a contiguous IPA string against the known IPA inventory."""
    import unicodedata

    keys = sorted(ipa_to_arpa_table().keys(), key=len, reverse=True)
    t = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    t = t.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    out, i = [], 0
    while i < len(t):
        hit = next((k for k in keys if k and t.startswith(k, i)), None)
        if hit is None:
            out.append(t[i])
            i += 1
        else:
            out.append(hit)
            i += len(hit)
    return out


def phone_models(coverage: pd.DataFrame, models: list[str], min_frac: float) -> list[str]:
    if coverage is None or coverage.empty:
        return []
    c = coverage[(coverage["split"] == "train") & coverage["model_key"].isin(models)]
    frac = c["n_phone_strings"] / c["n_metadata_clips"].replace(0, np.nan)
    return sorted(c.loc[frac >= float(min_frac), "model_key"])


def availability(coverage: pd.DataFrame, models: list[str], cfg_ph: dict, g2p) -> tuple[list[str], pd.DataFrame]:
    min_frac = float(cfg_ph.get("min_nonempty_frac", 0.3))
    pm = phone_models(coverage, models, min_frac)
    rows = []
    for m in models:
        c = coverage[(coverage["model_key"] == m) & (coverage["split"] == "train")]
        n = int(c["n_phone_strings"].iloc[0]) if not c.empty else 0
        tot = int(c["n_metadata_clips"].iloc[0]) if not c.empty else 0
        rows.append({"model_key": m, "n_phone_strings_train": n, "n_train_clips": tot,
                     "status": "AVAILABLE" if m in pm else "UNAVAILABLE",
                     "reason": "" if m in pm else f"phone strings on {n}/{tot} train clips (< {min_frac:.0%}) — the "
                                                  "export carries no measured phonetic transcription for this model"})
    ok_g2p = g2p is not None and (getattr(g2p, "exact", False) or bool(cfg_ph.get("allow_approximate_g2p", False)))
    if pm and not ok_g2p:
        rows.append({"model_key": "*", "status": "UNAVAILABLE",
                     "reason": "no exact G2P backend (g2p_en / nltk cmudict) for the expected-text reference phones"})
        pm = []
    if not pm:
        rows.append({"model_key": "*", "status": "UNAVAILABLE",
                     "reason": "no measured phone predictions: phonetic experiments (M3, H5) are not executed"})
    return pm, pd.DataFrame(rows)


def build_clip_phone_table(joined: pd.DataFrame, models: list[str], g2p, bins: int = 3) -> pd.DataFrame:
    d = joined[joined["model_key"].isin(models) & joined["asr_ok"] & joined["has_phones"].astype(bool)]
    ref_cache: dict[str, list[str]] = {}
    rows = []
    for r in d.itertuples(index=False):
        ref = str(r.expected_text or "")
        if ref not in ref_cache:
            ref_cache[ref] = list(g2p.phonemes(ref))
        rp = ref_cache[ref]
        hp, rate = parse_phones(r.phones)
        if not rp:
            continue
        ops = align_ops(rp, hp)
        f = {**error_summary(rp, hp, "ph"), **position_profile(ops, len(rp), bins, "ph"), **cv_profile(ops, "ph")}
        f["ph_map_rate"] = rate
        rows.append({FILE: getattr(r, FILE), "model_key": r.model_key, "split": r.split, ID: getattr(r, ID),
                     "task": r.task, "asr_ok": True, "ref_phones": " ".join(rp), "hyp_phones": " ".join(hp),
                     "ops": ops_to_string(ops), **f})
    return pd.DataFrame(rows)


def consensus_phone_table(long: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if long is None or long.empty:
        return pd.DataFrame(columns=[FILE])
    cols = [c for c in cols if c in long.columns]
    g = long.groupby(FILE)
    t = g[cols].mean()
    t["n_phone_models"] = g["model_key"].nunique().astype(float)
    return t.reset_index()


def confusion_matrix(long: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    from ..phonetic.error_features import confusion_counts

    if long is None or long.empty or "ops" not in long:
        return pd.DataFrame()
    tr = long[long["split"] == "train"]
    return confusion_counts(tr["ops"].tolist(), top=top)
