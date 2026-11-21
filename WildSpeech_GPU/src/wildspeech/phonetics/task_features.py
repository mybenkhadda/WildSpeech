"""Timing and task-specific features (label-free).

timing_features(spans, expected, frame_s, pause_frames)    (CTC forced-alignment spans; NaN without spans)
    tm_onset_s            start of the first aligned expected token
    tm_phone_dur_mean/cv  aligned token durations; tm_vowel_dur_mean, tm_consonant_dur_mean
    tm_gap_mean/cv        inter-token transition gaps (sound-transition timing)
    tm_n_pauses           gaps >= pause_frames; tm_pause_within_word / tm_pause_between_words (placement)
    tm_longest_pause_pos  relative position (0..1) of the longest pause in the expected sequence
    tm_artic_rate         aligned tokens per second of speech (excluding pauses)
    tm_s_per_phone, tm_s_per_syllable   complexity-normalised timing
task_features(task, word_f, phon_f, ops, expected, timing)
    blending            bl_integrated (one recognised word and near-correct phones), bl_segmented (pauses
                        between phones of the target word), bl_transition_cv, bl_partial (some but not all
                        phones), bl_multiword_attempt (several short words recognised for one target)
    deletion            dl_exact, dl_excessive (phones missing, nothing added), dl_retained (extra phones kept:
                        insertions), dl_retained_initial/final, dl_manipulation (substitutions), dl_len_diff
    sentence_repetition sr_omission_rate, sr_sequencing (word order), sr_completeness, sr_n_words,
                        sr_first_half_err / sr_second_half_err (position of breakdown)
"""
from __future__ import annotations

import numpy as np

from .alignment import Op
from .g2p import is_vowel


def timing_features(spans: list[dict], expected: dict | None, frame_s: float = 0.02, pause_frames: int = 5) -> dict:
    keys = ("tm_onset_s", "tm_phone_dur_mean", "tm_phone_dur_cv", "tm_vowel_dur_mean", "tm_consonant_dur_mean",
            "tm_gap_mean", "tm_gap_cv", "tm_n_pauses", "tm_pause_within_word", "tm_pause_between_words",
            "tm_longest_pause_pos", "tm_artic_rate", "tm_s_per_phone", "tm_s_per_syllable")
    out = {k: np.nan for k in keys}
    sp = [s for s in spans if str(s.get("p")) != "|"]
    if not sp:
        return out
    st = np.array([int(s["s"]) for s in sp])
    en = np.array([int(s["e"]) for s in sp])
    dur = (en - st) * frame_s
    gaps = np.clip(st[1:] - en[:-1], 0, None) if len(sp) > 1 else np.zeros(0)
    labs = [str(s["p"]) for s in sp]
    out["tm_onset_s"] = float(st[0] * frame_s)
    out["tm_phone_dur_mean"] = float(dur.mean())
    out["tm_phone_dur_cv"] = float(dur.std() / max(dur.mean(), 1e-6))
    v = [d for d, lab in zip(dur, labs) if is_vowel(lab)]
    c = [d for d, lab in zip(dur, labs) if not is_vowel(lab)]
    out["tm_vowel_dur_mean"] = float(np.mean(v)) if v else np.nan
    out["tm_consonant_dur_mean"] = float(np.mean(c)) if c else np.nan
    if len(gaps):
        g = gaps * frame_s
        out["tm_gap_mean"] = float(g.mean())
        out["tm_gap_cv"] = float(g.std() / max(g.mean(), 1e-6)) if g.mean() > 0 else 0.0
        pause = gaps >= pause_frames
        out["tm_n_pauses"] = float(pause.sum())
        wo = (expected or {}).get("word_of", [])
        if len(wo) == len(sp):
            within = [pause[k] and wo[k] == wo[k + 1] for k in range(len(gaps))]
            between = [pause[k] and wo[k] != wo[k + 1] for k in range(len(gaps))]
            out["tm_pause_within_word"] = float(np.sum(within))
            out["tm_pause_between_words"] = float(np.sum(between))
        if pause.any():
            out["tm_longest_pause_pos"] = float(np.argmax(gaps) / max(len(gaps) - 1, 1))
        speech_s = float(dur.sum() + (gaps[~pause].sum() * frame_s if len(gaps) else 0.0))
    else:
        out["tm_gap_mean"], out["tm_gap_cv"], out["tm_n_pauses"] = 0.0, 0.0, 0.0
        speech_s = float(dur.sum())
    out["tm_artic_rate"] = float(len(sp) / max(speech_s, 1e-3))
    total = float((en[-1] - st[0]) * frame_s)
    n_ph = len((expected or {}).get("phones", [])) or len(sp)
    out["tm_s_per_phone"] = total / max(n_ph, 1)
    ns = (expected or {}).get("n_syllables") or 0
    out["tm_s_per_syllable"] = total / ns if ns else np.nan
    return out


def _blending(word_f: dict, phon_f: dict, timing: dict) -> dict:
    n_hyp = word_f.get("w_n_hyp", np.nan)
    near = phon_f.get("ph_near_correct", np.nan)
    within = timing.get("tm_pause_within_word", np.nan)
    npz = timing.get("tm_n_pauses", np.nan)
    seg = within if np.isfinite(within) else npz
    per = phon_f.get("ph_per", np.nan)
    return {"bl_integrated": float(n_hyp == 1 and np.isfinite(near) and near >= 0.75) if np.isfinite(n_hyp) else np.nan,
            "bl_segmented": float(seg > 0) if np.isfinite(seg) else np.nan,
            "bl_transition_cv": timing.get("tm_gap_cv", np.nan),
            "bl_partial": float(0 < per < 1) if np.isfinite(per) else np.nan,
            "bl_multiword_attempt": float(n_hyp >= 2) if np.isfinite(n_hyp) else np.nan}


def _deletion(phon_f: dict, ops: list[Op]) -> dict:
    if not ops:
        return {k: np.nan for k in ("dl_exact", "dl_excessive", "dl_retained", "dl_retained_initial", "dl_retained_final",
                                     "dl_manipulation", "dl_len_diff")}
    n_ref = phon_f.get("ph_n_ref", np.nan)
    ins = [o for o in ops if o.op == "I"]
    first_ref = next((k for k, o in enumerate(ops) if o.op != "I"), len(ops))
    last_ref = max((k for k, o in enumerate(ops) if o.op != "I"), default=-1)
    return {"dl_exact": phon_f.get("ph_exact", np.nan),
            "dl_excessive": float(phon_f.get("ph_del_rate", 0) > 0 and not ins),
            "dl_retained": float(len(ins) / max(n_ref, 1)),
            "dl_retained_initial": float(any(k < first_ref for k, o in enumerate(ops) if o.op == "I")),
            "dl_retained_final": float(any(k > last_ref for k, o in enumerate(ops) if o.op == "I")),
            "dl_manipulation": phon_f.get("ph_sub_rate", np.nan),
            "dl_len_diff": float(phon_f.get("ph_n_hyp", np.nan) - n_ref)}


def _sentence(word_f: dict, hyp_words: list[str], ref_words: list[str]) -> dict:
    from .alignment import align

    ops = align(ref_words, hyp_words)
    n = len(ref_words)
    half = n / 2.0
    e1 = sum(o.op != "M" for o in ops if o.op != "I" and o.i < half)
    e2 = sum(o.op != "M" for o in ops if o.op != "I" and o.i >= half)
    return {"sr_omission_rate": word_f.get("w_del_rate", np.nan), "sr_sequencing": word_f.get("w_order", np.nan),
            "sr_completeness": word_f.get("w_completeness", np.nan), "sr_n_words": float(n),
            "sr_first_half_err": e1 / max(np.ceil(half), 1), "sr_second_half_err": e2 / max(n - np.ceil(half), 1)}


def task_features(task: str, word_f: dict, phon_f: dict, ops: list[Op], timing: dict,
                  hyp_words: list[str] | None = None, ref_words: list[str] | None = None) -> dict:
    t = str(task).lower()
    if t == "blending":
        return _blending(word_f, phon_f, timing)
    if t == "deletion":
        return _deletion(phon_f, ops)
    if t == "sentence_repetition":
        return _sentence(word_f, hyp_words or [], ref_words or [])
    return {}
