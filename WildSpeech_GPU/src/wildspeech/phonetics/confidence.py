"""Confidence features from what a model ACTUALLY exposes (never fabricated).

token_confidence(row)        only when has_token_probs: mean/min/p10 token log-prob, low-confidence token share,
                             mean next-token entropy (Whisper), token-probability CV. Otherwise every value is NaN
                             and `cf_has_token_probs` = 0.
posterior_confidence(row, …) CTC models only (frame posteriors):
    cf_expected_ll           log P(expected | audio) per target token (CTC forward)
    cf_free_ll               mean best-path frame log-prob
    cf_expected_gap          cf_expected_ll - (free ll per token): <= 0; large negative = the audio supports the
                             model's free hypothesis much better than the expected response
    cf_gop_mean/min/p10      forced-alignment goodness of pronunciation of the expected phones
    cf_low_conf_frac         share of expected tokens with mean target log-posterior < threshold
    cf_vowel_gop, cf_consonant_gop, cf_cluster_gop
    cf_post_mean             mean target posterior over aligned frames
    cf_frame_entropy         speech-frame posterior entropy (uncertainty); cf_blank_frac
    cf_supported_frac        share of expected phones that are BOTH aligned with posterior >= 0.5 AND matched in
                             the free recognition (phonological alignment). The forced alignment alone always
                             "finds" every expected phone; this measure requires independent acoustic support.
"""
from __future__ import annotations

import json

import numpy as np

from .alignment import Op, parse_spans
from .g2p import is_vowel


def _num(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def token_confidence(row: dict, low: float = -2.0) -> dict:
    keys = ("cf_tok_lp_mean", "cf_tok_lp_min", "cf_tok_lp_p10", "cf_tok_low_frac", "cf_tok_lp_cv", "cf_tok_entropy")
    has = bool(row.get("has_token_probs")) and bool(row.get("token_logprobs"))
    if not has:
        return {**{k: np.nan for k in keys}, "cf_has_token_probs": 0.0}
    try:
        lp = np.array(json.loads(row["token_logprobs"]), float)
    except Exception:  # noqa: BLE001
        return {**{k: np.nan for k in keys}, "cf_has_token_probs": 0.0}
    lp = lp[np.isfinite(lp)]
    if not len(lp):
        return {**{k: np.nan for k in keys}, "cf_has_token_probs": 0.0}
    p = np.exp(lp)
    return {"cf_tok_lp_mean": float(lp.mean()), "cf_tok_lp_min": float(lp.min()), "cf_tok_lp_p10": float(np.percentile(lp, 10)),
            "cf_tok_low_frac": float((lp < low).mean()), "cf_tok_lp_cv": float(p.std() / max(p.mean(), 1e-9)),
            "cf_tok_entropy": _num(row.get("token_entropy_mean")), "cf_has_token_probs": 1.0}


POST_KEYS = ("cf_expected_ll", "cf_free_ll", "cf_expected_gap", "cf_gop_mean", "cf_gop_min", "cf_gop_p10",
             "cf_low_conf_frac", "cf_vowel_gop", "cf_consonant_gop", "cf_cluster_gop", "cf_post_mean",
             "cf_frame_entropy", "cf_blank_frac", "cf_supported_frac")


def posterior_confidence(row: dict, expected: dict | None, free_ops: list[Op] | None, low: float = -2.0,
                         phone_level: bool = True) -> dict:
    out = {k: np.nan for k in POST_KEYS}
    out["cf_has_posteriors"] = 0.0
    if not np.isfinite(_num(row.get("frame_entropy_mean"))) and not np.isfinite(_num(row.get("greedy_ll"))):
        return out                                  # not a frame-posterior model
    out["cf_has_posteriors"] = 1.0
    out["cf_frame_entropy"] = _num(row.get("frame_entropy_mean"))
    out["cf_blank_frac"] = _num(row.get("blank_frac"))
    out["cf_free_ll"] = _num(row.get("greedy_ll"))
    ell = _num(row.get("expected_ll"))
    out["cf_expected_ll"] = ell
    spans = parse_spans(row.get("expected_alignment"))
    nf = _num(row.get("n_frames"))
    if np.isfinite(ell) and spans and np.isfinite(out["cf_free_ll"]) and np.isfinite(nf):
        out["cf_expected_gap"] = float(ell - out["cf_free_ll"] * nf / max(len(spans), 1))
    if not spans:
        return out
    g = np.array([_num(s.get("gop")) for s in spans])
    lp = np.array([_num(s.get("lp")) for s in spans])
    post = np.array([_num(s.get("post")) for s in spans])
    out.update({"cf_gop_mean": float(np.nanmean(g)), "cf_gop_min": float(np.nanmin(g)),
                "cf_gop_p10": float(np.nanpercentile(g, 10)), "cf_low_conf_frac": float(np.nanmean(lp < low)),
                "cf_post_mean": float(np.nanmean(post))})
    if phone_level:
        labs = [str(s.get("p")) for s in spans]
        v = [g[k] for k, lab in enumerate(labs) if is_vowel(lab)]
        c = [g[k] for k, lab in enumerate(labs) if not is_vowel(lab) and lab != "|"]
        out["cf_vowel_gop"] = float(np.nanmean(v)) if v else np.nan
        out["cf_consonant_gop"] = float(np.nanmean(c)) if c else np.nan
        if expected is not None:
            cl = expected.get("in_cluster", [])
            # spans follow the mappable expected phones in order; aligned positions = expected positions when
            # coverage is complete, otherwise cluster GOP is left undefined rather than mis-assigned
            if len(spans) == len(expected.get("phones", [])) and cl:
                cg = [g[k] for k in range(len(spans)) if cl[k]]
                out["cf_cluster_gop"] = float(np.nanmean(cg)) if cg else np.nan
            if free_ops is not None and len(spans) == len(expected.get("phones", [])):
                matched = {o.i for o in free_ops if o.op == "M"}
                sup = [(post[k] >= 0.5) and (k in matched) for k in range(len(spans))]
                out["cf_supported_frac"] = float(np.mean(sup)) if sup else np.nan
    return out
