"""Recognized-text deviation from the expected response, per (recognizer, clip), and cross-recognizer disagreement.

IMPORTANT — what these features are: the deviation of a RECOGNIZER'S OUTPUT from the expected text. It mixes the
child's response with recognizer error on child speech, so a substitution here is a *recognized-text error*, never
a verified pronunciation error. Feature names keep the `w_` (word) / `c_` (character) prefixes of
phonetic.error_features; measured phonetic errors live in phonetic_features (ASRPHN) only.

Per clip and recognizer (existing phonetic.error_features.clip_error_features + V4 additions):
  w_sub/ins/del (+ _rate), w_er (= WER), w_seq_preservation, w_partial_correct, w_exact, c_* (= CER level),
  n_words_hyp, n_words_ref, len_ratio, empty, repetitions, restarts,
  coverage            share of expected word types present in the recognized text
  near_match          c_er <= v4.text.near_match_cer;   text_correct  w_exact or c_er <= text_correct_cer
  unique_ratio        lexical diversity of the recognized words
  task-specific       blending: bl_exact, bl_segmented (several short fragments instead of one blended word),
                      bl_partial_cer, bl_extra_words; deletion: dl_exact, dl_excess_chars (residual of the sound
                      that should be deleted), dl_contains_expected, dl_len_diff; sentence repetition: sr_fidelity
                      (1 - WER), sr_omission_rate, sr_order (LCS preservation), sr_first/second_half_err
  uncertainty         asr_conf, asr_mean_token_logprob (only if the export carries them; never fabricated)
  g2pd_*              (optional, off by default) G2P of the recognized WORDS vs G2P of the expected text — a
                      DERIVED quantity, explicitly not a measured phonetic transcription
Cross-recognizer (per clip, >= 2 usable recognizers): dis_word_mean/max, dis_char_mean, agree_exact_frac, wer_sd.
Failed inferences (asr_ok False) or missing transcripts give NaN features (never 0) and lower the coverage.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from .. import FILE, ID
from ..phonetic.error_features import align_ops, clip_error_features

TASK_PREFIX = {"blending": "bl_", "deletion": "dl_", "sentence_repetition": "sr_"}


def words(text) -> list[str]:
    return [w for w in re.sub(r"[^a-z' ]+", " ", str(text or "").lower()).split() if w]


def _half_errors(rw: list[str], hw: list[str]) -> tuple[float, float]:
    if not rw:
        return np.nan, np.nan
    ops = align_ops(rw, hw)
    half = len(rw) / 2.0
    e1 = sum(o[0] != "M" for o in ops if o[3] < half)
    e2 = sum(o[0] != "M" for o in ops if o[3] >= half)
    n1, n2 = max(int(np.ceil(half)), 1), max(len(rw) - int(np.ceil(half)), 1)
    return e1 / n1, e2 / n2


def task_features(task: str, rw: list[str], hw: list[str], base: dict) -> dict:
    rc, hc = "".join(rw), "".join(hw)
    if task == "blending":
        frag = len(hw) >= 2 and np.mean([len(w) for w in hw]) <= 2.5
        return {"bl_exact": base["w_exact"], "bl_segmented": float(frag or (len(hw) >= 2 and hc == rc)),
                "bl_partial_cer": base["c_er"], "bl_extra_words": float(max(0, len(hw) - len(rw)))}
    if task == "deletion":
        return {"dl_exact": base["w_exact"], "dl_excess_chars": float(max(0, len(hc) - len(rc))),
                "dl_contains_expected": float(bool(rc) and rc in hc), "dl_len_diff": float(len(hc) - len(rc))}
    if task == "sentence_repetition":
        h1, h2 = _half_errors(rw, hw)
        return {"sr_fidelity": float(max(0.0, 1.0 - min(base["w_er"], 1.0))), "sr_omission_rate": base["w_del_rate"],
                "sr_order": base["w_seq_preservation"], "sr_first_half_err": h1, "sr_second_half_err": h2}
    return {}


def clip_text_features(hyp: str | None, ref: str, task: str, near_cer: float = 0.34, correct_cer: float = 0.2,
                       g2p=None) -> dict:
    if hyp is None:
        return {}
    f, _ = clip_error_features(hyp, ref, None)
    hw, rw = words(hyp), words(ref)
    f["coverage"] = float(len(set(rw) & set(hw)) / len(set(rw))) if rw else np.nan
    f["near_match"] = float(f["c_er"] <= near_cer)
    f["text_correct"] = float(f["w_exact"] == 1.0 or f["c_er"] <= correct_cer)
    f["unique_ratio"] = float(len(set(hw)) / len(hw)) if hw else np.nan
    f.update(task_features(task, rw, hw, f))
    if g2p is not None:
        g, _ = clip_error_features(hyp, ref, g2p)
        f.update({k.replace("ph_", "g2pd_", 1): v for k, v in g.items() if k.startswith("ph_")})
    return f


class _PairCache:
    """Many children give identical responses: features depend only on (hyp, ref, task) -> compute once."""

    def __init__(self, fn):
        self.fn, self.mem = fn, {}

    def __call__(self, hyp, ref, task):
        k = (hyp, ref, task)
        if k not in self.mem:
            self.mem[k] = self.fn(hyp, ref, task)
        return self.mem[k]


def build_clip_text_table(joined: pd.DataFrame, models: list[str], cfg_text: dict, g2p=None, progress: bool = True
                          ) -> pd.DataFrame:
    """Long table: FILE, model_key, split, ID, task, asr_ok, <features>."""
    near, corr = float(cfg_text.get("near_match_cer", 0.34)), float(cfg_text.get("text_correct_cer", 0.2))
    use_g2p = g2p if cfg_text.get("include_g2p_derived") else None
    calc = _PairCache(lambda h, r, t: clip_text_features(h, r, t, near, corr, use_g2p))
    d = joined[joined["model_key"].isin(models)].reset_index(drop=True)
    it = d.itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, total=len(d), desc="recognized-text features", dynamic_ncols=True)
        except ImportError:
            pass
    rows = []
    for r in it:
        hyp = r.transcript if (bool(r.asr_ok) and r.transcript is not None and not (isinstance(r.transcript, float))) else None
        f = calc(hyp, str(r.expected_text or ""), str(r.task)) if hyp is not None else {}
        rows.append({FILE: getattr(r, FILE), "model_key": r.model_key, "split": r.split, ID: getattr(r, ID),
                     "task": r.task, "asr_ok": bool(r.asr_ok) and hyp is not None,
                     "asr_conf": float(r.confidence) if r.confidence is not None and np.isfinite(r.confidence) else np.nan,
                     "asr_mean_token_logprob": float(r.mean_token_logprob) if np.isfinite(r.mean_token_logprob) else np.nan,
                     **f})
    return pd.DataFrame(rows)


def consensus_table(long: pd.DataFrame, feature_cols: list[str], strategy: str = "consensus",
                    coverage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per clip: mean of each feature over usable recognizers (consensus) or the best-coverage recognizer."""
    ok = long[long["asr_ok"]]
    cols = [c for c in feature_cols if c in ok.columns]
    if ok.empty or not cols:
        return pd.DataFrame(columns=[FILE] + cols)
    if strategy == "best_coverage" and coverage is not None and not coverage.empty:
        best = coverage[coverage["split"] == "train"].sort_values("usable_frac", ascending=False)["model_key"].iloc[0]
        t = ok[ok["model_key"] == best].drop_duplicates(FILE)[[FILE] + cols].copy()
        t["n_models"] = 1.0
        return t
    g = ok.groupby(FILE)
    t = g[cols].mean()
    t["n_models"] = g["model_key"].nunique().astype(float)
    return t.reset_index()


def _ned(a: list, b: list) -> float:
    if not a and not b:
        return 0.0
    ops = align_ops(a, b)
    return sum(o[0] != "M" for o in ops) / max(len(a), len(b), 1)


def disagreement_table(joined: pd.DataFrame, long: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per clip with >= 2 usable recognizers: pairwise normalized word / character edit distances, exact-agreement
    share and the SD of WER across recognizers."""
    d = joined[joined["model_key"].isin(models) & joined["asr_ok"] & joined["transcript"].notna()]
    if d["model_key"].nunique() < 2:
        return pd.DataFrame(columns=[FILE])
    wer = long[long["asr_ok"]].groupby(FILE)["w_er"].std(ddof=0) if "w_er" in long else pd.Series(dtype=float)
    cache: dict = {}
    rows = []
    for f, g in d.groupby(FILE):
        ts = [" ".join(words(t)) for t in g.drop_duplicates("model_key")["transcript"]]
        if len(ts) < 2:
            continue
        wd, cd, ag = [], [], []
        for a, b in itertools.combinations(ts, 2):
            k = (a, b) if a <= b else (b, a)
            if k not in cache:
                cache[k] = (_ned(a.split(), b.split()), _ned(list(a.replace(" ", "")), list(b.replace(" ", ""))), float(a == b))
            w_, c_, e_ = cache[k]
            wd.append(w_)
            cd.append(c_)
            ag.append(e_)
        rows.append({FILE: f, "dis_word_mean": float(np.mean(wd)), "dis_word_max": float(np.max(wd)),
                     "dis_char_mean": float(np.mean(cd)), "agree_exact_frac": float(np.mean(ag)),
                     "wer_sd": float(wer.get(f, np.nan)), "n_models": float(len(ts))})
    return pd.DataFrame(rows)


def reliability_table(joined: pd.DataFrame, models: list[str], dis: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Clip-level ASR reliability (gate inputs of the reliability-gated mixture of experts, M7):
    rel_any_ok (>= 1 usable recognizer), rel_frac_ok (usable share), rel_agree (1 - mean word disagreement),
    rel_conf (mean recognizer confidence z when exported)."""
    d = joined[joined["model_key"].isin(models)]
    g = d.groupby(FILE)
    t = pd.DataFrame({"rel_frac_ok": g["asr_ok"].mean().astype(float)})
    t["rel_any_ok"] = (t["rel_frac_ok"] > 0).astype(float)
    if dis is not None and not dis.empty:
        t["rel_agree"] = 1.0 - dis.set_index(FILE)["dis_word_mean"].reindex(t.index)
    if "asr_conf" in long and long["asr_conf"].notna().any():
        z = long.groupby("model_key")["asr_conf"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
        t["rel_conf"] = long.assign(z=z).groupby(FILE)["z"].mean().reindex(t.index)
    return t.reset_index()


def transcript_examples(joined: pd.DataFrame, per_task: int = 4, seed: int = 0) -> pd.DataFrame:
    """Training-split examples with the word alignment (for the edit-alignment figure)."""
    from ..phonetic.error_features import ops_to_string

    d = joined[(joined["split"] == "train") & joined["asr_ok"] & joined["transcript"].notna()]
    rows = []
    for t, g in d.groupby("task"):
        files = g[FILE].drop_duplicates().sample(min(per_task, g[FILE].nunique()), random_state=seed)
        for f in files:
            for r in g[g[FILE] == f].itertuples(index=False):
                rw, hw = words(r.expected_text), words(r.transcript)
                rows.append({"task": t, FILE: f, "model_key": r.model_key, "expected_text": r.expected_text,
                             "transcript": r.transcript, "item_score": r.item_score,
                             "word_ops": ops_to_string(align_ops(rw, hw))})
    return pd.DataFrame(rows)
