"""P25 clip-level phonetic feature tables, one per ASR model (label-free, cached).

PhoneticFeatureBuilder(ctx).build(model_key, split)
    ASR rows (P24 cache) x clip metadata (task, expected_text) ->
      cache/v3/phonetic/<model>/<split>/features.parquet   FILE + '<MODEL>::<feature>' numeric columns
      cache/v3/phonetic/<model>/<split>/details.parquet    FILE, transcript, ref/hyp phones, op string, phones_source
    Hypothesis phones: the model's own phone output when it has one (phones_source='model'); for word/char
    models the G2P of the recognised transcript (phones_source='g2p_of_transcript', flagged, so derived phone
    errors are never confused with recognised phone errors).
    The table is rebuilt only when the ASR cache fingerprint or the phonetics settings change.
Nothing here reads item_score or outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE
from ..io_utils import atomic_write_parquet, read_json, stable_hash, write_json
from .alignment import parse_spans
from .confidence import posterior_confidence, token_confidence
from .error_analysis import clip_details, phoneme_features, word_features, words_of
from .g2p import ExpectedPhones
from .task_features import task_features, timing_features

VERSION = "v3-phon-1.0.0"


def model_prefix(model_key: str) -> str:
    return "M_" + str(model_key).upper()


class PhoneticFeatureBuilder:
    def __init__(self, ctx):
        self.ctx = ctx
        self.p = ctx.cfg.get_path("v3.phonetics", {}) or {}
        self.exp = ExpectedPhones(self.p.get("g2p_backend", "auto"), bool(self.p.get("allow_approximate_g2p", False)))
        self.low = float(ctx.cfg.get_path("v3.asr.low_confidence_logprob", -2.0))
        self.bins = int(self.p.get("position_bins", 3))
        self.frame_s = float(self.p.get("frame_seconds", 0.02))
        self.pause_frames = int(self.p.get("pause_frames", 5))
        self.root = ctx.paths.v3_cache / "phonetic"

    def _dir(self, model_key: str, split: str):
        d = self.root / model_key.lower() / split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fingerprint(self, asr_fp: str | None) -> str:
        return stable_hash({"asr": asr_fp, "p": self.p, "low": self.low, "g2p": self.exp.g2p.backend, "v": VERSION})

    # ---------------------------------------------------------------- one clip
    def clip(self, row: dict, task: str, expected_text: str, output: str) -> tuple[dict, dict]:
        exp = self.exp(expected_text)
        transcript = str(row.get("transcript") or "")
        if output == "phones":
            hyp_ph = str(row.get("phones") or "").split()
            src = "model"
        else:
            hyp_ph = self.exp.phonemes(transcript) if self.exp.usable else []
            src = "g2p_of_transcript"
        f: dict = {}
        wf = word_features(transcript, expected_text) if output in ("words", "chars") else {}
        f.update(wf)
        pf, ops = phoneme_features(exp, hyp_ph, self.bins) if (exp["phones"] and self.exp.usable) else ({}, [])
        f.update(pf)
        f.update(token_confidence(row, self.low))
        f.update(posterior_confidence(row, exp, ops if output == "phones" else None, self.low, phone_level=output == "phones"))
        spans = parse_spans(row.get("expected_alignment"))
        tf = timing_features(spans, exp if output == "phones" else None, self.frame_s, self.pause_frames)
        f.update(tf)
        f.update(task_features(task, wf, pf, ops, tf, words_of(transcript), words_of(expected_text)))
        f["phones_from_model"] = float(src == "model")
        f["audio_seconds"] = float(row.get("audio_seconds", np.nan) or np.nan)
        f["truncated"] = float(bool(row.get("truncated")))
        f["exp_n_phones"] = float(len(exp["phones"]))
        f["exp_n_syllables"] = float(exp["n_syllables"])
        f["exp_n_clusters"] = float(exp["n_clusters"])
        det = {"transcript": transcript, **clip_details(ops, exp, hyp_ph), "phones_source": src}
        return f, det

    # ---------------------------------------------------------------- one model
    def build(self, model_key: str, split: str, asr: pd.DataFrame, output: str, asr_fp: str | None,
              force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = self._dir(model_key, split)
        fp = self.fingerprint(asr_fp)
        meta = read_json(d / "_fingerprint.json", {}) or {}
        if not force and meta.get("fingerprint") == fp and (d / "features.parquet").exists():
            return pd.read_parquet(d / "features.parquet"), pd.read_parquet(d / "details.parquet")
        b = self.ctx.bundle()
        clips = (b.train_clips if split == "train" else b.test_clips)[[FILE, "task", "expected_text"]].drop_duplicates(FILE)
        a = asr[asr["error"].fillna("") == ""] if "error" in asr else asr
        m = a.merge(clips, on=FILE, how="inner")
        feats, dets = [], []
        for r in m.to_dict("records"):
            f, det = self.clip(r, r["task"], r["expected_text"], output)
            feats.append({FILE: r[FILE], **f})
            dets.append({FILE: r[FILE], "task": r["task"], "expected_text": r["expected_text"], **det})
        F = pd.DataFrame(feats)
        if not F.empty:
            pre = model_prefix(model_key)
            num = [c for c in F.columns if c != FILE]
            F[num] = F[num].apply(pd.to_numeric, errors="coerce").astype(np.float32)
            F = F.rename(columns={c: f"{pre}::{c}" for c in num})
        D = pd.DataFrame(dets)
        if not D.empty:
            D["model_key"] = model_key
        atomic_write_parquet(F, d / "features.parquet")
        atomic_write_parquet(D, d / "details.parquet")
        write_json(d / "_fingerprint.json", {"fingerprint": fp, "asr_fingerprint": asr_fp, "n": len(F),
                                             "g2p_backend": self.exp.g2p.backend, "g2p_exact": self.exp.exact})
        return F, D


def load_phonetic_tables(ctx, splits=("train",), models: list[str] | None = None,
                         only_real: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide clip features (all models joined on FILE) + long details (one row per clip x model).

    only_real (default): a (model, split) table is loaded only if its source ASR output is currently usable
    (asr.outputs.usable_pairs: accepted P24 provenance + real transcripts/phones + legal/provenance still PASS),
    so P26-P29 can never consume features derived from unverified or no-longer-eligible ASR caches."""
    root = ctx.paths.v3_cache / "phonetic"
    keys = sorted(p.name for p in root.glob("*") if p.is_dir()) if root.exists() else []
    if models:
        keys = [k for k in keys if k in {m.lower() for m in models}]
    usable = None
    if only_real:
        from ..asr.outputs import usable_pairs

        usable = {(m.lower(), s) for m, s in usable_pairs(ctx, tuple(splits))}
    wide, dets = None, []
    for k in keys:
        fs, ds = [], []
        for s in splits:
            if usable is not None and (k, s) not in usable:
                continue
            p = root / k / s
            if (p / "features.parquet").exists():
                fs.append(pd.read_parquet(p / "features.parquet"))
                ds.append(pd.read_parquet(p / "details.parquet"))
        if not fs:
            continue
        F = pd.concat(fs, ignore_index=True).drop_duplicates(FILE, keep="last")
        wide = F if wide is None else wide.merge(F, on=FILE, how="outer")
        dets += ds
    return (wide if wide is not None else pd.DataFrame(columns=[FILE]),
            pd.concat(dets, ignore_index=True) if dets else pd.DataFrame())


def feature_columns(wide: pd.DataFrame, model_key: str | None = None, names: tuple[str, ...] | None = None,
                    prefixes: tuple[str, ...] | None = None) -> list[str]:
    cols = [c for c in wide.columns if "::" in c]
    if model_key:
        cols = [c for c in cols if c.startswith(model_prefix(model_key) + "::")]
    if names:
        cols = [c for c in cols if c.split("::", 1)[1] in set(names)]
    if prefixes:
        cols = [c for c in cols if c.split("::", 1)[1].startswith(prefixes)]
    return cols
