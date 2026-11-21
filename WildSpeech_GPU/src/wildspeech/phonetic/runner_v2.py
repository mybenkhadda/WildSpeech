"""P18 runner: content-conditioned phonetic features from eligible LOCAL models, cached per model.

Adapters (configured in v2.phonetic, gated by configs/foundation_models.yaml):
  whisper_asr  -> family ph_whisper_<key>: free transcription + teacher-forced expected-text likelihood + word /
                  character / phoneme (G2P) error analysis, repetitions, restarts
  ctc_char     -> family ph_ctc_<key>:     CTC expected likelihood, forced-alignment GOP / entropy / durations,
                  greedy-text error analysis
  ctc_phoneme  -> family ph_phn_<key>:     phoneme-level forced alignment against G2P(expected text): GOP by phone
                  class and position, phone error analysis, posterior summaries post_<ARPAbet>
  child models -> same adapters (registry kind decides) with child_speech: true
Detail rows (hypothesis text, phone ops, aligned spans) -> family phd_<key> (strings) for confusion matrices,
error-position heatmaps and alignment plots. A feature is produced only when the model and metadata support it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import FILE
from ..acoustic.store import FeatureStore
from ..io_utils import stable_hash
from ..model_registry import ModelNotEligible, load_registry, resolve_model_source
from .error_features import align_ops, clip_error_features, cv_profile, error_summary, ops_to_string, position_profile
from .g2p import G2P

VERSION = "phon-2.0.0"
KEY_FEATURES = {
    "whisper": ["w_er", "c_er", "ph_per", "ph_seq_preservation", "ph_partial_correct", "expected_ll_mean", "ll_ratio",
                "avg_logprob", "repetitions", "restarts", "len_ratio", "empty", "ph_consonant_err", "ph_vowel_err",
                "ph_err_initial", "ph_err_medial", "ph_err_final"],
    "ctc": ["expected_ll", "ll_ratio", "gop_mean", "gop_min", "low_conf_frac", "align_entropy", "tok_dur_mean",
            "tok_dur_cv", "c_er", "w_er", "gop_initial", "gop_final"],
    "phn": ["expected_ll", "ll_ratio", "gop_mean", "gop_min", "gop_p10", "low_conf_frac", "align_entropy", "tok_dur_mean",
            "tok_dur_cv", "ph_per", "ph_seq_preservation", "ph_consonant_err", "ph_vowel_err", "vowel_gop_mean",
            "consonant_gop_mean", "ph_err_initial", "ph_err_medial", "ph_err_final", "target_coverage"],
}


def v2_store(ctx) -> FeatureStore:
    return FeatureStore(ctx.paths.v2_cache / "features")


def model_jobs(ctx) -> tuple[list[dict], pd.DataFrame]:
    """Resolve configured adapters -> eligible jobs + an eligibility table (license / overlap verification)."""
    ph = ctx.cfg.get_path("v2.phonetic", {}) or {}
    reg = load_registry(ctx.paths.configs)
    rows, jobs = [], []
    wanted = ([(k, "whisper") for k in ph.get("asr_models", [])] + [(k, "ctc") for k in ph.get("ctc_char_models", [])]
              + [(k, "phn") for k in ph.get("ctc_phoneme_models", [])] + [(k, None) for k in ph.get("child_models", [])])
    for key, kind in wanted:
        entry = reg.get(key, {})
        rk = entry.get("kind")
        k = kind or {"whisper_asr": "whisper", "ctc_char": "ctc", "ctc_phoneme": "phn"}.get(rk)
        row = {"model": key, "adapter": k, "registry_kind": rk, "hf_id": entry.get("hf_id"), "license": entry.get("license"),
               "child_speech": entry.get("child_speech"), "license_verified": bool(entry.get("license_verified")),
               "overlap_checked": bool(entry.get("overlap_checked"))}
        if k is None:
            rows.append({**row, "eligible": False, "reason": f"unsupported kind {rk}"})
            continue
        try:
            src, kw = resolve_model_source(key, reg, ctx.cfg)
            rows.append({**row, "eligible": True, "reason": "verified", "source": src})
            jobs.append({"key": key, "adapter": k, "source": src, "kwargs": kw, "family": f"ph_{k}_{key}", "detail": f"phd_{key}"})
        except ModelNotEligible as e:
            rows.append({**row, "eligible": False, "reason": str(e)})
    return jobs, pd.DataFrame(rows)


class PhoneticV2Runner:
    def __init__(self, ctx):
        self.ctx, self.cfg, self.log = ctx, ctx.cfg, ctx.logger
        self.ph = ctx.cfg.get_path("v2.phonetic", {}) or {}
        self.store = v2_store(ctx)
        self.g2p = G2P(self.ph.get("g2p_backend", "auto"))
        self.allow_approx = bool(self.ph.get("allow_approximate_g2p", False))
        self.bins = int(self.ph.get("position_bins", 3))
        self.low = float(self.ph.get("low_confidence_logprob", -2.0))
        self.sr = int(ctx.cfg.get_path("audio.sample_rate", 16000))

    def fingerprint(self, job: dict) -> str:
        return stable_hash({"job": {k: job[k] for k in ("key", "adapter", "source")}, "g2p": self.g2p.backend,
                            "approx": self.allow_approx, "bins": self.bins, "low": self.low,
                            "max_s": self.ph.get("max_seconds"), "v": VERSION})

    def _load(self, job: dict):
        dev, fp16 = self.ctx.device, bool(self.ph.get("fp16", True))
        if job["adapter"] == "whisper":
            from .asr import WhisperExpectedScorer, WhisperTranscriber

            t = WhisperTranscriber(job["source"], job["kwargs"], dev, fp16)
            return t, WhisperExpectedScorer(t)
        from .alignment import CTCModel

        return CTCModel(job["source"], job["kwargs"], "ctc_phoneme" if job["adapter"] == "phn" else "ctc_char", dev, fp16), None

    def _process_fn(self, job: dict, model, scorer, texts: pd.Series):
        g2p, bins, low, allow = self.g2p, self.bins, self.low, self.allow_approx
        exact = g2p.exact or allow

        def whisper(waves, names):
            refs = [texts.get(n, "") for n in names]
            hyp = model.transcribe(waves, self.sr)
            sc = scorer.score(waves, refs, self.sr)
            out = []
            for h, s, ref in zip(hyp, sc, refs):
                f, d = clip_error_features(h["text"], ref, g2p, bins, allow)
                f.update({"avg_logprob": h["avg_logprob"], "min_logprob": h["min_logprob"], "n_tokens": h["n_tokens"], **s})
                f["ll_ratio"] = h["avg_logprob"] - s["expected_ll_mean"] if np.isfinite(s["expected_ll_mean"]) else np.nan
                out.append({"num": f, "detail": {"hyp_text": h["text"], **d, "spans": ""}})
            return out

        def ctc(waves, names):
            lps = model.log_posteriors(waves, self.sr)
            out = []
            for lp, n in zip(lps, names):
                ref = texts.get(n, "")
                f, spans = model.align_features(lp, model.char_targets(ref), None, low, bins)
                hyp = model.decode_text(model.greedy(lp))
                e, d = clip_error_features(hyp, ref, g2p, bins, allow)
                out.append({"num": {**f, **e}, "detail": {"hyp_text": hyp, **d, "spans": spans}})
            return out

        def phn(waves, names):
            lps = model.log_posteriors(waves, self.sr)
            out = []
            for lp, n in zip(lps, names):
                ref = texts.get(n, "")
                rp = g2p.phonemes(ref) if exact else []
                tgt, cov = model.phone_targets(rp)
                labels = [p for p in rp if p in model.arpa_tok]
                f, spans = model.align_features(lp, tgt, labels, low, bins) if exact else ({}, "")
                hp = model.decode_phones(model.greedy(lp))
                f.update(model.posterior_summary(lp))
                f["target_coverage"] = float(cov) if exact else np.nan
                if exact and rp:
                    ops = align_ops(rp, hp)
                    f.update(error_summary(rp, hp, "ph"))
                    f.update(position_profile(ops, len(rp), bins, "ph"))
                    f.update(cv_profile(ops, "ph"))
                    f["ph_per"] = f["ph_er"]
                    d = {"ref_phones": " ".join(rp), "hyp_phones": " ".join(hp), "ops": ops_to_string(ops)}
                else:
                    d = {"ref_phones": "", "hyp_phones": " ".join(hp), "ops": ""}
                out.append({"num": f, "detail": {"hyp_text": "", **d, "spans": spans}})
            return out

        return {"whisper": whisper, "ctc": ctc, "phn": phn}[job["adapter"]]

    def run(self, split: str) -> dict:
        from ..acoustic.batching import VRAMMonitor
        from ..v2.audio_pass import AudioPass

        jobs, elig = model_jobs(self.ctx)
        summary = {"split": split, "g2p_backend": self.g2p.backend, "g2p_exact": self.g2p.exact, "jobs": {}}
        if not jobs:
            return {**summary, "eligibility": elig}
        ap = AudioPass(self.ctx, split, "P18", float(self.ph.get("max_seconds", 30.0)),
                       max_batch_size=int(self.ph.get("batch_size", 8)),
                       batch_seconds=float(self.ph.get("batch_size", 8)) * float(self.ph.get("max_seconds", 30.0)))
        cand = ap.clips(self.ph.get("max_clips"))
        texts = cand.set_index(FILE)["expected_text"]
        for job in jobs:
            fp = self.fingerprint(job)
            for fam in (job["family"], job["detail"]):
                self.store.validate(fam, split, fp, True, self.log)
                self.store.set_fingerprint(fam, split, fp, {"job": job["key"]})
            todo = sorted(set(cand[FILE]) - self.store.done(job["family"], split) - self.store.failed(job["family"], split))
            if not todo:
                summary["jobs"][job["key"]] = {"status": "cached"}
                continue
            self.log.info("P18 %s (%s): loading %s", job["key"], job["adapter"], job["source"])
            model, scorer = self._load(job)
            fn = self._process_fn(job, model, scorer, texts)

            def flush(buf, job=job):
                if not buf:
                    return "-"
                self.store.append(job["detail"], split, pd.DataFrame([{FILE: r[FILE], **r["detail"]} for r in buf]))
                return self.store.append(job["family"], split, pd.DataFrame([{FILE: r[FILE], **r["num"]} for r in buf]))

            def fails(rows, job=job):
                self.store.add_failures(job["family"], split, rows)

            ap.desc = f"P18 {job['key']} {split}"
            res = ap.run(todo, fn, flush, fails, durations=cand.set_index(FILE)["duration_sec"])
            summary["jobs"][job["key"]] = {"status": "done", **res}
            del model, scorer
            VRAMMonitor.release()
        summary["eligibility"] = elig
        return summary


def load_phonetic_features(ctx, splits=("train",), key_only: bool = True) -> pd.DataFrame:
    """Wide clip table FILE + 'PH_<ADAPTER>_<KEY>::feature' for every cached phonetic family."""
    store = v2_store(ctx)
    root = ctx.paths.v2_cache / "features"
    fams = sorted(p.name for p in root.glob("ph_*") if p.is_dir()) if root.exists() else []
    out = None
    for fam in fams:
        parts = [store.load(fam, s, prefix=False) for s in splits]
        parts = [p for p in parts if not p.empty]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True).drop_duplicates(FILE, keep="last")
        adapter = fam.split("_")[1]
        keep = [c for c in df.columns if c != FILE and pd.api.types.is_numeric_dtype(df[c])
                and (not key_only or c in KEY_FEATURES.get(adapter, []) or c.startswith("post_"))]
        df = df[[FILE] + keep].rename(columns={c: f"{fam.upper()}::{c}" for c in keep})
        out = df if out is None else out.merge(df, on=FILE, how="outer")
    return out if out is not None else pd.DataFrame(columns=[FILE])


def load_phonetic_details(ctx, splits=("train",)) -> pd.DataFrame:
    store = v2_store(ctx)
    root = ctx.paths.v2_cache / "features"
    fams = sorted(p.name for p in root.glob("phd_*") if p.is_dir()) if root.exists() else []
    parts = []
    for fam in fams:
        for s in splits:
            df = store.load(fam, s, prefix=False)
            if not df.empty:
                parts.append(df.assign(model=fam[4:]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
