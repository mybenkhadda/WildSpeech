"""Resumable, OOM-safe local ASR inference (P24), tuned for an 8 GB RTX 3070.

ASRInferenceEngine.run(split) for ONE eligible model:
  1. checkpoint identity: fingerprint(model source + snapshot revision + decoding settings). A change invalidates
     only this model's cache. Resume skips clips already stored or recorded as failures.
  2. the model is loaded inside ModelSlot (one GPU model at a time). Precision comes from the per-model registry
     `precision` (else v3.asr.precision): bf16/fp16 on CUDA where supported, fp32 on CPU.
  3. batch policy: the parameter count and the GPU's total VRAM set the starting batch size and padded-seconds
     budget; registry batch_size / batch_seconds override. Batches are duration-sorted.
  4. AdaptiveMicroBatcher: inside each loader batch, CUDA OOM halves the micro-batch (down to 1 clip) and it grows
     back after `grow_after` clean micro-batches. A single clip that still OOMs goes to AudioPass, which does a
     single-clip CPU fallback or records a failure.
  5. precision guard: the first batch (and any later suspicious batch) is checked for non-finite model scores.
     If it is suspicious in fp16/bf16, the model switches to fp32 for the rest of the run, the batch is redone, and
     the switch is recorded (`precision_fallback`).
  6. v2.audio_pass.AudioPass: TAR byte-offset reads, threaded decoding with bounded prefetch, live tqdm (clips/s,
     ETA, RAM, VRAM, failures, last checkpoint), atomic Parquet checkpoints every `checkpoint_every` clips
  7. the throughput / memory log per batch goes to the P24 tables, and the model is unloaded (VRAM released) before
     the next model loads
Provenance of every cache carries producer="P24" plus the full eligibility record. asr.outputs only accepts such
caches as real outputs. Clips longer than max_audio_seconds are truncated by the decoder and flagged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import FILE
from .base import ModelSlot
from .checkpointing import ASRCheckpointStore, asr_fingerprint
from .eligibility import EligibilityRecord
from .model_registry import ASRModelSpec


TECHNICAL_DIMENSIONS = ("checkpoint_availability", "checkpoint_files", "architecture_compatibility",
                        "processor_compatibility", "framework_dependencies")


def technically_ready(rec: EligibilityRecord) -> tuple[bool, list[str]]:
    """Checkpoint + framework dimensions only (legal / provenance are NOT considered). Used for the load check,
    which touches synthetic audio only — never competition data."""
    bad = [f"{d}={rec.dimensions.get(d)}" for d in TECHNICAL_DIMENSIONS if rec.dimensions.get(d) != "PASS"]
    return (not bad), bad


def build_adapter(spec: ASRModelSpec, rec: EligibilityRecord, cfg, device: str, technical_only: bool = False):
    """Concrete adapter for an ELIGIBLE model (raises for anything else).
    technical_only=True (load diagnostics on synthetic audio): only the checkpoint / framework dimensions must pass;
    such an adapter is never handed competition audio (ASRInferenceEngine always calls with technical_only=False)."""
    if technical_only:
        ok, bad = technically_ready(rec)
        if not ok:
            raise PermissionError(f"{spec.key} is not technically ready: {', '.join(bad)}")
    elif not rec.eligible:
        raise PermissionError(f"{spec.key} is not eligible: {'; '.join(rec.reasons)}")
    kw = {"local_files_only": True}
    arch, src = rec.architecture, rec.source
    if spec.adapter == "pasketti_phonetic":
        from .pasketti_phonetic import PaskettiPhoneticAdapter

        return PaskettiPhoneticAdapter(spec, src, kw, device, cfg, arch)
    if spec.adapter == "pasketti_word":
        from .pasketti_word import build_pasketti_word

        return build_pasketti_word(spec, src, kw, device, cfg, arch)
    if spec.adapter == "whisper":
        if arch != "hf_whisper":
            from .pasketti_phonetic import UnsupportedArchitecture

            raise UnsupportedArchitecture(f"{spec.key}: adapter whisper but snapshot architecture is {arch}")
        from .whisper_adapter import WhisperAdapter

        return WhisperAdapter(spec, src, kw, device, cfg, output=spec.output)
    if spec.adapter == "ctc":
        if arch != "hf_ctc":
            from .pasketti_phonetic import UnsupportedArchitecture

            raise UnsupportedArchitecture(f"{spec.key}: adapter ctc but snapshot architecture is {arch}")
        from .ctc_adapter import CTCAdapter

        return CTCAdapter(spec, src, kw, device, cfg, output=spec.output)
    if spec.adapter == "encoder":
        if arch not in ("hf_encoder", "hf_ctc"):
            from .pasketti_phonetic import UnsupportedArchitecture

            raise UnsupportedArchitecture(f"{spec.key}: adapter encoder but snapshot architecture is {arch}")
        from .encoder_adapter import FrozenEncoderAdapter

        return FrozenEncoderAdapter(spec, src, kw, device, cfg)
    if spec.adapter == "qwen3_asr":
        from .qwen3_asr_adapter import Qwen3ASRAdapter

        return Qwen3ASRAdapter(spec, src, kw, device, cfg)
    raise ValueError(f"unknown adapter {spec.adapter}")


def batch_policy(n_params: int, cfg, vram_gb: float | None = None, dtype_bytes: int = 2, spec=None) -> dict:
    """Starting batch size / padded-seconds budget. The config sizes are an upper bound. With the GPU's VRAM known,
    the bound is reduced so that weights + a per-clip activation allowance fit with headroom. Registry overrides
    (spec.batch_size / spec.batch_seconds) win."""
    b = cfg.get_path("v3.asr.batch", {}) or {}
    large = n_params >= int(b.get("large_model_params", 300_000_000))
    size = int(b.get("large_model_batch_size", 2) if large else b.get("small_model_batch_size", 8))
    secs = float(b.get("batch_seconds_large", 60.0) if large else b.get("batch_seconds_small", 240.0))
    if vram_gb:
        usable = float(vram_gb) * float(b.get("vram_headroom", 0.80)) - n_params * dtype_bytes / 1e9 - float(b.get("cuda_context_gb", 0.8))
        per_clip = float(b.get("activation_gb_per_clip_large", 0.9) if large else b.get("activation_gb_per_clip_small", 0.35))
        fit = max(1, int(usable // per_clip)) if usable > 0 else 1
        size = max(1, min(size, fit))
    if spec is not None and getattr(spec, "runtime_backend", None) == "qwen_asr_vllm":
        q = cfg.get_path("v3.asr.qwen3_asr", {}) or {}
        size, secs = int(q.get("batch_size", 16)), float(q.get("batch_seconds", 240.0))   # vLLM batches internally
    if spec is not None and getattr(spec, "batch_size", None):
        size = int(spec.batch_size)
    if spec is not None and getattr(spec, "batch_seconds", None):
        secs = float(spec.batch_seconds)
    return {"large": large, "max_batch_size": size, "batch_seconds": secs}


def decoding_settings(cfg, spec: ASRModelSpec | None = None) -> dict:
    a = cfg.get_path("v3.asr", {}) or {}
    s = {k: a.get(k) for k in ("precision", "max_audio_seconds", "chunk_seconds", "chunk_overlap_seconds",
                               "whisper_max_new_tokens", "whisper_beam_size", "expected_text_scoring", "save_embeddings")}
    if spec is not None and getattr(spec, "precision", None):
        s["precision"] = spec.precision
    if spec is not None:
        s["unbatched_group_norm"] = a.get("unbatched_group_norm", True)
        s["phone_inventory"] = getattr(spec, "phone_inventory", None)
        if getattr(spec, "encoder_layers", None):
            s["encoder_layers"] = list(spec.encoder_layers)
        if getattr(spec, "runtime_backend", None) == "qwen_asr_vllm":
            q = cfg.get_path("v3.asr.qwen3_asr", {}) or {}
            s["qwen3_asr"] = {"language": q.get("language", "English"),
                              "transformers_backend": bool(q.get("allow_transformers_backend", False))}
    return s | {"g2p": cfg.get_path("v3.phonetics.g2p_backend", "auto"),
                "approx_g2p": cfg.get_path("v3.phonetics.allow_approximate_g2p", False)}


class AdaptiveMicroBatcher:
    """Runs fn on micro-batches. OOM halves the size (down to 1), and `grow_after` clean calls double it back up to
    `max_size`. An OOM on a single clip is re-raised so the caller's CPU fallback / failure path takes over."""

    def __init__(self, max_size: int, grow_after: int = 20, is_oom=None, release=None):
        from ..acoustic.batching import VRAMMonitor
        from ..acoustic.batching import is_oom as _is_oom

        self.max_size = max(1, int(max_size))
        self.size = self.max_size
        self.grow_after = max(1, int(grow_after))
        self.is_oom = is_oom or _is_oom
        self.release = release or VRAMMonitor.release
        self.clean = 0
        self.shrinks = 0
        self.min_seen = self.size

    def run(self, fn, waves: list, names: list) -> list:
        out, i = [], 0
        while i < len(waves):
            n = min(self.size, len(waves) - i)
            try:
                rows = fn(waves[i:i + n], names[i:i + n])
            except Exception as e:  # noqa: BLE001
                if self.is_oom(e) and n > 1:
                    self.release()
                    self.size = max(1, n // 2)
                    self.min_seen = min(self.min_seen, self.size)
                    self.shrinks += 1
                    self.clean = 0
                    continue
                raise
            out += list(rows)
            i += n
            self.clean += 1
            if self.clean >= self.grow_after and self.size < self.max_size:
                self.size = min(self.max_size, self.size * 2)
                self.clean = 0
        return out


def suspicious_rows(rows: list[dict], exposes_token_probs: bool, exposes_frames: bool) -> int:
    """Rows whose model scores look numerically broken (non-finite values dropped by ASRResult.row)."""
    bad = 0
    for r in rows:
        if r.get("error") == "embedding_nonfinite":
            bad += 1
            continue
        has_text = bool(str(r.get("transcript") or "").strip() or str(r.get("phones") or "").strip())
        if exposes_token_probs and has_text and not r.get("has_token_probs"):
            bad += 1
            continue
        g = r.get("greedy_ll")
        if exposes_frames and g is not None and not np.isfinite(float(g)) and float(r.get("n_frames", 1) or 1) > 0:
            bad += 1
    return bad


class ASRInferenceEngine:
    """One eligible model. `store_root` defaults to cache/v3 (P24 outputs); the smoke test uses cache/v3/asr_smoke
    so its rows can never be mistaken for P24 inference output."""

    def __init__(self, ctx, spec: ASRModelSpec, rec: EligibilityRecord, store_root=None, max_clips: int | None = None,
                 only_files: list[str] | None = None):
        self.ctx, self.spec, self.rec = ctx, spec, rec
        self.only_files = list(only_files) if only_files else None
        self.cfg, self.log = ctx.cfg, ctx.logger
        self.a = dict(ctx.cfg.get_path("v3.asr", {}) or {})
        if max_clips is not None:
            self.a["max_clips"] = int(max_clips)
        self.sr = int(getattr(spec, "sample_rate", None) or ctx.cfg.get_path("audio.sample_rate", 16000))
        self.smoke = store_root is not None
        self.store = ASRCheckpointStore(store_root or ctx.paths.v3_cache, spec.key)
        self.batch_log: list[dict] = []

    def run_locked(self, split: str, retry_failures: bool = False) -> dict:
        """run() inside an INFERENCE_RUNNING heartbeat lock; records the outcome in _last_run.json."""
        from .run_state import HeartbeatLock, record_last_run, run_lock_path

        if self.smoke:
            return self.run(split, retry_failures)
        with HeartbeatLock(run_lock_path(self.ctx, self.spec.key), {"model_key": self.spec.key, "split": split}):
            try:
                res = self.run(split, retry_failures)
            except BaseException as e:  # noqa: BLE001 - KeyboardInterrupt too: checkpoints already on disk
                record_last_run(self.ctx, self.spec.key, "failed" if not isinstance(e, KeyboardInterrupt) else "interrupted",
                                error=f"{type(e).__name__}: {e}"[:800], split=split, revision=self.rec.snapshot_revision)
                raise
        n_p, n_f = int(res.get("processed", 0) or 0), int(res.get("failures", 0) or 0)
        bad = res.get("status") == "done" and n_p > 0 and n_f >= n_p          # every attempted clip failed
        record_last_run(self.ctx, self.spec.key, "failed" if bad else "ok", split=split,
                        revision=self.rec.snapshot_revision, processed=res.get("processed"), failures=res.get("failures"))
        return res

    def _device(self) -> str:
        pref = self.a.get("device", "auto")
        return self.ctx.device if pref == "auto" else str(pref)

    def provenance(self, split: str, settings: dict) -> dict:
        return {"producer": "P24_SMOKE" if self.smoke else "P24", "lifecycle_at_start": self.rec.lifecycle,
                "snapshot_kind": self.rec.snapshot_kind, "output": self.spec.output, "role": self.spec.role,
                "encoder_layers": list(getattr(self.spec, "encoder_layers", []) or []),
                "model_key": self.spec.key, "hf_id": self.spec.hf_id, "local_path": self.spec.local_path,
                "resolved_source": self.rec.source, "snapshot_origin": self.rec.snapshot_origin,
                "snapshot_revision": self.rec.snapshot_revision, "pinned_revision": self.spec.revision,
                "architecture": self.rec.architecture, "license": self.spec.license,
                "training_sources": self.spec.evidence.training_sources, "model_card_url": self.spec.evidence.model_card_url,
                "eligibility": self.rec.as_row(), "settings": settings, "split": split}

    def run(self, split: str, retry_failures: bool = False) -> dict:
        from ..acoustic.batching import VRAMMonitor
        from ..v2.audio_pass import AudioPass

        settings = decoding_settings(self.cfg, self.spec)
        if self.smoke:
            settings = {**settings, "smoke_run": time.strftime("%Y%m%dT%H%M%S")}   # every smoke test re-decodes
        fp = asr_fingerprint(self.spec.key, self.rec.source, self.rec.snapshot_revision, settings)
        probe = AudioPass(self.ctx, split, f"P24 {self.spec.key} {split}", float(self.a.get("max_audio_seconds", 120.0)),
                          sample_rate=self.sr)
        cand = probe.clips(None if self.only_files else self.a.get("max_clips"))
        if self.only_files:
            cand = cand[cand[FILE].isin(set(self.only_files))].reset_index(drop=True)
        state = self.store.prepare(split, fp, self.provenance(split, settings), self.log)
        todo = self.store.todo(split, cand[FILE].tolist(), retry_failures)
        summary = {"model_key": self.spec.key, "split": split, "cache_state": state, "candidates": len(cand),
                   "todo": len(todo), "fingerprint": fp}
        if not todo:
            return {**summary, "status": "cached"}
        device = self._device()
        adapter = build_adapter(self.spec, self.rec, self.cfg, device)
        texts = cand.set_index(FILE)["expected_text"].fillna("").astype(str)
        durs = cand.set_index(FILE)["duration_sec"]
        max_s = float(self.a.get("max_audio_seconds", 120.0))
        guard = {"checked": False, "fallback": False, "suspicious": 0}
        with ModelSlot(self.spec.key):
            t_load = time.perf_counter()
            adapter.load()
            load_s = time.perf_counter() - t_load
            info = adapter.version_info()
            vm = VRAMMonitor(device)
            vm.reset()
            vram_gb = (vm.total / 1e9) if (vm.total and str(device).startswith("cuda")) else None
            pol = batch_policy(int(info.get("n_params") or 0), self.cfg, vram_gb,
                               4 if str(info.get("dtype", "")).startswith("fp32") else 2, self.spec)
            micro = AdaptiveMicroBatcher(pol["max_batch_size"], int((self.a.get("batch") or {}).get("grow_after", 20)))
            ap = AudioPass(self.ctx, split, f"{'SMOKE' if self.smoke else 'P24'} {self.spec.key} {split}", max_s,
                           max_batch_size=pol["max_batch_size"], batch_seconds=pol["batch_seconds"], sample_rate=self.sr)
            ap.checkpoint_every = int(self.a.get("checkpoint_every", 250))
            save_tp = bool(self.a.get("save_token_probabilities", True))
            ver = {"v": f"{self.spec.key}@{self.rec.snapshot_revision or 'local'}:{info.get('dtype')}"}

            def decode(waves, names):
                res = adapter.transcribe_batch(waves, [texts.get(n, "") for n in names], self.sr)
                rows = []
                for n, r in zip(names, res):
                    row = r.row(save_tp)
                    row["truncated"] = bool(float(durs.get(n, 0) or 0) > max_s)
                    row["_embedding"] = r.embedding
                    rows.append(row)
                return rows

            def process(waves, names):
                t0 = time.perf_counter()
                rows = micro.run(decode, waves, names)
                bad = suspicious_rows(rows, bool(getattr(adapter, "exposes_token_probs", False)),
                                      bool(getattr(adapter, "exposes_frame_posteriors", False)))
                low = str(getattr(adapter, "dtype_name", "fp32")) in ("fp16", "bf16")
                if bad and low and bool(self.a.get("precision_guard", True)):
                    self.log.warning("P24 %s: %d/%d rows with non-finite scores in %s -> switching to fp32",
                                     self.spec.key, bad, len(rows), adapter.dtype_name)
                    adapter.set_fp32()
                    guard.update({"fallback": True, "suspicious": guard["suspicious"] + bad})
                    ver["v"] = f"{self.spec.key}@{self.rec.snapshot_revision or 'local'}:fp32(guard)"
                    rows = micro.run(decode, waves, names)
                guard["checked"] = True
                for r in rows:
                    r["precision"] = adapter.dtype_name
                self.batch_log.append({"model_key": self.spec.key, "split": split, "n_clips": len(waves),
                                       "audio_seconds": float(sum(len(w) for w in waves) / self.sr),
                                       "seconds": time.perf_counter() - t0,
                                       "peak_vram_gb": vm.peak_fraction() * vm.total / 1e9 if vm.total else 0.0,
                                       "device": adapter.device, "micro_batch": micro.size, "dtype": adapter.dtype_name})
                return rows

            def cpu_fallback(waves, names):
                adapter.to_cpu()
                try:
                    rows = [dict(r, cpu_fallback=True) for r in decode(waves, names)]
                    return rows
                finally:
                    adapter.to_gpu()

            def flush(buf):
                return self.store.append(split, [dict(r) for r in buf], ver["v"])

            res = ap.run(todo, process, flush, lambda rows: self.store.add_failures(split, rows),
                         cpu_fallback=cpu_fallback if (self.a.get("cpu_fallback_on_oom", True) and str(device).startswith("cuda")) else None,
                         durations=durs)
            peak = vm.snapshot()
            adapter.unload()
        return {**summary, "status": "done", **res, "load_seconds": load_s, "model_version": ver["v"],
                "dtype": adapter.dtype_name, "n_params": info.get("n_params"), "batch_size": pol["max_batch_size"],
                "batch_seconds": pol["batch_seconds"], "micro_batch_min": micro.min_seen, "oom_shrinks": micro.shrinks,
                "precision_fallback": guard["fallback"], "suspicious_rows": guard["suspicious"],
                "peak_vram_gb": peak.get("peak_reserved_gb", 0.0), "token_probs": info.get("token_probs"),
                "frame_posteriors": info.get("frame_posteriors"), "architecture": self.rec.architecture,
                "device": device}

    def batch_table(self) -> pd.DataFrame:
        b = pd.DataFrame(self.batch_log)
        if not b.empty:
            b["clips_per_s"] = b["n_clips"] / b["seconds"].clip(lower=1e-6)
            b["rtf"] = b["seconds"] / b["audio_seconds"].clip(lower=1e-6)
        return b


def throughput_summary(batch_logs: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        b = batch_logs[(batch_logs["model_key"] == r["model_key"]) & (batch_logs["split"] == r["split"])] if not batch_logs.empty else batch_logs
        rows.append({"model_key": r["model_key"], "split": r["split"], "status": r.get("status"),
                     "clips": int(b["n_clips"].sum()) if not b.empty else 0,
                     "audio_hours": float(b["audio_seconds"].sum() / 3600) if not b.empty else 0.0,
                     "clips_per_s": float(b["n_clips"].sum() / max(b["seconds"].sum(), 1e-6)) if not b.empty else np.nan,
                     "rtf": float(b["seconds"].sum() / max(b["audio_seconds"].sum(), 1e-6)) if not b.empty else np.nan,
                     "peak_vram_gb": float(r.get("peak_vram_gb") or (b["peak_vram_gb"].max() if not b.empty else 0.0)),
                     "failures": r.get("failures"), "ooms": r.get("ooms"), "oom_shrinks": r.get("oom_shrinks"),
                     "dtype": r.get("dtype"), "precision_fallback": r.get("precision_fallback"),
                     "n_params": r.get("n_params"), "batch_size": r.get("batch_size"), "load_seconds": r.get("load_seconds")})
    return pd.DataFrame(rows)


# ========================================================================================== smoke test
def validate_smoke_rows(rows: pd.DataFrame, output: str, n_emb: int = 0, failures: int = 0) -> list[str]:
    """Problems that make a smoke decode FAIL (empty list = pass). A recogniser that returns only empty strings, or
    whose scores are all non-finite, is not ready — even though nothing raised."""
    p = []
    if rows is None or rows.empty:
        return ["no rows decoded"]
    if failures:
        p.append(f"{failures} failed clip(s)")
    if output == "embeddings":
        if n_emb <= 0:
            p.append("no pooled embeddings written")
        return p
    col = "phones" if output == "phones" else "transcript"
    txt = rows.get(col, pd.Series([""] * len(rows))).fillna("").astype(str).str.strip()
    if output == "phones" and "ipa" in rows:
        txt = txt.where(txt.ne(""), rows["ipa"].fillna("").astype(str).str.strip())
    if txt.ne("").sum() == 0:
        p.append(f"every {col} is empty (decoder produced no output)")
    if "has_token_probs" in rows and "mean_token_logprob" in rows:
        tp = rows[rows["has_token_probs"].fillna(False).astype(bool)]
        if len(tp) and not np.isfinite(pd.to_numeric(tp["mean_token_logprob"], errors="coerce")).any():
            p.append("token probabilities present but all non-finite")
    if "error" in rows:
        errs = rows["error"].fillna("").astype(str).str.strip()
        if errs.ne("").all():
            p.append("every row carries an error: " + errs.iloc[0][:120])
    return p


def synthetic_probe_audio(sr: int = 16000) -> list[np.ndarray]:
    """Deterministic NON-competition audio for load diagnostics: 1 s of faint noise, and 3 s of a voiced-like
    harmonic tone (f0 glide 140->220 Hz) with an amplitude envelope. Never derived from any recording."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(sr) * 1e-3).astype(np.float32)
    t = np.arange(3 * sr) / sr
    f0 = 140 + 80 * t / 3.0
    ph = 2 * np.pi * np.cumsum(f0) / sr
    env = np.clip(np.sin(np.pi * t / 3.0), 0, None) ** 0.5
    tone = (0.1 * env * sum(np.sin(k * ph) / k for k in range(1, 6))).astype(np.float32)
    return [noise, tone + (rng.standard_normal(len(t)) * 1e-3).astype(np.float32)]


def load_check(ctx, spec: ASRModelSpec, rec: EligibilityRecord, device: str | None = None) -> dict:
    """TECHNICAL readiness, independent of the legal / provenance reviews: load the local weights (local_files_only,
    never trust_remote_code), run a forward pass on synthetic audio, check the output contract, unload.
    Result -> cache/v3/asr_load_checks/<model>.json (keyed by revision). No competition audio is used."""
    import gc
    from pathlib import Path as _P

    from ..acoustic.batching import VRAMMonitor
    from ..io_utils import write_json
    from .base import ModelSlot

    device = device or ctx.device
    out = {"model_key": spec.key, "revision": rec.snapshot_revision, "architecture": rec.architecture,
           "source": rec.source, "device": device, "ok": False, "stage": "build", "error": "",
           "checked": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.perf_counter()
    ad = None
    try:
        ad = build_adapter(spec, rec, ctx.cfg, device, technical_only=True)
        vm = VRAMMonitor(device)
        vm.reset()
        with ModelSlot(spec.key):
            out["stage"] = "load"
            t1 = time.perf_counter()
            ad.load()
            out["load_seconds"] = round(time.perf_counter() - t1, 2)
            info = ad.version_info()
            out.update({"dtype": info.get("dtype"), "n_params": info.get("n_params"),
                        "token_probs": info.get("token_probs"), "frame_posteriors": info.get("frame_posteriors")})
            exp = float(spec.expected_params_millions or 0) * 1e6
            if exp and info.get("n_params"):
                out["params_ratio"] = round(float(info["n_params"]) / exp, 3)
            out["stage"] = "forward"
            waves = synthetic_probe_audio(int(spec.sample_rate or 16000))
            res = ad.transcribe_batch(waves, ["", ""], int(spec.sample_rate or 16000))
            rows = [r.row(False) for r in res]
            out["stage"] = "contract"
            probs = []
            if len(res) != len(waves):
                probs.append(f"{len(res)} results for {len(waves)} inputs")
            if spec.output == "embeddings":
                dims = [len(r.embedding) for r in res if r.embedding is not None]
                if len(dims) != len(res) or not all(np.all(np.isfinite(r.embedding)) for r in res if r.embedding is not None):
                    probs.append("missing or non-finite embedding")
                out["embedding_dim"] = dims[0] if dims else 0
            else:
                for r in rows:
                    if r.get("error"):
                        probs.append(f"decoder error: {r['error']}")
                    if getattr(ad, "exposes_frame_posteriors", False) and not np.isfinite(float(r.get("greedy_ll", np.nan))):
                        probs.append("non-finite CTC posteriors")
                out["outputs"] = [{"transcript": r.get("transcript", "")[:80], "phones": r.get("phones", "")[:80],
                                   "n_tokens": r.get("n_tokens")} for r in rows]
            if out.get("params_ratio") and not (0.5 <= out["params_ratio"] <= 2.0):
                probs.append(f"parameter count {info.get('n_params')} far from registry {spec.expected_params_millions} M")
            out["peak_vram_gb"] = vm.snapshot().get("peak_reserved_gb", 0.0)
            ad.unload()
            ad = None
        out.update({"ok": not probs, "error": "; ".join(probs), "stage": "done" if not probs else "contract"})
    except Exception as e:  # noqa: BLE001 - reported with the stage that failed
        out["error"] = f"{type(e).__name__}: {e}"[:800]
        out["dependency_hint"] = dependency_hint(e)
        ctx.logger.exception("load check %s failed at %s", spec.key, out["stage"])
    finally:
        if ad is not None:
            try:
                ad.unload()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        VRAMMonitor.release()
    out["seconds"] = round(time.perf_counter() - t0, 2)
    d = _P(ctx.paths.v3_cache) / "asr_load_checks"
    write_json(d / f"{spec.key}.json", out)
    return out


def dependency_hint(e: BaseException) -> str:
    """Actionable diagnosis for the common loading failures on this stack (torch 2.5.1 / transformers 4.57)."""
    s = f"{type(e).__name__}: {e}"
    low = s.lower()
    if "torch.load" in low or "cve-2025-32434" in low or "weights_only" in low:
        return ("transformers refuses torch.load with torch < 2.6: run `python -m wildspeech v3-install --model <key> "
                "--execute` (sha256-verified .bin -> safetensors conversion) instead of upgrading torch")
    if "feature extractor" in low or "preprocessor_config" in low:
        return "preprocessor_config.json missing at the pinned revision: v3-install fetches it at that commit"
    if "tokenizer" in low and ("can't load" in low or "not found" in low):
        return "tokenizer files missing/incomplete in the snapshot: re-install the pinned revision"
    if "out of memory" in low:
        return "CUDA OOM while loading: close other GPU processes or run the check with device='cpu'"
    if "qwen_asr" in low or "vllm" in low:
        return "pasketti_word needs qwen_asr + vLLM (Linux / WSL2): scripts/v3_pasketti_word_wsl.sh"
    if "trust_remote_code" in low or "auto_map" in low:
        return "checkpoint needs remote code; the pipeline never enables trust_remote_code for unreviewed code"
    if "no module named" in low:
        return "missing Python package: scripts\\v3_install_asr_env.ps1"
    if "permission" in low and "technically ready" in low:
        return "checkpoint or framework dimension not PASS: see model_eligibility.csv for the failing check"
    return ""


def load_check_record(ctx, model_key: str) -> dict:
    from pathlib import Path as _P

    from ..io_utils import read_json

    return read_json(_P(ctx.paths.v3_cache) / "asr_load_checks" / f"{model_key}.json", {}) or {}


def smoke_files(ctx, split: str = "train", n: int = 8) -> list[str]:
    """Task-stratified, deterministic small clip set (shortest clips per task first, then one long clip so the
    chunking path is exercised). Training split only by default."""
    b = ctx.bundle()
    c = (b.train_clips if split == "train" else b.test_clips)[[FILE, "task", "duration_sec"]].drop_duplicates(FILE)
    if c.empty:
        return []
    per = max(1, int(n) // max(c["task"].nunique(), 1))
    pick = (c.sort_values(["task", "duration_sec", FILE]).groupby("task").head(per)[FILE].tolist())
    longest = c.sort_values(["duration_sec", FILE], ascending=[False, True])[FILE].head(1).tolist()
    return list(dict.fromkeys(pick + longest))[: max(int(n), 1) + 1]


def run_smoke(ctx, spec: ASRModelSpec, rec: EligibilityRecord, n_clips: int = 8, split: str = "train") -> dict:
    """Decode a few clips into cache/v3/asr_smoke (never accepted as P24 output) and record _smoke.json.
    The result feeds the inference_readiness dimension; a failure makes the lifecycle FAILED."""
    from pathlib import Path as _P

    from .run_state import record_smoke

    files = smoke_files(ctx, split, n_clips)
    root = _P(ctx.paths.v3_cache) / "asr_smoke"
    out = {"model_key": spec.key, "revision": rec.snapshot_revision, "n_clips": len(files), "split": split, "ok": False}
    t0 = time.perf_counter()
    try:
        eng = ASRInferenceEngine(ctx, spec, rec, store_root=root, only_files=files)
        res = eng.run(split, retry_failures=True)
        rows = eng.store.load(split)
        n_text = int(rows.get("transcript", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()
                     + rows.get("phones", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum()) if not rows.empty else 0
        n_emb = len(eng.store.load_embeddings(split)) if spec.output == "embeddings" else 0
        fails = len(eng.store.failures(split))
        bt = eng.batch_table()
        problems = validate_smoke_rows(rows, spec.output, n_emb, fails)
        ok = not problems
        out.update({"ok": bool(ok), "rows": int(len(rows)), "failures": int(fails), "nonempty_outputs": n_text,
                    "problems": "; ".join(problems),
                    "embeddings": n_emb, "dtype": res.get("dtype"), "device": res.get("device"),
                    "load_seconds": res.get("load_seconds"), "peak_vram_gb": res.get("peak_vram_gb"),
                    "clips_per_s": float(bt["n_clips"].sum() / max(bt["seconds"].sum(), 1e-6)) if not bt.empty else 0.0,
                    "precision_fallback": res.get("precision_fallback"),
                    "error": "" if ok else "; ".join(problems)})
        if not rows.empty:
            keep = [c for c in (FILE, "transcript", "phones", "ipa", "mean_token_logprob", "expected_ll", "error") if c in rows]
            out["examples"] = rows[keep].head(5).to_dict("records")
    except Exception as e:  # noqa: BLE001 - recorded as a failed smoke test
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:800]})
        ctx.logger.exception("smoke test %s failed", spec.key)
    out["seconds"] = round(time.perf_counter() - t0, 2)
    record_smoke(ctx, spec.key, {k: v for k, v in out.items() if k != "examples"})
    return out
