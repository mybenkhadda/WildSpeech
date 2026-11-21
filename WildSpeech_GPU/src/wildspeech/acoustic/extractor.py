"""Resumable acoustic extraction: TAR offsets -> threaded decode -> GPU batches -> per-family Parquet parts."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import FILE
from ..data.audio_io import PrefetchingBatchLoader, StreamingTarBatchLoader, make_reader
from ..data.tar_manifest import build_tar_manifest
from ..io_utils import stable_hash
from ..logging_utils import ResourceProgress
from .batching import AdaptiveBatchController, VRAMMonitor, is_oom, plan_duration_batches
from .dsp import AcousticParams
from .store import FeatureStore
from .wavelet_features import wavelet_features

FAMILY_PARAM_KEYS = {
    "quality": ["sr", "n_fft", "hop", "win", "vad_noise_percentile", "vad_peak_percentile", "vad_offset_db",
                "vad_max_below_peak_db", "vad_min_dynamic_range_db", "min_speech_s", "min_silence_s",
                "bandlimit_drop_db", "hf_split_hz", "spectral_min_frames"],
    "temporal": ["sr", "n_fft", "hop", "win", "vad_noise_percentile", "vad_peak_percentile", "vad_offset_db",
                 "vad_max_below_peak_db", "min_speech_s", "min_silence_s", "long_pause_s", "nucleus_prominence_db",
                 "nucleus_min_gap_s"],
    "spectral": ["sr", "n_fft", "hop", "win", "vad_noise_percentile", "vad_offset_db", "vad_max_below_peak_db",
                 "min_speech_s", "min_silence_s", "spectral_min_frames"],
    "mfcc": ["sr", "n_fft", "hop", "win", "n_mels", "n_mfcc", "fmin", "fmax", "vad_offset_db", "min_speech_s",
             "min_silence_s", "spectral_min_frames"],
    "mel": ["sr", "n_fft", "hop", "win", "n_mels", "fmin", "fmax", "mel_summary_bands", "vad_offset_db",
            "min_speech_s", "min_silence_s", "spectral_min_frames"],
    "pitch": ["sr", "hop", "yin_frame_length", "yin_threshold", "yin_voicing_max", "f0_min_hz", "f0_max_hz",
              "vad_offset_db", "min_speech_s", "min_silence_s"],
}
CPU_FAMILIES = ("wavelet",)


def active_families(cfg) -> list[str]:
    fams = list(cfg.get_path("audio.families", ["quality", "temporal", "spectral", "mfcc", "mel"]))
    if cfg.get_path("audio.enable_pitch", False) and "pitch" not in fams:
        fams.append("pitch")
    if cfg.get_path("audio.enable_wavelet", False) and "wavelet" not in fams:
        fams.append("wavelet")
    return fams


class AcousticExtractionRunner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.p = AcousticParams.from_cfg(ctx.cfg)
        self.store = FeatureStore(ctx.paths.cache / "features")
        self.device = ctx.device
        self.log = ctx.logger
        self._gpu = None
        self._cpu_fallback = None
        g = ctx.cfg.section("gpu")
        bs = g.get("batch_size", 24)
        if bs == "auto":
            from ..io_utils import read_json

            rec = read_json(ctx.paths.cache / "runtime" / "batch_recommendation.json", {}) or {}
            bs = int(rec.get("recommended_batch_size", 24))
        self.controller = AdaptiveBatchController(batch_size=int(bs), max_batch_size=int(g.get("max_batch_size", 64)),
                                                  min_batch_size=int(g.get("min_batch_size", 1)),
                                                  max_batch_seconds=float(g.get("max_batch_seconds", 480.0)),
                                                  safety_fraction=float(g.get("vram_safety_fraction", 0.8)),
                                                  grow_after=int(g.get("grow_after_batches", 40)))
        self.vram = VRAMMonitor(self.device)

    # ------------------------------------------------------------ extractors
    @property
    def gpu(self):
        if self._gpu is None:
            from .gpu_features import GPUAcousticExtractor

            self._gpu = GPUAcousticExtractor(self.p, self.device)
        return self._gpu

    @property
    def cpu_fallback(self):
        if self._cpu_fallback is None:
            from .gpu_features import GPUAcousticExtractor

            self._cpu_fallback = GPUAcousticExtractor(self.p, "cpu")
        return self._cpu_fallback

    def family_fingerprint(self, family: str) -> str:
        pd_ = self.p.as_dict()
        if family == "wavelet":
            params = {k: self.cfg.get_path(f"audio.{k}") for k in ("wavelet_family", "wavelet_level", "wavelet_max_seconds", "sample_rate")}
        else:
            params = {k: pd_[k] for k in FAMILY_PARAM_KEYS[family]}
        params["max_clip_seconds"] = self.cfg.get_path("audio.max_clip_seconds")
        params["version"] = self.cfg.get_path("audio.extractor_version")
        return stable_hash({"family": family, **params})

    def _gpu_extract(self, waves: list[np.ndarray], fams: list[str]) -> dict[str, list[dict]]:
        try:
            return self.gpu.extract_batch(waves, fams)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            VRAMMonitor.release()
            self.controller.on_oom()
            if len(waves) == 1:
                self.log.warning("OOM on a single clip (%.1fs) -> CPU fallback", len(waves[0]) / self.p.sr)
                return self.cpu_fallback.extract_batch(waves, fams)
            mid = len(waves) // 2
            a, b = self._gpu_extract(waves[:mid], fams), self._gpu_extract(waves[mid:], fams)
            return {f: a[f] + b[f] for f in fams}

    # ------------------------------------------------------------------ main
    def manifest(self, split: str) -> pd.DataFrame | None:
        key = f"{split}_audio"
        if key not in self.ctx.paths.files:
            return None
        return build_tar_manifest(self.ctx.paths.files[key], self.ctx.paths.cache / "manifests", logger=self.log)

    def run(self, split: str, families: list[str] | None = None, filenames: list[str] | None = None,
            retry_failures: bool = False) -> dict:
        bundle = self.ctx.bundle()
        clips = bundle.train_clips if split == "train" else bundle.test_clips
        fams = list(families or active_families(self.cfg))
        gpu_fams = [f for f in fams if f in FAMILY_PARAM_KEYS]
        cpu_fams = [f for f in fams if f in CPU_FAMILIES]
        max_clips = self.cfg.get_path("audio.max_clips")
        cand = clips[[FILE, "duration_sec"]].drop_duplicates(FILE)
        if filenames is not None:
            cand = cand[cand[FILE].isin(set(filenames))]
        elif max_clips:
            cand = cand.sort_values(FILE).head(int(max_clips))
        auto = bool(self.cfg.get_path("runtime.auto_invalidate_stale_cache", True))
        if self.cfg.get_path("runtime.overwrite_cache", False):
            for f in fams:
                self.store.invalidate(f, split)
        todo: dict[str, set[str]] = {}
        for f in fams:
            fp = self.family_fingerprint(f)
            state = self.store.validate(f, split, fp, auto, self.log)
            self.store.set_fingerprint(f, split, fp, {"family": f})
            if retry_failures:
                self.store.clear_failures(f, split)
            skip = self.store.done(f, split) | self.store.failed(f, split)
            todo[f] = set(cand[FILE]) - skip
            self.log.info("%s/%s cache=%s todo=%d", f, split, state, len(todo[f]))
        need = set().union(*todo.values()) if todo else set()
        summary = {"split": split, "families": fams, "requested": len(cand), "todo": len(need), "processed": 0,
                   "failures": 0, "seconds": 0.0}
        if not need:
            return summary
        man = self.manifest(split)
        if man is None:
            raise FileNotFoundError(f"{split}_audio TAR not found")
        in_tar = set(man.loc[man["safe"], FILE])
        missing = need - in_tar
        if missing:
            for f in fams:
                self.store.add_failures(f, split, [{FILE: m, "stage": "manifest", "error": "missing_in_tar"}
                                                   for m in missing if m in todo[f]])
            need -= missing
        durs = cand.set_index(FILE).loc[sorted(need), "duration_sec"]
        cap = float(self.cfg.get_path("audio.max_clip_seconds", 30.0))
        g = self.cfg.section("gpu")
        batches = plan_duration_batches(durs, int(g.get("max_batch_size", 64)), float(g.get("max_batch_seconds", 480.0)) * 1.5, cap)
        rt = self.cfg.section("runtime")
        reader = make_reader(self.ctx.paths.files[f"{split}_audio"], man)
        if reader is not None:
            loader = PrefetchingBatchLoader(reader, batches, self.p.sr, cap, int(rt.get("decode_threads", 6)),
                                            int(rt.get("prefetch_batches", 2)), rt.get("ram_soft_limit_gb"))
        else:
            loader = StreamingTarBatchLoader(self.ctx.paths.files[f"{split}_audio"], need,
                                             self.controller.batch_size, self.p.sr, cap)
        every = int(self.cfg.get_path("audio.checkpoint_every", 1000))
        buffers: dict[str, list[dict]] = {f: [] for f in fams}
        fail_buf: list[dict] = []
        prog = ResourceProgress(total=len(need), desc=f"P02 {split} acoustic")
        t0 = time.perf_counter()
        wl = (self.cfg.get_path("audio.wavelet_family", "db4"), int(self.cfg.get_path("audio.wavelet_level", 6)),
              float(self.cfg.get_path("audio.wavelet_max_seconds", 10.0)))
        try:
            for batch in loader:
                for fname, err in batch.failures:
                    fail_buf.append({FILE: fname, "stage": "decode", "error": err})
                if batch.waves:
                    meta_rows = [{FILE: f, "true_duration_s": m["true_duration_s"], "decoded_duration_s": m["decoded_duration_s"],
                                  "cropped": float(m["cropped"]), "file_sr": float(m["file_sr"])}
                                 for f, m in zip(batch.filenames, batch.metas)]
                    want_gpu = [f for f in gpu_fams if any(fn in todo[f] for fn in batch.filenames)]
                    results: dict[str, list[dict]] = {f: [] for f in want_gpu}
                    if want_gpu:
                        for sl in self.controller.split(batch.waves, self.p.sr):
                            self.vram.reset()
                            try:
                                res = self._gpu_extract(batch.waves[sl], want_gpu)
                            except Exception as e:  # noqa: BLE001 - non-OOM failure of the whole sub-batch
                                self.log.warning("batch failure %s: %s", type(e).__name__, e)
                                res = {f: [None] * (sl.stop - sl.start) for f in want_gpu}
                                for fn in batch.filenames[sl]:
                                    fail_buf.append({FILE: fn, "stage": "gpu", "error": f"{type(e).__name__}: {e}"})
                            for f in want_gpu:
                                results[f].extend(res[f])
                            self.controller.on_success(self.vram.peak_fraction())
                    for f in want_gpu:
                        for meta, row in zip(meta_rows, results[f]):
                            if row is not None and meta[FILE] in todo[f]:
                                buffers[f].append({**meta, **row})
                    for f in cpu_fams:
                        for meta, y in zip(meta_rows, batch.waves):
                            if meta[FILE] in todo[f]:
                                try:
                                    buffers[f].append({**meta, **wavelet_features(y, self.p.sr, *wl)})
                                except Exception as e:  # noqa: BLE001
                                    fail_buf.append({FILE: meta[FILE], "stage": f, "error": f"{type(e).__name__}: {e}"})
                ckpt = None
                if max(len(b) for b in buffers.values()) >= every:
                    ckpt = self._flush(buffers, fail_buf, split, fams)
                    fail_buf = []
                prog.update(len(batch.filenames) + len(batch.failures), failures=len(batch.failures), checkpoint=ckpt,
                            bs=self.controller.batch_size)
            ckpt = self._flush(buffers, fail_buf, split, fams)
            prog.update(0, checkpoint=ckpt)
        finally:
            prog.close()
            if reader is not None:
                reader.close()
        summary.update({"processed": len(need), "failures": prog.failures, "seconds": time.perf_counter() - t0,
                        "clips_per_s": len(need) / max(time.perf_counter() - t0, 1e-6), **self.controller.state(),
                        **self.vram.snapshot()})
        return summary

    def _flush(self, buffers: dict[str, list[dict]], fails: list[dict], split: str, fams: list[str]) -> str:
        name = "-"
        for f in fams:
            if buffers[f]:
                name = self.store.append(f, split, pd.DataFrame(buffers[f]))
                buffers[f].clear()
        if fails:
            for f in fams:
                self.store.add_failures(f, split, fails)
        return name

    def load_clip_features(self, split: str, families: list[str] | None = None) -> pd.DataFrame:
        """Wide clip-level table: filename + FAMILY::feature columns for all cached families."""
        fams = families or active_families(self.cfg)
        out = None
        for f in fams:
            df = self.store.load(f, split)
            if df.empty:
                continue
            meta_cols = [c for c in df.columns if c.split("::")[-1] in ("true_duration_s", "decoded_duration_s", "cropped", "file_sr")]
            if out is None:
                keep = df.rename(columns={c: "META::" + c.split("::")[-1] for c in meta_cols})
                out = keep
            else:
                out = out.merge(df.drop(columns=meta_cols), on=FILE, how="outer")
        return out if out is not None else pd.DataFrame(columns=[FILE])
