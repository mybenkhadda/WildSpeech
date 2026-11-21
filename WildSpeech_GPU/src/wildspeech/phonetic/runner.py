"""P03 runner: local ASR + CTC alignment per clip, cached as feature families ASR and ALIGN."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import FILE
from ..acoustic.batching import VRAMMonitor, is_oom, plan_duration_batches
from ..acoustic.store import FeatureStore
from ..data.audio_io import PrefetchingBatchLoader, make_reader
from ..data.tar_manifest import build_tar_manifest
from ..io_utils import stable_hash
from ..logging_utils import ResourceProgress
from ..model_registry import ModelNotEligible, load_registry, resolve_model_source
from .asr import CTCAligner, WhisperTranscriber
from .text_features import deviation_metrics


class PhoneticRunner:
    def __init__(self, ctx):
        self.ctx, self.cfg, self.log = ctx, ctx.cfg, ctx.logger
        self.store = FeatureStore(ctx.paths.cache / "features")
        self.reg = load_registry(ctx.paths.configs)
        self.sr = int(self.cfg.get_path("audio.sample_rate", 16000))

    def _fp(self, family: str, model_key: str) -> str:
        return stable_hash({"family": family, "model": model_key, "max_s": self.cfg.get_path("phonetic.max_seconds"),
                            "low": self.cfg.get_path("phonetic.low_confidence_logprob"), "v": "phon-1.0"})

    def run(self, split: str) -> dict:
        ph = self.cfg.section("phonetic")
        bundle = self.ctx.bundle()
        clips = (bundle.train_clips if split == "train" else bundle.test_clips).drop_duplicates(FILE)
        if self.cfg.get_path("audio.max_clips"):
            clips = clips.sort_values(FILE).head(int(self.cfg.get_path("audio.max_clips")))
        jobs = {}
        for family, key, cls in [("asr", ph.get("asr_model"), WhisperTranscriber), ("align", ph.get("ctc_model"), CTCAligner)]:
            if not key:
                continue
            try:
                src, kw = resolve_model_source(key, self.reg, self.cfg)
            except ModelNotEligible as e:
                self.log.warning("P03 %s skipped: %s", family, e)
                continue
            fp = self._fp(family, key)
            self.store.validate(family, split, fp, True, self.log)
            todo = set(clips[FILE]) - self.store.done(family, split) - self.store.failed(family, split)
            if todo:
                jobs[family] = (key, src, kw, cls, todo)
        summary = {"split": split, "families": list(jobs), "processed": {}}
        if not jobs:
            return summary
        man = build_tar_manifest(self.ctx.paths.files[f"{split}_audio"], self.ctx.paths.cache / "manifests", logger=self.log)
        reader = make_reader(self.ctx.paths.files[f"{split}_audio"], man)
        if reader is None:
            raise RuntimeError("P03 requires an uncompressed TAR for random access")
        text = clips.set_index(FILE)["expected_text"]
        cap = float(ph.get("max_seconds", 30.0))
        try:
            for family, (key, src, kw, cls, todo) in jobs.items():
                self.log.info("P03 %s: loading %s (%s)", family, key, src)
                model = cls(src, kw, self.ctx.device, fp16=bool(self.cfg.get_path("gpu.use_fp16_embeddings", True)))
                todo = sorted(todo & set(man[FILE]))
                bs = int(ph.get("asr_batch_size" if family == "asr" else "ctc_batch_size", 8))
                batches = plan_duration_batches(clips.set_index(FILE).loc[todo, "duration_sec"], bs, bs * cap, cap)
                loader = PrefetchingBatchLoader(reader, batches, self.sr, cap, int(self.cfg.get_path("runtime.decode_threads", 6)), 2)
                prog = ResourceProgress(len(todo), f"P03 {family} {split}")
                buf, fails, t0 = [], [], time.perf_counter()
                for b in loader:
                    fails += [{FILE: f, "stage": "decode", "error": e} for f, e in b.failures]
                    if b.waves:
                        rows = self._safe(model, family, b.waves, [text[f] for f in b.filenames])
                        for f, r in zip(b.filenames, rows):
                            if r is None:
                                fails.append({FILE: f, "stage": family, "error": "model_failure"})
                            else:
                                buf.append({FILE: f, **r})
                    ck = None
                    if len(buf) >= int(self.cfg.get_path("audio.checkpoint_every", 1000)):
                        ck = self.store.append(family, split, pd.DataFrame(buf))
                        buf = []
                    prog.update(len(b.filenames) + len(b.failures), failures=len(b.failures), checkpoint=ck)
                self.store.append(family, split, pd.DataFrame(buf))
                self.store.add_failures(family, split, fails)
                prog.close()
                summary["processed"][family] = {"clips": len(todo), "failures": len(fails), "seconds": time.perf_counter() - t0}
                del model
                VRAMMonitor.release()
        finally:
            reader.close()
        return summary

    def _safe(self, model, family: str, waves, texts) -> list[dict | None]:
        try:
            if family == "asr":
                res = model.transcribe(waves, self.sr)
                return [{**{"asr_text": r["text"], "avg_logprob": r["avg_logprob"], "min_logprob": r["min_logprob"],
                            "n_tokens": r["n_tokens"]}, **deviation_metrics(r["text"], t)} for r, t in zip(res, texts)]
            return model.score(waves, texts, self.sr, float(self.cfg.get_path("phonetic.low_confidence_logprob", -2.0)))
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                VRAMMonitor.release()
            if len(waves) == 1:
                self.log.warning("P03 %s failure: %s", family, e)
                return [None]
            mid = len(waves) // 2
            return self._safe(model, family, waves[:mid], texts[:mid]) + self._safe(model, family, waves[mid:], texts[mid:])
