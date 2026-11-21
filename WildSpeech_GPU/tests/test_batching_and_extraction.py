import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from wildspeech.acoustic.batching import AdaptiveBatchController, plan_duration_batches  # noqa: E402
from wildspeech.acoustic.extractor import AcousticExtractionRunner  # noqa: E402
from wildspeech.acoustic.store import FeatureStore  # noqa: E402


def test_plan_respects_budget():
    d = pd.Series(np.random.default_rng(0).uniform(0.5, 20, 200), index=[f"f{i}" for i in range(200)])
    batches = plan_duration_batches(d, max_batch_size=16, max_batch_seconds=60.0)
    assert sorted(sum(batches, [])) == sorted(d.index)
    for b in batches:
        assert len(b) <= 16
        assert len(b) == 1 or d[b].max() * len(b) <= 60.0 + 1e-9


def test_controller_split_and_oom():
    c = AdaptiveBatchController(batch_size=8, max_batch_seconds=10.0)
    waves = [np.zeros(16000 * 2)] * 12
    sl = c.split(waves, 16000)
    assert sum(s.stop - s.start for s in sl) == 12
    assert all((s.stop - s.start) * 2 <= 10 for s in sl)
    c.on_oom()
    assert c.batch_size == 4 and c.max_batch_seconds == 5.0


def test_oom_recovery_splits_batch(ctx, monkeypatch):
    import torch

    r = AcousticExtractionRunner(ctx)
    calls = {"n": 0}
    real = r.gpu.extract_batch

    def flaky(waves, fams):
        calls["n"] += 1
        if len(waves) > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(waves, fams)

    monkeypatch.setattr(r.gpu, "extract_batch", flaky)
    waves = [np.random.default_rng(i).standard_normal(8000).astype(np.float32) * 0.1 for i in range(4)]
    out = r._gpu_extract(waves, ["quality"])
    assert len(out["quality"]) == 4
    assert r.controller.ooms >= 1


def test_extraction_resume_and_family_caches(ctx):
    ctx.cfg.set_path("audio.max_clips", 20)
    ctx.cfg.set_path("audio.checkpoint_every", 7)
    r = AcousticExtractionRunner(ctx)
    s1 = r.run("train", ["quality", "temporal"])
    assert s1["processed"] == 20
    s2 = r.run("train", ["quality", "temporal"])
    assert s2["todo"] == 0                      # fully resumed from parts
    store = FeatureStore(ctx.paths.cache / "features")
    assert len(store.parts("quality", "train")) >= 1
    assert store.done("quality", "train") == store.done("temporal", "train")
    # changing a temporal-only parameter invalidates only the temporal cache
    ctx.cfg.set_path("audio.long_pause_s", 0.9)
    r2 = AcousticExtractionRunner(ctx)
    s3 = r2.run("train", ["quality", "temporal"])
    assert s3["todo"] == 20
    assert store.validate("quality", "train", r2.family_fingerprint("quality"), False) == "valid"
    wide = r2.load_clip_features("train", ["quality", "temporal"])
    assert len(wide) == 20 and any(c.startswith("TEMPORAL::") for c in wide.columns)
