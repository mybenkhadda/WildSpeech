"""GPU memory recovery (OOM splitting / CPU fallback), checkpoint resume and submission schema."""
import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE, ID, TARGET


class _OOM(RuntimeError):
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_oom_splits_batches_until_they_fit(ctx):
    from wildspeech.v2.audio_pass import AudioPass

    ap = AudioPass(ctx, "train", "test", 30.0)
    calls = []

    def process(waves, names):
        calls.append(len(waves))
        if len(waves) > 2:
            raise _OOM()
        return [{"n": len(w)} for w in waves]

    waves = [np.zeros(10 + i) for i in range(8)]
    out = ap.safe_process(process, waves, [f"f{i}" for i in range(8)])
    assert len(out) == 8 and all(o is not None for o in out)
    assert ap.ooms >= 1 and max(calls[1:]) <= 4


def test_single_clip_oom_uses_cpu_fallback(ctx):
    from wildspeech.v2.audio_pass import AudioPass

    ap = AudioPass(ctx, "train", "test", 30.0)
    out = ap.safe_process(lambda w, n: (_ for _ in ()).throw(_OOM()), [np.zeros(5)], ["a"],
                          cpu_fallback=lambda w, n: [{"cpu": True}])
    assert out == [{"cpu": True}]
    out2 = ap.safe_process(lambda w, n: (_ for _ in ()).throw(ValueError("bad clip")), [np.zeros(5)], ["a"])
    assert out2 == [None]                                    # recorded as a failure, never a fabricated row


def test_batch_policy_for_large_models():
    from wildspeech.asr.inference import batch_policy
    from wildspeech.config import Config

    cfg = Config({"v3": {"asr": {"batch": {"large_model_params": 300_000_000, "large_model_batch_size": 2,
                                           "small_model_batch_size": 8, "batch_seconds_large": 60.0,
                                           "batch_seconds_small": 240.0}}}})
    assert batch_policy(315_000_000, cfg)["max_batch_size"] == 2
    assert batch_policy(95_000_000, cfg)["max_batch_size"] == 8


def test_checkpoint_resume_and_invalidation(tmp_path):
    from wildspeech.asr.checkpointing import ASRCheckpointStore, asr_fingerprint, load_all_asr

    st = ASRCheckpointStore(tmp_path, "fake_model")
    fp1 = asr_fingerprint("fake_model", "src", "rev1", {"a": 1})
    assert st.prepare("train", fp1, {"model_key": "fake_model"}) == "empty"
    rows = [{FILE: f"c{i}.flac", "transcript": "cat", "_embedding": np.arange(4, dtype=np.float32)} for i in range(5)]
    st.append("train", rows[:3], "fake@rev1")
    st.add_failures("train", [{FILE: "c4.flac", "stage": "process", "error": "x"}])
    assert st.done("train") == {"c0.flac", "c1.flac", "c2.flac"}
    assert st.todo("train", [r[FILE] for r in rows]) == ["c3.flac"]                 # resume skips done + failed
    assert st.todo("train", [r[FILE] for r in rows], retry_failures=True) == ["c3.flac", "c4.flac"]
    assert st.prepare("train", fp1, {"model_key": "fake_model"}) == "valid"
    emb = st.load_embeddings("train")
    assert emb.shape == (3, 5) and emb["e0003"].iloc[0] == 3
    long = load_all_asr(tmp_path)
    assert set(long["model_key"]) == {"fake_model"} and len(long) == 3
    fp2 = asr_fingerprint("fake_model", "src", "rev2", {"a": 1})
    assert st.prepare("train", fp2, {"model_key": "fake_model"}) == "invalidated"
    assert st.done("train") == set()
    assert len(st.provenance("train")["runs"]) == 3


def test_submission_schema_validation():
    from wildspeech.submission import validate_submission

    fmt = pd.DataFrame({ID: ["a", "b", "c"], TARGET: 0.0})
    ok = pd.DataFrame({ID: ["a", "b", "c"], TARGET: [100.0, 95.0, 110.0]})
    assert (validate_submission(ok, fmt)["status"] == "PASS").all()
    bad = pd.DataFrame({ID: ["a", "c", "b"], TARGET: [100.0, np.nan, 100.0]})
    chk = validate_submission(bad, fmt).set_index("check")["status"]
    assert chk["same order"] == "FAIL" and chk["finite target"] == "FAIL"
    const = pd.DataFrame({ID: ["a", "b", "c"], TARGET: 100.0})
    assert validate_submission(const, fmt).set_index("check").loc["non-constant predictions", "status"] == "FAIL"
