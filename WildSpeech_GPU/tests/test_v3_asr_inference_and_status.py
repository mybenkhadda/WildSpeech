"""Adaptive micro-batching, precision guard helpers, VRAM-aware batch policy, framework checks, and orchestrator status
semantics (an eligibility-only P24 can never be recorded as completed ASR inference)."""
import json
from pathlib import Path

import numpy as np
import pytest

from wildspeech.config import Config


# --------------------------------------------------------------------------------------- micro-batching
class FakeOOM(MemoryError):
    pass


def _batcher(max_size, grow_after=2):
    from wildspeech.asr.inference import AdaptiveMicroBatcher

    return AdaptiveMicroBatcher(max_size, grow_after, is_oom=lambda e: isinstance(e, FakeOOM), release=lambda: None)


def test_micro_batcher_halves_on_oom_and_regrows():
    calls = []

    def fn(w, n):
        calls.append(len(w))
        if len(w) > 2:
            raise FakeOOM()
        return [x * 10 for x in w]

    b = _batcher(8, grow_after=2)
    out = b.run(fn, list(range(8)), [str(i) for i in range(8)])
    assert out == [x * 10 for x in range(8)]                 # order preserved, nothing dropped
    # 8 OOM -> 4 OOM -> 2 ok, 2 ok -> regrow to 4 -> OOM -> 2 ok, 2 ok -> regrow to 4
    assert calls == [8, 4, 2, 2, 4, 2, 2]
    assert b.shrinks == 3 and b.min_seen == 2 and b.size == 4


def test_micro_batcher_single_clip_oom_is_reraised():
    b = _batcher(4)
    with pytest.raises(FakeOOM):
        b.run(lambda w, n: (_ for _ in ()).throw(FakeOOM()), [1], ["a"])


def test_micro_batcher_non_oom_errors_propagate():
    b = _batcher(4)
    with pytest.raises(ValueError):
        b.run(lambda w, n: (_ for _ in ()).throw(ValueError("bad")), [1, 2], ["a", "b"])


# -------------------------------------------------------------------------------------- precision guard
def test_suspicious_rows():
    from wildspeech.asr.inference import suspicious_rows

    ok = {"transcript": "cat", "has_token_probs": True, "greedy_ll": -0.2, "n_frames": 10}
    nan_tok = {"transcript": "cat", "has_token_probs": False, "greedy_ll": np.nan}
    nan_ctc = {"transcript": "", "phones": "K AE T", "has_token_probs": True, "greedy_ll": float("nan"), "n_frames": 10}
    silent = {"transcript": "", "phones": "", "has_token_probs": False, "greedy_ll": -0.1}
    assert suspicious_rows([ok, silent], True, True) == 0
    assert suspicious_rows([nan_tok], True, False) == 1
    assert suspicious_rows([nan_ctc], False, True) == 1


def test_set_fp32_switches_model_dtype():
    torch = pytest.importorskip("torch")
    from wildspeech.asr.base import BaseASRAdapter

    ad = BaseASRAdapter(None, "src", {}, "cpu", Config({"v3": {"asr": {}}}))
    ad.model = torch.nn.Linear(2, 2).to(torch.float16)
    ad.dtype_name = "fp16"
    ad.set_fp32()
    assert next(ad.model.parameters()).dtype == torch.float32 and ad.dtype_name == "fp32"


def test_registry_precision_overrides_config():
    from types import SimpleNamespace

    from wildspeech.asr.base import BaseASRAdapter
    from wildspeech.asr.inference import decoding_settings

    cfg = Config({"v3": {"asr": {"precision": "auto"}}})
    spec = SimpleNamespace(precision="fp32")
    assert BaseASRAdapter(spec, "s", {}, "cuda", cfg).precision == "fp32"
    assert decoding_settings(cfg, spec)["precision"] == "fp32"
    assert decoding_settings(cfg, None)["precision"] == "auto"


# --------------------------------------------------------------------------------------------- batch policy
BATCH = Config({"v3": {"asr": {"batch": {"large_model_params": 300_000_000, "large_model_batch_size": 2,
                                         "small_model_batch_size": 8, "batch_seconds_large": 60.0,
                                         "batch_seconds_small": 240.0}}}})


def test_batch_policy_vram_aware_on_8gb():
    from wildspeech.asr.inference import batch_policy

    assert batch_policy(315_000_000, BATCH, vram_gb=8.0)["max_batch_size"] <= 2
    assert batch_policy(242_000_000, BATCH, vram_gb=8.0)["max_batch_size"] <= 8
    assert batch_policy(242_000_000, BATCH, vram_gb=1.0)["max_batch_size"] == 1          # never 0
    assert batch_policy(95_000_000, BATCH)["max_batch_size"] == 8                       # unchanged without VRAM info


def test_batch_policy_registry_override():
    from types import SimpleNamespace

    from wildspeech.asr.inference import batch_policy

    p = batch_policy(95_000_000, BATCH, 8.0, 2, SimpleNamespace(batch_size=3, batch_seconds=45.0))
    assert p["max_batch_size"] == 3 and p["batch_seconds"] == 45.0


# ------------------------------------------------------------------------------------------ environment
def test_version_parsing_and_missing_package():
    from wildspeech.asr.environment import _ver_tuple, package_version

    assert _ver_tuple("2.1.0+cu121") == (2, 1, 0) and _ver_tuple("4.38.0.dev0") >= (4, 38, 0)
    ok, v, why = package_version("definitely_not_a_package_xyz")
    assert not ok and v is None and "not importable" in why


def test_framework_check_contract(monkeypatch):
    from types import SimpleNamespace

    from wildspeech.asr import environment as E

    monkeypatch.setattr(E, "package_version", lambda n: (True, "9.9", ""))
    spec = SimpleNamespace(adapter="whisper", expected_architectures=["hf_whisper"])
    assert E.framework_check(spec, "hf_whisper").ok
    bad = E.framework_check(spec, "hf_ctc")
    assert not bad.ok and "not in expected" in bad.detail and "cannot drive" in bad.detail
    ps = SimpleNamespace(adapter="pasketti_phonetic", expected_architectures=["hf_ctc", "hf_whisper"])
    assert E.framework_check(ps, "hf_ctc").ok and not E.framework_check(ps, "nemo").ok


def test_component_check_on_synthetic_ctc_snapshot(tmp_path):
    pytest.importorskip("transformers")
    from types import SimpleNamespace

    from wildspeech.asr.environment import component_check

    d = tmp_path / "ctc"
    d.mkdir()
    letters = {c: i + 2 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    (d / "vocab.json").write_text(json.dumps({"<pad>": 0, "|": 1, **letters}), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"],
                                               "vocab_size": 28}), encoding="utf-8")
    (d / "preprocessor_config.json").write_text(json.dumps({"feature_extractor_type": "Wav2Vec2FeatureExtractor",
                                                            "sampling_rate": 16000, "feature_size": 1,
                                                            "padding_value": 0.0, "do_normalize": True,
                                                            "return_attention_mask": False}), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Wav2Vec2CTCTokenizer", "pad_token": "<pad>",
                                                         "word_delimiter_token": "|"}), encoding="utf-8")
    spec = SimpleNamespace(key="toy", output="chars")
    out = component_check(spec, str(d), "hf_ctc")
    assert out["ok"], out["problems"]
    assert out["letters"] == 26 and out["word_delimiter"] == "|" and out["sampling_rate"] == 16000


# ------------------------------------------------------------------------------------- orchestrator status
def test_final_status_mapping():
    from wildspeech.orchestrator import NOT_EXECUTED_STATUS, final_status

    assert final_status("P24", {"status": "NOT_EXECUTED"}) == NOT_EXECUTED_STATUS
    assert final_status("P27", {"status": "NOT_EXECUTED"}) == NOT_EXECUTED_STATUS
    assert final_status("P24", {"status": "done", "stage_outcome": "INFERENCE_PARTIAL"}) == "completed"
    assert final_status("P18", {"status": "NOT_EXECUTED"}) == "completed"     # V1/V2 semantics untouched
    assert final_status("P24", {}) == "completed"


def test_stage_specs_gate_on_outputs_not_on_p24_status():
    from wildspeech.orchestrator import STAGES

    assert STAGES["P24"].state and STAGES["P25"].state
    assert "P24" not in STAGES["P25"].deps and "P24" in STAGES["P25"].soft
    assert STAGES["P26"].deps == ("P25",)


@pytest.fixture()
def v3ctx(project):
    from wildspeech.context import build_context

    return build_context("FAST", overrides={"runtime": {"device": "cpu"},
                                            "v3": {"asr": {"enabled": True}, "phonetics": {"enabled": True},
                                                   "discovery": {"enabled": True}}}, root=project)


def test_eligibility_only_p24_is_not_completed_and_blocks_dependants(v3ctx, monkeypatch):
    from wildspeech.orchestrator import Pipeline
    from wildspeech.stages import p24_asr, p25_phonetic_features, p26_correct_only

    monkeypatch.setattr(p24_asr, "run", lambda ctx: {"status": "NOT_EXECUTED", "reason": "eligibility only",
                                                     "stage_outcome": "ELIGIBILITY_ONLY", "inference_executed": False})
    monkeypatch.setattr(p25_phonetic_features, "run", lambda ctx: {"status": "NOT_EXECUTED", "reason": "no outputs"})
    monkeypatch.setattr(p26_correct_only, "run", lambda ctx: pytest.fail("P26 must not run without P25"))
    pipe = Pipeline(v3ctx)
    res = pipe.run(["P24", "P25", "P26"], force=["P24", "P25", "P26"]).set_index("stage")["status"]
    assert res["P24"] == "not_executed" and res["P25"] == "not_executed" and res["P26"] == "blocked"
    st = pipe.status("P24")
    assert st["status"] == "not_executed" and st["summary"]["inference_executed"] is False
    assert "eligibility only" in st["note"]
    again = pipe.run(["P24"]).set_index("stage")["status"]
    assert again["P24"] == "not_executed"                      # never cached as a success


def test_p24_fingerprint_follows_reviews_file(v3ctx):
    from wildspeech.orchestrator import Pipeline

    pipe = Pipeline(v3ctx)
    f0 = pipe.fingerprint("P24")
    assert pipe.fingerprint("P24") == f0                        # deterministic
    p = Path(v3ctx.paths.configs) / "v3_asr_reviews.yaml"
    old = p.read_text(encoding="utf-8")
    try:
        p.write_text(old + "\n# a reviewer touched this file\n", encoding="utf-8")
        assert pipe.fingerprint("P24") != f0
    finally:
        p.write_text(old, encoding="utf-8")


def test_v1_v2_fingerprints_do_not_include_state(v3ctx):
    from wildspeech.orchestrator import STAGES

    assert all(STAGES[s].state is None for s in STAGES if s not in ("P24", "P25"))
