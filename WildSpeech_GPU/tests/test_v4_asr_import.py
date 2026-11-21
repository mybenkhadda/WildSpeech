"""V4 external ASR import: schema inspection, canonical adapter, failures, duplicates, provenance gate.
Synthetic data only (no Drive, no models)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE
from wildspeech.v4 import EXTERNAL_PROVENANCE
from wildspeech.v4 import asr_import as AI

CFG4 = {"asr_import": {"success_values": ["ok", "success", "true", "1"], "failure_values": ["error", "failed", "false", "0"],
                       "permission_review": {"reviewed": True, "reviewer": "me", "date": "2026-11-21", "statement": "ok"},
                       "min_model_coverage": 0.5},
        "joins": {"audio_extensions": [".flac", ".wav"]}}


@pytest.mark.parametrize("raw,expected", [
    ("train_audio/ABC_01.FLAC", "abc_01"), ("C:\\x\\abc_01.wav", "abc_01"), ("abc_01", "abc_01"),
    ("  abc_01.flac ", "abc_01"), (None, ""), (float("nan"), ""), ("dir/sub/abc.def.flac", "abc.def"),
])
def test_normalize_clip_name(raw, expected):
    assert AI.normalize_clip_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [("train", "train"), ("TEST", "test"), ("train_audio.tar", "train"),
                                          ("test_audio/x.flac", "test"), ("contest", None), ("", None), (None, None)])
def test_normalize_split(raw, expected):
    assert AI.normalize_split(raw) == expected


def _long_export():
    return pd.DataFrame({
        "model": ["whisper", "whisper", "whisper", "w2v", "w2v", "w2v", "whisper"],
        "audio_name": ["a.flac", "b.flac", "c.flac", "a.flac", "b.flac", "c.flac", "a.flac"],
        "split_hint": ["train"] * 7,
        "text": ["pot", "", None, "pot", "boy", "zoo", "pot pot"],
        "status": ["ok", "ok", "error", "ok", "ok", "ok", "ok"],
        "error": ["", "", "CUDA OOM", "", "", "", ""],
        "inference_ms": [100, 120, np.nan, 80, 90, 85, 110],
        "avg_logprob": [-0.2, -1.5, np.nan, -0.1, -0.3, -0.4, -0.5],
        "outcome_score": [1, 2, 3, 4, 5, 6, 7],                 # a label field that must be ignored
    })


def test_inspect_long_schema_and_warnings():
    s = AI.inspect_asr_schema(_long_export())
    assert s["layout"] == "long"
    assert s["roles"]["model_id"] == "model" and s["roles"]["transcript"] == "text"
    assert s["roles"]["filename"] == "audio_name" and s["roles"]["split"] == "split_hint"
    assert s["roles"]["decode_seconds"] == "inference_ms" and s["roles"]["confidence"] == "avg_logprob"
    assert "outcome_score" in s["label_fields_present"]
    assert {m["model_id"] for m in s["models"]} == {"whisper", "w2v"}


def test_canonical_keeps_failures_and_distinguishes_empty_from_missing():
    raw = _long_export()
    s = AI.inspect_asr_schema(raw)
    c = AI.ms_to_seconds(AI.canonicalize(AI.to_long(raw, s), CFG4), s)
    assert len(c) == len(raw)                                         # nothing dropped
    fail = c[(c["model_key"] == "whisper") & (c["filename_norm"] == "c")].iloc[0]
    assert not fail["asr_ok"] and fail["failure_kind"] in ("error", "no_output", "status_failed")
    empty = c[(c["model_key"] == "whisper") & (c["filename_norm"] == "b")].iloc[0]
    assert empty["asr_ok"] and empty["transcript"] == ""              # recognised silence is an observation
    assert c["decode_seconds"].max() == pytest.approx(0.12)           # ms -> s
    assert (c["provenance_type"] == EXTERNAL_PROVENANCE).all()
    assert any(col.startswith("src__") for col in c.columns)
    assert "outcome_score" not in c.columns                           # only as src__ original field


def test_resolve_duplicates_reports_discarded_rows():
    raw = _long_export()
    s = AI.inspect_asr_schema(raw)
    c = AI.canonicalize(AI.to_long(raw, s), CFG4)
    c["split"], c[FILE] = "train", c["filename_norm"] + ".flac"
    kept, gone = AI.resolve_duplicates(c, ("model_key", "split", FILE))
    assert len(kept) == 6 and len(gone) == 1
    assert not kept.duplicated(["model_key", "split", FILE]).any()
    assert "duplicate_reason" in gone.columns


def test_wide_layout_detection_and_melt():
    raw = pd.DataFrame({"filename": ["a.flac", "b.flac"], "split": ["test", "test"],
                        "whisper_small_text": ["pot", "boy"], "parakeet_text": ["pot", None],
                        "whisper_small_confidence": [-0.1, -0.2]})
    s = AI.inspect_asr_schema(raw)
    assert s["layout"] == "wide"
    assert set(s["wide_model_columns"]) == {"whisper_small", "parakeet"}
    c = AI.canonicalize(AI.to_long(raw, s), CFG4)
    assert len(c) == 4
    assert set(c["model_key"]) == {"whisper_small", "parakeet"}
    miss = c[(c["model_key"] == "parakeet") & (c["filename_norm"] == "b")].iloc[0]
    assert not miss["asr_ok"]
    assert c.loc[c["model_key"] == "whisper_small", "confidence"].notna().all()


def test_explicit_column_map_overrides_detection():
    raw = pd.DataFrame({"m": ["x"], "wav_path": ["train_audio/q.flac"], "out": ["cat"]})
    s = AI.inspect_asr_schema(raw, column_map={"model_id": "m", "member": "wav_path", "transcript": "out"})
    c = AI.canonicalize(AI.to_long(raw, s), CFG4)
    assert c.loc[0, "filename_norm"] == "q" and c.loc[0, "split_src"] == "train" and c.loc[0, "transcript"] == "cat"
    with pytest.raises(KeyError):
        AI.inspect_asr_schema(raw, column_map={"transcript": "nope"})


def test_numeric_status_and_token_probabilities():
    raw = pd.DataFrame({"model": ["m", "m"], "file": ["a.wav", "b.wav"], "text": ["a", "b"], "success": [1, 0],
                        "token_probs": [[0.9, 0.5], "[0.2, 0.1]"]})
    s = AI.inspect_asr_schema(raw)
    c = AI.canonicalize(AI.to_long(raw, s), CFG4)
    assert bool(c.loc[0, "asr_ok"]) and not bool(c.loc[1, "asr_ok"])
    assert c.loc[0, "has_token_probs"] and c.loc[0, "mean_token_logprob"] < 0


def test_gate_requires_permission_review_and_coverage():
    joined = pd.DataFrame({"model_key": ["m", "m"], "model_id": ["m", "m"], "split": ["train", "train"],
                           FILE: ["a.flac", "b.flac"], "transcript": ["pot", "boy"], "has_phones": [False, False],
                           "asr_ok": [True, True]})
    cov = pd.DataFrame([{"model_key": "m", "split": "train", "usable_frac": 1.0, "n_usable": 2, "n_metadata_clips": 2,
                         "n_failed": 0}])
    prov = {"provenance_type": EXTERNAL_PROVENANCE, "source_sha256": "abc"}
    gate, usable = AI.validate_import(joined, cov, prov, CFG4, ("train",), need_test=False)
    assert usable == ["m"] and (gate["status"] != "FAIL").all()
    no_review = {**CFG4, "asr_import": {**CFG4["asr_import"], "permission_review": {"reviewed": False}}}
    gate2, usable2 = AI.validate_import(joined, cov, prov, no_review, ("train",), need_test=False)
    assert usable2 == [] and (gate2.loc[gate2["check"] == "permission_review", "status"] == "FAIL").all()


def test_external_provenance_is_never_a_p24_output():
    from wildspeech.asr.outputs import _origin

    origin, accepted, why = _origin({"provenance_type": EXTERNAL_PROVENANCE, "producer": EXTERNAL_PROVENANCE,
                                     "eligibility": {"eligible": True}}, accept_fixture=False)
    assert origin == EXTERNAL_PROVENANCE
    assert accepted is False
    assert "v4" in why.lower()
