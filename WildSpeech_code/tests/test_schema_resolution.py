import numpy as np
import pandas as pd
import pytest

from wildspeech.data_audit import (SchemaResolutionError, normalize_correct, resolve_fields, resolve_thetas,
                                   task_family)


def test_unambiguous_clip_fields_resolve():
    df = pd.DataFrame({"Student ID": ["a"], "clip_id": ["c1"], "filename": ["c1.wav"], "task": ["Blending"],
                       "item_id": ["i1"], "expected_text": ["cat"], "is_correct": [1]})
    res = resolve_fields(df, "clip", {})
    a = res["assigned"]
    assert a["student_id"] == "Student ID"
    assert a["clip_id"] == "clip_id"
    assert a["audio_file"] == "filename"
    assert a["correct"] == "is_correct"
    assert not res["missing_required"]


def test_ambiguous_fields_are_reported_not_guessed():
    df = pd.DataFrame({"student": ["a"], "child_id": ["a"], "score_a": [1.0], "score_b": [2.0]})
    res = resolve_fields(df, "labels", {})
    assert "student_id" in res["ambiguous"]
    assert "student_id" not in res["assigned"]
    assert "target" in res["ambiguous"]


def test_overrides_take_precedence_and_are_validated():
    df = pd.DataFrame({"student": ["a"], "child_id": ["a"], "y_eoy": [1.0]})
    res = resolve_fields(df, "labels", {"labels": {"student_id": "child_id"}})
    assert res["assigned"]["student_id"] == "child_id"
    with pytest.raises(SchemaResolutionError):
        resolve_fields(df, "labels", {"labels": {"student_id": "missing_col"}})


def test_theta_resolution_separates_standard_errors():
    df = pd.DataFrame({"student_id": ["a", "b"], "blending_theta": [0.1, 0.2], "blending_theta_se": [0.3, 0.3],
                       "deletion_theta": [0.0, 1.0], "sentence_repetition_theta": [1.0, 2.0]})
    th = resolve_thetas(df, {"student_id"}, {})
    assert th["theta"] == {"blending": "blending_theta", "deletion": "deletion_theta", "repetition": "sentence_repetition_theta"}
    assert th["theta_se"]["blending"] == "blending_theta_se"


def test_correctness_normalisation():
    c, _, info = normalize_correct(pd.Series(["Correct", "incorrect", None, "weird"]))
    assert c.tolist()[:2] == [1.0, 0.0] and np.isnan(c.iloc[2]) and np.isnan(c.iloc[3])
    assert "weird" in info["unmapped_values"]
    c, raw, info = normalize_correct(pd.Series([0.0, 0.5, 1.0]))
    assert info["mode"] == "PARTIAL_SCORE" and c.isna().all() and raw.tolist() == [0.0, 0.5, 1.0]


def test_task_family_mapping():
    assert task_family("Phoneme Blending", {}) == "blending"
    assert task_family("Sentence Repetition", {}) == "repetition"
    assert task_family("elision", {}) == "deletion"
    assert task_family("mystery", {}).startswith("other_")
    assert task_family("mystery", {"mystery": "blending"}) == "blending"
