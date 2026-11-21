"""V4 joins (split + normalized filename, one-to-one, clip_id via manifest), recognized-text features, measured
phonetic parsing / availability. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE, ID
from wildspeech.v4 import joins as J
from wildspeech.v4 import phonetic_features as PF
from wildspeech.v4 import text_features as TF


def _meta():
    return pd.DataFrame({"split": ["train", "train", "test", "test"], FILE: ["A1.flac", "B2.flac", "C3.flac", "D4.flac"],
                         "filename_norm": ["a1", "b2", "c3", "d4"], ID: ["s1", "s1", "t1", "t2"],
                         "task": ["blending", "deletion", "blending", "sentence_repetition"],
                         "expected_text": ["pot", "sun", "boy", "the dog ran"], "item_score": [1.0, 0.0, np.nan, 1.0],
                         "duration_sec": [1.0, 1.2, 0.9, 2.5]})


def _canon(**over):
    base = {"model_key": ["m", "m", "m", "m"], "model_id": ["m"] * 4, "model_version": ["v"] * 4,
            "source_row": [0, 1, 2, 3], "clip_ref": [None] * 4, "audio_ref": ["a1.flac", "B2", "x/c3.wav", "zz.flac"],
            "filename_norm": ["a1", "b2", "c3", "zz"], "split_src": [None, "train", "train", None],
            "transcript": ["pot", "sun", "boy", "x"], "phones": [None] * 4, "has_phones": [False] * 4,
            "asr_ok": [True] * 4, "error": [""] * 4, "confidence": [np.nan] * 4, "mean_token_logprob": [np.nan] * 4,
            "has_token_probs": [False] * 4, "decode_seconds": [np.nan] * 4, "audio_seconds": [np.nan] * 4,
            "sample_rate": [np.nan] * 4, "task_src": [None] * 4}
    base.update(over)
    return pd.DataFrame(base)


def test_join_uses_metadata_split_and_reports_conflicts_and_unmatched():
    joined, unmatched, audit = J.join_asr(_canon(), _meta(), None)
    assert set(joined[FILE]) == {"A1.flac", "B2.flac", "C3.flac"}
    assert joined.set_index(FILE).loc["C3.flac", "split"] == "test"          # metadata split wins over the label
    assert len(unmatched) == 1 and unmatched["audio_ref"].iloc[0] == "zz.flac"
    conflict = [a for a in audit if a["check"] == "ASR split label agrees with metadata"][0]
    assert conflict["value"] == 1 and conflict["status"] == "WARN"


def test_join_resolves_clip_id_only_through_manifest():
    c = _canon(clip_ref=["h1", "h2", None, None], audio_ref=[None, None, "c3", "zz"], filename_norm=["", "", "c3", "zz"])
    man = J.normalize_manifest(pd.DataFrame({"clip_id": ["h1", "h2"], "audio_name": ["A1.flac", "B2.flac"],
                                             "split_hint": ["train", "train"]}), None, (".flac",), "external:x")
    joined, _, _ = J.join_asr(c, _meta(), man)
    assert set(joined[FILE]) >= {"A1.flac", "B2.flac"}
    # without a manifest a hashed clip id is never assumed to be a filename
    joined2, unmatched2, _ = J.join_asr(c, _meta(), None)
    assert "A1.flac" not in set(joined2[FILE]) and len(unmatched2) >= 2


def test_join_asserts_one_to_one_metadata():
    m = _meta()
    m.loc[1, "filename_norm"] = "a1"                                         # collision inside the train split
    with pytest.raises(Exception):
        J.join_asr(_canon(), m, None)


def test_coverage_counts():
    joined, unmatched, _ = J.join_asr(_canon(), _meta(), None)
    cov = J.coverage_table(joined, unmatched, pd.DataFrame(columns=["model_key", "split"]), _meta())
    tr = cov[(cov["model_key"] == "m") & (cov["split"] == "train")].iloc[0]
    assert tr["n_metadata_clips"] == 2 and tr["n_usable"] == 2 and tr["usable_frac"] == pytest.approx(1.0)
    te = cov[(cov["model_key"] == "m") & (cov["split"] == "test")].iloc[0]
    assert te["n_usable"] == 1 and te["n_missing_predictions"] == 1


def test_text_features_task_specific_and_missing():
    f = TF.clip_text_features("s w ee t", "sweet", "blending")
    assert f["bl_segmented"] == 1.0 and f["bl_exact"] == 0.0
    g = TF.clip_text_features("sand", "and", "deletion")
    assert g["dl_excess_chars"] == 1.0 and g["dl_contains_expected"] == 1.0
    h = TF.clip_text_features("the dog", "the dog ran fast", "sentence_repetition")
    assert h["sr_omission_rate"] == pytest.approx(0.5) and h["sr_second_half_err"] == pytest.approx(1.0)
    assert TF.clip_text_features(None, "pot", "blending") == {}              # missing prediction -> no features
    e = TF.clip_text_features("", "pot", "blending")
    assert e["empty"] == 1.0 and e["w_er"] == 1.0


def test_disagreement_needs_two_recognizers():
    joined = pd.DataFrame({FILE: ["a", "a", "b"], "model_key": ["m1", "m2", "m1"], "asr_ok": [True, True, True],
                           "transcript": ["the dog", "the dot", "pot"]})
    long = pd.DataFrame({FILE: ["a", "a", "b"], "model_key": ["m1", "m2", "m1"], "asr_ok": [True] * 3,
                         "w_er": [0.0, 0.5, 0.0]})
    d = TF.disagreement_table(joined, long, ["m1", "m2"])
    assert list(d[FILE]) == ["a"] and d["dis_word_mean"].iloc[0] == pytest.approx(0.5)
    assert TF.disagreement_table(joined, long, ["m1"]).empty


def test_parse_phones_arpabet_and_ipa():
    p, r = PF.parse_phones("P AA1 T")
    assert p == ["P", "AA", "T"] and r == 1.0
    q, _ = PF.parse_phones("p ɑ t")
    assert q == ["P", "AA", "T"]
    assert PF.parse_phones("")[0] == []


def test_phonetic_unavailable_without_phone_strings():
    cov = pd.DataFrame([{"model_key": "m", "split": "train", "n_phone_strings": 0, "n_metadata_clips": 100}])

    class FakeG2P:
        exact = True

    pm, av = PF.availability(cov, ["m"], {"min_nonempty_frac": 0.3}, FakeG2P())
    assert pm == [] and (av["status"] == "UNAVAILABLE").all()
