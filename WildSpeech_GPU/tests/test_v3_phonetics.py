"""Phoneme alignment, expected-text comparison, confidence without fabricated probabilities, timing, tasks."""
import json

import numpy as np

from wildspeech.phonetics.alignment import align, lcs_ratio, order_preservation, ops_from_string, ops_to_string, phonological_align
from wildspeech.phonetics.confidence import posterior_confidence, token_confidence
from wildspeech.phonetics.error_analysis import confusion_matrix, confusion_table, phoneme_features, word_features
from wildspeech.phonetics.g2p import consonant_clusters, phone_distance, substitution_type, syllable_count
from wildspeech.phonetics.task_features import task_features, timing_features


def _exp(phones, word_of=None):
    cl = [False] * len(phones)
    for s, e in consonant_clusters(phones):
        for k in range(s, e):
            cl[k] = True
    return {"phones": phones, "word_of": word_of or [0] * len(phones), "in_cluster": cl, "n_syllables": syllable_count(phones)}


def test_phone_distance_and_types():
    assert phone_distance("B", "B") == 0.0
    assert phone_distance("B", "P") < phone_distance("B", "IY") == 1.0
    assert substitution_type("B", "P") == "voicing"
    assert substitution_type("T", "K") == "place"
    assert substitution_type("B", "IY") == "cv_swap"


def test_alignment_roundtrip_and_counts():
    ops = align(list("kitten"), list("sitting"))
    assert sum(o.op != "M" for o in ops) == 3
    assert [o.op for o in ops_from_string(ops_to_string(ops))] == [o.op for o in ops]


def test_phonological_alignment_prefers_near_substitution():
    ops = phonological_align(["B", "AE", "T"], ["P", "AE", "T"])
    assert [o.op for o in ops] == ["S", "M", "M"]


def test_order_and_lcs():
    assert order_preservation(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert order_preservation(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert np.isnan(order_preservation(["a"], ["a"]))
    assert lcs_ratio(["a", "b", "c"], ["a", "c"]) == 2 / 3


def test_word_features_expected_text():
    f = word_features("the dog the dog by tree", "the dog by the tree")
    assert 0 < f["w_wer"] < 1 and f["w_completeness"] < 1 and f["w_repetitions"] >= 1
    g = word_features("", "cat")
    assert g["w_empty"] == 1.0 and g["w_wer"] == 1.0
    assert word_features("cat", "cat")["w_exact"] == 1.0


def test_phoneme_features_clusters_and_omissions():
    exp = _exp(["S", "W", "IY", "T"])                       # 'sweet': initial cluster S W
    f, ops = phoneme_features(exp, ["W", "IY", "T"])         # cluster reduction (S deleted)
    assert f["ph_initial_omission"] == 1.0 and f["ph_cluster_reduction"] > 0 and f["ph_cluster_acc"] < 1
    assert f["ph_per"] == 0.25 and f["ph_exact"] == 0.0
    f2, _ = phoneme_features(exp, ["S", "W", "IY", "T"])
    assert f2["ph_exact"] == 1.0 and f2["ph_near_correct"] == 1.0
    f3, _ = phoneme_features(exp, ["Z", "W", "IY", "T"])     # voicing substitution counts as near-correct
    assert f3["ph_sub_voicing_rate"] > 0 and f3["ph_near_correct"] > f["ph_near_correct"] - 1


def test_missing_token_probabilities_are_nan():
    f = token_confidence({"has_token_probs": False, "token_logprobs": ""})
    assert f["cf_has_token_probs"] == 0.0 and all(np.isnan(v) for k, v in f.items() if k != "cf_has_token_probs")
    g = token_confidence({"has_token_probs": True, "token_logprobs": json.dumps([-0.1, -3.0]), "token_entropy_mean": 0.5})
    assert g["cf_has_token_probs"] == 1.0 and g["cf_tok_low_frac"] == 0.5


def test_forced_alignment_is_not_evidence_without_free_match():
    exp = _exp(["K", "AE", "T"])
    spans = [{"p": "K", "s": 0, "e": 3, "lp": -0.1, "gop": 0.0, "post": 0.9},
             {"p": "AE", "s": 3, "e": 6, "lp": -0.2, "gop": 0.0, "post": 0.8},
             {"p": "T", "s": 6, "e": 9, "lp": -0.1, "gop": 0.0, "post": 0.9}]
    row = {"frame_entropy_mean": 0.3, "greedy_ll": -0.1, "blank_frac": 0.2, "expected_ll": -0.2, "n_frames": 9,
           "expected_alignment": json.dumps(spans)}
    _, ops_good = phoneme_features(exp, ["K", "AE", "T"])
    _, ops_bad = phoneme_features(exp, ["B", "IY"])          # free recognition heard something else
    good = posterior_confidence(row, exp, ops_good)
    bad = posterior_confidence(row, exp, ops_bad)
    assert good["cf_supported_frac"] == 1.0
    assert bad["cf_supported_frac"] < 1.0                     # high forced-alignment posterior alone is not support
    assert good["cf_gop_mean"] == bad["cf_gop_mean"]
    none = posterior_confidence({"expected_alignment": ""}, exp, None)
    assert none["cf_has_posteriors"] == 0.0 and np.isnan(none["cf_gop_mean"])


def test_timing_and_tasks():
    exp = _exp(["K", "AE", "T"])
    spans = [{"p": "K", "s": 10, "e": 14}, {"p": "AE", "s": 14, "e": 20}, {"p": "T", "s": 30, "e": 33}]
    tf = timing_features(spans, exp, 0.02, 5)
    assert abs(tf["tm_onset_s"] - 0.2) < 1e-9 and tf["tm_n_pauses"] == 1.0 and tf["tm_pause_within_word"] == 1.0
    wf = word_features("cat", "cat")
    pf, ops = phoneme_features(exp, ["K", "AE", "T"])
    bl = task_features("blending", wf, pf, ops, tf)
    assert bl["bl_integrated"] == 1.0 and bl["bl_segmented"] == 1.0
    pf2, ops2 = phoneme_features(_exp(["K", "AE", "T"]), ["K", "AE", "T", "S"])
    dl = task_features("deletion", {}, pf2, ops2, {})
    assert dl["dl_retained"] > 0 and dl["dl_retained_final"] == 1.0 and dl["dl_excessive"] == 0.0
    sr = task_features("sentence_repetition", word_features("the dog", "the dog by the tree"), {}, [], {},
                       ["the", "dog"], ["the", "dog", "by", "the", "tree"])
    assert sr["sr_second_half_err"] > sr["sr_first_half_err"]
    assert all(np.isnan(v) for v in timing_features([], exp).values())


def test_confusion_matrix_row_normalised():
    _, ops = phoneme_features(_exp(["S", "AE", "N", "D"]), ["S", "AE", "N"])
    long = confusion_table([ops_to_string(ops)] * 3)
    m = confusion_matrix(long, 10)
    assert np.allclose(m.sum(1).dropna(), 1.0)
    assert m.loc["D", "∅"] == 1.0
