"""Phonetic error calculations: Levenshtein ops, S/I/D, sequence preservation, positions, consonant/vowel split,
repetitions / restarts, ARPAbet <-> IPA mapping, confusion counts."""
import numpy as np

from wildspeech.phonetic.error_features import (align_ops, clip_error_features, confusion_counts, cv_profile, error_summary, lcs_len,
                                                ops_from_string, ops_to_string, position_profile, repetitions, restarts)
from wildspeech.phonetic.g2p import G2P, ipa_to_arpa, is_vowel


def test_align_ops_counts():
    ref, hyp = list("KAET"), list("KAT")
    ops = align_ops(ref, hyp)
    kinds = [o[0] for o in ops]
    assert kinds.count("D") == 1 and kinds.count("M") == 3
    s = error_summary(["P", "AA", "T"], ["B", "AA", "T", "S"], "ph")
    assert s["ph_sub"] == 1 and s["ph_ins"] == 1 and s["ph_del"] == 0
    assert np.isclose(s["ph_er"], 2 / 3) and np.isclose(s["ph_seq_preservation"], 2 / 3)
    assert error_summary(["A"], ["A"], "x")["x_exact"] == 1.0


def test_ops_roundtrip_and_confusion():
    ops = align_ops(["P", "AA", "T"], ["B", "AA"])
    s = ops_to_string(ops)
    assert ops_from_string(s) == [(k, r, h, p) for k, r, h, p in ops]
    mat = confusion_counts([s])
    assert mat.loc["P", "B"] == 1 and mat.loc["T", "∅"] == 1 and mat.loc["AA", "AA"] == 1


def test_lcs_and_positions():
    assert lcs_len(list("ABCBDAB"), list("BDCABA")) == 4
    ops = align_ops(["S", "W", "IY", "T"], ["W", "IY", "T"])          # initial deletion
    prof = position_profile(ops, 4, 3, "ph")
    assert prof["ph_err_initial"] > 0 and prof["ph_err_final"] == 0


def test_consonant_vowel_profile():
    ops = align_ops(["P", "AA", "T"], ["P", "IY", "T"])
    cv = cv_profile(ops)
    assert cv["ph_vowel_err"] == 1.0 and cv["ph_consonant_err"] == 0.0
    assert is_vowel("AA1") and not is_vowel("T")


def test_repetitions_and_restarts():
    assert repetitions(["the", "the", "dog"]) >= 1
    assert restarts(["p", "pot"], ["pot"]) == 1
    assert restarts(["sweet"], ["sweet"]) == 0


def test_ipa_mapping():
    assert ipa_to_arpa("ɑː") == "AA" and ipa_to_arpa("ˈæ") == "AE" and ipa_to_arpa("tʃ") == "CH"
    assert ipa_to_arpa("ɾ") == "T" and ipa_to_arpa("") is None


def test_clip_error_features_word_and_char_without_exact_g2p():
    g = G2P(backend="rule")
    f, d = clip_error_features("put the chicken in freezer", "Put the chicken in the freezer.", g, 3, allow_approx=False)
    assert f["w_del"] == 1 and f["w_sub"] == 0 and np.isclose(f["w_er"], 1 / 6)
    assert "ph_per" not in f and d["ops"] == ""                   # approximate G2P -> no phoneme metrics
    f2, d2 = clip_error_features("pot", "Pot", g, 3, allow_approx=True)
    assert f2["ph_per"] == 0.0 and d2["ref_phones"]
