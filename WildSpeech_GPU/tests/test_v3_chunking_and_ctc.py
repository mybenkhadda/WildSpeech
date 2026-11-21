"""Audio chunking + reconstruction, frame stitching, transcript merging, CTC greedy decoding, ASR output schema."""
import json

import numpy as np
import pytest

from wildspeech.asr.base import (ASR_COLUMNS, ASRResult, ModelSlot, merge_transcripts, plan_chunks, reconstruct_waveform,
                                 split_waveform, stitch_frames)


@pytest.mark.parametrize("n", [100, 16000, 16000 * 25, 16000 * 61 + 7, 16000 * 130])
def test_chunks_cover_and_reconstruct_exactly(n):
    sr = 16000
    w = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    pieces, plan = split_waveform(w, sr, 25.0, 2.0)
    assert plan[0].start == 0 and plan[-1].end == n
    cores = [(c.core_start, c.core_end) for c in plan]
    assert cores[0][0] == 0 and cores[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(cores, cores[1:]))           # cores tile the signal
    assert all(c.start <= c.core_start < c.core_end <= c.end for c in plan)
    assert all(len(p) <= 25 * sr for p in pieces)
    np.testing.assert_array_equal(reconstruct_waveform(pieces, plan), w)


def test_overlap_must_be_shorter():
    with pytest.raises(ValueError):
        plan_chunks(100000, 16000, 2.0, 2.0)


def test_stitch_frames_length_matches_audio():
    sr, spf = 16000, 320
    n = sr * 70
    plan = plan_chunks(n, sr, 25.0, 2.0)
    frames = [np.ones(((c.end - c.start) // spf, 5)) * c.index for c in plan]
    st = stitch_frames(frames, plan, spf)
    assert abs(st.shape[0] - n // spf) <= len(plan)
    assert (np.diff(st[:, 0]) >= 0).all()                                 # chunks appear in order


def test_merge_transcripts_removes_boundary_duplicates():
    assert merge_transcripts(["the dog by the tree", "the tree in the park"]) == "the dog by the tree in the park"
    assert merge_transcripts(["a b", "c d"]) == "a b c d"
    assert merge_transcripts(["", "x"]) == "x"


def test_asr_row_never_fabricates_probabilities():
    r = ASRResult(transcript="cat", raw_tokens="c a t", token_logprobs=None).row()
    assert set(ASR_COLUMNS) <= set(r)
    assert r["has_token_probs"] is False and np.isnan(r["mean_token_logprob"]) and r["token_logprobs"] == ""
    r2 = ASRResult(transcript="cat", token_logprobs=[-0.1, -0.5]).row()
    assert r2["has_token_probs"] is True and json.loads(r2["token_logprobs"]) == [-0.1, -0.5]
    assert abs(r2["mean_token_logprob"] - (-0.3)) < 1e-9


def test_model_slot_is_exclusive():
    with ModelSlot("a"):
        with pytest.raises(RuntimeError):
            with ModelSlot("b"):
                pass
    with ModelSlot("b"):
        assert ModelSlot.owner() == "b"
    assert ModelSlot.owner() is None


def _fake_ctc(kind="ctc_char"):
    from wildspeech.asr.ctc_adapter import CTCAdapter

    a = object.__new__(CTCAdapter)
    vocab = {"<pad>": 0, "|": 1, "C": 2, "A": 3, "T": 4}
    a.vocab, a.inv, a.blank, a.delim, a.kind = vocab, {v: k for k, v in vocab.items()}, 0, 1, kind
    a.special, a.upper = {0}, True
    return a


def _lp(path, V=5, p=0.9):
    lp = np.full((len(path), V), np.log((1 - p) / (V - 1)))
    for t, k in enumerate(path):
        lp[t, k] = np.log(p)
    return lp


def test_ctc_greedy_collapses_repeats_and_blanks():
    a = _fake_ctc()
    toks, scores = a.greedy(_lp([0, 2, 2, 0, 3, 3, 3, 0, 4, 0, 1, 0, 2, 0, 0]))
    assert toks == [2, 3, 4, 1, 2]
    assert len(scores) == len(toks) and all(s <= 0 for s in scores)
    text, phones, raw = a.decode(toks)
    assert text == "cat c" and phones == ""


def test_ctc_repeated_letter_needs_blank():
    a = _fake_ctc()
    toks, _ = a.greedy(_lp([2, 2, 0, 2]))
    assert toks == [2, 2]


def test_ctc_char_targets():
    a = _fake_ctc()
    ids, labs = a.char_targets("Cat!")
    assert ids == [2, 3, 4] and labs == ["c", "a", "t"]


def test_forced_alignment_spans():
    pytest.importorskip("torchaudio")
    import torch

    a = _fake_ctc()
    a.torch = torch
    lp = _lp([0, 2, 2, 0, 3, 0, 4, 4, 0])
    ell, spans = a.expected_alignment(lp.astype(np.float32), [2, 3, 4], ["c", "a", "t"])
    assert np.isfinite(ell) and spans is not None and [s["p"] for s in spans] == ["c", "a", "t"]
    assert all(s["gop"] <= 1e-6 for s in spans)
    ell2, sp2 = a.expected_alignment(lp[:2].astype(np.float32), [2, 3, 4], ["c", "a", "t"])
    assert sp2 is None                                                    # too few frames: no alignment invented


def test_parse_phone_string():
    from wildspeech.asr.whisper_adapter import parse_phone_string

    assert parse_phone_string("K AE1 T") == ["K", "AE", "T"]
    assert parse_phone_string("kæt") == ["K", "AE", "T"]
    assert parse_phone_string("tʃ") == ["CH"]
