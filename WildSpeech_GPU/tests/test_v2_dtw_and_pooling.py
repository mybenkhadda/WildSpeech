"""DTW (numpy reference vs batched torch, invariances) and embedding pooling / masking."""
import numpy as np
import pytest

from wildspeech.reference.dtw import dtw_numpy, dtw_with_path


def _seq(n, d=13, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, d)), axis=0).astype(np.float32) / 5


def test_dtw_identical_sequences_cost_zero():
    a = _seq(40)
    r = dtw_numpy(a, a)
    assert r["cost"] < 1e-3 and r["path_len"] == 40 and r["nondiag_frac"] == 0.0 and r["diag_dev"] < 1e-12


def test_dtw_time_warped_copy_is_closer_than_unrelated():
    a = _seq(60, seed=1)
    warped = np.repeat(a, 2, axis=0)[::3]                     # tempo change
    other = _seq(60, seed=9)
    assert dtw_numpy(a, warped)["norm_cost"] < dtw_numpy(a, other)["norm_cost"]
    assert dtw_numpy(a, warped)["nondiag_frac"] > 0


def test_dtw_path_is_monotone_and_complete():
    a, b = _seq(30, seed=2), _seq(45, seed=3)
    _, C, path = dtw_with_path(a, b, None)
    assert path[0] == (0, 0) and path[-1] == (29, 44)
    steps = np.diff(np.array(path), axis=0)
    assert np.all((steps >= 0) & (steps <= 1)) and np.all(steps.sum(1) >= 1)


@pytest.mark.parametrize("band", [None, 0.35])
def test_dtw_torch_batch_matches_numpy(band):
    pytest.importorskip("torch")
    from wildspeech.reference.dtw import dtw_batch_torch

    A = [_seq(n, seed=s) for n, s in ((25, 1), (40, 2), (12, 3), (33, 4))]
    B = [_seq(n, seed=s) for n, s in ((30, 5), (22, 6), (18, 7), (33, 8))]
    got = dtw_batch_torch(A, B, "cpu", band, window=5)
    for a, b, g in zip(A, B, got):
        ref = dtw_numpy(a, b, band, window=5)
        for k in ("cost", "path_len", "norm_cost", "nondiag_frac", "diag_dev", "local_max", "local_max_pos"):
            assert abs(ref[k] - g[k]) <= 1e-3 * (1 + abs(ref[k])), (k, ref[k], g[k])


def test_masked_pooling_is_padding_invariant():
    torch = pytest.importorskip("torch")
    from wildspeech.foundation.pooling import lengths_to_mask, masked_mean, masked_std, pool_layers

    rng = np.random.default_rng(0)
    h = torch.from_numpy(rng.normal(size=(1, 7, 4)).astype(np.float32))
    padded = torch.cat([h, torch.full((1, 5, 4), 99.0)], 1)
    m1, m2 = lengths_to_mask([7], 7, "cpu"), lengths_to_mask([7], 12, "cpu")
    assert torch.allclose(masked_mean(h, m1), masked_mean(padded, m2), atol=1e-6)
    assert torch.allclose(masked_std(h, m1), masked_std(padded, m2), atol=1e-6)
    out = pool_layers([padded, padded * 2], [0, 1], m2, m2, ("mean", "std", "speech_mean"))
    assert tuple(out.shape) == (1, 2, 3, 4)
    assert torch.allclose(out[0, 1, 0], 2 * out[0, 0, 0], atol=1e-5)


def test_speech_mask_marks_energy_frames():
    pytest.importorskip("torch")
    from wildspeech.foundation.pooling import speech_mask_from_waves

    sr = 16000
    y = np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr), np.zeros(sr // 2)]).astype(np.float32)
    m = speech_mask_from_waves([y], [75], 80, sr, "cpu").numpy()[0]
    assert m[30:45].mean() > 0.9 and m[:20].mean() < 0.1 and m[75:].sum() == 0


def test_linear_cka_properties():
    from wildspeech.foundation.compression import linear_cka

    X = np.random.default_rng(0).normal(size=(200, 10))
    assert abs(linear_cka(X, X) - 1) < 1e-9
    assert abs(linear_cka(X, 3 * X @ np.linalg.qr(np.random.default_rng(1).normal(size=(10, 10)))[0]) - 1) < 1e-6
    assert linear_cka(X, np.random.default_rng(2).normal(size=(200, 10))) < 0.3
