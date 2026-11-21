"""Numerical tests of the torch extractor on the CPU device (identical code path to CUDA)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wildspeech.acoustic.cpu_reference import compare_parity, numpy_parity_stats  # noqa: E402
from wildspeech.acoustic.dsp import AcousticParams  # noqa: E402
from wildspeech.acoustic.gpu_features import GPU_FAMILIES, GPUAcousticExtractor  # noqa: E402


def _tone(f0, seconds, sr=16000, lead=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    y = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
    pad = np.zeros(int(lead * sr))
    return (np.concatenate([pad, y, pad]) + 1e-3 * rng.standard_normal(len(y) + 2 * len(pad))).astype(np.float32)


@pytest.fixture(scope="module")
def ext():
    return GPUAcousticExtractor(AcousticParams(), "cpu")


def test_parity_with_numpy(ext):
    waves = [_tone(200, 1.0), _tone(310, 0.4, seed=1), _tone(150, 2.2, seed=2)]
    gpu = ext.parity_stats(waves)
    ref = [numpy_parity_stats(w, ext.p) for w in waves]
    tab = compare_parity(gpu, ref, rtol=2e-3, atol=2e-3)
    assert tab["pass"].all(), tab[~tab["pass"]]


def test_batch_invariance_masking(ext):
    """A clip's features must not depend on what it is padded with in the batch."""
    a, b = _tone(220, 0.8), _tone(180, 3.0, seed=3)
    alone = ext.extract_batch([a], GPU_FAMILIES)
    batched = ext.extract_batch([a, b], GPU_FAMILIES)
    for fam in GPU_FAMILIES:
        for k, v in alone[fam][0].items():
            w = batched[fam][0][k]
            if np.isnan(v):
                assert np.isnan(w), (fam, k)
            else:
                assert abs(v - w) <= 1e-3 + 1e-3 * abs(v), (fam, k, v, w)


def test_vad_and_temporal(ext):
    y = _tone(200, 1.0, lead=0.5)
    r = ext.extract_batch([y], ["temporal", "quality"])
    t, q = r["temporal"][0], r["quality"][0]
    assert 0.35 <= t["onset_latency_s"] <= 0.65
    assert 0.7 <= t["speech_dur_s"] <= 1.3
    assert q["snr_db"] > 20
    silence = np.zeros(16000, np.float32)
    r2 = ext.extract_batch([silence], ["temporal", "quality"])
    assert r2["quality"][0]["no_speech"] in (0.0, 1.0)


def test_pitch_tracks_f0(ext):
    for f0 in (150.0, 250.0):
        p = ext.extract_batch([_tone(f0, 1.0)], ["pitch"])["pitch"][0]
        expected = 12 * np.log2(f0 / 100)
        assert abs(p["f0_median_st"] - expected) < 0.7, (f0, p)
        assert p["voiced_ratio"] > 0.6


def test_bandlimit_detects_lowpass(ext):
    rng = np.random.default_rng(0)
    noise = 0.1 * rng.standard_normal(16000).astype(np.float32)
    from wildspeech.acoustic.perturb import lowpass_fft

    lp = lowpass_fft(noise, 16000, 4000)
    r = ext.extract_batch([noise, lp], ["quality"])["quality"]
    assert r[0]["bandlimit_hz"] > 7000
    assert r[1]["bandlimit_hz"] < 4500


def test_short_clip_does_not_crash(ext):
    r = ext.extract_batch([np.zeros(100, np.float32), _tone(200, 0.2)], GPU_FAMILIES)
    assert len(r["spectral"]) == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu(ext):
    waves = [_tone(200, 1.0), _tone(260, 2.0, seed=4)]
    g = GPUAcousticExtractor(AcousticParams(), "cuda")
    a, b = g.extract_batch(waves, GPU_FAMILIES), ext.extract_batch(waves, GPU_FAMILIES)
    for fam in GPU_FAMILIES:
        tab = compare_parity(a[fam], b[fam], 1e-2, 1e-2)
        assert tab["pass"].all(), tab[~tab["pass"]]
