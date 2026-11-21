"""V2 signal processing: bandwidth harmonization, amplitude normalisation, robust family, padding/batch invariance,
CPU/GPU numerical parity (GPU checks skip without CUDA). Synthetic signals only."""
import numpy as np
import pytest

from wildspeech.acoustic.dsp import AcousticParams
from wildspeech.harmonization.bandwidth import (BandwidthHarmonizer, Variant, active_level_db, amp_normalize_numpy,
                                                bandlimit_numpy, raised_cosine_gain, variant_params)

SR = 16000


def _tones(freqs, seconds=1.0, sr=SR, amp=0.2):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * sum(np.sin(2 * np.pi * f * t) for f in freqs)).astype(np.float32)


def _band_energy(y, lo, hi, sr=SR):
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return Y[(f >= lo) & (f < hi)].sum()


def test_raised_cosine_gain_shape():
    f = np.linspace(0, 8000, 801)
    g = raised_cosine_gain(f, 4000, None, 200)
    assert np.allclose(g[f <= 3900], 1.0) and np.allclose(g[f >= 4100], 0.0)
    assert np.all(np.diff(g) <= 1e-12)                      # monotone non-increasing low-pass
    gh = raised_cosine_gain(f, None, 300, 100)
    assert np.allclose(gh[f <= 250], 0.0) and np.allclose(gh[f >= 350], 1.0)


@pytest.mark.parametrize("cut", [5000, 4000, 3400])
def test_bandlimit_removes_energy_above_cutoff(cut):
    y = _tones([500, 1500, 6500, 7200])
    z = bandlimit_numpy(y, SR, cut, None, 250)
    # > 99% of the out-of-band energy removed (residual = edge transients of a finite clip); pass band intact
    assert _band_energy(z, cut + 300, 8000) < 1e-2 * _band_energy(y, cut + 300, 8000)
    assert abs(_band_energy(z, 300, 2000) / _band_energy(y, 300, 2000) - 1) < 0.03
    assert len(z) == len(y) and z.dtype == np.float32


def test_narrowband_highpass():
    y = _tones([100, 1000])
    z = bandlimit_numpy(y, SR, 3400, 300, 100)
    assert _band_energy(z, 0, 200) < 1e-2 * _band_energy(y, 0, 200)


def test_amp_normalisation_reaches_target_level():
    y = 0.01 * _tones([440]) / 0.2
    z = amp_normalize_numpy(y, SR, -26.0)
    assert abs(active_level_db(z, SR) - (-26.0)) < 0.5
    loud = amp_normalize_numpy(_tones([440], amp=0.9), SR, -1.0)
    assert np.abs(loud).max() <= 0.99 + 1e-6               # peak cap


def test_harmonizer_cpu_torch_matches_numpy_and_respects_lengths():
    pytest.importorskip("torch")
    waves = [_tones([300, 6000], 0.7), _tones([800, 7000], 1.3)]
    v = Variant("bw4k", 4000)
    out = BandwidthHarmonizer(SR, "cpu", 250).apply(waves, v)
    for w, o in zip(waves, out):
        ref = bandlimit_numpy(w, SR, 4000, None, 250)
        assert len(o) == len(w)
        assert np.max(np.abs(o - ref)) < 5e-4          # same filter; FFT length differs with the batch maximum
    assert BandwidthHarmonizer(SR, "cpu").apply(waves, Variant("orig"))[0] is not None


def test_variant_params():
    p = variant_params(AcousticParams(), Variant("nb", 3400, 300))
    assert p.fmax == 3400 and p.fmin == 300 and p.hf_split_hz == 1700


def _speechlike(seconds, f0, rng):
    t = np.arange(int(SR * seconds)) / SR
    env = np.clip(np.sin(np.pi * t / seconds), 0, None)
    y = 0.2 * env * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    pad = np.zeros(int(0.25 * SR))
    return (np.concatenate([pad, y, pad]) + 0.003 * rng.standard_normal(len(y) + 2 * len(pad))).astype(np.float32)


def test_robust_family_padding_and_batch_invariance():
    pytest.importorskip("torch")
    from wildspeech.harmonization.normalization import RobustFeatureExtractor

    rng = np.random.default_rng(0)
    short, long_ = _speechlike(0.8, 180, rng), _speechlike(2.5, 220, rng)
    ext = RobustFeatureExtractor(variant_params(AcousticParams(), Variant("bw4k", 4000)), "cpu")
    alone = ext.extract_batch([short])[0]
    batched = ext.extract_batch([short, long_])[0]
    assert set(alone) == set(batched) and len(alone) > 20
    for k in alone:
        if np.isfinite(alone[k]):
            assert abs(alone[k] - batched[k]) <= 1e-3 * (1 + abs(alone[k])), k


def test_cmvn_numpy_zero_mean_unit_var():
    from wildspeech.harmonization.normalization import cmvn_numpy

    X = np.random.default_rng(1).normal(3, 2, size=(200, 13))
    Z = cmvn_numpy(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-9)


@pytest.mark.gpu
def test_cpu_gpu_parity_harmonizer_and_robust():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from wildspeech.harmonization.normalization import RobustFeatureExtractor

    rng = np.random.default_rng(2)
    waves = [_speechlike(1.2, 200, rng), _speechlike(2.0, 250, rng)]
    v = Variant("bw4k", 4000)
    g = BandwidthHarmonizer(SR, "cuda").apply(waves, v)
    c = BandwidthHarmonizer(SR, "cpu").apply(waves, v)
    assert max(np.max(np.abs(a - b)) for a, b in zip(g, c)) < 1e-4
    p = variant_params(AcousticParams(), v)
    rg = RobustFeatureExtractor(p, "cuda").extract_batch(g)
    rc = RobustFeatureExtractor(p, "cpu").extract_batch(g)
    for a, b in zip(rg, rc):
        for k in a:
            if np.isfinite(a[k]) and np.isfinite(b[k]):
                assert abs(a[k] - b[k]) <= 3e-3 * (1 + abs(b[k])), k
