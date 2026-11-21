import numpy as np

from wildspeech.audio_processing import energy_vad, mask_to_segments
from wildspeech.irt import fit_irt, score_students
from wildspeech.stats_utils import bh_fdr, icc1, partial_corr, robust_stats, safe_corr


def test_robust_stats_with_insufficient_data():
    out = robust_stats([], "x", ["mean", "std", "skewness"])
    assert all(np.isnan(v) for v in out.values())
    out = robust_stats([1.0], "x", ["mean", "std", "kurtosis"])
    assert out["x_mean"] == 1.0 and np.isnan(out["x_std"]) and np.isnan(out["x_kurtosis"])


def test_correlations_return_nan_not_errors():
    r, p, n = safe_corr([1, 1, 1, 1] * 5, list(range(20)))
    assert np.isnan(r) and n == 20
    r, p, n = partial_corr([1, 2, 3], [3, 2, 1], [[0], [1], [2]])
    assert np.isnan(r)
    assert np.isnan(icc1([1, 2], ["a", "b"]))


def test_fdr_handles_nan():
    q = bh_fdr([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and q[0] <= q[2]


def test_vad_on_constant_and_silent_signals():
    mask, info = energy_vad(np.full(100, -120.0), 0.01, {"vad_min_dynamic_range_db": 6, "vad_energy_offset_db": 12,
                                                         "vad_max_below_peak_db": 40, "vad_min_speech_s": 0.08, "vad_min_silence_s": 0.15})
    assert not mask.any() and info["vad_status"] == "LOW_DYNAMIC_RANGE"
    assert mask_to_segments(mask, 0.01, 1.0) == []


def test_irt_handles_perfect_items_with_priors():
    import pandas as pd

    rows = [{"student_id": f"s{s}", "item_key": f"i{i}", "correct": 1.0 if i == 0 else float((s + i) % 2)}
            for s in range(20) for i in range(4)]
    df = pd.DataFrame(rows)
    fit = fit_irt(df, model="rasch", min_obs_per_item=1)
    assert np.isfinite(fit.items["b"]).all() and np.isfinite(fit.items["b_se"]).all()
    ab = score_students(df, fit.items)
    assert np.isfinite(ab["theta"]).all()
