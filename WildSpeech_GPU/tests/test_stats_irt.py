import numpy as np
import pandas as pd

from wildspeech.irt import IRTModel
from wildspeech.stats import bh_fdr, cronbach_alpha, delta_r2_bootstrap, icc1, ols_hc3, partial_corr, r2, spearman


def test_bh_fdr_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = bh_fdr(p)
    assert np.isnan(q[-1])
    np.testing.assert_allclose(q[:4], [0.02, 0.04, 0.04, 0.02])


def test_partial_corr_removes_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x, y = z + 0.3 * rng.normal(size=500), z + 0.3 * rng.normal(size=500)
    assert spearman(x, y)[0] > 0.8
    assert abs(partial_corr(x, y, z[:, None])[0]) < 0.15


def test_icc_and_r2():
    g = np.repeat(np.arange(10), 20)
    v = np.random.default_rng(1).normal(size=200) + np.repeat(np.arange(10), 20)
    assert icc1(v, g) > 0.8
    y = np.arange(10.0)
    assert r2(y, y) == 1.0 and r2(y, np.full(10, y.mean())) == 0.0
    est, lo, hi = delta_r2_bootstrap(y, np.full(10, y.mean()), y, n_boot=50)
    assert est == 1.0


def test_ols_hc3_recovers_slope():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = 2 * x + rng.normal(size=400) * (1 + np.abs(x))
    t = ols_hc3(y, x[:, None], ["x"])
    assert abs(t.set_index("term").loc["x", "coef"] - 2) < 0.3


def test_cronbach():
    rng = np.random.default_rng(3)
    f = rng.normal(size=300)
    X = np.column_stack([f + 0.3 * rng.normal(size=300) for _ in range(4)])
    assert cronbach_alpha(X) > 0.8


def test_rasch_recovers_difficulty_order():
    rng = np.random.default_rng(4)
    theta = rng.normal(size=300)
    b = np.array([-1.5, -0.5, 0.0, 0.7, 1.6])
    rows = [{"subject_id": f"s{i}", "item_id": f"i{j}", "item_score": float(rng.random() < 1 / (1 + np.exp(-(theta[i] - b[j]))))}
            for i in range(300) for j in range(5)]
    m = IRTModel("rasch").fit(pd.DataFrame(rows))
    est = m.items_.set_index("item_id").loc[[f"i{j}" for j in range(5)], "b"].to_numpy()
    assert np.all(np.diff(est) > 0)
    sc = m.score_persons(pd.DataFrame(rows))
    assert spearman(sc["theta"], theta[[int(s[1:]) for s in sc["subject_id"]]])[0] > 0.6
