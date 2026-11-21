"""Student-disjoint cross-fitting, reference-statistic leakage prevention and feature aggregation (synthetic project)."""
import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE, ID


def _wide(ctx, seed=0):
    rng = np.random.default_rng(seed)
    c = ctx.bundle().all_clips.drop_duplicates(FILE)
    s = c["item_score"].fillna(0.5).to_numpy(float)
    return pd.DataFrame({FILE: c[FILE].to_numpy(),
                         "M_FAKE::ph_near_correct": np.clip(0.5 + 0.4 * s + 0.1 * rng.standard_normal(len(c)), 0, 1),
                         "M_FAKE::cf_gop_mean": -1.0 + 0.8 * s + 0.3 * rng.standard_normal(len(c)),
                         "M_FAKE::ph_per": np.clip(0.5 - 0.4 * s + 0.1 * rng.standard_normal(len(c)), 0, None),
                         "M_FAKE::ph_cluster_acc": rng.random(len(c)),
                         "M_FAKE::tm_s_per_phone": 0.1 + 0.05 * rng.random(len(c))})


def _split(ctx):
    ids = sorted(ctx.bundle().y.index.astype(str))
    return ids[: len(ids) * 2 // 3], ids[len(ids) * 2 // 3:]


def test_correct_response_training_rows_ignore_validation_clips(ctx):
    from wildspeech.discovery.correct_response import CorrectResponseProvider, pick_role_columns

    wide = _wide(ctx)
    roles = pick_role_columns(wide, "fake")
    assert roles["conf"] == "M_FAKE::cf_gop_mean"
    tr, va = _split(ctx)
    T1 = CorrectResponseProvider(ctx, "a", wide, roles, "fake").fit_table(tr, va)
    w2 = wide.copy()
    vfiles = set(ctx.bundle().train_clips.loc[ctx.bundle().train_clips[ID].isin(va), FILE])
    w2.loc[w2[FILE].isin(vfiles), roles["conf"]] += 50.0                      # corrupt validation clips only
    T2 = CorrectResponseProvider(ctx, "a", w2, roles, "fake").fit_table(tr, va)
    pd.testing.assert_frame_equal(T1.loc[tr], T2.loc[tr])                      # reference fitted on training clips only
    assert set(T1.index) == set(tr) | set(va)
    assert any(c.endswith("corr_conf_mean__value") for c in T1.columns)


def test_nested_soft_correctness_is_student_disjoint(ctx):
    from wildspeech.discovery.item_conditioning import v3_clip_frame
    from wildspeech.representations.student_aggregation import NestedSoftCorrectnessProvider

    wide = _wide(ctx)
    cols = ["M_FAKE::ph_near_correct", "M_FAKE::cf_gop_mean"]
    tr, va = _split(ctx)
    p = NestedSoftCorrectnessProvider(ctx, "n", wide, cols, "LOGISTIC", 3)
    T1 = p.fit_table(tr, va)
    col = "ASRN__all__p__mean"
    assert col in T1
    # flipping ONE training student's own labels must not change the probabilities that score that student
    victim = tr[0]
    cf = v3_clip_frame(ctx)
    m = (cf[ID] == victim) & (cf["is_scored"] == 1)
    cf.loc[m, "item_score"] = 1.0 - cf.loc[m, "item_score"]
    try:
        T2 = p.fit_table(tr, va)
    finally:
        cf.loc[m, "item_score"] = 1.0 - cf.loc[m, "item_score"]
    assert np.isclose(T1.loc[victim, col], T2.loc[victim, col], equal_nan=True)
    # validation students are scored by a model that never saw their labels
    vv = va[0]
    mv = (cf[ID] == vv) & (cf["is_scored"] == 1)
    cf.loc[mv, "item_score"] = 1.0 - cf.loc[mv, "item_score"]
    try:
        T3 = p.fit_table(tr, va)
    finally:
        cf.loc[mv, "item_score"] = 1.0 - cf.loc[mv, "item_score"]
    assert np.isclose(T1.loc[vv, col], T3.loc[vv, col], equal_nan=True)


def test_consistency_and_family_provider(ctx):
    from wildspeech.representations.student_aggregation import ConsistencyProvider, phonetic_family_provider

    wide = _wide(ctx)
    tr, va = _split(ctx)
    T = ConsistencyProvider(ctx, "c", wide, ["M_FAKE::cf_gop_mean", "M_FAKE::ph_near_correct"]).fit_table(tr, va)
    assert T.index.name == ID and len(T) == len(tr) + len(va)
    assert all(c.startswith("ASRS__") for c in T.columns)
    fp = phonetic_family_provider(ctx, "ASRP", wide, ["M_FAKE::ph_per", "M_FAKE::ph_near_correct"])
    A = fp.table(tr, sorted(set(tr) | set(va)))
    assert all(c.startswith("ASRP__") for c in A.columns) and A.shape[0] == len(tr) + len(va)
    assert any("__correct_mean" in c for c in A.columns)


@pytest.mark.parametrize("method", ["select", "pca", "pls", "rrr"])
@pytest.mark.parametrize("budget", [2, 5, 50])
def test_compact_reducer_budget_is_respected(method, budget):
    from wildspeech.representations.compact_features import CompactReducer

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((80, 30)), columns=[f"F__all__x{i}__value" for i in range(30)])
    y = X.iloc[:, 0] * 2 + rng.standard_normal(80)
    r = CompactReducer(method, budget, rank_grid=(2, 4), alphas=(1.0, 10.0)).fit(X, y)
    A = r.transform(X, prefix="F")
    expected = min(budget, 30)
    if method == "rrr":
        assert A.shape[1] == min(r.rank_, expected)
    else:
        assert A.shape[1] == expected                  # components follow the budget (V2 capped them at 10)
    assert A.index.equals(X.index)


def test_aggregate_student_task_shapes():
    from wildspeech.features.conditional_aggregation import aggregate_student_task

    c = pd.DataFrame({ID: ["a", "a", "b", "b"], "task": ["blending"] * 4, "item_score": [1, 0, 1, 1]})
    out = aggregate_student_task(c, np.array([[1.0], [3.0], [2.0], [4.0]]), ["x"], "ASRC")
    assert out.loc["a", "ASRC__blending__x__mean"] == 2.0
    assert out.loc["a", "ASRC__blending__x__correct_mean"] == 1.0
    assert out.loc["b", "ASRC__all__x__mean"] == 3.0
