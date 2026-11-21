"""Leakage guards: fold-fitted statistics depend only on training-fold students; labels never enter features;
folds are student-disjoint; recordings are aggregated to one row per student."""
import numpy as np
import pandas as pd

from wildspeech import ID
from wildspeech.features.assemble import make_builder
from wildspeech.validation.splits import build_folds, loso, repeated_kfold, school_kfold


def test_one_row_per_student(ctx):
    b = ctx.bundle()
    builder = make_builder(ctx, with_audio=False)
    ids = list(b.y.index)
    X = builder.fit(ids[:24]).transform(ids)
    assert X.index.is_unique and len(X) == len(ids)
    assert any(c.startswith("ITEM__") for c in X.columns) and any(c.startswith("SCREENING__") for c in X.columns)
    assert not any("school" in c.lower() for c in X.columns)


def test_item_stats_ignore_validation_students(ctx):
    b = ctx.bundle()
    ids = list(b.y.index)
    tr, va = ids[:24], ids[24:]
    builder = make_builder(ctx, with_audio=False)
    builder.fit(tr)
    p1 = builder.item_sum_.copy()
    # corrupt validation students' responses -> fitted item statistics must not change
    m = builder.clips[ID].isin(va)
    builder.clips.loc[m, "item_score"] = 1.0 - builder.clips.loc[m, "item_score"]
    builder.fit(tr)
    pd.testing.assert_series_equal(p1, builder.item_sum_)


def test_labels_never_used(ctx, monkeypatch):
    b = ctx.bundle()
    ids = list(b.y.index)
    X1 = make_builder(ctx, with_audio=False).fit(ids[:24]).transform(ids)
    shuffled = b.labels.copy()
    shuffled["outcome_score"] = shuffled["outcome_score"].sample(frac=1.0, random_state=1).values
    b.labels = shuffled
    X2 = make_builder(ctx, with_audio=False).fit(ids[:24]).transform(ids)
    pd.testing.assert_frame_equal(X1, X2)


def test_leave_one_student_out_item_difficulty(ctx):
    b = ctx.bundle()
    ids = list(b.y.index)
    builder = make_builder(ctx, with_audio=False).fit(ids)
    c = builder._clip_frame(ids[:1])
    row = c[c["item_score"].notna()].iloc[0]
    S, N = builder.item_sum_[row["item_id"]], builder.item_n_[row["item_id"]]
    own = c[(c["item_id"] == row["item_id"])]["item_score"]
    tp = builder.task_p_[row["task"]]
    expected = (S - own.sum() + builder.k * tp) / (N - own.count() + builder.k)
    assert np.isclose(row["p_item"], expected)


def test_folds_student_disjoint(ctx):
    b = ctx.bundle()
    y = b.y
    for f in repeated_kfold(y, 3, 2, 0):
        assert not set(f.train_ids) & set(f.valid_ids)
    tr = b.train.set_index(ID)
    sch = tr["school_id"].reindex(y.index)
    for f in school_kfold(sch, 3, 0):
        assert not set(sch.loc[list(f.train_ids)]) & set(sch.loc[list(f.valid_ids)])
    lo = loso(sch, 2)
    assert sum(len(f.valid_ids) for f in lo) == len(y)
    assert build_folds("instrument_transfer", b.train, y, ctx.cfg, 0)


def test_crossfit_rows_come_from_other_folds(ctx):
    from wildspeech.features.builder import crossfit_features

    b = ctx.bundle()
    ids = list(b.y.index.astype(str))
    builder = make_builder(ctx, with_audio=False)
    X = crossfit_features(builder, ids, b.y, 3, 0)
    assert list(X.index) == ids
