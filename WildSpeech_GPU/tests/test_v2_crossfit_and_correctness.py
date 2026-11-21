"""Student-disjoint clip classification, nested cross-fitting leakage prevention, item encoding, calibration and
soft-correctness features (synthetic clip tables)."""
import numpy as np
import pandas as pd
import pytest

from wildspeech.correctness.calibration import IsotonicCalibrator, PlattCalibrator, classification_metrics, ece, reliability_table
from wildspeech.correctness.classifiers import ItemEncoder
from wildspeech.correctness.crossfit import clip_oof, nested_predictions, student_groups_kfold
from wildspeech.correctness.soft_features import soft_student_features
from wildspeech.v2.validation_registry import inner_student_folds


def _clips(n_students=60, per=12, seed=0, student_effect=0.0, signal=1.5):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_students):
        u = rng.normal() * student_effect
        for k in range(per):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(signal * x + u)))
            rows.append({"filename": f"f{s}_{k}", "subject_id": f"s{s:03d}", "task": ["blending", "deletion"][k % 2],
                         "item_id": f"it{k % 6}", "item_score": float(rng.random() < p), "x": x, "sid_num": s})
    return pd.DataFrame(rows)


class _Recorder:
    """fit/predict stub that records the students it was trained on and predicts with a logistic on x."""
    log: list = []

    def __init__(self, c):
        self.c = c

    def fit(self, rows):
        from sklearn.linear_model import LogisticRegression

        d = self.c.iloc[rows]
        _Recorder.log.append(set(d["subject_id"]))
        self.m = LogisticRegression().fit(d[["x"]], d["item_score"].astype(int))
        self.train_students = set(d["subject_id"])
        return self

    def predict_proba(self, rows):
        d = self.c.iloc[rows]
        assert not (set(d["subject_id"]) & self.train_students), "prediction for a student seen in training"
        return self.m.predict_proba(d[["x"]])[:, 1]


def test_student_groups_kfold_disjoint():
    c = _clips()
    folds = student_groups_kfold(c["subject_id"].to_numpy(), 5, 0)
    seen = set()
    for tr, va in folds:
        assert not (set(c.iloc[tr]["subject_id"]) & set(c.iloc[va]["subject_id"]))
        seen |= set(va)
    assert seen == set(range(len(c)))


def test_clip_oof_is_student_disjoint_and_informative():
    c = _clips()
    rows = np.arange(len(c))
    p = clip_oof(c, rows, lambda: _Recorder(c), 5, 0)
    assert np.isfinite(p).all()
    assert classification_metrics(c["item_score"], p)["auroc"] > 0.7


def test_nested_predictions_never_use_own_student():
    c = _clips()
    train_ids = sorted(c["subject_id"].unique())[:45]
    rows = np.arange(len(c))
    inner = inner_student_folds(train_ids, 4, 1)
    _Recorder.log = []
    out = nested_predictions(c, train_ids, rows, lambda: _Recorder(c), inner, None)
    assert (out.loc[out["subject_id"].isin(train_ids), "source"] == "inner_oof").all()
    assert (out.loc[~out["subject_id"].isin(train_ids), "source"] == "outer").all()
    assert np.isfinite(out["p"]).all()
    outer_model_students = _Recorder.log[-1]
    assert outer_model_students <= set(train_ids)           # outer model never sees validation students


def test_nested_crossfit_prevents_student_memorisation_leakage():
    """Labels depend ONLY on a student effect that a student-id feature could memorise; nested OOF predictions
    for training students must therefore carry no information beyond chance about their own labels."""
    c = _clips(n_students=80, per=10, student_effect=3.0, signal=0.0, seed=3)

    class Memoriser:
        def __init__(self):
            pass

        def fit(self, rows):
            d = c.iloc[rows]
            self.rate = d.groupby("subject_id")["item_score"].mean().to_dict()
            self.base = float(d["item_score"].mean())
            return self

        def predict_proba(self, rows):
            return c.iloc[rows]["subject_id"].map(self.rate).fillna(self.base).to_numpy(float)

    train_ids = sorted(c["subject_id"].unique())
    out = nested_predictions(c, train_ids, np.arange(len(c)), Memoriser, inner_student_folds(train_ids, 4, 0), None)
    from wildspeech.diagnostics.scientific_validation import auc_fast

    assert abs(auc_fast(c["item_score"], out["p"]) - 0.5) < 0.1
    # the same memoriser evaluated WITHOUT nesting (fit and predict on the same students) leaks badly
    m = Memoriser().fit(np.arange(len(c)))
    assert auc_fast(c["item_score"], m.predict_proba(np.arange(len(c)))) > 0.8


def test_item_encoder_leave_one_student_out():
    c = pd.DataFrame({"subject_id": ["a", "a", "b", "c"], "item_id": ["i", "i", "i", "i"], "task": "t",
                      "item_score": [1.0, 1.0, 0.0, 0.0]})
    enc = ItemEncoder(k=0.0).fit(c)
    full = enc.p_item(c, loso=False)
    loso = enc.p_item(c, loso=True)
    assert np.allclose(full, 0.5)
    assert np.isclose(loso[0], 0.0) and np.isclose(loso[2], 2 / 3)        # own responses removed


def test_calibrators_and_metrics():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p_true).astype(float)
    p_bad = p_true ** 3                                                     # badly calibrated
    for cal in (IsotonicCalibrator(), PlattCalibrator()):
        q = cal.fit(p_bad[:2000], y[:2000]).transform(p_bad[2000:])
        assert ece(y[2000:], q) < ece(y[2000:], p_bad[2000:])
    rt = reliability_table(y, p_true, 10)
    assert rt["n"].sum() == 4000 and len(rt) == 10
    m = classification_metrics(y, p_true)
    assert 0.5 < m["auroc"] < 1 and 0 < m["brier"] < 0.25


def test_soft_student_features_definitions():
    c = pd.DataFrame({"subject_id": ["a"] * 4 + ["b"] * 4, "task": ["blending"] * 8,
                      "item_score": [1, 1, 0, 0, 1, 0, 1, 0]})
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.8, 0.3, 0.7, 0.1])
    d = np.array([0.0, 1.0, 0.5, 1.5, 0.0, 1.0, 0.5, 1.5])
    T = soft_student_features(c, p, d, "SOFT", p_meta=np.full(8, 0.5))
    a = T.loc["a"]
    assert np.isclose(a["SOFT__blending__p_correct_mean__value"], 0.65)
    assert np.isclose(a["SOFT__blending__near_correct_frac__value"], 0.5)        # incorrect with p > 0.5
    assert np.isclose(a["SOFT__blending__doubtful_correct_frac__value"], 0.5)    # correct with p < 0.5
    assert np.isclose(a["SOFT__blending__discrepancy__value"], np.mean([0.1, 0.6, 0.6, 0.2]))
    assert "SOFT__blending__audio_lift__value" in T.columns
    assert T.index.name == "subject_id" and set(T.index) == {"a", "b"}


def test_soft_provider_end_to_end_nested(ctx):
    """SoftCorrectnessProvider on the synthetic project: training rows are inner-OOF, others outer."""
    from wildspeech.correctness.classifiers import ClipFeatureBank
    from wildspeech.correctness.soft_features import SoftCorrectnessProvider

    bank = ClipFeatureBank(ctx)
    ids = sorted(ctx.bundle().train_students["subject_id"].astype(str))
    prov = SoftCorrectnessProvider(ctx, bank, "META", "LOGISTIC", prefix="SOFTMETA", inner_splits=3)
    pa = prov.clip_predictions(ids[:24], ids)
    assert set(pa.loc[pa["subject_id"].isin(ids[:24]), "source"]) == {"inner_oof"}
    assert set(pa.loc[~pa["subject_id"].isin(ids[:24]), "source"]) == {"outer"}
    T = prov.fit_table(ids[:24], ids)
    assert len(T) == len(ids) and any(c.startswith("SOFTMETA__") for c in T.columns)
