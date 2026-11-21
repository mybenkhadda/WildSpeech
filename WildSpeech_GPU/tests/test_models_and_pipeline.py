import numpy as np
import pandas as pd
import pytest

from wildspeech.models.fusion import LateFusionRegressor, ResidualStackRegressor
from wildspeech.models.preprocessing import FoldModel, SupervisedSelector, TabularPreprocessor
from wildspeech.models.zoo import make_model
from wildspeech.registry import Registry
from wildspeech.submission import validate_submission


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"SCREENING__a": rng.normal(size=n), "SCREENING__b": rng.normal(size=n),
                      "TEMPORAL__blending__x__mean": rng.normal(size=n), "SPECTRAL__deletion__y__mean": rng.normal(size=n),
                      "MFCC__all__z__mean": rng.normal(size=n)})
    y = 3 * X["SCREENING__a"] + X["TEMPORAL__blending__x__mean"] + 0.3 * rng.normal(size=n)
    X.iloc[::7, 2] = np.nan
    return X, y.values


def test_preprocessor_fits_on_train_only():
    X, y = _toy()
    tr, te = X.iloc[:80], X.iloc[80:] * 100
    p = TabularPreprocessor().fit(tr)
    out = p.transform(te)
    assert np.allclose(p.mean_, tr[p.cols_].fillna(tr[p.cols_].median()).mean())
    assert out.shape[0] == 40 and not out.isna().any().any()


def test_selector_keeps_screening():
    X, y = _toy()
    s = SupervisedSelector(2).fit(X.fillna(0), y)
    assert {"SCREENING__a", "SCREENING__b"} <= set(s.cols_)


@pytest.mark.parametrize("name", ["RIDGE", "ELASTIC_NET", "PLS", "GAM", "MEAN"])
def test_zoo_models(ctx, name):
    X, y = _toy()
    fm = FoldModel(make_model(name, ctx.cfg), None).fit(X.iloc[:90], y[:90])
    p = fm.predict(X.iloc[90:])
    assert p.shape == (30,) and np.isfinite(p).all()


def test_fusion_estimators():
    X, y = _toy(200)
    Xp = TabularPreprocessor().fit_transform(X)
    lf = LateFusionRegressor().fit(Xp.iloc[:150], y[:150])
    assert (lf.weights_ >= 0).all() and lf.predict(Xp.iloc[150:]).shape == (50,)
    rs = ResidualStackRegressor().fit(Xp.iloc[:150], y[:150])
    assert rs.alpha_ in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert np.corrcoef(rs.predict(Xp.iloc[150:]), y[150:])[0, 1] > 0.9


def test_moe_and_mil_smoke(ctx):
    pytest.importorskip("torch")
    from wildspeech.models.neural import MoERegressor

    ctx.cfg.set_path("multimodal.neural.epochs", 3)
    X, y = _toy(100)
    Xp = TabularPreprocessor().fit_transform(X)
    m = MoERegressor(cfg=ctx.cfg, seed=0).fit(Xp, y)
    assert np.isfinite(m.predict(Xp)).all()


def test_cv_runner_end_to_end(ctx):
    from wildspeech.stages._common import get_runner
    from wildspeech.validation.runner import ModelSpec

    r = get_runner(ctx, with_audio=False)
    res = r.evaluate(ModelSpec("t", "RIDGE", ["SCREENING", "ITEM"]), "repeated_kfold", stage="TEST")
    assert res.status == "ok"
    assert res.oof["subject_id"].is_unique             # one repeat -> one prediction per student
    assert np.isfinite(res.summary["r2_mean"])


def test_registry_upsert(tmp_path):
    r = Registry(tmp_path / "r.parquet", ["feature"])
    r.upsert([{"feature": "a", "x": 1.0}, {"feature": "b", "x": 2.0}])
    r.upsert([{"feature": "a", "y": "hello"}])
    f = r.frame().set_index("feature")
    assert f.loc["a", "x"] == 1.0 and f.loc["a", "y"] == "hello" and len(f) == 2
    assert Registry(tmp_path / "r.parquet", ["feature"]).frame().shape[0] == 2


def test_submission_validation():
    fmt = pd.DataFrame({"subject_id": ["a", "b", "c"], "outcome_score": 0.0})
    good = pd.DataFrame({"subject_id": ["a", "b", "c"], "outcome_score": [90.0, 100.0, 110.0]})
    assert (validate_submission(good, fmt)["status"] == "PASS").all()
    bad = pd.DataFrame({"subject_id": ["b", "a", "c"], "outcome_score": [90.0, np.nan, 110.0]})
    v = validate_submission(bad, fmt).set_index("check")["status"]
    assert v["same order"] == "FAIL" and v["finite target"] == "FAIL"


def test_orchestrator_plan(ctx):
    from wildspeech.orchestrator import STAGES, Pipeline

    p = Pipeline(ctx)
    plan = p.resolve(["P13"])
    assert plan[:2] == ["P00", "P01"] and plan[-1] == "P13" and "P10" in plan
    assert not p.enabled("P14") and not p.enabled("P03")
    res = p.run(["P07"], dry_run=True)
    assert set(res["status"]) <= {"would_run", "blocked", "cached", "disabled"}
    assert len(STAGES) == 23            # P00-P15 + the V2 research line P16-P22
