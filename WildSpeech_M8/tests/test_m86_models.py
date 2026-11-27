"""Train-only nested selection, monotonic model, penalized pooling."""
from __future__ import annotations
import numpy as np
import pandas as pd
from test_m85_features import screening_fixture
from wildspeech_m86.modeling import MonotoneHingeGAM,PartialPoolingRidge
from wildspeech_m86.nested_cv import FoldWorkspace

def test_monotone_theta_partial_effect_in_all_three_tasks():
    rng=np.random.default_rng(7)
    tasks=["blending","deletion","sentence_repetition"]
    x=pd.DataFrame({t+"_theta":rng.normal(size=35) for t in tasks})
    x["is_wj4"]=(np.arange(35)%2).astype(float)
    x["deletion_n_clips"]=2+np.arange(35)%4
    x["s3_blending_spline_0"]=rng.normal(size=35)
    x["s4_wj4_x_blending_theta"]=x.is_wj4*x.blending_theta
    y=95+2*x.blending_theta+2*x.deletion_theta+2*x.sentence_repetition_theta
    model=MonotoneHingeGAM(alpha=10.,n_knots=3).fit(x,y)
    for task in tasks:
        low=x.iloc[:5].copy()
        high=low.copy()
        high[task+"_theta"]=low[task+"_theta"]+1.
        assert np.all(model.predict(high)>=model.predict(low)-1e-8)

def test_partial_pooling_experts_share_information():
    rng=np.random.default_rng(11)
    x=pd.DataFrame({
        "blending_theta":rng.normal(size=52),
        "deletion_n_clips":rng.integers(0,6,size=52).astype(float),
        "is_wj4":np.arange(52)%2,
        "sentence_repetition_theta":rng.normal(size=52)})
    y=97+5*x.blending_theta+.5*x.is_wj4+rng.normal(size=52)
    for gate in ("instrument","deletion"):
        model=PartialPoolingRidge(alpha=10.,gate=gate).fit(x,y)
        p=model.predict(x.iloc[:7])
        assert np.isfinite(p).all() and len(p)==7

def test_nested_oof_models_run_on_synthetic_student_splits():
    st,clips,_,_=screening_fixture(n=27,ntest=5)
    y=100+3*st.blending_theta.fillna(0).to_numpy(float)
    ctx=FoldWorkspace(st,clips,y,np.arange(20),np.arange(20,27),
                      seed=42,rasch_iters=3,rasch_crossfit=3)
    for mid in ("B1","B2","B3","I1","I2","I3","I4","I5","D2","D3","T1"):
        pred,config=ctx.predict(mid,iterations=8,seeds=(42,))
        assert len(pred)==7 and np.isfinite(pred).all(),mid
        assert config
