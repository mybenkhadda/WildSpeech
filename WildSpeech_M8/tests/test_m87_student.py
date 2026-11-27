"""Student-safe audio aggregation and grouped clip correctness control."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from test_m85_features import screening_fixture
from wildspeech_m87.clip_models import ClipCorrectness
from wildspeech_m87.student_aggregation import SpeechAggregator,available_families
from wildspeech_m87.fold_models import FoldStudy

def fake_clips(train,clip):
    d=clip.copy()
    rng=np.random.default_rng(7)
    d["uid"]=d.filename
    d["status"]="VALID"
    d["whole_phone_edit_rate"]=rng.uniform(0,1,len(d))
    d["child_phone_edit_rate"]=rng.uniform(0,1,len(d))
    d["child_target_logp_per_phone"]=rng.normal(-2,.4,len(d))
    d["child_target_vs_free_per_frame"]=rng.normal(-.05,.02,len(d))
    d["whole_target_logp_per_phone"]=rng.normal(-2.5,.3,len(d))
    d["whole_target_vs_free_per_frame"]=rng.normal(-.1,.02,len(d))
    d["whole_phone_weighted_edit_rate"]=d.whole_phone_edit_rate*.9
    d["child_phone_weighted_edit_rate"]=d.child_phone_edit_rate*.9
    d["child_posterior_entropy"]=rng.uniform(0.5,1.5,len(d))
    d["child_posterior_margin"]=rng.uniform(.2,1.5,len(d))
    d["child_aligned_phone_confidence"]=rng.uniform(.1,.9,len(d))
    d["child_aligned_phone_margin"]=rng.uniform(.1,.9,len(d))
    d["child_aligned_phone_entropy"]=rng.uniform(.1,1.5,len(d))
    d["child_phone_coverage"]=rng.uniform(.2,1.,len(d))
    d["task_phone_graded_accuracy"]=rng.uniform(.1,1.,len(d))
    d["task_phone_omission_rate"]=rng.uniform(0,.5,len(d))
    d["task_phone_substitution_rate"]=rng.uniform(0,.5,len(d))
    d["task_phone_order_ratio"]=rng.uniform(.2,1.,len(d))
    d["localization_confidence"]=rng.uniform(.4,.9,len(d))
    d["rms"]=rng.uniform(.02,.08,len(d))
    d["noise_floor"]=rng.uniform(.001,.01,len(d))
    d["clipping_frac"]=np.zeros(len(d))
    d["silence_frac"]=rng.uniform(.05,.3,len(d))
    d["duration_s"]=d.duration_sec
    d["original_sr"]=16000
    d["vad_segments"]=2
    d["vad_speech_seconds"]=d.duration_sec*.7
    d["word_coverage"]=rng.uniform(0,1,len(d))
    d["word_lcs_ratio"]=rng.uniform(0,1,len(d))
    for c in ("word_content_coverage","word_function_coverage",
              "word_endings_coverage","word_normalized_edits",
              "word_deletions","word_substitutions"):
        d[c]=rng.uniform(0,1,len(d))
    for layer in (6,7,8,9,10):
        for k in range(24):
            d[f"layer_{layer:02d}_{k:02d}"]=rng.normal(size=len(d))
    d["random_phone_edit_rate"]=rng.uniform(0,1,len(d))
    d["random_posterior_entropy"]=rng.uniform(.2,1.,len(d))
    d["random_target_logp_per_phone"]=rng.normal(-2,.4,len(d))
    return d

def test_clip_correctness_group_heldout_and_unscored():
    tr,clips,te,tclips=screening_fixture(n=27,ntest=6)
    d=fake_clips(tr,clips);v=fake_clips(te,tclips)
    m=ClipCorrectness(n_splits=3)
    dt,dv=m.fit_predict(d,v)
    assert len(dt)==len(d) and np.isfinite(
        dt.loc[dt.item_score.notna(),"clip_pred_correct"]).all()
    assert np.isfinite(dv.clip_pred_correct).all()
    # The held-out clip predictor is fit using TRAIN, never hidden-test outcomes.
    second=v.copy()
    second["item_score"]=1-second.item_score
    assert np.allclose(
        m._enrich(second,dv.clip_pred_correct).clip_pred_correct.to_numpy(),
        dv.clip_pred_correct.to_numpy(),equal_nan=True)

def test_training_normalization_loo_and_student_isolation():
    tr,clips,te,tclips=screening_fixture(n=27,ntest=6)
    train=fake_clips(tr,clips);test=fake_clips(te,tclips)
    agg=SpeechAggregator(("A02","A03"))
    x=agg.fit(train).transform_train(tr,train)
    xv=agg.transform(te,test)
    assert x.shape[0]==len(tr) and xv.shape[0]==len(te)
    assert x.columns.tolist()==xv.columns.tolist()
    for sid in te.index:
        one=agg.transform(te.loc[[sid]],test[test.subject_id==sid])
        pd.testing.assert_frame_equal(one,xv.loc[[sid]])
    with pytest.raises(ValueError,match="Training"):
        agg.transform(tr.iloc[:1],train[train.subject_id==tr.index[0]])

def test_all_reported_arm_ids_and_model_synthetic():
    tr,clips,_,_=screening_fixture(n=36,ntest=6)
    tr=tr.copy()
    tr["school_id"]=[f"s{i%9}" for i in range(len(tr))]
    d=fake_clips(tr,clips)
    y=np.random.default_rng(11).normal(100,8,len(tr))
    ctx=FoldStudy(tr,clips,d,y,np.arange(26),np.arange(26,36),
                  seed=42,rasch_iters=3,inner_k=3)
    for arm in ("A00","A02","A04","A07","A12","A14","A16"):
        pred,meta=ctx.predict(arm)
        assert len(pred)==10 and np.isfinite(pred).all(),arm
        assert meta["model_family"]
    assert available_families(d)["A13"]["eligible"]
