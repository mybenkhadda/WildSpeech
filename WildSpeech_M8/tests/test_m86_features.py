"""M8.6 screening features: no target leakage, no test-fit, LOO/crossfit invariants."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from test_m85_features import screening_fixture
from wildspeech_m86.contracts import prompt_key,validate_inputs,ensure_predictor_columns
from wildspeech_m86.item_matrix import ItemMatrix
from wildspeech_m86.item_difficulty import DifficultyProfile
from wildspeech_m86.rasch_ability import RaschAbility
from wildspeech_m86.duration_controls import DurationControls
from wildspeech_m86.item_complexity import complexity_features,text_complexity
from wildspeech_m86.features import M86Features,select_features
from wildspeech_m86.modeling import SPECS

def test_prompt_identity_and_complexity_no_unlicensed_frequency():
    assert prompt_key("blending","  APPLE ")==prompt_key("blending","apple")
    assert prompt_key("deletion","apple")!=prompt_key("blending","apple")
    assert text_complexity("What a CAT! ")["words"]==3.
    assert text_complexity(None)["missing_text"]==1.

def test_item_matrix_leave_one_student_out_and_unknown_prompt():
    a,clips,test,ct=screening_fixture(n=27,ntest=6)
    m=ItemMatrix(min_students=1).fit(a,clips)
    first=m.transform_train(a,clips)
    outside=m.transform(test,ct)
    assert len(first)==len(a) and first.columns.tolist()==outside.columns.tolist()
    assert np.isfinite(first.to_numpy()).all()
    assert np.isfinite(outside.to_numpy()).all()
    sid=a.index[0]
    changed=clips.copy()
    mask=(changed.subject_id==sid)&(changed.task=="blending")&(changed.expected_text.str.strip()=="APPLE")
    # LOO item prior must not be influenced by any of this student's attempts.
    first_prior=m._prior(prompt_key("blending","APPLE"),sid,True)
    changed.loc[mask,"item_score"]=1.-changed.loc[mask,"item_score"]
    after=ItemMatrix(min_students=1).fit(a,changed)
    assert abs(first_prior-after._prior(prompt_key("blending","APPLE"),sid,True))<1e-12
    with pytest.raises(ValueError,match="training students"):
        m.transform(a.iloc[:1],clips[clips.subject_id==sid])
    unseen=ct.copy();unseen["expected_text"]="this prompt is not in training"
    assert np.isfinite(m.transform(test,unseen).to_numpy()).all()

def test_crossfitted_rasch_and_independent_test_scoring():
    a,clips,test,ct=screening_fixture(n=27,ntest=6)
    model=RaschAbility(outer_iters=3,inner_iters=4,crossfit=3)
    x=model.fit(a,clips).transform_train(a,clips)
    xt=model.transform(test,ct)
    assert x.shape==(27,15) and xt.shape==(6,15)
    assert np.isfinite(x.to_numpy(float)).all()
    for sid in test.index:
        one=model.transform(test.loc[[sid]],ct[ct.subject_id==sid])
        pd.testing.assert_frame_equal(one,xt.loc[[sid]])
    with pytest.raises(ValueError,match="Training students"):
        model.transform(a.iloc[:1],clips[clips.subject_id==a.index[0]])

def test_difficulty_and_duration_training_loo_and_independence():
    train,clips,test,testclips=screening_fixture(n=27,ntest=6)
    for cls in (DifficultyProfile,DurationControls):
        model=cls().fit(train,clips)
        xr=model.transform_train(train,clips)
        xt=model.transform(test,testclips)
        assert np.isfinite(xr.to_numpy(float)).all()
        for sid in test.index:
            one=model.transform(test.loc[[sid]],testclips[testclips.subject_id==sid])
            pd.testing.assert_frame_equal(one,xt.loc[[sid]])
    v=complexity_features(test,testclips)
    assert len(v)==len(test) and np.isfinite(v.to_numpy(float)).all()

def test_joint_screening_features_and_per_student_independence():
    train,clips,test,testclips=screening_fixture(n=27,ntest=6)
    b=M86Features(("I1","I2","I3","I4","T1"),rasch_iters=3)
    x=b.fit_transform(train,clips)
    xt=b.transform(test,testclips)
    assert x.columns.tolist()==xt.columns.tolist()
    assert all(any(c.startswith(s+"_") for c in x.columns)
               for s in ("i1","i2","i3","i4","t1"))
    for sid in test.index:
        one=b.transform(test.loc[[sid]],testclips[testclips.subject_id==sid])
        pd.testing.assert_frame_equal(one,xt.loc[[sid]],check_dtype=False)
    for mid in SPECS:
        view=select_features(x,mid)
        assert view.shape[1]>0
        ensure_predictor_columns(view)
        assert not any("school_id" in c or "score_status" in c for c in view)
    assert not any(c.startswith("s4_") for c in select_features(x,"B3"))
    assert not any(c.startswith("i1_") for c in select_features(x,"B1"))
    with pytest.raises(ValueError,match="Training student"):
        b.transform(train.iloc[:1],clips[clips.subject_id==train.index[0]])
