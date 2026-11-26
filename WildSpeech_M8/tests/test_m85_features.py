"""M8.5 raw per-student feature invariants and LOO item scoring tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from wildspeech_m85.contracts import normalize_prompt,item_contract
from wildspeech_m85.s2_items import S2ItemEncoder
from wildspeech_m85.features import ScreeningFeatures
from wildspeech_m85.nested_cv import subset_columns,FULL_BLOCKS


def screening_fixture(n=30,ntest=8):
    rng=np.random.default_rng(701)
    def meta(ids,test=False):
        m=pd.DataFrame({
            "subject_id":ids,
            "instrument":["wj4" if i%2 else "ktea3" for i in range(len(ids))],
            "original_sample_rate":[11025 if i%3 else 16000 for i in range(len(ids))],
        }).set_index("subject_id")
        for name in ("blending","deletion","sentence_repetition"):
            m[name+"_theta"]=rng.normal(0,1,len(ids))
            m.loc[m.index[::5],name+"_theta"]=np.nan
        return m
    train=meta([f"p{i:03d}" for i in range(n)])
    test=meta([f"v{i:03d}" for i in range(ntest)],True)
    def clips(students):
        out=[]
        for i,sid in enumerate(students.index):
            for task in ("blending","deletion","sentence_repetition"):
                if task=="deletion" and i%4==0:continue
                for j in range(3):
                    out.append({"subject_id":sid,"task":task,
                        "filename":f"{sid}_{task}_{j}.wav",
                        "duration_sec":float(1.+j*.4),
                        "expected_text":"  APPLE  " if j==0 else "cat" if j==1 else "banana",
                        "item_score":float((i+j)%2) if not(j==2 and i%7==0) else np.nan})
        return pd.DataFrame(out)
    return train,clips(train),test,clips(test)


def test_normalization_and_no_assumed_chronology():
    assert normalize_prompt("  HéLLo   WORLD ")=="héllo world"
    train,tc,test,vc=screening_fixture()
    contract=item_contract(tc,vc)
    assert "last_three_errors" in contract["disabled"]
    assert contract["test"]["item_score_present"]


def test_leave_one_student_out_prompt_success_is_invariant_to_own_score():
    train,tc,test,vc=screening_fixture()
    a=S2ItemEncoder(kappa=10.).fit(train,tc)
    sid=train.index[0]
    p0,n0=a.expected_probability("blending","  APPLE  ",sid,exclude_self=True)
    modified=tc.copy()
    mask=((modified.subject_id==sid)&(modified.task=="blending")&
          (modified.expected_text.str.strip()=="APPLE"))
    modified.loc[mask,"item_score"]=1.-modified.loc[mask,"item_score"]
    b=S2ItemEncoder(kappa=10.).fit(train,modified)
    p1,n1=b.expected_probability("blending","apple",sid,exclude_self=True)
    assert p0==p1 and n0==n1
    assert a.expected_probability("blending","  APPLE  ",sid,False)[0] != \
           b.expected_probability("blending","apple",sid,False)[0]


def test_training_and_test_singleton_feature_independence():
    train,tc,test,vc=screening_fixture()
    builder=ScreeningFeatures(FULL_BLOCKS)
    x=builder.fit_transform(train,tc)
    all_test=builder.transform(test,vc)
    assert x.shape[1]>65 and x.columns.tolist()==all_test.columns.tolist()
    assert x.index.tolist()==train.index.tolist()
    assert not any("school" in c or "score_status" in c for c in x.columns)
    for sid in test.index:
        one_meta=test.loc[[sid]]
        one_clips=vc[vc.subject_id==sid]
        one=builder.transform(one_meta,one_clips)
        pd.testing.assert_frame_equal(one,all_test.loc[[sid]],check_dtype=False)
    other=vc.copy()
    other.loc[other.subject_id==test.index[1],"item_score"]=0.
    pd.testing.assert_frame_equal(
        all_test.loc[[test.index[0]]],
        builder.transform(test,other).loc[[test.index[0]]])
    with pytest.raises(ValueError,match="training IDs"):
        builder.transform(train.iloc[:1],tc[tc.subject_id==train.index[0]])


def test_feature_block_ablation_has_exact_contract():
    train,tc,test,vc=screening_fixture()
    x=ScreeningFeatures(FULL_BLOCKS).fit_transform(train,tc)
    assert any(s.startswith("s2_") for s in subset_columns(x,["S2"]).columns)
    assert not any(s.startswith("s3_") for s in subset_columns(x,["S2"]).columns)
    assert any(s.startswith("s3_") for s in subset_columns(x,["S3"]).columns)
    assert any(s.startswith("s4_") for s in subset_columns(x,["S4"]).columns)
    assert any(s.startswith("s5_") for s in subset_columns(x,["S5"]).columns)
    assert not any(s.endswith("_mean_duration") or s=="is_11khz"
                   for s in subset_columns(x,FULL_BLOCKS,True).columns)


def test_unknown_prompt_fallback_and_no_test_fit():
    train,tc,test,vc=screening_fixture()
    builder=ScreeningFeatures(("S2",))
    builder.fit_transform(train,tc)
    before=builder.s2.item_global.copy()
    altered=vc.copy()
    altered.loc[altered.subject_id==test.index[0],"expected_text"]="totally unseen prompt"
    feature=builder.transform(test.iloc[:1],altered[altered.subject_id==test.index[0]])
    assert np.isfinite(feature.to_numpy(dtype=float)).all()
    assert before==builder.s2.item_global


def test_blocked_leakage_inputs():
    train,tc,test,vc=screening_fixture()
    with pytest.raises(ValueError,match="training students"):
        # Concatenate same student metadata to test, invalid duplicate/overlap.
        ScreeningFeatures(("S2",)).fit(train,tc).transform(train.iloc[:1],
            tc[tc.subject_id==train.index[0]])
