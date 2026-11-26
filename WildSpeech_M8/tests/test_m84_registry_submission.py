import numpy as np
import pandas as pd
import pytest
from wildspeech_m84.registry import build_registry,identify_feature,registry_summary
from wildspeech_m84.reference import make_reference,reference_frames
from wildspeech_m84.submission import build_submission,predict_independent


def test_provenance_with_known_problematic_names():
    assert identify_feature("m8__audio__n_correct_items")["family"]=="screening_correctness"
    assert identify_feature("is_11khz")["nuisance"]
    assert identify_feature("school_id")["family"]=="PROHIBITED"
    assert identify_feature("blending_theta")["family"]=="screening_correctness"
    train=pd.DataFrame({"blending_theta":[1.,2.,3.],
                        "studentmeta__blending_theta":[1.,2.,3.],
                        "m8__audio__n_correct_items":[2,3,4]})
    test=train.copy()
    registry=build_registry(train,test)
    assert registry.is_duplicate.sum()==1
    assert "blending_theta" in set(registry.canonical_of)
    with pytest.raises(ValueError,match="Forbidden"):
        build_registry(train.assign(school_id=1),test.assign(school_id=1))


def test_singleton_independence_and_strict_schema(tmp_path):
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(48,14)),columns=[f"f{i}" for i in range(14)])
    y=x.f0*4+rng.normal(size=len(x))
    xt=pd.DataFrame(rng.normal(size=(12,14)),columns=x.columns,
                    index=[f"s{i}" for i in range(12)])
    fmt=pd.DataFrame({"outcome_score":np.zeros(len(xt))},index=xt.index)
    model=make_reference().fit(x,y)
    pred=predict_independent(model,xt)
    assert len(pred)==len(xt)
    output=build_submission(x,y,xt,fmt,tmp_path)
    assert output["singleton_batch_test"] and output["permutation_invariance_test"]
    assert pd.read_csv(output["submission_path"]).columns.tolist()==["subject_id","outcome_score"]
    wrong=fmt.iloc[::-1]
    with pytest.raises(ValueError,match="ordering"):
        build_submission(x,y,xt,wrong,tmp_path/"other")
