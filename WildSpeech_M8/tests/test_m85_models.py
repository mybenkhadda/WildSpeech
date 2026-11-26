import numpy as np
import pytest
from wildspeech_m85.nested_cv import FoldCache,predict_fold
from wildspeech_m85.submission import fit_and_export
from wildspeech_m84.data import Dataset
from test_m85_features import screening_fixture


def _as_dataset():
    tr,cl,te,ct=screening_fixture(n=36,ntest=8)
    train=tr.copy()
    train["school_id"]=[f"s{i%9}" for i in range(len(train))]
    y=95.+4*train.blending_theta.fillna(0).to_numpy(float)+np.random.default_rng(10).normal(size=len(train))
    test=te.copy()
    sub=te.iloc[:,0:0].copy()
    sub["outcome_score"]=0.
    return Dataset(root=None,train_students=train,test_students=test,
                   labels=__import__("pandas").Series(y,index=train.index),
                   train_clips=cl,test_clips=ct,submission_format=sub,
                   order_manifest={})


def test_nested_cv_all_arms_on_small_student_holdout():
    d=_as_dataset()
    y=d.labels.to_numpy(float)
    tr=np.arange(24);va=np.arange(24,len(y))
    cache=FoldCache(d.train_students,d.train_clips,y,tr,va,inner_k=3)
    for j in range(1,10):
        model=f"E{j:02d}"
        pred,info=predict_fold(cache,model,seeds=(42,),iterations=12)
        assert len(pred)==len(va) and np.isfinite(pred).all()
        assert "model_family" in info
    with pytest.raises(ValueError):
        FoldCache(d.train_students,d.train_clips,y,tr,np.arange(20,28))


def test_promoted_model_raw_test_inference_and_checksum(tmp_path):
    d=_as_dataset()
    out=fit_and_export(d,"E04",tmp_path,seed_count=1,iterations=12)
    assert out["singleton_raw_pipeline_test"]
    assert out["test_permutation_invariance"]
    assert out["not_submitted_online"]
    assert (tmp_path/"m0plus_final.joblib").is_file()
    assert (tmp_path/"m0plus_submission.sha256").is_file()
    with pytest.raises(ValueError):
        fit_and_export(d,"E09",tmp_path/"bad",seed_count=1,iterations=12)
