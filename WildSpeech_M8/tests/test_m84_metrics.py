import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from wildspeech_m84.folds import build_folds
from wildspeech_m84.metrics import corrected_t,validate_oof,summarize_oof,compare_runs,promotion_gate


def fixture_oof(factor=0.7):
    n=80
    ids=[f"p{i}" for i in range(n)]
    groups=np.asarray([f"g{i%10}" for i in range(n)])
    folds=build_folds(ids,groups)
    y=np.linspace(65,135,n)+np.random.default_rng(0).normal(size=n)
    rows=[]
    for row in folds.itertuples():
        for p in row.valid_pos:
            rows.append({"scheme":row.scheme,"repeat":row.repeat,"fold":row.fold,
                         "pos":p,"subject_id":ids[p],"y_true":y[p],
                         "y_pred":y.mean()+factor*(y[p]-y.mean())})
    return pd.DataFrame(rows),folds,ids,y


def test_pooled_r2_and_permutation_storage():
    oof,folds,ids,y=fixture_oof()
    fold,pooled,inst=summarize_oof(oof,folds,ids,y,["wj4" if i%2 else "ktea3" for i in range(len(ids))])
    assert len(fold)==100 and len(pooled)==20
    assert abs(pooled[(pooled.scheme=="student") & (pooled.repeat==0)].r2.iloc[0]-
               r2_score(y,y.mean()+.7*(y-y.mean())))<1e-12
    shuffled=oof.sample(frac=1,random_state=13)
    fold2,pool2,_=summarize_oof(shuffled,folds,ids,y)
    pd.testing.assert_frame_equal(fold,fold2)
    pd.testing.assert_frame_equal(pooled,pool2)
    assert len(inst)==40


def test_paired_and_hash_and_gate():
    a,folds,ids,y=fixture_oof(.7)
    b,_,_,_=fixture_oof(.9)
    cmp=compare_runs(a,b,folds,ids,y,["wj4" if i%2 else "ktea3" for i in range(len(ids))],
                     "FROZEN","FROZEN")
    assert cmp["mean_fold_delta_r2"]>0 and cmp["win_fraction"]==1.
    assert cmp["school_pooled_delta_r2"]>0
    gate,why=promotion_gate(cmp)
    assert gate=="PROMOTE"
    with pytest.raises(ValueError,match="FOLD HASH"):
        compare_runs(a,b,folds,ids,y,["wj4"]*80,"A","B")
    c=b.copy()
    c.loc[c.index[0],"subject_id"]="WRONG"
    with pytest.raises(ValueError):
        compare_runs(a,c,folds,ids,y,["wj4"]*80,"FROZEN","FROZEN")


def test_undefined_subgroup_zero_variance():
    oof,folds,ids,y=fixture_oof()
    foldscores,pooled,inst=summarize_oof(oof,folds,ids,y,["only"]*80,min_instrument_n=100)
    assert inst.r2.isna().all()
    assert corrected_t([0]*50)==0.
    assert np.isinf(corrected_t([1]*50))
    assert corrected_t([.01,.02,.03,.04,.05])>0
