"""M8.1 tests. No official competition audio/labels or model downloads."""
from pathlib import Path
import sys
# pip install -e . installs src/wildspeech_m8
import numpy as np
import pandas as pd
from wildspeech_m8.m8_research import (bh_adjust,feature_sets,conditional_associations,
                         paired_student_bootstrap,feature_shift_audit,ResearchConfig,experiment_ablation)
from wildspeech_m8.m8_pipeline import Config,fit_predict_bundle
from wildspeech_m8.m8_phonetic import clean_ipa,phone_distance
from wildspeech_m8.m8_audio import aggregate_audio


def test_benjamini_hochberg():
    q=bh_adjust([.001,.02,.03,.8,np.nan]);assert q[0]<=q[1]<=q[2]<=q[3]
    assert np.isnan(q[-1]) and q[0]==.004


def test_ipa_normalization_preserves_non_ascii():
    assert clean_ipa('ˈkæt ɪ')=='kætɪ'
    assert phone_distance('kæt','kæt')==0
    assert phone_distance('kæt','bæt')>0


def test_feature_family_ablation_and_conditional(tmp_path):
    rng=np.random.default_rng(2);n=70
    theta=rng.normal(size=n);z=rng.normal(size=n)
    df=pd.DataFrame({'blending_theta':theta,'studentmeta__instrument':['A','B']*(n//2),
                     'm8__audio__quality_coverage':rng.uniform(.3,.9,n),
                     'm8__audio__pause_count_proxy__mean':z,
                     'm8__phonetic__ref_distance__mean':rng.normal(size=n),
                     'profile__mean_duration':rng.uniform(2,8,n)},index=[f'p{i}' for i in range(n)])
    sets=feature_sets(df)
    assert 'base_plus_acoustic' in sets and 'base_plus_speech_text' in sets
    y=110+5*theta+3*z+rng.normal(size=n)
    table,resid=conditional_associations(df,y,tmp_path,ResearchConfig(folds=3,quick=True))
    assert len(table)>=2 and np.isfinite(table.conditional_rho).all()
    assert len(resid)==n and np.isfinite(resid.screening_oof_residual).all()


def test_bootstrap_school_cluster():
    r=np.random.default_rng(3);n=36
    tab=[]
    for repeat in range(2):
        for i in range(n):
            y=100+i*.4;school=f's{i%6}'
            for model,pred in [('screening_context',y+r.normal(scale=3)),('all_features',y+r.normal(scale=2))]:
                tab.append({'subject_id':f'x{i}','school':school,'repeat':repeat,'experiment':model,'y':y,'pred':pred})
    t=pd.DataFrame(tab)
    for clustered in [False,True]:
        p=paired_student_bootstrap(t,nboot=40,cluster_by_school=clustered)
        assert len(p)==1 and np.isfinite(p.delta_r2).all() and p.n_schools.iloc[0]==6


def test_audio_aggregation_quality():
    d=pd.DataFrame({'filename':['x1','x2','y1'],'subject_id':['x','x','y'],
                    'task':['blending','blending','deletion'],'item_score':[1,0,1],
                    'expected_text':['cat','dog','good morning'],
                    'audio__valid':[1,1,1],'audio__quality_usable':[1,0,1],
                    'audio__speech_fraction_proxy':[.5,.1,.7],
                    'audio__active_speech_s_proxy':[1,2,4],
                    'audio__onset_s_proxy':[.2,1,.5],
                    'audio__pause_count_proxy':[2,3,1]})
    out=aggregate_audio(d)
    assert len(out)==2 and 'm8__audio__reliable_n' in out
    assert 'm8__audio__n_correct_items' in out and 'm8__audio__speech_time_per_expected_word_mean' in out


def test_cached_fitted_candidates():
    r=np.random.default_rng(44);n=50
    f=pd.DataFrame({'blending_theta':r.normal(size=n),
                    'deletion_theta':r.normal(size=n),'task':['a','b']*(n//2)})
    y=5*f.blending_theta.to_numpy()+r.normal(size=n)
    cfg=Config(inner_folds=2,quick=True)
    specs=[('global',None),('metric_a',('metric',2,.5)),('metric_b',('metric',2,.5)),('rules',('rules',0,.25))]
    out=fit_predict_bundle(f.iloc[:40],f.iloc[40:],y[:40],cfg,44,specs)
    assert np.array_equal(out['metric_a'][0],out['metric_b'][0])
    assert len(out['rules'][0])==10 and np.isfinite(out['global'][0]).all()


def test_train_test_shift(tmp_path):
    r=np.random.default_rng(7)
    a=pd.DataFrame({'m8__audio__value':r.normal(size=80),'other':r.normal(size=80)})
    b=pd.DataFrame({'m8__audio__value':r.normal(loc=2,size=40),'other':r.normal(size=40)})
    t=feature_shift_audit(a,b,tmp_path)
    assert t.iloc[0].feature=='m8__audio__value'
    assert (tmp_path/'feature_distribution_shift.csv').is_file()


def test_audio_negative_control_experiment(tmp_path):
    r=np.random.default_rng(12);n=39
    df=pd.DataFrame({'blending_theta':r.normal(size=n),
        'studentmeta__instrument':['KTEA','WJ','KTEA']*13,
        'm8__audio__pause_count_proxy__mean':r.normal(size=n),
        'm8__phonetic__ref_distance__mean':r.normal(size=n)},index=[f'z{i}' for i in range(n)])
    y=100+df.blending_theta.to_numpy()*4+r.normal(size=n)
    _,_,s=experiment_ablation(df,y,np.array(['a','b','c']*13),tmp_path,
        ResearchConfig(quick=True,folds=3,repeats=1,permutations=1))
    exp=set(s.experiment)
    assert 'control_shuffle_audio_0' in exp and 'control_shuffle_speech_text_0' in exp