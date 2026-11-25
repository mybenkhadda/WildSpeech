"""Deterministic, synthetic tests. Synthetic outcomes are not competition results."""
import numpy as np
import pandas as pd
import pytest
from wildspeech_m8.m82_views import ViewBuilder,select_views
from wildspeech_m8.m82_consensus import ConsensusConfig,consensus_ensemble
from wildspeech_m8.m82_pair_analysis import attach_pair_diagnostics,stable_graph_nodes
from wildspeech_m8.m82_prediction import ConsensusPrototype


def fake_frame(n=42,seed=123):
    r=np.random.default_rng(seed)
    ability=r.normal(size=n);style=r.normal(size=n);other=r.normal(size=n)
    z=pd.DataFrame({
        'profile__theta':ability,
        'profile__score_mean':40+ability*10+r.normal(size=n),
        'studentmeta__task_behavior':style+.2*r.normal(size=n),
        'm8__task__blending__accuracy':1/(1+np.exp(-style)),
        'm8__audio__pause_duration_mean':1+style*.1+r.normal(size=n)*.2,
        'm8__audio__onset_s_proxy':.5+style*.3+r.normal(size=n)*.2,
        'm8__audio__f0_contour':other+r.normal(size=n)*.2,
        'm8__audio__spectral_flux':other*.5+r.normal(size=n)*.2,
        'm8__phonetic__asr_ipa_distance':other*.2+style*.3+r.normal(size=n)*.2,
        'm8__speech__phoneme_accuracy':1/(1+np.exp(-other)),
        'm8__audio__original_sample_rate':r.integers(3,8,n)*1000,
        'm8__audio__reference_text_length':r.integers(2,30,n),
    },index=pd.Index([f'S{i:03d}' for i in range(n)],name='subject_id'))
    y=100+ability*4+style*2+r.normal(size=n)*5
    schools=np.array([f'SCHOOL_{i%4}' for i in range(n)])
    return z,y,schools


def test_view_separation_and_nuisance():
    frame,_,_=fake_frame()
    names=select_views(frame)
    assert 'nuisance_audit' in names
    assert 'screening' in names
    assert not set(names['nuisance_audit']) & set(names['screening'])
    v=ViewBuilder().fit_transform(frame)
    assert 'nuisance_audit' not in v
    assert 'conditional_speech' in v
    assert all('sample_rate' not in col for k,z in v.items() for col in z.columns)


def test_outcome_blind_consensus_and_no_self_pairs():
    frame,y,school=fake_frame()
    views=ViewBuilder(max_per_view=10).fit_transform(frame)
    cfg=ConsensusConfig(cluster_counts=(2,),algorithms=('kmeans','ward'),resamples=2,
                        row_fraction=.9,min_pair_runs=1,seed=12)
    a=consensus_ensemble(views,cfg)
    b=consensus_ensemble(views,cfg)
    assert len(a.pairs)==42*41//2
    assert (a.pairs.i<a.pairs.j).all()
    pd.testing.assert_frame_equal(a.pairs,b.pairs)
    p=attach_pair_diagnostics(a.pairs,views,frame.index,school,y)
    assert (p.subject_a!=p.subject_b).all()
    assert np.isfinite(p.score_gap).all()
    assert 'category' in p


def test_coverage_masks_missing_values_and_no_bogus_similarity():
    x,y,_=fake_frame()
    x.loc[x.index[:5],['m8__audio__pause_duration_mean','m8__audio__onset_s_proxy']]=np.nan
    views=ViewBuilder(max_per_view=16,min_nonmissing=.5).fit_transform(x)
    t=views['acoustic_timing']
    assert np.isfinite(t.matrix).all()
    assert t.pair_coverage()[0,1]<t.pair_coverage()[10,11]
    assert not t.observed[0].all()


def test_bad_features_rejected():
    df,_,_=fake_frame()
    df['outcome_score']=1.
    with pytest.raises(ValueError,match='leakage'):
        ViewBuilder().fit(df)


def test_frozen_training_centers_inductive_unseen_student():
    frame,y,_=fake_frame()
    model=ConsensusPrototype(seed=5).fit(frame.iloc[:30])
    a=model.transform(frame.iloc[30:])
    b=model.transform(frame.iloc[30:].copy().assign(profile__theta=100.))
    assert a.shape==b.shape
    assert np.isfinite(a).all()
    assert not np.allclose(a,b)


def test_stable_graph_keeps_singletons():
    p=pd.DataFrame({'i':[0,1,2],'j':[1,2,3],
                    'category':['stable_together','uncertain','stable_together']})
    d=stable_graph_nodes(p,5)
    assert d.component_size.tolist()==[2,2,2,2,1]
    assert d.graph_assigned.sum()==4