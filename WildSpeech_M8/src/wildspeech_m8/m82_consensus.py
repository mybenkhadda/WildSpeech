"""M8.2 multiple-view co-association with honest missing-pair denominators.

No outcomes are read. Labels from different partitions are compared only via
pairwise co-membership. Chance correction uses each run's actual cluster sizes;
this is a descriptive null, not an inferential p-value.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from .m82_views import View

@dataclass
class ConsensusConfig:
    cluster_counts:tuple[int,...]=(2,3,4,6,8)
    algorithms:tuple[str,...]=('kmeans','gmm','ward')
    resamples:int=8
    row_fraction:float=.82
    column_fraction:float=.8
    min_pair_coverage:float=.38
    min_observed_student:float=.45
    seed:int=2026
    min_pair_runs:int=8
    minimum_views:int=2
    stable_together:float=.9
    stable_apart:float=.08
    min_excess_together:float=.22
    min_excess_apart:float=.12

@dataclass
class ConsensusResult:
    pairs:pd.DataFrame
    matrices:dict[str,dict[str,np.ndarray]]
    runs:pd.DataFrame
    summary:pd.DataFrame


def _fit_partition(X,k,algorithm,seed):
    if algorithm=='kmeans':
        return KMeans(n_clusters=k,n_init=5,random_state=seed).fit_predict(X)
    if algorithm=='gmm':
        return GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-3,
                               n_init=1,max_iter=100,random_state=seed).fit_predict(X)
    if algorithm=='ward':
        return AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(X)
    raise ValueError('Unknown clustering algorithm '+str(algorithm))


def _empty(n):
    return {'co':np.zeros((n,n),np.float32),
            'support':np.zeros((n,n),np.float32),
            'expected':np.zeros((n,n),np.float32)}


def _accumulate(m,indices,labels,pair_ok):
    """Count only pairs that were sampled and have jointly observed features."""
    n=len(indices)
    assert pair_ok.shape==(n,n)
    pair_ok=pair_ok.astype(np.float32)
    eq=(labels[:,None]==labels[None,:]).astype(np.float32)
    counts=np.bincount(labels-labels.min())
    chance=float(np.dot(counts,counts-1)/max(1,n*(n-1)))
    ix=np.ix_(indices,indices)
    m['support'][ix]+=pair_ok
    m['co'][ix]+=eq*pair_ok
    m['expected'][ix]+=chance*pair_ok
    return chance


def consensus_ensemble(views:dict[str,View],cfg:ConsensusConfig)->ConsensusResult:
    if not views:raise ValueError('No usable independent feature views')
    n=next(iter(views.values())).matrix.shape[0]
    if any(v.matrix.shape[0]!=n for v in views.values()):raise ValueError('View rows mismatch')
    if n<8:raise ValueError('At least 8 students required')
    mats={name:_empty(n) for name in views}
    logs=[]
    for vi,(name,v) in enumerate(sorted(views.items())):
        if v.matrix.shape[1]<2:continue
        observed_frac=v.observed.mean(axis=1)
        eligible=np.flatnonzero(observed_frac>=cfg.min_observed_student)
        if len(eligible)<8:continue
        for repeat in range(cfg.resamples):
            for k in cfg.cluster_counts:
                if k<2 or k>len(eligible)//4:continue
                for algorithm in cfg.algorithms:
                    # Same seed is reproducible across platforms, but we vary
                    # row/column subsets for independent perturbation tests.
                    seed=cfg.seed+vi*100_003+repeat*4_201+k*101+cfg.algorithms.index(algorithm)*13
                    rng=np.random.default_rng(seed)
                    take=min(len(eligible),max(k*4,int(len(eligible)*cfg.row_fraction)))
                    ids=np.sort(rng.choice(eligible,size=take,replace=False))
                    n_features=v.matrix.shape[1]
                    feat_count=max(2,int(np.ceil(n_features*cfg.column_fraction)))
                    cols=np.sort(rng.choice(n_features,size=min(feat_count,n_features),replace=False))
                    x=v.matrix[ids][:,cols]
                    pair_fraction=(v.observed[ids][:,cols].astype(np.float32) @
                                   v.observed[ids][:,cols].astype(np.float32).T)/len(cols)
                    valid=pair_fraction>=cfg.min_pair_coverage
                    np.fill_diagonal(valid,False)
                    if valid.sum()<10:continue
                    try:
                        labels=_fit_partition(x,k,algorithm,seed)
                    except (ValueError,RuntimeError) as exc:
                        logs.append({'view':name,'algorithm':algorithm,'k':k,'repeat':repeat,
                                     'status':'failed','error':str(exc)[:150]})
                        continue
                    realized=len(np.unique(labels))
                    if realized<2:continue
                    chance=_accumulate(mats[name],ids,labels,valid)
                    logs.append({'view':name,'algorithm':algorithm,'k':k,'repeat':repeat,
                                 'status':'complete','n_rows':len(ids),'n_features':len(cols),
                                 'realized_k':realized,'chance_coassignment':chance,
                                 'n_valid_pairs':int(np.count_nonzero(np.triu(valid,1)))})
    tri=np.triu_indices(n,1)
    pair={'i':tri[0],'j':tri[1]}
    active=[]
    for name,m in mats.items():
        support=m['support'];good=support>0
        c=np.full((n,n),np.nan,np.float32)
        e=np.full((n,n),np.nan,np.float32)
        c[good]=m['co'][good]/support[good]
        e[good]=m['expected'][good]/support[good]
        m['consensus']=c;m['null_expected']=e
        np.fill_diagonal(c,1.)
        pair[f'co__{name}']=c[tri]
        pair[f'support__{name}']=support[tri].astype(np.int32)
        pair[f'null__{name}']=e[tri]
        if (support[tri]>=cfg.min_pair_runs).any():active.append(name)
    if not active:raise RuntimeError('No supported pair consensus; lower min_pair_runs or increase resamples')
    frame=pd.DataFrame(pair)
    # First average correlated views within one feature DOMAIN, then average
    # domains. Conditional speech uses the same acoustic inputs and does not
    # earn an extra independent vote. This avoids feature-view duplication.
    domains={'screening':'screening','task_behavior':'task_behavior',
             'acoustic_timing':'acoustic','acoustic_signal':'acoustic',
             'phonetic_speech':'phonetic','conditional_speech':'acoustic'}
    vals=np.array([frame[f'co__{x}'].where(frame[f'support__{x}']>=cfg.min_pair_runs).to_numpy(float)
                   for x in active])
    base=np.array([frame[f'null__{x}'].where(frame[f'support__{x}']>=cfg.min_pair_runs).to_numpy(float)
                   for x in active])
    frame['n_views']=np.sum(np.isfinite(vals),axis=0)
    domain_values=[];domain_null=[]
    for domain in sorted(set(domains.get(x,x) for x in active)):
        sel=[i for i,x in enumerate(active) if domains.get(x,x)==domain]
        a=vals[sel];b=base[sel]
        na=np.sum(np.isfinite(a),axis=0);nb=np.sum(np.isfinite(b),axis=0)
        domain_values.append(np.divide(np.nansum(a,axis=0),na,
                                       out=np.full(len(frame),np.nan),where=na>0))
        domain_null.append(np.divide(np.nansum(b,axis=0),nb,
                                     out=np.full(len(frame),np.nan),where=nb>0))
    dv=np.array(domain_values);de=np.array(domain_null)
    frame['n_domains']=np.sum(np.isfinite(dv),axis=0)
    total=frame.n_domains.to_numpy()
    frame['consensus']=np.divide(np.nansum(dv,axis=0),total,
                                 out=np.full(len(frame),np.nan),where=total>0)
    frame['chance_expected']=np.divide(np.nansum(de,axis=0),total,
                                       out=np.full(len(frame),np.nan),where=total>0)
    frame['chance_excess']=frame.consensus-frame.chance_expected
    frame['view_disagreement']=np.nanmax(np.where(np.isfinite(vals),vals,-np.inf),axis=0) - np.nanmin(np.where(np.isfinite(vals),vals,np.inf),axis=0)
    frame.loc[frame.n_views<2,'view_disagreement']=np.nan
    high=np.all(np.isnan(vals)|(vals>=.75),axis=0)
    low=np.all(np.isnan(vals)|(vals<=.20),axis=0)
    valid=frame.n_domains>=min(cfg.minimum_views,len(domain_values))
    stable_pos=valid & high & (frame.consensus>=cfg.stable_together) & (frame.chance_excess>=cfg.min_excess_together)
    stable_neg=valid & low & (frame.consensus<=cfg.stable_apart) & ((-frame.chance_excess)>=cfg.min_excess_apart)
    cross=valid & ~stable_pos & ~stable_neg & (frame.view_disagreement>=.55) & (np.nanmin(vals,axis=0)<=.25) & (np.nanmax(vals,axis=0)>=.75)
    frame['category']=np.select([stable_pos,stable_neg,cross],['stable_together','stable_apart','cross_view_disagreement'],default='uncertain')
    frame.loc[~valid,'category']='insufficient_coverage'
    desc=[]
    for name,m in mats.items():
        pp=m['support'][tri]
        selected=pp>=cfg.min_pair_runs
        desc.append({'view':name,'n_features':views[name].matrix.shape[1],
                     'n_runs_complete':sum(x['view']==name and x['status']=='complete' for x in logs),
                     'pairs_supported':int(selected.sum()),
                     'supported_fraction':float(selected.mean()),
                     'mean_consensus':float(np.nanmean(m['consensus'][tri][selected])) if selected.any() else np.nan,
                     'mean_null_expected':float(np.nanmean(m['null_expected'][tri][selected])) if selected.any() else np.nan})
    return ConsensusResult(frame,mats,pd.DataFrame(logs),pd.DataFrame(desc))