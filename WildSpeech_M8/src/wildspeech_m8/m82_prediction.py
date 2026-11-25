"""Strictly nested M8.2 prediction comparator.

Uses feature-only, training-fitted multi-view prototypes; then fits an optional
fixed-strength ridge correction on training-OOF baseline residuals. Validation
outcomes are never touched by representation learning, preprocessing or fitting.
The method is intentionally simple: no claims of train/test graph transduction.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from .m82_views import ViewBuilder
from .m8_pipeline import FoldEncoder,make_estimator

@dataclass
class PredictionConfig:
    folds:int=5
    repeats:int=1
    inner_folds:int=3
    seed:int=2026
    shrinkage:float=.25
    ridge_alpha:float=100.
    quick:bool=False


class ConsensusPrototype:
    """Compact, inductive alternative to using transductive train graph labels."""
    def __init__(self,seed=2026,ks=(2,4),max_per_view=24):
        self.seed=seed;self.ks=ks;self.max_per_view=max_per_view

    def fit(self,frame):
        self.builder=ViewBuilder(max_per_view=self.max_per_view,min_nonmissing=.55).fit(frame)
        v=self.builder.transform(frame)
        self.parts=[]
        for z,(name,view) in enumerate(sorted(v.items())):
            x=view.matrix
            for k in self.ks:
                if k>=len(x)//3:continue
                km=KMeans(n_clusters=k,n_init=5,random_state=self.seed+z*101+k).fit(x)
                ds=km.transform(x)
                scale=max(np.median(np.min(ds,axis=1)),.1)
                self.parts.append((name,km,scale))
        if not self.parts:raise ValueError('No eligible prototype views')
        return self

    def transform(self,frame):
        v=self.builder.transform(frame)
        groups=[]
        for name,km,scale in self.parts:
            dd=km.transform(v[name].matrix)
            sim=np.exp(-np.square(dd/(2*scale)))
            groups.append(sim)
        return np.column_stack(groups).astype(np.float32)


def _baseline_predict(a,b,y,seed,fast=True):
    enc=FoldEncoder().fit(a)
    reg=make_estimator(seed,fast=fast).fit(enc.transform(a),y)
    return np.asarray(reg.predict(enc.transform(b)),float)


def crossfit_residuals(frame,y,n_splits=3,seed=2026,fast=True):
    if len(frame)<n_splits*3:raise ValueError('Not enough samples for cross-fitting')
    r=np.full(len(y),np.nan,dtype=float)
    for fold,(a,b) in enumerate(KFold(n_splits=n_splits,shuffle=True,random_state=seed).split(frame)):
        p=_baseline_predict(frame.iloc[a],frame.iloc[b],y[a],seed+fold,fast=fast)
        r[b]=y[b]-p
    if not np.isfinite(r).all():raise AssertionError('Incomplete OOF residuals')
    return r


def evaluate_predictive(tr:pd.DataFrame,y,schools,cfg:PredictionConfig):
    """Paired outer CV; correction trained only on cross-fitted training residuals."""
    n=len(tr);y=np.asarray(y,float);schools=np.asarray(schools).astype(str)
    if n<12:raise ValueError('At least 12 students required')
    folds=min(cfg.folds,3 if cfg.quick else cfg.folds)
    if n<folds*4:folds=max(2,n//4)
    splits=RepeatedKFold(n_splits=folds,n_repeats=1 if cfg.quick else cfg.repeats,
                         random_state=cfg.seed)
    preds=[];folds_out=[]
    for fold_id,(train_ids,val_ids) in enumerate(splits.split(tr)):
        a=tr.iloc[train_ids];b=tr.iloc[val_ids];train_y=y[train_ids]
        # Exactly identical global predictions are used for both comparison arms.
        base=_baseline_predict(a,b,train_y,cfg.seed+fold_id,fast=True)
        inner=min(cfg.inner_folds,max(2,len(a)//5))
        residuals=crossfit_residuals(a,train_y,inner,cfg.seed+1000+fold_id,fast=True)
        representation=ConsensusPrototype(seed=cfg.seed+fold_id).fit(a)
        xtrain=representation.transform(a);xval=representation.transform(b)
        correction=Ridge(alpha=cfg.ridge_alpha).fit(xtrain,residuals)
        local=base+cfg.shrinkage*correction.predict(xval)
        for name,pr in [('global',base),('consensus_prototype_residual',local)]:
            folds_out.append({'fold':fold_id,'model':name,'n':len(val_ids),
                              'r2':float(r2_score(y[val_ids],pr)),
                              'mae':float(mean_absolute_error(y[val_ids],pr)),
                              'rmse':float(np.sqrt(mean_squared_error(y[val_ids],pr)))})
            preds.extend({'subject_id':str(tr.index[ix]),'school':str(schools[ix]),
                          'fold':fold_id,'model':name,'y':float(y[ix]),'pred':float(p)}
                         for ix,p in zip(val_ids,pr))
    return pd.DataFrame(preds),pd.DataFrame(folds_out)


def paired_bootstrap(pred,n_boot=300,seed=2026,school_cluster=False):
    """Paired uncertainty on repeat-averaged STUDENT predictions (not pair rows)."""
    if pred.empty:return pd.DataFrame()
    g=pred.groupby(['subject_id','school','model'],as_index=False).agg(y=('y','first'),pred=('pred','mean'))
    wide=g.pivot(index=['subject_id','school','y'],columns='model',values='pred').reset_index()
    if not {'global','consensus_prototype_residual'}.issubset(wide):return pd.DataFrame()
    y=wide.y.to_numpy(float);a=wide.global_.to_numpy(float) if 'global_' in wide else wide['global'].to_numpy(float)
    b=wide['consensus_prototype_residual'].to_numpy(float)
    delta=float(r2_score(y,b)-r2_score(y,a))
    rng=np.random.default_rng(seed);samples=[]
    school=wide.school.astype(str).to_numpy()
    clusters=[np.where(school==sc)[0] for sc in np.unique(school)]
    for _ in range(n_boot):
        if school_cluster:
            inds=np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))])
        else:inds=rng.integers(0,len(y),len(y))
        if np.std(y[inds])<1e-10:continue
        samples.append(r2_score(y[inds],b[inds])-r2_score(y[inds],a[inds]))
    return pd.DataFrame([{'delta_r2':delta,'lo95':np.quantile(samples,.025) if samples else np.nan,
                          'hi95':np.quantile(samples,.975) if samples else np.nan,
                          'delta_mae':mean_absolute_error(y,b)-mean_absolute_error(y,a),
                          'resampling':'school_cluster' if school_cluster else 'student',
                          'n_students':len(y),'n_schools':len(clusters)}])