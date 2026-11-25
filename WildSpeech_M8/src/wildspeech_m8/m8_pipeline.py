"""WildSpeech M8: score-aware behavioral and speech discovery.

Reproducible research pipeline. All outcome-supervised operations are fitted inside
training partitions. Never identifies a child or infers unknown test targets.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, re, tarfile, time, warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.special import logit
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=2026
LEAK_PATTERNS=('outcome','target_score','y_true','y_pred','true_band','relative_q3','residual','school','cluster_label','split','subject_id','teacher_id','examiner_id','speaker_id','recording_id','clip_id','filename')
TASKS=('blending','deletion','sentence_repetition')

@dataclass
class Config:
    data_root: str = '/content/drive/MyDrive/WildSpeech'
    profile_csv: str = ''
    output_root: str = '/content/drive/MyDrive/WildSpeech_M8'
    audio: str = 'light'  # off, light; foundation embeddings opt-in separate stage
    audio_limit_per_split: int = 0 # 0 = all; nonzero for smoke only
    max_audio_sec: float = 75.0
    target_sr: int = 16000
    cv_folds: int = 5
    cv_repeats: int = 2
    inner_folds: int = 3
    loso: bool = True
    loso_min_train: int = 30
    seed: int = SEED
    bootstrap_n: int = 400
    quick: bool = False
    max_features: int = 450
    version: str = 'M8.1'


def safe_save(df, p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False)
    return str(p)

def report_json(obj,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,default=lambda o:float(o) if isinstance(o,np.generic) else str(o)))

def discover_profile(cfg):
    paths=[cfg.profile_csv,os.getenv('M71_PROFILE_CSV',''),
         '/content/drive/MyDrive/WildSpeech_M71_PHENOTYPE_BOOSTING/tables/student_profiles_all_covariates.csv',
         '/content/drive/MyDrive/WildSpeech_M7_DEEP_EDA/tables/student_profiles_all_covariates.csv']
    for x in paths:
        if x and Path(x).is_file():return Path(x)
    base=Path('/content/drive/MyDrive')
    if base.is_dir():
        for p in base.glob('*WildSpeech*/tables/student_profiles_all_covariates.csv'):
            return p
    return None

def identify_columns(df):
    return [c for c in df.columns if c!='subject_id' and not any(z in c.lower() for z in LEAK_PATTERNS)]

def assert_safe_features(columns):
    bad=[c for c in columns if any(p in c.lower() for p in LEAK_PATTERNS)]
    if bad:raise ValueError('Potential target leakage or school identity: '+str(bad[:15]))

def read_official(cfg):
    root=Path(cfg.data_root)
    files=['train_student_metadata.csv','test_student_metadata.csv','train_labels.csv','submission_format.csv','train_clip_metadata.csv','test_clip_metadata.csv']
    missing=[f for f in files if not (root/f).is_file()]
    if missing:raise FileNotFoundError(f'Missing official files in {root}: {missing}. Set --data-root.')
    t=pd.read_csv(root/files[0],dtype={'subject_id':str})
    e=pd.read_csv(root/files[1],dtype={'subject_id':str})
    labs=pd.read_csv(root/files[2],dtype={'subject_id':str})
    fmt=pd.read_csv(root/files[3],dtype={'subject_id':str})
    if not {'subject_id','outcome_score'}.issubset(labs.columns):raise ValueError('Label file must include subject_id,outcome_score')
    for d in [t,e,labs,fmt]:
        if d.subject_id.isna().any() or not d.subject_id.is_unique:raise ValueError('Missing or duplicate student identities')
    if set(t.subject_id)!=set(labs.subject_id) or set(e.subject_id)!=set(fmt.subject_id):raise ValueError('Official ID mismatch')
    if set(t.subject_id)&set(e.subject_id):raise ValueError('Train/test identities overlap')
    clips={s:pd.read_csv(root/f'{s}_clip_metadata.csv',dtype={'subject_id':str,'filename':str}) for s in ('train','test')}
    for s,d in clips.items():
        ref=set(t.subject_id if s=='train' else e.subject_id)
        if not set(d.subject_id).issubset(ref):raise ValueError(s+' clip metadata has invalid subject_id')
    return t,e,labs,fmt,clips

def basic_clip_features(clips):
    """Clip-level official metadata: never use target or school as a predictive feature."""
    d=clips.copy();keys=['subject_id']; n=len(d)
    for c in identify_columns(d):
        if c in ['filename','subject_id']:continue
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c]=pd.to_numeric(d[c],errors='coerce')
    out=d.groupby('subject_id',sort=False).size().rename('m8__n_clips').to_frame()
    numeric=[c for c in identify_columns(d) if c not in ['filename'] and pd.api.types.is_numeric_dtype(d[c])]
    for c in numeric[:50]:
        g=d.groupby('subject_id')[c]
        out[f'm8__meta__{c}__mean']=g.mean()
        out[f'm8__meta__{c}__std']=g.std()
    task_col=next((c for c in ['task','task_type','task_name','assessment_task'] if c in d),None)
    if task_col:
        for task,part in d.groupby(task_col):
            tag=re.sub('[^a-z0-9]+','_',str(task).lower())[:35]
            out[f'm8__task__{tag}__n']=part.groupby('subject_id').size().reindex(out.index).fillna(0)
            for c in numeric[:30]:
                out[f'm8__task__{tag}__{c}__mean']=part.groupby('subject_id')[c].mean().reindex(out.index)
    return out.reset_index()

def transform_student_frame(df):
    """Deterministic row-wise transformations only: no corpus-level fitting/leakage."""
    out=df.copy()
    cols=identify_columns(df)
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):continue
        x=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        lo=c.lower()
        if any(t in lo for t in ('duration','latency','pause','time_','n_clips','count','n_scored','speech_sec')):
            if x.dropna().ge(0).all() and x.notna().any():out['m8__log__'+c]=np.log1p(x.clip(lower=0))
        if ('accuracy' in lo or 'fraction' in lo or 'share' in lo or 'ratio' in lo) and x.dropna().between(0,1).all() and x.notna().any():
            out['m8__logit__'+c]=logit(x.clip(0.015,0.985))
        if ('std' in lo or 'variance' in lo) and x.dropna().ge(0).all() and x.notna().any():
            out['m8__sqrt__'+c]=np.sqrt(x.clip(lower=0))
    # Contrasts are constructed only for explicitly matching numeric summary fields.
    for a,b in [('blending_theta','deletion_theta'),('blending_theta','sentence_repetition_theta'),('deletion_theta','sentence_repetition_theta')]:
        if a in out and b in out:
            out[f'm8__contrast__{a}__{b}']=pd.to_numeric(out[a],errors='coerce')-pd.to_numeric(out[b],errors='coerce')
    # Task contrasts align the *same measurement* in the same student's tasks.
    # Comparisons are unit-preserving, deterministic, and do not inspect labels.
    matched={}
    for c in list(out.columns):
        low=c.lower()
        for task in TASKS:
            token='__task_'+task+'__'
            token2='__'+task+'__'
            if token in low:
                matched.setdefault(low.replace(token,'__TASK__'),{})[task]=c
            elif token2 in low and any(p in low for p in ('m8__phonetic','m8__speech')):
                matched.setdefault(low.replace(token2,'__TASK__'),{})[task]=c
    n_contrasts=0
    for identity,group in sorted(matched.items()):
        for ta,tb in [('blending','deletion'),('blending','sentence_repetition'),('deletion','sentence_repetition')]:
            if ta in group and tb in group and n_contrasts<48:
                x=pd.to_numeric(out[group[ta]],errors='coerce')
                y=pd.to_numeric(out[group[tb]],errors='coerce')
                label=re.sub('[^a-z0-9]+','_',identity.lower())[-60:]
                out[f'm8__taskcontrast__{label}__{ta}_minus_{tb}']=x-y
                n_contrasts+=1
    # Targeted acoustic × screening interactions, interpreted only through
    # held-out family ablations and robust conditional comparisons.
    ability=next((z for z in ('profile__score_mean','blending_theta') if z in out),None)
    speech=next((z for z in out if ('m8__audio__reliable_onset' in z or
                 'm8__phonetic__phonetic_ref_distance__mean' in z)),None)
    if ability is not None and speech is not None:
        a=pd.to_numeric(out[ability],errors='coerce');b=pd.to_numeric(out[speech],errors='coerce')
        out['m8__interaction__ability_times_speech']=a*b
    return out.loc[:,~out.columns.duplicated()].copy()

class RobustWinsorizer(BaseEstimator,TransformerMixin):
    """Fold-fitted extreme-value guard; no global train/test quantile leakage."""
    def __init__(self,low=.005,high=.995):self.low=low;self.high=high
    def fit(self,X,y=None):
        x=np.asarray(X,dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.lo_=np.nanquantile(x,self.low,axis=0)
            self.hi_=np.nanquantile(x,self.high,axis=0)
        self.lo_=np.nan_to_num(self.lo_,nan=-np.inf)
        self.hi_=np.nan_to_num(self.hi_,nan=np.inf)
        return self
    def transform(self,X):return np.clip(np.asarray(X,dtype=float),self.lo_,self.hi_)
    def get_feature_names_out(self,input_features=None):
        return np.asarray(input_features if input_features is not None else [],dtype=object)

def preprocess():
    try:
        cat=OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse_output=False)
    except TypeError:
        cat=OneHotEncoder(handle_unknown='ignore',min_frequency=3,sparse=False)
    return ColumnTransformer([
      ('num',Pipeline([('clip',RobustWinsorizer()),('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',RobustScaler())]),make_numeric),
      ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',cat)]),make_categorical)
    ],remainder='drop',sparse_threshold=0)

def make_numeric(df):return list(df.select_dtypes(include='number').columns)
def make_categorical(df):return list(df.select_dtypes(exclude='number').columns)

def make_estimator(seed,fast=False):
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=350 if fast else 650,depth=4,learning_rate=.045,
                                 l2_leaf_reg=7,loss_function='RMSE',verbose=False,
                                 thread_count=4,random_seed=seed,allow_writing_files=False)
    except ImportError:
        return HistGradientBoostingRegressor(max_iter=180 if fast else 300,l2_regularization=10,
                                              max_leaf_nodes=10,learning_rate=.045,random_state=seed)

class FoldEncoder:
    def fit(self,df):
        self.columns=list(df.columns)
        self.pipe=preprocess().fit(df[self.columns])
        return self
    def transform(self,df):
        z=np.asarray(self.pipe.transform(df[self.columns]),dtype=np.float32)
        if not np.isfinite(z).all():raise ValueError('Nonfinite encoded features')
        return np.clip(z,-25,25)


def feature_families(columns):
    fam={}
    for c in columns:
        low=c.lower()
        if 'm8__audio' in low or 'acoustic__' in low or ('m8__taskcontrast' in low and 'audio' in low):
            role='audio'
        elif 'asr' in low or 'phoneme' in low or 'phonetic' in low or 'transcript' in low or 'm8__speech' in low or 'ability_times_speech' in low or ('m8__taskcontrast' in low and 'speech' in low):role='speech_text'
        elif any(x in low for x in ['theta','score','correct','accuracy','n_scored','irt']):role='screening'
        elif any(x in low for x in ['duration','latency','pause','timing','rate','energy']):role='temporal'
        else:role='context_behavior'
        fam[c]=role
    return fam

def choose_features(df,max_features):
    """Predeclared family-balanced selection; no target-guided cherry-picking.

    Prefer core screening summaries but always give available audio, speech and
    contextual evidence a chance to enter the comparison. The feature cap is
    applied without inspecting labels, train/test outcomes or model scores.
    """
    cols=identify_columns(df)
    assert_safe_features(cols)
    if len(cols)<=max_features:return cols
    family=feature_families(cols)
    buckets={k:sorted([c for c in cols if family[c]==k],key=lambda c:
         (0 if any(x in c.lower() for x in ('theta','accuracy','score_mean','contrast','quality_coverage')) else 1,
          0 if 'reliable_' in c.lower() else 1,c))
         for k in ('screening','context_behavior','audio','speech_text','temporal')}
    quotas={'screening':.30,'context_behavior':.19,'audio':.29,'speech_text':.17,'temporal':.05}
    selected=[]
    for fam,part in buckets.items():
        cap=int(max_features*quotas[fam]);selected.extend(part[:cap]);buckets[fam]=part[cap:]
    while len(selected)<max_features and any(buckets.values()):
        for fam in ('screening','speech_text','audio','context_behavior','temporal'):
            if buckets[fam] and len(selected)<max_features:selected.append(buckets[fam].pop(0))
    return selected

def score(y,p):
    return {'r2':float(r2_score(y,p)) if len(y)>1 else np.nan,
            'mae':float(mean_absolute_error(y,p)),
            'rmse':float(np.sqrt(mean_squared_error(y,p)))}

def crossfit_baseline(X,y,cfg,seed):
    """Each training residual is scored by a baseline NOT trained on that row."""
    n=len(y); k=min(cfg.inner_folds,n)
    if k<2:raise ValueError('Not enough training samples for cross-fitting')
    p=np.full(n,np.nan)
    for a,b in KFold(n_splits=k,shuffle=True,random_state=seed).split(X):
        m=make_estimator(seed+int(b[0]),cfg.quick).fit(X[a],y[a]);p[b]=m.predict(X[b])
    if not np.isfinite(p).all():raise ValueError('Invalid OOF baseline residual')
    return p

def residual_embedding(X,res,dim):
    d=max(1,min(dim,X.shape[1],len(X)-2))
    if float(np.std(res))<1e-8:
        return PCA(n_components=d).fit(X), 'pca'
    try:
        m=PLSRegression(n_components=d,scale=False).fit(X,res)
        return m,'pls'
    except Exception:
        return PCA(n_components=d).fit(X),'pca'

def embed_transform(m,X,kind):
    return np.asarray(m.transform(X),dtype=float)

class ResidualDiscovery:
    """Outcome-aware representation -> GMM soft membership -> shrunken local residual model.
    Label supervision enters only via training-fold cross-fitted residuals.
    """
    def __init__(self,k=2,dim=3,lam=.5,mode='metric',seed=SEED):
        self.k=k;self.dim=dim;self.lam=lam;self.mode=mode;self.seed=seed
    def fit(self,X,y,base_oof):
        residual=y-base_oof
        self.projector,self.projector_kind=residual_embedding(X,residual,self.dim)
        Z=embed_transform(self.projector,X,self.projector_kind)
        self.embedding_scale=StandardScaler().fit(Z)
        Z=np.clip(self.embedding_scale.transform(Z),-10,10)
        self.mix=GaussianMixture(n_components=self.k,covariance_type='diag',reg_covar=1e-2,
                                n_init=4,random_state=self.seed).fit(Z)
        q=self.mix.predict_proba(Z)
        # Regularize regression heavily; correct only based on observed OOF errors.
        self.expert=Ridge(alpha=100.0,fit_intercept=True).fit(np.hstack([q,Z]),residual)
        self.train_membership=q
        self.train_embed=Z
        return self
    def transform(self,X):
        Z=np.clip(self.embedding_scale.transform(embed_transform(self.projector,X,self.projector_kind)),-10,10)
        q=self.mix.predict_proba(Z)
        return Z,q
    def predict_correction(self,X):
        z,q=self.transform(X)
        return self.lam*self.expert.predict(np.hstack([q,z]))

class RuleDiscovery:
    """Small interpretable residual tree; no rule searches on validation labels."""
    def __init__(self,lam=.5,seed=SEED):self.lam=lam;self.seed=seed
    def fit(self,X,y,base_oof):
        r=y-base_oof
        self.tree=DecisionTreeRegressor(max_depth=2,min_samples_leaf=max(15,int(len(y)*.12)),
                                        ccp_alpha=.05,random_state=self.seed).fit(X,r)
        self.leaf_ids=self.tree.apply(X)
        return self
    def predict_correction(self,X):return self.lam*self.tree.predict(X)

CANDIDATES=[('global',None),('metric_k2_l025',('metric',2,.25)),('metric_k2_l050',('metric',2,.5)),
            ('metric_k3_l025',('metric',3,.25)),('metric_k3_l050',('metric',3,.5)),
            ('rules_l025',('rules',0,.25)),('rules_l050',('rules',0,.5))]

def fit_predict(train,val,ytr,cfg,seed,candidate,details=False):
    """Fold-safe: preprocessing, supervised embedding, residual correction within train only."""
    enc=FoldEncoder().fit(train)
    Xt=enc.transform(train);Xv=enc.transform(val)
    global_model=make_estimator(seed,cfg.quick).fit(Xt,ytr)
    p=global_model.predict(Xv)
    kind=candidate[0] if candidate else 'global'
    if kind=='global':return (p,None,None) if details else p
    oof=crossfit_baseline(Xt,ytr,cfg,seed+101)
    if kind=='metric':
        _,k,lam=candidate
        m=ResidualDiscovery(k=k,lam=lam,seed=seed).fit(Xt,ytr,oof)
        z,q=m.transform(Xv); group=q.argmax(axis=1)
    else:
        _,_,lam=candidate
        m=RuleDiscovery(lam=lam,seed=seed).fit(Xt,ytr,oof)
        group=m.tree.apply(Xv); q=None
    pred=p+m.predict_correction(Xv)
    return (pred,group,q) if details else pred

def inner_select(tr,y,cfg,seed):
    """Cached inner CV: fit global model and cross-fit residuals ONCE per inner fold.

    Every candidate shares the same fold-specific preprocessing/baseline and OOF
    residuals. This avoids repeated CatBoost fits and is a fairer comparison.
    """
    splits=list(KFold(n_splits=min(cfg.inner_folds,len(y)),shuffle=True,random_state=seed).split(tr))
    cand=CANDIDATES[:3] if cfg.quick else CANDIDATES
    preds={label:np.full(len(y),np.nan) for label,_ in cand}
    for j,(a,b) in enumerate(splits):
        enc=FoldEncoder().fit(tr.iloc[a]);Xt=enc.transform(tr.iloc[a]);Xv=enc.transform(tr.iloc[b])
        base=make_estimator(seed+j*11,cfg.quick).fit(Xt,y[a]);p_base=base.predict(Xv)
        preds['global'][b]=p_base
        if len(cand)>1:
            base_oof=crossfit_baseline(Xt,y[a],cfg,seed+j*11+101)
            corrections={}
            for label,spec in cand:
                if spec is None:continue
                kind=spec[0]
                if kind=='metric':
                    key=('metric',spec[1])
                    if key not in corrections:
                        m=ResidualDiscovery(k=spec[1],lam=1.,seed=seed+j*43).fit(Xt,y[a],base_oof)
                        corrections[key]=m.predict_correction(Xv)
                elif kind=='rules':
                    key=('rules',)
                    if key not in corrections:
                        m=RuleDiscovery(lam=1.,seed=seed+j*43).fit(Xt,y[a],base_oof)
                        corrections[key]=m.predict_correction(Xv)
                else:raise ValueError('Unknown candidate '+label)
                preds[label][b]=p_base+float(spec[2])*corrections[key]
    rows=[{'candidate':label,**score(y,yp)} for label,yp in preds.items()]
    tab=pd.DataFrame(rows).sort_values(['rmse','candidate'])
    best=tab.iloc[0]['candidate']; return best,dict(cand)[best],tab

def fit_predict_bundle(train,val,ytr,cfg,seed,specs):
    """Fit fold preprocessing / baseline / OOF residuals once for all candidates.
    Identical candidate specifications (e.g. a nested-selected duplicate) reuse
    predictions; corrections with a different lam reuse their fitted mechanism.
    """
    enc=FoldEncoder().fit(train);xt=enc.transform(train);xv=enc.transform(val)
    base=make_estimator(seed,cfg.quick).fit(xt,ytr)
    pbase=np.asarray(base.predict(xv),dtype=float)
    requires=any(spec is not None for _,spec in specs)
    oof=crossfit_baseline(xt,ytr,cfg,seed+101) if requires else None
    cache={};outputs={}
    for label,spec in specs:
        if spec is None:
            outputs[label]=(pbase.copy(),None,None);continue
        kind=spec[0]
        if kind=='metric':
            k,lam=int(spec[1]),float(spec[2]);key=('metric',k)
            if key not in cache:
                m=ResidualDiscovery(k=k,lam=1.,seed=seed+k).fit(xt,ytr,oof)
                z,q=m.transform(xv)
                cache[key]=(m.predict_correction(xv),q.argmax(axis=1),q)
            delta,g,q=cache[key]
        elif kind=='rules':
            lam=float(spec[2]);key=('rules',)
            if key not in cache:
                m=RuleDiscovery(lam=1.,seed=seed).fit(xt,ytr,oof)
                cache[key]=(m.predict_correction(xv),m.tree.apply(xv),None)
            delta,g,q=cache[key]
        else:raise ValueError('Unknown candidate kind '+str(kind))
        outputs[label]=(pbase+lam*delta,g,q)
    return outputs

def evaluate_cv(df,y,schools,cfg,out):
    out=Path(out);cvdir=out/'tables';cvdir.mkdir(parents=True,exist_ok=True)
    outer=RepeatedKFold(n_splits=cfg.cv_folds,n_repeats=cfg.cv_repeats,random_state=cfg.seed)
    predrows=[];tuning=[];mem=[]
    all_specs=[('global',None),('metric_k2_l050',('metric',2,.5)),('rules_l050',('rules',0,.5))]
    for fold,(a,b) in enumerate(outer.split(df)):
        repeat=fold//cfg.cv_folds
        print(f'[M8 CV] repeat={repeat} fold={fold%cfg.cv_folds} train={len(a)} val={len(b)}',flush=True)
        best,spec,grid=inner_select(df.iloc[a],y[a],cfg,cfg.seed+fold*7)
        grid['outer_fold']=fold;tuning.append(grid)
        tests=all_specs+[('nested_selected',spec)]
        fitted=fit_predict_bundle(df.iloc[a],df.iloc[b],y[a],cfg,cfg.seed+fold*71,tests)
        for name,s in tests:
            p,g,q=fitted[name]
            for j,idx in enumerate(b):
                predrows.append({'subject_id':str(df.index[idx]),'repeat':repeat,'fold':fold,
                                 'model':name,'selected':best if name=='nested_selected' else name,
                                 'y':float(y[idx]),'pred':float(p[j]),'school':str(schools[idx])})
                if g is not None:
                    mem.append({'subject_id':str(df.index[idx]),'repeat':repeat,'fold':fold,
                                'model':name,'group':int(g[j]),'confidence':float(q[j].max()) if q is not None else np.nan})
        if cfg.quick and fold>=1:break
    oof=pd.DataFrame(predrows);safe_save(oof,cvdir/'student_oof_predictions.csv')
    safe_save(pd.concat(tuning,ignore_index=True),cvdir/'inner_cv_model_selection.csv')
    safe_save(pd.DataFrame(mem),cvdir/'heldout_memberships.csv')
    rows=[]
    for (r,name),g in oof.groupby(['repeat','model']):
        if len(g)!=len(df):continue
        rows.append({'repeat':r,'model':name,**score(g.y,g.pred)})
    scores=pd.DataFrame(rows);safe_save(scores,cvdir/'cv_metrics.csv')
    if scores.empty:print('QUICK mode: incomplete outer CV; no complete-repeat R²')
    return oof,scores

def evaluate_loso(df,y,schools,cfg,out):
    if not cfg.loso or cfg.quick or len(np.unique(schools))<3:return pd.DataFrame()
    o=[]
    for i,(a,b) in enumerate(LeaveOneGroupOut().split(df,y,schools)):
        if len(a)<cfg.loso_min_train:continue
        school=schools[b[0]];print(f'[M8 LOSO] {i+1} school={school} heldout={len(b)}',flush=True)
        specs=[('global',None),('metric_k2_l050',('metric',2,.5)),('rules_l050',('rules',0,.5))]
        fitted=fit_predict_bundle(df.iloc[a],df.iloc[b],y[a],cfg,cfg.seed+i*41,specs)
        for name,spec in specs:
            pred=fitted[name][0]
            for j,idx in enumerate(b):
                o.append({'subject_id':str(df.index[idx]),'school':school,'model':name,'y':float(y[idx]),'pred':float(pred[j])})
    t=pd.DataFrame(o)
    if t.empty:return t
    safe_save(t,Path(out)/'tables'/'loso_predictions.csv')
    summaries=[]
    for (sch,m),g in t.groupby(['school','model']):summaries.append({'school':sch,'model':m,'n':len(g),**score(g.y,g.pred),'bias':float((g.pred-g.y).mean())})
    safe_save(pd.DataFrame(summaries),Path(out)/'tables'/'loso_per_school.csv')
    return t

def bootstrap_paired(oof,n,seed):
    """Paired student bootstrap on repeat-averaged OOF predictions.

    Repeated predictions of the same student are correlated observations and
    therefore must not be resampled as independent rows.
    """
    if oof.empty:return pd.DataFrame()
    agg=oof.groupby(['subject_id','model'],as_index=False).agg(y=('y','first'),pred=('pred','mean'))
    r=agg.pivot(index='subject_id',columns='model',values='pred')
    yy=agg.drop_duplicates('subject_id').set_index('subject_id')['y']
    if not {'global','metric_k2_l050'}.issubset(r.columns):return pd.DataFrame()
    r=r.join(yy).dropna();rng=np.random.default_rng(seed);rows=[]
    for name in [c for c in r if c not in ('global','y')]:
        d=[];v=r[['y','global',name]].to_numpy()
        for _ in range(n):
            sample=v[rng.integers(0,len(v),len(v))]
            if np.std(sample[:,0])<1e-8:continue
            d.append(r2_score(sample[:,0],sample[:,2])-r2_score(sample[:,0],sample[:,1]))
        rows.append({'model':name,'delta_r2':float(r2_score(v[:,0],v[:,2])-r2_score(v[:,0],v[:,1])),
                     'lo95':float(np.quantile(d,.025)),'hi95':float(np.quantile(d,.975)),
                     'resampling':'student_repeat_averaged'})
    return pd.DataFrame(rows)

def subgroup_atlas(df,y,cfg,out):
    """Descriptive training-only grouping, NOT evidence of heldout utility."""
    if len(df)<30:return pd.DataFrame()
    enc=FoldEncoder().fit(df);X=enc.transform(df)
    r=crossfit_baseline(X,y,cfg,cfg.seed+400)
    m=ResidualDiscovery(k=2,lam=.5,seed=cfg.seed).fit(X,y,r)
    z,q=m.transform(X);g=q.argmax(axis=1)
    tab=pd.DataFrame({'subject_id':df.index,'group':g,'membership_confidence':q.max(axis=1),
                      'outcome':y,'baseline_oof':r,'residual_oof':y-r})
    safe_save(tab,Path(out)/'tables'/'descriptive_group_assignments.csv')
    matrix=df.select_dtypes(include='number').copy().replace([np.inf,-np.inf],np.nan)
    mat=matrix.groupby(g).mean(numeric_only=True)
    zmat=(mat-matrix.mean(numeric_only=True))/(matrix.std(numeric_only=True).replace(0,np.nan))
    zmat=zmat.loc[:,zmat.notna().all()]
    if len(zmat.columns):
        ranked=zmat.abs().max().sort_values(ascending=False).head(30).index
        safe_save(zmat[ranked].reset_index(),Path(out)/'tables'/'group_signature_zscores.csv')
        fig,ax=plt.subplots(figsize=(max(12,len(ranked)*.52),7))
        im=ax.imshow(zmat[ranked].to_numpy(),aspect='auto',cmap='coolwarm',vmin=-2,vmax=2)
        ax.set_xticks(range(len(ranked)),ranked,rotation=75,ha='right',fontsize=7);ax.set_yticks(range(len(mat)),[f'Group {i}' for i in mat.index]);fig.colorbar(im,ax=ax,label='Group mean z-score')
        fig.subplots_adjust(bottom=.54,top=.92);fig.savefig(Path(out)/'figures'/'group_feature_signatures.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return tab

def permutation_negative_control(df,y,cfg,out,nperm=15):
    """Train-only supervised projected structure vs target-permuted controls.
    Descriptive negative control; do not interpret training cluster score separation as validation.
    """
    if len(df)<40:return pd.DataFrame()
    enc=FoldEncoder().fit(df);X=enc.transform(df)
    p=crossfit_baseline(X,y,cfg,cfg.seed+850);r=y-p
    rng=np.random.default_rng(cfg.seed)
    rows=[]
    for i in range(nperm+1):
        rr=r if i==0 else rng.permutation(r)
        m=ResidualDiscovery(k=2,seed=cfg.seed+i).fit(X,p+rr,p)
        z,q=m.transform(X)
        sil=silhouette_score(z,q.argmax(axis=1)) if len(np.unique(q.argmax(axis=1)))>1 else np.nan
        rows.append({'control':'observed' if i==0 else 'permuted','iteration':i,'silhouette':sil,
                     'residual_variance_by_group':float(np.var([rr[q.argmax(axis=1)==j].mean() for j in range(2)]))})
    t=pd.DataFrame(rows);safe_save(t,Path(out)/'tables'/'permutation_group_diagnostics.csv');return t

def prepare_dataset(cfg):
    t,e,labs,fmt,clips=read_official(cfg)
    profile=discover_profile(cfg)
    if profile is None:
        warnings.warn('M7 profile not found; using official metadata-derived features. Supply --profile-csv for full M7 baseline and comparison.')
        base=pd.concat([t.assign(split='train'),e.assign(split='test')],ignore_index=True)
    else:
        base=pd.read_csv(profile,dtype={'subject_id':str})
        if not base.subject_id.is_unique:raise ValueError('Repeated subject_id in profile CSV')
        if not (set(t.subject_id)|set(e.subject_id)).issubset(base.subject_id):raise ValueError('M7 profile missing students')
        base=base.drop(columns=[c for c in base if any(p in c.lower() for p in LEAK_PATTERNS) and c!='subject_id'],errors='ignore')
    counts=pd.concat([basic_clip_features(clips[s]) for s in ('train','test')],ignore_index=True)
    if counts.subject_id.duplicated().any():raise ValueError('clip metadata features duplicate student')
    base=base.merge(counts,on='subject_id',how='left',validate='one_to_one')
    # add non-school, non-target student metadata; never school_id as a model feature.
    m=pd.concat([t,e],ignore_index=True)
    keep=['subject_id']+[c for c in identify_columns(m) if c!='subject_id']
    base=base.merge(m[keep].rename(columns={c:'studentmeta__'+c for c in keep if c!='subject_id'}),on='subject_id',how='left',validate='one_to_one')
    speech_file=Path(cfg.output_root)/'tables'/'speech_student_features.csv'
    if speech_file.is_file():
        sp=pd.read_csv(speech_file,dtype={'subject_id':str})
        if sp.subject_id.duplicated().any():raise ValueError('Duplicate speech student features')
        if 'subject_id' not in sp:raise ValueError('Speech feature table requires subject_id')
        base=base.merge(sp,on='subject_id',how='left',validate='one_to_one')
    audio_file=Path(cfg.output_root)/'tables'/'audio_student_features.csv'
    if audio_file.is_file() and not cfg.quick:
        summary_file=Path(cfg.output_root)/'tables'/'audio_extraction_summary.json'
        if summary_file.exists():
            info=json.loads(summary_file.read_text())
            if not info.get('is_complete',False):
                raise ValueError('Partial SMOKE audio feature cache detected. Re-run m8_audio.py with --limit 0 before FULL modeling.')
    if audio_file.is_file():
        af=pd.read_csv(audio_file,dtype={'subject_id':str})
        if af.subject_id.duplicated().any():raise ValueError('Duplicate audio student features')
        base=base.merge(af,on='subject_id',how='left',validate='one_to_one')
    # Optional pretrained phonetic model features, extracted only with explicit opt-in.
    ph_file=Path(cfg.output_root)/'tables'/'phonetic_student_features.csv'
    if ph_file.is_file():
        ph_summary=Path(cfg.output_root)/'tables'/'phonetic_extraction_summary.json'
        if ph_summary.exists() and not cfg.quick and not json.loads(ph_summary.read_text()).get('is_complete',False):
            raise ValueError('Partial SMOKE phonetic cache detected. Re-run m8_phonetic.py without --limit before FULL modeling.')
        ph=pd.read_csv(ph_file,dtype={'subject_id':str})
        if ph.subject_id.duplicated().any():raise ValueError('Duplicate phonetic student IDs')
        base=base.merge(ph,on='subject_id',how='left',validate='one_to_one')
    base=transform_student_frame(base)
    base=base.set_index('subject_id')
    tr=t.subject_id.to_numpy();te=e.subject_id.to_numpy()
    y=labs.set_index('subject_id').reindex(tr)['outcome_score'].to_numpy(float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite outcome_score')
    cols=choose_features(base,cfg.max_features)
    Xtrain=base.reindex(tr)[cols];Xtest=base.reindex(te)[cols]
    if Xtrain.isna().all().any():
        bad=Xtrain.columns[Xtrain.isna().all()].tolist();warnings.warn('Dropping all-missing train columns: '+str(bad[:10]));cols=[c for c in cols if c not in bad]
        Xtrain=Xtrain[cols];Xtest=Xtest[cols]
    if not cols:raise ValueError('No usable features')
    schools=t.set_index('subject_id').reindex(tr)['school_id'].fillna('UNKNOWN').to_numpy() if 'school_id' in t else np.repeat('UNKNOWN',len(tr))
    manifest=pd.DataFrame({'feature':cols,'family':[feature_families(cols)[c] for c in cols], 'train_missing':[Xtrain[c].isna().mean() for c in cols]})
    return Xtrain,Xtest,y,schools,fmt,clips,manifest

def plot_all(df,y,schools,oof,cv,loso,groups,bootstrap,manifest,cfg,out):
    figdir=Path(out)/'figures';figdir.mkdir(parents=True,exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,4));plt.hist(y,bins=22,edgecolor='white');plt.xlabel('End-of-year score');plt.ylabel('Students');plt.title('Target distribution');save('01_target_distribution')
    vc=pd.Series(schools).value_counts();plt.figure(figsize=(9,4));vc.plot.bar();plt.ylabel('Train students');plt.title('School sample size imbalance');save('02_school_coverage')
    f=manifest.groupby('family').agg(n=('feature','size'),missing=('train_missing','mean')).sort_values('n',ascending=False)
    fig,ax=plt.subplots(figsize=(8,4));f['n'].plot.bar(ax=ax);ax.set_ylabel('Feature count');ax.set_title('Feature bank by family');save('03_feature_family_inventory')
    if not oof.empty:
        complete=oof[(oof.repeat==0)&(oof.model=='global')]
        plt.figure(figsize=(6,5));plt.scatter(complete.y,complete.pred,s=12,alpha=.55);a=min(complete.y.min(),complete.pred.min());b=max(complete.y.max(),complete.pred.max());plt.plot([a,b],[a,b],'k--');plt.xlabel('Observed');plt.ylabel('Predicted');plt.title('Held-out global model calibration');save('04_oof_calibration')
        plt.figure(figsize=(7,4));plt.scatter(complete.pred,complete.y-complete.pred,s=13,alpha=.5);plt.axhline(0,color='k',ls='--');plt.xlabel('OOF prediction');plt.ylabel('Observed - predicted');plt.title('Baseline residual structure');save('05_oof_residuals')
    if not cv.empty:
        s=cv.groupby('model',as_index=False)['r2'].mean().sort_values('r2');plt.figure(figsize=(9,4));plt.barh(s.model,s.r2);plt.xlabel('R² (mean of complete repeats)');plt.title('Outer CV model comparison');save('06_model_comparison')
    if not bootstrap.empty:
        b=bootstrap.sort_values('delta_r2');plt.figure(figsize=(8,4));plt.errorbar(b.delta_r2,range(len(b)),xerr=[b.delta_r2-b.lo95,b.hi95-b.delta_r2],fmt='o',capsize=3);plt.yticks(range(len(b)),b.model);plt.axvline(0,color='k',ls='--');plt.xlabel('Paired ΔR² vs global, student bootstrap');save('07_paired_bootstrap')
    if not groups.empty:
        plt.figure(figsize=(7,4));data=[groups[groups.group==g].residual_oof.to_numpy() for g in sorted(groups.group.unique())];plt.boxplot(data,labels=[str(g) for g in sorted(groups.group.unique())]);plt.axhline(0,color='k',ls='--');plt.xlabel('Descriptive group');plt.ylabel('Baseline OOF residual');plt.title('Group association (exploratory, supervised)');save('08_group_residuals')
        plt.figure(figsize=(7,4));plt.hist(groups.membership_confidence,bins=20);plt.xlabel('Soft assignment maximum (not calibrated probability)');plt.title('Assignment certainty diagnostic');save('09_group_assignment')
    if not loso.empty:
        l=loso.groupby(['school','model']).apply(lambda g:pd.Series({'mae':mean_absolute_error(g.y,g.pred)}),include_groups=False).reset_index()
        piv=l.pivot(index='school',columns='model',values='mae');piv.plot.bar(figsize=(11,4));plt.ylabel('MAE');plt.title('Unseen-school error (equal school visibility)');save('10_loso_per_school')
    # Inspect top univariate associations; descriptive only (not predictive ranking).
    rows=[]
    for c in df.select_dtypes(include='number').columns:
        x=df[c].to_numpy(dtype=float);ok=np.isfinite(x)&np.isfinite(y)
        if ok.sum()>=25 and np.std(x[ok])>1e-10:
            rho,p=stats.spearmanr(x[ok],y[ok]);rows.append({'feature':c,'rho':rho,'p_uncorrected':p,'n':ok.sum()})
    assoc=pd.DataFrame(rows)
    if not assoc.empty:
        assoc['absrho']=assoc.rho.abs();assoc=assoc.sort_values('absrho',ascending=False)
        safe_save(assoc,Path(out)/'tables'/'exploratory_univariate_associations.csv')
        a=assoc.head(15).iloc[::-1];plt.figure(figsize=(10,6));plt.barh(a.feature,a.rho);plt.xlabel('Spearman correlation (exploratory, not causal)');plt.title('Strongest univariate outcome associations');save('11_feature_score_associations')
        for i,c in enumerate(assoc.head(8).feature):
            x=pd.to_numeric(df[c],errors='coerce');ok=x.notna()&np.isfinite(x)
            plt.figure(figsize=(6,4));plt.scatter(x[ok],y[ok],s=12,alpha=.45)
            if ok.sum()>=30:
                try:
                    z=pd.qcut(x[ok],q=min(8,x[ok].nunique()),duplicates='drop')
                    means=pd.DataFrame({'x':x[ok].to_numpy(),'y':y[ok]}).groupby(z,observed=True).mean(numeric_only=True)
                    plt.plot(means.x,means.y,color='tab:orange',lw=2)
                except Exception:pass
            plt.xlabel(c);plt.ylabel('Outcome score');plt.title('Exploratory feature–score relationship');save(f'12_feature_response_{i:02d}')

def run(cfg):
    start=time.time();out=Path(cfg.output_root)
    for sub in ('tables','figures','reports','predictions','cache'):(out/sub).mkdir(parents=True,exist_ok=True)
    report_json(asdict(cfg),out/'reports'/'config.json')
    tr,te,y,schools,fmt,clips,manifest=prepare_dataset(cfg)
    print(f'[DATA] train={len(tr)} test={len(te)} features={tr.shape[1]} schools={len(np.unique(schools))}')
    safe_save(manifest,out/'tables'/'feature_manifest.csv')
    safe_save(pd.DataFrame({'subject_id':tr.index,'school':schools,'outcome':y}),out/'tables'/'training_index_audit.csv')
    oof,cv=evaluate_cv(tr,y,schools,cfg,out)
    loso=evaluate_loso(tr,y,schools,cfg,out)
    groups=subgroup_atlas(tr,y,cfg,out) if not cfg.quick else pd.DataFrame()
    neg=permutation_negative_control(tr,y,cfg,out,nperm=10 if cfg.quick else 30) if not cfg.quick else pd.DataFrame()
    bt=bootstrap_paired(oof,cfg.bootstrap_n,cfg.seed) if not cfg.quick else pd.DataFrame()
    if not bt.empty:safe_save(bt,out/'tables'/'paired_bootstrap.csv')
    plot_all(tr,y,schools,oof,cv,loso,groups,bt,manifest,cfg,out)
    # final fit: all training outcomes available; select on nested train-only CV only.
    if not cfg.quick:
        choice,spec,table=inner_select(tr,y,cfg,cfg.seed+1234)
        pred=fit_predict(tr,te,y,cfg,cfg.seed+2000,spec)
        sub=fmt[['subject_id']].copy()
        if not set(sub.subject_id).issubset(te.index):raise ValueError('Submission identities not in test set')
        sub['outcome_score']=pd.Series(pred,index=te.index).reindex(sub.subject_id).to_numpy()
        if len(sub)!=len(fmt) or not np.isfinite(sub.outcome_score).all():raise ValueError('Invalid submission')
        safe_save(sub,out/'predictions'/'m8_submission.csv')
        report_json({'selected':choice,'n_train':len(tr),'n_test':len(te),'features':len(tr.columns),
                     'runtime_minutes':round((time.time()-start)/60,2)},out/'reports'/'run_summary.json')
    print('[DONE] outputs:',out,'elapsed_mins',round((time.time()-start)/60,1))
    return out

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default=os.getenv('WILDSPEECH_DATA_ROOT','/content/drive/MyDrive/WildSpeech'))
    p.add_argument('--profile-csv',default=os.getenv('M71_PROFILE_CSV',''))
    p.add_argument('--output-root',default=os.getenv('M8_OUTPUT_ROOT','/content/drive/MyDrive/WildSpeech_M8'))
    p.add_argument('--quick',action='store_true');p.add_argument('--skip-loso',action='store_true')
    p.add_argument('--cv-repeats',type=int,default=2);p.add_argument('--cv-folds',type=int,default=5)
    p.add_argument('--max-features',type=int,default=450)
    a=p.parse_args()
    cfg=Config(data_root=a.data_root,profile_csv=a.profile_csv,output_root=a.output_root,quick=a.quick,
               loso=not a.skip_loso,cv_repeats=a.cv_repeats,cv_folds=a.cv_folds,max_features=a.max_features)
    run(cfg)
if __name__=='__main__':cli()