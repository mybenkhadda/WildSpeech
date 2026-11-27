"""M8.7 fold-safe experimental design and predeclared A00–A16 regressors.

A00/A01 test whole-clip evidence; A02/A03 localized evidence;
A04/A05/A15 nuisance and selection controls; A06 correctness strata;
A07–A09 task scorers; A10–A11 clip correctness and missing labels;
A12 confidence weighting; A13 WavLM layers; A14 residual comparator;
A16 two-head, INNER-OOF trained stacking.

All student-outcome fitting stays inside outer train. Every supervised
clip corrector uses GroupKFold on training students. Each inner fold
re-fits screening/item statistics and audio normalization independently.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from wildspeech_m86.features import M86Features,select_features
from .clip_models import ClipCorrectness
from .student_aggregation import SpeechAggregator,FAMILIES

ALPHAS=(10.,100.,1000.)
ALL_FAMILIES=tuple(FAMILIES)
CONTROL_ARMS=("A04","A05","A15")
PRIMARY=("A02","A03","A07","A08","A09","A16")
BASELINE="I5"
ITEM_BLOCKS=("I1","I2","I3","I4")

def new_ridge(alpha):
    return make_pipeline(
        SimpleImputer(strategy="median",keep_empty_features=True),
        StandardScaler(),Ridge(alpha=float(alpha),solver="lsqr"))

def _student_clips(df,students):
    return df[df.subject_id.astype(str).isin(set(students.index.astype(str)))].copy()

def _select_audio(agg,arm):
    cols=FAMILIES[arm]
    use=[c for c in agg.columns
         if c.startswith("speech_") and
         (any("_"+x+"_" in c for x in cols) or
          c.endswith(("_n_valid","_fail_frac","_n_unscored","_loc_conf_mean")))]
    if arm=="A06":
        use=[c for c in use if c.endswith(("_correct","_incorrect","_observed",
                                          "_n_valid","_fail_frac","_n_unscored"))]
    if arm=="A11":
        use=[c for c in use if c.endswith(("_mean","_observed",
                                          "_n_valid","_n_unscored","_fail_frac"))]
    if not use:raise RuntimeError(f"No acoustic predictors available for {arm}")
    return agg[use].copy()

def _project_layers(train,val,cap=8):
    layer=[c for c in train if "_layer_" in c and c.endswith("_mean")]
    if not layer:return train,val
    n=min(cap,len(train)-1,len(layer))
    if n<1:return train,val
    im=SimpleImputer(strategy="constant",fill_value=0)
    a=im.fit_transform(train[layer]);b=im.transform(val[layer])
    scaler=StandardScaler().fit(a)
    a=scaler.transform(a);b=scaler.transform(b)
    pca=PCA(n_components=n,random_state=42)
    pc=pca.fit_transform(a);pv=pca.transform(b)
    ta=train.drop(columns=layer).copy()
    va=val.drop(columns=layer).copy()
    for i in range(n):
        name=f"speech_layer_pca_{i:02d}"
        ta[name]=pc[:,i];va[name]=pv[:,i]
    return ta,va

@dataclass
class Representation:
    train_ids:list[str]
    val_ids:list[str]
    x_train:pd.DataFrame
    x_val:pd.DataFrame
    audio_train:pd.DataFrame
    audio_val:pd.DataFrame
    weighted_train:pd.DataFrame
    weighted_val:pd.DataFrame

    def design(self,arm):
        if arm not in FAMILIES:raise KeyError(arm)
        if arm=="A12":
            t=_select_audio(self.weighted_train,arm)
            v=_select_audio(self.weighted_val,arm)
        else:
            t=_select_audio(self.audio_train,arm)
            v=_select_audio(self.audio_val,arm)
        if arm=="A13":
            t,v=_project_layers(t,v)
        if arm=="A04":
            # Recording-condition-only control; no screening predictors.
            return t,v
        if arm=="A15":
            # No child-selected features; random segment control.
            pass
        return (pd.concat([self.x_train,t],axis=1),
                pd.concat([self.x_val,v],axis=1))

    def base(self):return self.x_train,self.x_val
    def speech(self,arm):
        if arm=="A12":return (_select_audio(self.weighted_train,arm),
                              _select_audio(self.weighted_val,arm))
        a=_select_audio(self.audio_train,arm)
        b=_select_audio(self.audio_val,arm)
        return _project_layers(a,b) if arm=="A13" else (a,b)

def representation(train_meta,valid_meta,train_clips,valid_clips,
                   rasch_iters=7,correctness=True):
    # M8.6 original I5 exactly (I1–I4) + no new outcome leakage.
    builder=M86Features(ITEM_BLOCKS,rasch_iters=rasch_iters)
    sx=select_features(builder.fit_transform(train_meta,train_clips),"I5")
    sv=select_features(builder.transform(valid_meta,valid_clips),"I5")
    if correctness:
        cc=ClipCorrectness(n_splits=3)
        train_clips,valid_clips=cc.fit_predict(train_clips,valid_clips)
    families=ALL_FAMILIES
    agg=SpeechAggregator(families)
    ax=agg.fit(train_clips).transform_train(train_meta,train_clips)
    av=agg.transform(valid_meta,valid_clips)
    weighted=SpeechAggregator(("A12",),weighted=True)
    wx=weighted.fit(train_clips).transform_train(train_meta,train_clips)
    wv=weighted.transform(valid_meta,valid_clips)
    if sx.columns.tolist()!=sv.columns.tolist() or ax.columns.tolist()!=av.columns.tolist():
        raise RuntimeError("Train/heldout feature schema mismatch")
    return Representation(train_meta.index.astype(str).tolist(),
                          valid_meta.index.astype(str).tolist(),
                          sx,sv,ax,av,wx,wv)

class FoldStudy:
    def __init__(self,meta,metadata_clips,features,y,train_pos,val_pos,
                 seed=42,rasch_iters=7,inner_k=3):
        self.meta=meta;self.y=np.asarray(y,float)
        self.tr=np.asarray(train_pos,int);self.va=np.asarray(val_pos,int)
        if len(set(self.tr)&set(self.va)):raise ValueError("Outer validation leakage")
        self.seed=int(seed)
        self.mt=meta.iloc[self.tr]
        self.mv=meta.iloc[self.va]
        # Feature cache carries item_score for training/test but no outcome_score.
        self.ct=_student_clips(features,self.mt)
        self.cv=_student_clips(features,self.mv)
        self.outer=representation(self.mt,self.mv,self.ct,self.cv,rasch_iters)
        self.inner=[]
        kfold=KFold(n_splits=inner_k,shuffle=True,random_state=seed)
        for it,iv in kfold.split(self.mt):
            a=self.mt.iloc[it];b=self.mt.iloc[iv]
            ca=_student_clips(self.ct,a)
            cb=_student_clips(self.ct,b)
            z=representation(a,b,ca,cb,rasch_iters)
            self.inner.append((np.asarray(it,int),np.asarray(iv,int),z))

    def _choose_alpha(self,arm):
        scores={}
        for alpha in ALPHAS:
            err=0.
            for it,iv,z in self.inner:
                x,v=z.design(arm)
                p=new_ridge(alpha).fit(x,self.y[self.tr[it]]).predict(v)
                err+=float(np.square(self.y[self.tr[iv]]-p).sum())
            scores[alpha]=err
        return min(scores,key=lambda a:(scores[a],a)),scores

    def _direct(self,arm):
        alpha,scores=self._choose_alpha(arm)
        x,v=self.outer.design(arm)
        p=new_ridge(alpha).fit(x,self.y[self.tr]).predict(v)
        return p,{"model_family":"joint_ridge","alpha":alpha,"inner_sse":scores}

    def _residual(self,arm="A14"):
        # Cross-fitted baseline predictions ONLY within this outer train.
        alpha=100.
        base=np.full(len(self.tr),np.nan)
        for it,iv,z in self.inner:
            a,b=z.base()
            base[iv]=new_ridge(alpha).fit(a,self.y[self.tr[it]]).predict(b)
        if not np.isfinite(base).all():raise ValueError("Incomplete inner baseline OOF")
        base_tr,base_va=self.outer.base()
        speech_tr,speech_va=self.outer.speech(arm)
        baseline=new_ridge(alpha).fit(base_tr,self.y[self.tr])
        correction=new_ridge(100.).fit(speech_tr,self.y[self.tr]-base)
        # Fixed shrinkage, selected before outer evaluation.
        p=baseline.predict(base_va)+.25*correction.predict(speech_va)
        return p,{"model_family":"crossfit_residual","shrinkage":.25,
                  "base_alpha":100.,"correction_alpha":100.}

    def _stack(self,arm="A16"):
        # 3-way inner OOF makes BOTH level-one predictors independent of
        # the student's outcome; val data never fits the meta learner.
        pb=np.full(len(self.tr),np.nan)
        ps=np.full(len(self.tr),np.nan)
        for it,iv,z in self.inner:
            xb,vb=z.base()
            xs,vs=z.speech(arm)
            pb[iv]=new_ridge(100.).fit(xb,self.y[self.tr[it]]).predict(vb)
            ps[iv]=new_ridge(100.).fit(xs,self.y[self.tr[it]]).predict(vs)
        if not np.isfinite(pb).all() or not np.isfinite(ps).all():
            raise ValueError("Incomplete train-only stacking OOF")
        # Fixed alpha and no outer-validation tuning.
        meta=new_ridge(100.).fit(
            pd.DataFrame({"screen":pb,"speech":ps}),self.y[self.tr])
        xb,vb=self.outer.base()
        xs,vs=self.outer.speech(arm)
        b=new_ridge(100.).fit(xb,self.y[self.tr]).predict(vb)
        s=new_ridge(100.).fit(xs,self.y[self.tr]).predict(vs)
        val=pd.DataFrame({"screen":b,"speech":s})
        return meta.predict(val),{"model_family":"inner_oof_two_head_stack",
                                  "base_alpha":100.,"speech_alpha":100.,
                                  "meta_alpha":100.}

    def predict(self,arm):
        if arm=="A14":return self._residual()
        if arm=="A16":return self._stack()
        return self._direct(arm)

def audio_contract(features,inventory):
    if features.uid.duplicated().any():raise ValueError("Duplicate acoustic clip UID")
    if set(features.uid)!=set(inventory.uid):
        raise RuntimeError("Audio cache does not account for all expected clips")
    if not {"subject_id","task","expected_text","item_score","status","uid"}.issubset(features):
        raise ValueError("Missing required clip feature identifiers")
    if (features.status!="VALID").mean()>.05:
        raise RuntimeError("More than 5% extraction failure; resolve before predictive claims")
    forbidden={"outcome_score","school_id","score_status"}
    if forbidden&set(features):raise ValueError("Target or school leaked into audio feature cache")
    return True
