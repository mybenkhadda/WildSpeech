"""Leakage-free feature-view construction for M8.2 student similarity research.

Every estimated transform is fitted on training rows only. Explicit nuisance views
are *audits*, not inputs to the behavioral consensus or pairwise predictive model.
Missing values never count as positive pairwise evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from .m8_pipeline import feature_families, assert_safe_features

# Token-based, not substring matching: 'rms' is intentionally treated as a
# possible recording condition and is examined separately from literacy behavior.
NUISANCE = re.compile(
    r"(?:sample[_]?rate|original[_]?sr|source[_]?sr|reference|expected|"
    r"target[_]?text|ref[_]?text|text[_]?length|n[_]?syllable|"
    r"n[_]?character|word[_]?count|item[_]?id|prompt[_]?length|"
    r"clip[_]?duration|recording|microphone|peak[_]?amplitude|"
    r"rms|snr|quality[_]?usable|quality[_]?coverage|silence[_]?noise)",re.I)
TIMING = re.compile(r"(?:pause|onset|latency|duration|speech[_]?fraction|speaking|"
                    r"speech[_]?rate|fluency|burst|voic|rhythm|timing|articulation[_]?time)",re.I)
PHONETIC = re.compile(r"(?:asr|phonem|phonetic|transcript|ipa|pronunc|edit[_]?distance|wer|cer)",re.I)

@dataclass
class View:
    name: str
    matrix: np.ndarray
    observed: np.ndarray
    columns: list[str]

    def pair_coverage(self) -> np.ndarray:
        """Jointly observed fraction, rather than overlap after imputation."""
        obs = self.observed.astype(np.float32)
        return (obs @ obs.T) / max(1, obs.shape[1])


def select_views(df:pd.DataFrame,max_per_view:int=36,min_nonmissing:float=.55) -> dict[str,list[str]]:
    """Predeclared feature taxonomy, independent of scores and school membership."""
    assert_safe_features(df.columns)
    numeric=df.select_dtypes(include='number').columns.tolist()
    fam=feature_families(numeric)
    candidates={k:[] for k in ('screening','task_behavior','acoustic_timing','acoustic_signal','phonetic_speech','nuisance_audit')}
    for col in numeric:
        x=pd.to_numeric(df[col],errors='coerce')
        if x.notna().mean()<min_nonmissing or x.dropna().nunique()<3:
            continue
        low=col.lower()
        if NUISANCE.search(low):
            candidates['nuisance_audit'].append(col)
        elif PHONETIC.search(low) or fam[col]=='speech_text':
            candidates['phonetic_speech'].append(col)
        elif TIMING.search(low) and fam[col] in ('audio','temporal','context_behavior'):
            candidates['acoustic_timing'].append(col)
        elif fam[col]=='audio':
            candidates['acoustic_signal'].append(col)
        elif fam[col]=='screening':
            candidates['screening'].append(col)
        elif fam[col] in ('context_behavior','temporal'):
            candidates['task_behavior'].append(col)
    # Deterministic, label-free selection. Prevent dozens of mathematically
    # equivalent transformations from swamping a view with near duplicates.
    result={}
    for name,cols in candidates.items():
        priority=lambda c:(0 if ('theta' in c.lower() or 'accuracy' in c.lower() or 'contrast' in c.lower()) else 1,
                           0 if '__log__' not in c and '__sqrt__' not in c and '__logit__' not in c else 1,c)
        cols=sorted(cols,key=priority)
        chosen=[];seen=set()
        for col in cols:
            stem=re.sub(r'm8__(?:log|sqrt|logit)__','',col.lower())
            if stem in seen:continue
            chosen.append(col);seen.add(stem)
            if len(chosen)>=max_per_view:break
        if len(chosen)>=2:result[name]=chosen
    return result


class ViewBuilder:
    """Fit train-only robust transforms and conditional speech residualization."""
    def __init__(self,max_per_view=36,min_nonmissing=.55,clip=6.0):
        self.max_per_view=max_per_view
        self.min_nonmissing=min_nonmissing
        self.clip=clip

    def fit(self,df:pd.DataFrame):
        self.columns_=select_views(df,self.max_per_view,self.min_nonmissing)
        self.params_={}
        for name,cols in self.columns_.items():
            a=df.reindex(columns=cols).apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float,copy=True)
            a[~np.isfinite(a)]=np.nan
            means=np.nanmedian(a,axis=0)
            means=np.nan_to_num(means,nan=0.,posinf=0.,neginf=0.)
            fill=np.where(np.isfinite(a),a,means)
            center=np.nanmedian(fill,axis=0)
            q75=np.percentile(fill,75,axis=0);q25=np.percentile(fill,25,axis=0)
            spread=np.maximum(q75-q25, np.std(fill,axis=0)*.15)
            spread=np.maximum(spread,1e-6)
            self.params_[name]=(means,center,spread)
        # Conditioning is performed without outcomes. Source: acoustic timing and
        # phonetic features only, adjusted for screening. Ridge is regularized to
        # avoid spurious correction with 541 students and correlated inputs.
        self.conditional_sources_=[x for x in ('acoustic_timing','acoustic_signal','phonetic_speech') if x in self.columns_]
        self.conditional_model_=None
        if 'screening' in self.columns_ and self.conditional_sources_:
            source=self._transform_basic(df)
            sx=source['screening'].matrix
            sy=np.concatenate([source[v].matrix for v in self.conditional_sources_],axis=1)
            self.conditional_model_=Ridge(alpha=60.).fit(sx,sy)
        return self

    def _transform_basic(self,df):
        out={}
        for name,cols in self.columns_.items():
            data=df.reindex(columns=cols).apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float,copy=True)
            data[~np.isfinite(data)]=np.nan
            obs=np.isfinite(data)
            med,center,spread=self.params_[name]
            x=np.where(obs,data,med)
            x=np.clip((x-center)/spread,-self.clip,self.clip).astype(np.float32)
            out[name]=View(name,x,obs,cols)
        return out

    def transform(self,df:pd.DataFrame,include_nuisance:bool=False):
        v=self._transform_basic(df)
        if self.conditional_model_ is not None:
            source=np.concatenate([v[z].matrix for z in self.conditional_sources_],axis=1)
            observed=np.concatenate([v[z].observed for z in self.conditional_sources_],axis=1)
            adjusted=np.clip(source-self.conditional_model_.predict(v['screening'].matrix),-self.clip,self.clip)
            names=[z for family in self.conditional_sources_ for z in v[family].columns]
            v['conditional_speech']=View('conditional_speech',adjusted.astype(np.float32),observed,names)
        if not include_nuisance:v.pop('nuisance_audit',None)
        return v

    def fit_transform(self,df,include_nuisance=False):
        return self.fit(df).transform(df,include_nuisance=include_nuisance)