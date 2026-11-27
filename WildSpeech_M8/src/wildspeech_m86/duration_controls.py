"""T1: clip DURATION, not response latency, normalized by training item & score.

No audio is read here. Each training student's durations are removed from
their own normalization references; test/validation never change fitted
moments. Task+status moments back off sparse (task,prompt,status) moments.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import TASKS,prompt_key,validate_inputs

def _status(score):
    if not np.isfinite(score):return "unscored"
    return "correct" if score>=.5 else "incorrect"

def _moments(d,keys):
    return {tuple(key) if isinstance(key,tuple) else (key,):
            (int(row["count"]),float(row["sum"]),float(row["sum_sq"]))
            for key,row in d.groupby(keys).agg(
                count=("logdur","size"),sum=("logdur","sum"),
                sum_sq=("sq","sum")).iterrows()}

class DurationControls:
    def __init__(self,kappa=8.):
        self.kappa=float(kappa)
    def _clean(self,clips):
        d=clips[["subject_id","task","expected_text","item_score","duration_sec"]].copy()
        score=pd.to_numeric(d.item_score,errors="coerce").to_numpy(float)
        dur=pd.to_numeric(d.duration_sec,errors="coerce").to_numpy(float)
        d["logdur"]=np.log1p(np.maximum(0.,dur))
        d=d[np.isfinite(d.logdur)].copy()
        d["sq"]=np.square(d.logdur)
        d["status"]=[_status(s) for s in pd.to_numeric(d.item_score,errors="coerce")]
        d["key"]=[prompt_key(t,s) for t,s in zip(d.task,d.expected_text)]
        return d

    def fit(self,students,clips):
        validate_inputs(students,clips)
        self.train_ids=set(students.index.astype(str))
        d=self._clean(clips)
        self.item_stats=_moments(d,["task","key","status"])
        self.task_stats=_moments(d,["task","status"])
        self.self_items=_moments(d,["task","key","status","subject_id"])
        self.self_tasks=_moments(d,["task","status","subject_id"])
        self.global_mean=float(d.logdur.mean()) if len(d) else 0.
        self.global_std=float(d.logdur.std(ddof=0)) if len(d)>1 else 1.
        self.global_std=max(.15,self.global_std)
        self.columns_=[f"t1_{task}_{c}" for task in TASKS for c in (
            "n_valid","duration_missing_frac","mean_logduration",
            "mean_norm_deviation","std_norm_deviation",
            "positive_deviation_frac","mean_item_support")]
        return self

    def _expected(self,task,key,status,sid,loo):
        i=(task,key,status);t=(task,status)
        c,s,q=self.item_stats.get(i,(0,0.,0.))
        ct,st,qt=self.task_stats.get(t,(0,0.,0.))
        if loo:
            a,b,_=self.self_items.get((task,key,status,sid),(0,0.,0.))
            c-=a;s-=b
            a,b,_=self.self_tasks.get((task,status,sid),(0,0.,0.))
            ct-=a;st-=b
        ref=st/ct if ct>0 else self.global_mean
        expected=(s+self.kappa*ref)/(c+self.kappa)
        return expected,max(0,c)

    def _build(self,students,clips,loo):
        validate_inputs(students,clips)
        if loo and not set(students.index.astype(str)).issubset(self.train_ids):
            raise ValueError("Train ID invalid")
        if not loo and set(students.index.astype(str))&self.train_ids:
            raise ValueError("Training student in inference")
        d=self._clean(clips)
        groups={str(s):g for s,g in d.groupby("subject_id",sort=False)}
        empty=d.iloc[:0]
        raw_counts=clips.groupby(["subject_id","task"]).size()
        rows=[]
        for sid in students.index.astype(str):
            sub=groups.get(sid,empty)
            out={}
            for task in TASKS:
                g=sub[sub.task==task]
                z=[];support=[]
                for row in g.itertuples():
                    ref,n=self._expected(row.task,row.key,row.status,sid,loo)
                    z.append(float(np.clip((row.logdur-ref)/self.global_std,-5.,5.)))
                    support.append(n)
                values=np.asarray(z,float)
                nraw=int(raw_counts.get((sid,task),0))
                pref=f"t1_{task}_"
                out.update({
                    pref+"n_valid":float(len(values)),
                    pref+"duration_missing_frac":float(1.-len(values)/nraw) if nraw else 0.,
                    pref+"mean_logduration":float(g.logdur.mean()) if len(g) else 0.,
                    pref+"mean_norm_deviation":float(values.mean()) if len(values) else 0.,
                    pref+"std_norm_deviation":float(values.std()) if len(values)>1 else 0.,
                    pref+"positive_deviation_frac":float((values>0).mean()) if len(values) else 0.,
                    pref+"mean_item_support":float(np.mean(np.log1p(support))) if support else 0.,
                })
            rows.append(out)
        return pd.DataFrame(rows,index=students.index).reindex(columns=self.columns_)
    def transform_train(self,students,clips):return self._build(students,clips,True)
    def transform(self,students,clips):return self._build(students,clips,False)
