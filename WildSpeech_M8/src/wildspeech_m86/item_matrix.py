"""I1: sparse task×prompt response matrix, fitted only on training students.

Training encodings exclude the student's own attempts from item difficulty.
Missing/unknown items are explicit; this is NOT a chronological sequence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import TASKS, prompt_key,validate_inputs

class ItemMatrix:
    def __init__(self, min_students=6, max_items=160, kappa=12., scale=.5):
        self.min_students=int(min_students)
        self.max_items=int(max_items)
        self.kappa=float(kappa)
        self.scale=float(scale)
        if self.min_students<1 or self.max_items<1 or self.kappa<=0:
            raise ValueError("Invalid item matrix hyperparameters")

    def fit(self, students, clips):
        validate_inputs(students, clips)
        self.train_ids=set(students.index.astype(str))
        d=clips.copy()
        d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
        d=d[d.item_score.notna()].copy()
        d["key"]=[prompt_key(t,s) for t,s in zip(d.task,d.expected_text)]
        g=d.groupby("key").subject_id.nunique().sort_values(ascending=False)
        selected=[k for k in sorted(g.index,key=lambda k:(-int(g[k]),k))
                  if g[k]>=self.min_students][:self.max_items]
        self.keys=tuple(selected)
        self.stats=d.groupby("key").item_score.agg(["sum","count"])
        by_student=d.groupby(["key","subject_id"]).item_score.agg(["sum","count"])
        self.student_stats={(str(k),str(s)):(float(r["sum"]),int(r["count"]))
                            for (k,s),r in by_student.iterrows()}
        self.prior_global=float(d.item_score.mean()) if len(d) else .5
        self.index={k:i for i,k in enumerate(self.keys)}
        self.columns_=[f"i1_item_{i:03d}_excess" for i in range(len(self.keys))]+[
                       f"i1_item_{i:03d}_observed" for i in range(len(self.keys))]
        return self

    def _prior(self,key,sid,loo):
        if key not in self.stats.index:
            return self.prior_global
        s=float(self.stats.loc[key,"sum"]);n=int(self.stats.loc[key,"count"])
        if loo:
            a,b=self.student_stats.get((key,sid),(0.,0))
            s-=a;n-=b
        return (s+self.kappa*self.prior_global)/(n+self.kappa)

    def _transform(self,students,clips,loo):
        validate_inputs(students,clips)
        subject_ids=students.index.astype(str)
        if loo and not set(subject_ids).issubset(self.train_ids):
            raise ValueError("LOO requires original training students")
        if not loo and set(subject_ids)&self.train_ids:
            raise ValueError("Cannot run inference on fitted training students without LOO")
        n=len(students);k=len(self.keys)
        excess=np.zeros((n,k),float)
        observed=np.zeros((n,k),float)
        if k:
            d=clips[["subject_id","task","expected_text","item_score"]].copy()
            d["key"]=[prompt_key(t,s) for t,s in zip(d.task,d.expected_text)]
            d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
            d=d[d.item_score.notna() & d.key.isin(self.index)]
            g=d.groupby(["subject_id","key"]).item_score.agg(["sum","count"])
            row_index={s:i for i,s in enumerate(subject_ids)}
            for (sid,key),vals in g.iterrows():
                sid=str(sid); idx=row_index.get(sid); col=self.index.get(key)
                if idx is None or col is None:continue
                p=self._prior(key,sid,loo)
                excess[idx,col]=self.scale*(float(vals["sum"])/int(vals["count"])-p)
                observed[idx,col]=self.scale
        result=pd.DataFrame(np.c_[excess,observed],columns=self.columns_,index=students.index)
        return result

    def transform_train(self,students,clips):
        return self._transform(students,clips,True)
    def transform(self,students,clips):
        return self._transform(students,clips,False)
