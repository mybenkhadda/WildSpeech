"""I2: regularized one-parameter Rasch model, with student-out-of-fit scores.

Fixed discrimination=1. No future-reading target is used. All item difficulty
parameters are fitted inside the training fold. Training-student scores are
cross-fitted on *other students' responses*, rather than fitting their own
items and then reusing that training representation on an unseen student.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import KFold
from .contracts import TASKS,prompt_key,validate_inputs

FIELDS=("theta_map","theta_se","n_scored","n_known_items","theta_minus_official")

def _scored(clips):
    if clips.empty:return pd.DataFrame(columns=["subject_id","task","key","item_score"])
    d=clips[["subject_id","task","expected_text","item_score"]].copy()
    d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
    d=d[d.item_score.notna()].copy()
    d["key"]=[prompt_key(t,x) for t,x in zip(d.task,d.expected_text)]
    return d

class RaschAbility:
    def __init__(self, person_precision=1., item_precision=1.5, outer_iters=7, inner_iters=6, crossfit=3):
        self.person_precision=float(person_precision)
        self.item_precision=float(item_precision)
        self.outer_iters=int(outer_iters)
        self.inner_iters=int(inner_iters)
        self.crossfit=int(crossfit)
        if min(self.person_precision,self.item_precision)<=0:
            raise ValueError("Rasch regularization must be positive")

    def fit(self,students,clips):
        validate_inputs(students,clips)
        self.train_ids=set(students.index.astype(str))
        scored=_scored(clips)
        self.difficulties={}
        for task in TASKS:
            d=scored[scored.task==task]
            if d.empty:
                self.difficulties[task]={}
                continue
            items=sorted(d.key.unique())
            persons=students.index.astype(str).tolist()
            ii={v:k for k,v in enumerate(items)}
            pp={v:k for k,v in enumerate(persons)}
            u=np.asarray([pp[str(x)] for x in d.subject_id],int)
            v=np.asarray([ii[x] for x in d.key],int)
            y=d.item_score.to_numpy(float)
            sums=np.bincount(v,weights=y,minlength=len(items))
            counts=np.bincount(v,minlength=len(items))
            b=-np.log(np.maximum(.02,np.minimum(.98,(sums+.5)/(counts+1.)))/
                      np.maximum(.02,1.-np.minimum(.98,(sums+.5)/(counts+1.))))
            theta=np.zeros(len(persons),float)
            # Alternating MAP Newton updates, vectorized over observed responses.
            # This is a strongly regularized, 1PL model, not a 2PL.
            for _ in range(self.outer_iters):
                for _ in range(self.inner_iters):
                    p=expit(theta[u]-b[v])
                    g=np.bincount(u,weights=y-p,minlength=len(persons))-self.person_precision*theta
                    h=np.bincount(u,weights=p*(1-p),minlength=len(persons))+self.person_precision
                    theta+=np.clip(g/h,-1.5,1.5)
                p=expit(theta[u]-b[v])
                grad=np.bincount(v,weights=p-y,minlength=len(items))-self.item_precision*b
                hess=np.bincount(v,weights=p*(1-p),minlength=len(items))+self.item_precision
                b+=np.clip(grad/hess,-1.0,1.0)
                b=np.clip(b,-5.,5.)
            self.difficulties[task]={key:float(z) for key,z in zip(items,b)}
        self.columns_=[f"i2_{task}_{field}" for task in TASKS for field in FIELDS]
        return self

    def _score(self,students,clips):
        validate_inputs(students,clips)
        d=_scored(clips)
        by_student={str(sid):g for sid,g in d.groupby("subject_id",sort=False)}
        empty=d.iloc[:0]
        rows=[]
        for sid in students.index.astype(str):
            sub=by_student.get(sid,empty)
            row={}
            for task in TASKS:
                g=sub[sub.task==task]
                known=self.difficulties.get(task,{})
                observed=[(float(r.item_score),known[r.key])
                          for r in g.itertuples() if r.key in known]
                theta=0.; info=self.person_precision
                if observed:
                    y=np.asarray([r[0] for r in observed],float)
                    b=np.asarray([r[1] for r in observed],float)
                    for _ in range(12):
                        p=expit(theta-b)
                        grad=float((y-p).sum()-self.person_precision*theta)
                        info=float((p*(1-p)).sum()+self.person_precision)
                        step=np.clip(grad/info,-1.,1.)
                        theta+=step
                        if abs(step)<1e-7:break
                    p=expit(theta-b)
                    info=float((p*(1-p)).sum()+self.person_precision)
                t=float(pd.to_numeric(students.loc[sid,task+"_theta"],errors="coerce"))
                row.update({
                    f"i2_{task}_theta_map":float(theta),
                    f"i2_{task}_theta_se":float(1./np.sqrt(info)),
                    f"i2_{task}_n_scored":float(len(g)),
                    f"i2_{task}_n_known_items":float(len(observed)),
                    f"i2_{task}_theta_minus_official":float(theta-t) if np.isfinite(t) else 0.
                })
            rows.append(row)
        return pd.DataFrame(rows,index=students.index).reindex(columns=self.columns_)

    def transform(self,students,clips):
        if not hasattr(self,"difficulties"):
            raise RuntimeError("Fit Rasch parameters before scoring")
        if set(students.index.astype(str))&self.train_ids:
            raise ValueError("Training students cannot be scored as unseen students")
        return self._score(students,clips)

    def transform_train(self,students,clips):
        ids=students.index.astype(str).tolist()
        if len(ids)!=len(self.train_ids) or set(ids)!=self.train_ids:
            raise ValueError("Wrong Rasch training student set")
        if len(students)<self.crossfit:
            raise ValueError("Too few students to cross-fit Rasch")
        output=pd.DataFrame(index=students.index,columns=self.columns_,dtype=float)
        splitter=KFold(n_splits=self.crossfit,shuffle=True,random_state=42)
        for tr,va in splitter.split(students):
            mtrain=students.iloc[tr];mvalid=students.iloc[va]
            ct=clips[clips.subject_id.isin(mtrain.index)]
            cv=clips[clips.subject_id.isin(mvalid.index)]
            other=RaschAbility(self.person_precision,self.item_precision,
                               self.outer_iters,self.inner_iters,self.crossfit).fit(mtrain,ct)
            output.iloc[va]=other.transform(mvalid,cv).to_numpy(float)
        if output.isna().any().any():
            raise ValueError("Incomplete train-side student-crossfitted Rasch")
        return output.astype(float)
