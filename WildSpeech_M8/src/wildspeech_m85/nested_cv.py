"""Strict inner CV: rebuild S2/item statistics and S3 knots on inner train only."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from wildspeech_m84.reference import make_reference
from .contracts import TASKS
from .features import ScreeningFeatures
from .models import make_ridge,MultiSeedCatBoost,ConvexBlend,RIDGE_ALPHAS

FULL_BLOCKS=("S2","S3","S4","S5")


def subset_columns(frame,blocks,remove_nuisance=False):
    keep=list(frame.columns)
    active=set(blocks)
    # Prefixes are provenance labels, not heuristics. Preserve exactly the
    # explicitly active blocks; never discard an active family accidentally.
    keep=[c for c in keep if not any(c.startswith(s.lower()+"_") for s in FULL_BLOCKS if s not in active)]
    if remove_nuisance:
        nuisance={"blending_mean_duration","deletion_mean_duration",
                  "sentence_repetition_mean_duration","is_11khz"}
        keep=[c for c in keep if c not in nuisance]
    return frame[keep].copy()


def _students_with_clips(students,clips,positions):
    meta=students.iloc[np.asarray(positions,int)].copy()
    raw=clips.loc[clips.subject_id.isin(meta.index)].copy()
    return meta,raw


class FoldCache:
    """Outer train/validation plus three train-only inner fold representations."""
    def __init__(self,students,clips,y,tr,va,seed=42,kappa=10.,n_knots=4,inner_k=3):
        if set(tr)&set(va):raise ValueError("Outer train/validation overlap")
        self.students=students;self.clips=clips;self.y=np.asarray(y,float)
        self.tr=np.asarray(tr,int);self.va=np.asarray(va,int)
        self.kappa=kappa;self.n_knots=n_knots;self.seed=seed
        self.tr_meta,self.tr_clips=_students_with_clips(students,clips,tr)
        self.va_meta,self.va_clips=_students_with_clips(students,clips,va)
        builder=ScreeningFeatures(FULL_BLOCKS,kappa,n_knots)
        self.outer_x=builder.fit_transform(self.tr_meta,self.tr_clips)
        self.outer_v=builder.transform(self.va_meta,self.va_clips)
        self.builder=builder
        self.inner=[]
        splitter=KFold(n_splits=inner_k,shuffle=True,random_state=seed)
        for itr,iva in splitter.split(self.tr):
            inner_train_meta,inner_train_clips=_students_with_clips(self.tr_meta,self.tr_clips,itr)
            inner_valid_meta,inner_valid_clips=_students_with_clips(self.tr_meta,self.tr_clips,iva)
            ib=ScreeningFeatures(FULL_BLOCKS,kappa,n_knots)
            inner_x=ib.fit_transform(inner_train_meta,inner_train_clips)
            inner_v=ib.transform(inner_valid_meta,inner_valid_clips)
            self.inner.append((itr,iva,inner_x,inner_v))

    def ridge(self,blocks,remove_nuisance=False,alpha_grid=RIDGE_ALPHAS):
        scores=[]
        for alpha in alpha_grid:
            weighted_sse=0.;n=0
            for itr,iva,xi,xv in self.inner:
                cols=subset_columns(xi,blocks,remove_nuisance)
                val=subset_columns(xv,blocks,remove_nuisance)
                m=make_ridge(alpha).fit(cols,self.y[self.tr[itr]])
                residual=self.y[self.tr[iva]]-m.predict(val)
                weighted_sse+=float(np.square(residual).sum());n+=len(iva)
            scores.append(weighted_sse/max(1,n))
        best_idx=int(np.argmin(scores))
        alpha=float(alpha_grid[best_idx])
        x=subset_columns(self.outer_x,blocks,remove_nuisance)
        v=subset_columns(self.outer_v,blocks,remove_nuisance)
        model=make_ridge(alpha).fit(x,self.y[self.tr])
        return model.predict(v),alpha,model

    def cat(self,seeds=(42,43,44,45,46),iterations=600):
        x=subset_columns(self.outer_x,(),False)
        v=subset_columns(self.outer_v,(),False)
        model=MultiSeedCatBoost(seeds=seeds,iterations=iterations)
        model.fit(x,self.y[self.tr])
        return model.predict(v),model

    def blend(self,seeds=(42,43,44,45,46),iterations=600,
              weights=(0.,.25,.5,.75,1.)):
        # E03: S0-only fixed reference vs fixed CatBoost, both re-fit in each
        # inner student split. Candidate weights optimize inner OOF *SSE*.
        inner_pred_r=np.empty(len(self.tr),dtype=float)
        inner_pred_c=np.empty(len(self.tr),dtype=float)
        for itr,iva,xi,xv in self.inner:
            x=subset_columns(xi,(),False)
            v=subset_columns(xv,(),False)
            ref=make_reference().fit(x,self.y[self.tr[itr]])
            cat=MultiSeedCatBoost(seeds=seeds,iterations=iterations)
            cat.fit(x,self.y[self.tr[itr]])
            inner_pred_r[iva]=ref.predict(v)
            inner_pred_c[iva]=cat.predict(v)
        if not np.isfinite(inner_pred_r).all() or not np.isfinite(inner_pred_c).all():
            raise ValueError("Inner OOF incomplete or invalid")
        target=self.y[self.tr]
        mse=[float(np.mean(np.square(target-((1-w)*inner_pred_r+w*inner_pred_c))))
             for w in weights]
        chosen=float(weights[int(np.argmin(mse))])
        x=subset_columns(self.outer_x,(),False)
        v=subset_columns(self.outer_v,(),False)
        linear=make_reference().fit(x,target)
        cat=MultiSeedCatBoost(seeds=seeds,iterations=iterations).fit(x,target)
        model=ConvexBlend(chosen,linear,cat)
        return model.predict(v),chosen,model


def model_spec(model_id):
    """All experimental families fixed in advance; no outer-CV adaptive tuning."""
    specs={
        "E01":{"kind":"ridge","blocks":["S3"],"description":"Spline Ridge on S0"},
        "E02":{"kind":"cat","blocks":[],"description":"Shallow fixed CatBoost on S0"},
        "E03":{"kind":"blend","blocks":[],"description":"Inner-OOF convex blend S0"},
        "E04":{"kind":"ridge","blocks":list(FULL_BLOCKS),"description":"S2–S5 bundled Ridge"},
        "E05":{"kind":"ridge","blocks":["S3","S4","S5"],"description":"E04 minus S2"},
        "E06":{"kind":"ridge","blocks":["S2","S4","S5"],"description":"E04 minus S3"},
        "E07":{"kind":"ridge","blocks":["S2","S3","S5"],"description":"E04 minus S4"},
        "E08":{"kind":"ridge","blocks":["S2","S3","S4"],"description":"E04 minus S5"},
        "E09":{"kind":"ridge","blocks":list(FULL_BLOCKS),"remove_nuisance":True,
               "description":"S2–S5 Ridge without S0 durations/sample-rate"},
    }
    if model_id not in specs:raise KeyError(f"Unknown preregistered model ID: {model_id}")
    return specs[model_id]


def predict_fold(cache,model_id,seeds=(42,43,44,45,46),iterations=600):
    spec=model_spec(model_id)
    if spec["kind"]=="ridge":
        pred,alpha,model=cache.ridge(spec["blocks"],spec.get("remove_nuisance",False))
        return pred,{"alpha":alpha,"model_family":"inner_cv_ridge"}
    if spec["kind"]=="cat":
        pred,model=cache.cat(seeds=seeds,iterations=iterations)
        return pred,{"seeds":list(seeds),"model_family":"catboost"}
    if spec["kind"]=="blend":
        pred,weight,model=cache.blend(seeds=seeds,iterations=iterations)
        return pred,{"weight_cat":weight,"model_family":"inner_oof_blend"}
    raise RuntimeError("Unreachable experiment model family")
