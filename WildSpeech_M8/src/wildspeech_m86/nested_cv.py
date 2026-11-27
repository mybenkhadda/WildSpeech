"""M8.6 nested evaluation: all feature fitting is inside corresponding train fold.

One FoldWorkspace constructs its outer and three inner feature representations
exactly once, sharing those *training-only* caches across preregistered arms.
No label is ever passed to a feature-fitting transformation. For B1 the
CatBoost blend weight and Ridge alpha are selected on inner OOF only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .contracts import ITEM_BLOCKS
from .features import M86Features,select_features
from .modeling import (SPECS, RIDGE_ALPHAS,BLEND_WEIGHTS,new_model)
from wildspeech_m85.models import make_ridge,MultiSeedCatBoost,ConvexBlend


def split_subset(students,clips,pos):
    sub=students.iloc[np.asarray(pos,int)].copy()
    child=clips[clips.subject_id.isin(sub.index)].copy()
    return sub,child

@dataclass
class FoldRepresentation:
    train_pos:np.ndarray
    valid_pos:np.ndarray
    train_features:pd.DataFrame
    valid_features:pd.DataFrame


class FoldWorkspace:
    def __init__(self,students,clips,y,tr,va,seed=42,
                 inner_k=3,extra=ITEM_BLOCKS,rasch_iters=7,rasch_crossfit=3):
        self.students=students;self.clips=clips
        self.y=np.asarray(y,dtype=float)
        self.tr=np.asarray(tr,dtype=int);self.va=np.asarray(va,dtype=int)
        self.seed=int(seed);self.inner_k=int(inner_k)
        self.extra=tuple(extra)
        if set(self.tr)&set(self.va):
            raise ValueError("Outer student leakage")
        mt,ct=split_subset(students,clips,self.tr)
        mv,cv=split_subset(students,clips,self.va)
        self.mt=mt;self.ct=ct
        self.builder=M86Features(self.extra,rasch_iters,rasch_crossfit)
        self.x=self.builder.fit_transform(mt,ct)
        self.xv=self.builder.transform(mv,cv)
        self.inner=[]
        splitter=KFold(n_splits=inner_k,shuffle=True,random_state=self.seed)
        for itr,iva in splitter.split(self.tr):
            imt,ict=split_subset(mt,ct,itr)
            imv,icv=split_subset(mt,ct,iva)
            b=M86Features(self.extra,rasch_iters,rasch_crossfit)
            xi=b.fit_transform(imt,ict)
            vi=b.transform(imv,icv)
            self.inner.append(FoldRepresentation(
                train_pos=np.asarray(itr,int),valid_pos=np.asarray(iva,int),
                train_features=xi,valid_features=vi))

    def _screen(self,model_id,alpha,iterations,seeds):
        spec=SPECS[model_id]
        score=0.
        for fold in self.inner:
            xi=select_features(fold.train_features,model_id)
            vi=select_features(fold.valid_features,model_id)
            model=new_model(spec,alpha=alpha,seeds=seeds,iterations=iterations)
            model.fit(xi,self.y[self.tr[fold.train_pos]])
            predictions=np.asarray(model.predict(vi),float)
            score+=float(np.square(
                self.y[self.tr[fold.valid_pos]]-predictions).sum())
        return score

    def _ridge_like(self,model_id,iterations,seeds):
        errors={float(alpha):self._screen(model_id,alpha,iterations,seeds)
                for alpha in RIDGE_ALPHAS}
        alpha=min(errors,key=lambda a:(errors[a],a))
        x=select_features(self.x,model_id)
        v=select_features(self.xv,model_id)
        model=new_model(SPECS[model_id],alpha=alpha,seeds=seeds,iterations=iterations)
        model.fit(x,self.y[self.tr])
        p=np.asarray(model.predict(v),float)
        return p,{"alpha":alpha,"inner_sse":errors,"kind":SPECS[model_id]["kind"]}

    def _full_feature_blend(self,iterations,seeds,inner_cat_seeds=(42,)):
        model_id="B1"
        pred_cat=np.full(len(self.tr),np.nan)
        ridge_oo={a:np.full(len(self.tr),np.nan) for a in RIDGE_ALPHAS}
        for fold in self.inner:
            xi=select_features(fold.train_features,model_id)
            vi=select_features(fold.valid_features,model_id)
            yy=self.y[self.tr[fold.train_pos]]
            for alpha in RIDGE_ALPHAS:
                m=make_ridge(alpha).fit(xi,yy)
                ridge_oo[alpha][fold.valid_pos]=m.predict(vi)
            cat=MultiSeedCatBoost(seeds=inner_cat_seeds,iterations=iterations).fit(xi,yy)
            pred_cat[fold.valid_pos]=cat.predict(vi)
        y_inner=self.y[self.tr]
        if not np.isfinite(pred_cat).all() or any(not np.isfinite(p).all() for p in ridge_oo.values()):
            raise RuntimeError("Incomplete nested full-feature blend OOF")
        # Choose alpha jointly with blend weight on the training-only OOF.
        pairs={(float(a),float(w)):float(np.square(
            y_inner-((1-w)*r+w*pred_cat)).sum())
            for a,r in ridge_oo.items() for w in BLEND_WEIGHTS}
        alpha,weight=min(pairs,key=lambda x:(pairs[x],x[0],x[1]))
        train=select_features(self.x,model_id)
        valid=select_features(self.xv,model_id)
        ridge=make_ridge(alpha).fit(train,self.y[self.tr])
        cat=MultiSeedCatBoost(seeds=seeds,iterations=iterations).fit(train,self.y[self.tr])
        blend=ConvexBlend(weight,ridge,cat)
        return blend.predict(valid),{"alpha":alpha,"weight_cat":weight,
                                    "inner_sse":float(pairs[(alpha,weight)]),
                                    "outer_seeds":list(seeds),"inner_cat_seeds":list(inner_cat_seeds)}

    def predict(self,model_id,iterations=600,seeds=(42,43,44,45,46)):
        if model_id not in SPECS:raise KeyError(model_id)
        if model_id=="B1":
            return self._full_feature_blend(iterations,seeds)
        return self._ridge_like(model_id,iterations,seeds)


def fit_full_model(data,model_id,iterations=600,seeds=(42,43,44,45,46),
                   inner_k=3,rasch_iters=7,rasch_crossfit=3):
    """Select hyperparameters by full-train internal CV and fit final model."""
    if model_id not in SPECS:raise KeyError(model_id)
    spec=SPECS[model_id]
    extra=spec.get("extra",())
    train=data.train_students
    y=data.labels.reindex(train.index).to_numpy(float)
    clips=data.train_clips
    kfold=KFold(n_splits=inner_k,shuffle=True,random_state=42)
    parts=[]
    for itr,iva in kfold.split(train):
        mt,ct=split_subset(train,clips,itr)
        mv,cv=split_subset(train,clips,iva)
        b=M86Features(extra,rasch_iters,rasch_crossfit)
        xi=b.fit_transform(mt,ct)
        vi=b.transform(mv,cv)
        parts.append((itr,iva,select_features(xi,model_id),select_features(vi,model_id)))
    if model_id=="B1":
        ro={a:np.full(len(y),np.nan) for a in RIDGE_ALPHAS}
        co=np.full(len(y),np.nan)
        for itr,iva,xi,vi in parts:
            for a in RIDGE_ALPHAS:
                ro[a][iva]=make_ridge(a).fit(xi,y[itr]).predict(vi)
            co[iva]=MultiSeedCatBoost(seeds=(42,),iterations=iterations).fit(xi,y[itr]).predict(vi)
        scores={(a,w):float(np.square(y-((1-w)*pred+w*co)).sum())
                for a,pred in ro.items() for w in BLEND_WEIGHTS}
        alpha,weight=min(scores,key=lambda z:(scores[z],z[0],z[1]))
    else:
        errors={}
        for a in RIDGE_ALPHAS:
            errors[a]=sum(float(np.square(y[iva]-new_model(
                spec,alpha=a,seeds=seeds,iterations=iterations
            ).fit(xi,y[itr]).predict(vi)).sum())
              for itr,iva,xi,vi in parts)
        alpha=min(errors,key=lambda x:(errors[x],x))
        weight=None
    builder=M86Features(extra,rasch_iters,rasch_crossfit)
    x=select_features(builder.fit_transform(train,clips),model_id)
    if model_id=="B1":
        linear=make_ridge(alpha).fit(x,y)
        cat=MultiSeedCatBoost(seeds=seeds,iterations=iterations).fit(x,y)
        model=ConvexBlend(weight,linear,cat)
    else:
        model=new_model(spec,alpha=alpha,seeds=seeds,iterations=iterations).fit(x,y)
    return builder,model,{"alpha":alpha,"weight_cat":weight,
                         "selected_on":"train_only_3fold_inner_oof",
                         "model_id":model_id,"n_training_students":len(train)}
