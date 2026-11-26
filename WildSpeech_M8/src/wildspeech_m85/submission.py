"""Full raw-screening fit and singleton-inference-controlled submission.

Only the preregistered promoted model is exported. The official reference
submission is left untouched if no candidate meets every held-out gate.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold
from wildspeech_m84.reference import make_reference
from wildspeech_m84.data import atomic_json,digest_bytes
from .features import ScreeningFeatures
from .nested_cv import FULL_BLOCKS,subset_columns,model_spec
from .models import RIDGE_ALPHAS,make_ridge,MultiSeedCatBoost,ConvexBlend


def _final_select_alpha(data,y,blocks,remove_nuisance,kappa,n_knots,seed=42):
    kfold=KFold(n_splits=3,shuffle=True,random_state=seed)
    performance={float(a):0. for a in RIDGE_ALPHAS}
    for train,valid in kfold.split(data.train_students):
        meta_tr=data.train_students.iloc[train]
        meta_va=data.train_students.iloc[valid]
        clips_tr=data.train_clips[data.train_clips.subject_id.isin(meta_tr.index)]
        clips_va=data.train_clips[data.train_clips.subject_id.isin(meta_va.index)]
        builder=ScreeningFeatures(FULL_BLOCKS,kappa,n_knots)
        xt=builder.fit_transform(meta_tr,clips_tr)
        xv=builder.transform(meta_va,clips_va)
        tr=subset_columns(xt,blocks,remove_nuisance)
        va=subset_columns(xv,blocks,remove_nuisance)
        for a in RIDGE_ALPHAS:
            m=make_ridge(a).fit(tr,y[train])
            performance[float(a)]+=float(np.square(y[valid]-m.predict(va)).sum())
    return min(performance,key=lambda alpha:(performance[alpha],alpha))


def _final_blend_weight(data,y,seeds,iterations):
    kfold=KFold(n_splits=3,shuffle=True,random_state=42)
    pr=np.full(len(y),np.nan)
    pc=np.full(len(y),np.nan)
    for tr,va in kfold.split(data.train_students):
        train=data.train_students.iloc[tr]
        valid=data.train_students.iloc[va]
        tclips=data.train_clips[data.train_clips.subject_id.isin(train.index)]
        vclips=data.train_clips[data.train_clips.subject_id.isin(valid.index)]
        b=ScreeningFeatures(())
        x=b.fit_transform(train,tclips)
        xv=b.transform(valid,vclips)
        pr[va]=make_reference().fit(x,y[tr]).predict(xv)
        pc[va]=MultiSeedCatBoost(seeds,iterations).fit(x,y[tr]).predict(xv)
    if not np.isfinite(pr).all() or not np.isfinite(pc).all():
        raise ValueError("Incomplete final blend cross-fitting")
    candidates=(0.,.25,.5,.75,1.)
    error=[np.square(y-((1-w)*pr+w*pc)).sum() for w in candidates]
    return float(candidates[int(np.argmin(error))])


def fit_and_export(data,model_id,outdir,seed_count=5,iterations=600,kappa=10.,n_knots=4):
    """Fit only labeled students, then verify raw per-student inference invariance."""
    if model_id not in ("E01","E02","E03","E04"):
        raise ValueError("Only preregistered eligible E01–E04 can produce a submission")
    spec=model_spec(model_id)
    if "S2" in spec["blocks"]:
        if "item_score" not in data.test_clips.columns or not data.test_clips.item_score.notna().any():
            raise RuntimeError("S2 promotion requires officially available test item_score")
    meta=data.train_students
    y=data.labels.reindex(meta.index).to_numpy(float)
    xmeta=data.test_students.reindex(data.submission_format.index)
    if xmeta.isna().all(axis=1).any():
        raise ValueError("Submission student has missing metadata")
    blocks=FULL_BLOCKS if spec["kind"]=="ridge" else ()
    builder=ScreeningFeatures(blocks,kappa,n_knots)
    train_features=builder.fit_transform(meta,data.train_clips)
    no_nuisance=spec.get("remove_nuisance",False)
    train_x=subset_columns(train_features,spec["blocks"],no_nuisance)
    seeds=tuple(42+i for i in range(seed_count))
    info={"model_id":model_id,"seeds":list(seeds),"iterations":iterations,
          "kappa":kappa,"n_knots":n_knots,"test_students":len(xmeta)}
    if spec["kind"]=="ridge":
        alpha=_final_select_alpha(data,y,spec["blocks"],no_nuisance,kappa,n_knots)
        model=make_ridge(alpha).fit(train_x,y)
        info["selected_alpha"]=alpha
    elif spec["kind"]=="cat":
        model=MultiSeedCatBoost(seeds,iterations).fit(train_x,y)
    else:
        w=_final_blend_weight(data,y,seeds,iterations)
        model=ConvexBlend(w,make_reference().fit(train_x,y),
                          MultiSeedCatBoost(seeds,iterations).fit(train_x,y))
        info["catboost_weight"]=w
    class RawScreeningModel:
        def __init__(self,builder,model,blocks,no_nuisance):
            self.builder=builder;self.model=model
            self.blocks=tuple(blocks);self.no_nuisance=no_nuisance
        def predict(self,students,clips):
            z=self.builder.transform(students,clips)
            return np.asarray(self.model.predict(subset_columns(
                z,self.blocks,self.no_nuisance)),dtype=float)
    raw=RawScreeningModel(builder,model,spec["blocks"],no_nuisance)
    # Validate the same final representation used for inference and test every student.
    all_x=subset_columns(builder.transform(xmeta,data.test_clips),
                         spec["blocks"],no_nuisance)
    predictions=np.asarray(model.predict(all_x),dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Submission contains nonfinite model predictions")
    for pos,sid in enumerate(xmeta.index.astype(str)):
        one=xmeta.iloc[[pos]]
        clips=data.test_clips[data.test_clips.subject_id.astype(str)==sid]
        local=raw.predict(one,clips)
        if not np.isclose(local[0],predictions[pos],rtol=0,atol=1e-7):
            raise RuntimeError(f"Raw singleton inference failed for position {pos}")
    permutation=np.random.default_rng(42).permutation(len(xmeta))
    shuffled=raw.predict(xmeta.iloc[permutation],data.test_clips)
    if not np.allclose(shuffled,predictions[permutation],rtol=0,atol=1e-7):
        raise RuntimeError("Inference changed under test row permutation")
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    filename=outdir/"m0plus_submission.csv"
    frame=pd.DataFrame({"subject_id":xmeta.index.astype(str),"outcome_score":predictions})
    assert frame.subject_id.tolist()==data.submission_format.index.astype(str).tolist()
    temp=outdir/"m0plus_submission.tmp.csv"
    frame.to_csv(temp,index=False,float_format="%.12f",lineterminator="\n")
    temp.replace(filename)
    dump({"builder":builder,"model":model,"feature_blocks":spec["blocks"],
          "remove_nuisance":no_nuisance,"info":info},outdir/"m0plus_final.joblib",compress=3)
    digest=digest_bytes(filename)
    (outdir/"m0plus_submission.sha256").write_text(digest+"\n",encoding="utf-8")
    info.update({"file":str(filename),"sha256":digest,
                 "singleton_raw_pipeline_test":True,"test_permutation_invariance":True,
                 "test_fit_count":0,"not_submitted_online":True})
    atomic_json(outdir/"submission_manifest.json",info)
    return info
