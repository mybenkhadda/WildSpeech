"""Train-only final M8.7 model, with independent raw student predictions.

Export only an eligible FULL candidate that passed the engineering/robustness
gate and whose localization QA is reviewed and acceptable. No automatic web
submission. Fits all parameters on TRAIN; frozen phonetic inference for each
test clip can run independently. This source consumes cached acoustic
inference, never trains ASR on hidden-test clips.
"""
from __future__ import annotations
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import KFold
from wildspeech_m84.data import atomic_json,digest_bytes
from wildspeech_m86.features import M86Features,select_features
from .clip_models import ClipCorrectness
from .student_aggregation import SpeechAggregator
from .fold_models import (new_ridge,Representation,representation,
                          _select_audio,_project_layers,ALPHAS,ITEM_BLOCKS,
                          _student_clips,CONTROL_ARMS)

class FrozenSpeechStudentPredictor:
    def __init__(self,screen_builder,clip_correctness,aggregator,
                 weighted_aggregator,model,arm,
                 baseline_model=None,speech_model=None,meta_model=None,
                 resid_model=None,shrinkage=.25,layer_transform=None):
        self.screen_builder=screen_builder
        self.clip_correctness=clip_correctness
        self.aggregator=aggregator
        self.weighted_aggregator=weighted_aggregator
        self.model=model
        self.arm=arm
        self.baseline_model=baseline_model
        self.speech_model=speech_model
        self.meta_model=meta_model
        self.resid_model=resid_model
        self.shrinkage=shrinkage
        self.layer_transform=layer_transform

    def prepare(self,students,clips):
        x=select_features(self.screen_builder.transform(students,clips),"I5")
        okay=clips.status.eq("VALID")
        predicted=pd.Series(np.nan,index=clips.index,dtype=float)
        if okay.any():
            scores=self.clip_correctness.model.predict(
                self.clip_correctness._x(clips.loc[okay]))
            predicted.loc[okay]=np.clip(scores,0,1)
        enriched=self.clip_correctness._enrich(clips,predicted)
        a=self.aggregator.transform(students,enriched)
        w=self.weighted_aggregator.transform(students,enriched)
        return x,a,w

    def _design(self,screen,agg,weighted):
        source=weighted if self.arm=="A12" else agg
        s=_select_audio(source,self.arm)
        if self.layer_transform is not None:
            # Stage-one train-fitted layer PCA, never refit on test.
            im,scaler,pca,cols,keep=self.layer_transform
            vectors=pca.transform(scaler.transform(im.transform(s[cols])))
            s=s[keep].copy()
            for k in range(vectors.shape[1]):s[f"speech_layer_pca_{k:02d}"]=vectors[:,k]
        if self.arm=="A04":return s
        return pd.concat([screen,s],axis=1)

    def predict(self,students,clips):
        screen,agg,weighted=self.prepare(students,clips)
        if self.arm=="A16":
            s=_select_audio(agg,self.arm)
            b=self.baseline_model.predict(screen)
            a=self.speech_model.predict(s)
            return np.asarray(self.meta_model.predict(
                pd.DataFrame({"screen":b,"speech":a},index=students.index)),float)
        if self.arm=="A14":
            s=_select_audio(agg,self.arm)
            return np.asarray(self.baseline_model.predict(screen)+
                              self.shrinkage*self.resid_model.predict(s),float)
        x=self._design(screen,agg,weighted)
        return np.asarray(self.model.predict(x),float)

def _final_fit(data,features,arm,rasch_iters=7):
    if arm in CONTROL_ARMS:raise ValueError("Controls cannot produce submissions")
    from .evaluation import _joined_features
    from .contracts import clip_inventory
    clips=_joined_features(clip_inventory(data),features)
    tr=_student_clips(clips,data.train_students)
    te=_student_clips(clips,data.test_students)
    y=data.labels.reindex(data.train_students.index).to_numpy(float)
    cc=ClipCorrectness(n_splits=4)
    tr_clip,te_clip=cc.fit_predict(tr,te)
    builder=M86Features(ITEM_BLOCKS,rasch_iters=rasch_iters)
    sx=select_features(builder.fit_transform(data.train_students,tr_clip),"I5")
    sv=select_features(builder.transform(data.test_students,te_clip),"I5")
    agg=SpeechAggregator(tuple(__import__(
        "wildspeech_m87.student_aggregation",fromlist=["FAMILIES"]).FAMILIES))
    ax=agg.fit(tr_clip).transform_train(data.train_students,tr_clip)
    av=agg.transform(data.test_students,te_clip)
    weighted=SpeechAggregator(("A12",),weighted=True)
    wx=weighted.fit(tr_clip).transform_train(data.train_students,tr_clip)
    wv=weighted.transform(data.test_students,te_clip)
    # Choose hyperparameter and stack weights only by TRAIN-student OOF.
    kfold=KFold(n_splits=3,shuffle=True,random_state=42)
    splits=[]
    for it,iv in kfold.split(data.train_students):
        mtr=data.train_students.iloc[it];mva=data.train_students.iloc[iv]
        ci=_student_clips(tr,mtr);cv=_student_clips(tr,mva)
        r=representation(mtr,mva,ci,cv,rasch_iters)
        splits.append((np.asarray(it,int),np.asarray(iv,int),r))
    from .student_aggregation import FAMILIES
    from .fold_models import _project_layers
    layer_transform=None
    if arm=="A13":
        selected=_select_audio(ax,arm)
        selected_test=_select_audio(av,arm)
        cols=[c for c in selected if "_layer_" in c and c.endswith("_mean")]
        if cols:
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            im=SimpleImputer(strategy="constant",fill_value=0)
            a=im.fit_transform(selected[cols])
            scaler=StandardScaler().fit(a)
            pca=PCA(n_components=min(8,len(a)-1,len(cols)),random_state=42).fit(
                scaler.transform(a))
            keep=[c for c in selected if c not in cols]
            layer_transform=(im,scaler,pca,cols,keep)

    if arm=="A16":
        pbase=np.full(len(y),np.nan);pspeech=np.full(len(y),np.nan)
        for it,iv,r in splits:
            xb,vb=r.base()
            xs,vs=r.speech(arm)
            pbase[iv]=new_ridge(100.).fit(xb,y[it]).predict(vb)
            pspeech[iv]=new_ridge(100.).fit(xs,y[it]).predict(vs)
        meta=new_ridge(100.).fit(pd.DataFrame({"screen":pbase,"speech":pspeech}),y)
        bmodel=new_ridge(100.).fit(sx,y)
        smodel=new_ridge(100.).fit(_select_audio(ax,arm),y)
        artifact=FrozenSpeechStudentPredictor(builder,cc,agg,weighted,None,arm,
                      baseline_model=bmodel,speech_model=smodel,meta_model=meta)
        return artifact,{"model":"crossfit_stack","alpha":100.}
    if arm=="A14":
        pred=np.full(len(y),np.nan)
        for it,iv,r in splits:
            xb,vb=r.base()
            pred[iv]=new_ridge(100.).fit(xb,y[it]).predict(vb)
        base=new_ridge(100.).fit(sx,y)
        resid=new_ridge(100.).fit(_select_audio(ax,arm),y-pred)
        artifact=FrozenSpeechStudentPredictor(builder,cc,agg,weighted,None,arm,
                      baseline_model=base,resid_model=resid)
        return artifact,{"model":"residual_shrink","shrinkage":.25}
    # Ordinary joint model, all hyperparameter selection from three inner splits.
    err={}
    for alpha in ALPHAS:
        error=0.
        for it,iv,r in splits:
            x,v=r.design(arm)
            p=new_ridge(alpha).fit(x,y[it]).predict(v)
            error+=float(np.square(y[iv]-p).sum())
        err[float(alpha)]=error
    alpha=min(err,key=lambda z:(err[z],z))
    selected=_select_audio(wx if arm=="A12" else ax,arm)
    if layer_transform is not None:
        im,scaler,pca,cols,keep=layer_transform
        vec=pca.transform(scaler.transform(im.transform(selected[cols])))
        selected=selected[keep].copy()
        for j in range(vec.shape[1]):selected[f"speech_layer_pca_{j:02d}"]=vec[:,j]
    x=selected if arm=="A04" else pd.concat([sx,selected],axis=1)
    model=new_ridge(alpha).fit(x,y)
    return FrozenSpeechStudentPredictor(builder,cc,agg,weighted,model,arm,
                     layer_transform=layer_transform),{
                     "model":"joint_ridge","alpha":alpha,"inner_sse":err}

def export_if_eligible(drive_root,cache_root,eval_root,output_dir="",
                       data_root="",rasch_iters=7,
                       require_qa=True):
    from wildspeech_m84.data import load_official,locate_data_root
    from .gpu_cache import load_features
    eval_root=Path(eval_root)
    report=json.loads((eval_root/"reports"/"M87_RESULTS.json").read_text())
    if report["mode"]!="FULL" or report["decision"]!="ENGINEERING_CANDIDATE":
        raise RuntimeError("No FULL eligible candidate")
    arm=report["winner"]
    if arm in CONTROL_ARMS:raise RuntimeError("Control selected as candidate")
    if require_qa:
        qa=Path(cache_root)/"qa"/"localization_qa_results.json"
        if not qa.is_file():raise RuntimeError("Training localization QA must be reviewed")
        q=json.loads(qa.read_text())
        if not q.get("localization_gate_passed"):
            raise RuntimeError("Localizer did not pass reviewed training QA gate")
    data=load_official(locate_data_root(drive_root,data_root))
    feature=load_features(cache_root)
    if feature.uid.duplicated().any():raise ValueError("Duplicate cached acoustic ID")
    artifact,settings=_final_fit(data,feature,arm,rasch_iters)
    from .evaluation import _joined_features
    from .contracts import clip_inventory
    test=_joined_features(clip_inventory(data),feature)
    test=test[test.split=="test"]
    students=data.test_students.reindex(data.submission_format.index)
    batch=artifact.predict(students,test)
    if not np.isfinite(batch).all():raise ValueError("Nonfinite final test predictions")
    for idx,sid in enumerate(students.index):
        single=artifact.predict(students.iloc[[idx]],
                                test[test.subject_id.astype(str)==str(sid)])
        if len(single)!=1 or not np.allclose(single[0],batch[idx],atol=1e-7,rtol=0):
            raise RuntimeError(f"Raw independent test student prediction failed: {idx}")
    perm=np.random.default_rng(42).permutation(len(students))
    shuffled=artifact.predict(students.iloc[perm],test)
    if not np.allclose(shuffled,batch[perm],atol=1e-7,rtol=0):
        raise RuntimeError("Test row permutation changed predictions")
    out=Path(output_dir) if output_dir else eval_root/"submission"
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({"subject_id":students.index.astype(str),
                        "outcome_score":batch})
    if frame.subject_id.tolist()!=data.submission_format.index.astype(str).tolist():
        raise RuntimeError("Submission order mismatch")
    temp=out/"m87_submission.tmp.csv"
    final=out/"m87_submission.csv"
    frame.to_csv(temp,index=False,float_format="%.12f",lineterminator="\n")
    os.replace(temp,final)
    sha=digest_bytes(final)
    (out/"m87_submission.sha256").write_text(sha+"\n")
    dump({"artifact":artifact,"settings":settings,"model_id":arm},
         out/"m87_predictor.joblib",compress=3)
    result={"model":arm,"file":str(final),"sha256":sha,
            "rows":len(batch),"fitted_on_test":False,
            "all_test_student_singleton_verified":True,
            "permutation_invariance_verified":True,
            "model_path":str(out/"m87_predictor.joblib"),
            "not_submitted_online":True,
            "qa_gate_checked":require_qa,"settings":settings}
    atomic_json(out/"M87_SUBMISSION_MANIFEST.json",result)
    return result
