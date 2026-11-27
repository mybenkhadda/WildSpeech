"""Student-local, item-normalized and uncertainty-weighted speech aggregation.

Item moments are learned only on outer-training clips. For train rows,
normalization can optionally subtract the student's own contributions.
Missing signals remain missing; they are never called incorrect. The output
is compact task-level summary, not one predictor per frame/clip.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FAMILIES={
 "A00":("whole_phone_edit_rate",),
 "A01":("whole_target_logp_per_phone","whole_target_vs_free_per_frame",
        "whole_phone_weighted_edit_rate","whole_phone_edit_rate"),
 "A02":("child_target_logp_per_phone","child_target_vs_free_per_frame",
        "child_phone_weighted_edit_rate","child_phone_edit_rate"),
 "A03":("child_aligned_phone_confidence","child_aligned_phone_margin",
        "child_aligned_phone_entropy","child_posterior_entropy",
        "child_posterior_margin","child_phone_coverage"),
 "A04":("rms","noise_floor","silence_frac","clipping_frac","duration_s",
        "original_sr","vad_segments","vad_speech_seconds"),
 "A05":("random_target_logp_per_phone","random_phone_weighted_edit_rate"),
 "A06":("child_phone_edit_rate","child_target_logp_per_phone",
        "child_posterior_entropy"),
 "A07":("word_coverage","word_lcs_ratio","word_content_coverage",
        "word_function_coverage","word_endings_coverage",
        "word_normalized_edits","word_deletions","word_substitutions"),
 "A08":("task_phone_graded_accuracy","task_phone_omission_rate",
        "task_phone_substitution_rate","task_phone_order_ratio",
        "task_phone_initial_errors","task_phone_middle_errors",
        "task_phone_final_errors","task_phone_segmented_proxy"),
 "A09":("task_phone_graded_accuracy","task_phone_omission_rate",
        "child_phone_edit_rate","word_coverage","word_lcs_ratio"),
 "A10":("clip_pred_correct","clip_surprise","clip_pred_on_correct",
        "clip_pred_on_incorrect"),
 "A11":("clip_unscored_recovery","clip_pred_correct"),
 "A12":("child_target_logp_per_phone","child_phone_edit_rate",
        "task_phone_graded_accuracy","word_coverage"),
 "A13":tuple(f"layer_{layer:02d}_{i:02d}"
             for layer in (6,7,8,9,10) for i in range(24)),
 "A14":("child_phone_edit_rate","child_target_logp_per_phone",
        "word_coverage","task_phone_graded_accuracy"),
 "A15":("random_phone_edit_rate","random_posterior_entropy",
        "random_target_logp_per_phone"),
 "A16":("child_target_logp_per_phone","child_posterior_entropy",
        "child_aligned_phone_confidence","child_phone_weighted_edit_rate",
        "task_phone_graded_accuracy","task_phone_omission_rate",
        "word_coverage","word_lcs_ratio","clip_pred_correct",
        "clip_unscored_recovery"),
}
PHONETIC_MANDATORY={"A00","A01","A02","A03","A05","A06","A08","A09",
                     "A10","A11","A12","A13","A14","A15","A16"}
WORD_REQUIRED={"A07"}
CORRECTNESS_REQUIRED={"A10","A11","A16"}
RELIABILITY_FAMILY={"A12","A16"}
CORRECTNESS_ONLY={"A06"}
LAYER_FAMILY={"A13"}

class SpeechAggregator:
    def __init__(self,families=("A16",),weighted=False,student_loo=True,
                 min_item_support=5):
        self.families=tuple(families)
        if not set(self.families).issubset(FAMILIES):raise ValueError("Unknown family")
        self.names=tuple(dict.fromkeys(k for family in self.families
                                      for k in FAMILIES[family]))
        self.weighted=bool(weighted)
        self.student_loo=bool(student_loo)
        self.min_item_support=int(min_item_support)

    def fit(self,train_clips):
        d=train_clips.copy()
        self.train_students=set(d.subject_id.astype(str))
        self.train_order=d.uid.astype(str).tolist()
        self.task_stats={};self.item_stats={};self.self_task={};self.self_item={}
        numeric=d[list(self.names)].apply(pd.to_numeric,errors="coerce") if all(
            col in d.columns for col in self.names) else pd.DataFrame({
            col:pd.to_numeric(d[col],errors="coerce") if col in d else np.nan
            for col in self.names},index=d.index)
        self.cols=tuple(numeric.columns)
        # store per-column count/sum/square, excluding NaN
        for task,g in d.groupby("task",sort=False):
            idx=g.index
            self.task_stats[str(task)]=self._moments(numeric.loc[idx])
        for (task,prompt),g in d.groupby(["task","expected_text"],dropna=False):
            self.item_stats[(str(task),str(prompt))]=self._moments(numeric.loc[g.index])
        for (sid,task),g in d.groupby(["subject_id","task"],sort=False):
            self.self_task[(str(sid),str(task))]=self._moments(numeric.loc[g.index])
        for (sid,task,prompt),g in d.groupby(
            ["subject_id","task","expected_text"],dropna=False):
            self.self_item[(str(sid),str(task),str(prompt))]=self._moments(numeric.loc[g.index])
        return self

    def _moments(self,x):
        vals=x.to_numpy(float)
        okay=np.isfinite(vals)
        return np.stack([okay.sum(axis=0),
            np.nansum(vals,axis=0),np.nansum(np.where(okay,vals*vals,0.),axis=0)],axis=0)

    def _norm(self,name,task,prompt,sid,loo):
        j=self.cols.index(name)
        taskmom=self.task_stats.get(task)
        itemmom=self.item_stats.get((task,prompt))
        tm=np.zeros((3,len(self.cols))) if taskmom is None else taskmom.copy()
        im=np.zeros((3,len(self.cols))) if itemmom is None else itemmom.copy()
        if loo:
            tm-=self.self_task.get((sid,task),0)
            im-=self.self_item.get((sid,task,prompt),0)
        if im[0,j]<self.min_item_support:
            n,s,ss=tm[:,j]
        else:n,s,ss=im[:,j]
        mean=s/n if n>0 else 0.
        var=max(ss/n-mean*mean,0.) if n>0 else 1.
        std=max(np.sqrt(var),.2)
        return mean,std

    def _features(self,students,clips,loo):
        if not hasattr(self,"cols"):raise RuntimeError("Fit aggregator first")
        ids=students.index.astype(str).tolist()
        if loo and not set(ids).issubset(self.train_students):
            raise ValueError("Training LOO cannot include validation/test students")
        if not loo and set(ids)&self.train_students:
            raise ValueError("Inference contains fitted training student")
        d=clips[clips.status=="VALID"].copy()
        by_student={str(s):g for s,g in d.groupby("subject_id",sort=False)}
        empty=d.iloc[:0]
        cols=self.cols
        rows=[]
        for sid in ids:
            g=by_student.get(sid,empty)
            out={}
            for task in ("blending","deletion","sentence_repetition"):
                sub=g[g.task==task]
                full=clips[(clips.subject_id.astype(str)==sid)&(clips.task==task)]
                out[f"speech_{task}_n_valid"]=float(len(sub))
                out[f"speech_{task}_fail_frac"]=float(
                    1-len(sub)/len(full)) if len(full) else 0.
                out[f"speech_{task}_n_unscored"]=float(pd.to_numeric(
                    full.item_score,errors="coerce").isna().sum()) if "item_score" in full else 0.
                if "localization_confidence" in sub:
                    out[f"speech_{task}_loc_conf_mean"]=float(pd.to_numeric(
                        sub.localization_confidence,errors="coerce").mean()) if len(sub) else 0.
                for name in cols:
                    vals=[];weights=[];statuses=[]
                    for row in sub.itertuples(index=False):
                        value=getattr(row,name,np.nan)
                        try:value=float(value)
                        except (ValueError,TypeError):value=np.nan
                        if not np.isfinite(value):continue
                        mean,std=self._norm(name,task,str(row.expected_text),sid,loo)
                        norm=float(np.clip((value-mean)/std,-10,10))
                        confidence=float(getattr(row,"localization_confidence",.5))
                        rms=float(getattr(row,"rms",.03))
                        w=max(.05,min(1.,confidence))*np.clip(rms/.06,.1,1.)
                        vals.append(norm);weights.append(w)
                        try:statuses.append(float(getattr(row,"item_score",np.nan)))
                        except (ValueError,TypeError):statuses.append(np.nan)
                    vals=np.asarray(vals,float);weights=np.asarray(weights,float)
                    yy=np.asarray(statuses,float)
                    prefix=f"speech_{task}_{name}"
                    if len(vals):
                        out[prefix+"_mean"]=float(np.average(vals,weights=weights)
                            if self.weighted else np.mean(vals))
                        out[prefix+"_std"]=float(np.std(vals))
                        out[prefix+"_observed"]=float(len(vals))
                        for label,mask in (("correct",yy>=.5),
                                           ("incorrect",np.isfinite(yy)&(yy<.5))):
                            out[prefix+"_"+label]=float(np.mean(vals[mask])) if mask.any() else 0.
                    else:
                        out[prefix+"_mean"]=np.nan
                        out[prefix+"_std"]=np.nan
                        out[prefix+"_observed"]=0.
                        out[prefix+"_correct"]=0.
                        out[prefix+"_incorrect"]=0.
            rows.append(out)
        return pd.DataFrame(rows,index=students.index)

    def transform_train(self,students,clips):
        return self._features(students,clips,self.student_loo)
    def transform(self,students,clips):
        return self._features(students,clips,False)

def available_families(clips):
    """Never run a phonetic or word experiment with missing required features."""
    present=set(clips)
    eligible={}
    for arm,names in FAMILIES.items():
        have=sum(c in present and pd.to_numeric(clips[c],errors="coerce").notna().any()
                 for c in names)
        eligible[arm]={"available_columns":have,"requested_columns":len(names),
                      "eligible":bool(have>=1 and
                         (arm not in WORD_REQUIRED or "word_coverage" in present) and
                         (arm not in CORRECTNESS_REQUIRED or "clip_pred_correct" in present))}
    return eligible
