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
        # Vectorized 2-D per-row normalization: one NumPy array operation per
        # observed student×item group, not a Python loop over every feature
        # of every clip of every student.
        d=clips.loc[clips.status=="VALID"].copy()
        values=pd.DataFrame({k:pd.to_numeric(d[k],errors="coerce") if k in d
                             else np.nan for k in self.cols},index=d.index)
        raw=values.to_numpy(float)
        n,k=raw.shape
        means=np.zeros((n,k),float)
        stds=np.ones((n,k),float)
        if n:
            group_index=d.groupby(["task","expected_text","subject_id"],
                                  dropna=False,sort=False).indices
            blank=np.zeros((3,k),float)
            for (task,prompt,sid),positions in group_index.items():
                task=str(task);prompt=str(prompt);sid=str(sid)
                tm=self.task_stats.get(task,blank).copy()
                im=self.item_stats.get((task,prompt),blank).copy()
                if loo:
                    tm-=self.self_task.get((sid,task),0)
                    im-=self.self_item.get((sid,task,prompt),0)
                use=np.where(im[0]>=self.min_item_support,im,tm)
                count=use[0]
                mean=np.divide(use[1],count,out=np.zeros(k,float),where=count>0)
                var=np.divide(use[2],count,out=np.ones(k,float),where=count>0)-mean*mean
                std=np.maximum(np.sqrt(np.maximum(0.,var)),.2)
                means[positions,:]=mean
                stds[positions,:]=std
        norm=np.where(np.isfinite(raw),np.clip((raw-means)/stds,-10,10),np.nan)
        norm_df=pd.DataFrame(norm,index=d.index,columns=self.cols)
        norm_df["subject_id"]=d.subject_id.astype(str)
        norm_df["task"]=d.task.astype(str)
        score=pd.to_numeric(d.item_score,errors="coerce") if "item_score" in d else pd.Series(np.nan,index=d.index)
        norm_df["correct"]=score>=.5
        norm_df["incorrect"]=score.notna()&(score<.5)
        conf=pd.to_numeric(d["localization_confidence"],errors="coerce") if "localization_confidence" in d else pd.Series(.5,index=d.index)
        rms=pd.to_numeric(d["rms"],errors="coerce") if "rms" in d else pd.Series(.03,index=d.index)
        norm_df["_weight"]=(conf.fillna(.05).clip(.05,1.)*
                            (rms.fillna(0).clip(lower=0)/.06).clip(.1,1.))
        full=clips.copy()
        full["subject_id"]=full.subject_id.astype(str)
        full["task"]=full.task.astype(str)
        full_score=pd.to_numeric(full.item_score,errors="coerce") if "item_score" in full else pd.Series(np.nan,index=full.index)
        full["_unscored"]=full_score.isna()
        full_counts=full.groupby(["subject_id","task"]).size()
        unscored=full.groupby(["subject_id","task"])["_unscored"].sum()
        valid_counts=d.groupby(["subject_id","task"]).size()
        result={}  # build column arrays once; avoid 2,000 fragmented inserts
        for task in ("blending","deletion","sentence_repetition"):
            part=norm_df[norm_df.task==task]
            by=part.groupby("subject_id",sort=False)
            if len(part):
                st=by[list(self.cols)].std(ddof=0).reindex(ids)
                ncol=by[list(self.cols)].count().reindex(ids).fillna(0)
                corr=part[part.correct].groupby("subject_id")[list(self.cols)].mean().reindex(ids).fillna(0)
                wrong=part[part.incorrect].groupby("subject_id")[list(self.cols)].mean().reindex(ids).fillna(0)
                if self.weighted:
                    weights=part["_weight"].to_numpy(float)
                    a=part[list(self.cols)]
                    num=a.mul(weights,axis=0).groupby(part.subject_id).sum(min_count=1).reindex(ids)
                    den=pd.DataFrame(np.where(np.isfinite(a.to_numpy(float)),
                                  weights[:,None],0.),
                                  columns=self.cols,index=part.index).groupby(
                                  part.subject_id).sum().reindex(ids).fillna(0)
                    mu=num.div(den.where(den>0))
                else:
                    mu=by[list(self.cols)].mean().reindex(ids)
                confidence=by["_weight"].mean().reindex(ids).fillna(0)
            else:
                st=mu=pd.DataFrame(np.nan,index=ids,columns=self.cols)
                ncol=corr=wrong=pd.DataFrame(0.,index=ids,columns=self.cols)
                confidence=pd.Series(0.,index=ids)
            valid=valid_counts.reindex(
                pd.MultiIndex.from_product([ids,[task]])).fillna(0).to_numpy(float)
            total=full_counts.reindex(
                pd.MultiIndex.from_product([ids,[task]])).fillna(0).to_numpy(float)
            us=unscored.reindex(
                pd.MultiIndex.from_product([ids,[task]])).fillna(0).to_numpy(float)
            result[f"speech_{task}_n_valid"]=valid
            result[f"speech_{task}_fail_frac"]=np.divide(total-valid,total,
                                         out=np.zeros(len(ids)),where=total>0)
            result[f"speech_{task}_n_unscored"]=us
            result[f"speech_{task}_loc_conf_mean"]=confidence.to_numpy(float)
            for key in self.cols:
                prefix=f"speech_{task}_{key}"
                result[prefix+"_mean"]=mu[key].to_numpy(float)
                result[prefix+"_std"]=st[key].to_numpy(float)
                result[prefix+"_observed"]=ncol[key].to_numpy(float)
                result[prefix+"_correct"]=corr[key].to_numpy(float)
                result[prefix+"_incorrect"]=wrong[key].to_numpy(float)
        return pd.DataFrame(result,index=students.index)

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
                         (arm not in CORRECTNESS_REQUIRED or "child_phone_edit_rate" in present))}
    return eligible
