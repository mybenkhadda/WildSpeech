"""Stratified training-only human QA; never samples or exposes test recordings."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from wildspeech_m84.data import atomic_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_template(data,inventory,features,output,n=150,seed=42):
    clips=inventory[inventory.split=="train"].copy()
    clips=clips.merge(features[["uid","status","response_start_s",
                               "response_end_s","localization_confidence"]],
                      on="uid",how="left",validate="one_to_one")
    clips["instrument"]=clips.subject_id.map(data.train_students.instrument)
    score=pd.to_numeric(clips.item_score,errors="coerce")
    clips["score_group"]=np.select([score>=.5,score<.5],
                                   ["correct","incorrect"],default="unscored")
    eligible=clips[clips.status=="VALID"]
    if len(eligible)==0:raise RuntimeError("No successful training audio")
    rng=np.random.default_rng(seed)
    groups=list(eligible.groupby(["task","instrument","score_group"],dropna=False))
    per=max(1,int(np.ceil(n/max(1,len(groups)))))
    sampled=pd.concat([g.sample(min(per,len(g)),random_state=seed)
                       for _,g in groups],ignore_index=True)
    if len(sampled)>n:sampled=sampled.sample(n,random_state=seed)
    sample=sampled.sort_values(["task","instrument","score_group","uid"]).copy()
    sample["human_child_present"]=pd.NA
    sample["human_start_s"]=np.nan
    sample["human_end_s"]=np.nan
    sample["human_adult_selected"]=pd.NA
    sample["human_multiple_attempts"]=pd.NA
    sample["review_note"]=""
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    sample.to_csv(path,index=False)
    return sample

def evaluate_qa(table,output):
    d=pd.read_csv(table)
    if "human_child_present" not in d:raise ValueError("Missing review fields")
    reviewed=d.human_child_present.notna()
    if not reviewed.any():
        return {"status":"NOT_REVIEWED","reviewed":0,
                "localization_gate_passed":False}
    x=d.loc[reviewed].copy()
    present=x.human_child_present.astype(str).str.lower().isin(("true","1","yes"))
    adult=x.human_adult_selected.astype(str).str.lower().isin(("true","1","yes"))
    start=pd.to_numeric(x.human_start_s,errors="coerce").to_numpy(float)
    end=pd.to_numeric(x.human_end_s,errors="coerce").to_numpy(float)
    pred_s=x.response_start_s.to_numpy(float)
    pred_e=x.response_end_s.to_numpy(float)
    intersects=np.maximum(0,np.minimum(end,pred_e)-np.maximum(start,pred_s))
    union=np.maximum(end,pred_e)-np.minimum(start,pred_s)
    iou=np.where((union>0)&np.isfinite(union),intersects/np.maximum(union,1e-6),np.nan)
    okay=present.to_numpy(bool)&np.isfinite(iou)&(iou>=.5)&(~adult.to_numpy(bool))
    correct=np.where(present.to_numpy(bool),okay,~present.to_numpy(bool))
    per=x.assign(correct=correct,present=present,adult=adult,iou=iou).groupby(
        ["task","instrument"]).agg(
            n=("correct","size"),accuracy=("correct","mean"),
            false_adult=("adult","mean"),mean_iou=("iou","mean")).reset_index()
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    per.to_csv(path/"localization_qa_by_group_PRIVATE.csv",index=False)
    score=float(np.mean(correct))
    worst=float(per[per.n>=5].accuracy.min()) if (per.n>=5).any() else np.nan
    summary={"status":"REVIEWED","reviewed":len(x),"accuracy":score,
             "mean_iou":float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
             "false_adult_frac":float(adult.mean()),
             "worst_group_accuracy_n5":worst,
             "localization_gate_passed":bool(
                 score>=.8 and np.isfinite(worst) and worst>=.65
                 and float(adult.mean())<=.1)}
    atomic_json(path/"localization_qa_results.json",summary)
    f,ax=plt.subplots(figsize=(9,4))
    ax.bar(range(len(per)),per.accuracy)
    ax.set(xticks=range(len(per)),
           xticklabels=[str(a)+"/"+str(b) for a,b in zip(per.task,per.instrument)],
           ylim=(0,1),ylabel="Training QA answer localization accuracy",
           title="Human reviewed training clips only")
    ax.tick_params(axis="x",rotation=40)
    f.tight_layout();f.savefig(path/"localization_qa_accuracy.png",dpi=160);plt.close(f)
    return summary
