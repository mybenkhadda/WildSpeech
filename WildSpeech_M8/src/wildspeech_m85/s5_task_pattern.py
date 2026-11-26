"""S5: joint administration patterns and item-score availability, student-local."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import TASKS


def task_pattern_features(students,clips):
    scored=clips.copy()
    scored["_scored"]=pd.to_numeric(scored.item_score,errors="coerce").notna().astype(int)
    grp=scored.groupby(["subject_id","task"]).agg(n_clips=("task","size"),
                                               n_scored=("_scored","sum"))
    n=len(students)
    result={}
    admin=[];scoring=[];theta_available=[]
    for task in TASKS:
        keys=pd.MultiIndex.from_arrays([students.index.astype(str),[task]*n],
                                       names=["subject_id","task"])
        vals=grp.reindex(keys).fillna(0)
        nclips=vals.n_clips.to_numpy(float)
        nscored=vals.n_scored.to_numpy(float)
        th=pd.to_numeric(students[task+"_theta"],errors="coerce").to_numpy(float)
        available=np.isfinite(th)
        given=nclips>0
        has_score=nscored>0
        result[f"s5_{task}_administered"]=given.astype(float)
        result[f"s5_{task}_has_score"]=has_score.astype(float)
        result[f"s5_{task}_unscored_fraction"]=np.divide(
            nclips-nscored,nclips,out=np.zeros(n),where=nclips>0)
        result[f"s5_{task}_administered_no_scoring"]=(given&~has_score).astype(float)
        result[f"s5_{task}_scored_theta_missing"]=(has_score&~available).astype(float)
        result[f"s5_{task}_theta_without_score"]=(available&~has_score).astype(float)
        admin.append(given);scoring.append(has_score);theta_available.append(available)
    mask=np.zeros(n,dtype=int)
    for i,v in enumerate(admin):
        mask+=((v.astype(int))<<i)
    for i in range(8):
        result[f"s5_admin_pattern_{i:03b}"]=(mask==i).astype(float)
    for j in range(3):
        for k in range(j+1,3):
            result[f"s5_pair_{TASKS[j]}_{TASKS[k]}"]=(admin[j]&admin[k]).astype(float)
    result["s5_administered_tasks"]=np.sum(admin,axis=0).astype(float)
    result["s5_scored_tasks"]=np.sum(scoring,axis=0).astype(float)
    result["s5_theta_tasks"]=np.sum(theta_available,axis=0).astype(float)
    return pd.DataFrame(result,index=students.index)
