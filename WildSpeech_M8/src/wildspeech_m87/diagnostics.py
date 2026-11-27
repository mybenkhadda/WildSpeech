"""M8.7 research figures and tables, all in the PRIVATE artifact directory.

No fake scores, no public export of individual child/school data. Figures
mark model-derived proxies, do not imply manual verified child phonetics.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from wildspeech_m84.data import atomic_json,atomic_text

def _save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)

def extraction_atlas(root,inventory,features):
    root=Path(root);fig=root/"figures";table=root/"tables"
    fig.mkdir(parents=True,exist_ok=True);table.mkdir(parents=True,exist_ok=True)
    d=inventory.merge(features.drop(columns=[c for c in
        ("subject_id","task","split","filename","item_score","expected_text")
        if c in features],errors="ignore"),on="uid",how="left",validate="one_to_one")
    if len(d)!=len(inventory):raise RuntimeError("Incomplete figure/inventory join")
    status=d.status.fillna("NOT_EXTRACTED")
    by=d.assign(cache_status=status).groupby(
        ["split","task","cache_status"]).size().rename("n").reset_index()
    by.to_csv(table/"01_coverage_by_task_PRIVATE.csv",index=False)
    figure,ax=plt.subplots(figsize=(10,4))
    by.pivot_table(index=["split","task"],columns="cache_status",values="n",
                   fill_value=0).plot(kind="bar",stacked=True,ax=ax)
    ax.set(ylabel="Clip count",xlabel="Split / task",title="C3: complete accounting of audio inventory")
    _save(figure,fig/"01_cache_coverage.png")
    valid=d[d.status=="VALID"].copy()
    if valid.empty:
        atomic_json(table/"extraction_atlas_summary.json",
                    {"valid":0,"failed":int(len(d))})
        return
    figure,ax=plt.subplots(figsize=(9,4))
    for task,group in valid.groupby("task"):
        ax.hist(group.duration_s.dropna(),bins=36,alpha=.45,label=task)
    ax.set(xlabel="Decoded duration (seconds)",ylabel="Clips",
           title="Recording duration distribution (not child response latency)")
    ax.legend();_save(figure,fig/"02_duration_by_task.png")
    figure,ax=plt.subplots(figsize=(8,4))
    for task,g in valid.groupby("task"):
        ax.hist(g.localization_confidence.dropna(),bins=np.linspace(0,1,26),
                density=True,alpha=.45,label=task)
    ax.set(xlabel="Uncalibrated localizer confidence",ylabel="Density",
           title="C5: task-specific selection confidence")
    ax.legend();_save(figure,fig/"03_localization_confidence.png")
    figure,axs=plt.subplots(1,2,figsize=(11,4))
    for j,column in enumerate(("whole_phone_edit_rate","child_phone_edit_rate")):
        if column in valid:
            for task,g in valid.groupby("task"):
                axs[j].hist(pd.to_numeric(g[column],errors="coerce").dropna().clip(0,3),
                            bins=25,density=True,alpha=.45,label=task)
        axs[j].set(xlabel="Phone edit rate (0–3 clipped)",ylabel="Density",
                    title=("Whole clip" if j==0 else "Chosen region"))
        axs[j].legend()
    _save(figure,fig/"04_whole_vs_segment_phone_edits.png")
    figure,ax=plt.subplots(figsize=(9,4))
    if all(x in valid for x in ("whole_target_logp_per_phone",
                                 "child_target_logp_per_phone")):
        a=pd.to_numeric(valid.whole_target_logp_per_phone,errors="coerce")
        b=pd.to_numeric(valid.child_target_logp_per_phone,errors="coerce")
        diff=(b-a).replace([np.inf,-np.inf],np.nan).dropna().clip(-10,10)
        ax.hist(diff,bins=35)
        ax.axvline(0,ls="--",color="black")
    ax.set(xlabel="Segment − whole target CTC score per target token",
           ylabel="Clips",title="A2: effect of candidate localization")
    _save(figure,fig/"05_ctc_segment_delta.png")
    if "item_score" in valid:
        score=pd.to_numeric(valid.item_score,errors="coerce")
        figure,ax=plt.subplots(figsize=(8,4))
        for group,label in ((score>=.5,"Scored correct"),
                            (score<.5,"Scored incorrect")):
            subset=valid.loc[group]
            if "task_phone_graded_accuracy" in subset:
                x=pd.to_numeric(subset.task_phone_graded_accuracy,errors="coerce")
                ax.hist(x.dropna().clip(0,1),bins=20,alpha=.5,density=True,label=label)
        ax.set(xlabel="ASR-derived phonetic graded accuracy",ylabel="Density",
               title="Within-correctness speech evidence (not human transcriptions)")
        ax.legend();_save(figure,fig/"06_grading_within_correctness.png")
    # Full feature schema is private and gives model implementers a typed source.
    audit=[]
    for name in valid.select_dtypes(include=[np.number]).columns:
        z=pd.to_numeric(valid[name],errors="coerce").to_numpy(float)
        audit.append({"feature":name,"nonmissing_frac":float(np.isfinite(z).mean()),
                      "std":float(np.nanstd(z)) if np.isfinite(z).any() else None})
    pd.DataFrame(audit).to_csv(table/"02_clip_feature_coverage_PRIVATE.csv",index=False)
    summary={"clips_expected":len(d),"clips_valid":len(valid),
             "clips_failed":int((status=="FAILED").sum()),
             "target_mapping_fraction":float(valid.target_valid.mean())
             if "target_valid" in valid else None,
             "mean_localization_confidence":float(valid.localization_confidence.mean()),
             "ambiguous_fraction":float(valid.overlap_or_ambiguity.mean()),
             "no_response_flag_fraction":float(valid.no_child_response.mean())}
    atomic_json(table/"extraction_atlas_summary.json",summary)
    return summary

def prediction_atlas(root,metrics,comparisons,baseline_oof=None,candidates=None,
                     data=None,clip_features=None):
    root=Path(root);fig=root/"figures";table=root/"tables"
    fig.mkdir(parents=True,exist_ok=True);table.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(table/"03_experiment_metrics.csv",index=False)
    if frame.empty:return
    figure,ax=plt.subplots(figsize=(13,5))
    x=np.arange(len(frame))
    ax.bar(x-.2,frame.student_pooled_r2.to_numpy(float),width=.4,label="Student pooled")
    ax.bar(x+.2,frame.school_pooled_r2.to_numpy(float),width=.4,label="School pooled")
    ax.set(xticks=x,xticklabels=frame.model,ylim=(min(0,frame.school_pooled_r2.min()-.05),
           max(.28,frame.student_pooled_r2.max()+.04)),
           ylabel="OOF R²",title="A0–A16: student vs held-out-school generalization")
    ax.legend();_save(figure,fig/"07_model_student_school_r2.png")
    compare=pd.DataFrame([{"model":c["candidate"],
                           "mean_fold_delta_r2":c["result"]["mean_fold_delta_r2"],
                           "school_pooled_delta_r2":c["result"]["school_pooled_delta_r2"],
                           "win_fraction":c["result"]["win_fraction"],
                           "corrected_t":c["result"]["corrected_t"]}
                          for c in comparisons])
    compare.to_csv(table/"04_paired_comparisons.csv",index=False)
    if len(compare):
        figure,ax=plt.subplots(figsize=(12,4))
        ax.bar(compare.model,compare.mean_fold_delta_r2)
        ax.axhline(0,color="black",lw=1)
        ax.axhline(.01,color="gray",ls="--",label="+.010 science magnitude")
        ax.set(ylabel="Mean paired fold ΔR²",xlabel="Candidate",
               title="Paired improvements against frozen I5 (not public leaderboard)")
        ax.legend();_save(figure,fig/"08_paired_delta_r2.png")
        figure,ax=plt.subplots(figsize=(10,5))
        ax.scatter(compare.mean_fold_delta_r2,compare.school_pooled_delta_r2,s=40)
        for row in compare.itertuples():
            ax.annotate(row.model,(row.mean_fold_delta_r2,row.school_pooled_delta_r2),
                        xytext=(3,4),textcoords="offset points",fontsize=8)
        ax.axhline(-.005,ls="--",color="gray")
        ax.axvline(0,ls="-",color="gray")
        ax.set(xlabel="Student mean fold ΔR²",ylabel="School pooled ΔR²",
               title="Transfer frontier: useful speech vs school shortcuts")
        _save(figure,fig/"09_transfer_frontier.png")
    if baseline_oof is not None and data is not None:
        oof=baseline_oof[baseline_oof.scheme=="student"]
        student=oof.groupby("pos").agg(y_true=("y_true","first"),
                                        y_pred=("y_pred","mean"))
        student["residual"]=student.y_true-student.y_pred
        student["instrument"]=data.instruments[student.index.to_numpy(int)]
        student.to_csv(table/"05_student_oof_residuals_PRIVATE.csv")
        figure,ax=plt.subplots(figsize=(8,5))
        for instrument,s in student.groupby("instrument"):
            ax.scatter(s.y_pred,s.y_true,alpha=.5,s=14,label=instrument)
        ax.plot([60,150],[60,150],ls="--",color="black")
        ax.set(xlabel="Frozen I5 mean OOF prediction",ylabel="True reading score",
               title="Baseline regression compression and instrument differences")
        ax.legend();_save(figure,fig/"10_baseline_prediction_scatter.png")
    if clip_features is not None and len(clip_features):
        if all(c in clip_features for c in
               ("child_phone_edit_rate","item_score","task")):
            d=clip_features.copy()
            d["item_score"]=pd.to_numeric(d.item_score,errors="coerce")
            d["edit"]=pd.to_numeric(d.child_phone_edit_rate,errors="coerce")
            rows=[]
            for (task,correct),g in d[d.item_score.notna()].groupby(
                ["task",d.item_score>=.5]):
                rows.append({"task":task,"correct":bool(correct),"n":len(g),
                             "mean_child_phone_edit":float(g.edit.mean())})
            pd.DataFrame(rows).to_csv(table/"06_within_correctness_PRIVATE.csv",index=False)

def qa_training_waveforms(root,inventory,features,audio_source,
                          n=12,seed=42):
    """Training clips ONLY; never manually inspect test waveform."""
    from .audio_io import AudioSource
    root=Path(root);qa=root/"figures"/"qa_training"
    qa.mkdir(parents=True,exist_ok=True)
    d=inventory[inventory.split=="train"].merge(
        features[["uid","status","response_start_s","response_end_s"]],
        on="uid",how="inner",validate="one_to_one")
    d=d[d.status=="VALID"].copy()
    if d.empty:return 0
    d["score_stratum"]=np.where(pd.to_numeric(d.item_score,errors="coerce")>=.5,
                               "correct","incorrect")
    rng=np.random.default_rng(seed)
    samples=[]
    for (_,group) in d.groupby(["task","score_stratum"]):
        samples.extend(group.sample(min(2,len(group)),
                                   random_state=seed).to_dict("records"))
    for idx,row in enumerate(samples[:n]):
        wave,sr,_,_=audio_source.decode(row["filename"])
        times=np.arange(len(wave))/sr
        f,ax=plt.subplots(figsize=(10,2.5))
        # Display envelope only, not speaker identity or recording data publicly.
        ax.plot(times,wave,lw=.3)
        ax.axvspan(row["response_start_s"],row["response_end_s"],
                   alpha=.15,label="predicted response region")
        ax.set(xlabel="Seconds",ylabel="Amplitude",
               title=f"Training QA {idx+1}: {row['task']} ({row['score_stratum']})")
        ax.legend();_save(f,qa/f"train_clip_{idx+1:02d}.png")
    return min(len(samples),n)
