"""Private, evidence-linked M8.6 research atlas.

Every row-level student/school statistic stays on private Google Drive. Plot
school cells only when n>=5. No training labels are used to fit item features.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wildspeech_m84.data import atomic_json,atomic_text
from .contracts import TASKS
from .item_complexity import text_complexity

def diagnostic_tables(root,data,reference,baseline):
    root=Path(root);fig=root/"figures";fig.mkdir(parents=True,exist_ok=True)
    tables=root/"tables";tables.mkdir(parents=True,exist_ok=True)
    students=data.train_students.copy()
    students["outcome_score"]=data.labels.reindex(students.index).to_numpy(float)
    students["instrument"]=students.instrument.astype(str)
    students["school_id"]=students.school_id.astype(str)
    oof=baseline[baseline.repeat==0].copy()
    a=oof[oof.scheme=="student"].sort_values("pos")
    b=oof[oof.scheme=="school"].sort_values("pos")
    y=data.labels.to_numpy(float)
    out=[]
    for name,subset in (("student",a),("school",b)):
        pos=subset.pos.astype(int).to_numpy()
        error=y[pos]-subset.y_pred.to_numpy(float)
        tmp=pd.DataFrame({"school_id":data.groups[pos],
                          "instrument":data.instruments[pos],
                          "abs_error":np.abs(error),"signed_error":error,
                          "squared_error":np.square(error)})
        grouped=tmp.groupby(["school_id","instrument"]).agg(
            n=("abs_error","size"),mae=("abs_error","mean"),
            bias=("signed_error","mean"),sse=("squared_error","sum")).reset_index()
        grouped["scheme"]=name
        out.append(grouped)
    school=pd.concat(out,ignore_index=True)
    school.to_csv(tables/"school_instrument_errors_PRIVATE.csv",index=False)
    shared=school[school.n>=5]
    agg=shared.groupby(["scheme","instrument"]).agg(
        n=("n","sum"),mae=("mae","mean"),
        avg_bias=("bias","mean")).reset_index()
    agg.to_csv(tables/"instrument_transfer_suppressed.csv",index=False)
    clip_data=data.train_clips.copy()
    clip_data["has_score"]=pd.to_numeric(clip_data.item_score,errors="coerce").notna()
    coverage=clip_data.groupby(["task","subject_id"]).agg(
        n=("task","size"),scored=("has_score","sum"))
    patterns=[]
    for task in TASKS:
        s=students[[task+"_theta","instrument"]].copy()
        values=coverage.loc[task] if task in coverage.index.get_level_values(0) else pd.DataFrame()
        n=values["n"].reindex(s.index).fillna(0) if not values.empty else pd.Series(0,index=s.index)
        s["administered"]=n>0
        s["theta_present"]=s[task+"_theta"].notna()
        pat=s.groupby("instrument")[["administered","theta_present"]].mean().reset_index()
        pat["task"]=task
        patterns.append(pat)
    presence=pd.concat(patterns,ignore_index=True)
    presence.to_csv(tables/"task_missingness_by_instrument.csv",index=False)
    # Distribution of item support and correctness, not score-based selection.
    d=clip_data[["task","expected_text","subject_id","item_score"]].copy()
    d["prompt"]=(d.task.astype(str)+"\x1f"+
                 d.expected_text.fillna("").astype(str).str.casefold().str.strip())
    item=d.groupby("prompt").agg(
        task=("task","first"),n_students=("subject_id","nunique"),
        n_clips=("subject_id","size"),
        item_correct=("item_score","mean")).reset_index()
    item.to_csv(tables/"item_support_PRIVATE.csv",index=False)
    summary={
        "n_item_keys":int(len(item)),
        "n_rare_le_2":int((item.n_students<=2).sum()),
        "n_schools":int(students.school_id.nunique()),
        "n_instruments":int(students.instrument.nunique()),
        "n_students":len(students),
        "school_min":int(students.school_id.value_counts().min()),
        "school_max":int(students.school_id.value_counts().max()),
    }
    atomic_json(tables/"diagnostic_summary.json",summary)
    figure,ax=plt.subplots(figsize=(8,4))
    x=item.n_students.clip(upper=60)
    ax.hist(x,bins=25,alpha=.85)
    ax.set(xlabel="Number of training students exposed to prompt (clipped 60)",
           ylabel="Prompt keys",title="Item support: training metadata only")
    figure.tight_layout();figure.savefig(fig/"01_item_support.png",dpi=160)
    plt.close(figure)
    figure,ax=plt.subplots(figsize=(9,4))
    for inst,g in presence.groupby("instrument"):
        ax.plot(range(3),g.set_index("task").reindex(TASKS).administered.values,
                marker="o",label=str(inst))
    ax.set(xticks=range(3),xticklabels=["Blending","Deletion","Sentence repetition"],
           ylim=(0,1.05),ylabel="Administered share",title="Task missingness by instrument")
    ax.legend();figure.tight_layout()
    figure.savefig(fig/"02_task_admin_by_instrument.png",dpi=160)
    plt.close(figure)
    atomic_text(root/"reports"/"M86_DIAGNOSTICS.md",
      "# M8.6 – Screening, item and instrument atlas\n\n"
      "All student/school-level records remain in private Drive.\n\n"
      +pd.DataFrame([summary]).to_markdown(index=False)+"\n\n"
      "## Task administration\n\n"+presence.to_markdown(index=False)+"\n\n"
      "## Suppressed school-by-instrument errors (minimum n=5)\n\n"
      +agg.to_markdown(index=False)+"\n\n"
      "**Caveats:** observational associations; no inferred item chronology; "
      "clip duration is not child response delay.\n")
    return summary

def score_plots(root,results):
    root=Path(root);fig=root/"figures";fig.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(results)
    if not len(df):return
    plot=df.sort_values("student_pooled_r2",ascending=False)
    f,ax=plt.subplots(figsize=(12,5))
    ax.bar(plot.model,plot.student_pooled_r2)
    baseline=df.loc[df.model=="E04","student_pooled_r2"]
    if not baseline.empty:ax.axhline(baseline.iloc[0],ls="--",color="black",label="Frozen E04")
    ax.set(ylabel="10-repeat student pooled OOF R²",xlabel="Experiment",
           title="M8.6 candidate comparison")
    ax.legend();f.tight_layout();f.savefig(fig/"03_model_performance.png",dpi=160)
    plt.close(f)
    if "paired_fold_delta_r2" in df:
        tmp=df[df.model!="E04"]
        f,ax=plt.subplots(figsize=(12,5))
        ax.bar(tmp.model,tmp.paired_fold_delta_r2)
        ax.axhline(0.,color="black",lw=1)
        ax.axhline(.01,color="gray",ls="--",label="Science ΔR² magnitude gate")
        ax.set(xlabel="Experiment",ylabel="Mean paired 50-fold ΔR²",
               title="Paired improvements against E04, not leaderboard R²")
        ax.legend();f.tight_layout();f.savefig(fig/"04_paired_deltas.png",dpi=160)
        plt.close(f)
    f,ax=plt.subplots(figsize=(8,5))
    ax.scatter(df.student_pooled_r2,df.school_pooled_r2,s=45)
    for r in df.itertuples():
        ax.annotate(r.model,(r.student_pooled_r2,r.school_pooled_r2),
                    xytext=(4,3),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Student pooled OOF R²",ylabel="School-disjoint pooled OOF R²",
           title="Predictive accuracy vs school transfer")
    f.tight_layout();f.savefig(fig/"05_school_transfer.png",dpi=160)
    plt.close(f)
