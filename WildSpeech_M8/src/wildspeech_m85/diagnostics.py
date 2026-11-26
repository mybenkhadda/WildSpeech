"""Private school/instrument, missing-task and item-identity research atlas.

All row-level data remain in the user's private Drive. Plots suppress school ×
instrument counts below five; no school or student ID becomes a model feature.
"""
from __future__ import annotations
from collections import Counter
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from .contracts import TASKS,normalize_prompt,item_contract
from wildspeech_m84.data import atomic_json,atomic_text


def run_diagnostics(data,reference_oof,output_root,min_cell=5):
    root=Path(output_root);path=root/"diagnostics";fig=root/"figures"
    path.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    ids=data.ids
    tr=data.train_students.copy()
    tr["outcome_score"]=data.labels.reindex(tr.index).to_numpy(float)
    tr["school_id"]=tr.school_id.astype(str)
    tr["instrument"]=tr.instrument.astype(str)
    for task in TASKS:
        clips=data.train_clips.loc[data.train_clips.task==task]
        counts=clips.groupby("subject_id").size().reindex(tr.index).fillna(0)
        score=clips.assign(scored=pd.to_numeric(clips.item_score,errors="coerce").notna()).groupby(
            "subject_id").scored.sum().reindex(tr.index).fillna(0)
        tr[f"{task}_given"]=(counts>0).astype(int)
        tr[f"{task}_scored"]=(score>0).astype(int)
        tr[f"{task}_theta_known"]=tr[f"{task}_theta"].notna().astype(int)

    # Fair comparison: one OOF prediction per student per repeat, then compare
    # repeat-level error aggregates, not 10 repeats as 10 distinct students.
    oof=reference_oof[["scheme","repeat","pos","y_true","y_pred"]].copy()
    oof["school_id"]=tr.school_id.to_numpy()[oof.pos.to_numpy(int)]
    oof["instrument"]=tr.instrument.to_numpy()[oof.pos.to_numpy(int)]
    oof["error"]=oof.y_true-oof.y_pred
    oof["squared_error"]=np.square(oof.error)
    oof["abs_error"]=np.abs(oof.error)
    by_inst=oof.groupby(["scheme","repeat","instrument"]).agg(
        n=("pos","size"),sse=("squared_error","sum"),
        mae=("abs_error","mean"),bias=("error","mean")).reset_index()
    by_inst.to_csv(path/"instrument_transfer_by_repeat.csv",index=False)

    counts=tr.groupby(["school_id","instrument"]).agg(
        n=("outcome_score","size"),mean_outcome=("outcome_score","mean"),
        sd_outcome=("outcome_score","std"),
        blending_theta_mean=("blending_theta","mean"),
        deletion_theta_mean=("deletion_theta","mean"),
        repetition_theta_mean=("sentence_repetition_theta","mean"),
        deletion_admin_frac=("deletion_given","mean"),
        deletion_theta_cov=("deletion_theta_known","mean")
    ).reset_index()
    counts["suppressed_in_shared_figures"]=counts.n<min_cell
    counts.to_csv(path/"instrument_school_atlas_PRIVATE.csv",index=False)

    school_err=oof.groupby(["scheme","repeat","school_id","instrument"]).agg(
        n=("pos","size"),sse=("squared_error","sum"),
        mae=("abs_error","mean"),bias=("error","mean")).reset_index()
    school_err.to_csv(path/"school_instrument_transfer_PRIVATE.csv",index=False)
    availability=tr.groupby("instrument")[[f"{t}_given" for t in TASKS]+
        [f"{t}_scored" for t in TASKS]+[f"{t}_theta_known" for t in TASKS]].mean()
    availability.to_csv(path/"missingness_by_instrument.csv")
    mask=sum(tr[f"{t}_given"].to_numpy(int)*(2**i) for i,t in enumerate(TASKS))
    patterns=pd.DataFrame({"instrument":tr.instrument.to_numpy(),"pattern":mask,
                           "student":np.ones(len(tr))})
    pattern_table=patterns.groupby(["instrument","pattern"]).student.agg(
        n="count").reset_index()
    pattern_table.to_csv(path/"task_availability_patterns_PRIVATE.csv",index=False)
    # Test coverage only: no outcome associations and no fitted statistics.
    test_cov=[]
    for name,clips,students in [
        ("train",data.train_clips,data.train_students),("test",data.test_clips,data.test_students)]:
        for task in TASKS:
            subset=clips[clips.task==task]
            n=subset.groupby("subject_id").size()
            nscored=subset.groupby("subject_id").item_score.agg(
                lambda s:pd.to_numeric(s,errors="coerce").notna().sum()) if "item_score" in subset else pd.Series(dtype=float)
            test_cov.append({"split":name,"task":task,"n_students":len(students),
                             "administered_fraction":float((n.reindex(students.index).fillna(0)>0).mean()),
                             "scored_fraction":float((nscored.reindex(students.index).fillna(0)>0).mean())})
    pd.DataFrame(test_cov).to_csv(path/"task_availability_train_test_coverage_only.csv",index=False)
    contract=item_contract(data.train_clips,data.test_clips)
    original=data.train_clips[["task","expected_text"]].copy()
    original["normalized"]=original.expected_text.map(normalize_prompt)
    variants=original.groupby(["task","normalized"]).expected_text.nunique(dropna=False)
    contract["n_train_prompt_keys"]=int(len(variants))
    contract["n_normalization_variant_keys"]=int((variants>1).sum())
    contract["n_rare_prompt_keys_at_most_2_clips"]=int(
        (original.groupby(["task","normalized"]).size()<=2).sum())
    atomic_json(path/"item_contract.json",contract)

    matrix=counts.pivot(index="school_id",columns="instrument",values="n").fillna(0).astype(int)
    figure,ax=plt.subplots(figsize=(9,max(5,.38*len(matrix)+1)))
    img=ax.imshow(matrix.to_numpy(float),aspect="auto",cmap="Blues")
    ax.set(yticks=range(len(matrix)),yticklabels=matrix.index,
           xticks=range(len(matrix.columns)),xticklabels=matrix.columns,
           title="School × assessment instrument (cells <5 suppressed)")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            val=int(matrix.iloc[i,j])
            ax.text(j,i,str(val) if val>=min_cell else "<5",
                    ha="center",va="center",fontsize=8)
    figure.colorbar(img,ax=ax,label="Students",fraction=.025)
    figure.tight_layout();figure.savefig(fig/"01_school_instrument_counts_private.png",dpi=165)
    plt.close(figure)

    means=by_inst.groupby(["scheme","instrument"]).agg(
        mae=("mae","mean"),bias=("bias","mean"),sse=("sse","mean")).reset_index()
    display=means.pivot(index="instrument",columns="scheme",values="mae")
    figure,ax=plt.subplots(figsize=(8,4))
    display.plot(kind="bar",ax=ax)
    ax.set(xlabel="Instrument",ylabel="Reference OOF MAE",
           title="Student and school-disjoint held-out errors")
    figure.tight_layout();figure.savefig(fig/"02_school_instrument_transfer.png",dpi=165)
    plt.close(figure)

    figure,ax=plt.subplots(figsize=(9,4))
    (1-availability[[f"{t}_theta_known" for t in TASKS]]).T.plot(kind="bar",ax=ax)
    ax.set(xlabel="Screener task",ylabel="Fraction theta missing",
           title="Task-ability missingness by outcome instrument")
    figure.tight_layout();figure.savefig(fig/"03_missingness_by_instrument.png",dpi=165)
    plt.close(figure)

    school_inst_summary=tr.groupby("instrument").agg(
        n=("outcome_score","size"),mean=("outcome_score","mean"),
        std=("outcome_score","std"))
    summary=[
        "# M8.5 — School/instrument and task missingness diagnostic atlas","",
        "**Observational, training-only diagnostics.** No school IDs enter the model.",
        "Repeated OOF errors are correlated; per-repeat aggregates are not independent student samples.","",
        "## Outcome distributions by instrument","",
        school_inst_summary.to_markdown(),"",
        "## Held-out reference errors by instrument and CV scheme","",
        means.to_markdown(index=False),"",
        "## Deletion and joint missingness","",
        availability.to_markdown(),"",
        "## Item contract","",
        f"- Train prompt keys: {contract['n_train_prompt_keys']}",
        f"- Keys with multiple normalized raw variants: {contract['n_normalization_variant_keys']}",
        f"- Rare item keys (≤2 clips): {contract['n_rare_prompt_keys_at_most_2_clips']}",
        "- **Never** derive chronology from CSV row order, filename or item IDs.",
        "- Do not infer school causal effects from instrument-linked school errors.",
        "- Raw school/item/student records stay in private Drive.",
        ""]
    atomic_text(path/"school_transfer_diagnostics.md","\n".join(summary))
    return {"n_schools":int(tr.school_id.nunique()),
            "n_instruments":int(tr.instrument.nunique()),
            "n_prompt_keys":contract["n_train_prompt_keys"],
            "mean_instrument_error_by_scheme":means.to_dict("records"),
            "item_score_test_coverage":contract["test"]["item_score_coverage"],
            "disabled_order_features":contract["disabled"]}
