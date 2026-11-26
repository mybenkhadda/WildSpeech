"""Run M8.4 Phase 0 reproducibly against a mounted Google Drive dataset.

Usage: python -m wildspeech_m84.runner --drive-root /content/drive/MyDrive
Creates private data manifests, frozen folds, OOF predictions, reference
submission and a report. Does NOT upload a submission to the competition.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm

from .data import load_official,locate_data_root,sha_json,atomic_json,atomic_text,digest_bytes
from .folds import build_folds,freeze_folds,fold_digest
from .reference import reference_frames,make_reference,reference_feature_hash,REFERENCE_COLUMNS
from .metrics import summarize_oof,compare_runs,promotion_gate
from .ledger import preregister,Ledger,make_run_id,utc_now
from .registry import build_registry,registry_summary
from .submission import build_submission
from .legacy_adapters import locate_m7_profile,load_m83_legacy_bank,fit_predict_m83,documented_legacy_status


def git_head(repository_root=None):
    folder=Path(repository_root or Path(__file__).resolve().parents[3])
    try:
        return subprocess.check_output(["git","-C",str(folder),"rev-parse","HEAD"],
                                       text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN_SOURCE_SHA"


def _write_parquet(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp.parquet")
    df.to_parquet(tmp,index=False)
    os.replace(tmp,path)


def _check_model_output(x,y,train,valid,predictor,seed):
    if set(train)&set(valid):
        raise ValueError("Train/validation overlap")
    p=np.asarray(predictor(x,y,train,valid,seed),dtype=float).reshape(-1)
    if len(p)!=len(valid) or not np.isfinite(p).all():
        raise ValueError("Model produced invalid fold predictions")
    return p


def _reference_predictor(x,y,tr,va,seed):
    model=make_reference()
    model.fit(x.iloc[tr],y[tr])
    return model.predict(x.iloc[va])


def _m83_predictor(columns):
    def pred(x,y,tr,va,seed):
        return fit_predict_m83(x,y,tr,va,columns,seed)
    return pred


def train_locked_model(model_id,frame,y,ids,folds,root,git_sha,fold_hash,
                       feature_hash,data_manifest_hash,model_config,predictor,
                       resume=True,notes="",compliance="reference_train_only"):
    """Fold-level cache with immutable run identity and explicit ledger states."""
    ledger=Ledger(root)
    cfg_hash=sha_json(model_config)
    run_id=make_run_id(model_id,git_sha,feature_hash,fold_hash,cfg_hash)
    folder=Path(root)/"runs"/run_id
    output=folder/"oof_predictions_PRIVATE.parquet"
    prev=ledger.latest(run_id)
    if prev and prev["status"]=="COMPLETE":
        oof=pd.read_parquet(output)
        from .metrics import validate_oof
        validate_oof(oof,folds,ids,y)
        print(f"[M84] Reused validated {model_id} run={run_id}")
        return run_id,oof
    if prev and prev["status"] not in ("PLANNED","RUNNING"):
        raise RuntimeError(f"Run {run_id} is terminal ({prev['status']}); change config/version")
    if prev is None:
        ledger.transition(run_id,"PLANNED",model_id=model_id,git_sha=git_sha,
                          created_utc=utc_now(),feature_set_hash=feature_hash,
                          fold_hash=fold_hash,data_manifest_hash=data_manifest_hash,
                          model_config=model_config,config_hash=cfg_hash,
                          comparison_plan_hash=json.loads((Path(root)/"ledger"/"planned_comparisons.json").read_text())["plan_hash"],
                          notes=notes,compliance=compliance)
    if ledger.latest(run_id)["status"]=="PLANNED":
        ledger.transition(run_id,"RUNNING",started_utc=utc_now())
    oof_parts=[]
    started=time.perf_counter()
    try:
        for fold in tqdm(folds.itertuples(),total=len(folds),
                         desc=f"{model_id} locked CV",unit="fold"):
            pth=folder/"folds"/f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet"
            valid=np.asarray(fold.valid_pos,dtype=int)
            train=np.asarray(fold.train_pos,dtype=int)
            if resume and pth.is_file():
                part=pd.read_parquet(pth)
                if (set(part.pos.astype(int))!=set(valid) or len(part)!=len(valid) or
                    part.subject_id.tolist()!=[ids[p] for p in valid] or
                    not np.allclose(part.y_true,y[valid],rtol=0,atol=0)):
                    raise RuntimeError(f"Checkpoint mismatch, delete only after audit: {pth}")
            else:
                predictions=_check_model_output(frame,y,train,valid,predictor,
                              seed=42+int(fold.repeat)*13+int(fold.fold))
                part=pd.DataFrame({
                    "scheme":str(fold.scheme),"repeat":int(fold.repeat),
                    "fold":int(fold.fold),"pos":valid,
                    "subject_id":[ids[p] for p in valid],
                    "y_true":y[valid],"y_pred":predictions})
                _write_parquet(part,pth)
            oof_parts.append(part)
        oof=pd.concat(oof_parts,ignore_index=True)
        from .metrics import validate_oof
        validate_oof(oof,folds,ids,y)
        _write_parquet(oof,output)
        scores,pool,inst=summarize_oof(oof,folds,ids,y)
        _write_parquet(scores,folder/"fold_metrics.parquet")
        _write_parquet(pool,folder/"pooled_metrics.parquet")
        if not inst.empty:
            _write_parquet(inst,folder/"instrument_metrics.parquet")
        subset=scores[scores.scheme=="student"]
        pooled=pool[pool.scheme=="student"]
        if not len(subset)==50 or not len(pooled)==10:
            raise ValueError("Incorrect student-fold summary size")
        metrics={
            "mean_fold_r2":float(subset.r2.mean()),
            "sd_fold_r2":float(subset.r2.std(ddof=0)),
            "mean_pooled_oof_r2":float(pooled.r2.mean()),
            "sd_pooled_oof_r2":float(pooled.r2.std(ddof=0)),
            "mean_fold_mae":float(subset.mae.mean()),
            "mean_fold_rmse":float(subset.rmse.mean()),
            "mean_school_pooled_r2":float(pool[pool.scheme=="school"].r2.mean()),
            "elapsed_sec":time.perf_counter()-started,
        }
        atomic_json(folder/"metrics.json",metrics)
        manifest={"model_id":model_id,"run_id":run_id,"git_sha":git_sha,
                  "fold_hash":fold_hash,"feature_hash":feature_hash,
                  "data_manifest_hash":data_manifest_hash,
                  "model_config":model_config,"compliance":compliance,
                  "oof_path":str(output),"metrics":metrics}
        atomic_json(folder/"run_manifest.json",manifest)
        ledger.transition(run_id,"COMPLETE",finished_utc=utc_now(),**metrics,
                          oof_path=str(output),metrics_path=str(folder/"metrics.json"),
                          run_manifest_path=str(folder/"run_manifest.json"))
        return run_id,oof
    except Exception as e:
        if ledger.latest(run_id)["status"]=="RUNNING":
            ledger.transition(run_id,"FAILED",finished_utc=utc_now(),
                              error=str(e),traceback=traceback.format_exc()[-5000:])
        raise


def _report(root,reference_run,score,parity,order,hashes,run_ids,comparisons,
            legacy_status,registry_results,submission):
    lines=[
        "# M8.4 — Phase 0 Baseline Parity and Frozen Evaluation",
        "",
        "## Evidence status",
        "",
        "**Official training-data run:** "+("PASSED parity" if parity else "FAILED parity — investigate before promotion"),
        "**No official online submission has been made by this pipeline.**",
        "",
        "## 1. Reproducibility",
        "",
        f"- Reference canonical order: metadata row order; labels-order-equal = {order['labels_original_order_equal']}",
        f"- Canonical student order SHA-256: {order['canonical_student_order_sha256']}",
        f"- Frozen fold SHA-256: {hashes['fold_hash']}",
        f"- Official input manifest SHA-256: {hashes['data_manifest_hash']}",
        f"- Reference feature matrix SHA-256: {hashes['reference_feature_hash']}",
        f"- Code SHA: {hashes['git_sha']}",
        "",
        "## 2. Exact DrivenData 14-feature reference",
        "",
        "| Metric | Official-data result | Reference benchmark |",
        "| --- | ---: | ---: |",
        f"| Mean 50-fold R² | {score['mean_fold_r2']:.6f} | 0.193 ±0.005 parity range |",
        f"| 50-fold R² SD | {score['sd_fold_r2']:.6f} | 0.043 reference fold SD |",
        f"| Mean 10-repeat pooled OOF R² | {score['mean_pooled_oof_r2']:.6f} | separately reported |",
        f"| School-disjoint pooled OOF R² | {score['mean_school_pooled_r2']:.6f} | robustness only |",
        f"| Mean fold MAE | {score['mean_fold_mae']:.6f} | secondary |",
        f"| Mean fold RMSE | {score['mean_fold_rmse']:.6f} | secondary |",
        "",
        f"**Parity check: {'PASS' if parity else 'FAIL'}**. Reference parity targets mean-fold R² in [0.188, 0.198].",
        "The public leaderboard 0.2888 is an external historical result, NOT a gate or new submission score.",
        "",
        "## 3. Model run inventory",
        "",
        "| Model | Status | Run ID |",
        "| --- | --- | --- |",
    ]
    for name,state in legacy_status.items():
        lines.append(f"| {name} | {state} | {run_ids.get(name,'—')} |")
    lines.extend(["","## 4. Preregistered paired comparisons","","Only compare results from identical frozen folds. Legacy diagnostics are NOT promotion eligible until their train-only feature formation is audited.",""])
    if comparisons:
        lines.append("| Challenger vs baseline | Δ mean-fold R² | corrected t | wins | pooled school Δ R² | gate |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for c in comparisons:
            lines.append(f"| {c['challenger']} / {c['baseline']} | {c['mean_fold_delta_r2']:+.5f} | "
                         f"{c['corrected_t']:.3f} | {c['win_fraction']:.1%} | "
                         f"{c['school_pooled_delta_r2']:+.5f} | RESEARCH ONLY |")
    else:
        lines.append("No clean historical comparison completed. No missing score has been invented.")
    lines.extend(["","## 5. Feature provenance",
                  f"- Reference features: {len(REFERENCE_COLUMNS)} (exactly 14).",
                  f"- Registry: {registry_results}",
                  "- Test coverage is audit-only; it never determines training transforms or model selection.",
                  "- M83 legacy taxonomy incorrectly calls correctness-derived features audio.",
                  "",
                  "## 6. Submission readiness",
                  f"- Submission file: {submission.get('submission_path','NOT GENERATED')}",
                  f"- SHA-256: {submission.get('sha256','UNAVAILABLE')}",
                  f"- Singleton/batch invariance: {submission.get('singleton_batch_test',False)}",
                  "- Upload to the competition must be performed separately and the actual receipt recorded.",
                  "",
                  "## 7. Decision and Phase 1 handoff",
                  "",
                  "**"+("CONDITIONAL GO" if parity else "NO GO")+"**: "+
                  ("Exact reference parity and submission invariance passed; unavailable/legacy-only models remain explicitly unresolved."
                   if parity else "Reference parity failed; fix reference feature order, imputation, versions, or CSV alignment before moving to M0+."),
                  "",
                  "Pre-register Phase 1 model candidates P1-A (reference with regularized model stack), "
                  "P1-B (S2–S5 screening bundle), P1-C (leave-one-block-out ablations).",
                  "Do not use outer folds to tune blend weights or feature-building decisions.",
                  "",
                  "## 8. Limits",
                  "- This harness does not infer actual leaderboard scores.",
                  "- Nadeau–Bengio corrected t is approximate and does not correct selection over many trials.",
                  "- The school-disjoint scheme is robustness only, not the leaderboard split.",
                  "- Instrument subgroup R² is undefined for constant-target or small subsets.",
                  "- Restored M7.3 source and independently audited M8.3 transformation code are prerequisites for true historical parity.",
                  ])
    atomic_text(Path(root)/"parity"/"M84_BASELINE_PARITY.md","\n".join(lines)+"\n")


def run_phase0(drive_root, data_root_override="", output_root_override="", m8_root_override="",
               m7_profile_override="", run_legacy_research=False, repository_root=None,
               require_parity=True):
    drive=Path(drive_root).expanduser()
    if not drive.is_dir():
        raise FileNotFoundError(f"Drive unavailable: {drive}")
    data_root=locate_data_root(drive,data_root_override)
    root=Path(output_root_override).expanduser() if output_root_override else drive/"WildSpeech_M84"/"phase0"
    root.mkdir(parents=True,exist_ok=True)
    print(f"[M84] Official data: {data_root}")
    print(f"[M84] Private output root: {root}")
    data=load_official(data_root,root)
    ids=data.ids
    y=data.labels.to_numpy(float)
    order=data.order_manifest
    folds=build_folds(ids,data.groups)
    fold_hash=freeze_folds(folds,ids,data.groups,root)
    preregister(root)
    ledger=Ledger(root)
    git_sha=git_head(repository_root)
    data_hash=digest_bytes(root/"manifests"/"data_manifest.json")
    x,xt,y=reference_frames(data)
    feature_hash=reference_feature_hash(x)
    reg=build_registry(x,xt)
    reg_path=root/"features"/"registry.csv"
    reg_path.parent.mkdir(parents=True,exist_ok=True)
    reg.to_csv(reg_path,index=False)
    reg_result=registry_summary(reg)
    manifest={
        "python":sys.version,"platform":platform.platform(),
        "numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,
        "git_sha":git_sha,"fold_hash":fold_hash,
        "reference_feature_hash":feature_hash,"data_manifest_sha256":data_hash,
        "executed_utc":utc_now(),"n_students":len(ids),
        "n_school_groups":len(set(data.groups)),"reference_columns":REFERENCE_COLUMNS}
    atomic_json(root/"manifests"/"environment.json",manifest)
    config={"pipeline":"SimpleImputer(median,indicator)->StandardScaler->RidgeCV(logspace(-2,5,50))",
            "reference_columns":REFERENCE_COLUMNS,"model_version":"drivendata-2026-11-24"}
    ref_run,ref_oof=train_locked_model(
        "REF_RIDGE14",x,y,ids,folds,root,git_sha,fold_hash,feature_hash,data_hash,
        config,_reference_predictor,
        notes="Exact DrivenData metadata-only reference; student metadata order")
    ref_metrics=json.loads((root/"runs"/ref_run/"metrics.json").read_text())
    parity=(.188 <= ref_metrics["mean_fold_r2"] <= .198)
    submission={}
    if parity:
        submission=build_submission(x,y,xt,data.submission_format,root/"submissions")
    else:
        print("[M84] REFERENCE PARITY FAILED; no accepted submission written")
    legacy_status=documented_legacy_status()
    run_ids={"REF_RIDGE14":ref_run}
    comparisons=[]
    if run_legacy_research:
        print("[M84] LEGACY RESEARCH ONLY: feature formation has not passed M8.4 test-row fit audit")
        cache_root=Path(m8_root_override).expanduser() if m8_root_override else drive/"WildSpeech_M8"
        profile=locate_m7_profile(drive,m7_profile_override)
        try:
            bx,bxt,by,bgroups,families,bank_manifest=load_m83_legacy_bank(data_root,cache_root,profile)
            if bx.index.astype(str).tolist()!=ids or not np.array_equal(by,y):
                raise ValueError("M8.3 feature-bank student/label order mismatch")
            # Keep original M8.3 feature selection and naming for historical traceability.
            legacy_reg=build_registry(bx,bxt)
            legacy_reg.to_csv(root/"features"/"legacy_registry_RESEARCH_ONLY.csv",index=False)
            for model_id,cols in [
                ("M83_ALL",list(bx.columns)),
                ("M83_WITHOUT_AUDIO_NAMING",[c for c in bx.columns if families[c]!="audio"])
            ]:
                feature_hash_legacy=sha_json({
                    "columns":cols,
                    "shape":list(bx[cols].shape),
                    "train_hash":pd.util.hash_pandas_object(bx[cols].astype(str),index=False).to_numpy().tolist(),
                })
                legacy_id,legacy_oof=train_locked_model(
                    model_id,bx[cols],y,ids,folds,root,git_sha,fold_hash,
                    feature_hash_legacy,data_hash,
                    {"implementation":"M8.3 legacy fit_predict fast=True", "columns":cols,
                     "legacy_feature_warning":True},
                    _m83_predictor(cols),
                    notes="RESEARCH_ONLY: past M8 feature formation may inspect test values",
                    compliance="LEGACY_RESEARCH_ONLY")
                run_ids[model_id]=legacy_id
                comp=compare_runs(ref_oof,legacy_oof,folds,ids,y,data.instruments,
                                  fold_hash,fold_hash,"REF_RIDGE14",model_id)
                comp["gate"]="RESEARCH_ONLY"
                comparisons.append(comp)
                atomic_json(root/"runs"/legacy_id/"comparison_to_reference_RESEARCH_ONLY.json",comp)
                legacy_status[model_id]="COMPLETE_RESEARCH_ONLY (not promotion-eligible)"
            if all(x in run_ids for x in ("M83_ALL","M83_WITHOUT_AUDIO_NAMING")):
                aa=pd.read_parquet(root/"runs"/run_ids["M83_ALL"]/"oof_predictions_PRIVATE.parquet")
                bb=pd.read_parquet(root/"runs"/run_ids["M83_WITHOUT_AUDIO_NAMING"]/"oof_predictions_PRIVATE.parquet")
                comp=compare_runs(bb,aa,folds,ids,y,data.instruments,fold_hash,fold_hash,
                                  "M83_WITHOUT_AUDIO_NAMING","M83_ALL")
                comp["gate"]="RESEARCH_ONLY"
                comparisons.append(comp)
        except Exception as exc:
            legacy_status["M83_ALL"]="NOT_REPRODUCED: "+str(exc)
            legacy_status["M83_WITHOUT_AUDIO_NAMING"]="NOT_REPRODUCED: "+str(exc)
            print("[M84] Legacy diagnostics unavailable:",exc)
    # M7.3 cannot be called "reproduced" without exact source. Preserve this fact.
    if ledger.latest("M73_GLOBAL_UNAVAILABLE") is None:
        ledger.transition("M73_GLOBAL_UNAVAILABLE","NOT_REPRODUCED",
                          model_id="M73_GLOBAL",notes=legacy_status["M73_GLOBAL"])
    if not run_legacy_research and ledger.latest("M83_LEGACY_NOT_AUDITED") is None:
        ledger.transition("M83_LEGACY_NOT_AUDITED","NOT_REPRODUCED",
                          model_id="M83_LEGACY",notes="Train/test feature-formation compliance review required")
    parity_report={
        "parity_pass":parity,"reference_run_id":ref_run,
        "reference_metrics":ref_metrics,
        "run_ids":run_ids,
        "historical_status":legacy_status,
        "comparisons":comparisons,
        "submission":submission,"fold_hash":fold_hash,
        "phase0_gate":"CONDITIONAL_GO" if parity else "NO_GO",
    }
    atomic_json(root/"parity"/"M84_RESULTS.json",parity_report)
    _report(root,ref_run,ref_metrics,parity,order,
            {"fold_hash":fold_hash,"data_manifest_hash":data_hash,
             "reference_feature_hash":feature_hash,"git_sha":git_sha},
            run_ids,comparisons,legacy_status,reg_result,submission)
    print("[M84] Mean 50-fold R²:",round(ref_metrics["mean_fold_r2"],6))
    print("[M84] Phase 0 gate:",parity_report["phase0_gate"])
    print("[M84] Report:",root/"parity"/"M84_BASELINE_PARITY.md")
    if require_parity and not parity:
        raise RuntimeError("Reference parity FAILED (report written). Fix before Phase 1.")
    return parity_report


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive-root",default="/content/drive/MyDrive")
    p.add_argument("--data-root",default="")
    p.add_argument("--output-root",default="")
    p.add_argument("--m8-root",default="")
    p.add_argument("--m7-profile",default="")
    p.add_argument("--legacy-research-only",action="store_true",
                   help="Run historical M8 adapters; NEVER promotion eligible")
    p.add_argument("--allow-parity-failure",action="store_true",
                   help="Write diagnostics without raising when reference differs")
    args=p.parse_args()
    result=run_phase0(args.drive_root,args.data_root,args.output_root,
                      args.m8_root,args.m7_profile,args.legacy_research_only,
                      require_parity=not args.allow_parity_failure)
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    cli()
