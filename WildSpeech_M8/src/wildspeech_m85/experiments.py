"""Locked M8.5 E00–E09 experiments on M8.4 folds and frozen reference OOF.

Use --mode SCREEN for one repeat of each scheme (exploratory and NEVER
promotable) and --mode FULL for all 100 fixed folds and all promotion gates.
Private row-level data, predictions and fitted models are written only to Drive.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from wildspeech_m84.data import (FILES,sha_json,atomic_json,digest_bytes,
                                  load_official,locate_data_root)
from wildspeech_m84.folds import fold_digest,validate_folds
from wildspeech_m84.metrics import (validate_oof,summarize_oof,compare_runs,
                                    promotion_gate)
from wildspeech_m84.ledger import Ledger,preregister,make_run_id,utc_now
from wildspeech_m84.reference import reference_frames,reference_feature_hash
from wildspeech_m84.runner import git_head
from .contracts import source_hash,item_contract
from .nested_cv import FoldCache,model_spec,predict_fold
from .diagnostics import run_diagnostics
from .reporting import render_results
from .submission import fit_and_export

EVAL_HASH_2026="6ee0f59c8cf5ea3d8f20f5fd500353419587660e00949ac6f1e6a16e092c9ce1"
CANDIDATES=tuple(f"E{i:02d}" for i in range(1,10))
PROMOTION_PRIORITY=("E04","E03","E02","E01")


def locked_plan():
    return [
        {"comparison_id":f"{model}_VS_E00","baseline":"E00",
         "challenger":model,"model_spec":model_spec(model),
         "status":"PREREGISTERED","eligible_for_promotion":model in PROMOTION_PRIORITY}
        for model in CANDIDATES
    ]+[
        {"comparison_id":"E04_VS_E01","baseline":"E01","challenger":"E04",
         "status":"PREREGISTERED","scope":"incremental S2–S5 bundle"},
    ]+[
        {"comparison_id":f"E{j:02d}_VS_E04","baseline":"E04","challenger":f"E{j:02d}",
         "status":"PREREGISTERED","scope":"conditional explanatory ablation only"}
        for j in range(5,10)
    ]


def _record(path,df):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+".tmp.parquet")
    df.to_parquet(tmp,index=False);os.replace(tmp,path)


def _load_reference(root, data, expected_hash=EVAL_HASH_2026):
    folder=Path(root)
    manifest=json.loads((folder/"manifests"/"fold_manifest.json").read_text())
    frozen_hash=manifest["fold_hash"]
    if expected_hash and frozen_hash!=expected_hash:
        raise RuntimeError(f"Unexpected M8.4 frozen fold hash {frozen_hash}. "
                           "Do NOT silently regenerate folds.")
    old_files=json.loads((folder/"manifests"/"data_manifest.json").read_text())["files"]
    for name in FILES:
        if digest_bytes(data.root/name)!=old_files[name]["sha256"]:
            raise RuntimeError(f"Official source CSV has changed since reference fit: {name}")
    fold=pd.read_parquet(folder/"folds"/"folds.parquet")
    validate_folds(fold,data.ids,data.groups)
    computed=fold_digest(fold,data.ids,data.groups)
    if computed!=frozen_hash:
        raise RuntimeError("Frozen M8.4 folds failed logical hash verification")
    info=json.loads((folder/"parity"/"M84_RESULTS.json").read_text())
    if info["fold_hash"]!=frozen_hash or not info["parity_pass"]:
        raise RuntimeError("M8.4 reference parity was not passed on frozen folds")
    ref_run=info["reference_run_id"]
    reference_dir=folder/"runs"/ref_run
    manifest_run=json.loads((reference_dir/"run_manifest.json").read_text())
    x,xt,y=reference_frames(data)
    if reference_feature_hash(x)!=manifest_run["feature_hash"]:
        raise RuntimeError("Reference feature values/order changed since M8.4")
    oof_path=reference_dir/"oof_predictions_PRIVATE.parquet"
    oof=pd.read_parquet(oof_path)
    validate_oof(oof,fold,data.ids,y)
    metrics=json.loads((reference_dir/"metrics.json").read_text())
    if not (.188<=metrics["mean_fold_r2"]<=.198):
        raise RuntimeError("Reference R² parity no longer passes")
    return fold,oof,frozen_hash,digest_bytes(oof_path),metrics


def _model_snapshot(oof,fold,ids,y,instruments):
    fold_metrics,pool,inst=summarize_oof(oof,fold,ids,y,instruments)
    st=fold_metrics[fold_metrics.scheme=="student"]
    sp=pool[pool.scheme=="student"]
    sc=pool[pool.scheme=="school"]
    return {
        "mean_fold_r2":float(st.r2.mean()),
        "mean_student_pooled_r2":float(sp.r2.mean()),
        "mean_school_pooled_r2":float(sc.r2.mean()),
        "student_mae":float(sp.mae.mean()),
        "student_rmse":float(sp.rmse.mean()),
        "n_folds":len(fold),
    },fold_metrics,pool,inst


def _checkpoint_ok(part,fold,ids,y):
    expected=np.asarray(fold.valid_pos,int)
    return (len(part)==len(expected) and
            part.pos.astype(int).tolist()==expected.tolist() and
            part.subject_id.astype(str).tolist()==[ids[p] for p in expected] and
            np.array_equal(part.y_true.to_numpy(float),y[expected]) and
            np.isfinite(part.y_pred.to_numpy(float)).all())


def _run_candidates(models,folds,data,mode,root,git_sha,ref_hash,ref_oof_sha,
                    screening_hash,iterations,seeds,kappa,n_knots,inner_k):
    ledger=Ledger(root)
    y=data.labels.to_numpy(float)
    ids=data.ids
    plan_hash=json.loads((root/"ledger"/"planned_comparisons.json").read_text())["plan_hash"]
    runs={}
    for mid in models:
        config={"model_id":mid,"spec":model_spec(mid),"mode":mode,
                "iterations":iterations,"seeds":list(seeds),
                "kappa":kappa,"n_knots":n_knots,"inner_folds":inner_k}
        feature_hash=sha_json({"sources":screening_hash,
                               "feature_construction":"m85_v1_LOO_and_fold_splines",
                               "blocks":model_spec(mid).get("blocks",[]),"kappa":kappa,
                               "n_knots":n_knots})
        cfg_hash=sha_json(config)
        run_id=make_run_id(mid,git_sha,feature_hash,ref_hash,cfg_hash)
        root_run=root/"runs"/run_id
        row=ledger.latest(run_id)
        if row is None:
            ledger.transition(run_id,"PLANNED",model_id=mid,git_sha=git_sha,
                              feature_set_hash=feature_hash,fold_hash=ref_hash,
                              reference_oof_sha256=ref_oof_sha,config_hash=cfg_hash,
                              model_config=config,comparison_plan_hash=plan_hash,
                              mode=mode,compliance="FOLD_FIT_AND_STUDENT_LOO")
        elif row["status"] not in ("PLANNED","RUNNING","COMPLETE"):
            raise RuntimeError(f"Cannot resume terminal {run_id} ({row['status']})")
        runs[mid]={"run_id":run_id,"dir":root_run,"config":config}
    incomplete={}
    for mid,v in runs.items():
        path=v["dir"]/"oof_predictions_PRIVATE.parquet"
        if ledger.latest(v["run_id"])["status"]=="COMPLETE":
            if not path.is_file():
                raise RuntimeError(f"Completed run missing OOF: {path}")
            oof=pd.read_parquet(path)
            validate_oof(oof,folds,ids,y)
            old=ledger.latest(v["run_id"])
            if old.get("oof_sha256")!=digest_bytes(path):
                raise RuntimeError(f"OOF fingerprint changed since COMPLETE: {mid}")
            v["oof"]=oof
        else:
            incomplete[mid]=v
            if ledger.latest(v["run_id"])["status"]=="PLANNED":
                ledger.transition(v["run_id"],"RUNNING",started_utc=utc_now())
    if incomplete:
        for fold in tqdm(folds.itertuples(),total=len(folds),
                         desc=f"M8.5 {mode} matched folds",unit="fold"):
            missing=[]
            for mid,v in incomplete.items():
                part_path=v["dir"]/"folds"/f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet"
                if part_path.is_file():
                    part=pd.read_parquet(part_path)
                    if not _checkpoint_ok(part,fold,ids,y):
                        raise RuntimeError(f"Corrupt fold checkpoint: {part_path}")
                else:
                    missing.append(mid)
            if not missing:
                continue
            cache=FoldCache(data.train_students,data.train_clips,y,
                            fold.train_pos,fold.valid_pos,seed=42+int(fold.repeat)*13+int(fold.fold),
                            kappa=kappa,n_knots=n_knots,inner_k=inner_k)
            for mid in missing:
                pred,extra=predict_fold(cache,mid,seeds=seeds,iterations=iterations)
                val=np.asarray(fold.valid_pos,int)
                if len(pred)!=len(val) or not np.isfinite(pred).all():
                    raise ValueError(f"Invalid {mid} predictions in {fold.scheme}/{fold.repeat}/{fold.fold}")
                part=pd.DataFrame({"scheme":fold.scheme,"repeat":int(fold.repeat),
                    "fold":int(fold.fold),"pos":val,
                    "subject_id":[ids[p] for p in val],"y_true":y[val],"y_pred":pred})
                save=incomplete[mid]["dir"]/"folds"/f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet"
                _record(save,part)
                atomic_json(save.with_suffix(".model.json"),extra)
        for mid,v in incomplete.items():
            parts=[]
            for fold in folds.itertuples():
                p=v["dir"]/"folds"/f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet"
                part=pd.read_parquet(p)
                if not _checkpoint_ok(part,fold,ids,y):
                    raise RuntimeError("Partial/mismatched fold predictions")
                parts.append(part)
            oof=pd.concat(parts,ignore_index=True)
            validate_oof(oof,folds,ids,y)
            _record(v["dir"]/"oof_predictions_PRIVATE.parquet",oof)
            m,f,p,i=_model_snapshot(oof,folds,ids,y,data.instruments)
            _record(v["dir"]/"fold_metrics.parquet",f)
            _record(v["dir"]/"pooled_metrics.parquet",p)
            _record(v["dir"]/"instrument_metrics.parquet",i)
            atomic_json(v["dir"]/"metrics.json",m)
            digest=digest_bytes(v["dir"]/"oof_predictions_PRIVATE.parquet")
            ledger.transition(v["run_id"],"COMPLETE",finished_utc=utc_now(),
                              oof_path=str(v["dir"]/"oof_predictions_PRIVATE.parquet"),
                              oof_sha256=digest,mean_fold_r2=m["mean_fold_r2"],
                              mean_pooled_oof_r2=m["mean_student_pooled_r2"],
                              mean_school_pooled_r2=m["mean_school_pooled_r2"],
                              paired_fold_count=len(folds))
            v["oof"]=oof
    return runs


def _comparisons(models,runs,folds,reference_oof,data,ref_hash,mode):
    y=data.labels.to_numpy(float)
    ids=data.ids
    base=reference_oof
    summaries=[];detail=[]
    for mid in models:
        oof=runs[mid]["oof"]
        snapshot,_,_,_=_model_snapshot(oof,folds,ids,y,data.instruments)
        row={"model":mid,**snapshot}
        if mode=="FULL":
            c=compare_runs(base,oof,folds,ids,y,data.instruments,
                           ref_hash,ref_hash,"E00",mid)
            g,why=promotion_gate(c)
            if mid not in PROMOTION_PRIORITY:
                g="DIAGNOSTIC_ONLY"
                why="Ablation or nuisance audit; not eligible for promotion"
            row.update({k:c[k] for k in ("mean_fold_delta_r2","corrected_t","win_fraction",
                       "school_pooled_delta_r2","mean_student_pooled_delta_r2")})
            row["gate"]=g;row["gate_reason"]=why
            row["instrument_delta_r2"]=c["instrument_delta_r2"]
            detail.append({"comparison_id":mid+"_VS_E00","result":c,"gate":g,"reason":why})
        else:
            row.update({"mean_fold_delta_r2":float("nan"),
                        "corrected_t":float("nan"),"win_fraction":float("nan"),
                        "school_pooled_delta_r2":float("nan"),
                        "gate":"SCREEN_ONLY","gate_reason":"Only 1/10 repeats; cannot promote"})
        summaries.append(row)
    if mode=="FULL":
        # Conditional comparisons: E04 vs E01, and ablations vs E04.
        extra=[("E01","E04")]+[("E04",m) for m in ("E05","E06","E07","E08","E09")]
        for baseline,challenger in extra:
            if baseline not in runs or challenger not in runs:continue
            c=compare_runs(runs[baseline]["oof"],runs[challenger]["oof"],
                           folds,ids,y,data.instruments,ref_hash,ref_hash,
                           baseline,challenger)
            detail.append({"comparison_id":challenger+"_VS_"+baseline,
                           "result":c,"gate":"DIAGNOSTIC_ONLY"})
    return summaries,detail


def execute(drive_root="/content/drive/MyDrive",data_root_override="",
            phase0_root_override="",output_root_override="",mode="SCREEN",
            run_models=CANDIDATES,seed_count=None,iterations=None,
            kappa=10.,n_knots=4,inner_k=3,
            expected_fold_hash=EVAL_HASH_2026,export_final=True):
    if mode not in ("SCREEN","FULL"):
        raise ValueError("mode must be SCREEN or FULL")
    rootdrive=Path(drive_root)
    root=Path(output_root_override) if output_root_override else rootdrive/"WildSpeech_M85"/"m0plus"
    phase0=Path(phase0_root_override) if phase0_root_override else rootdrive/"WildSpeech_M84"/"phase0"
    data=load_official(locate_data_root(rootdrive,data_root_override))
    folds,reference,fold_hash,ref_sha,ref_metrics=_load_reference(
        phase0,data,expected_fold_hash)
    # SCREEN uses the exact first repeat already frozen in Phase 0.
    if mode=="SCREEN":
        folds=folds[folds.repeat==0].reset_index(drop=True)
        reference=reference[reference.repeat==0].copy()
    chosen=tuple(run_models)
    if len(chosen)!=len(set(chosen)) or any(m not in CANDIDATES for m in chosen):
        raise ValueError("Unknown or repeated experiment ID")
    if not chosen:
        raise ValueError("No experiments selected")
    seeds=tuple(42+j for j in range(
        5 if seed_count is None and mode=="FULL" else
        1 if seed_count is None else seed_count))
    iterations=600 if iterations is None and mode=="FULL" else 250 if iterations is None else int(iterations)
    if mode=="FULL" and (len(seeds)!=5 or iterations!=600 or kappa!=10. or n_knots!=4 or inner_k!=3):
        raise RuntimeError("FULL is the locked preregistered 5-seed, 600-iteration, kappa10, "
                           "4-knot, 3-inner-fold protocol; use SCREEN for exploratory settings")
    root.mkdir(parents=True,exist_ok=True)
    # Preregister BEFORE any new model fit; use a separate M8.5 ledger.
    preregister(root,locked_plan())
    code_sha=git_head()
    screening_hash=source_hash(data.train_students,data.train_clips)
    atomic_json(root/"manifest.json",{
        "code_sha":code_sha,"mode":mode,"fold_hash":fold_hash,
        "reference_oof_sha256":ref_sha,"reference_run":ref_metrics,
        "train_screening_source_hash":screening_hash,
        "frozen_reference_unchanged":True,"n_students":len(data.ids),
        "candidate_models":list(chosen),
        "no_test_fitted_features":True,"run_at":utc_now()})
    diagnostic=run_diagnostics(data,reference,root)
    contract=json.loads((root/"diagnostics"/"item_contract.json").read_text())
    if "item_score" not in data.test_clips or not data.test_clips.item_score.notna().any():
        # No M0+ S2 promotion can be considered without eligible test item scores.
        if any("S2" in model_spec(m)["blocks"] for m in chosen):
            raise RuntimeError("Official test item_score unavailable; S2 feature family cannot be deployed")
    runs=_run_candidates(chosen,folds,data,mode,root,code_sha,fold_hash,
                         ref_sha,screening_hash,iterations,seeds,kappa,n_knots,inner_k)
    reference_summary,_,_,_=_model_snapshot(reference,folds,data.ids,
                           data.labels.to_numpy(float),data.instruments)
    rows,comparisons=_comparisons(chosen,runs,folds,reference,data,fold_hash,mode)
    candidates={row["model"]:row for row in rows}
    winner=next((x for x in PROMOTION_PRIORITY if
                 x in candidates and candidates[x]["gate"]=="PROMOTE"),None)
    if mode=="FULL" and len(chosen)==len(CANDIDATES) and winner:
        decision="PROMOTE"
        explanation=(f"{winner} is the first predeclared-priority model passing all five "
                     "paired reference, school and instrument gates; all E05–E09 remain diagnostic.")
    elif mode=="FULL" and len(chosen)==len(CANDIDATES):
        decision="HOLD_REFERENCE"
        explanation="No eligible model passed every preregistered gate; frozen M8.4 Ridge remains M0+."
    elif mode=="SCREEN":
        decision="SCREEN_EXPLORATORY"
        explanation="Only first frozen repeat evaluated. Results cannot be promoted or submitted."
    else:
        decision="INCOMPLETE_FULL"
        explanation="Full 10 repeats exist for only a subset of preregistered arms; no promotion allowed."
    submission={}
    if decision=="PROMOTE" and export_final:
        submission=fit_and_export(data,winner,root/"submission",
                                  seed_count=len(seeds),iterations=iterations,
                                  kappa=kappa,n_knots=n_knots)
    (root/"tables").mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{"comparison_id":i["comparison_id"],"gate":i["gate"],
                   "mean_fold_delta_r2":i["result"]["mean_fold_delta_r2"],
                   "school_pooled_delta_r2":i["result"]["school_pooled_delta_r2"],
                   "win_fraction":i["result"]["win_fraction"],
                   "corrected_t":i["result"]["corrected_t"],
                   "instrument_deltas":json.dumps(i["result"]["instrument_delta_r2"])}
                  for i in comparisons]).to_csv(
                      root/"tables"/"paired_comparison_audit.csv",index=False)
    result={
        "mode":mode,"decision":decision,"decision_explanation":explanation,
        "winner":winner if decision=="PROMOTE" else None,
        "fold_hash":fold_hash,"reference_oof_sha256":ref_sha,
        "reference":reference_summary,"model_metrics":[{"model":"E00",**reference_summary}]+rows,
        "comparisons":comparisons,"submission":submission,
        "run_ids":{k:v["run_id"] for k,v in runs.items()},
        "completed_all_E01_E09":len(chosen)==9,
        "screen_results_not_confirmatory":mode!="FULL",
        "finished_utc":utc_now(),
    }
    report=render_results(root,result,diagnostic)
    print(f"[M85] Mode: {mode} | Decision: {decision} | Winner: {result['winner']}")
    print(f"[M85] Private report: {report}")
    if submission: print(f"[M85] Verified candidate submission: {submission['file']}")
    return result


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root",default="/content/drive/MyDrive")
    parser.add_argument("--data-root",default="")
    parser.add_argument("--phase0-root",default="")
    parser.add_argument("--output-root",default="")
    parser.add_argument("--mode",choices=("SCREEN","FULL"),default="SCREEN")
    parser.add_argument("--models",nargs="+",default=list(CANDIDATES))
    parser.add_argument("--no-submit-export",action="store_true")
    parser.add_argument("--accept-other-frozen-hash",action="store_true",
                        help="Synthetic/independent benchmark only; documented nonofficial folds")
    args=parser.parse_args()
    execute(args.drive_root,args.data_root,args.phase0_root,args.output_root,
            args.mode,tuple(args.models),
            expected_fold_hash=None if args.accept_other_frozen_hash else EVAL_HASH_2026,
            export_final=not args.no_submit_export)


if __name__=="__main__":
    cli()
