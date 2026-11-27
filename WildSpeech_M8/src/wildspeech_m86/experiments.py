"""M8.6 M0++ scientific engine on frozen M8.4/M8.5 outer folds.

FULL requires exact source CSV hashes, M8.4 fold SHA, completed M8.5 E04,
and 11 preregistered arms. SCREEN is a one-repeat software smoke/exploration
and can NEVER promote or export a new challenger.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wildspeech_m84.data import load_official,locate_data_root,digest_bytes,sha_json,atomic_json
from wildspeech_m84.metrics import compare_runs,validate_oof,summarize_oof
from wildspeech_m84.ledger import Ledger,preregister,make_run_id,utc_now
from wildspeech_m84.runner import git_head
from wildspeech_m85.experiments import _load_reference,_checkpoint_ok
from wildspeech_m85.contracts import source_hash
from .contracts import FULL_FOLD_HASH,EXPERIMENTS,raw_train_fingerprint
from .modeling import SPECS,SCREEN_PRIORITY,CONFIRMATORY
from .nested_cv import FoldWorkspace
from .diagnostics import diagnostic_tables
from .report import write_report
from .submission import export_m86_candidate,export_e04_diagnostic


def plan():
    """Predeclare all comparisons and BOTH different decisions before model fit."""
    p=[]
    for model in EXPERIMENTS:
        p.append({
          "comparison_id":f"{model}_VS_E04","challenger":model,
          "baseline":"E04_M85","spec":SPECS[model],
          "full_required":True,
          "engineering_rule":{
              "positive_pooled_student_delta":True,"min_fold_wins":.65,
              "min_school_pooled_delta":-.005,"min_each_instrument_delta":-.010
          },
          "science_rule":{
              "min_fold_mean_delta":.010,"min_corrected_t":2.,
              "min_fold_wins":.60,
              "min_school_pooled_delta":-.005,"min_each_instrument_delta":-.010
          },
          "primary_science_test":model in CONFIRMATORY,
        })
    return p


def _load_bases(data,p0,p85,mode,expected_fold_hash,synthetic):
    if synthetic and (mode!="SCREEN" or expected_fold_hash is not None):
        raise ValueError("Synthetic fixture allowed only for SCREEN without official fold hash")
    f,reference,fold_hash,ref_sha,ref_metrics=_load_reference(
        p0,data,expected_fold_hash,strict_parity=not synthetic)
    if not synthetic and fold_hash!=FULL_FOLD_HASH:
        raise RuntimeError("M8.4 expected official fold hash was changed")
    m85_result_path=p85/"reports"/"M85_RESULTS.json"
    if not m85_result_path.is_file():
        raise FileNotFoundError(f"Completed M8.5 results unavailable: {m85_result_path}")
    result=json.loads(m85_result_path.read_text())
    if result["fold_hash"]!=fold_hash or result["reference_oof_sha256"]!=ref_sha:
        raise RuntimeError("M8.5 did not use the exact M8.4 reference / fold hash")
    if not synthetic and (result.get("mode")!="FULL" or
                           not result.get("completed_all_E01_E09")):
        raise RuntimeError("Full M8.5 nine-arm reference is required")
    m85_manifest=json.loads((p85/"manifest.json").read_text())
    if m85_manifest["train_screening_source_hash"]!=source_hash(
        data.train_students,data.train_clips):
        raise RuntimeError("Raw training screening inputs changed since M8.5")
    runid=result["run_ids"].get("E04")
    if not runid:raise RuntimeError("M8.5 E04 run absent")
    oof_file=p85/"runs"/runid/"oof_predictions_PRIVATE.parquet"
    e04=pd.read_parquet(oof_file)
    ledger=Ledger(p85)
    last=ledger.latest(runid)
    if last is None or last["status"]!="COMPLETE" or last.get("oof_sha256")!=digest_bytes(oof_file):
        raise RuntimeError("M8.5 E04 ledger/OOF checksum mismatch or incomplete run")
    if not synthetic:
        validate_oof(e04,f,data.ids,data.labels.to_numpy(float))
    if mode=="SCREEN":
        f=f[f.repeat==0].reset_index(drop=True)
        reference=reference[reference.repeat==0].copy()
        e04=e04[e04.repeat==0].copy()
    validate_oof(reference,f,data.ids,data.labels.to_numpy(float))
    validate_oof(e04,f,data.ids,data.labels.to_numpy(float))
    # Baselines are returned as read-only DataFrames, not mutated.
    return f,reference,e04,fold_hash,ref_sha,digest_bytes(oof_file),result,runid


def _write_parquet(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+".tmp.parquet")
    frame.to_parquet(temp,index=False)
    os.replace(temp,path)


def _scores(oof,folds,data):
    fm,pool,inst=summarize_oof(
        oof,folds,data.ids,data.labels.to_numpy(float),data.instruments)
    sr=fm[fm.scheme=="student"];ps=pool[pool.scheme=="student"]
    sc=pool[pool.scheme=="school"]
    return {
        "mean_fold_r2":float(sr.r2.mean()),
        "student_pooled_r2":float(ps.r2.mean()),
        "school_pooled_r2":float(sc.r2.mean()),
        "student_mae":float(ps.mae.mean()),
        "student_rmse":float(ps.rmse.mean()),
        "n_folds":len(folds),
    },fm,pool,inst


def _decisions(c):
    z=c["instrument_delta_r2"]
    robust=(c["school_pooled_delta_r2"] is not None and
            np.isfinite(c["school_pooled_delta_r2"]) and
            c["school_pooled_delta_r2"]>=-.005 and
            z and all(v is not None and np.isfinite(v) and v>=-.01 for v in z.values()))
    engineering=bool(robust and c["mean_student_pooled_delta_r2"]>0 and
                     c["win_fraction"]>=.65)
    science=bool(robust and c["mean_fold_delta_r2"]>=.01 and
                 c["corrected_t"]>=2 and c["win_fraction"]>=.60)
    return engineering,science


def _checkpoint_meta(part,row,ids,y):
    return _checkpoint_ok(part,row,ids,y)


def run_models(models,folds,data,mode,root,git_sha,fold_hash,
               e04_sha,raw_hash,rasch_iters,rasch_crossfit,cat_iterations,cat_seeds):
    ledger=Ledger(root)
    plans={}
    for name in models:
        spec=SPECS[name]
        config={"mode":mode,"model":name,"spec":spec,
                "rasch_iters":rasch_iters,"rasch_crossfit":rasch_crossfit,
                "cat_iterations":cat_iterations,"cat_seeds":list(cat_seeds),
                "inner_splits":3,"ridge_alphas":[1,10,100,1000],
                "baseline_e04_sha":e04_sha}
        feature_hash=sha_json({"raw":raw_hash,"spec":spec,
                               "versions":"m86-v1-student-crossfit-irt"})
        run_id=make_run_id("M86_"+name,git_sha,feature_hash,fold_hash,sha_json(config))
        dest=root/"runs"/run_id
        if ledger.latest(run_id) is None:
            ledger.transition(run_id,"PLANNED",model_id="M86_"+name,
                              git_sha=git_sha,feature_set_hash=feature_hash,
                              fold_hash=fold_hash,config=config,
                              baseline_e04_sha256=e04_sha,
                              mode=mode,source_hash=raw_hash)
        plans[name]={"run_id":run_id,"dir":dest}
    active={}
    for name,entry in plans.items():
        last=ledger.latest(entry["run_id"])
        if last["status"]=="COMPLETE":
            path=entry["dir"]/"oof_predictions_PRIVATE.parquet"
            if not path.is_file() or digest_bytes(path)!=last.get("oof_sha256"):
                raise RuntimeError(f"Completed M8.6 OOF changed: {name}")
            frame=pd.read_parquet(path)
            validate_oof(frame,folds,data.ids,data.labels.to_numpy(float))
            entry["oof"]=frame
        elif last["status"] in ("PLANNED","RUNNING"):
            if last["status"]=="PLANNED":
                ledger.transition(entry["run_id"],"RUNNING",started_utc=utc_now())
            active[name]=entry
        else:
            raise RuntimeError(f"Terminal run {name}: {last['status']}")
    if active:
        y=data.labels.to_numpy(float);ids=data.ids
        union=tuple(b for b in ("I1","I2","I3","I4","T1")
                    if any(b in SPECS[m].get("extra",()) for m in active))
        for fold in tqdm(folds.itertuples(),total=len(folds),
                         desc="M8.6 "+mode,unit="fold"):
            needed={}
            for model,entry in active.items():
                path=(entry["dir"]/"folds"/
                      f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet")
                if path.is_file():
                    previous=pd.read_parquet(path)
                    if not _checkpoint_meta(previous,fold,ids,y):
                        raise RuntimeError(f"Corrupted private fold checkpoint: {path}")
                else:
                    needed[model]=path
            if not needed:continue
            workspace=FoldWorkspace(
                data.train_students,data.train_clips,y,
                fold.train_pos,fold.valid_pos,
                seed=42+13*int(fold.repeat)+int(fold.fold),
                inner_k=3,extra=union,
                rasch_iters=rasch_iters,rasch_crossfit=rasch_crossfit)
            for name,path in needed.items():
                predictions,hyper=workspace.predict(
                    name,iterations=cat_iterations,seeds=cat_seeds)
                valid=np.asarray(fold.valid_pos,int)
                if len(predictions)!=len(valid) or not np.isfinite(predictions).all():
                    raise ValueError(f"Invalid predictions for {name}")
                frame=pd.DataFrame({
                    "scheme":fold.scheme,"repeat":int(fold.repeat),
                    "fold":int(fold.fold),"pos":valid,
                    "subject_id":[ids[i] for i in valid],
                    "y_true":y[valid],"y_pred":np.asarray(predictions,float)})
                _write_parquet(frame,path)
                atomic_json(path.with_suffix(".hyper.json"),hyper)
        for name,entry in active.items():
            parts=[]
            for fold in folds.itertuples():
                file=entry["dir"]/"folds"/f"{fold.scheme}_r{fold.repeat:02}_f{fold.fold}.parquet"
                p=pd.read_parquet(file)
                if not _checkpoint_meta(p,fold,ids,y):
                    raise ValueError("Incomplete or inconsistent saved fold prediction")
                parts.append(p)
            oof=pd.concat(parts,ignore_index=True)
            validate_oof(oof,folds,ids,y)
            output=entry["dir"]/"oof_predictions_PRIVATE.parquet"
            _write_parquet(oof,output)
            stats,fm,pool,inst=_scores(oof,folds,data)
            _write_parquet(fm,entry["dir"]/"fold_metrics.parquet")
            _write_parquet(pool,entry["dir"]/"pooled_metrics.parquet")
            _write_parquet(inst,entry["dir"]/"instrument_metrics.parquet")
            atomic_json(entry["dir"]/"metrics.json",stats)
            sha=digest_bytes(output)
            ledger.transition(entry["run_id"],"COMPLETE",
                              finished_utc=utc_now(),
                              oof_path=str(output),oof_sha256=sha,
                              mean_fold_r2=stats["mean_fold_r2"],
                              mean_pooled_oof_r2=stats["student_pooled_r2"])
            entry["oof"]=oof
    return plans


def execute(drive_root="/content/drive/MyDrive",data_root_override="",
            phase0_root_override="",phase1_root_override="",
            output_root_override="",mode="SCREEN",models=EXPERIMENTS,
            export_diagnostic_e04=False,export_candidate=True,
            rasch_iters=7,rasch_crossfit=3,
            cat_iterations=None,cat_seed_count=None,
            expected_fold_hash=FULL_FOLD_HASH,
            allow_synthetic_fixture=False):
    if mode not in ("SCREEN","FULL"):raise ValueError("Use SCREEN or FULL")
    if allow_synthetic_fixture and (mode!="SCREEN" or expected_fold_hash is not None or
                                    export_candidate or export_diagnostic_e04):
        raise ValueError("Synthetic fixture only SCREEN, no fixed hash, no exports")
    rootdrive=Path(drive_root)
    phase0=Path(phase0_root_override) if phase0_root_override else rootdrive/"WildSpeech_M84"/"phase0"
    phase1=Path(phase1_root_override) if phase1_root_override else rootdrive/"WildSpeech_M85"/"m0plus"
    base=Path(output_root_override) if output_root_override else rootdrive/"WildSpeech_M86"/"m0pp"
    root=base/mode.lower()
    dataset=load_official(locate_data_root(rootdrive,data_root_override))
    folds,ref,e04,fold_hash,ref_sha,e04_sha,m85_report,e04_id=_load_bases(
        dataset,phase0,phase1,mode,expected_fold_hash,allow_synthetic_fixture)
    chosen=tuple(models)
    if not chosen or len(set(chosen))!=len(chosen) or set(chosen)-set(EXPERIMENTS):
        raise ValueError("Unknown or duplicated experiment")
    if mode=="FULL" and set(chosen)!=set(EXPERIMENTS):
        raise ValueError("FULL requires all 11 preregistered M8.6 experiments")
    cat_iterations=(600 if mode=="FULL" else 30) if cat_iterations is None else int(cat_iterations)
    nseeds=(5 if mode=="FULL" else 1) if cat_seed_count is None else int(cat_seed_count)
    if mode=="FULL" and (cat_iterations!=600 or nseeds!=5 or
                          rasch_iters!=7 or rasch_crossfit!=3):
        raise RuntimeError("Locked FULL protocol may not be adjusted: use SCREEN for exploratory settings")
    if not 1<=nseeds<=5 or cat_iterations<5:
        raise ValueError("Invalid screening compute configuration")
    seeds=tuple(range(42,42+nseeds))
    root.mkdir(parents=True,exist_ok=True)
    preregister(root,plan())
    raw_hash=raw_train_fingerprint(dataset.train_students,dataset.train_clips)
    git_sha=git_head()
    manifest={"mode":mode,"fold_hash":fold_hash,"git_sha":git_sha,
              "reference_oof_sha256":ref_sha,"e04_oof_sha256":e04_sha,
              "e04_run_id":e04_id,"train_source_hash":raw_hash,
              "run_models":chosen,"cat_iterations":cat_iterations,
              "cat_seeds":seeds,"rasch_iters":rasch_iters,
              "rasch_crossfit":rasch_crossfit,
              "fitted_test_statistics":False,"executed_utc":utc_now()}
    # Manifest is immutable by configuration; never overwrite a different run.
    path=root/"manifest.json"
    if path.is_file():
        old=json.loads(path.read_text())
        invariant=("fold_hash","reference_oof_sha256","e04_oof_sha256","train_source_hash",
                   "cat_iterations","cat_seeds","rasch_iters","rasch_crossfit","mode")
        if any(old.get(k)!=manifest.get(k) for k in invariant):
            raise RuntimeError("M86 frozen execution manifest changed; select a versioned output directory")
    else:
        atomic_json(path,manifest)
    atlas=diagnostic_tables(root,dataset,ref,e04)
    runs=run_models(chosen,folds,dataset,mode,root,git_sha,fold_hash,e04_sha,
                    raw_hash,rasch_iters,rasch_crossfit,cat_iterations,seeds)
    baseline_stats,_,_,_=_scores(e04,folds,dataset)
    ref_stats,_,_,_=_scores(ref,folds,dataset)
    rows=[{"model":"E04",**baseline_stats},
          {"model":"E00",**ref_stats}]
    comparisons=[]
    for model in chosen:
        stats,_,_,_=_scores(runs[model]["oof"],folds,dataset)
        row={"model":model,**stats}
        if mode=="FULL":
            c=compare_runs(e04,runs[model]["oof"],folds,dataset.ids,
                           dataset.labels.to_numpy(float),dataset.instruments,
                           fold_hash,fold_hash,"E04",model)
            engineering,science=_decisions(c)
            row.update({
                "paired_fold_delta_r2":c["mean_fold_delta_r2"],
                "paired_student_pooled_delta_r2":c["mean_student_pooled_delta_r2"],
                "paired_school_delta_r2":c["school_pooled_delta_r2"],
                "win_fraction":c["win_fraction"],
                "corrected_t":c["corrected_t"],
                "submission_eligible":engineering,
                "science_gate":science,
            })
            comparisons.append({"comparison":f"{model}_VS_E04","baseline":"E04",
                                "candidate":model,"result":c,
                                "submission_eligible":engineering,
                                "science_gate":science})
        else:
            row.update({"paired_fold_delta_r2":np.nan,
                        "submission_eligible":False,
                        "science_gate":False})
        rows.append(row)
    if mode!="FULL":
        decision="SCREEN_EXPLORATORY";winner=None
        explanation="Only one of ten repeated outer rounds; no promotion or statistical claim."
    else:
        eligible=[r for r in rows if r.get("submission_eligible",False)]
        eligible.sort(key=lambda r:(-r["student_pooled_r2"],
                                     SCREEN_PRIORITY.index(r["model"])))
        winner=eligible[0]["model"] if eligible else None
        decision="ENGINEERING_CANDIDATE" if winner else "HOLD_E04"
        explanation=(
            f"{winner} passes preregistered engineering robustness and paired win gates. "
            "It was selected on the same repeated folds, so the corrected statistic "
            "is not an independent confirmatory result." if winner else
            "No M8.6 candidate passes the independent engineering robustness vetoes; "
            "M8.5 E04 remains an exploratory baseline, and M8.4 is the formally approved fallback.")
    diagnostic={}
    if export_diagnostic_e04:
        if mode!="FULL":raise RuntimeError("E04 diagnostic must be tied to completed FULL results")
        diagnostic=export_e04_diagnostic(dataset,base/"diagnostic_e04",fold_hash,
                                         e04_id,m85_report["decision"])
    candidate={}
    if mode=="FULL" and winner and export_candidate:
        candidate=export_m86_candidate(dataset,winner,root/"submission",
                    iterations=cat_iterations,seeds=seeds,
                    rasch_iters=rasch_iters,rasch_crossfit=rasch_crossfit)
    result={
        "mode":mode,"decision":decision,"decision_explanation":explanation,
        "winner":winner,"fold_hash":fold_hash,
        "reference_oof_sha256":ref_sha,"e04_oof_sha256":e04_sha,
        "n_student_folds":int(sum(folds.scheme=="student")),
        "n_school_folds":int(sum(folds.scheme=="school")),
        "reference":ref_stats,"e04":baseline_stats,
        "model_metrics":rows,"comparisons":comparisons,
        "run_ids":{m:runs[m]["run_id"] for m in chosen},
        "diagnostic_e04":diagnostic,
        "candidate_submission":candidate,
        "science_primary":{c["candidate"]:c["science_gate"] for c in comparisons
                           if c["candidate"] in CONFIRMATORY},
        "time_utc":utc_now(),"no_online_submission":True,
        "test_data_fitted":False,
    }
    report=write_report(root,result)
    print(f"[M86] mode={mode} decision={decision} winner={winner}")
    print(f"[M86] report={report}")
    return result


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive-root",default="/content/drive/MyDrive")
    p.add_argument("--data-root",default="")
    p.add_argument("--phase0-root",default="")
    p.add_argument("--phase1-root",default="")
    p.add_argument("--output-root",default="")
    p.add_argument("--mode",choices=["SCREEN","FULL"],default="SCREEN")
    p.add_argument("--models",nargs="+",default=list(EXPERIMENTS))
    p.add_argument("--diagnostic-e04-export",action="store_true")
    p.add_argument("--no-candidate-export",action="store_true")
    args=p.parse_args()
    execute(drive_root=args.drive_root,data_root_override=args.data_root,
            phase0_root_override=args.phase0_root,
            phase1_root_override=args.phase1_root,
            output_root_override=args.output_root,
            mode=args.mode,models=tuple(args.models),
            export_diagnostic_e04=args.diagnostic_e04_export,
            export_candidate=not args.no_candidate_export)

if __name__=="__main__":cli()
