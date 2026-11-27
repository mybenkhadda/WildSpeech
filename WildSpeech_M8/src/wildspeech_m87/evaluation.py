"""A00–A16 original-fold controlled outcome experiments, checkpointed to Drive.

All baselines are READ ONLY. Each outer-fold workspace includes frozen
screening I5, student-local speech aggregations and three INNER STUDENT
folds with independent feature fitting. Fold-specific clip correctness
training is grouped by student. NO test rows participate in any fit.
"""
from __future__ import annotations
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from wildspeech_m84.data import load_official,locate_data_root,atomic_json,digest_bytes,sha_json
from wildspeech_m84.ledger import Ledger,preregister,make_run_id,utc_now
from wildspeech_m84.metrics import validate_oof,summarize_oof,compare_runs
from wildspeech_m84.runner import git_head
from .contracts import EXPERIMENTS,validate_baselines,clip_inventory,inventory_hash
from .student_aggregation import FAMILIES,available_families
from .fold_models import FoldStudy,CONTROL_ARMS,PRIMARY,audio_contract
from .gpu_cache import load_features
from .diagnostics import extraction_atlas,prediction_atlas

def experiment_plan():
    return [{
        "comparison_id":arm+"_VS_I5","candidate":arm,"baseline":"I5_M86",
        "feature_family":FAMILIES[arm],
        "model":"stack_two_head" if arm=="A16" else
                "residual_shrink" if arm=="A14" else "joint_ridge",
        "negative_control":arm in CONTROL_ARMS,
        "main_hypothesis":arm in PRIMARY,
        "scientific_threshold":{"delta_fold_r2":.01,"corrected_t":2.,
              "student_fold_wins":.60,"min_school_delta":-.005,
              "min_instrument_delta":-.010},
        "engineering_threshold":{"student_delta_positive":True,
             "student_fold_wins":.65,"min_school_delta":-.005,
             "min_instrument_delta":-.010}
    } for arm in EXPERIMENTS]

def _write_parquet(x,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".partial");x.to_parquet(tmp,index=False);os.replace(tmp,p)

def load_anchor(root,data,folds,model):
    from wildspeech_m84.metrics import validate_oof
    summary=json.loads((Path(root)/"reports"/"M86_RESULTS.json").read_text())
    runid=summary["run_ids"][model]
    file=Path(root)/"runs"/runid/"oof_predictions_PRIVATE.parquet"
    frame=pd.read_parquet(file)
    validate_oof(frame,folds,data.ids,data.labels.to_numpy(float))
    return frame,digest_bytes(file)

def score_oof(frame,folds,data):
    f,p,i=summarize_oof(frame,folds,data.ids,data.labels.to_numpy(float),data.instruments)
    return {
       "mean_fold_r2":float(f[f.scheme=="student"].r2.mean()),
       "student_pooled_r2":float(p[p.scheme=="student"].r2.mean()),
       "school_pooled_r2":float(p[p.scheme=="school"].r2.mean()),
       "student_mae":float(p[p.scheme=="student"].mae.mean()),
       "student_rmse":float(p[p.scheme=="student"].rmse.mean())
    },f,p,i

def decision(c,negative=False):
    if negative:return False,False
    deltas=c.get("instrument_delta_r2") or {}
    sch=c.get("school_pooled_delta_r2")
    robust=(sch is not None and np.isfinite(sch) and sch>=-.005
            and bool(deltas)
            and all(z is not None and np.isfinite(z) and z>=-.01
                    for z in deltas.values()))
    engineering=bool(robust and c["mean_student_pooled_delta_r2"]>0
                     and c["win_fraction"]>=.65)
    scientific=bool(robust and c["mean_fold_delta_r2"]>=.01
                    and c["corrected_t"]>=2. and c["win_fraction"]>=.60)
    return engineering,scientific

def _joined_features(inventory,cache):
    if cache.uid.duplicated().any():raise RuntimeError("Duplicate cached UID")
    keep=[c for c in cache if c not in ("filename","split","subject_id","task",
                    "expected_text","duration_sec","item_score")]
    # Use official item_score and metadata rather than any cached variant.
    d=inventory.merge(cache[keep],on="uid",how="left",validate="one_to_one")
    if "status" not in d:raise RuntimeError("No audio extraction status")
    if d.uid.isna().any():raise RuntimeError("Missing UID")
    for c in ("item_score","expected_text","duration_sec","filename"):
        if c not in d:raise RuntimeError(f"Missing official column {c}")
    d["status"]=d.status.fillna("FAILED")
    return d

def _checkpoint_matches(part,fold,ids,y):
    v=np.asarray(fold.valid_pos,int)
    return (len(part)==len(v) and part.pos.astype(int).tolist()==v.tolist()
            and part.subject_id.astype(str).tolist()==[ids[k] for k in v]
            and np.array_equal(part.y_true.to_numpy(float),y[v])
            and np.isfinite(part.y_pred.to_numpy(float)).all())

def evaluate(drive_root="/content/drive/MyDrive",data_root="",cache_root="",
             output_root="",phase0_root="",m86_root="",
             mode="SCREEN",models=EXPERIMENTS,rasch_iters=7,
             require_word=False,official=True):
    if mode not in ("SCREEN","FULL"):raise ValueError("Use SCREEN or FULL")
    drive=Path(drive_root)
    data=load_official(locate_data_root(drive,data_root))
    p0=Path(phase0_root) if phase0_root else drive/"WildSpeech_M84"/"phase0"
    m86=Path(m86_root) if m86_root else drive/"WildSpeech_M86"/"m0pp"/"full"
    clip_root=Path(cache_root) if cache_root else drive/"WildSpeech_M87"/"speech"
    result=Path(output_root) if output_root else clip_root/"evaluations"/mode.lower()
    folds,reference,inventory,bases=validate_baselines(data,p0,m86,official)
    baseline_i5,i5_sha=load_anchor(m86,data,folds,"I5")
    baseline_i4,i4_sha=load_anchor(m86,data,folds,"I4")
    cache=load_features(clip_root)
    features=_joined_features(inventory,cache)
    audio_contract(features,inventory)
    chosen=tuple(models)
    if not chosen or len(set(chosen))!=len(chosen) or set(chosen)-set(EXPERIMENTS):
        raise ValueError("Unknown or repeated model arm")
    if mode=="FULL" and set(chosen)!=set(EXPERIMENTS):
        raise ValueError("FULL must include A00–A16, including negative controls")
    if mode=="SCREEN":
        folds=folds[folds.repeat==0].reset_index(drop=True)
        reference=reference[reference.repeat==0].copy()
        baseline_i5=baseline_i5[baseline_i5.repeat==0].copy()
        baseline_i4=baseline_i4[baseline_i4.repeat==0].copy()
    status=available_families(features[features.status=="VALID"])
    unavailable={}
    for arm in chosen:
        if arm=="A07" and (("word_coverage" not in features) or
                           not pd.to_numeric(features.word_coverage,errors="coerce").notna().any()):
            unavailable[arm]="PASKETTI_WORD_NOT_EXECUTED"
        elif not status[arm]["eligible"]:
            unavailable[arm]="REQUIRED_AUDIO_FEATURES_NOT_AVAILABLE"
    if require_word and "A07" in unavailable:
        raise RuntimeError("Word ASR required but sentence-repetition word cache is absent")
    result.mkdir(parents=True,exist_ok=True)
    preregister(result,experiment_plan())
    provenance={"mode":mode,"git_sha":git_head(),
                "fold_hash":bases["fold_hash"],
                "i5_oof_sha256":i5_sha,"i4_oof_sha256":i4_sha,
                "reference_sha256":bases["reference_oof_sha256"],
                "source_inventory_sha256":inventory_hash(inventory),
                "clip_status_hash":sha_json(features[["uid","status"]].sort_values("uid").to_dict("records")),
                "audio_extract_contract":"m87-v1",
                "model_selection_policy":"student_oof_plus_school_and_instrument_veto",
                "unavailable":unavailable,
                "config":{"rasch_iters":rasch_iters,"models":list(chosen),
                          "mode":mode,"official":official}}
    manifest=result/"manifest.json"
    if manifest.is_file():
        old=json.loads(manifest.read_text())
        for k in ("fold_hash","i5_oof_sha256","source_inventory_sha256",
                  "clip_status_hash","config"):
            if old.get(k)!=provenance.get(k):
                raise RuntimeError("M8.7 immutable experiment manifest changed")
    else:atomic_json(manifest,provenance)
    extraction_atlas(result,inventory,features)
    ledger=Ledger(result)
    rows={}
    active=[m for m in chosen if m not in unavailable]
    for arm in active:
        codehash=sha_json({"git":provenance["git_sha"],"arm":arm,
                           "audio":provenance["clip_status_hash"],"i5":i5_sha})
        id=make_run_id("M87_"+arm,provenance["git_sha"],codehash,
                       bases["fold_hash"],sha_json(provenance["config"]))
        folder=result/"runs"/id
        if ledger.latest(id) is None:
            ledger.transition(id,"PLANNED",model_id="M87_"+arm,
                              git_sha=provenance["git_sha"],
                              fold_hash=bases["fold_hash"],
                              feature_set_hash=codehash,
                              baseline_oof_sha256=i5_sha,mode=mode)
        rows[arm]={"id":id,"folder":folder}
    incomplete={}
    for arm,entry in rows.items():
        last=ledger.latest(entry["id"])
        path=entry["folder"]/"oof_predictions_PRIVATE.parquet"
        if last["status"]=="COMPLETE":
            if not path.is_file() or digest_bytes(path)!=last.get("oof_sha256"):
                raise RuntimeError(f"Completed candidate changed {arm}")
            oof=pd.read_parquet(path)
            validate_oof(oof,folds,data.ids,data.labels.to_numpy(float))
            entry["oof"]=oof
        elif last["status"] in ("PLANNED","RUNNING"):
            if last["status"]=="PLANNED":
                ledger.transition(entry["id"],"RUNNING",started_utc=utc_now())
            incomplete[arm]=entry
        else:raise RuntimeError(f"Terminal arm status {arm}/{last['status']}")
    y=data.labels.to_numpy(float);ids=data.ids
    if incomplete:
        for fold in tqdm(folds.itertuples(),total=len(folds),
                         desc="M87 matched folds (shared inner cache)"):
            needed={}
            for arm,entry in incomplete.items():
                path=entry["folder"]/"folds"/(
                    f"{fold.scheme}_r{int(fold.repeat):02}_f{int(fold.fold)}.parquet")
                if path.is_file():
                    part=pd.read_parquet(path)
                    if not _checkpoint_matches(part,fold,ids,y):
                        raise RuntimeError(f"Corrupt private checkpoint {path}")
                else:needed[arm]=path
            if not needed:continue
            # All arms reuse EXACT SAME fold-fit M8.6 + speech representations
            # and student-grouped clip correctness predictions.
            study=FoldStudy(data.train_students,data.train_clips,features,y,
                           fold.train_pos,fold.valid_pos,
                           seed=42+int(fold.repeat)*13+int(fold.fold),
                           rasch_iters=rasch_iters,inner_k=3)
            for arm,path in needed.items():
                pred,settings=study.predict(arm)
                v=np.asarray(fold.valid_pos,int)
                if len(pred)!=len(v) or not np.isfinite(pred).all():
                    raise RuntimeError(f"Nonfinite or misaligned prediction {arm}")
                frame=pd.DataFrame({"scheme":fold.scheme,"repeat":int(fold.repeat),
                   "fold":int(fold.fold),"pos":v,
                   "subject_id":[ids[k] for k in v],"y_true":y[v],
                   "y_pred":np.asarray(pred,float)})
                _write_parquet(frame,path)
                atomic_json(path.with_suffix(".settings.json"),settings)
        for arm,entry in incomplete.items():
            parts=[]
            for fold in folds.itertuples():
                path=entry["folder"]/"folds"/(
                    f"{fold.scheme}_r{int(fold.repeat):02}_f{int(fold.fold)}.parquet")
                part=pd.read_parquet(path)
                if not _checkpoint_matches(part,fold,ids,y):
                    raise RuntimeError(f"Missing saved prediction {path}")
                parts.append(part)
            full=pd.concat(parts,ignore_index=True)
            validate_oof(full,folds,ids,y)
            path=entry["folder"]/"oof_predictions_PRIVATE.parquet"
            _write_parquet(full,path)
            scores,fm,pooled,inst=score_oof(full,folds,data)
            for file,table in (("fold_metrics.parquet",fm),
                               ("pooled_metrics.parquet",pooled),
                               ("instrument_metrics.parquet",inst)):
                _write_parquet(table,entry["folder"]/file)
            atomic_json(entry["folder"]/"metrics.json",scores)
            sha=digest_bytes(path)
            ledger.transition(entry["id"],"COMPLETE",finished_utc=utc_now(),
                              oof_path=str(path),oof_sha256=sha,
                              student_pooled_r2=scores["student_pooled_r2"])
            entry["oof"]=full
    row_metrics=[]
    for mid,frame in (("I5",baseline_i5),("I4",baseline_i4),("E00",reference)):
        score,_,_,_=score_oof(frame,folds,data);row_metrics.append({"model":mid,**score})
    comparisons=[]
    for arm,entry in rows.items():
        score,_,_,_=score_oof(entry["oof"],folds,data)
        record={"model":arm,**score,"engineering_eligible":False,
                "science_gate":False}
        if mode=="FULL":
            cmp=compare_runs(baseline_i5,entry["oof"],folds,ids,y,data.instruments,
                             bases["fold_hash"],bases["fold_hash"],"M86_I5",arm)
            engineering,science=decision(cmp,negative=arm in CONTROL_ARMS)
            record.update({"mean_fold_delta_r2":cmp["mean_fold_delta_r2"],
                           "corrected_t":cmp["corrected_t"],
                           "win_fraction":cmp["win_fraction"],
                           "school_delta_r2":cmp["school_pooled_delta_r2"],
                           "engineering_eligible":engineering,
                           "science_gate":science})
            comparisons.append({"candidate":arm,"result":cmp,
                                "engineering_eligible":engineering,
                                "science_gate":science})
        row_metrics.append(record)
    for arm,why in unavailable.items():
        row_metrics.append({"model":arm,"status":"SKIPPED","reason":why})
    candidates=[r for r in row_metrics if r.get("engineering_eligible")]
    candidates.sort(key=lambda r:(-r["student_pooled_r2"],r["model"]))
    winner=candidates[0]["model"] if candidates else None
    decision_status=("SCREEN_ONLY" if mode=="SCREEN" else
                     "ENGINEERING_CANDIDATE" if winner else "NO_PROMOTION")
    # An extraction/paired study never silently exports a leaderboard CSV.
    result_data={"mode":mode,"decision":decision_status,"winner":winner,
       "fold_hash":bases["fold_hash"],"n_student_folds":int(sum(folds.scheme=="student")),
       "n_school_folds":int(sum(folds.scheme=="school")),
       "i5_oof_sha256":i5_sha,"i4_oof_sha256":i4_sha,
       "reference_oof_sha256":bases["reference_oof_sha256"],
       "clip_cache_coverage":json.loads((clip_root/"manifests"/"coverage.json").read_text()),
       "available":list(rows),"skipped":unavailable,
       "model_metrics":row_metrics,"comparisons":comparisons,
       "run_ids":{k:v["id"] for k,v in rows.items()},
       "no_public_submission":True,"time_utc":utc_now()}
    prediction_atlas(result,
          [r for r in row_metrics if r.get("status")!="SKIPPED"],
          comparisons,baseline_i5,data=data,clip_features=features)
    from .report import write_report
    write_report(result,result_data)
    return result_data
