"""Immutable clip inventory, frozen evaluation anchors and model provenance."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,os,platform,sys
import numpy as np
import pandas as pd
from wildspeech_m84.data import sha_json,atomic_json
from wildspeech_m86.contracts import FULL_FOLD_HASH

TASKS=("blending","deletion","sentence_repetition")
EXPERIMENTS=tuple(f"A{i:02d}" for i in range(17))
PHONE_REPO="drivendata/pasketti-phonetic"
WORD_REPO="drivendata/pasketti-word"
ASSETS={
 "pasketti_phonetic":{"repo":PHONE_REPO,"license":"bigcode-openrail-m","eligibility":"REVIEW_REQUIRED","required":True},
 "pasketti_word":{"repo":WORD_REPO,"license":"bigcode-openrail-m","eligibility":"REVIEW_REQUIRED","required":False},
 "silero_vad":{"package":"silero-vad","license":"VERIFY_RELEASE","eligibility":"VERIFY_PIN","required":False},
 "eng_to_ipa":{"package":"eng_to_ipa","license":"VERIFY_RELEASE","eligibility":"VERIFY_PIN","required":False},
}
@dataclass(frozen=True)
class Config:
    drive_root:str="/content/drive/MyDrive"
    data_root:str=""
    output_root:str=""
    train_tar:str=""
    test_tar:str=""
    train_audio_dir:str=""
    test_audio_dir:str=""
    phonetic_revision:str=""
    word_revision:str=""
    acknowledge_license_review:bool=False
    batch_size:int=2
    max_seconds:float=30.
    layer_indices:tuple[int,...]=(6,7,8,9,10)
    seed:int=42
    mode:str="SCREEN"
    def validate(self):
        if self.mode not in ("SMOKE","SCREEN","FULL"):raise ValueError("Invalid mode")
        if not self.acknowledge_license_review:raise PermissionError(
            "Read the exact model license and competition rules and set "
            "acknowledge_license_review=True before external model inference")
        if self.batch_size<1 or not 0<self.max_seconds<=60:raise ValueError("Invalid audio settings")
        return self
    def output(self):
        return Path(self.output_root) if self.output_root else Path(self.drive_root)/"WildSpeech_M87"/"speech"

def stable_uid(split,filename):
    if split not in ("train","test") or not str(filename):raise ValueError("Invalid ID")
    return split+"_"+hashlib.sha256((split+"\0"+str(filename)).encode()).hexdigest()[:24]

def clip_inventory(data):
    tables=[]
    for split,clips in (("train",data.train_clips),("test",data.test_clips)):
        d=clips.copy()
        if d.filename.isna().any() or d.filename.duplicated().any() or "expected_text" not in d:
            raise ValueError("Missing or duplicate filename or expected_text")
        d["split"]=split;d["uid"]=[stable_uid(split,f) for f in d.filename]
        d["subject_id"]=d.subject_id.astype(str)
        d["filename"]=d.filename.astype(str)
        d["duration_sec"]=pd.to_numeric(d.duration_sec,errors="coerce")
        if not np.isfinite(d.duration_sec.to_numpy(float)).all() or (d.duration_sec<0).any():
            raise ValueError("Invalid clip duration")
        cols=["uid","split","filename","subject_id","task","expected_text","duration_sec"]
        if "item_score" in d:cols.append("item_score")
        tables.append(d[cols])
    out=pd.concat(tables,ignore_index=True)
    if out.uid.duplicated().any():raise ValueError("Clip ID collision")
    return out

def inventory_hash(d):
    cols=["uid","split","filename","subject_id","task","expected_text","duration_sec"]
    return sha_json(d[cols].sort_values("uid").fillna("").to_dict("records"))

def validate_baselines(data,phase0_root,m86_full_root=None,official=True):
    from wildspeech_m85.experiments import _load_reference
    from wildspeech_m84.metrics import validate_oof
    from wildspeech_m84.data import digest_bytes
    folds,reference,fold_hash,ref_sha,_=_load_reference(
        Path(phase0_root),data,FULL_FOLD_HASH if official else None,
        strict_parity=official)
    inventory=clip_inventory(data)
    stats={"fold_hash":fold_hash,"reference_oof_sha256":ref_sha,
           "inventory_sha256":inventory_hash(inventory),
           "n_train_students":len(data.train_students),"n_test_students":len(data.test_students),
           "n_train_clips":len(data.train_clips),"n_test_clips":len(data.test_clips)}
    if m86_full_root:
        import json
        root=Path(m86_full_root)
        summary=json.loads((root/"reports"/"M86_RESULTS.json").read_text())
        if summary["fold_hash"]!=fold_hash or summary["reference_oof_sha256"]!=ref_sha:
            raise RuntimeError("M8.6 baseline does not match frozen M8.4")
        if official and summary["mode"]!="FULL":raise RuntimeError("Need M8.6 FULL")
        for model in ("I4","I5"):
            id=summary["run_ids"][model]
            file=root/"runs"/id/"oof_predictions_PRIVATE.parquet"
            frame=pd.read_parquet(file)
            validate_oof(frame,folds,data.ids,data.labels.to_numpy(float))
            stats[model+"_oof_sha256"]=digest_bytes(file)
            stats[model+"_oof_path"]=str(file)
    return folds,reference,inventory,stats

def write_provenance(config,output,revisions=None):
    import importlib.metadata as im
    versions={}
    for k in ["torch","transformers","soundfile","librosa","numpy",
              "silero-vad","qwen-asr","scikit-learn","scipy"]:
        try:versions[k]=im.version(k)
        except im.PackageNotFoundError:versions[k]=None
    doc={"assets":ASSETS,"model_revisions":revisions or {},
         "license_review_acknowledged":config.acknowledge_license_review,
         "not_automatic_eligibility_confirmation":True,
         "competition_rules":"https://platform.k12-ai-infrastructure.org/competitions/4/literacy-speech-modeling/rules/",
         "python":sys.version,"platform":platform.platform(),
         "git_sha":os.getenv("WILDSPEECH_COMMIT","UNKNOWN"),
         "versions":versions,"config":asdict(config),
         "no_audio_uploaded":True,"no_test_model_adaptation":True}
    atomic_json(Path(output)/"manifests"/"model_provenance_PRIVATE.json",doc)
    return doc
