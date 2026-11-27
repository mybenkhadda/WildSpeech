"""Resumable, private clip-level GPU inference, evidence and optional word cache.

Audio never leaves local machine/Colab VM. Only model weights are fetched.
All 24,518 expected clips must appear as VALID or FAILED; never silently skip.
Each completed shard is atomic; on restart its clip IDs are reused without
decoding. Posterior and pooled WavLM tensors are stored in compressed shards.
"""
from __future__ import annotations
import json,hashlib,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from wildspeech_m84.data import atomic_json,digest_bytes,sha_json
from .contracts import Config,inventory_hash
from .audio_io import AudioSource
from .vad_quality import VAD,quality
from .child_localizer import localize,segment_frame_bounds
from .task_scorers import clip_task_features,word_partial_credit
from .targets import TargetRegistry,normalize_text

def _atomic_parquet(frame,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".partial")
    frame.to_parquet(tmp,index=False);os.replace(tmp,p)

def _atomic_npz(path,arrays):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".partial")
    with tmp.open("wb") as fp:np.savez_compressed(fp,**arrays)
    os.replace(tmp,p)

def _fixed_projection(v,layer,n=24):
    """Deterministic, NOT learned from train/test; full embedding also cached."""
    x=np.asarray(v,float);rng=np.random.default_rng(5000+int(layer))
    proj=rng.choice([-1.,1.],size=(len(x),n))
    return (x@proj/np.sqrt(len(x))).astype(np.float32)

def create_sources(config,output):
    c=config
    train=AudioSource(c.train_tar,c.train_audio_dir,
                      Path(output)/"indexes"/"train_tar.parquet" if c.train_tar else None)
    test=AudioSource(c.test_tar,c.test_audio_dir,
                     Path(output)/"indexes"/"test_tar.parquet" if c.test_tar else None)
    return {"train":train,"test":test}

def _one(row,wave,model_result,target,vad_spans,qual,original_sr,blank):
    logp=model_result.logprobs
    loc=localize(row.uid,row.task,vad_spans,logp,model_result.duration_s,
                 target.get("ids",[]),blank)
    loc["duration_s"]=model_result.duration_s
    a,b=segment_frame_bounds(loc["response_start_s"],loc["response_end_s"],
                             model_result.duration_s,len(logp))
    ra,rb=segment_frame_bounds(loc["random_start_s"],loc["random_end_s"],
                               model_result.duration_s,len(logp))
    feats=clip_task_features(row.task,target.get("ids",[]),logp,logp[a:b],
                             logp[ra:rb],blank,loc,row.expected_text)
    # Main feature frame contains only compact scalar features.
    record={"uid":row.uid,"split":row.split,"filename":row.filename,
            "subject_id":row.subject_id,"task":row.task,
            "item_score":getattr(row,"item_score",np.nan),
            "expected_text":row.expected_text,
            "status":"VALID","phonetic_transcript":model_result.transcript,
            "phonetic_revision":model_result.revision,
            "target_ipa":target.get("ipa",""),
            "target_valid":bool(target.get("valid")),
            "target_failure":target.get("reason","UNKNOWN"),
            "original_sr":int(original_sr),"n_ctc_frames":len(logp),
            **{k:float(v) for k,v in qual.items()},**loc,**feats}
    for layer,vec in model_result.hidden.items():
        sketch=_fixed_projection(vec,layer)
        for i,value in enumerate(sketch):
            record[f"layer_{int(layer):02d}_{i:02d}"]=float(value)
    return record

def _load_existing(root):
    shards=sorted((root/"shards").glob("*.parquet"))
    seen={}
    for file in shards:
        frame=pd.read_parquet(file,columns=["uid","status"])
        if frame.uid.duplicated().any():raise RuntimeError(f"Duplicate clip in {file}")
        for uid in frame.uid:
            if uid in seen:raise RuntimeError(f"Duplicate cached clip across shards: {uid}")
            seen[uid]=file.name
        arrayfile=file.with_suffix(".npz")
        if not arrayfile.is_file():raise RuntimeError(f"Missing frame tensors for {file}")
    return seen

def _run_slice(rows,sources,phone,vad,registry,target_map):
    work=[];out=[];tensors={}
    for row in rows:
        try:
            wave,sr,audio_sha,original_sr=sources[row.split].decode(row.filename)
            spans=vad.segments(wave,sr)
            q=quality(wave,sr)
            work.append((row,wave,spans,q,original_sr,audio_sha))
        except Exception as e:
            out.append({"uid":row.uid,"split":row.split,"filename":row.filename,
                        "subject_id":row.subject_id,"task":row.task,
                        "item_score":getattr(row,"item_score",np.nan),
                        "status":"FAILED","failure_stage":"DECODE_VAD",
                        "failure_reason":str(e)[:350]})
    if not work:return out,tensors
    try:preds=phone.infer_batch([r[1] for r in work])
    except Exception as exception:
        # A bad single clip must not kill the other entries in its batch.
        preds=[]
        for sample in work:
            try:preds.append(phone.infer_one(sample[1]))
            except Exception as e:preds.append(e)
    for i,(row,wave,spans,q,original_sr,audio_sha) in enumerate(work):
        pred=preds[i]
        if isinstance(pred,Exception):
            out.append({"uid":row.uid,"split":row.split,"filename":row.filename,
                        "subject_id":row.subject_id,"task":row.task,
                        "status":"FAILED","failure_stage":"PHONETIC_ASR",
                        "failure_reason":str(pred)[:350]})
            continue
        try:
            target=target_map.get((row.task,normalize_text(row.expected_text)),
                     {"valid":False,"ids":[],"reason":"NO_TARGET"})
            result=_one(row,wave,pred,target,spans,q,original_sr,phone.blank)
            result["audio_sha256"]=audio_sha
            out.append(result)
            tensors["posterior_"+row.uid]=pred.logprobs.astype(np.float16)
            for layer,vec in pred.hidden.items():
                tensors["hidden_"+str(layer)+"_"+row.uid]=vec.astype(np.float16)
        except Exception as e:
            out.append({"uid":row.uid,"split":row.split,"filename":row.filename,
                        "subject_id":row.subject_id,"task":row.task,
                        "status":"FAILED","failure_stage":"FEATURES",
                        "failure_reason":str(e)[:350]})
    return out,tensors

def extract(config,inventory,phone=None,vad=None,shard_size=48,max_clips=0,
            vad_method="silero"):
    config.validate()
    root=config.output();root.mkdir(parents=True,exist_ok=True)
    # Resolve the immutable MODEL revision *before* deciding whether a shard
    # is reusable. An unpinned moving "main" revision must never share
    # posterior caches with a different set of model weights.
    if phone is None:
        from .phonetic_asr import PhoneticASR
        phone=PhoneticASR(revision=config.phonetic_revision,
                          layers=config.layer_indices,max_seconds=config.max_seconds)
    resolved_revision=str(getattr(phone,"revision",config.phonetic_revision))
    if not resolved_revision:
        raise RuntimeError("Model revision must be pinned before extraction")
    manifest=root/"manifests"/"extraction.json"
    fingerprint=sha_json({"inventory":inventory_hash(inventory),
        "phonetic_revision":resolved_revision,
        "max_seconds":config.max_seconds,"layers":config.layer_indices,
        "vad_method":vad_method,"cache_contract":"m87-v2"})
    if manifest.is_file():
        old=json.loads(manifest.read_text())
        if old["fingerprint"]!=fingerprint:raise RuntimeError(
            "Cache input/model/config changed. Select a NEW output_root; do not mix versions.")
    else:atomic_json(manifest,{"fingerprint":fingerprint,"expected":len(inventory),
                               "contract":"m87-v2","resolved_phonetic_revision":resolved_revision})
    if vad is None:vad=VAD(vad_method)
    sources=create_sources(config,root)
    registry=TargetRegistry(phone.processor.tokenizer)
    targets=registry.make(inventory)
    target_path=root/"manifests"/"target_registry_PRIVATE.parquet"
    _atomic_parquet(targets,target_path)
    target_map=registry.lookups(targets)
    seen=_load_existing(root)
    todo=inventory[~inventory.uid.isin(seen)].sort_values(
        ["split","duration_sec","uid"],kind="stable")
    if max_clips:todo=todo.head(int(max_clips))
    shards_dir=root/"shards"
    counter=0
    existing=[int(p.stem.split("_")[0]) for p in shards_dir.glob("*.parquet")
              if p.stem.split("_")[0].isdigit()] if shards_dir.exists() else []
    start=max(existing,default=-1)+1
    for begin in tqdm(range(0,len(todo),shard_size),desc="M87 private GPU shards"):
        subset=todo.iloc[begin:begin+shard_size]
        rows=[];frames={}
        for offset in range(0,len(subset),config.batch_size):
            batch=list(subset.iloc[offset:offset+config.batch_size].itertuples(index=False))
            result,tensors=_run_slice(batch,sources,phone,vad,registry,target_map)
            rows.extend(result);frames.update(tensors)
        if len(rows)!=len(subset) or set(r["uid"] for r in rows)!=set(subset.uid):
            raise RuntimeError("Missing/duplicate clip in GPU shard")
        shard_num=start+counter;counter+=1
        base=shards_dir/f"{shard_num:05d}_phonetic"
        _atomic_npz(base.with_suffix(".npz"),frames)
        _atomic_parquet(pd.DataFrame(rows),base.with_suffix(".parquet"))
    return summarize_cache(root,inventory,strict=not bool(max_clips))

def summarize_cache(root,inventory,strict=True):
    root=Path(root)
    parts=[pd.read_parquet(p) for p in sorted((root/"shards").glob("*.parquet"))]
    if not parts:raise RuntimeError("No cached audio shards")
    rows=pd.concat(parts,ignore_index=True)
    if rows.uid.duplicated().any():raise RuntimeError("Duplicate cached UID")
    unexpected=set(rows.uid)-set(inventory.uid)
    if unexpected:raise RuntimeError("Cache contains foreign input clips")
    counts={"expected":len(inventory),"accounted":len(rows),
            "valid":int((rows.status=="VALID").sum()),
            "failed":int((rows.status=="FAILED").sum()),
            "missing":int(len(inventory)-len(rows))}
    if strict and counts["missing"]:raise RuntimeError(f"Incomplete audio accounting: {counts}")
    _atomic_parquet(rows,root/"features"/"phonetic_clip_features_PRIVATE.parquet")
    atomic_json(root/"manifests"/"coverage.json",counts)
    return counts

def extract_words(config,inventory,max_clips=0,batch_size=8,
                  sentence_only=True):
    """Separate GPU phase: avoid loading Pasketti Word and WavLM simultaneously."""
    config.validate()
    from .word_asr import WordASR
    root=config.output();out=root/"features"/"word_asr_PRIVATE.parquet"
    old=pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["uid"])
    have=set(old.uid)
    todo=inventory[(inventory.task=="sentence_repetition") if sentence_only
                   else np.ones(len(inventory),dtype=bool)]
    todo=todo[~todo.uid.isin(have)].sort_values(["split","duration_sec"])
    if max_clips:todo=todo.head(int(max_clips))
    sources=create_sources(config,root)
    model=WordASR(revision=config.word_revision,batch_size=batch_size)
    records=[]
    for start in tqdm(range(0,len(todo),batch_size),desc="Pasketti Word local"):
        subset=todo.iloc[start:start+batch_size]
        work=[]
        for row in subset.itertuples(index=False):
            try:
                wave,_,_,_=sources[row.split].decode(row.filename)
                work.append((row,wave))
            except Exception as e:
                records.append({"uid":row.uid,"word_status":"FAILED","word_failure":str(e)[:250]})
        if not work:continue
        try:decoded=model.transcribe([x[1] for x in work])
        except Exception as e:
            for row,_ in work:
                records.append({"uid":row.uid,"word_status":"FAILED","word_failure":str(e)[:250]})
            continue
        for (row,_),hyp in zip(work,decoded):
            stats=word_partial_credit(row.expected_text,hyp["word_transcript"])
            records.append({"uid":row.uid,"word_status":"VALID",**hyp,**stats})
        # Save on every batch, not only at end of long Qwen inference.
        tmp=pd.concat([old,pd.DataFrame(records)],ignore_index=True)
        if tmp.uid.duplicated().any():raise RuntimeError("Repeated Word ASR UID")
        _atomic_parquet(tmp,out)
    if not out.exists():_atomic_parquet(old,out)
    return {"n_completed":int(len(pd.read_parquet(out))),"path":str(out)}

def load_features(root,word_optional=True):
    root=Path(root)
    f=pd.read_parquet(root/"features"/"phonetic_clip_features_PRIVATE.parquet")
    if word_optional and (root/"features"/"word_asr_PRIVATE.parquet").is_file():
        w=pd.read_parquet(root/"features"/"word_asr_PRIVATE.parquet")
        # Preserve raw word result only once; no prompt-dependent ASR decoder.
        f=f.merge(w,on="uid",how="left",validate="one_to_one")
    return f
