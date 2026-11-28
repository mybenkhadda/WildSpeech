"""Resumable, private clip-level GPU inference, evidence and optional word cache.

Audio never leaves local machine/Colab VM. Only model weights are fetched.
All 24,518 expected clips must appear as VALID or FAILED; never silently skip.
Each completed shard is atomic; on restart its clip IDs are reused without
decoding. Posterior and pooled WavLM tensors are stored in compressed shards.
"""
from __future__ import annotations
import json,hashlib,os,time,traceback,shutil,zipfile,tempfile
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
    """Standard .npz with fast lossless DEFLATE-1, same tensors as before."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+".partial")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1,allowZip64=True) as archive:
        for name,arr in arrays.items():
            with archive.open(name+".npy","w",force_zip64=True) as out:
                np.lib.format.write_array(out,np.asanyarray(arr),allow_pickle=False)
    os.replace(tmp,p)

def _copy_atomic(local,destination):
    """Upload completed local shard to Drive, then publish by atomic rename."""
    local=Path(local);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".partial")
    with local.open("rb") as source, temporary.open("wb") as sink:
        shutil.copyfileobj(source,sink,length=16*1024*1024)
    if temporary.stat().st_size!=local.stat().st_size:
        raise IOError("Truncated staging upload: "+str(destination))
    os.replace(temporary,destination)

def _is_cuda_oom(exc):
    msg=str(exc).lower()
    return ("out of memory" in msg and ("cuda" in msg or "gpu" in msg))

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

def _run_slice(rows,sources,phone,vad,registry,target_map,timings=None):
    work=[];out=[];tensors={}
    def add(key,seconds):
        if timings is not None:timings[key]=timings.get(key,0.)+seconds
    for row in rows:
        try:
            tick=time.perf_counter()
            wave,sr,audio_sha,original_sr=sources[row.split].decode(row.filename)
            add("decode",time.perf_counter()-tick)
            tick=time.perf_counter()
            spans=vad.segments(wave,sr)
            add("vad",time.perf_counter()-tick)
            tick=time.perf_counter()
            q=quality(wave,sr)
            add("quality",time.perf_counter()-tick)
            work.append((row,wave,spans,q,original_sr,audio_sha))
        except Exception as e:
            if _is_cuda_oom(e):raise
            out.append({"uid":row.uid,"split":row.split,"filename":row.filename,
                        "subject_id":row.subject_id,"task":row.task,
                        "item_score":getattr(row,"item_score",np.nan),
                        "status":"FAILED","failure_stage":"DECODE_VAD",
                        "failure_reason":str(e)[:350]})
    if not work:return out,tensors
    tick=time.perf_counter()
    try:preds=phone.infer_batch([r[1] for r in work])
    except Exception as exception:
        if _is_cuda_oom(exception):raise
        # A corrupt clip should not suppress unrelated valid entries.
        preds=[]
        for sample in work:
            try:preds.append(phone.infer_one(sample[1]))
            except Exception as e:
                if _is_cuda_oom(e):raise
                preds.append(e)
    add("inference",time.perf_counter()-tick)
    tick=time.perf_counter()
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
            if _is_cuda_oom(e):raise
            out.append({"uid":row.uid,"split":row.split,"filename":row.filename,
                        "subject_id":row.subject_id,"task":row.task,
                        "status":"FAILED","failure_stage":"FEATURES",
                        "failure_reason":str(e)[:350]})
    add("features",time.perf_counter()-tick)
    return out,tensors

def extract(config,inventory,phone=None,vad=None,shard_size=48,max_clips=0,
            vad_method="silero",staging_dir=None,profile_every=4):
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
    stage=Path(staging_dir).expanduser() if staging_dir else None
    if stage is not None:
        stage.mkdir(parents=True,exist_ok=True)
        if stage.resolve()==shards_dir.resolve():
            raise ValueError("Staging must be separate from final shard directory")
    timeline=[]
    total=len(todo)
    bar=tqdm(range(0,total,shard_size),desc="M87 private GPU shards")
    for begin in bar:
        subset=todo.iloc[begin:begin+shard_size]
        rows=[];frames={};timer={}
        shard_tick=time.perf_counter()
        for offset in range(0,len(subset),config.batch_size):
            batch=list(subset.iloc[offset:offset+config.batch_size].itertuples(index=False))
            result,tensors=_run_slice(batch,sources,phone,vad,registry,target_map,timer)
            rows.extend(result);frames.update(tensors)
        if len(rows)!=len(subset) or set(r["uid"] for r in rows)!=set(subset.uid):
            raise RuntimeError("Missing/duplicate clip in GPU shard")
        shard_num=start+counter
        base=shards_dir/f"{shard_num:05d}_phonetic"
        tick=time.perf_counter()
        if stage is None:
            _atomic_npz(base.with_suffix(".npz"),frames)
            _atomic_parquet(pd.DataFrame(rows),base.with_suffix(".parquet"))
            timer["serialization_and_drive"]=time.perf_counter()-tick
        else:
            # Compress on ephemeral local disk, but persist EVERY completed
            # shard to Drive before moving to the next one. On interruption,
            # Drive always contains complete (NPZ, Parquet) pairs for resume.
            local_base=stage/f"{shard_num:05d}_phonetic"
            _atomic_npz(local_base.with_suffix(".npz"),frames)
            _atomic_parquet(pd.DataFrame(rows),local_base.with_suffix(".parquet"))
            timer["serialization"]=time.perf_counter()-tick
            tick=time.perf_counter()
            _copy_atomic(local_base.with_suffix(".npz"),base.with_suffix(".npz"))
            _copy_atomic(local_base.with_suffix(".parquet"),base.with_suffix(".parquet"))
            timer["drive_upload"]=time.perf_counter()-tick
            local_base.with_suffix(".npz").unlink(missing_ok=True)
            local_base.with_suffix(".parquet").unlink(missing_ok=True)
        counter+=1
        timer.update({"shard":shard_num,"clips":len(subset),
                      "elapsed":time.perf_counter()-shard_tick,
                      "failed":sum(r["status"]=="FAILED" for r in rows)})
        timeline.append(timer)
        bar.set_postfix({"s/shard":round(timer["elapsed"],1),
                         "gpu":round(timer.get("inference",0.),1),
                         "features":round(timer.get("features",0.),1)})
        if profile_every and counter%int(profile_every)==0:
            recent=pd.DataFrame(timeline[-int(profile_every):])
            sums=recent.select_dtypes(include="number").sum()
            detail=" | ".join(f"{c}={sums[c]:.1f}s" for c in
                ("decode","vad","quality","inference","features",
                 "serialization","drive_upload","serialization_and_drive")
                if c in sums)
            tqdm.write(f"[M87] {counter} new shards | {detail}")
            # Small non-sensitive timing report, useful for cross-GPU reviews.
            _atomic_parquet(pd.DataFrame(timeline),
                root/"manifests"/"timing_profile_PRIVATE.parquet")
    if timeline:
        _atomic_parquet(pd.DataFrame(timeline),
            root/"manifests"/"timing_profile_PRIVATE.parquet")
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
    """Separate GPU process: unprompted word ASR on LOCALIZED answer only.

    Never decode the adult-containing whole recording for partial-credit
    features; doing so could achieve fake success by transcribing the prompt.
    Unknown/no-response localization is a FAILED clip, not a zero score.
    """
    config.validate()
    from .word_asr import WordASR
    root=config.output();out=root/"features"/"word_asr_PRIVATE.parquet"
    original=root/"features"/"phonetic_clip_features_PRIVATE.parquet"
    if not original.is_file():
        raise RuntimeError("Run Pasketti phonetic and child localizer BEFORE word ASR")
    loc=pd.read_parquet(original,columns=[
        "uid","status","response_start_s","response_end_s",
        "localization_confidence","no_child_response"])
    if loc.uid.duplicated().any():raise RuntimeError("Duplicate localization UID")
    lookup=loc.set_index("uid")
    old=pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["uid"])
    if old.uid.duplicated().any():raise RuntimeError("Duplicate word ASR cache UID")
    have=set(old.uid)
    todo=inventory[(inventory.task=="sentence_repetition") if sentence_only
                   else np.ones(len(inventory),dtype=bool)]
    todo=todo[~todo.uid.isin(have)].sort_values(["split","duration_sec"])
    if max_clips:todo=todo.head(int(max_clips))
    sources=create_sources(config,root)
    model=WordASR(revision=config.word_revision,batch_size=batch_size)
    records=[]
    for start in tqdm(range(0,len(todo),batch_size),desc="Pasketti Word localized"):
        subset=todo.iloc[start:start+batch_size]
        work=[]
        for row in subset.itertuples(index=False):
            try:
                if row.uid not in lookup.index:
                    raise ValueError("Missing phonetic/localization artifact")
                z=lookup.loc[row.uid]
                if str(z.status)!="VALID" or bool(z.no_child_response):
                    raise ValueError("No valid selected child-answer candidate")
                s=float(z.response_start_s);e=float(z.response_end_s)
                wave,sr,_,_=sources[row.split].decode(row.filename)
                if not np.isfinite([s,e]).all() or s<0 or e-s<.15:
                    raise ValueError("Invalid/too short response candidate")
                left=max(0,int((s-.05)*sr))
                right=min(len(wave),int((e+.05)*sr))
                if right-left<.15*sr:
                    raise ValueError("Insufficient selected response audio")
                work.append((row,wave[left:right],float(z.localization_confidence)))
            except Exception as exc:
                records.append({"uid":row.uid,"word_status":"FAILED",
                    "word_failure":str(exc)[:250],
                    "word_segment_source":"selected_response_v1"})
        if work:
            try:
                decoded=model.transcribe([x[1] for x in work])
            except Exception as exc:
                for row,_,confidence in work:
                    records.append({"uid":row.uid,"word_status":"FAILED",
                                    "word_failure":str(exc)[:250],
                                    "word_segment_source":"selected_response_v1",
                                    "word_localization_confidence":confidence})
            else:
                for (row,_,confidence),hyp in zip(work,decoded):
                    stats=word_partial_credit(row.expected_text,hyp["word_transcript"])
                    records.append({"uid":row.uid,"word_status":"VALID",
                        "word_segment_source":"selected_response_v1",
                        "word_localization_confidence":confidence,**hyp,**stats})
        # Save even when all clips fail; never silently discard failures.
        tmp=pd.concat([old,pd.DataFrame(records)],ignore_index=True)
        if tmp.uid.duplicated().any():raise RuntimeError("Repeated Word ASR UID")
        _atomic_parquet(tmp,out)
    if not out.exists():_atomic_parquet(old,out)
    final=pd.read_parquet(out)
    return {"n_completed":len(final),
            "n_valid":int((final.get("word_status",pd.Series(dtype=str))=="VALID").sum()),
            "path":str(out),"word_segment_source":"selected_response_v1"}

def load_features(root,word_optional=True):
    root=Path(root)
    f=pd.read_parquet(root/"features"/"phonetic_clip_features_PRIVATE.parquet")
    if word_optional and (root/"features"/"word_asr_PRIVATE.parquet").is_file():
        w=pd.read_parquet(root/"features"/"word_asr_PRIVATE.parquet")
        # The phonetic extraction marks word_partial_available=0 (unavailable)
        # until the optional word recognizer has run. Replace that PLACEHOLDER
        # with the actual word result; do not create misleading _x/_y columns.
        shared=(set(f.columns)&set(w.columns))-{"uid"}
        if shared-{"word_partial_available"}:
            raise RuntimeError(f"Unexpected word/phonetic feature name collision: {shared}")
        if "word_partial_available" in shared:
            f=f.drop(columns=["word_partial_available"])
        if w.uid.duplicated().any():
            raise RuntimeError("Duplicate word cache IDs")
        f=f.merge(w,on="uid",how="left",validate="one_to_one")
    return f
