"""Task-aware, uncertainty-preserving child response candidate selection.

This is NOT speaker identification. Selection may pick an adult prompt; QA
must measure that failure. No target words are fed into the free ASR decoder.
Uncertain cases are flagged; no-response is not imputed as wrong.
"""
from __future__ import annotations
import hashlib
import numpy as np
from .ctc_alignment import ctc_align,greedy_tokens

def candidates_from_vad(spans,duration,task,min_s=.12,max_s=12.):
    candidates=[]
    for start,end in spans:
        start=max(0.,float(start));end=min(float(duration),float(end))
        if end-start<min_s:continue
        if end-start<=max_s:
            candidates.append((start,end))
        else:
            # A merged prompt+answer VAD span can contain multiple speakers.
            # Search overlapping portions instead of assuming one speaker.
            length=max_s
            stride=max(1.,length*.55)
            steps=int(np.ceil((end-start-length)/stride))+1
            for j in range(max(1,steps)):
                left=min(start+j*stride,end-length)
                candidates.append((left,min(end,left+length)))
    if not candidates and duration>=min_s:
        # Full clip fallback flagged as low-confidence, never auto-claimed child.
        candidates=[(0.,float(duration))]
    return sorted(set((round(a,3),round(b,3)) for a,b in candidates))

def segment_frame_bounds(start,end,duration,nframes):
    a=max(0,min(nframes-1,int(np.floor(start/duration*nframes))))
    b=max(a+1,min(nframes,int(np.ceil(end/duration*nframes))))
    return a,b

def _score(logprobs,target,blank,start,end,duration,task):
    a,b=segment_frame_bounds(start,end,duration,len(logprobs))
    sub=logprobs[a:b]
    if len(target):
        forced=ctc_align(sub,target,blank,with_path=False)
        free=ctc_align(sub,greedy_tokens(sub,blank),blank,with_path=False)
        if forced.get("valid") and free.get("valid"):
            # Compare two CTC sequence likelihoods normalized by frame
            # count; higher (less negative) suggests target support.
            acoustic=(forced["log_likelihood"]-free["log_likelihood"])/len(sub)
            matched=True
        else:acoustic=-2.0;matched=False
    else:
        acoustic=0.;matched=False
    center=.5*(start+end)/max(duration,1e-6)
    # Weak late-turn preference; not an assertion about speaker age.
    timing=.15*center
    return float(np.clip(acoustic,-10,0)+timing),matched

def localize(uid,task,spans,logprobs,duration,target_ids,blank,
             min_confidence=.53,margin=.10):
    if task not in ("blending","deletion","sentence_repetition"):
        raise ValueError("Unknown task")
    if duration<=0 or not len(logprobs):
        return {"response_start_s":0.,"response_end_s":0.,
                "localization_confidence":0.,"candidate_count":0,
                "no_child_response":True,"multiple_attempts":False,
                "overlap_or_ambiguity":True,"prompt_detected":False,
                "localization_method":"invalid","random_start_s":0.,
                "random_end_s":0.}
    candidates=candidates_from_vad(spans,duration,task)
    if not candidates:raise RuntimeError("Candidate generation failed")
    stats=[]
    for a,b in candidates:
        val,valid=_score(logprobs,target_ids,blank,a,b,duration,task)
        stats.append((val,a,b,valid))
    stats.sort(key=lambda x:(-x[0],-x[1]))
    best=stats[0]
    second=stats[1][0] if len(stats)>1 else best[0]-.15
    diff=max(0.,best[0]-second)
    # Confidence is provisional and must be CALIBRATED on training-only QA.
    confidence=float((1./(1.+np.exp(-8*(diff-.08))))*
                     (0.75 if best[3] else .30))
    close=[v for v in stats if best[0]-v[0]<margin]
    ambiguous=len(close)>1 or not best[3] or len(spans)<2
    # A no-response label from VAD alone is not definitive; keep a flag.
    no_response=not spans
    random_i=int(hashlib.sha256(str(uid).encode()).hexdigest()[:8],16)%len(candidates)
    random_start,random_end=candidates[random_i]
    return {"response_start_s":float(best[1]),
            "response_end_s":float(best[2]),
            "localization_confidence":confidence,
            "candidate_count":len(candidates),
            "no_child_response":bool(no_response),
            "multiple_attempts":bool(len(spans)>=3),
            "overlap_or_ambiguity":bool(ambiguous),
            "prompt_detected":bool(len(spans)>=2),
            "localization_method":"task_ctc_late_prior_v1",
            "random_start_s":float(random_start),
            "random_end_s":float(random_end),
            "ctc_candidate_score":float(best[0]),
            "ctc_candidate_margin":float(diff),
            "vad_segments":len(spans),
            "vad_speech_seconds":float(sum(max(0,e-s) for s,e in spans))}
