"""Local speech activity detection and recording nuisance descriptors."""
from __future__ import annotations
import numpy as np

def quality(wav,sr=16000):
    x=np.asarray(wav,np.float32)
    if not len(x):raise ValueError("Empty wave")
    hop=max(1,sr//50)
    rms=np.asarray([np.sqrt(np.mean(np.square(x[i:i+hop])))
                    for i in range(0,len(x),hop)])
    return {"rms":float(np.sqrt(np.mean(np.square(x)))),
            "peak":float(np.max(np.abs(x))),
            "clipping_frac":float(np.mean(np.abs(x)>=.995)),
            "silence_frac":float(np.mean(rms<.008)),
            "noise_floor":float(np.percentile(rms,10)),
            "duration_s":float(len(x)/sr)}

def energy_vad(wav,sr=16000,frame_ms=20,min_speech_s=.1,gap_s=.16):
    x=np.asarray(wav,np.float32)
    if not len(x):return []
    hop=max(1,round(sr*frame_ms/1000))
    levels=np.asarray([np.sqrt(np.mean(np.square(x[i:i+hop])))
                       for i in range(0,len(x),hop)])
    cutoff=max(.014,2*float(np.percentile(levels,10)))
    mask=levels>cutoff
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
    spans=[]
    for a,b in edges:
        left=a*hop/sr;right=min(len(x)/sr,b*hop/sr)
        if spans and left-spans[-1][1]<=gap_s:spans[-1][1]=right
        else:spans.append([left,right])
    return [(float(a),float(b)) for a,b in spans if b-a>=min_speech_s]

class VAD:
    def __init__(self,method="silero"):
        self.method=method
        if method=="silero":
            try:
                from silero_vad import load_silero_vad,get_speech_timestamps
            except ImportError as e:
                raise ImportError("Install optional silero-vad for neural VAD") from e
            self.model=load_silero_vad()
            self.get=get_speech_timestamps
        elif method!="energy":raise ValueError("VAD method must be silero or energy")
    def segments(self,wav,sr=16000):
        if self.method=="energy":return energy_vad(wav,sr)
        import torch
        ts=self.get(torch.from_numpy(np.asarray(wav,np.float32)),self.model,
                    sampling_rate=sr,min_speech_duration_ms=100,
                    min_silence_duration_ms=100)
        return [(float(x["start"])/sr,float(x["end"])/sr) for x in ts]
