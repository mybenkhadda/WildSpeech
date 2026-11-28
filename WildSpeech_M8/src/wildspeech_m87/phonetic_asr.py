"""Pasketti Phonetic: fp32 WavLMForCTC, true padded lengths and layer vectors.

No expected text is supplied to ASR. Full frame log probabilities can be
stored as float16 for offline feature extraction, after FP32 model inference.
WavLM relative position bias can yield NaNs in half precision: NEVER convert
model weights or inference to half precision.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .contracts import PHONE_REPO

@dataclass
class PhoneticResult:
    logprobs:np.ndarray
    transcript:str
    hidden:dict
    n_frames:int
    duration_s:float
    revision:str

def pinned_revision(repo,revision=""):
    from huggingface_hub import HfApi
    if revision and len(revision)>=40 and all(c in "0123456789abcdef" for c in revision.lower()):
        return revision
    return HfApi().model_info(repo,revision=revision or "main").sha

class PhoneticASR:
    def __init__(self,repo=PHONE_REPO,revision="",device=None,layers=(6,7,8,9,10),
                 max_seconds=30.):
        import torch
        from transformers import Wav2Vec2Processor,WavLMForCTC
        self.torch=torch
        self.repo=repo;self.revision=pinned_revision(repo,revision)
        self.processor=Wav2Vec2Processor.from_pretrained(repo,revision=self.revision,
                                                        trust_remote_code=False)
        self.model=WavLMForCTC.from_pretrained(repo,revision=self.revision,
                                               trust_remote_code=False)
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model=self.model.to(self.device,dtype=torch.float32).eval()
        self.layers=tuple(int(k) for k in layers)
        self.max_seconds=float(max_seconds)
        self.sr=16000
        self.blank=int(self.processor.tokenizer.pad_token_id)
        self.vocab=self.processor.tokenizer.get_vocab()
        if self.blank is None:raise RuntimeError("Pasketti tokenizer CTC blank unspecified")
        if self.device=="cuda":torch.cuda.empty_cache()

    def _infer_short(self,waves):
        torch=self.torch
        for x in waves:
            if len(x)>self.max_seconds*self.sr:
                raise ValueError("Long wave must be passed through infer_one chunker")
        inputs=self.processor([np.asarray(x,np.float32) for x in waves],
                              sampling_rate=self.sr,padding=True,
                              return_attention_mask=True,return_tensors="pt")
        lens=inputs["attention_mask"].sum(-1)
        tensor={k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode():
            out=self.model(**tensor,output_hidden_states=True,return_dict=True)
            nframes=self.model._get_feat_extract_output_lengths(lens).cpu().numpy()
            logits=out.logits.float()
            logp=torch.log_softmax(logits,dim=-1)
            hidden=out.hidden_states
            data=[]
            for i,count in enumerate(nframes):
                n=int(count)
                if n<1:raise RuntimeError("No CTC frames")
                z=logp[i,:n].cpu().numpy().astype(np.float32)
                if not np.isfinite(z).all():raise RuntimeError("Nonfinite FP32 phonetic logits")
                pred=np.argmax(z,axis=-1)
                text=self.processor.decode(pred)
                pooled={}
                for layer in self.layers:
                    if layer<0 or layer>=len(hidden):
                        raise ValueError(f"Requested WavLM hidden layer {layer} out of range")
                    pooled[str(layer)]=hidden[layer][i,:n].float().mean(dim=0).cpu().numpy().astype(np.float32)
                data.append(PhoneticResult(z,str(text),pooled,n,len(waves[i])/self.sr,self.revision))
        del inputs,tensor,logp,out,hidden
        return data

    def infer_one(self,wave):
        x=np.asarray(wave,np.float32)
        if not len(x) or not np.isfinite(x).all():raise ValueError("Invalid wave")
        max_samples=int(self.max_seconds*self.sr)
        if len(x)<=max_samples:return self._infer_short([x])[0]
        # Model is trained on short utterances; long clips are segmented with
        # deterministic 0.5s overlaps. We trim overlaps before concatenating
        # posterior frames. This is explicitly an approximation.
        overlap=int(.5*self.sr)
        step=max_samples-overlap
        chunks=[];starts=[]
        for start in range(0,len(x),step):
            piece=x[start:start+max_samples]
            if len(piece)<self.sr//5 and chunks:break
            starts.append(start);chunks.append(piece)
        parts=self._infer_short(chunks)
        pieces=[];pooled={str(i):[] for i in self.layers};weights=[]
        for j,p in enumerate(parts):
            left=int(round(.25*p.n_frames/max(p.duration_s,.001))) if j else 0
            right=int(round(.25*p.n_frames/max(p.duration_s,.001))) if j<len(parts)-1 else 0
            arr=p.logprobs[left:len(p.logprobs)-right if right else None]
            if len(arr)==0:continue
            pieces.append(arr)
            for layer in self.layers:pooled[str(layer)].append(p.hidden[str(layer)]*len(arr))
            weights.append(len(arr))
        merged=np.concatenate(pieces)
        states={str(layer):sum(pooled[str(layer)])/sum(weights) for layer in self.layers}
        transcript=self.processor.decode(np.argmax(merged,axis=-1))
        return PhoneticResult(merged,str(transcript),states,len(merged),
                              len(x)/self.sr,self.revision)

    def infer_batch(self,waves):
        """Adaptive FP32 batches: bisect CUDA OOM, never silently fail a clip."""
        if not waves:return []
        try:
            if all(len(x)<=self.max_seconds*self.sr for x in waves):
                return self._infer_short(waves)
            return [self.infer_one(x) for x in waves]
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(waves)==1:
                raise RuntimeError(
                    "Single-clip CUDA OOM: stop without caching FAILED; "
                    "reduce max input duration only in a fresh cache"
                )
            middle=len(waves)//2
            return self.infer_batch(waves[:middle])+self.infer_batch(waves[middle:])
