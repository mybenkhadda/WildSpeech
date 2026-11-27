"""Optional Pasketti Word Qwen3-ASR local vLLM inference.

Install in a separate compatible Linux GPU environment if necessary:
qwen_asr[vllm]==0.0.6, vllm==0.14.0, transformers==4.57.6.
No prompt/expected text is passed, to prevent conditioned hallucination.
Token confidence is unavailable from the verified transcribe interface:
never fabricate log-probabilities.
"""
from __future__ import annotations
import numpy as np
from .phonetic_asr import pinned_revision
from .contracts import WORD_REPO

class WordASR:
    def __init__(self,repo=WORD_REPO,revision="",gpu_memory_utilization=.8,batch_size=8):
        from qwen_asr import Qwen3ASRModel
        self.revision=pinned_revision(repo,revision)
        # Use HF snapshot path to pin exact weights (downloads MODEL only).
        from huggingface_hub import snapshot_download
        local=snapshot_download(repo_id=repo,revision=self.revision)
        self.model=Qwen3ASRModel.LLM(local,
                         gpu_memory_utilization=gpu_memory_utilization,
                         max_inference_batch_size=int(batch_size),
                         max_new_tokens=256)
        self.batch_size=int(batch_size)

    def transcribe(self,waves,sr=16000):
        result=self.model.transcribe(
            audio=[(np.asarray(x,np.float32),int(sr)) for x in waves],
            language="English")
        if len(result)!=len(waves):raise RuntimeError("Word ASR lost input order")
        return [{"word_transcript":str(x.text),
                 "word_confidence_available":False,
                 "word_revision":self.revision} for x in result]
