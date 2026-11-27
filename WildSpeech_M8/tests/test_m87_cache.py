"""End-to-end resumable extraction using REAL WAV bytes and a synthetic CTC model.

No model downloads, no private audio. Tests image/manifest/coverage creation,
TAR-or-directory decoding, voice activity, shard tensors and restart behavior.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from wildspeech_m87.contracts import Config,stable_uid
from wildspeech_m87.phonetic_asr import PhoneticResult
from wildspeech_m87.gpu_cache import extract,load_features
from wildspeech_m87.diagnostics import extraction_atlas

class Tokenizer:
    pad_token_id=0
    unk_token_id=4
    def get_vocab(self):return {"<pad>":0,"a":1,"b":2,"|":3,"<unk>":4}
    def tokenize(self,s):return list(s.replace(" ","|"))
    def convert_tokens_to_ids(self,tokens):return [self.get_vocab().get(t,4) for t in tokens]

class Processor:
    tokenizer=Tokenizer()

class FakePhonetic:
    revision="synthetic-2026-audio-ctc"
    blank=0
    processor=Processor()
    def __init__(self):self.called=0
    def infer_batch(self,waves):
        self.called+=1
        output=[]
        for wave in waves:
            frames=max(12,int(len(wave)/320))
            p=np.full((frames,5),.01,dtype=np.float32)
            p[:,0]=.96
            p[frames//3:frames//3+3,0]=.02
            p[frames//3:frames//3+3,1]=.95
            p/=p.sum(axis=1,keepdims=True)
            states={str(k):np.ones(32,np.float32)*(k+len(wave)/16000)
                    for k in (6,7,8,9,10)}
            output.append(PhoneticResult(np.log(p),"a",states,frames,
                                         len(wave)/16000,self.revision))
        return output

class FakeVAD:
    def segments(self,wav,sr):
        return [(0.1,min(len(wav)/sr,0.7))]

def inventory_and_files(tmp_path):
    train=tmp_path/"train";test=tmp_path/"test"
    train.mkdir();test.mkdir()
    rows=[]
    for split,folder,n in (("train",train,4),("test",test,3)):
        for i in range(n):
            name=f"clip_{i:02d}.wav"
            wave=(.2*np.sin(2*np.pi*440*np.arange(16000)/16000)).astype(np.float32)
            sf.write(folder/name,wave,16000)
            rows.append({"uid":stable_uid(split,name),"split":split,
                         "filename":name,"subject_id":f"{split}_person{i}",
                         "task":("blending" if i%2==0 else "sentence_repetition"),
                         "expected_text":"a","duration_sec":1.,
                         "item_score":float(i%2) if split=="train" else np.nan})
    return pd.DataFrame(rows),train,test

def test_extract_all_audio_once_resume_and_figures(tmp_path):
    inventory,train,test=inventory_and_files(tmp_path)
    config=Config(drive_root=str(tmp_path),output_root=str(tmp_path/"private"),
                  train_audio_dir=str(train),test_audio_dir=str(test),
                  mode="SCREEN",acknowledge_license_review=True,
                  batch_size=2)
    backend=FakePhonetic()
    counts=extract(config,inventory,phone=backend,vad=FakeVAD(),
                   shard_size=3,vad_method="energy")
    assert counts=={"expected":7,"accounted":7,"valid":7,"failed":0,"missing":0}
    assert backend.called>=4
    rows=load_features(config.output())
    assert rows.uid.is_unique and rows.status.eq("VALID").all()
    assert rows.audio_sha256.str.len().eq(64).all()
    assert rows.localization_confidence.notna().all()
    shard=list((config.output()/"shards").glob("*.npz"))
    assert len(shard)==3
    with np.load(shard[0]) as z:
        assert any(k.startswith("posterior_") for k in z.files)
        assert any(k.startswith("hidden_6_") for k in z.files)
    before=backend.called
    repeated=extract(config,inventory,phone=backend,vad=FakeVAD(),
                     shard_size=3,vad_method="energy")
    assert repeated==counts and backend.called==before
    result=extraction_atlas(config.output(),inventory,rows)
    assert result["clips_valid"]==7
    assert (config.output()/"figures"/"01_cache_coverage.png").is_file()
    assert (config.output()/"figures"/"06_grading_within_correctness.png").is_file()
