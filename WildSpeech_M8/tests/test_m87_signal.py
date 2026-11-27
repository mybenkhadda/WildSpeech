"""Fast CPU tests: TAR decoding, VAD, CTC paths and speech evidence."""
from __future__ import annotations
import io,tarfile
import numpy as np
import soundfile as sf
from wildspeech_m87.audio_io import AudioSource
from wildspeech_m87.vad_quality import energy_vad,quality
from wildspeech_m87.ctc_alignment import ctc_align,greedy_tokens,posterior_features,edit_counts
from wildspeech_m87.child_localizer import localize
from wildspeech_m87.task_scorers import word_partial_credit,phone_task_score

def test_archive_offsets_and_decode(tmp_path):
    wav=np.zeros(16000,np.float32);wav[3200:6400]=.2
    buf=io.BytesIO();sf.write(buf,wav,16000,format="WAV")
    tar=tmp_path/"train.tar"
    with tarfile.open(tar,"w") as a:
        data=buf.getvalue();m=tarfile.TarInfo("nested/hello.wav");m.size=len(data)
        a.addfile(m,io.BytesIO(data))
    source=AudioSource(tar_path=tar,index_path=tmp_path/"idx.parquet")
    x,sr,sha,original=source.decode("hello.wav")
    assert sr==original==16000 and len(x)==16000 and len(sha)==64
    again=AudioSource(tar_path=tar,index_path=tmp_path/"idx.parquet")
    assert again.get_bytes("nested/hello.wav")==buf.getvalue()
    assert quality(x)["rms"]>0 and energy_vad(x)

def test_ctc_forward_viterbi_and_blank_repetitions():
    # frames: blank, a, blank, b, blank
    probs=np.full((5,3),.025,float)
    probs[np.arange(5),[0,1,0,2,0]]=.95
    z=np.log(probs/probs.sum(axis=1,keepdims=True))
    aligned=ctc_align(z,[1,2],blank=0)
    assert aligned["valid"] and len(aligned["path"])==5
    assert len(aligned["phone_frames"])==2
    assert greedy_tokens(z,blank=0)==[1,2]
    evidence=posterior_features(z,[1,2],blank=0)
    assert evidence["ctc_valid"]==1 and evidence["aligned_phone_confidence"]>0.7
    repeat=ctc_align(z,[1,1,1,1],blank=0)
    assert not repeat["valid"] and repeat["reason"]=="INSUFFICIENT_FRAMES"
    counts=edit_counts([1,2,3],[1,3])
    assert counts["deletions"]==1 and counts["normalized_edits"]==1/3

def test_task_localization_and_random_control_is_deterministic():
    x=np.full((80,3),np.log(.025),float)
    x[:,0]=np.log(.95)
    x[52:58]=np.log([.3,.35,.35])
    x[53:56,1]=np.log(.95)
    x[56:59,2]=np.log(.95)
    first=localize("clip-aa","blending",[(.2,1.),(3.,4.8)],x,5.,[1,2],0)
    second=localize("clip-aa","blending",[(.2,1.),(3.,4.8)],x,5.,[1,2],0)
    assert first==second
    assert first["candidate_count"]==2
    assert first["response_end_s"]>=first["response_start_s"]
    assert "no_child_response" in first and "overlap_or_ambiguity" in first

def test_sentence_partial_credit_and_task_phone_errors():
    f=word_partial_credit("the red cat jumped","red cat jumps")
    assert f["word_partial_available"]==1
    assert 0<f["word_coverage"]<1 and f["word_lcs_ratio"]>0
    p=phone_task_score([1,2,3,4],[1,3,4],"blending",confidence=.9)
    assert p["task_phone_graded_accuracy"]<1
    assert p["task_phone_omission_rate"]>0
