"""Pasketti Word optional adapter uses only selected responses, not adult prompt.

No real ASR/model download: a fake local transcriber records input durations.
The word merge must preserve original phonetic localization confidence.
"""
import numpy as np
import pandas as pd
from wildspeech_m87.contracts import Config
from wildspeech_m87.gpu_cache import extract,extract_words,load_features
from test_m87_cache import inventory_and_files,FakePhonetic,FakeVAD

def test_word_asr_is_selected_answer_only_and_merge_does_not_rename_confidence(
        tmp_path,monkeypatch):
    inventory,train,test=inventory_and_files(tmp_path)
    c=Config(drive_root=str(tmp_path),output_root=str(tmp_path/"private"),
             train_audio_dir=str(train),test_audio_dir=str(test),
             acknowledge_license_review=True,mode="SCREEN")
    extract(c,inventory,phone=FakePhonetic(),vad=FakeVAD(),
            vad_method="energy",shard_size=4)
    durations=[]
    class FakeWord:
        def __init__(self,revision="",batch_size=8):self.revision="fake-word-sha"
        def transcribe(self,waves):
            durations.extend(len(wave)/16000 for wave in waves)
            return [{"word_transcript":"a","word_confidence_available":False,
                     "word_revision":"fake-word-sha"} for _ in waves]
    import wildspeech_m87.word_asr as word
    monkeypatch.setattr(word,"WordASR",FakeWord)
    result=extract_words(c,inventory,batch_size=2,sentence_only=True)
    assert result["n_completed"]==3 and result["n_valid"]==3
    assert all(.15<duration<.85 for duration in durations)
    f=load_features(c.output())
    assert "localization_confidence" in f.columns
    assert "word_localization_confidence" in f.columns
    assert not any(x.endswith(("_x","_y")) for x in f.columns)
    assert f.loc[f.task=="sentence_repetition","word_partial_available"].eq(1).all()
    before=len(durations)
    extract_words(c,inventory,batch_size=2,sentence_only=True)
    assert len(durations)==before
