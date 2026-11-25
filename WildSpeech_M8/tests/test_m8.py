import sys,io
from pathlib import Path
# pip install -e . installs src/wildspeech_m8
import numpy as np,pandas as pd,soundfile as sf
from wildspeech_m8.m8_pipeline import (Config,FoldEncoder,crossfit_baseline,ResidualDiscovery,fit_predict,transform_student_frame,assert_safe_features,score)
from wildspeech_m8.m8_audio import audio_features
from wildspeech_m8.m8_speech import pair_features

def test_leakage_rejection():
    try:assert_safe_features(['profile__score_mean','school_id'])
    except ValueError:pass
    else:raise AssertionError('school_id should be rejected')

def test_pair_features():
    a=pair_features('cat','cat');assert a['asr__exact_match']==1
    b=pair_features('cat','bat');assert b['asr__exact_match']==0 and b['asr__char_sim']<1

def test_audio():
    t=np.arange(16000)/16000
    wav=np.sin(2*np.pi*320*t).astype('float32')*.10
    file=io.BytesIO();sf.write(file,wav,16000,format='WAV')
    z=audio_features(file.getvalue());assert z['audio__valid']==1 and np.isfinite(z['audio__spectral_centroid_mean'])

def test_oof_and_transform():
    rng=np.random.default_rng(5);n=55
    frame=pd.DataFrame({'blending_theta':rng.normal(size=n),'deletion_theta':rng.normal(size=n),
                        'profile__score_mean':rng.normal(size=n),'duration_sec':rng.uniform(1,10,n),
                        'instrument':rng.choice(['a','b'],n)})
    d=transform_student_frame(frame);assert 'm8__contrast__blending_theta__deletion_theta' in d
    enc=FoldEncoder().fit(d.iloc[:45]);z=enc.transform(d.iloc[:45]); zv=enc.transform(d.iloc[45:]);assert z.shape[1]==zv.shape[1]
    y=110+4*frame.blending_theta.to_numpy()+rng.normal(size=n)
    cfg=Config(quick=True,inner_folds=3)
    p=crossfit_baseline(z,y[:45],cfg,5);assert np.isfinite(p).all()
    m=ResidualDiscovery(k=2,seed=5).fit(z,y[:45],p)
    pred=m.predict_correction(zv);assert len(pred)==10 and np.isfinite(pred).all()
    pv=fit_predict(d.iloc[:45],d.iloc[45:],y[:45],cfg,5,('metric',2,.5))
    assert len(pv)==10 and np.isfinite(pv).all()