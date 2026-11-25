"""WildSpeech M8 audio feature bank (CPU-first, cacheable, tar-streaming).

Audio features are signal measurements, not verified speaker identity, phoneme
accuracy, or clinical measures. No external speech API and no adult/child claims.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, tarfile, time, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import signal, stats
import soundfile as sf

FEATURE_VERSION='m8_audio_1.0'

def audio_features(blob,max_sec=75,target_sr=16000):
    with sf.SoundFile(io.BytesIO(blob)) as f:
        sr=f.samplerate;n=min(f.frames,int(max_sec*sr));y=f.read(n,dtype='float32',always_2d=True)
    x=y.mean(axis=1);length=len(x)
    if length<max(320,int(.15*sr)):
        return {'audio__valid':0.,'audio__duration_s':length/sr,'audio__quality_usable':0.}
    if sr!=target_sr:
        import math
        g=math.gcd(sr,target_sr)
        x=signal.resample_poly(x,target_sr//g,sr//g).astype('float32');sr=target_sr
    x=np.nan_to_num(x,nan=0,posinf=0,neginf=0)
    dur=len(x)/sr
    rms=np.sqrt(np.mean(x*x)+1e-12);clip=np.mean(np.abs(x)>=.98)
    dc=float(np.abs(x.mean()))
    # 25 ms frames at 10 ms hops. Frame energies are proxies, not speaker diarization.
    frame=max(16,int(sr*.025));hop=max(8,int(sr*.010))
    if len(x)<frame:return {'audio__valid':0.,'audio__duration_s':dur,'audio__quality_usable':0.}
    frames=np.lib.stride_tricks.sliding_window_view(x,frame)[::hop]
    if len(frames)>20000:frames=frames[:20000]
    # O(n) short-time RMS using cumulative energy, avoiding the enormous
    # temporary array produced by multiplying all overlapping sample windows.
    power=np.cumsum(np.r_[0.0,x.astype(np.float64)**2])
    starts=np.arange(0,len(x)-frame+1,hop,dtype=int)
    e=np.sqrt((power[starts+frame]-power[starts])/frame+1e-12)
    loud=np.quantile(e,.80);noise=np.quantile(e,.15)
    # Noise-floor-relative detector; do not claim isolated child speech.
    threshold=max(float(noise*2.8),float(loud*.12),1e-5)
    voice=e>threshold
    # Remove isolated one-frame bursts; this is *not* a speaker VAD.
    if len(voice)>=5:voice=signal.medfilt(voice.astype('uint8'),kernel_size=5).astype(bool)
    speech_fraction=float(voice.mean())
    on=np.flatnonzero(voice)
    onset=float(on[0]*hop/sr) if len(on) else np.nan
    offset=float(on[-1]*hop/sr) if len(on) else np.nan
    boundaries=np.diff(np.r_[False,voice,False].astype(int))
    onsets=np.where(boundaries==1)[0];ends=np.where(boundaries==-1)[0]
    pauses=[]
    if len(onsets)>1:
        pauses=np.maximum((onsets[1:]-ends[:-1])*hop/sr,0)
        pauses=np.asarray(pauses)[np.asarray(pauses)>.15]
    # STFT features, downsample to bound memory.
    f,t,z=signal.stft(x,fs=sr,nperseg=512,noverlap=256,boundary=None,padded=False)
    mag=np.abs(z).astype('float64')+1e-12
    norm=mag/(mag.sum(axis=0,keepdims=True)+1e-12)
    centroid=(norm*f[:,None]).sum(axis=0)
    flatness=np.exp(np.log(mag).mean(axis=0))/(mag.mean(axis=0)+1e-12)
    entropy=-(norm*np.log(norm+1e-12)).sum(axis=0)/np.log(len(f))
    spectral_flux=np.maximum(np.diff(norm,axis=1),0).sum(axis=0) if norm.shape[1]>1 else np.array([0.])
    snr_proxy=20*np.log10((loud+1e-8)/(noise+1e-8))
    quality=float((rms>5e-5) and (clip<.05) and (.015<speech_fraction<.98) and (snr_proxy>3))
    feats={
      'audio__valid':1.,'audio__duration_s':dur,'audio__quality_usable':quality,
      'audio__rms':float(rms),'audio__rms_log':float(np.log(rms+1e-10)),
      'audio__peak':float(np.max(np.abs(x))),'audio__clip_fraction':float(clip),
      'audio__dc_abs':dc,'audio__silence_fraction':1-speech_fraction,
      'audio__speech_fraction_proxy':speech_fraction,'audio__onset_s_proxy':onset,
      'audio__trailing_s_proxy':float(max(0,dur-offset)) if np.isfinite(offset) else np.nan,
      'audio__speech_bursts_proxy':len(onsets),'audio__pause_count_proxy':len(pauses),
      'audio__active_speech_s_proxy':float(speech_fraction*dur),
      'audio__pause_rate_per_s_proxy':float(len(pauses)/max(speech_fraction*dur,0.05)),
      'audio__pause_sum_s_proxy':float(np.sum(pauses)),
      'audio__pause_mean_s_proxy':float(np.mean(pauses)) if len(pauses) else 0.,
      'audio__pause_p90_s_proxy':float(np.quantile(pauses,.9)) if len(pauses) else 0.,
      'audio__snr_proxy_db':float(snr_proxy),'audio__frame_rms_cv':float(e.std()/(e.mean()+1e-8)),
      'audio__spectral_centroid_mean':float(np.mean(centroid)),
      'audio__spectral_centroid_std':float(np.std(centroid)),
      'audio__spectral_flatness_mean':float(np.mean(flatness)),
      'audio__spectral_centroid_cv':float(np.std(centroid)/(np.mean(centroid)+1e-5)),
      'audio__spectral_entropy_mean':float(np.mean(entropy)),
      'audio__spectral_flux_mean':float(np.mean(spectral_flux)),
      'audio__zero_crossing_rate':float(np.mean(np.diff(np.signbit(x))!=0)),
    }
    return feats

def extract_split(root,split,max_sec=75,limit=0):
    root=Path(root)
    meta=pd.read_csv(root/f'{split}_clip_metadata.csv',dtype={'subject_id':str,'filename':str})
    if not {'filename','subject_id'}.issubset(meta):raise ValueError('Invalid clip metadata')
    # basename mapping is safe only with unique filenames; ambiguous recordings fail loudly.
    if meta.filename.duplicated().any():raise ValueError('Duplicate audio filenames in metadata')
    if limit>0:meta=meta.head(limit).copy()
    lookup={Path(k).name:k for k in meta.filename}
    if len(lookup)!=len(meta):raise ValueError('Ambiguous metadata audio basenames')
    tarpath=root/f'{split}_audio.tar'
    if not tarpath.is_file():raise FileNotFoundError(tarpath)
    rows=[];found=set();bad=[];start=time.time()
    from tqdm.auto import tqdm
    with tarfile.open(tarpath,'r|*') as archive:
        with tqdm(total=len(meta),desc=f'M8 audio {split}',unit='clip') as bar:
            for member in archive:
                if not member.isfile():continue
                name=Path(member.name).name
                if name not in lookup:continue
                original=lookup[name]
                if original in found:raise ValueError('Duplicate audio file in tar: '+original)
                found.add(original)
                try:
                    file=archive.extractfile(member)
                    if file is None:raise IOError('empty archive member')
                    feat=audio_features(file.read(),max_sec)
                    feat.update({'filename':original,'audio__status':'ok','audio__error':''})
                except Exception as ex:
                    feat={'filename':original,'audio__status':'error','audio__error':type(ex).__name__+': '+str(ex)[:120],'audio__valid':0.,'audio__quality_usable':0.}
                    bad.append(feat['audio__error'])
                rows.append(feat);bar.update(1)
                if len(found)>=len(lookup):break
    missing=set(meta.filename)-found
    if missing:raise RuntimeError(f'{len(missing)} metadata clips not found in {tarpath}. Example: {list(missing)[:5]}')
    out=meta.merge(pd.DataFrame(rows),on='filename',how='left',validate='one_to_one')
    print(f'[audio] {split}: {len(out)} clips, errors={len(bad)}, runtime={round((time.time()-start)/60,2)}min')
    return out

def aggregate_audio(df):
    num=[c for c in df if c.startswith('audio__') and pd.api.types.is_numeric_dtype(df[c])]
    group=df.groupby('subject_id')
    result=group.size().rename('m8__audio__n_recordings').to_frame()
    for c in num:
        g=group[c]
        for stat,suffix in [('mean','mean'),('std','std')]:
            result[f'm8__audio__{c[7:]}__{suffix}']=getattr(g,stat)()
    if 'audio__quality_usable' in df:
        result['m8__audio__quality_coverage']=group['audio__quality_usable'].mean()
        usable=df[df['audio__quality_usable']==1].copy()
        if not usable.empty:
            gu=usable.groupby('subject_id')
            for c in [z for z in num if any(k in z for k in
                      ('onset','pause','speech_fraction','spectral_entropy','snr_proxy','flux'))]:
                result[f'm8__audio__reliable_{c[7:]}__mean']=gu[c].mean().reindex(result.index)
            result['m8__audio__reliable_n']=gu.size().reindex(result.index).fillna(0)
    if 'expected_text' in df:
        # Expected-text length is from item metadata; NOT an observed transcript.
        d=df.copy()
        d['m8__expected_words']=d['expected_text'].fillna('').astype(str).str.split().str.len().clip(lower=1)
        if 'audio__active_speech_s_proxy' in d:
            d['m8__speech_time_per_expected_word_proxy']=d['audio__active_speech_s_proxy']/d['m8__expected_words']
            result['m8__audio__speech_time_per_expected_word_mean']=d.groupby('subject_id')['m8__speech_time_per_expected_word_proxy'].mean().reindex(result.index)
    if 'item_score' in df:
        # Contrast speech signals in scored-correct and scored-incorrect items.
        d=df.copy();d['m8__item_correct']=pd.to_numeric(d['item_score'],errors='coerce')
        for score_value,suffix in [(1,'correct'),(0,'incorrect')]:
            p=d[d['m8__item_correct']==score_value]
            gg=p.groupby('subject_id')
            result[f'm8__audio__n_{suffix}_items']=gg.size().reindex(result.index).fillna(0)
            for c in [z for z in num if any(k in z for k in ('onset','speech_fraction','pause_count','spectral_entropy'))]:
                result[f'm8__audio__{c[7:]}__{suffix}_mean']=gg[c].mean().reindex(result.index)
        for c in [z for z in num if any(k in z for k in ('onset','pause_count','speech_fraction'))]:
            a=result.get(f'm8__audio__{c[7:]}__correct_mean')
            b=result.get(f'm8__audio__{c[7:]}__incorrect_mean')
            if a is not None and b is not None:
                result[f'm8__audio__{c[7:]}__correct_minus_incorrect']=a-b
    taskcol=next((c for c in ['task','task_type','task_name','assessment_task'] if c in df.columns),None)
    if taskcol:
        for task,p in df.groupby(taskcol):
            tag=re.sub('[^a-z0-9]+','_',str(task).lower())[:35]
            gg=p.groupby('subject_id')
            result[f'm8__audio__task_{tag}__n']=gg.size().reindex(result.index).fillna(0)
            for c in [c for c in num if any(x in c for x in ('onset','pause','speech_fraction','spectral_entropy','snr_proxy'))]:
                result[f'm8__audio__task_{tag}__{c[7:]}']=gg[c].mean().reindex(result.index)
    return result.reset_index()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--limit',type=int,default=0,help='0 full; nonzero is smoke mode only')
    p.add_argument('--max-sec',type=float,default=75)
    args=p.parse_args();out=Path(args.output_root)/'tables';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for split in ('train','test'):
        cache=out/f'audio_clip_features_{split}.csv'
        if cache.exists():
            previous=pd.read_csv(cache,dtype={'subject_id':str,'filename':str})
            meta=pd.read_csv(Path(args.data_root)/f'{split}_clip_metadata.csv',dtype={'subject_id':str,'filename':str})
            if args.limit==0 and set(previous.filename)==set(meta.filename):
                print('[cache hit]',cache);records.append(previous);continue
        data=extract_split(args.data_root,split,args.max_sec,args.limit)
        data.to_csv(cache,index=False);records.append(data)
    allclips=pd.concat(records,ignore_index=True)
    ag=aggregate_audio(allclips)
    ag.to_csv(out/'audio_student_features.csv',index=False)
    summary={'version':FEATURE_VERSION,'n_clips':len(allclips),'n_students':len(ag),
             'is_complete':args.limit==0,'invalid':int((allclips.audio__valid.fillna(0)==0).sum()),
             'limits': 'No validated diarization/ASR/phoneme correctness; proxy features only'}
    (out/'audio_extraction_summary.json').write_text(json.dumps(summary,indent=2))
    print('[DONE]',summary)
if __name__=='__main__':main()