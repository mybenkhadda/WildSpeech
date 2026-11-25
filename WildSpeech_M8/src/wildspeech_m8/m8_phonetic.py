"""Optional fully local GPU-capable Pasketti phonetic inference for WildSpeech M8.1.

Downloads pretrained weights from Hugging Face only when invoked; never uploads
competition audio. Output is a noisy model hypothesis, *not* gold child phonemes.
The official public reference uses drivendata/pasketti-phonetic with WavLMForCTC.

Run after installing optional-requirements.txt.
"""
from __future__ import annotations
import argparse, csv, io, json, math, re, tarfile, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from difflib import SequenceMatcher
from tqdm.auto import tqdm

MODEL='drivendata/pasketti-phonetic'


def clean_ipa(s):
    if pd.isna(s):return ''
    # Preserve IPA symbols (unlike ASCII-only ASR normalization).
    s=unicodedata.normalize('NFKC',str(s)).lower()
    return ''.join(ch for ch in s if (unicodedata.category(ch).startswith('L') or
                  unicodedata.category(ch).startswith('M') or ch in ('ə','ɚ','ɹ','ɑ','ʊ','ʌ','ɪ','æ','ɔ','ɛ','θ','ð','ŋ','ʃ','ʒ','ʧ','ʤ')) and ch not in 'ˌˈ').replace(' ', '')


def phone_distance(ref,hyp):
    a=clean_ipa(ref);b=clean_ipa(hyp)
    if not a or not b:return np.nan
    return float(1-SequenceMatcher(None,a,b,autojunk=False).ratio())


def expected_to_ipa(text):
    import eng_to_ipa as epi
    if pd.isna(text) or not str(text).strip():return ('',False)
    ipa=epi.convert(str(text),stress_marks=False,keep_punct=False)
    # eng_to_ipa marks OOV words with '*'; do not silently score orthographic fallback.
    return ipa,('*' not in ipa and bool(clean_ipa(ipa)))


def decode_audio(blob,sr=16000,max_sec=75):
    with sf.SoundFile(io.BytesIO(blob)) as snd:
        input_sr=snd.samplerate;frames=min(snd.frames,round(max_sec*input_sr))
        data=snd.read(frames,dtype='float32',always_2d=True).mean(axis=1)
    if input_sr!=sr:
        g=math.gcd(input_sr,sr);data=signal.resample_poly(data,sr//g,input_sr//g).astype('float32')
    data=np.nan_to_num(data,nan=0,posinf=0,neginf=0)
    return data


def inference_batch(model,processor,waves,device):
    import torch
    if not waves:return []
    ordering=sorted(range(len(waves)),key=lambda i:len(waves[i]))
    # One batch (small by default). Sorting lets the smallest clip be decoded first;
    # all predictions are restored in the original order.
    sorted_waves=[waves[i] for i in ordering]
    inputs=processor(sorted_waves,sampling_rate=16000,padding=True,return_tensors='pt')
    lens=np.asarray([len(x) for x in sorted_waves],dtype=int)
    model_args={'input_values':inputs.input_values.to(device)}
    if 'attention_mask' in inputs:
        model_args['attention_mask']=inputs.attention_mask.to(device)
    with torch.inference_mode():
        logits=model(**model_args).logits.float()
    # WavLMForCTC downsamples samples into CTC frame sequences; cropping avoids
    # decoding padding as words/phones for short clips.
    if hasattr(model,'_get_feat_extract_output_lengths'):
        outlens=model._get_feat_extract_output_lengths(torch.tensor(lens,device=logits.device)).cpu().numpy()
    else:
        outlens=np.ceil(lens/max(lens)*logits.shape[1]).astype(int)
    blank=getattr(model.config,'pad_token_id',0)
    result=[]
    for i,limit in enumerate(outlens):
        l=logits[i,:int(limit)].detach()
        top=l.argmax(dim=-1)
        hypo=processor.decode(top.cpu().numpy())
        probability=l.softmax(dim=-1).max(dim=-1).values
        active=(top!=blank)
        conf=float(probability[active].mean().cpu()) if active.any() else np.nan
        result.append({'phonetic_hypothesis':str(hypo),
                       'phonetic_confidence_proxy':conf,
                       'phonetic_ctc_blank_fraction':float((~active).float().mean().cpu()),
                       'phonetic_frames':int(limit)})
    restored=[None]*len(ordering)
    for idx,item in zip(ordering,result):restored[idx]=item
    return restored


def aggregate_phonetics(table,outdir):
    # Accuracy is only a predictor when the *observed item score* is available.
    # This is not a verified phoneme error rate.
    tab=table.copy()
    tab['phonetic_ref_distance']= [phone_distance(r,h) if good else np.nan
             for r,h,good in zip(tab.phonetic_reference,tab.phonetic_hypothesis,tab.phonetic_reference_valid)]
    tab['phonetic_valid']=tab.phonetic_ref_distance.notna().astype(float)
    tab['phonetic_hyp_len']=tab.phonetic_hypothesis.fillna('').astype(str).str.len()
    tab['phonetic_ref_len']=tab.phonetic_reference.fillna('').astype(str).str.len()
    tab.to_csv(outdir/'phonetic_clip_features.csv',index=False)
    gg=tab.groupby('subject_id',sort=False)
    result=gg.size().rename('m8__phonetic__total_clips').to_frame()
    for col in ['phonetic_ref_distance','phonetic_valid','phonetic_confidence_proxy','phonetic_ctc_blank_fraction',
                'phonetic_hyp_len','phonetic_ref_len']:
        for stat in ['mean','std']:
            result[f'm8__phonetic__{col}__{stat}']=gg[col].agg(stat)
    for task,part in tab.groupby('task'):
        label=re.sub('[^a-z0-9]+','_',str(task).lower())[:35]
        g=part.groupby('subject_id')
        result[f'm8__phonetic__{label}__n']=g.size().reindex(result.index).fillna(0)
        for col in ['phonetic_ref_distance','phonetic_valid','phonetic_confidence_proxy']:
            result[f'm8__phonetic__{label}__{col}_mean']=g[col].mean().reindex(result.index)
    if 'item_score' in tab:
        scores=pd.to_numeric(tab.item_score,errors='coerce')
        for val,tag in [(1,'scored_correct'),(0,'scored_incorrect')]:
            part=tab[scores==val]
            result[f'm8__phonetic__ref_distance__{tag}']=part.groupby('subject_id').phonetic_ref_distance.mean().reindex(result.index)
        result['m8__phonetic__distance_correct_minus_incorrect']=(
            result['m8__phonetic__ref_distance__scored_correct']-
            result['m8__phonetic__ref_distance__scored_incorrect'])
    result.reset_index().to_csv(outdir/'phonetic_student_features.csv',index=False)
    return result


def process(data_root,output_root,model_id=MODEL,batch_size=3,max_sec=75,device='auto',limit=0,quality_only=False):
    import torch
    from transformers import AutoProcessor,WavLMForCTC
    root=Path(data_root);out=Path(output_root)/'tables';out.mkdir(parents=True,exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[phonetic] using {model_id} on {device}; batch={batch_size}',flush=True)
    processor=AutoProcessor.from_pretrained(model_id,trust_remote_code=False)
    model=WavLMForCTC.from_pretrained(model_id,trust_remote_code=False).to(device).eval()
    records=[]
    for split in ('train','test'):
        metadata=pd.read_csv(root/f'{split}_clip_metadata.csv',dtype={'subject_id':str,'filename':str})
        if limit>0:metadata=metadata.head(limit)
        assert metadata.filename.is_unique,'Duplicate filenames; cannot match tar members'
        cached=out/f'phonetic_progress_{split}.csv'
        done=pd.read_csv(cached,dtype={'filename':str}) if cached.exists() else pd.DataFrame()
        completed=set(done.filename) if len(done) else set()
        wanted={Path(f).name:f for f in metadata.filename if f not in completed}
        if len(wanted)!=(len(metadata)-len(completed.intersection(metadata.filename))):
            raise ValueError('Duplicate basenames in official clip metadata')
        if quality_only:
            qp=out/f'audio_clip_features_{split}.csv'
            if not qp.exists():raise FileNotFoundError('Run audio extraction first for quality-only phonetics')
            q=pd.read_csv(qp,dtype={'filename':str})
            good=set(q.loc[q.audio__quality_usable==1,'filename'])
            wanted={k:v for k,v in wanted.items() if v in good}
        batch=[]
        def flush(items):
            if not items:return
            wave=[w for _,w in items]
            try:results=inference_batch(model,processor,wave,device)
            except torch.cuda.OutOfMemoryError:
                if device!='cuda' or len(items)==1:raise
                torch.cuda.empty_cache()
                results=[]
                for _,w in items:results.extend(inference_batch(model,processor,[w],device))
            rows=[]
            for (filename,w),res in zip(items,results):rows.append({'filename':filename,**res})
            pd.DataFrame(rows).to_csv(cached,mode='a',header=not cached.exists(),index=False)
        with tarfile.open(root/f'{split}_audio.tar','r|*') as arc:
            with tqdm(total=len(wanted),desc=f'Pasketti {split}',unit='clip') as bar:
                found=set()
                for mem in arc:
                    if not mem.isfile():continue
                    name=Path(mem.name).name
                    if name not in wanted:continue
                    if name in found:raise ValueError(f'Repeated tar member {name}')
                    raw=arc.extractfile(mem)
                    if raw is None:continue
                    wav=decode_audio(raw.read(),max_sec=max_sec)
                    if len(wav)<1600:continue
                    batch.append((wanted[name],wav));found.add(name);bar.update(1)
                    if len(batch)>=batch_size:flush(batch);batch=[]
                flush(batch)
                if len(found)!=len(wanted):warnings.warn(f'{len(wanted)-len(found)} requested clips not found in {split} tar')
        current=pd.read_csv(cached,dtype={'filename':str}) if cached.exists() else pd.DataFrame(columns=['filename'])
        current=current.drop_duplicates('filename',keep='last')
        if current.filename.duplicated().any():raise ValueError('phonetic cache corrupt: duplicate filename')
        merged=metadata.merge(current,on='filename',how='left',validate='one_to_one')
        merged['split']=split
        refs=[expected_to_ipa(x) for x in merged.expected_text]
        merged['phonetic_reference']=[x[0] for x in refs]
        merged['phonetic_reference_valid']=[x[1] for x in refs]
        merged['phonetic_hypothesis']=merged.phonetic_hypothesis.fillna('')
        records.append(merged)
    combined=pd.concat(records,ignore_index=True)
    agg=aggregate_phonetics(combined,out)
    info={'model_id':model_id,'n_clips':len(combined),'n_students':len(agg),
          'n_hypotheses':int(combined.phonetic_hypothesis.fillna('').str.len().gt(0).sum()),
          'is_complete':limit==0,'quality_only':bool(quality_only),
          'description':'ASR/CTC phonetic hypotheses, not gold phoneme accuracy or speaker identity'}
    (out/'phonetic_extraction_summary.json').write_text(json.dumps(info,indent=2))
    print(f'[phonetic] {len(combined)} clips; {len(agg)} students; reference coverage {combined.phonetic_reference_valid.mean():.1%}',flush=True)
    print('Phonetic edit distance is an imperfect model/reference agreement score, not child phoneme correctness.')


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--model',default=MODEL);p.add_argument('--batch-size',type=int,default=3)
    p.add_argument('--device',default='auto');p.add_argument('--max-sec',type=float,default=75)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--quality-only',action='store_true')
    a=p.parse_args()
    process(a.data_root,a.output_root,a.model,a.batch_size,a.max_sec,a.device,a.limit,a.quality_only)
if __name__=='__main__':cli()