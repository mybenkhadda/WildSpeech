"""Optional existing-ASR integration: reference-conditioned, uncertainty-aware.

Expected text is NOT a verified child transcript. Edit distance describes the
ASR hypothesis versus task prompt, never certified phoneme correctness.
No audio or transcripts are sent to a third-party API.
"""
from __future__ import annotations
import argparse,re,unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

def normalize(x):
    if pd.isna(x):return ''
    s=unicodedata.normalize('NFKC',str(x)).lower()
    return ' '.join(re.findall(r"[a-z0-9']+",s))

def edit_distance(a,b):
    # Dynamic program for short word/sentence hypotheses.
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]

def pair_features(reference,hypothesis):
    a=normalize(reference);b=normalize(hypothesis)
    if not a or not b:return {'asr__valid':0.,'asr__char_sim':np.nan,'asr__word_error_rate':np.nan,'asr__word_count':len(b.split()),'asr__ref_word_count':len(a.split())}
    aa=a.split();bb=b.split();char=edit_distance(a,b);word=edit_distance(aa,bb)
    return {'asr__valid':1.,'asr__char_sim':max(0,1-char/max(len(a),len(b))),
            'asr__word_error_rate':word/max(len(aa),1),
            'asr__word_count':len(bb),'asr__ref_word_count':len(aa),
            'asr__word_count_diff':len(bb)-len(aa),
            'asr__exact_match':float(a==b),
            'asr__length_ratio':len(b)/max(len(a),1)}

def build(data_root,asr_path,output_root):
    root=Path(data_root);a=Path(asr_path);out=Path(output_root)/'tables';out.mkdir(parents=True,exist_ok=True)
    asr=pd.read_parquet(a) if a.suffix=='.parquet' else pd.read_csv(a)
    if 'filename' not in asr:raise ValueError('ASR export must have filename')
    hyp=next((c for c in ('asr_transcript','transcript','hypothesis','predicted_text') if c in asr),None)
    if hyp is None:raise ValueError('Missing recognized transcript column')
    clips=pd.concat([pd.read_csv(root/f'{s}_clip_metadata.csv',dtype={'subject_id':str,'filename':str}).assign(split=s) for s in ('train','test')])
    if 'expected_text' not in clips:raise ValueError('Official clip metadata needs expected_text')
    # Split-qualified identity to prevent cross-split filename collisions.
    if 'split' in asr.columns:
        if asr.duplicated(['split','filename']).any():raise ValueError('Duplicate split/filename ASR')
        keys=['split','filename']
    else:
        if clips.filename.duplicated().any():raise ValueError('Ambiguous filename across train/test: require split in ASR file')
        keys=['filename']
    joined=clips.merge(asr[keys+[hyp]],how='left',on=keys,validate='one_to_one')
    rows=[pair_features(r.expected_text,getattr(r,hyp)) for r in joined.itertuples(index=False)]
    features=pd.concat([joined[['split','subject_id','filename','task']].reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    features.to_csv(out/'speech_clip_features.csv',index=False)
    nums=[c for c in features if c.startswith('asr__')]
    agg=features.groupby('subject_id')[nums].agg(['mean','std']);agg.columns=['m8__speech__'+a+'__'+b for a,b in agg.columns]
    for task,part in features.groupby('task'):
        suffix=re.sub('[^a-z0-9]+','_',str(task).lower())[:30]
        for col in ('asr__word_error_rate','asr__char_sim','asr__exact_match'):
            agg[f'm8__speech__{suffix}__{col}']=part.groupby('subject_id')[col].mean().reindex(agg.index)
    agg.reset_index().to_csv(out/'speech_student_features.csv',index=False)
    print(f'[M8 speech] {len(features)} clips, {len(agg)} students, ASR coverage {features.asr__valid.mean():.2%}')
    print('WARNING: reference-conditioned ASR similarities are not verified child phonetic errors.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--asr-path',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args();build(a.data_root,a.asr_path,a.output_root)