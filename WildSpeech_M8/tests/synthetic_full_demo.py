"""FULL synthetic integration test; NEVER cite its scores as competition results.
Run: python tests/synthetic_full_demo.py --output /tmp/m81_demo
"""
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
# pip install -e . installs src/wildspeech_m8
from wildspeech_m8.m8_pipeline import Config,run
from wildspeech_m8.m8_research import ResearchConfig,run_research
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output)/'fake_official_data';root.mkdir(parents=True,exist_ok=True)
out=Path(a.output)/'generated_outputs';(out/'tables').mkdir(parents=True,exist_ok=True)
r=np.random.default_rng(2026);n=80;m=24
ids=[f'SYNTRAIN{i:03d}' for i in range(n)]+[f'SYNTEST{i:03d}' for i in range(m)]
bl=r.normal(size=n+m);de=.6*bl+.65*r.normal(size=n+m)
sentence=.5*bl+.7*r.normal(size=n+m)
pause=np.clip(.4-.18*bl+.4*r.normal(size=n+m),.05,1.7)
phon=np.clip(.55-.08*bl+.13*r.normal(size=n+m),.02,.98)
prof=pd.DataFrame({'subject_id':ids,'blending_theta':bl,'deletion_theta':de,
                   'sentence_repetition_theta':sentence,'profile__score_mean':.5*bl+.5*de,
                   'profile__mean_duration':np.exp(1.8+pause),
                   'acoustic__spectral_centroid':1500+250*pause+200*r.normal(size=n+m),
                   'task_level_accuracy_blending':np.clip(.55+.18*bl,0,1),
                   'task_level_accuracy_deletion':np.clip(.4+.22*de,0,1)})
prof.to_csv(root/'profiles.csv',index=False)
tr=pd.DataFrame({'subject_id':ids[:n],'school_id':[f'S{i%6}' for i in range(n)],'instrument':r.choice(['KTEA','WJ'],n)})
te=pd.DataFrame({'subject_id':ids[n:],'school_id':[f'S{i%6}' for i in range(m)],'instrument':r.choice(['KTEA','WJ'],m)})
tr.to_csv(root/'train_student_metadata.csv',index=False);te.to_csv(root/'test_student_metadata.csv',index=False)
y=100+7*bl[:n]+3*de[:n]-6*pause[:n]-3*phon[:n]+4*((bl[:n]>.3)&(pause[:n]>.35))+r.normal(0,8,n)
pd.DataFrame({'subject_id':tr.subject_id,'outcome_score':y}).to_csv(root/'train_labels.csv',index=False)
te[['subject_id']].to_csv(root/'submission_format.csv',index=False)
for split,meta in [('train',tr),('test',te)]:
    rows=[]
    for i,sid in enumerate(meta.subject_id):
        for j in range(9):
            rows.append({'subject_id':sid,'filename':f'{sid}_{j}.wav',
                         'task':['blending','deletion','sentence_repetition'][j%3],
                         'item_score':int(r.random()<.6),'expected_text':['cat','bat','Put the chicken in the freezer.'][j%3],
                         'duration_sec':float(r.uniform(1,8))})
    pd.DataFrame(rows).to_csv(root/f'{split}_clip_metadata.csv',index=False)
audio=pd.DataFrame({'subject_id':ids,
    'm8__audio__quality_coverage':r.uniform(.65,.95,n+m),
    'm8__audio__onset_s_proxy__mean':pause,
    'm8__audio__reliable_onset_s_proxy__mean':pause+r.normal(0,.1,n+m),
    'm8__audio__pause_count_proxy__mean':2+2*pause+r.normal(size=n+m),
    'm8__audio__spectral_entropy_mean__mean':r.normal(.5,.1,n+m),
    'm8__audio__task_blending__onset_s_proxy':pause+.05*r.normal(size=n+m),
    'm8__audio__task_deletion__onset_s_proxy':pause+.08*r.normal(size=n+m),
    'm8__audio__task_sentence_repetition__onset_s_proxy':pause+.1*r.normal(size=n+m)})
audio.to_csv(out/'tables'/'audio_student_features.csv',index=False)
(out/'tables'/'audio_extraction_summary.json').write_text(json.dumps({'is_complete':True,'n_clips':(n+m)*9,'note':'SYNTHETIC_GENERATED_DO_NOT_USE_FOR_CLAIMS'}))
ph=pd.DataFrame({'subject_id':ids,'m8__phonetic__phonetic_ref_distance__mean':phon,
                 'm8__phonetic__blending__phonetic_ref_distance_mean':phon+.1*r.normal(size=n+m),
                 'm8__phonetic__deletion__phonetic_ref_distance_mean':phon+.1*r.normal(size=n+m)})
ph.to_csv(out/'tables'/'phonetic_student_features.csv',index=False)
(out/'SYNTHETIC_NOT_COMPETITION_RESULTS.txt').write_text('ALL SCORES AND FIGURES HERE ARE FROM SYNTHETIC DATA, NOT OFFICIAL WILDSPEECH DATA.\n')
cfg=Config(data_root=str(root),profile_csv=str(root/'profiles.csv'),output_root=str(out),quick=False,
           cv_folds=3,cv_repeats=1,inner_folds=2,bootstrap_n=50,max_features=220)
run(cfg)
res=run_research(cfg,ResearchConfig(folds=3,repeats=1,permutations=2,bootstrap=80,stability_resamples=5,quick=False))
sub=pd.read_csv(out/'predictions'/'m8_submission.csv')
assert len(sub)==m and sub.outcome_score.notna().all()
assert len(list((out/'figures').glob('E*.png')))>=10
print('FULL SYNTHETIC PASS:',res)