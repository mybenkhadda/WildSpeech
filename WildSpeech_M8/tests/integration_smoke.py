import tempfile,sys
from pathlib import Path
import numpy as np,pandas as pd
# pip install -e . installs src/wildspeech_m8
from wildspeech_m8.m8_pipeline import Config,run
from wildspeech_m8.m8_research import ResearchConfig,run_research
r=np.random.default_rng(42);n=45;m=16
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'data';root.mkdir();out=Path(tmp)/'outputs'
    tr=pd.DataFrame({'subject_id':[f't{i}' for i in range(n)],'school_id':[f's{i%5}' for i in range(n)],'instrument':r.choice(['KTEA','WJ'],n)})
    te=pd.DataFrame({'subject_id':[f'e{i}' for i in range(m)],'school_id':[f's{i%5}' for i in range(m)],'instrument':r.choice(['KTEA','WJ'],m)})
    x=r.normal(size=n+m)
    prof=pd.DataFrame({'subject_id':pd.concat([tr.subject_id,te.subject_id],ignore_index=True),
                       'blending_theta':x,'deletion_theta':r.normal(size=n+m),'profile__score_mean':r.normal(size=n+m),
                       'profile__mean_duration':r.uniform(1,20,n+m),'acoustic__spectral_centroid':r.normal(size=n+m)})
    prof.to_csv(root/'profiles.csv',index=False)
    y=100+8*x[:n]+r.normal(0,8,n)
    pd.DataFrame({'subject_id':tr.subject_id,'outcome_score':y}).to_csv(root/'train_labels.csv',index=False)
    te[['subject_id']].to_csv(root/'submission_format.csv',index=False)
    tr.to_csv(root/'train_student_metadata.csv',index=False);te.to_csv(root/'test_student_metadata.csv',index=False)
    for split,frame in [('train',tr),('test',te)]:
        clips=pd.DataFrame([{'subject_id':sid,'filename':f'{sid}_{j}.wav','task':['blending','deletion'][j%2], 'item_score':r.binomial(1,.5),'duration_sec':r.uniform(1,6),'expected_text':'abc'} for sid in frame.subject_id for j in range(3)])
        clips.to_csv(root/f'{split}_clip_metadata.csv',index=False)
    cfg=Config(data_root=str(root),profile_csv=str(root/'profiles.csv'),output_root=str(out),quick=True,loso=False,cv_folds=3,cv_repeats=1,inner_folds=2)
    run(cfg)
    assert (out/'figures'/'04_oof_calibration.png').exists()
    res=run_research(cfg,ResearchConfig(folds=3,repeats=1,permutations=1,bootstrap=45,stability_resamples=3,quick=True))
    assert (out/'RESEARCH_ATLAS.html').exists()
    assert (out/'RESEARCH_SUMMARY.md').exists()
    assert res['num_figures']>=7
    print('PASS: integration smoke; enhanced figures =',res['num_figures'])