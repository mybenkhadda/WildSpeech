"""Run: python tests/m82_synthetic_demo.py; non-competition fixture."""
import sys
from pathlib import Path
from test_m82 import fake_frame
from wildspeech_m8.m82_pipeline import M82Config,run_feature_frame

if __name__=='__main__':
    X,y,schools=fake_frame(75)
    output=Path(sys.argv[1] if len(sys.argv)>1 else '/tmp/m82_synthetic_demo')
    cfg=M82Config(mode='SMOKE',max_per_view=12,run_predictive=True,
                  predictive_folds=3,seed=1337,resamples=2)
    result=run_feature_frame(X,y,schools,output,cfg)
    assert result['pairs']==75*74//2
    assert len(result['figures'])>=15,result['figures']
    assert (output/'tables'/'pairwise_predictive_oof.csv').is_file()
    assert (output/'M82_RESEARCH_ATLAS.html').is_file()
    print('Synthetic integration passed; outputs:',output)