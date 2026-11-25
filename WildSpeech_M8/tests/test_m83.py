"""M8.3 synthetic tests; not official competition evidence."""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

def load_module():
    package=types.ModuleType("wildspeech_m8");package.__path__=[]
    prior=types.ModuleType("wildspeech_m8.m8_pipeline")
    prior.Config=lambda **kw:None
    prior.prepare_dataset=lambda cfg:None
    class Encoder:
        def fit(self,df):self.cols=list(df);self.imp=SimpleImputer().fit(df);return self
        def transform(self,df):return self.imp.transform(df[self.cols])
    prior.FoldEncoder=Encoder
    prior.make_estimator=lambda seed,fast:HistGradientBoostingRegressor(max_iter=25,min_samples_leaf=4,random_state=seed)
    prior.choose_features=lambda f,limit:list(f.columns)[:limit]
    prior.feature_families=lambda cols:{c:("screening" if c.startswith("screening") else "audio") for c in cols}
    prior.assert_safe_features=lambda cols:None
    sys.modules["wildspeech_m8"]=package
    sys.modules["wildspeech_m8.m8_pipeline"]=prior
    path=Path(__file__).resolve().parents[1]/"src/wildspeech_m8/m83_bottleneck.py"
    spec=importlib.util.spec_from_file_location("wildspeech_m8.m83_bottleneck",path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def test_synthetic_integration(tmp_path):
    m=load_module();rng=np.random.default_rng(123);n=72
    frame=pd.DataFrame({"screening_skill":rng.normal(size=n),"audio_quality":rng.normal(size=n),"audio_duration":rng.normal(size=n)})
    y=3*frame.screening_skill.to_numpy()+rng.normal(size=n)
    groups=np.repeat(["s1","s2","s3","s4","s5","s6"],12)
    cfg=m.AuditConfig(folds=3,repeats=1,quick=True,bootstrap=10)
    report=m.run_frame(frame,y,groups,tmp_path,cfg)
    assert report["rows"]==n
    assert (tmp_path/"M83_BOTTLENECK_REPORT.md").is_file()
    fold=pd.read_csv(tmp_path/"tables/fold_metrics.csv")
    assert set(fold.split)=={"random","grouped"}
    for (_,_,name),g in pd.read_csv(tmp_path/"tables/oof_predictions_PRIVATE.csv").groupby(["split","repeat","model"]):
        assert sorted(g.row.tolist())==list(range(n))
    assert pd.read_csv(tmp_path/"tables/residual_probe.csv").mae_improvement.notna().all()

def test_group_disjoint():
    m=load_module();groups=np.repeat(["a","b","c","d"],6)
    for _,_,tr,va in m.make_splits(len(groups),groups,m.AuditConfig(folds=3,repeats=2),True):
        assert not set(groups[tr])&set(groups[va])
