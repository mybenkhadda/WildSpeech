"""Training-only audio perturbation stability: a cheap nuisance audit.

Reprocess a small stratified TRAIN subset at 0.9× / 1.1× speed and
Gaussian noise; compare WavLM posterior summary features. Does not retrain.
These transforms change phones/timing and are *stress tests*, not invariances
guaranteed by the data-generation process.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wildspeech_m84.data import atomic_json
from .ctc_alignment import posterior_features

def _resample_speed(x,rate):
    from scipy.signal import resample_poly
    import fractions
    frac=fractions.Fraction(1/rate).limit_denominator(100)
    return resample_poly(x,frac.numerator,frac.denominator).astype(np.float32)

def run_stability(inventory,source,model,target_map,root,max_clips=45,seed=42):
    train=inventory[inventory.split=="train"]
    n_per=max(1,max_clips//3)
    sample=pd.concat([g.sample(min(n_per,len(g)),random_state=seed)
                      for _,g in train.groupby("task")]).head(max_clips)
    rng=np.random.default_rng(seed)
    records=[]
    for row in sample.itertuples(index=False):
        try:
            x,sr,_,_=source.decode(row.filename)
            if len(x)>model.max_seconds*sr:continue
            base=model.infer_one(x)
            transforms={
                "slow_0.9":_resample_speed(x,.9),
                "fast_1.1":_resample_speed(x,1.1),
                "noise_0.01":(x+rng.normal(0,.01,size=len(x))).astype(np.float32),
            }
            target=target_map.get((row.task,str(row.expected_text)),[])
            before=posterior_features(base.logprobs,target,model.blank)
            for name,wav in transforms.items():
                if len(wav)>model.max_seconds*sr:continue
                changed=model.infer_one(wav)
                after=posterior_features(changed.logprobs,target,model.blank)
                for feature in ("posterior_entropy","posterior_margin",
                                "target_logp_per_frame"):
                    a=before.get(feature);b=after.get(feature)
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        records.append({"uid":row.uid,"task":row.task,"transform":name,
                                        "feature":feature,"before":float(a),"after":float(b),
                                        "abs_delta":abs(a-b)})
        except Exception as exc:
            records.append({"uid":row.uid,"transform":"FAILED","feature":"ERROR",
                            "failure":str(exc)[:250]})
    p=Path(root);p.mkdir(parents=True,exist_ok=True)
    d=pd.DataFrame(records)
    d.to_csv(p/"stability_train_PRIVATE.csv",index=False)
    valid=d[d.feature!="ERROR"]
    if len(valid):
        f,ax=plt.subplots(figsize=(9,5))
        valid.boxplot(column="abs_delta",by=["transform","feature"],
                      ax=ax,rot=70)
        f.suptitle("");ax.set(title="Training-only speech perturbation stability",
                              ylabel="Absolute scalar feature change")
        f.tight_layout();f.savefig(p/"stability_feature_deltas.png",dpi=160)
        plt.close(f)
    summary={"n_train_clips_attempted":len(sample),"n_scalar_comparisons":len(valid),
             "note":"Perturbations are stress tests; no cross-test training or generated test labels"}
    atomic_json(p/"stability_results.json",summary)
    return summary
