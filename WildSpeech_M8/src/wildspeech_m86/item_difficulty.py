"""I4: item-difficulty-conditional success and uncertainty; training-student LOO."""
from __future__ import annotations
import numpy as np
import pandas as pd
from wildspeech_m85.s2_items import S2ItemEncoder
from .contracts import TASKS,validate_inputs

class DifficultyProfile:
    def __init__(self,kappa=10.):
        self.kappa=float(kappa)
    def fit(self,students,clips):
        validate_inputs(students,clips)
        self.encoder=S2ItemEncoder(kappa=self.kappa).fit(students,clips)
        self.train_ids=set(students.index.astype(str))
        return self
    def _build(self,students,clips,loo):
        validate_inputs(students,clips)
        if loo and not set(students.index.astype(str)).issubset(self.train_ids):
            raise ValueError("Unfitted LOO ID")
        if not loo and set(students.index.astype(str))&self.train_ids:
            raise ValueError("Train ID cannot appear in inference")
        grouped={str(s):g for s,g in clips.groupby("subject_id",sort=False)}
        empty=clips.iloc[:0]
        rows=[]
        for sid in students.index.astype(str):
            g=grouped.get(sid,empty);res={}
            for task in TASKS:
                x=g[g.task==task]
                yy=pd.to_numeric(x.item_score,errors="coerce").to_numpy(float)
                okay=np.isfinite(yy)
                x=x.iloc[np.flatnonzero(okay)]
                yy=yy[okay]
                proba=np.asarray([self.encoder.expected_probability(task,row.expected_text,sid,loo)[0]
                                   for row in x.itertuples()],float)
                for name,mask in (("hard",proba<.33),
                                  ("mid",(proba>=.33)&(proba<=.67)),
                                  ("easy",proba>.67)):
                    res[f"i4_{task}_{name}_n"]=float(mask.sum())
                    res[f"i4_{task}_{name}_accuracy"]=float(yy[mask].mean()) if mask.any() else 0.
                    res[f"i4_{task}_{name}_excess"]=float((yy[mask]-proba[mask]).mean()) if mask.any() else 0.
                res[f"i4_{task}_expected_success_std"]=float(proba.std()) if len(proba)>1 else 0.
            rows.append(res)
        return pd.DataFrame(rows,index=students.index).fillna(0.)
    def transform_train(self,students,clips):return self._build(students,clips,True)
    def transform(self,students,clips):return self._build(students,clips,False)
