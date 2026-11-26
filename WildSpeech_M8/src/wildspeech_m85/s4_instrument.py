"""S4: deterministic observed-instrument interactions; no school correction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .contracts import TASKS


def instrument_features(students,reference):
    wj=(students.instrument.astype(str).str.casefold()=="wj4").astype(float).to_numpy()
    result={}
    for task in TASKS:
        theta=pd.to_numeric(students[task+"_theta"],errors="coerce").to_numpy(float)
        nclips=pd.to_numeric(reference[f"{task}_n_clips"],errors="coerce").fillna(0).to_numpy(float)
        result[f"s4_wj4_x_{task}_theta"]=wj*theta
        result[f"s4_wj4_x_{task}_administered"]=wj*(nclips>0)
        result[f"s4_wj4_x_{task}_log_clips"]=wj*np.log1p(nclips)
    return pd.DataFrame(result,index=students.index)
