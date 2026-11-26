"""S3: theta splines fitted on the outer-training cohort only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from .contracts import TASKS


class S3ThetaSplines:
    def __init__(self,n_knots=4,degree=3):
        if n_knots<3 or degree<1:raise ValueError("invalid spline basis")
        self.n_knots=n_knots;self.degree=degree

    def fit(self,students):
        self.models={};self.medians={}
        for task in TASKS:
            col=task+"_theta"
            x=pd.to_numeric(students[col],errors="coerce").to_numpy(float)
            values=x[np.isfinite(x)]
            self.medians[col]=float(np.median(values)) if len(values) else 0.
            # Constant values cannot support meaningful quantile knots.
            if len(np.unique(values))>=self.n_knots:
                model=SplineTransformer(n_knots=self.n_knots,degree=self.degree,
                                        include_bias=False,knots="quantile",
                                        extrapolation="constant")
                model.fit(values.reshape(-1,1))
                self.models[col]=model
            else:
                self.models[col]=None
        return self

    def transform(self,students):
        result={}
        expected=self.n_knots+self.degree-2
        for task in TASKS:
            col=task+"_theta"
            x=pd.to_numeric(students[col],errors="coerce").to_numpy(float)
            miss=~np.isfinite(x)
            x=np.where(miss,self.medians[col],x)
            model=self.models[col]
            basis=model.transform(x.reshape(-1,1)) if model is not None else np.zeros((len(x),expected))
            if basis.shape[1]!=expected:
                raise ValueError("Spline basis dimensionality changed")
            for i in range(expected):
                result[f"s3_{task}_spline_{i}"]=basis[:,i]
            result[f"s3_{task}_theta_missing"]=miss.astype(float)
        return pd.DataFrame(result,index=students.index)
