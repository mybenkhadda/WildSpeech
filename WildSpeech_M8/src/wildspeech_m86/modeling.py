"""Predeclared model arms and compact, interpretable constrained estimators.

B2 uses a monotone hinge GAM. Every coefficient multiplying a raw-theta basis
or nonnegative cross-theta interaction is constrained >=0. S3 free splines and
instrument×theta terms are omitted from this arm so they cannot silently
violate the monotonicity guarantee.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from wildspeech_m85.models import make_ridge,MultiSeedCatBoost,ConvexBlend

ALL=("S2","S3","S4","S5")
SPECS={
    "B1":{"kind":"blend","m85_blocks":ALL,"extra":(),"description":"E04 full-feature Ridge/CatBoost convex blend"},
    "B2":{"kind":"monotone","m85_blocks":ALL,"extra":(),"description":"Constrained additive theta hinge GAM with positive pair interactions"},
    "B3":{"kind":"ridge","m85_blocks":("S2","S3"),"extra":(),"description":"Compact S0+S2+S3 Ridge"},
    "I1":{"kind":"ridge","m85_blocks":ALL,"extra":("I1",),"description":"Response matrix increment"},
    "I2":{"kind":"ridge","m85_blocks":ALL,"extra":("I2",),"description":"Rasch ability and uncertainty increment"},
    "I3":{"kind":"ridge","m85_blocks":ALL,"extra":("I3",),"description":"Item complexity increment"},
    "I4":{"kind":"ridge","m85_blocks":ALL,"extra":("I4",),"description":"Difficulty conditional correctness increment"},
    "I5":{"kind":"ridge","m85_blocks":ALL,"extra":("I1","I2","I3","I4"),
          "description":"Full item intelligence bundle"},
    "D2":{"kind":"partial_pool","gate":"instrument","m85_blocks":ALL,"extra":(),
          "description":"Instrument penalized deviations from common coefficients"},
    "D3":{"kind":"partial_pool","gate":"deletion","m85_blocks":ALL,"extra":(),
          "description":"Deletion availability penalized deviations"},
    "T1":{"kind":"ridge","m85_blocks":ALL,"extra":("T1",),
          "description":"Clip duration item×score-group nuisance control"},
}
RIDGE_ALPHAS=(1.,10.,100.,1000.)
BLEND_WEIGHTS=(0.,.25,.5,.75,1.)
CONFIRMATORY=("B1","B2","I5")
SCREEN_PRIORITY=("B1","B2","I5","B3","I1","I2","I3","I4","D2","D3","T1")

class MonotoneHingeGAM:
    def __init__(self,alpha=10.,n_knots=4):
        self.alpha=float(alpha);self.n_knots=int(n_knots)
        if self.alpha<=0 or self.n_knots<2:raise ValueError("Invalid GAM specification")

    def _design(self,x,fit=False):
        tasks=("blending","deletion","sentence_repetition")
        theta_cols=[t+"_theta" for t in tasks]
        # Exclude unconstrained nonlinear theta effects and theta×instrument
        # deviations. They could otherwise undermine the monotone guarantee.
        nuisance_cols=[c for c in x.columns
                       if c not in theta_cols and not c.startswith("s3_")
                       and not (c.startswith("s4_") and c.endswith("_theta"))]
        if fit:
            self.other_cols=nuisance_cols
            self.imputer=SimpleImputer(strategy="median",add_indicator=True,
                                      keep_empty_features=True)
            self.scaler=StandardScaler()
            v=self.imputer.fit_transform(x[self.other_cols].astype(float))
            other=self.scaler.fit_transform(v)
        else:
            if self.other_cols!=nuisance_cols:
                raise ValueError("Changed GAM covariates between fit and inference")
            other=self.scaler.transform(self.imputer.transform(x[self.other_cols].astype(float)))
        monotone=[];z_list=[]; missing_flags=[]
        for col in theta_cols:
            t=pd.to_numeric(x[col],errors="coerce").to_numpy(float)
            good=np.isfinite(t)
            if fit:
                med=float(np.median(t[good])) if good.any() else 0.
                std=float(np.std(t[good])) if good.sum()>1 else 1.
                std=max(std,.1)
                knots=np.unique(np.quantile((t[good]-med)/std,
                              np.linspace(.1,.9,self.n_knots))) if good.any() else np.zeros(1)
                self.theta_params=getattr(self,"theta_params",{})
                self.theta_params[col]=(med,std,knots)
            med,std,knots=self.theta_params[col]
            z=np.where(good,(t-med)/std,0.)
            z_list.append(z)
            missing_flags.append((~good).astype(float))
            monotone.append(z)            # positive base slope
            monotone.extend([np.maximum(z-k,0.) for k in knots])
        # Nonnegative pairwise hinge products => nondecreasing in each theta.
        for i,j in ((0,1),(0,2),(1,2)):
            monotone.append(np.maximum(0.,z_list[i])*np.maximum(0.,z_list[j]))
        m=np.column_stack(monotone)
        base=np.c_[np.ones(len(x)),other,np.column_stack(missing_flags)]
        self.n_unconstrained=base.shape[1]
        return np.c_[base,m]

    def fit(self,x,y):
        x=x.copy();y=np.asarray(y,dtype=float)
        design=self._design(x,fit=True)
        n,p=design.shape
        # Solve the *same* convex nonnegative-coefficient ridge objective
        # with its precomputed Gram matrix and analytic gradient. L-BFGS-B
        # avoids repeatedly solving a large dense least-squares system when
        # 11 models are evaluated across 100 outer/300 inner folds.
        penalty=np.full(p,self.alpha,dtype=float)
        penalty[0]=0.  # unpenalized intercept
        gram=design.T@design+np.diag(penalty)
        rhs=design.T@y
        bounds=[(None,None)]*self.n_unconstrained+[
                (0.,None)]*(p-self.n_unconstrained)
        x0=np.zeros(p,float)
        x0[0]=float(np.mean(y))
        def objective(coef):
            residual=gram@coef-rhs
            return (.5*float(coef@(gram@coef))-float(rhs@coef),residual)
        result=minimize(objective,x0,method="L-BFGS-B",jac=True,
                        bounds=bounds,
                        options={"maxiter":700,"ftol":1e-11,"gtol":1e-7,"maxls":40})
        if not np.isfinite(result.x).all():
            raise RuntimeError("Constrained GAM optimizer returned nonfinite coefficients")
        if not result.success:
            # A finite feasible iterate remains a valid regularized model,
            # but its optimization status must not be hidden.
            grad=result.jac
            pg=np.asarray(grad,float).copy()
            pg[self.n_unconstrained:]=np.where(
                 result.x[self.n_unconstrained:]<=1e-8,
                 np.minimum(pg[self.n_unconstrained:],0.),pg[self.n_unconstrained:])
            if np.linalg.norm(pg,ord=np.inf)>0.01:
                raise RuntimeError(f"Constrained GAM failed KKT check: {result.message}")
        self.coef_=result.x
        self.columns_=x.columns.tolist()
        self.fit_success_=True
        return self

    def predict(self,x):
        if x.columns.tolist()!=self.columns_:raise ValueError("Mismatched GAM input")
        return self._design(x).dot(self.coef_)

class PartialPoolingRidge:
    """Shared regression plus strongly shrunken subgroup deviations.

    Instrument or deletion-available groups are pooled rather than fitting
    unrelated child-level models on fewer than 541 students.
    """
    def __init__(self,alpha=10.,deviation_multiplier=20.,gate="instrument"):
        self.alpha=float(alpha)
        self.deviation_multiplier=float(deviation_multiplier)
        self.gate=gate
        if self.gate not in ("instrument","deletion"):
            raise ValueError("Unsupported partial-pooling gate")

    def _gate(self,x):
        col="is_wj4" if self.gate=="instrument" else "deletion_n_clips"
        if col not in x:raise ValueError(f"Missing observed group signal: {col}")
        v=pd.to_numeric(x[col],errors="coerce").fillna(0).to_numpy(float)
        return ((v>.5) if self.gate=="instrument" else (v>0)).astype(float)

    def _design(self,x,fit=False):
        if fit:
            self.columns_=x.columns.tolist()
            self.imp=SimpleImputer(strategy="median",add_indicator=True,
                                   keep_empty_features=True)
            self.scaler=StandardScaler()
            z=self.scaler.fit_transform(self.imp.fit_transform(x.astype(float)))
            self.gate_center=float(self._gate(x).mean())
        else:
            if x.columns.tolist()!=self.columns_:raise ValueError("Pooling schema mismatch")
            z=self.scaler.transform(self.imp.transform(x.astype(float)))
        g=self._gate(x)-self.gate_center
        base=np.c_[np.ones(len(x)),z]
        dev=base*g[:,None]
        return np.c_[base,dev]

    def fit(self,x,y):
        design=self._design(x,True);y=np.asarray(y,float)
        m=design.shape[1]//2
        ridge=np.r_[0.,np.full(m-1,self.alpha),
                    np.full(m,self.alpha*self.deviation_multiplier)]
        matrix=design.T.dot(design)+np.diag(ridge)
        self.coef_=np.linalg.solve(matrix,design.T.dot(y))
        return self

    def predict(self,x):
        return self._design(x,False).dot(self.coef_)


def new_model(spec,alpha=10.,seeds=(42,),iterations=200):
    kind=spec["kind"]
    if kind=="ridge":return make_ridge(alpha)
    if kind=="monotone":return MonotoneHingeGAM(alpha=alpha)
    if kind=="partial_pool":return PartialPoolingRidge(alpha=alpha,gate=spec["gate"])
    if kind=="cat":return MultiSeedCatBoost(seeds=seeds,iterations=iterations)
    raise ValueError("Blends require explicit nested OOF model selection")
