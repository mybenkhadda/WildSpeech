"""CPU-efficient, fixed-budget screening estimators. All tuning is inner-fold-only."""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS=(.1,1.,10.,100.,1000.)
CAT_PARAMS={"iterations":600,"depth":2,"learning_rate":.03,
            "l2_leaf_reg":20.,"loss_function":"RMSE"}


def make_ridge(alpha):
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=True,
                                 keep_empty_features=True)),
        ("scaler",StandardScaler()),
        ("ridge",Ridge(alpha=float(alpha),solver="auto")),
    ])


class MultiSeedCatBoost:
    """Five fixed seeds for FULL; one seed for nonpromotable SCREEN."""
    def __init__(self,seeds=(42,43,44,45,46),iterations=600,depth=2,l2_leaf_reg=20.):
        self.seeds=tuple(seeds)
        self.iterations=int(iterations)
        self.depth=int(depth)
        self.l2_leaf_reg=float(l2_leaf_reg)
        if not self.seeds:
            raise ValueError("At least one CatBoost seed required")

    def fit(self,x,y):
        numeric=x.astype(float)
        self.models=[]
        for seed in self.seeds:
            model=CatBoostRegressor(
                iterations=self.iterations,depth=self.depth,
                learning_rate=.03,l2_leaf_reg=self.l2_leaf_reg,
                loss_function="RMSE",verbose=False,random_seed=int(seed),
                allow_writing_files=False,thread_count=2,
                random_strength=.5,
            )
            model.fit(numeric,np.asarray(y,dtype=float))
            self.models.append(model)
        return self

    def predict(self,x):
        vals=[np.asarray(m.predict(x.astype(float)),dtype=float) for m in self.models]
        out=np.mean(vals,axis=0)
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite CatBoost ensemble prediction")
        return out


class ConvexBlend:
    """Weights selected from training-side inner OOF, not outer valid targets."""
    def __init__(self,weight_cat,linear,cat):
        self.weight_cat=float(weight_cat)
        if not 0<=self.weight_cat<=1:
            raise ValueError("Nonconvex blend weight")
        self.linear=linear;self.cat=cat

    def predict(self,x):
        return ((1-self.weight_cat)*np.asarray(self.linear.predict(x),dtype=float)
                +self.weight_cat*np.asarray(self.cat.predict(x),dtype=float))
