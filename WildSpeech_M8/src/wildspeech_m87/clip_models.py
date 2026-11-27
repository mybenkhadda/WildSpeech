"""Group-cross-fitted, strictly training-only clip correctness and missing scores.

Targets are item_score (observed on training clips), NEVER future reading
outcome. Every training-clip OOF prediction is from a model that excluded its
entire student. New-student predictions come from a model fitted only on
outer-training students.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

AUDIO_INPUTS=("child_phone_edit_rate","child_target_logp_per_frame",
              "child_posterior_entropy","child_aligned_phone_confidence",
              "child_posterior_margin","whole_phone_edit_rate",
              "task_phone_graded_accuracy","task_phone_omission_rate",
              "task_phone_substitution_rate","task_phone_order_ratio",
              "localization_confidence","rms","noise_floor","clipping_frac",
              "vad_speech_seconds","word_coverage","word_lcs_ratio")

class ClipCorrectness:
    def __init__(self,alpha=30.,n_splits=4):
        self.alpha=float(alpha);self.n_splits=int(n_splits)
    def _x(self,d):
        f=pd.DataFrame(index=d.index)
        for c in AUDIO_INPUTS:
            f[c]=pd.to_numeric(d[c],errors="coerce") if c in d else np.nan
        for task in ("blending","deletion","sentence_repetition"):
            f["task_"+task]=(d.task.astype(str)==task).astype(float)
        f["expected_chars"]=d.expected_text.fillna("").astype(str).str.len().astype(float)
        f["expected_words"]=d.expected_text.fillna("").astype(str).str.split().str.len().astype(float)
        return f.fillna(np.nan)
    def _new(self):
        return make_pipeline(SimpleImputer(strategy="median",
                              keep_empty_features=True,add_indicator=False),
                             StandardScaler(),
                             Ridge(alpha=self.alpha))
    def fit_predict(self,train,heldout):
        d=train.copy()
        score=pd.to_numeric(d.item_score,errors="coerce")
        valid=score.notna() & d.status.eq("VALID")
        observed=d.loc[valid].copy()
        if len(observed)<10:raise ValueError("Too few scored clips to train correctness")
        groups=observed.subject_id.astype(str).to_numpy()
        n=len(np.unique(groups))
        if n<self.n_splits:raise ValueError("Too few training students")
        train_predictions=pd.Series(np.nan,index=d.index,dtype=float)
        oof=np.empty(len(observed),dtype=float);oof[:]=np.nan
        kfold=GroupKFold(n_splits=self.n_splits)
        features=self._x(observed)
        for it,iv in kfold.split(features,groups=groups):
            m=self._new().fit(features.iloc[it],score.loc[observed.index[it]].to_numpy(float))
            oof[iv]=m.predict(features.iloc[iv])
        if not np.isfinite(oof).all():raise RuntimeError("Incomplete grouped clip OOF")
        train_predictions.loc[observed.index]=np.clip(oof,0,1)
        self.model=self._new().fit(features,score.loc[observed.index].to_numpy(float))
        train_other=d.status.eq("VALID") & ~valid
        if train_other.any():
            train_predictions.loc[train_other]=np.clip(
                self.model.predict(self._x(d.loc[train_other])),0,1)
            # Unscored TRAIN clips were not included in supervised fits.
        valpred=pd.Series(np.nan,index=heldout.index,dtype=float)
        okay=heldout.status.eq("VALID")
        if okay.any():
            valpred.loc[okay]=np.clip(self.model.predict(self._x(heldout.loc[okay])),0,1)
        return self._enrich(d,train_predictions),self._enrich(heldout,valpred)

    @staticmethod
    def _enrich(d,pred):
        out=d.copy()
        out["clip_pred_correct"]=pred
        y=pd.to_numeric(out.item_score,errors="coerce") if "item_score" in out else pd.Series(np.nan,index=out.index)
        out["clip_surprise"]=np.where(y.notna(),pred-y,np.nan)
        out["clip_unscored_recovery"]=np.where(y.isna(),pred,np.nan)
        out["clip_pred_on_correct"]=np.where(y>=.5,pred,np.nan)
        out["clip_pred_on_incorrect"]=np.where(y<.5,pred,np.nan)
        return out
