"""Fold-fit, raw-metadata-to-student-features pipeline.

S0 mirrors the immutable reference. S2 uses leave-one-student-out statistics
for fitting students and training-only fitted statistics for all unseen students.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from wildspeech_m84.reference import metadata_features,REFERENCE_COLUMNS
from .contracts import validate_student_inputs,assert_disjoint,assert_predictive_columns
from .s2_items import S2ItemEncoder
from .s3_splines import S3ThetaSplines
from .s4_instrument import instrument_features
from .s5_task_pattern import task_pattern_features

BLOCKS=("S2","S3","S4","S5")


class ScreeningFeatures:
    def __init__(self,blocks=(),kappa=10.,n_knots=4):
        blocks=tuple(blocks)
        if len(set(blocks))!=len(blocks) or set(blocks)-set(BLOCKS):
            raise ValueError("Unknown or duplicate feature block")
        self.blocks=blocks;self.kappa=kappa;self.n_knots=n_knots

    def fit(self,students,clips):
        validate_student_inputs(students,clips)
        self.train_index=students.index.astype(str).tolist()
        self.train_id_set=set(self.train_index)
        self.s2=S2ItemEncoder(self.kappa).fit(students,clips) if "S2" in self.blocks else None
        self.s3=S3ThetaSplines(self.n_knots).fit(students) if "S3" in self.blocks else None
        self.fitted=True
        return self

    def _build(self,students,clips,train):
        if not getattr(self,"fitted",False):
            raise RuntimeError("Fit training cohort before feature inference")
        validate_student_inputs(students,clips)
        if train:
            if students.index.astype(str).tolist()!=self.train_index:
                raise ValueError("Training feature generation changed student order or membership")
        elif set(students.index.astype(str))&self.train_id_set:
            raise ValueError("No training students in validation/test transform")
        ref=metadata_features(clips,students)
        pieces=[ref]
        if "S2" in self.blocks:
            pieces.append(self.s2.transform_train(students,clips) if train else self.s2.transform(students,clips))
        if "S3" in self.blocks:
            pieces.append(self.s3.transform(students))
        if "S4" in self.blocks:
            pieces.append(instrument_features(students,ref))
        if "S5" in self.blocks:
            pieces.append(task_pattern_features(students,clips))
        result=pd.concat(pieces,axis=1)
        assert_predictive_columns(result.columns)
        if result.index.has_duplicates or not result.index.equals(students.index):
            raise ValueError("Feature index mismatch")
        return result

    def fit_transform(self,students,clips):
        self.fit(students,clips)
        features=self._build(students,clips,True)
        self.columns_=features.columns.tolist()
        return features

    def transform(self,students,clips):
        result=self._build(students,clips,False)
        if result.columns.tolist()!=self.columns_:
            raise ValueError("Different predictive columns at inference")
        return result

    def _test_singleton(self,students,clips,model,atol=1e-8):
        """Raw metadata+clips -> fitted preprocessing -> predictions, by student."""
        all_features=self.transform(students,clips)
        batched=np.asarray(model.predict(all_features),float).reshape(-1)
        if not np.isfinite(batched).all():
            raise ValueError("Nonfinite inference predictions")
        for i,sid in enumerate(students.index.astype(str)):
            single_meta=students.iloc[[i]]
            single_clips=clips.loc[clips.subject_id.astype(str)==sid].copy()
            local_features=self.transform(single_meta,single_clips)
            if not np.allclose(local_features.to_numpy(dtype=float),
                               all_features.iloc[[i]].to_numpy(dtype=float),
                               equal_nan=True,rtol=0,atol=atol):
                raise AssertionError(f"Raw student feature depends on neighboring rows: {sid}")
            single_prediction=float(model.predict(local_features)[0])
            if not np.isclose(single_prediction,batched[i],rtol=0,atol=atol):
                raise AssertionError(f"Single-student inference mismatch: {sid}")
        permutation=np.random.default_rng(42).permutation(len(students))
        shuffled=self.transform(students.iloc[permutation],clips)
        if not np.allclose(np.asarray(model.predict(shuffled),float),
                           batched[permutation],rtol=0,atol=atol):
            raise AssertionError("Batch order changes predictions")
        return batched
