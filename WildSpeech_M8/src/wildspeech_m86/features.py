"""Raw-screening-to-feature representation with nested, no-test-fit transforms.

M8.5 S0..S5 remain untouched. M8.6 I1/I2/I3/I4/T1 are optional,
composable additions. All train representations are student-LOO/cross-fitted
where reference statistics depend on a student's own answers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from wildspeech_m85.features import ScreeningFeatures
from wildspeech_m85.nested_cv import FULL_BLOCKS
from .contracts import ITEM_BLOCKS,validate_inputs,ensure_predictor_columns
from .item_matrix import ItemMatrix
from .rasch_ability import RaschAbility
from .item_complexity import complexity_features
from .item_difficulty import DifficultyProfile
from .duration_controls import DurationControls


class M86Features:
    def __init__(self,blocks=ITEM_BLOCKS,rasch_iters=7,rasch_crossfit=3):
        self.blocks=tuple(blocks)
        if len(set(self.blocks))!=len(self.blocks) or set(self.blocks)-set(ITEM_BLOCKS):
            raise ValueError("Unsupported or duplicated M8.6 feature block")
        self.rasch_iters=int(rasch_iters)
        self.rasch_crossfit=int(rasch_crossfit)

    def fit(self,students,clips):
        validate_inputs(students,clips)
        self.training_order=students.index.astype(str).tolist()
        self.training_ids=set(self.training_order)
        self.core=ScreeningFeatures(FULL_BLOCKS)
        self.core.fit(students,clips)
        self.modules={}
        if "I1" in self.blocks:self.modules["I1"]=ItemMatrix().fit(students,clips)
        if "I2" in self.blocks:
            self.modules["I2"]=RaschAbility(outer_iters=self.rasch_iters,
                                             crossfit=self.rasch_crossfit).fit(students,clips)
        if "I4" in self.blocks:self.modules["I4"]=DifficultyProfile().fit(students,clips)
        if "T1" in self.blocks:self.modules["T1"]=DurationControls().fit(students,clips)
        return self

    def _assemble(self,students,clips,train):
        validate_inputs(students,clips)
        ids=students.index.astype(str).tolist()
        if train and ids!=self.training_order:
            raise ValueError("Training student order or membership changed")
        if not train and set(ids)&self.training_ids:
            raise ValueError("Training student passed to holdout/test transform")
        pieces=[self.core._build(students,clips,train)]
        for block in self.blocks:
            if block=="I3":
                pieces.append(complexity_features(students,clips))
            else:
                m=self.modules[block]
                pieces.append(m.transform_train(students,clips) if train else
                              m.transform(students,clips))
        out=pd.concat(pieces,axis=1)
        ensure_predictor_columns(out)
        if not out.index.equals(students.index):
            raise ValueError("M8.6 features lost student order")
        return out

    def fit_transform(self,students,clips):
        self.fit(students,clips)
        x=self._assemble(students,clips,True)
        self.columns_=x.columns.tolist()
        return x

    def transform(self,students,clips):
        if not hasattr(self,"columns_"):
            raise RuntimeError("fit_transform required before independent prediction")
        x=self._assemble(students,clips,False)
        if x.columns.tolist()!=self.columns_:
            raise ValueError("M8.6 feature schema changed at inference")
        return x


def select_features(x,model_id):
    """Single canonical block selector; NEVER use naming heuristics for source."""
    from .modeling import SPECS
    if model_id not in SPECS:raise KeyError(model_id)
    spec=SPECS[model_id]
    m85=set(spec.get("m85_blocks",FULL_BLOCKS))
    keep=[]
    for column in x.columns:
        family=next((p.upper() for p in ("s2","s3","s4","s5")
                     if column.startswith(p+"_")),None)
        if family is not None and family not in m85:continue
        extra=next((p.upper() for p in ("i1","i2","i3","i4","t1")
                    if column.startswith(p+"_")),None)
        if extra is not None and extra not in spec.get("extra",()):continue
        keep.append(column)
    out=x[keep].copy()
    ensure_predictor_columns(out)
    return out


def assert_raw_independence(builder,model,predictor,students,clips,model_id,atol=1e-7):
    """Per-student RAW source through frozen transformations & final predictor."""
    from .features import select_features
    x=select_features(builder.transform(students,clips),model_id)
    batch=np.asarray(predictor(model,x),float)
    if len(batch)!=len(students) or not np.isfinite(batch).all():
        raise ValueError("Model returned missing/nonfinite test predictions")
    for i,sid in enumerate(students.index.astype(str)):
        one=students.iloc[[i]]
        one_clips=clips[clips.subject_id.astype(str)==sid]
        local_x=select_features(builder.transform(one,one_clips),model_id)
        xa=x.iloc[[i]]
        if local_x.columns.tolist()!=xa.columns.tolist() or not np.allclose(
             local_x.to_numpy(float),xa.to_numpy(float),equal_nan=True,rtol=0,atol=atol):
            raise AssertionError(f"Test feature depends on other rows: index={i}")
        pred=np.asarray(predictor(model,local_x),float)
        if not np.isclose(pred[0],batch[i],atol=atol,rtol=0):
            raise AssertionError(f"Test prediction depends on batch: index={i}")
    permutation=np.random.default_rng(42).permutation(len(students))
    perm=select_features(builder.transform(students.iloc[permutation],clips),model_id)
    if not np.allclose(np.asarray(predictor(model,perm),float),batch[permutation],rtol=0,atol=atol):
        raise AssertionError("Test order influences fitted-model prediction")
    return batch
