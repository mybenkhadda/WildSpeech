import numpy as np
import pandas as pd
import pytest
from wildspeech_m84.folds import build_folds,freeze_folds,fold_digest,validate_folds


def test_freeze_reproducible_and_school_disjoint(tmp_path):
    ids=[f"u{i}" for i in range(93)]
    groups=np.array([f"s{i%17}" for i in range(93)])
    f=build_folds(ids,groups)
    assert len(f)==100
    assert len(f[f.scheme=="school"])==50
    assert len(f[f.scheme=="student"])==50
    assert validate_folds(f,ids,groups)
    x=fold_digest(f,ids,groups)
    assert fold_digest(build_folds(ids,groups),ids,groups)==x
    assert freeze_folds(f,ids,groups,tmp_path)==x
    assert freeze_folds(f,ids,groups,tmp_path)==x
    assert (tmp_path/"folds"/"folds.parquet").is_file()


def test_changed_order_hash_rejected(tmp_path):
    ids=[f"u{i}" for i in range(85)]
    groups=np.array([f"s{i%17}" for i in range(85)])
    x=build_folds(ids,groups)
    freeze_folds(x,ids,groups,tmp_path)
    ids2=ids[::-1]
    with pytest.raises(RuntimeError,match="FROZEN FOLD HASH"):
        freeze_folds(build_folds(ids2,groups),ids2,groups,tmp_path)


def test_not_enough_school_groups():
    with pytest.raises(ValueError):
        build_folds([str(i) for i in range(25)],["one"]*25)
