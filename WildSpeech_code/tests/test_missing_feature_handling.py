import numpy as np
import pandas as pd

from wildspeech.families import feature_family, modeling_columns
from wildspeech.fold_features import FoldFeatureBuilder
from wildspeech.literacy_indices import compute_indices


def test_eda_only_and_fold_fitted_columns_excluded_from_direct_modelling():
    cols = ["theta_blending", "theta_common_factor", "t_active_dur__mean", "t_active_dur__diff=high__mean",
            "irt_blending_theta", "ia_t_active_dur__mean", "s_bg_level_db__mean"]
    out = modeling_columns(cols, ["SCREENING", "TEMPORAL", "ITEM_DIFFICULTY", "ITEM_ADJUSTED_AUDIO"])
    assert out == ["theta_blending", "t_active_dur__mean"]
    assert feature_family("s_bg_level_db__mean") == "BACKGROUND"


def test_builder_drops_columns_missing_in_training_fold(ctx, synthetic_students):
    base = synthetic_students[["theta_blending", "theta_deletion"]].copy()
    base["t_active_dur__mean"] = np.nan
    base.loc[base.index[50:], "t_active_dur__mean"] = 1.0
    b = FoldFeatureBuilder(ctx, base=base)
    tr, va = b.build(list(base.index[:40]), list(base.index[40:]), {"SCREENING", "TEMPORAL"})
    assert "t_active_dur__mean" not in tr.columns and list(tr.columns) == list(va.columns)


def test_indices_report_unavailable_components():
    base = pd.DataFrame({"acc_all": np.linspace(0, 1, 30)}, index=[f"s{i}" for i in range(30)])
    idx, rep = compute_indices(base, list(base.index), {"index_min_components": 2})
    assert idx.shape[1] == 0
    assert all(v["status"] == "UNAVAILABLE" for v in rep.values())
