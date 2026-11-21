"""End-to-end V2 integration on a fresh synthetic project (CPU): P00-P02 then P16 -> P17 -> P19 -> P21 -> P22.
Phonetic (P18) and foundation-model (P20) stages are disabled (no local models in a test environment), so the
run also verifies that missing optional modalities are handled. Test inference stays disabled."""
from pathlib import Path

import pandas as pd
import pytest

OVERRIDES = {
    "runtime": {"device": "cpu"},
    "audio": {"enable_pitch": False, "enable_wavelet": False, "checkpoint_every": 100},
    "gpu": {"benchmark_batch_sizes": [8], "benchmark_clips": 16, "parity_clips": 4},
    "validation": {"n_splits": 3, "n_repeats": 1, "bootstrap": 30, "schemes": ["repeated_kfold"]},
    "analysis": {"bootstrap": 30, "permutations": 30, "cluster_bootstrap": 3, "perturbation_clips": 12, "discovery_repeats": 2},
    "features": {"crossfit_folds": 3},
    "models": {"benchmark_models": ["RIDGE"], "benchmark_families": [["SCREENING"]], "ablation_max_evals": 2},
    "v2": {
        "validation": {"schemes": ["repeated_kfold", "sample_rate_transfer", "sample_rate_equal_n"], "inner_splits": 2,
                       "equal_n_repeats": 1, "max_repeats": None},
        "evaluation": {"student_schemes": ["repeated_kfold", "sample_rate_transfer"], "bootstrap": 30, "max_evals": 60,
                       "default_budget": 5},
        "harmonization": {"active_variants": ["orig", "bw4k"], "perturbation_clips": 12, "perturbations": ["noise_20db", "gain_minus_12db"],
                          "student_budget": 5, "parity_clips": 2},
        "correctness": {"feature_sets": ["META", "TEMPORAL", "HARMONIZED", "COMBINED"], "models": ["LOGISTIC"],
                        "crossfit_model": "LOGISTIC", "crossfit_feature_sets": ["META", "COMBINED"],
                        "crossfit_schemes": ["repeated_kfold"], "bootstrap": 30, "clip_cv_splits": 3},
        "reference": {"exemplars_per_item": 2, "min_correct_per_item": 2, "inner_splits": 2, "max_frames": 120},
        "multimodal": {"budgets": [5], "models": ["RIDGE"], "fusions": ["early", "residual", "late", "task_pool"],
                       "pairwise": True, "leave_one_out": False},
        "failure": {"bootstrap": 30},
        "phonetic": {"enabled": False},
        "foundation": {"enabled": False},
    },
}


@pytest.fixture(scope="module")
def v2_project(tmp_path_factory) -> Path:
    from conftest import make_project

    return make_project(tmp_path_factory.mktemp("wsv2"), n_train=36, n_test=9)


@pytest.mark.integration
def test_v2_pipeline_end_to_end(v2_project):
    from wildspeech.context import build_context
    from wildspeech.orchestrator import Pipeline

    ctx = build_context("V2_SMOKE", overrides=OVERRIDES, root=v2_project)
    pipe = Pipeline(ctx)
    pre = pipe.run(["P00", "P01", "P02"])
    assert set(pre["status"]) <= {"completed", "cached"}, pre
    res = pipe.run(["P16", "P17", "P19", "P21", "P22"])
    failed = res[res["status"] == "failed"]
    for sid in failed["stage"]:
        print(sid, pipe.status(sid).get("traceback"))
    assert failed.empty, res
    st = res.set_index("stage")["status"]
    for sid in ("P16", "P17", "P19", "P21", "P22"):
        assert st[sid] == "completed", (sid, st[sid])
    v2 = v2_project / "outputs" / "V2"
    for rel in ("P16_HARMONIZATION/tables/variant_summary.csv", "P16_HARMONIZATION/selection.json",
                "P17_CORRECTNESS/tables/clip_metrics.csv", "P17_CORRECTNESS/selection.json",
                "P19_REFERENCE/tables/reference_clip_auc.csv", "P21_MULTIMODAL/tables/model_comparison_p21.csv",
                "P21_MULTIMODAL/tables/family_value.csv", "P22_FAILURE_RECOVERY/tables/deployability.csv",
                "P22_FAILURE_RECOVERY/tables/candidates.csv", "validation_registry.parquet", "experiment_registry.csv",
                "model_comparison.csv", "ablation_results.csv", "hypothesis_registry.csv", "figure_index.csv",
                "scientific_report.html"):
        assert (v2 / rel).exists(), rel
    vs = pd.read_csv(v2 / "P16_HARMONIZATION/tables/variant_summary.csv")
    assert set(vs["variant"]) == {"orig", "bw4k"}
    subs = pd.read_csv(v2 / "P22_FAILURE_RECOVERY/tables/submissions.csv")
    assert subs["status"].astype(str).str.contains("disabled").all()        # no test inference by default
    assert not (v2_project / "submissions" / "V2").exists() or not list((v2_project / "submissions" / "V2").glob("*.csv"))
    # V1 stage statuses untouched by the V2 run (no recomputation of P00-P02)
    again = pipe.run(["P00", "P01", "P02"])
    assert set(again["status"]) == {"cached"}
    # rerun of the V2 line is fully cached (fingerprints unchanged)
    rerun = pipe.run(["P16", "P17", "P19", "P21", "P22"]).set_index("stage")["status"]
    assert all(rerun[s] == "cached" for s in ("P16", "P17", "P19", "P21", "P22"))
