"""V3 integration on a fresh synthetic project (CPU).

1. configuration / orchestration: V3 profiles load, P23-P29 are registered with outputs/V3 dirs, V3 never re-plans V2
2. end-to-end: P00-P02 + V2 P16/P17/P21, then P23 (audit), P24 (no eligible model -> stage status 'not_executed',
   never 'completed'), P25 without outputs (not_executed) and P26 (blocked); then synthetic ASR outputs written through
   the real checkpoint store (TEST FIXTURE ONLY, provenance fixture=true, accepted only because the test sets
   v3.asr.accept_fixture_outputs) and P25-P29 on them. Verifies every artefact path, registry, report and caching.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildspeech import FILE

BASE = {
    "runtime": {"device": "cpu"},
    "audio": {"enable_pitch": False, "enable_wavelet": False, "checkpoint_every": 100},
    "gpu": {"benchmark_batch_sizes": [8], "benchmark_clips": 16, "parity_clips": 4},
    "validation": {"n_splits": 3, "n_repeats": 1, "bootstrap": 30, "schemes": ["repeated_kfold"]},
    "analysis": {"bootstrap": 30, "permutations": 30, "cluster_bootstrap": 3, "perturbation_clips": 12, "discovery_repeats": 2},
    "features": {"crossfit_folds": 3},
    "models": {"benchmark_models": ["RIDGE"], "benchmark_families": [["SCREENING"]], "ablation_max_evals": 2},
    "v2": {"validation": {"schemes": ["repeated_kfold", "school_kfold", "sample_rate_transfer"], "inner_splits": 2},
           "evaluation": {"student_schemes": ["repeated_kfold"], "bootstrap": 30, "max_evals": 30, "default_budget": 5},
           "harmonization": {"active_variants": ["orig"], "perturbation_clips": 8, "perturbations": ["gain_minus_12db"],
                             "student_budget": 5, "parity_clips": 2},
           "correctness": {"feature_sets": ["META", "TEMPORAL", "COMBINED"], "models": ["LOGISTIC"], "crossfit_model": "LOGISTIC",
                           "crossfit_feature_sets": ["META", "COMBINED"], "crossfit_schemes": ["repeated_kfold"],
                           "bootstrap": 30, "clip_cv_splits": 3},
           "reference": {"enabled": False},
           "multimodal": {"budgets": [5], "models": ["RIDGE"], "fusions": ["early", "residual"], "pairwise": False,
                          "leave_one_out": False, "indices": {"enabled": False}},
           "phonetic": {"enabled": False}, "foundation": {"enabled": False}},
    "v3": {"audit": {"reaudit_specs": ["P21_SINGLE_SOFT"], "bootstrap": 30, "task_ensemble_audit": True},
           "asr": {"enabled": True, "models": ["fake_child_phon", "fake_general_word"], "accept_fixture_outputs": True},
           "phonetics": {"allow_approximate_g2p": True, "manual_review_clips": 10},
           "discovery": {"n_perm": 30, "bootstrap": 30, "clip_cv_splits": 3, "min_correct_clips_per_student": 1, "top_k_confirm": 5},
           "representations": {"budgets": [5], "methods": ["pca", "select"], "mil": {"enabled": False},
                               "nested_soft": {"inner_splits": 2}},
           "ablation": {"schemes": ["repeated_kfold", "school_kfold", "sample_rate_transfer"], "models": ["RIDGE"],
                        "fusions": ["early", "residual"], "bootstrap": 30, "max_evals": 80, "default_budget": 5,
                        "negative_controls": ["permuted_features", "shuffled_outcome"]}},
}
FAKE_REGISTRY = """allowed_licenses: [apache-2.0]
models:
  fake_child_phon: {hf_id: none/fake-phon, role: child_phonetic, adapter: pasketti_phonetic, output: phones, child_speech: true}
  fake_general_word: {hf_id: none/fake-word, role: general_word, adapter: whisper, output: words, child_speech: false}
"""


def test_v3_profiles_and_stage_registration(tmp_path):
    from conftest import make_project
    from wildspeech.context import build_context
    from wildspeech.orchestrator import STAGES, V2_STAGES, V3_STAGES, Pipeline

    root = make_project(tmp_path / "p", n_train=12, n_test=3)
    for prof in ("V3_SMOKE", "V3_AUDIT", "V3_ASR", "V3_SCIENCE", "V3_FULL", "V3_SUBMISSION"):
        ctx = build_context(prof, overrides={"runtime": {"device": "cpu"}}, root=root)
        assert ctx.cfg.get_path("v3.project.version") == "3.0.0"
        assert ctx.cfg.get_path("v2.phonetic.enabled") is True            # V2_FULL base_profile layered
    assert set(V3_STAGES) == {"P23", "P24", "P25", "P26", "P27", "P28", "P29"} and set(V3_STAGES) <= set(STAGES)
    assert ctx.stage_dir("P23").parts[-2:] == ("V3", "P23_FUSION_AUDIT")
    for s in V3_STAGES.values():
        assert not (set(s.deps) & set(V2_STAGES)), "V3 must never hard-depend on V2 stages"
    plan = Pipeline(ctx).resolve(["P29"])
    assert not (set(plan) & set(V2_STAGES))                                 # V2 is never re-planned by V3


def _fake_asr(ctx, rng):
    """Synthetic outputs with the real P24 schema (test fixture standing in for eligible models)."""
    from wildspeech.asr.base import ASRResult
    from wildspeech.asr.checkpointing import ASRCheckpointStore, asr_fingerprint
    from wildspeech.phonetics.g2p import ExpectedPhones

    exp = ExpectedPhones("auto", allow_approx=True)
    clips = ctx.bundle().train_clips.drop_duplicates(FILE)
    for key, out in (("fake_child_phon", "phones"), ("fake_general_word", "words")):
        st = ASRCheckpointStore(ctx.paths.v3_cache, key)
        st.prepare("train", asr_fingerprint(key, "fixture", None, {}), {"model_key": key, "fixture": True})
        rows = []
        for r in clips.to_dict("records"):
            ok = r["item_score"] == 1
            ph = exp.phonemes(r["expected_text"])
            if not ok and ph:
                ph = ph[1:] + ["S"]
            q = 0.85 if ok else 0.45
            spans, t = [], 5
            for p in exp.phonemes(r["expected_text"]):
                d = int(rng.integers(3, 8))
                spans.append({"p": p, "s": t, "e": t + d, "lp": float(np.log(q)), "gop": float(np.log(q) - np.log(0.9)), "post": q})
                t += d + int(rng.integers(0, 7))
            words = str(r["expected_text"]).lower().split()
            res = ASRResult(transcript="" if out == "phones" else " ".join(words if ok else words[1:] + ["um"]),
                            phones=" ".join(ph) if out == "phones" else "", raw_tokens=" ".join(ph),
                            token_logprobs=list(np.log(np.clip(q + 0.05 * rng.standard_normal(max(len(words), 1)), 0.01, 0.99))),
                            frame_entropy_mean=float(1 - q) if out == "phones" else np.nan,
                            blank_frac=0.4 if out == "phones" else np.nan, greedy_ll=-0.2 if out == "phones" else np.nan,
                            expected_ll=float(np.log(q)), expected_alignment=spans if out == "phones" else None,
                            audio_seconds=float(r["duration_sec"]), embedding=rng.standard_normal(16).astype(np.float32))
            row = res.row()
            row["n_frames"] = t + 5
            row[FILE] = r[FILE]
            row["_embedding"] = res.embedding
            rows.append(row)
        st.append("train", rows, f"{key}@fixture")


@pytest.mark.integration
def test_v3_pipeline_end_to_end(tmp_path_factory):
    from conftest import make_project
    from wildspeech.context import build_context
    from wildspeech.orchestrator import Pipeline

    root = make_project(tmp_path_factory.mktemp("wsv3"), n_train=36, n_test=9)
    (root / "configs" / "v3_asr_models.yaml").write_text(FAKE_REGISTRY, encoding="utf-8")
    ctx = build_context("V3_SMOKE", overrides=BASE, root=root)
    pipe = Pipeline(ctx)
    pre = pipe.run(["P00", "P01", "P02", "P10", "P16", "P17", "P21"])
    assert not (pre["status"] == "failed").any(), pre
    res = pipe.run(["P23", "P24"])
    for sid in res.loc[res["status"] == "failed", "stage"]:
        print(sid, pipe.status(sid).get("traceback"))
    st = res.set_index("stage")["status"]
    assert st["P23"] == "completed", res
    assert st["P24"] == "not_executed", res                     # eligibility-only P24 is never 'completed'
    assert pipe.status("P24")["status"] == "not_executed"
    assert pipe.status("P24")["summary"]["stage_outcome"] == "ELIGIBILITY_ONLY"
    assert pipe.status("P24")["summary"]["inference_executed"] is False
    v3 = root / "outputs" / "V3"
    elig = pd.read_csv(v3 / "P24_ASR" / "tables" / "model_eligibility.csv")
    assert set(elig["eligibility"]) == {"NOT_EXECUTED"} and elig["reasons"].str.len().gt(0).all()
    assert (v3 / "P24_ASR" / "tables" / "not_executed.csv").exists()
    for t in ("eligibility_checks", "required_actions", "framework_environment", "execution_state_before"):
        assert (v3 / "P24_ASR" / "tables" / f"{t}.csv").exists(), t
    # without real ASR output, P25 is NOT_EXECUTED and P26 (hard dependant) is blocked — nothing looks like a result
    pre25 = pipe.run(["P25", "P26"]).set_index("stage")["status"]
    assert pre25["P25"] == "not_executed" and pre25["P26"] == "blocked", pre25
    for t in ("fusion_audit", "fusion_verdicts", "baseline_reproduction", "lineage_audit", "v3_repairs", "task_ensemble_summary"):
        assert (v3 / "P23_FUSION_AUDIT" / "tables" / f"{t}.csv").exists(), t
    br = pd.read_csv(v3 / "P23_FUSION_AUDIT" / "tables" / "baseline_reproduction.csv")
    assert bool(br["reproduced"].iloc[0])                                   # V3 evaluator reproduces the V2 baseline
    _fake_asr(ctx, np.random.default_rng(0))
    res2 = pipe.run(["P25", "P26", "P27", "P28", "P29"], force=["P25"])
    for sid in res2.loc[res2["status"] == "failed", "stage"]:
        print(sid, pipe.status(sid).get("traceback"))
    assert (res2["status"] == "completed").all(), res2
    for rel in ("P25_PHONETIC_FEATURES/tables/correctness_validation_auc.csv", "P25_PHONETIC_FEATURES/tables/child_vs_general_auc.csv",
                "P25_PHONETIC_FEATURES/tables/manual_review_sheet.csv", "P26_CORRECT_ONLY/tables/residual_associations.csv",
                "P26_CORRECT_ONLY/tables/clip_correctness_increment.csv", "P26_CORRECT_ONLY/tables/student_incremental_r2.csv",
                "P27_ASR_DISAGREEMENT/tables/variance_decomposition.csv", "P27_ASR_DISAGREEMENT/tables/consensus.csv",
                "P28_REPRESENTATIONS/tables/representation_results.csv", "P29_READING_ABLATION/tables/ablation_results.csv",
                "P29_READING_ABLATION/tables/transfer_robustness.csv", "P29_READING_ABLATION/tables/champion_comparison.csv",
                "model_registry.csv", "phonetic_feature_registry.parquet", "hypothesis_registry.csv",
                "experiment_registry.csv", "ablation_results.csv", "scientific_report.html", "v3_champion.json"):
        assert (v3 / rel).exists(), rel
    ab = pd.read_csv(v3 / "P29_READING_ABLATION" / "tables" / "ablation_results.csv")
    assert {"ASR00", "ASR02", "ASR03"} <= set(ab["experiment"])
    assert ab.loc[ab["group"] != "baseline", "status"].isin(
        ["ok", "fallback_baseline", "partial_fallback", "no_added_features", "unavailable", "failed"]).all()
    rep = ab[ab["status"] == "fallback_baseline"]
    assert (rep["reportable"] == False).all()                               # noqa: E712 - fallbacks never reportable
    champ = json.loads((v3 / "v3_champion.json").read_text())
    assert "champion" in champ and "rule" in champ
    assert not (root / "submissions" / "V3").exists() or not list((root / "submissions" / "V3").glob("*.csv"))
    again = pipe.run(["P23", "P25", "P26", "P27", "P28", "P29"]).set_index("stage")["status"]
    assert all(again[s] == "cached" for s in ("P23", "P25", "P26", "P27", "P28", "P29"))
