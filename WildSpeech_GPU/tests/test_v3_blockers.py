"""Regression tests for the V3 ASR blockers found in the executed master notebook (2026-11-21):

* empty result CSVs crashed the notebook (pandas EmptyDataError)            -> NO_RESULTS, never a crash
* stale NOT_EXECUTED markers / tables shown as if they were current         -> table_state STALE, archived markers
* review workflow: deferred decisions, board, commands, Pasketti never auto-approved
* per-model download authorization
* cached snapshots split across revisions                                   -> inventory + sha256-blob offline repair
* technical readiness separate from legal readiness; smoke output validation
* empty cache dirs reported as rejected                                     -> never_run
* P23 protected from implicit re-runs
* P29 result classes: genuine negative vs baseline fallback
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildspeech.asr.model_registry import load_registry_document, parse_model
from wildspeech.asr.reviews import (ReviewEntry, assess_reviews, load_proposals, model_decisions, record_deferral,
                                    review_board, validate_entry)
from wildspeech.config import Config
from wildspeech.io_utils import NO_RESULTS, atomic_write_csv, is_no_results, read_csv_safe, read_table

REPO = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER = "b" * 40


# =========================================================================================== empty CSVs
def test_empty_dataframe_round_trips_as_no_results(tmp_path):
    p = atomic_write_csv(pd.DataFrame(), tmp_path / "t.csv")
    assert p.read_text(encoding="utf-8").strip() != ""          # never a header-less file again
    df = read_csv_safe(p)
    assert df.empty and df.attrs["status"] == NO_RESULTS and list(df.columns) == []


@pytest.mark.parametrize("content", ["", "\n", "\r\n"])
def test_legacy_empty_files_do_not_raise(tmp_path, content):
    p = tmp_path / "legacy.csv"
    p.write_text(content, encoding="utf-8")
    df = read_csv_safe(p)
    assert df.empty and is_no_results(df)
    assert is_no_results(read_table(tmp_path / "legacy"))


def test_header_only_is_no_results_but_keeps_columns(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a,b\n", encoding="utf-8")
    df = read_csv_safe(p)
    assert list(df.columns) == ["a", "b"] and df.attrs["status"] == NO_RESULTS


def test_non_empty_table_unchanged(tmp_path):
    p = atomic_write_csv(pd.DataFrame({"x": [1, 2]}), tmp_path / "ok.csv")
    df = read_csv_safe(p)
    assert df["x"].tolist() == [1, 2] and df.attrs.get("status") is None


def test_read_stage_table_never_raises_on_empty_file(ctx):
    d = ctx.stage_dir("P25") / "tables"
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature_tables.csv").write_text("\n", encoding="utf-8")           # what the old P25 wrote
    df = ctx.read_stage_table("P25", "feature_tables")
    assert df is not None and df.empty and df.attrs["status"] == NO_RESULTS


def test_notebook_table_reports_no_results(ctx, capsys):
    from wildspeech.v3 import notebook as nb

    d = ctx.stage_dir("P25") / "tables"
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature_tables.csv").write_text("", encoding="utf-8")
    assert nb.table(ctx, "P25", "feature_tables", n=None, allow_stale=True) is None
    assert "NO_RESULTS" in capsys.readouterr().out


# =========================================================================================== stale tables
def test_table_written_before_last_run_is_stale(ctx):
    from wildspeech.orchestrator import Pipeline
    from wildspeech.v3.readiness import table_state

    p = ctx.save_table(pd.DataFrame({"a": [1]}), "P26", "old_result")
    old = time.time() - 3600
    os.utime(p, (old, old))
    Pipeline(ctx)._write("P26", "not_executed", None, 5.0, note="no inputs")
    state, _ = table_state(ctx, "P26", "old_result")
    assert state == "STALE"
    ctx.save_table(pd.DataFrame({"a": [2]}), "P26", "new_result")
    assert table_state(ctx, "P26", "new_result")[0] == "CURRENT"
    assert table_state(ctx, "P26", "never_written")[0] == "MISSING"


def test_previous_not_executed_marker_is_archived_not_deleted(ctx):
    from wildspeech.v3.common import archive_previous_outcome, not_executed

    not_executed(ctx, "P27", "needs >= 2 recognisers")
    d = ctx.stage_dir("P27")
    assert (d / "tables" / "not_executed.csv").exists() and (d / "_outcome.json").exists()
    archive_previous_outcome(ctx, "P27")
    assert not (d / "tables" / "not_executed.csv").exists()
    assert list((d / "tables" / "_history").glob("not_executed_*.csv"))
    assert list((d / "tables" / "_history").glob("outcome_*.json"))


# ============================================================================================= reviews
def _spec(tmp_path, **over):
    m = {"family": "wav2vec2", "role": "general_char", "adapter": "ctc", "output": "chars", "hf_id": "org/model",
         "hf_id_status": "verified_public", "revision": SHA, "expected_architectures": ["hf_ctc"],
         "files": [{"name": "config.json", "size": 2, "role": "config"},
                   {"name": "model.safetensors", "size": 5, "role": "weights",
                    "sha256": hashlib.sha256(b"WEIGH").hexdigest()}],
         "evidence": {"model_card_url": "https://example.org/card", "license_id": "apache-2.0",
                      "license_url": "https://example.org/l", "training_sources": ["LibriSpeech"],
                      "training_evidence_url": "https://example.org/p", "enumerated_corpora": True},
         "restrictions": []}
    m.update(over)
    return parse_model("m", m, tmp_path, "models")


def _entry(**kw):
    base = dict(model_key="m", checks=["overlap"], decision="deferred", reviewer="Test Reviewer", date="2026-11-01",
                revision_reviewed=SHA, outstanding=["Awaiting an organiser ruling on corpus overlap"])
    base.update(kw)
    return ReviewEntry(**base)


def test_deferred_review_is_valid_and_keeps_check_pending(tmp_path):
    s = _spec(tmp_path)
    e = _entry()
    assert validate_entry(e, s, ["apache-2.0"]) == []
    rv, invalid = assess_reviews(s, [e], ["apache-2.0"], SHA)
    assert not invalid and rv["overlap"].status == "DEFERRED" and "organiser ruling" in rv["overlap"].detail


def test_deferred_needs_an_outstanding_question_and_cannot_resolve(tmp_path):
    s = _spec(tmp_path, restrictions=[{"id": "R1", "check": "overlap", "text": "x"}])
    assert any("outstanding" in p for p in validate_entry(_entry(outstanding=[]), s, ["apache-2.0"]))
    assert any("cannot resolve" in p for p in validate_entry(_entry(resolves=["R1"]), s, ["apache-2.0"]))


def test_later_approval_supersedes_deferral(tmp_path):
    s = _spec(tmp_path)
    ok = ReviewEntry(model_key="m", checks=["overlap"], decision="approved", reviewer="Test Reviewer",
                     date="2026-11-10", revision_reviewed=SHA, overlap_basis="enumerated_public_corpora_disjoint",
                     evidence=[{"url": "https://example.org/card", "summary": "corpora enumerated, disjoint"}])
    rv, _ = assess_reviews(s, [_entry(), ok], ["apache-2.0"], SHA)
    assert rv["overlap"].status == "APPROVED"


def test_deferred_review_never_makes_a_model_eligible(tmp_path):
    from wildspeech.asr.eligibility import EligibilityChecker
    from wildspeech.asr.model_registry import CompetitionPolicy, LicensePolicy

    s = _spec(tmp_path)
    ch = EligibilityChecker(Config({"runtime": {}, "paths": {}, "v3": {"asr": {}}}), ["apache-2.0"],
                            CompetitionPolicy(statement="ok", recorded_by="t", source_url="https://example.org",
                                              permitted_families=["wav2vec2"], named_families=["wav2vec2"]),
                            [_entry()], license_policy=LicensePolicy(permissive=["apache-2.0"]), env={})
    r = ch.check(s, probe_files=False)
    assert r.checks["overlap_review"] == "PENDING" and not r.download_allowed and not r.eligible


def test_record_deferral_writes_validated_entry(tmp_path):
    s = _spec(tmp_path)
    f = tmp_path / "reviews.yaml"
    assert record_deferral(f, s, ["overlap"], "Test Reviewer", SHA, ["Awaiting the organiser ruling on overlap"],
                           allowed_licenses=["apache-2.0"]) == []
    assert "deferred" in f.read_text(encoding="utf-8")
    assert record_deferral(f, s, ["overlap"], "claude", SHA, ["Awaiting the organiser ruling"],
                           allowed_licenses=["apache-2.0"])          # placeholder reviewer refused


def test_repo_board_never_approves_pasketti_and_lists_commands():
    doc = load_registry_document(REPO / "configs")
    props = load_proposals(REPO / "configs" / "v3_asr_review_proposals.yaml")
    b = review_board(doc.specs, [], props, doc.allowed_licenses, doc.license_policy, REPO)
    assert set(b["model_key"]) == set(doc.specs)
    psk = b[(b["model_key"].str.startswith("pasketti")) & (b["check"] == "overlap")]
    assert (psk["status"] != "APPROVED").all() and psk["decision_needed"].all()
    assert (psk["proposals_ready"] == "").all()                 # the overlap drafts are ready: false
    assert psk["cmd_accept_after_deciding"].str.contains("--mitigation").all()
    assert psk["cmd_defer"].str.contains("defer").all() and psk["cmd_reject"].str.contains("rejected").all()
    md = model_decisions(b)
    assert not md.loc[md["model_key"] == "pasketti_phonetic", "reviews_complete"].iloc[0]
    w = b[(b["model_key"] == "wav2vec2_base_960h") & (b["check"] == "license")]
    assert w["cmd_accept"].str.contains("P-B960-LP").all()


# ========================================================================================== downloads
def test_per_model_download_authorization(monkeypatch):
    from wildspeech.asr.model_manager import downloads_allowed

    monkeypatch.delenv("ALLOW_MODEL_DOWNLOADS", raising=False)
    monkeypatch.delenv("WILDSPEECH_ALLOW_MODEL_DOWNLOADS", raising=False)
    cfg = Config({"v3": {"asr": {"downloads": {"allow_model_downloads": True, "authorized_models": ["a"]}}}})
    assert downloads_allowed(cfg, None, "a")[0] is True
    ok, why = downloads_allowed(cfg, None, "b")
    assert ok is False and "authorized_models" in why
    assert downloads_allowed(Config({"v3": {"asr": {"downloads": {}}}}), None, "a")[0] is False
    assert downloads_allowed(cfg, False, "a")[0] is False           # explicit dry run wins


# ===================================================================================== cross-revision
def _hub(tmp_path, weights=b"WEIGH"):
    hub = tmp_path / "hub"
    base = hub / "models--org--model"
    sha = hashlib.sha256(weights).hexdigest()
    (base / "blobs").mkdir(parents=True)
    (base / "blobs" / sha).write_bytes(weights)
    pinned = base / "snapshots" / SHA
    pinned.mkdir(parents=True)
    (pinned / "config.json").write_text("{}", encoding="utf-8")
    other = base / "snapshots" / OTHER
    other.mkdir(parents=True)
    (other / "model.safetensors").write_bytes(weights)
    (other / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (base / "refs" / "pr").mkdir(parents=True)
    (base / "refs" / "main").write_text(SHA, encoding="utf-8")
    (base / "refs" / "pr" / "11").write_text(OTHER, encoding="utf-8")
    return hub, pinned


def test_revision_inventory_shows_files_split_across_revisions(tmp_path):
    from wildspeech.asr.snapshots import revision_inventory

    hub, _ = _hub(tmp_path)
    t = pd.DataFrame(revision_inventory(_spec(tmp_path), hub))
    w = t[t["file"] == "model.safetensors"].set_index("snapshot")
    assert bool(w.loc[OTHER, "present"]) and not bool(w.loc[SHA, "present"])
    assert w.loc[OTHER, "refs"] == "pr/11" and bool(w.loc[SHA, "is_pinned"])


def test_offline_repair_links_only_sha256_verified_blobs(tmp_path):
    from wildspeech.asr.model_manager import repair_from_blobs
    from wildspeech.asr.snapshots import manifest_problems, manifest_status

    hub, pinned = _hub(tmp_path)
    s = _spec(tmp_path)
    rows = repair_from_blobs(s, hub, {}, verify=True)
    done = {r["file"]: r for r in rows}
    assert done["model.safetensors"]["ok"] and done["model.safetensors"]["action"].startswith("linked_")
    assert (pinned / "model.safetensors").read_bytes() == b"WEIGH"
    assert not manifest_problems(manifest_status(pinned, s))
    assert (hub / "models--org--model" / "snapshots" / OTHER / "model.safetensors").exists()   # source untouched


def test_small_git_files_are_never_borrowed_from_another_revision(tmp_path):
    from wildspeech.asr.model_manager import repair_from_blobs

    hub, pinned = _hub(tmp_path)
    s = _spec(tmp_path, files=[{"name": "config.json", "size": 2, "role": "config"},
                               {"name": "preprocessor_config.json", "size": 2, "role": "processor"},
                               {"name": "model.safetensors", "size": 5, "role": "weights",
                                "sha256": hashlib.sha256(b"WEIGH").hexdigest()}])
    rows = {r["file"]: r for r in repair_from_blobs(s, hub, {}, verify=True)}
    assert rows["preprocessor_config.json"]["action"] == "needs_download"
    assert not (pinned / "preprocessor_config.json").exists()


def test_corrupt_blob_is_refused(tmp_path):
    from wildspeech.asr.model_manager import repair_from_blobs

    hub, pinned = _hub(tmp_path)
    good = hashlib.sha256(b"WEIGH").hexdigest()
    (hub / "models--org--model" / "blobs" / good).write_bytes(b"EVIL!")      # same size, wrong content
    rows = {r["file"]: r for r in repair_from_blobs(_spec(tmp_path), hub, {}, verify=True)}
    assert rows["model.safetensors"]["action"] == "blob_hash_mismatch"
    assert not (pinned / "model.safetensors").exists()


# =============================================================================== technical vs legal
def test_technical_readiness_ignores_reviews_and_build_adapter_still_gates_inference():
    from wildspeech.asr.eligibility import EligibilityRecord
    from wildspeech.asr.inference import TECHNICAL_DIMENSIONS, build_adapter, technically_ready

    rec = EligibilityRecord("m", "NOT_EXECUTED", dimensions={d: "PASS" for d in TECHNICAL_DIMENSIONS} |
                            {"license_approval": "PENDING"})
    assert technically_ready(rec)[0] is True
    with pytest.raises(PermissionError):
        build_adapter(_FakeSpec(), rec, Config({}), "cpu")                      # inference needs eligibility
    rec.dimensions["checkpoint_files"] = "FAIL"
    ok, bad = technically_ready(rec)
    assert not ok and "checkpoint_files=FAIL" in bad
    with pytest.raises(PermissionError):
        build_adapter(_FakeSpec(), rec, Config({}), "cpu", technical_only=True)


class _FakeSpec:
    key = "m"
    adapter = "ctc"
    output = "chars"


def test_synthetic_probe_audio_is_deterministic_and_finite():
    from wildspeech.asr.inference import synthetic_probe_audio

    a, b = synthetic_probe_audio(), synthetic_probe_audio()
    assert len(a) == 2 and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert all(np.isfinite(x).all() and x.dtype == np.float32 for x in a)


def test_dependency_hints():
    from wildspeech.asr.inference import dependency_hint

    assert "safetensors" in dependency_hint(ValueError("Due to a serious vulnerability issue in torch.load ... CVE-2025-32434"))
    assert "preprocessor_config" in dependency_hint(OSError("Can't load feature extractor for 'x'"))
    assert "WSL2" in dependency_hint(ModuleNotFoundError("No module named 'qwen_asr'"))


@pytest.mark.parametrize("rows,output,n_emb,fails,bad", [
    (pd.DataFrame(), "words", 0, 0, "no rows"),
    (pd.DataFrame({"transcript": ["", ""], "error": ["", ""]}), "words", 0, 0, "empty"),
    (pd.DataFrame({"transcript": ["cat", ""], "error": ["", ""]}), "words", 0, 0, None),
    (pd.DataFrame({"phones": ["", ""], "ipa": ["kæt", ""], "error": ["", ""]}), "phones", 0, 0, None),
    (pd.DataFrame({"transcript": ["cat"], "has_token_probs": [True], "mean_token_logprob": [np.nan], "error": [""]}),
     "words", 0, 0, "non-finite"),
    (pd.DataFrame({"transcript": [""], "error": [""]}), "embeddings", 0, 0, "embeddings"),
    (pd.DataFrame({"transcript": [""], "error": [""]}), "embeddings", 3, 0, None),
    (pd.DataFrame({"transcript": ["cat"], "error": [""]}), "words", 0, 2, "failed"),
])
def test_smoke_output_validation(rows, output, n_emb, fails, bad):
    from wildspeech.asr.inference import validate_smoke_rows

    p = validate_smoke_rows(rows, output, n_emb, fails)
    assert (not p) if bad is None else any(bad in x for x in p)


def test_oom_halves_micro_batch_and_recovers():
    from wildspeech.asr.inference import AdaptiveMicroBatcher

    calls = []

    def fn(w, n):
        calls.append(len(w))
        if len(w) > 2:
            raise RuntimeError("CUDA out of memory")
        return [x * 2 for x in w]

    mb = AdaptiveMicroBatcher(8, grow_after=100, is_oom=lambda e: "out of memory" in str(e), release=lambda: None)
    assert mb.run(fn, list(range(7)), list(range(7))) == [x * 2 for x in range(7)]
    assert mb.shrinks >= 2 and mb.min_seen <= 2 and max(calls[2:]) <= 2


# ================================================================================ outputs / schemas
def test_empty_cache_dir_is_never_run_not_rejected(ctx):
    from wildspeech.asr.outputs import execution_state, require_outputs

    (Path(ctx.paths.v3_cache) / "asr" / "ghost_model" / "train").mkdir(parents=True, exist_ok=True)
    st = execution_state(ctx, ["ghost_model"], ("train",))
    assert st.iloc[0]["state"] == "never_run" and not st.iloc[0]["accepted"]
    ok, acc, why = require_outputs(ctx, "P25", "text", 1, ("train",))
    assert not ok and acc == [] and "ghost_model" in why


def test_asr_result_row_schema_never_fabricates_probabilities():
    from wildspeech.asr.base import ASR_COLUMNS, ASRResult

    r = ASRResult(transcript="cat", raw_tokens="c a t").row()
    assert set(ASR_COLUMNS) <= set(r)
    assert r["has_token_probs"] is False and np.isnan(r["mean_token_logprob"]) and r["token_logprobs"] == ""
    r2 = ASRResult(transcript="cat", token_logprobs=[-0.1, float("nan"), -0.3]).row()
    assert r2["has_token_probs"] and r2["mean_token_logprob"] == pytest.approx(-0.2)


def test_ctc_ipa_character_merge():
    from wildspeech.asr.ctc_adapter import ipa_sequence_to_arpa

    out = ipa_sequence_to_arpa(["t", "ʃ", "|", "k", "æ", "t"], merge=True)
    assert out[0] == "CH" and out[-3:] == ["K", "AE", "T"]


# ============================================================================== orchestration / P23
def test_p23_is_protected_from_implicit_reruns(ctx):
    from wildspeech.orchestrator import Pipeline, protected_stages
    from wildspeech.v3 import notebook as nb

    assert "P23" in protected_stages(ctx.cfg)
    pipe = Pipeline(ctx)
    pipe._write("P23", "completed", "an-older-fingerprint", 1318.0)
    plan = pipe.run(["P23"], dry_run=True)
    assert plan.set_index("stage").loc["P23", "status"] == "protected"
    assert nb.force_list(True, ["P23", "P24"]) == ["P23", "P24"]
    nbplan = nb.run_stages(ctx, ["P23"], True, enabled=False)          # FORCE=True excludes protected stages
    assert nbplan.set_index("stage").loc["P23", "status"] == "protected"
    forced = pipe.run(["P23"], force=["P23"], dry_run=True)
    assert forced.set_index("stage").loc["P23", "status"] == "would_run"


def test_protection_key_is_outside_every_stage_fingerprint(ctx):
    from wildspeech.orchestrator import STAGES

    for sid, spec in STAGES.items():
        assert not any(k == "v3" or k.startswith("v3.orchestration") for k in spec.keys), sid


def test_plan_rows_are_annotated(ctx):
    from wildspeech.v3.readiness import explain_plan

    t = explain_plan(pd.DataFrame([{"stage": "P25", "status": "would_run"}, {"stage": "P26", "status": "not_executed"}]))
    assert t["meaning"].str.contains("PLANNED").iloc[0] and "NOT_EXECUTED" in t["meaning"].iloc[1]


# ================================================================================ P29 result classes
def test_result_classes_separate_fallback_from_genuine_negative():
    from wildspeech.evaluation.ablation import classify_results

    rows = pd.DataFrame([
        {"name": "B", "group": "baseline", "status": "done"},
        {"name": "FB", "group": "family", "status": "done", "fallback_folds": 3, "delta_r2": 0.0,
         "delta_ci_low": 0.0, "delta_ci_high": 0.0, "q": 1.0},
        {"name": "NEG", "group": "family", "status": "done", "fallback_folds": 0, "delta_r2": -0.01,
         "delta_ci_low": -0.02, "delta_ci_high": -0.004, "q": 0.01},
        {"name": "GAIN", "group": "family", "status": "done", "fallback_folds": 0, "delta_r2": 0.01,
         "delta_ci_low": 0.002, "delta_ci_high": 0.02, "q": 0.01},
        {"name": "NULL", "group": "family", "status": "done", "fallback_folds": 0, "delta_r2": 0.001,
         "delta_ci_low": -0.004, "delta_ci_high": 0.006, "q": 0.6},
        {"name": "FAIL", "group": "family", "status": "failed"},
        {"name": "NOFEAT", "group": "family", "status": "no_added_features", "fallback_folds": 0},
    ])
    c = classify_results(rows).set_index("name")["result_class"]
    assert c.to_dict() == {"B": "BASELINE", "FB": "BASELINE_FALLBACK", "NEG": "GENUINE_NEGATIVE",
                           "GAIN": "GENUINE_GAIN", "NULL": "NO_DETECTABLE_EFFECT", "FAIL": "NOT_REPORTABLE",
                           "NOFEAT": "BASELINE_FALLBACK"}


def test_baseline_reference_check_against_p10_gam(ctx):
    from wildspeech.stages.p29_reading_ablation import REFERENCE_GAM, baseline_reference_check

    ctx.save_table(pd.DataFrame([{"name": REFERENCE_GAM, "r2_mean": 0.215256}]), "P10", "model_benchmark")
    ok = baseline_reference_check(ctx, pd.DataFrame([{"scheme": "repeated_kfold", "r2_mean": 0.2149}]), "repeated_kfold")
    assert ok.iloc[0]["status"] == "MATCH"
    bad = baseline_reference_check(ctx, pd.DataFrame([{"scheme": "repeated_kfold", "r2_mean": 0.15}]), "repeated_kfold")
    assert bad.iloc[0]["status"] == "MISMATCH"


# ============================================================================== notebook contract
def test_repaired_notebook_is_clean_and_has_safe_controls():
    p = REPO / "notebooks" / "WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER_REPAIRED.ipynb"
    nbj = json.loads(p.read_text(encoding="utf-8"))
    code = [c for c in nbj["cells"] if c["cell_type"] == "code"]
    assert all(c["execution_count"] is None and c["outputs"] == [] for c in code)
    src = "".join("".join(c["source"]) for c in code)
    for flag in ("RUN_PIPELINE = False", "ALLOW_MODEL_DOWNLOADS = False", "RUN_SMOKE = False", "FORCE = False"):
        assert flag in src
    assert "pd.read_csv" not in src                                   # all tables go through nb.table / nb.tables
    for fn in ("review_board", "load_checks", "revision_inventory", "repair_plan", "model_readiness",
               "stage_readiness", "p23_results"):
        assert f"nb.{fn}(" in src
