"""Manual review records: validated on every run, refused when incomplete, never edit the registry."""
import datetime as dt
import shutil
from pathlib import Path

import pytest
import yaml

from wildspeech.asr.model_registry import load_registry_document, parse_model
from wildspeech.asr.reviews import (ReviewEntry, _cli, assess_reviews, load_reviews, record_review, validate_entry)

REPO = Path(__file__).resolve().parents[1]
SHA = "c" * 40
TODAY = dt.date(2026, 11, 20)
ALLOWED = ["apache-2.0"]


def _spec(tmp_path, restrictions=None, revision=None, license_id="apache-2.0"):
    return parse_model("m", {"family": "whisper", "role": "general_word", "adapter": "whisper", "output": "words",
                             "hf_id": "openai/whisper-small.en", "revision": revision,
                             "evidence": {"license_id": license_id, "training_sources": ["web audio"],
                                          "model_card_url": "https://example.org/card"},
                             "restrictions": restrictions or []}, tmp_path, "models/asr")


def _entry(**kw):
    base = dict(model_key="m", checks=["license", "provenance"], decision="approved", reviewer="A. Reviewer",
                date="2026-11-01", revision_reviewed=SHA, license_id="apache-2.0",
                evidence=[{"url": "https://example.org/card", "summary": "card read: apache-2.0, web audio"}])
    base.update(kw)
    return ReviewEntry(**base)


def test_valid_entry_has_no_problems(tmp_path):
    assert validate_entry(_entry(), _spec(tmp_path), ALLOWED, today=TODAY) == []


@pytest.mark.parametrize("over,needle", [
    ({"reviewer": "todo"}, "placeholder"),
    ({"reviewer": ""}, "placeholder"),
    ({"date": "2099-01-01"}, "future"),
    ({"date": "yesterday"}, "ISO"),
    ({"revision_reviewed": None}, "revision_reviewed missing"),
    ({"decision": "maybe"}, "decision"),
    ({"checks": []}, "no checks"),
    ({"checks": ["vibes"]}, "unknown checks"),
    ({"evidence": []}, "evidence item"),
    ({"evidence": [{"url": "http://insecure.example", "summary": "long enough summary"}]}, "https"),
    ({"evidence": [{"url": "https://example.org", "summary": "short"}]}, "summary"),
    ({"license_id": None}, "license_id"),
    ({"license_id": "mit"}, "!= registry"),
])
def test_invalid_entries(tmp_path, over, needle):
    probs = validate_entry(_entry(**over), _spec(tmp_path), ALLOWED, today=TODAY)
    assert any(needle in p for p in probs), probs


def test_license_not_allowed(tmp_path):
    probs = validate_entry(_entry(license_id="gpl-3.0"), _spec(tmp_path, license_id="gpl-3.0"), ALLOWED, today=TODAY)
    assert any("not in allowed_licenses" in p for p in probs)


def test_registry_without_license_cannot_be_license_approved(tmp_path):
    probs = validate_entry(_entry(), _spec(tmp_path, license_id=None), ALLOWED, today=TODAY)
    assert any("registry evidence has no license_id" in p for p in probs)


def test_pinned_revision_must_match(tmp_path):
    probs = validate_entry(_entry(revision_reviewed="d" * 40), _spec(tmp_path, revision=SHA), ALLOWED, today=TODAY)
    assert any("pinned revision" in p for p in probs)


def test_resolves_rules(tmp_path):
    rs = [{"id": "R1", "check": "overlap", "text": "t"}]
    spec = _spec(tmp_path, rs)
    assert any("unknown restriction" in p for p in validate_entry(_entry(resolves=["NOPE"]), spec, ALLOWED, today=TODAY))
    assert any("does not cover" in p for p in validate_entry(_entry(resolves=["R1"]), spec, ALLOWED, today=TODAY))
    rej = _entry(checks=["overlap"], decision="rejected", license_id=None, resolves=["R1"])
    assert any("only an approved review" in p for p in validate_entry(rej, spec, ALLOWED, today=TODAY))
    ok = _entry(checks=["overlap"], license_id=None, resolves=["R1"])
    assert validate_entry(ok, spec, ALLOWED, today=TODAY) == []


def test_evidence_file_must_exist(tmp_path):
    spec = _spec(tmp_path)
    bad = _entry(evidence=[{"file": "missing.pdf", "summary": "saved model card pdf"}])
    assert validate_entry(bad, spec, ALLOWED, base_dir=tmp_path, today=TODAY)
    (tmp_path / "card.pdf").write_bytes(b"%PDF")
    good = _entry(evidence=[{"file": "card.pdf", "summary": "saved model card pdf"}])
    assert validate_entry(good, spec, ALLOWED, base_dir=tmp_path, today=TODAY) == []


def test_unknown_model(tmp_path):
    assert validate_entry(_entry(model_key="zzz"), None, ALLOWED) == ["unknown model_key 'zzz'"]


def test_assess_latest_valid_wins_and_mismatch(tmp_path):
    spec = _spec(tmp_path)
    es = [_entry(), _entry(checks=["provenance"], decision="rejected", license_id=None, date="2026-11-05", index=1)]
    for i, e in enumerate(es):
        e.index = i
    out, invalid = assess_reviews(spec, es, ALLOWED, None, today=TODAY)
    assert out["license"].status == "APPROVED" and out["provenance"].status == "REJECTED" and not invalid
    out2, _ = assess_reviews(spec, [_entry()], ALLOWED, "e" * 40, today=TODAY)
    assert out2["license"].status == "REVISION_MISMATCH"
    assert out2["overlap"].status == "PENDING"


def test_record_refuses_invalid_and_writes_nothing(tmp_path):
    p = tmp_path / "v3_asr_reviews.yaml"
    p.write_text("# header kept\nreviews: []\n", encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    probs = record_review(p, _entry(reviewer="me"), _spec(tmp_path), ALLOWED)
    assert probs and p.read_text(encoding="utf-8") == before


def test_record_appends_valid_entry_and_keeps_header(tmp_path):
    p = tmp_path / "v3_asr_reviews.yaml"
    p.write_text("# header kept\n# second line\n\nreviews: []\n", encoding="utf-8")
    assert record_review(p, _entry(), _spec(tmp_path), ALLOWED) == []
    assert record_review(p, _entry(checks=["overlap"], license_id=None), _spec(tmp_path), ALLOWED) == []
    txt = p.read_text(encoding="utf-8")
    assert txt.startswith("# header kept\n# second line")
    es = load_reviews(p)
    assert [e.checks for e in es] == [["license", "provenance"], ["overlap"]]
    assert "index" not in yaml.safe_load(txt)["reviews"][0]


def test_cli_record_and_show(tmp_path, capsys):
    cfg = tmp_path / "configs"
    shutil.copytree(REPO / "configs", cfg)
    doc = load_registry_document(cfg)
    assert "whisper_small_en" in doc.specs
    rc = _cli(["--configs", str(cfg), "record", "--model", "whisper_small_en", "--checks", "license", "provenance",
               "overlap", "--decision", "approved", "--reviewer", "A. Reviewer", "--revision", SHA,
               "--license-id", "apache-2.0", "--evidence-url", "https://huggingface.co/openai/whisper-small.en",
               "--summary", "card read at revision; apache-2.0; web audio only", "--resolves", "WSP-OVERLAP",
               "--date", "2026-11-01"])
    assert rc == 0
    es = load_reviews(cfg / "v3_asr_reviews.yaml")
    assert len(es) == 1 and es[0].resolves == ["WSP-OVERLAP"]
    rc2 = _cli(["--configs", str(cfg), "record", "--model", "whisper_small_en", "--checks", "license",
                "--decision", "approved", "--reviewer", "todo", "--revision", SHA, "--license-id", "apache-2.0",
                "--evidence-url", "https://x.org", "--summary", "placeholder reviewer must fail"])
    assert rc2 == 1 and len(load_reviews(cfg / "v3_asr_reviews.yaml")) == 1
    assert _cli(["--configs", str(cfg), "show"]) == 0
    assert "VALID" in capsys.readouterr().out
    # the registry itself is never modified by recording a review
    assert (cfg / "v3_asr_models.yaml").read_text(encoding="utf-8") == (REPO / "configs" / "v3_asr_models.yaml").read_text(encoding="utf-8")


def test_repo_reviews_file_is_empty_by_default():
    assert load_reviews(REPO / "configs" / "v3_asr_reviews.yaml") == []
