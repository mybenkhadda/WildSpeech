"""Manual model reviews: the ONLY way a license / provenance / overlap / competition check can be approved.

configs/v3_asr_reviews.yaml holds a list of review entries. Each entry is a human attestation tied to one model,
one or more checks, one checkpoint revision, and concrete evidence:

    - model_key: whisper_small_en
      checks: [license, provenance, overlap]
      decision: approved                 # approved | rejected
      reviewer: "A. Person"
      date: "2026-11-21"
      revision_reviewed: "<commit sha>"
      license_id: apache-2.0             # required when `license` is among the checks
      acknowledged_conditions: []        # every condition id of a CONDITIONAL license (license_policy.conditional)
      overlap_basis: temporal_precedence # required when `overlap` is approved (see OVERLAP_BASES)
      mitigation: null                   # required for overlap_basis = acknowledged_overlap_permitted
      evidence:
        - url: https://huggingface.co/openai/whisper-small.en
          summary: "card at <sha>: license apache-2.0; training data = 680k h web audio"
      resolves: [WSP-OVERLAP]            # blocking restriction ids this review closes (same check only)
      notes: "..."

Safety properties (enforced by validate_entry on EVERY pipeline run, not only when recording):
  * incomplete entries are INVALID and ignored (reported with the problems); they never approve anything
  * the review must cover the revision that is pinned / present on disk (assess_reviews compares them)
  * a license approval must name the registry's license_id; that id must be permissive, or conditional with every
    condition acknowledged; prohibited / unknown licenses can never be approved
  * an overlap approval needs a POSITIVE basis. "No known overlap" is not a basis and is refused:
        enumerated_public_corpora_disjoint  registry evidence enumerates the training corpora (enumerated_corpora:
                                            true) and the reviewer states why none of them is this competition's data
        temporal_precedence                 the checkpoint (evidence.release_date / revision_date) predates the
                                            existence of the competition's released recordings
        organizer_statement                 an organiser statement (URL) that addresses this model's training data
        acknowledged_overlap_permitted      overlap is possible or likely, the rules permit the model anyway, and a
                                            written mitigation is recorded
  * a restriction is closed only by an approved, valid review of the restriction's own check that lists its id
  * a rejected (valid) review FAILS the check; for each check the latest valid entry wins (date, then file order)
record_review() validates first and refuses to write an invalid entry; it never edits the model registry.

configs/v3_asr_review_proposals.yaml holds DRAFT entries assembled from the registry evidence (prepared_by is not a
reviewer). A proposal approves nothing. `accept-proposal --id <id> --reviewer "<name>"` copies it into the reviews
file under the reviewer's name after full validation; `--edit-summary` lets the reviewer restate what they checked.

A third decision, `deferred`, DOCUMENTS an unresolved case (e.g. the Pasketti ReadNet overlap question) without
approving or rejecting anything: it needs a named reviewer, the revision and at least one `outstanding` question;
the check stays PENDING and the board shows the reviewer's open questions. A later approved/rejected entry
supersedes it (latest valid entry wins).

CLI (run by you):  python -m wildspeech.asr.reviews record | defer | show | proposals | accept-proposal | board | commands
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .model_registry import REVIEW_CHECKS, ASRModelSpec, LicensePolicy

DECISIONS = ("approved", "rejected", "deferred")
FINAL_DECISIONS = ("approved", "rejected")
MIN_OUTSTANDING_CHARS = 15
PLACEHOLDER_REVIEWERS = {"", "me", "todo", "tbd", "user", "reviewer", "<your name>", "<name>", "name", "xxx",
                         "claude", "assistant", "ai", "auto", "automatic", "pipeline", "wildspeech"}
MIN_SUMMARY_CHARS = 12
MIN_MITIGATION_CHARS = 30
OVERLAP_BASES = ("enumerated_public_corpora_disjoint", "temporal_precedence", "organizer_statement",
                 "acknowledged_overlap_permitted")
REFUSED_BASES = ("no_known_overlap", "absence_of_known_overlap", "none_known", "unknown", "assumed")


@dataclass
class ReviewEntry:
    model_key: str
    checks: list[str]
    decision: str
    reviewer: str
    date: str
    revision_reviewed: str | None = None
    license_id: str | None = None
    evidence: list[dict] = field(default_factory=list)
    resolves: list[str] = field(default_factory=list)
    notes: str | None = None
    acknowledged_conditions: list[str] = field(default_factory=list)
    overlap_basis: str | None = None
    mitigation: str | None = None
    proposal_id: str | None = None
    outstanding: list[str] = field(default_factory=list)   # deferred decisions: what is still unresolved
    index: int = -1                      # position in the file (tie-breaker); not serialised

    @classmethod
    def parse(cls, d: dict, index: int = -1) -> "ReviewEntry":
        d = dict(d or {})
        checks = d.get("checks") or ([d["check"]] if d.get("check") else [])
        ev = d.get("evidence") or []
        if isinstance(ev, dict):
            ev = [ev]
        return cls(model_key=str(d.get("model_key", "")).strip(), checks=[str(c).strip().lower() for c in checks],
                   decision=str(d.get("decision", "")).strip().lower(), reviewer=str(d.get("reviewer") or "").strip(),
                   date=str(d.get("date") or "").strip(),
                   revision_reviewed=(str(d["revision_reviewed"]).strip() or None) if d.get("revision_reviewed") else None,
                   license_id=(str(d["license_id"]).strip().lower() or None) if d.get("license_id") else None,
                   evidence=[dict(e or {}) for e in ev], resolves=[str(r).strip() for r in (d.get("resolves") or [])],
                   notes=d.get("notes"),
                   acknowledged_conditions=[str(c).strip() for c in (d.get("acknowledged_conditions") or [])],
                   overlap_basis=(str(d["overlap_basis"]).strip().lower() or None) if d.get("overlap_basis") else None,
                   mitigation=(str(d["mitigation"]).strip() or None) if d.get("mitigation") else None,
                   proposal_id=(str(d["proposal_id"]).strip() or None) if d.get("proposal_id") else None,
                   outstanding=[str(q).strip() for q in (d.get("outstanding") or []) if str(q).strip()],
                   index=index)

    def to_yaml_dict(self) -> dict:
        d = asdict(self)
        d.pop("index", None)
        return {k: v for k, v in d.items() if v not in (None, [], "")}


def _parse_date(s: str) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(str(s)[:10])
    except Exception:  # noqa: BLE001
        return None


def _overlap_problems(e: ReviewEntry, spec: ASRModelSpec) -> list[str]:
    p = []
    b = e.overlap_basis
    if not b:
        return ["overlap approval needs overlap_basis (one of " + ", ".join(OVERLAP_BASES) + "); absence of known "
                "overlap is not a basis"]
    if b in REFUSED_BASES:
        return [f"overlap_basis '{b}' is refused: absence of known overlap never approves the overlap check"]
    if b not in OVERLAP_BASES:
        return [f"unknown overlap_basis '{b}' (one of {OVERLAP_BASES})"]
    ev = spec.evidence
    if b == "enumerated_public_corpora_disjoint" and ev.enumerated_corpora is not True:
        p.append("basis enumerated_public_corpora_disjoint needs registry evidence.enumerated_corpora: true "
                 f"(registry says {ev.enumerated_corpora!r})")
    if b == "temporal_precedence" and not (ev.release_date or spec.revision_date):
        p.append("basis temporal_precedence needs evidence.release_date or revision_date in the registry")
    if b == "organizer_statement" and not any(str(x.get("url", "")).startswith("https://") and
                                              "huggingface.co" not in str(x.get("url", "")) for x in e.evidence):
        p.append("basis organizer_statement needs an https evidence URL of the organiser statement (not the model card)")
    if b == "acknowledged_overlap_permitted" and len(str(e.mitigation or "")) < MIN_MITIGATION_CHARS:
        p.append(f"basis acknowledged_overlap_permitted needs a written mitigation (>= {MIN_MITIGATION_CHARS} chars)")
    return p


def validate_entry(e: ReviewEntry, spec: ASRModelSpec | None, allowed_licenses: list[str],
                   base_dir: Path | None = None, today: _dt.date | None = None,
                   license_policy: LicensePolicy | None = None) -> list[str]:
    """Problems that make the entry INVALID (empty list = valid). Independent of the local snapshot."""
    p: list[str] = []
    today = today or _dt.date.today()
    if spec is None:
        return [f"unknown model_key {e.model_key!r}"]
    if not e.checks:
        p.append("no checks listed")
    bad = [c for c in e.checks if c not in REVIEW_CHECKS]
    if bad:
        p.append(f"unknown checks {bad} (allowed: {list(REVIEW_CHECKS)})")
    if e.decision not in DECISIONS:
        p.append(f"decision must be one of {DECISIONS}")
    if e.reviewer.strip().lower() in PLACEHOLDER_REVIEWERS:
        p.append("reviewer missing or placeholder (a named human reviewer is required)")
    d = _parse_date(e.date)
    if d is None:
        p.append("date missing or not ISO (YYYY-MM-DD)")
    elif d > today:
        p.append("date is in the future")
    if not e.revision_reviewed:
        p.append("revision_reviewed missing (the checkpoint commit you read the card at)")
    elif spec.revision and e.revision_reviewed != spec.revision:
        p.append(f"revision_reviewed {e.revision_reviewed} != registry pinned revision {spec.revision}")
    good_ev = 0
    for i, ev in enumerate(e.evidence):
        url, fpath, summ = str(ev.get("url") or ""), ev.get("file"), str(ev.get("summary") or "").strip()
        ok_src = url.startswith("https://") or (fpath and _file_exists(fpath, base_dir))
        if not ok_src:
            p.append(f"evidence[{i}]: needs an https url or an existing file")
        if len(summ) < MIN_SUMMARY_CHARS:
            p.append(f"evidence[{i}]: summary of what was verified is missing/too short")
        good_ev += bool(ok_src and len(summ) >= MIN_SUMMARY_CHARS)
    if e.decision == "deferred":
        if not [q for q in e.outstanding if len(q) >= MIN_OUTSTANDING_CHARS]:
            p.append(f"a deferred decision must list >= 1 outstanding question (>= {MIN_OUTSTANDING_CHARS} chars) "
                     "that keeps the check open")
        if e.resolves:
            p.append("a deferred decision cannot resolve restrictions")
    elif not good_ev:
        p.append("at least one complete evidence item is required")
    lp = license_policy or LicensePolicy(permissive=[a.lower() for a in allowed_licenses])
    if "license" in e.checks and e.decision == "approved":
        reg = (spec.evidence.license_id or "").lower()
        if not e.license_id:
            p.append("license review must state license_id")
        elif not reg:
            p.append("registry evidence has no license_id; record it in configs/v3_asr_models.yaml first")
        elif e.license_id != reg:
            p.append(f"license_id {e.license_id} != registry evidence license_id {reg}")
        else:
            cls = lp.classify(e.license_id)
            if cls in ("prohibited", "unknown", "missing"):
                p.append(f"license {e.license_id} is {cls} under license_policy (not in allowed_licenses)")
            elif cls == "conditional":
                need = set(lp.condition_ids(e.license_id))
                miss = sorted(need - set(e.acknowledged_conditions))
                if miss:
                    p.append(f"conditional license {e.license_id}: acknowledge every condition "
                             f"(missing: {', '.join(miss)})")
    if "overlap" in e.checks and e.decision == "approved":
        p += _overlap_problems(e, spec)
    known_r = {r.id: r for r in spec.restrictions}
    for rid in e.resolves:
        r = known_r.get(rid)
        if r is None:
            p.append(f"resolves unknown restriction {rid}")
        elif r.check not in e.checks:
            p.append(f"restriction {rid} belongs to check '{r.check}', which this review does not cover")
    if e.resolves and e.decision != "approved":
        p.append("only an approved review can resolve restrictions")
    return p


def _file_exists(fpath, base_dir: Path | None) -> bool:
    q = Path(str(fpath))
    if not q.is_absolute() and base_dir is not None:
        q = Path(base_dir) / q
    return q.exists()


def load_reviews(path: Path) -> list[ReviewEntry]:
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = raw.get("reviews") if isinstance(raw, dict) else raw
    return [ReviewEntry.parse(d, i) for i, d in enumerate(items or [])]


@dataclass
class CheckReview:
    """Outcome of the reviews for one (model, check)."""
    check: str
    status: str                          # APPROVED | REJECTED | DEFERRED | PENDING | REVISION_MISMATCH
    entry: ReviewEntry | None = None
    detail: str = ""
    resolved: set = field(default_factory=set)


def assess_reviews(spec: ASRModelSpec, reviews: list[ReviewEntry], allowed_licenses: list[str],
                   snapshot_revision: str | None = None, base_dir: Path | None = None,
                   today: _dt.date | None = None, license_policy: LicensePolicy | None = None
                   ) -> tuple[dict[str, CheckReview], list[dict]]:
    """-> ({check: CheckReview}, invalid-entry rows). The latest VALID entry per check decides; a valid approval
    whose revision differs from the local snapshot's revision is REVISION_MISMATCH (not approved)."""
    mine = [e for e in reviews if e.model_key == spec.key]
    invalid, valid = [], []
    for e in mine:
        probs = validate_entry(e, spec, allowed_licenses, base_dir, today, license_policy)
        if probs:
            invalid.append({"model_key": spec.key, "index": e.index, "checks": ",".join(e.checks),
                            "reviewer": e.reviewer, "date": e.date, "problems": "; ".join(probs)})
        else:
            valid.append(e)
    valid.sort(key=lambda e: (_parse_date(e.date) or _dt.date.min, e.index))
    out = {}
    for c in REVIEW_CHECKS:
        es = [e for e in valid if c in e.checks]
        if not es:
            out[c] = CheckReview(c, "PENDING", None, "no valid review recorded")
            continue
        e = es[-1]
        if e.decision == "rejected":
            out[c] = CheckReview(c, "REJECTED", e, f"rejected by {e.reviewer} on {e.date}: {e.notes or ''}".strip())
            continue
        if e.decision == "deferred":
            out[c] = CheckReview(c, "DEFERRED", e, f"deferred by {e.reviewer} on {e.date}; outstanding: "
                                                   + " | ".join(e.outstanding))
            continue
        if snapshot_revision and e.revision_reviewed != snapshot_revision:
            out[c] = CheckReview(c, "REVISION_MISMATCH", e,
                                 f"review covers revision {e.revision_reviewed}; local snapshot is {snapshot_revision}")
            continue
        res = {rid for x in es if x.decision == "approved" and (not snapshot_revision or x.revision_reviewed == snapshot_revision)
               for rid in x.resolves}
        extra = f"; basis={e.overlap_basis}" if (c == "overlap" and e.overlap_basis) else ""
        out[c] = CheckReview(c, "APPROVED", e, f"approved by {e.reviewer} on {e.date} (rev {e.revision_reviewed}){extra}", res)
    return out, invalid


def record_review(path: Path, entry: ReviewEntry, spec: ASRModelSpec, allowed_licenses: list[str],
                  base_dir: Path | None = None, license_policy: LicensePolicy | None = None) -> list[str]:
    """Validate, then append atomically. Returns the problems; nothing is written when there are any."""
    probs = validate_entry(entry, spec, allowed_licenses, base_dir, license_policy=license_policy)
    if probs:
        return probs
    from ..io_utils import atomic_write_text

    p = Path(path)
    raw = (yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None) or {}
    header = _header(p)
    items = list(raw.get("reviews") or []) if isinstance(raw, dict) else list(raw or [])
    items.append(entry.to_yaml_dict())
    body = yaml.safe_dump({"reviews": items}, sort_keys=False, allow_unicode=True, width=110)
    atomic_write_text(p, header + body)
    return []


def _header(p: Path) -> str:
    """Keep the leading comment block of the reviews file when rewriting it."""
    if not p.exists():
        return "# WildSpeech V3 manual ASR model reviews (see wildspeech/asr/reviews.py)\n\n"
    lines = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#") or not ln.strip():
            lines.append(ln)
        else:
            break
    return "\n".join(lines).rstrip() + "\n\n"


# ========================================================================================== proposals
@dataclass
class Proposal:
    id: str
    entry: ReviewEntry
    prepared_by: str = ""
    prepared_on: str = ""
    ready: bool = True                  # False = the draft itself says a human judgement is still missing
    open_questions: list[str] = field(default_factory=list)


def load_proposals(path: Path) -> list[Proposal]:
    p = Path(path)
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out = []
    for i, d in enumerate(raw.get("proposals") or []):
        d = dict(d or {})
        e = ReviewEntry.parse({**d, "reviewer": "", "date": ""}, i)
        e.proposal_id = str(d.get("id"))
        out.append(Proposal(id=str(d.get("id")), entry=e, prepared_by=str(d.get("prepared_by") or ""),
                            prepared_on=str(d.get("prepared_on") or ""), ready=d.get("ready", True) is not False,
                            open_questions=[str(q) for q in (d.get("open_questions") or [])]))
    return out


def proposal_table(proposals: list[Proposal]):
    import pandas as pd

    return pd.DataFrame([{"proposal_id": p.id, "model_key": p.entry.model_key, "checks": ",".join(p.entry.checks),
                          "decision": p.entry.decision, "ready": p.ready, "overlap_basis": p.entry.overlap_basis,
                          "acknowledged_conditions": ",".join(p.entry.acknowledged_conditions),
                          "resolves": ",".join(p.entry.resolves), "open_questions": " | ".join(p.open_questions),
                          "prepared_by": p.prepared_by, "prepared_on": p.prepared_on} for p in proposals])


def accept_proposal(reviews_file: Path, proposal: Proposal, reviewer: str, spec: ASRModelSpec,
                    allowed_licenses: list[str], base_dir: Path | None = None, date: str | None = None,
                    summary: str | None = None, mitigation: str | None = None, confirm_open: bool = False,
                    license_policy: LicensePolicy | None = None) -> list[str]:
    """A named human turns a draft into a review. Refused when the draft has open questions (unless the reviewer
    passes confirm_open=True, i.e. states they resolved them) or when the resulting entry fails validation."""
    if proposal.open_questions and not confirm_open:
        return [f"proposal {proposal.id} has open questions; resolve them, then pass --resolved-open-questions: "
                + " | ".join(proposal.open_questions)]
    e = ReviewEntry.parse(proposal.entry.to_yaml_dict())
    e.reviewer = reviewer
    e.date = date or _dt.date.today().isoformat()
    e.proposal_id = proposal.id
    if summary:
        e.evidence = [{**x, "summary": summary} for x in e.evidence]
    if mitigation:
        e.mitigation = mitigation
    e.notes = ((e.notes or "") + f" [accepted from proposal {proposal.id} prepared by {proposal.prepared_by}]").strip()
    return record_review(reviews_file, e, spec, allowed_licenses, base_dir, license_policy)


# ==================================================================================== review board
CHECK_LABEL = {"license": "license & restrictions", "provenance": "training-data provenance",
               "overlap": "competition-data overlap", "competition": "competition permission"}


def _q(s: str) -> str:
    return '"' + str(s).replace('"', "'") + '"'


def check_commands(spec: ASRModelSpec, check: str, proposals: list[Proposal]) -> dict:
    """Exact commands for one (model, check). Nothing is executed; the strings are shown to the reviewer."""
    mine = [p for p in proposals if p.entry.model_key == spec.key and check in p.entry.checks]
    ready = [p.id for p in mine if p.ready and not p.open_questions]
    blocked = [p.id for p in mine if not (p.ready and not p.open_questions)]
    base = "python -m wildspeech.asr.reviews"
    rev = spec.revision or "<commit sha>"
    open_r = [r.id for r in spec.blocking_restrictions(check)]
    out = {"accept": "", "accept_after_deciding": "", "reject": "", "defer": "", "record": ""}
    if ready:
        out["accept"] = f'{base} accept-proposal --id {" ".join(ready)} --reviewer "<your full name>"'
    if blocked:
        mit = ' --mitigation "<your own mitigation text>"' if check == "overlap" else ""
        out["accept_after_deciding"] = (f'{base} accept-proposal --id {" ".join(blocked)} --reviewer "<your full name>" '
                                        f"--resolved-open-questions{mit}")
    out["reject"] = (f"{base} record --model {spec.key} --checks {check} --decision rejected --reviewer \"<your full name>\" "
                     f"--revision {rev} --evidence-url {spec.evidence.model_card_url or '<source url>'} "
                     f"--summary \"<why the model is rejected>\"")
    out["defer"] = (f"{base} defer --model {spec.key} --checks {check} --reviewer \"<your full name>\" --revision {rev} "
                    f"--outstanding \"<the question that is still unresolved>\"")
    extra = (" --license-id " + (spec.evidence.license_id or "<id>")) if check == "license" else ""
    extra += " --overlap-basis <basis> " if check == "overlap" else ""
    extra += (" --resolves " + " ".join(open_r)) if open_r else ""
    out["record"] = (f"{base} record --model {spec.key} --checks {check} --decision approved --reviewer \"<your full name>\" "
                     f"--revision {rev} --evidence-url <source url> --summary \"<what you verified>\"{extra}")
    return out


def review_board(specs: dict[str, ASRModelSpec], reviews: list[ReviewEntry], proposals: list[Proposal],
                 allowed_licenses: list[str], license_policy: LicensePolicy | None = None,
                 base_dir: Path | None = None, today: _dt.date | None = None):
    """One row per (model, review check): current decision, who/when, open restrictions, drafts, open questions,
    whether a HUMAN decision is still needed, and the exact commands. Registry + review files only."""
    import pandas as pd

    rows = []
    for k, s in specs.items():
        rv, invalid = assess_reviews(s, reviews, allowed_licenses, s.revision, base_dir, today, license_policy)
        for c in REVIEW_CHECKS:
            cr = rv[c]
            restr = s.blocking_restrictions(c)
            open_r = [r.id for r in restr if r.id not in cr.resolved]
            mine = [p for p in proposals if p.entry.model_key == k and c in p.entry.checks]
            needs_review = c != "competition" or bool(restr)   # competition is a policy check unless restricted
            approved = cr.status == "APPROVED" and not open_r
            decision_needed = needs_review and not approved and cr.status not in ("REJECTED",)
            cmds = check_commands(s, c, proposals) if decision_needed else {}
            rows.append({"model_key": k, "priority": s.priority, "check": c, "label": CHECK_LABEL[c],
                         "review_required": needs_review, "status": cr.status if needs_review else "POLICY",
                         "detail": cr.detail, "reviewer": cr.entry.reviewer if cr.entry else "",
                         "date": cr.entry.date if cr.entry else "",
                         "basis": (cr.entry.overlap_basis or "") if (cr.entry and c == "overlap") else "",
                         "open_restrictions": ",".join(open_r),
                         "restriction_text": " | ".join(r.text for r in restr if r.id in open_r)[:600],
                         "proposals_ready": ",".join(p.id for p in mine if p.ready and not p.open_questions),
                         "proposals_need_decision": ",".join(p.id for p in mine if not (p.ready and not p.open_questions)),
                         "open_questions": " | ".join(q for p in mine for q in p.open_questions),
                         "outstanding": " | ".join(cr.entry.outstanding) if (cr.entry and cr.status == "DEFERRED") else "",
                         "invalid_entries": len([iv for iv in invalid if c in iv["checks"].split(",")]),
                         "decision_needed": decision_needed, **{f"cmd_{x}": y for x, y in cmds.items()}})
    t = pd.DataFrame(rows)
    return t.sort_values(["priority", "model_key", "check"]).reset_index(drop=True) if not t.empty else t


def model_decisions(board) -> "object":
    """Per model: one line of review state + the decisions that are still the reviewer's to make."""
    import pandas as pd

    if board is None or board.empty:
        return pd.DataFrame()
    out = []
    for k, g in board.groupby("model_key", sort=False):
        need = g[g["decision_needed"]]
        rej = g[g["status"] == "REJECTED"]
        out.append({"model_key": k, "priority": int(g["priority"].iloc[0]),
                    **{f"{c}": st for c, st in zip(g["check"], g["status"])},
                    "rejected": bool(len(rej)),
                    "reviews_complete": bool(need.empty and rej.empty),
                    "outstanding_decisions": " ; ".join(
                        f"{r.check}: " + (r.open_questions or r.outstanding or r.restriction_text or r.detail)[:240]
                        for r in need.itertuples()),
                    "ready_to_accept": ",".join(x for x in need["proposals_ready"] if x),
                    "needs_your_judgement": ",".join(x for x in need["proposals_need_decision"] if x)})
    return pd.DataFrame(out).sort_values("priority").reset_index(drop=True)


def record_deferral(path: Path, spec: ASRModelSpec, checks: list[str], reviewer: str, revision: str,
                    outstanding: list[str], notes: str | None = None, evidence: list[dict] | None = None,
                    allowed_licenses: list[str] | None = None, license_policy: LicensePolicy | None = None,
                    base_dir: Path | None = None, date: str | None = None) -> list[str]:
    e = ReviewEntry(model_key=spec.key, checks=list(checks), decision="deferred", reviewer=reviewer,
                    date=date or _dt.date.today().isoformat(), revision_reviewed=revision, evidence=list(evidence or []),
                    notes=notes, outstanding=[q for q in outstanding if q])
    return record_review(path, e, spec, list(allowed_licenses or []), base_dir, license_policy)


# ============================================================================================== CLI
def _cli(argv=None) -> int:
    from .model_registry import load_registry_document

    ap = argparse.ArgumentParser(prog="python -m wildspeech.asr.reviews")
    ap.add_argument("--configs", default=None, help="configs directory (default: auto-detected)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="validate and append one review")
    r.add_argument("--model", required=True)
    r.add_argument("--checks", nargs="+", required=True, choices=list(REVIEW_CHECKS))
    r.add_argument("--decision", required=True, choices=list(DECISIONS))
    r.add_argument("--reviewer", required=True)
    r.add_argument("--revision", required=True, help="checkpoint commit the card was read at")
    r.add_argument("--license-id", default=None)
    r.add_argument("--acknowledge", nargs="*", default=[], help="condition ids of a conditional license")
    r.add_argument("--overlap-basis", default=None, choices=list(OVERLAP_BASES))
    r.add_argument("--mitigation", default=None)
    r.add_argument("--evidence-url", action="append", default=[])
    r.add_argument("--evidence-file", action="append", default=[])
    r.add_argument("--summary", required=True, help="what you verified (applies to every evidence item)")
    r.add_argument("--resolves", nargs="*", default=[])
    r.add_argument("--notes", default=None)
    r.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    df = sub.add_parser("defer", help="document an UNRESOLVED decision (approves nothing; the check stays pending)")
    df.add_argument("--model", required=True)
    df.add_argument("--checks", nargs="+", required=True, choices=list(REVIEW_CHECKS))
    df.add_argument("--reviewer", required=True)
    df.add_argument("--revision", required=True)
    df.add_argument("--outstanding", action="append", required=True, help="an open question (repeatable)")
    df.add_argument("--evidence-url", action="append", default=[])
    df.add_argument("--summary", default=None, help="what you looked at so far (needed with --evidence-url)")
    df.add_argument("--notes", default=None)
    df.add_argument("--date", default=None)
    sub.add_parser("show", help="list entries and whether they validate")
    sub.add_parser("proposals", help="list the drafted review proposals")
    sub.add_parser("board", help="per model and check: decision, open restrictions, drafts, what you still decide")
    cm = sub.add_parser("commands", help="print the exact commands for every decision that is still open")
    cm.add_argument("--model", nargs="*", default=None)
    a2 = sub.add_parser("accept-proposal", help="accept a drafted proposal under YOUR name")
    a2.add_argument("--id", required=True, nargs="+")
    a2.add_argument("--reviewer", required=True)
    a2.add_argument("--date", default=None)
    a2.add_argument("--summary", default=None, help="restate what you verified (replaces the draft summary)")
    a2.add_argument("--mitigation", default=None)
    a2.add_argument("--resolved-open-questions", action="store_true",
                    help="you resolved the proposal's open questions yourself")
    a = ap.parse_args(argv)
    if a.configs:
        cfg_dir = Path(a.configs)
    else:
        from ..config import find_config_dir

        cfg_dir = find_config_dir()
    doc = load_registry_document(cfg_dir)
    path = cfg_dir / "v3_asr_reviews.yaml"
    prop_path = cfg_dir / "v3_asr_review_proposals.yaml"
    lp = doc.license_policy
    if a.cmd == "show":
        es = load_reviews(path)
        if not es:
            print(f"no reviews in {path}")
        for e in es:
            probs = validate_entry(e, doc.specs.get(e.model_key), doc.allowed_licenses, cfg_dir.parent, license_policy=lp)
            print(f"[{e.index}] {e.model_key} {e.checks} {e.decision} by {e.reviewer} {e.date} rev={e.revision_reviewed} "
                  f"-> {'VALID' if not probs else 'INVALID: ' + '; '.join(probs)}")
        return 0
    if a.cmd in ("board", "commands"):
        import pandas as pd

        pd.set_option("display.width", 240)
        pd.set_option("display.max_colwidth", 110)
        board = review_board(doc.specs, load_reviews(path), load_proposals(prop_path), doc.allowed_licenses, lp,
                             cfg_dir.parent)
        if a.cmd == "board":
            print(model_decisions(board).to_string(index=False))
            print()
            cols = ["model_key", "check", "status", "reviewer", "date", "open_restrictions", "proposals_ready",
                    "proposals_need_decision", "decision_needed"]
            print(board[cols].to_string(index=False))
            return 0
        sel = board[board["decision_needed"]]
        if a.model:
            sel = sel[sel["model_key"].isin(a.model)]
        if sel.empty:
            print("no open review decisions")
        for r in sel.to_dict("records"):
            print(f"\n## {r['model_key']} · {r['check']} ({r['label']}) — {r['status']}")
            if r["open_questions"] or r["outstanding"]:
                print("   open questions:", r["open_questions"] or r["outstanding"])
            if r["open_restrictions"]:
                print("   open restrictions:", r["open_restrictions"])
            for kind in ("accept", "accept_after_deciding", "record", "reject", "defer"):
                cmd = r.get(f"cmd_{kind}")
                if cmd:
                    print(f"   [{kind}] {cmd}")
        return 0
    if a.cmd == "defer":
        spec = doc.specs.get(a.model)
        if spec is None:
            print(f"unknown model {a.model}; registry has {sorted(doc.specs)}", file=sys.stderr)
            return 2
        ev = [{"url": u, "summary": a.summary or ""} for u in a.evidence_url]
        probs = record_deferral(path, spec, a.checks, a.reviewer, a.revision, a.outstanding, a.notes, ev,
                                doc.allowed_licenses, lp, cfg_dir.parent, a.date)
        if probs:
            print("REFUSED (nothing written):", file=sys.stderr)
            for q in probs:
                print("  -", q, file=sys.stderr)
            return 1
        print(f"recorded DEFERRED decision for {a.model} {a.checks}: {a.outstanding}")
        return 0
    if a.cmd == "proposals":
        import pandas as pd

        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 90)
        t = proposal_table(load_proposals(prop_path))
        print(t.to_string(index=False) if not t.empty else f"no proposals in {prop_path}")
        return 0
    if a.cmd == "accept-proposal":
        props = {p.id: p for p in load_proposals(prop_path)}
        rc = 0
        for pid in a.id:
            p = props.get(pid)
            if p is None:
                print(f"unknown proposal {pid}; known: {sorted(props)}", file=sys.stderr)
                rc = 2
                continue
            spec = doc.specs.get(p.entry.model_key)
            if spec is None:
                print(f"{pid}: unknown model {p.entry.model_key}", file=sys.stderr)
                rc = 2
                continue
            probs = accept_proposal(path, p, a.reviewer, spec, doc.allowed_licenses, cfg_dir.parent, a.date, a.summary,
                                    a.mitigation, a.resolved_open_questions, lp)
            if probs:
                print(f"REFUSED {pid} (nothing written):", file=sys.stderr)
                for q in probs:
                    print("  -", q, file=sys.stderr)
                rc = 1
            else:
                print(f"accepted {pid} -> {p.entry.model_key} {p.entry.checks} ({p.entry.decision}) by {a.reviewer}")
        return rc
    spec = doc.specs.get(a.model)
    if spec is None:
        print(f"unknown model {a.model}; registry has {sorted(doc.specs)}", file=sys.stderr)
        return 2
    ev = [{"url": u, "summary": a.summary} for u in a.evidence_url] + [{"file": f, "summary": a.summary} for f in a.evidence_file]
    e = ReviewEntry(model_key=a.model, checks=a.checks, decision=a.decision, reviewer=a.reviewer,
                    date=a.date or _dt.date.today().isoformat(), revision_reviewed=a.revision,
                    license_id=(a.license_id or "").lower() or None, evidence=ev, resolves=a.resolves, notes=a.notes,
                    acknowledged_conditions=list(a.acknowledge or []), overlap_basis=a.overlap_basis,
                    mitigation=a.mitigation)
    probs = record_review(path, e, spec, doc.allowed_licenses, cfg_dir.parent, lp)
    if probs:
        print("REFUSED (nothing written):", file=sys.stderr)
        for q in probs:
            print("  -", q, file=sys.stderr)
        return 1
    print(f"recorded review for {a.model} {a.checks} ({a.decision}) in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
