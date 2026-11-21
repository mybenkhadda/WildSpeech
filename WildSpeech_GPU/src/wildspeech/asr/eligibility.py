"""Eligibility of local speech models in 13 INDEPENDENT dimensions + one lifecycle status. No override switch.

  dimension                    checks                                          passes when
  registry_validity            registry_valid, repo_identity                   schema valid; official repo verified,
                                                                                revision pinned to a commit sha
  license_documentation        license_documented                              license id + URL recorded; class is
                                                                                permissive or conditional
  license_approval             license_review, license_restrictions            valid APPROVED license review (conditional
                                                                                licenses: every condition acknowledged);
                                                                                blocking license restrictions resolved
  training_provenance          training_documented, provenance_review          sources + evidence URL; approved review
  overlap_assessment           overlap_review, overlap_restrictions            approved review with a POSITIVE basis
                                                                                (never "no known overlap"); blocking
                                                                                overlap restrictions resolved
  competition_permission       competition_policy, competition_restrictions    family named by the organisers, or allowed
                                                                                by the general rule with a permissive
                                                                                license; statement + source URL recorded
  checkpoint_availability      checkpoint_present, revision_match              snapshot on disk at the pinned commit
  checkpoint_files             manifest_files, integrity                       every manifest file present at the
                                                                                published size; LFS sha256 verified when
                                                                                hashes are checked (verify_hashes)
  architecture_compatibility   architecture_expected, adapter_compatible,      config class/model_type as registered,
                               remote_code                                     adapter drives it, no unreviewed auto_map
  processor_compatibility      components                                      processor/tokenizer instantiate locally and
                                                                                meet the adapter contract (deep check)
  framework_dependencies       packages_importable, weights_loadable           packages + platform; weights loadable by the
                                                                                installed torch (.bin needs torch >= 2.6)
  gpu_readiness                gpu_fit                                         CUDA present and the model fits in VRAM
                                                                                (WARN: CPU-feasible fallback)
  inference_readiness          smoke_test                                      last smoke decode for this revision passed
                                                                                (informational; P24 runs it before the
                                                                                full pass)

Check statuses: PASS | WARN | PENDING | FAIL | NOT_PROBED | NOT_APPLICABLE. A dimension is the worst of its checks
(WARN / NOT_APPLICABLE count as PASS).

Lifecycle (first matching rule):
  INELIGIBLE                   registry invalid, prohibited license, a rejected review, or family not permitted
  LEGAL_REVIEW_PENDING         license documentation / approval / competition permission pending
  PROVENANCE_REVIEW_PENDING    training provenance / overlap assessment pending
  DOWNLOADING                  a live download lock exists for the model
  READY_FOR_DOWNLOAD           legal + provenance + competition PASS, checkpoint not on disk
  CHECKPOINT_INCOMPLETE        snapshot present but files missing / wrong size / hash mismatch / wrong revision
  FRAMEWORK_INCOMPATIBLE       architecture, processor, framework or GPU dimension FAIL
  INFERENCE_RUNNING            a live inference lock exists (heartbeat fresh)
  FAILED                       the last inference / smoke attempt failed and no complete accepted output exists
  INFERENCE_COMPLETE           accepted outputs cover every configured split
  READY_FOR_INFERENCE          everything above passes
`eligible` (inference may run) = dimensions 1-12 all PASS. `download_allowed` = legal + provenance + competition PASS
(the model manager additionally needs ALLOW_MODEL_DOWNLOADS). A completed eligibility check is NEVER inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import v3
from .model_registry import ASRModelSpec, CompetitionPolicy, LicensePolicy

PASS, WARN, PENDING, FAIL, NOT_PROBED, NA = "PASS", "WARN", "PENDING", "FAIL", "NOT_PROBED", "NOT_APPLICABLE"
RANK = {PASS: 0, WARN: 0, NA: 0, NOT_PROBED: 1, PENDING: 2, FAIL: 3}
DIM_STATUS = {0: "PASS", 1: "NOT_PROBED", 2: "PENDING", 3: "FAIL"}

DIMENSIONS = {
    "registry_validity": ("registry_valid", "repo_identity"),
    "license_documentation": ("license_documented",),
    "license_approval": ("license_review", "license_restrictions"),
    "training_provenance": ("training_documented", "provenance_review"),
    "overlap_assessment": ("overlap_review", "overlap_restrictions"),
    "competition_permission": ("competition_policy", "competition_restrictions"),
    "checkpoint_availability": ("checkpoint_present", "revision_match"),
    "checkpoint_files": ("manifest_files", "integrity"),
    "architecture_compatibility": ("architecture_expected", "adapter_compatible", "remote_code"),
    "processor_compatibility": ("components",),
    "framework_dependencies": ("packages_importable", "weights_loadable"),
    "gpu_readiness": ("gpu_fit",),
    "inference_readiness": ("smoke_test",),
}
CHECKS = tuple(c for cs in DIMENSIONS.values() for c in cs)
GATING = tuple(d for d in DIMENSIONS if d != "inference_readiness")
GROUPS = {"legal": ("registry_validity", "license_documentation", "license_approval", "competition_permission"),
          "provenance": ("training_provenance", "overlap_assessment"),
          "checkpoint": ("checkpoint_availability", "checkpoint_files"),
          "framework": ("architecture_compatibility", "processor_compatibility", "framework_dependencies", "gpu_readiness")}
GROUP_STATUS = {"PASS": "PASS", "NOT_PROBED": "NOT_PROBED", "PENDING": "PENDING_REVIEW", "FAIL": "FAIL"}
LIFECYCLE = ("LEGAL_REVIEW_PENDING", "PROVENANCE_REVIEW_PENDING", "READY_FOR_DOWNLOAD", "DOWNLOADING",
             "CHECKPOINT_INCOMPLETE", "FRAMEWORK_INCOMPATIBLE", "READY_FOR_INFERENCE", "INFERENCE_RUNNING",
             "INFERENCE_COMPLETE", "INELIGIBLE", "FAILED")
REVIEW_CMD = "python -m wildspeech.asr.reviews record --model {key} --checks {check} --decision approved " \
             "--reviewer \"<your name>\" --revision {rev} --evidence-url <source url> --summary \"<what you verified>\""
PROPOSAL_CMD = "python -m wildspeech.asr.reviews accept-proposal --id {ids} --reviewer \"<your name>\""


@dataclass
class CheckResult:
    status: str
    detail: str = ""
    action: str = ""


@dataclass
class EligibilityRecord:
    model_key: str
    status: str                                            # ELIGIBLE | NOT_EXECUTED (schema-1/2 callers)
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)              # check -> status string
    source: str | None = None
    architecture: str | None = None
    snapshot_revision: str | None = None
    results: dict = field(default_factory=dict)             # check -> CheckResult
    dimensions: dict = field(default_factory=dict)          # 13 dimensions -> PASS | NOT_PROBED | PENDING | FAIL
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    invalid_reviews: list[dict] = field(default_factory=list)
    snapshot_origin: str | None = None
    lifecycle: str = "LEGAL_REVIEW_PENDING"
    groups: dict = field(default_factory=dict)              # legal / provenance / checkpoint / framework
    download_allowed: bool = False
    snapshot_kind: str | None = None
    gpu: dict = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.status == v3.ELIGIBLE

    def as_row(self) -> dict:
        return {"lifecycle": self.lifecycle, "eligibility": self.status, "eligible": self.eligible,
                "download_allowed": self.download_allowed,
                **{f"dim_{d}": self.dimensions.get(d) for d in DIMENSIONS},
                **{f"{g}_status": self.groups.get(g) for g in GROUPS},
                "reasons": "; ".join(self.reasons), "required_actions": " | ".join(self.actions),
                "warnings": "; ".join(self.warnings), "invalid_reviews": len(self.invalid_reviews),
                "resolved_source": self.source, "snapshot_origin": self.snapshot_origin, "snapshot_kind": self.snapshot_kind,
                "architecture": self.architecture, "snapshot_revision": self.snapshot_revision,
                "gpu_need_gb": self.gpu.get("estimated_need_gb"), "gpu_fits": self.gpu.get("fits"),
                **{f"check_{k}": self.checks.get(k) for k in CHECKS}}

    def check_rows(self) -> list[dict]:
        out = []
        for d, cs in DIMENSIONS.items():
            for c in cs:
                r = self.results.get(c)
                if r is not None:
                    out.append({"model_key": self.model_key, "dimension": d, "check": c, "status": r.status,
                                "detail": r.detail, "action": r.action})
        return out

    def dimension_rows(self) -> list[dict]:
        return [{"model_key": self.model_key, "dimension": d, "status": self.dimensions.get(d),
                 "lifecycle": self.lifecycle} for d in DIMENSIONS]

    def evidence_json(self) -> dict:
        return {"model_key": self.model_key, "lifecycle": self.lifecycle, "eligible": self.eligible,
                "download_allowed": self.download_allowed, "dimensions": self.dimensions, "groups": self.groups,
                "checks": {c: {"status": r.status, "detail": r.detail, "action": r.action} for c, r in self.results.items()},
                "snapshot": {"path": self.source, "origin": self.snapshot_origin, "kind": self.snapshot_kind,
                             "revision": self.snapshot_revision, "architecture": self.architecture},
                "gpu": self.gpu, "warnings": self.warnings, "invalid_reviews": self.invalid_reviews,
                "required_actions": self.actions}


# ---------------------------------------------------------------------- back-compat re-exports
from .snapshots import detect_architecture  # noqa: E402,F401  (tests / callers import it from here)


def resolve_local_snapshot(spec: ASRModelSpec, cache_dir: str | None = None) -> tuple[str | None, str | None]:
    """(path, revision) or (None, reason) — filesystem only (kept for schema-1 callers)."""
    from .snapshots import resolve_snapshot

    s = resolve_snapshot(spec, cache_dir)
    return (s.path, s.revision) if s.path else (None, s.reason)


def framework_available(architecture: str) -> tuple[bool, str]:
    from .environment import ARCH_PACKAGES, package_version

    pk = ARCH_PACKAGES.get(architecture)
    if not pk:
        return False, f"architecture '{architecture}' has no supported local adapter"
    for n in pk:
        ok, _, why = package_version(n)
        if not ok:
            return False, why
    return True, ",".join(pk)


def _worst(statuses) -> str:
    ranks = [RANK[s] for s in statuses]
    return DIM_STATUS[max(ranks)] if ranks else "NOT_PROBED"


class EligibilityChecker:
    def __init__(self, cfg, allowed_licenses: list[str], policy: CompetitionPolicy | None = None,
                 reviews: list | None = None, base_dir: Path | None = None, today=None,
                 license_policy: LicensePolicy | None = None, runtime: dict | None = None, env: dict | None = None,
                 proposals: list | None = None, hash_cache: dict | None = None):
        self.cfg = cfg
        self.allowed = {a.lower() for a in allowed_licenses}
        self.policy = policy or CompetitionPolicy()
        self.lp = license_policy or LicensePolicy(permissive=sorted(self.allowed))
        self.reviews = list(reviews or [])
        self.base_dir = base_dir
        self.today = today
        self.runtime = dict(runtime or {})           # model_key -> {downloading, running, last_failed, complete, smoke}
        self._env = env
        self.proposals = list(proposals or [])
        self.hash_cache = hash_cache if hash_cache is not None else {}

    @property
    def env(self) -> dict:
        if self._env is None:
            from .environment import framework_environment

            self._env = framework_environment()
        return self._env

    # ---------------------------------------------------------------------------- helpers
    def _proposal_ids(self, key: str, check: str) -> list[str]:
        return [p.id for p in self.proposals if p.entry.model_key == key and check in p.entry.checks]

    def _review_action(self, spec, check, extra: str = "") -> str:
        ids = self._proposal_ids(spec.key, check)
        if ids:
            return PROPOSAL_CMD.format(ids=" ".join(ids)) + "  (draft in configs/v3_asr_review_proposals.yaml)"
        return REVIEW_CMD.format(key=spec.key, check=check, rev=spec.revision or "<sha>") + extra

    @staticmethod
    def _resolved(rv, check) -> set:
        r = rv.get(check)
        return set(r.resolved) if (r is not None and r.status == "APPROVED") else set()

    def _open(self, spec, check, rv) -> list[str]:
        return [r.id for r in spec.blocking_restrictions(check) if r.id not in self._resolved(rv, check)]

    def _review_result(self, spec, cr, check) -> CheckResult:
        if cr.status == "APPROVED":
            return CheckResult(PASS, cr.detail)
        if cr.status == "REJECTED":
            return CheckResult(FAIL, cr.detail, "model rejected by review (record a new review to change this)")
        if cr.status == "DEFERRED":
            return CheckResult(PENDING, cr.detail, "decide the outstanding question(s), then record approved/rejected: "
                               + self._review_action(spec, check))
        if cr.status == "REVISION_MISMATCH":
            return CheckResult(PENDING, cr.detail, "re-review the card at the local revision: " + self._review_action(spec, check))
        return CheckResult(PENDING, f"{check} review not recorded", self._review_action(spec, check))

    def _restr_result(self, spec, check, rv) -> CheckResult:
        open_ids = self._open(spec, check, rv)
        if not open_ids:
            n = len(spec.blocking_restrictions(check))
            return CheckResult(PASS if n else NA, f"{n} blocking restriction(s), all resolved" if n else "none")
        txt = "; ".join(f"{r.id}: {r.text}" for r in spec.blocking_restrictions(check) if r.id in open_ids)
        return CheckResult(PENDING, txt, self._review_action(spec, check, " --resolves " + " ".join(open_ids)))

    # ------------------------------------------------------------------------ 1-6: evidence
    def _registry(self, spec, R):
        errs = spec.validate()
        R["registry_valid"] = CheckResult(FAIL if errs else PASS, "; ".join(errs) or "schema valid",
                                          "fix configs/v3_asr_models.yaml" if errs else "")
        from .snapshots import SHA_RE

        if spec.hf_id_status != "verified_public" and spec.hf_id:
            R["repo_identity"] = CheckResult(PENDING, f"hf_id {spec.hf_id} is {spec.hf_id_status}",
                                             "confirm the official repository and set hf_id_status: verified_public")
        elif spec.hf_id and not (spec.revision and SHA_RE.match(spec.revision)):
            R["repo_identity"] = CheckResult(PENDING, "revision not pinned to a 40-character commit sha",
                                             "pin `revision` in configs/v3_asr_models.yaml (HF API `sha`)")
        else:
            R["repo_identity"] = CheckResult(PASS, f"{spec.hf_id or spec.local_path} @ {spec.revision or 'local'}")

    def _license(self, spec, rv, R):
        lic = (spec.evidence.license_id or "").lower()
        cls = self.lp.classify(lic)
        url = spec.evidence.license_url
        if cls == "missing":
            R["license_documented"] = CheckResult(PENDING, "no license_id in the registry evidence",
                                                  f"read {spec.evidence.model_card_url or 'the model card'} and record "
                                                  "evidence.license_id / license_url")
        elif cls in ("prohibited",):
            R["license_documented"] = CheckResult(FAIL, f"license '{lic}' is prohibited (non-commercial / restricted)",
                                                  "do not use this model")
        elif cls == "unknown" or lic not in self.allowed:
            R["license_documented"] = CheckResult(FAIL, f"license '{lic}' is not in license_policy (permissive / conditional)",
                                                  "do not use this model, or add the license to license_policy after legal review")
        elif not url:
            R["license_documented"] = CheckResult(PENDING, f"{lic}: no license_url recorded", "record evidence.license_url")
        else:
            conds = self.lp.condition_ids(lic)
            R["license_documented"] = CheckResult(PASS if cls == "permissive" else WARN,
                                                  f"{lic} [{cls}] ({url})" + (f"; conditions {conds}" if conds else ""))
        R["license_review"] = self._review_result(spec, rv["license"], "license")
        R["license_restrictions"] = self._restr_result(spec, "license", rv)

    def _provenance(self, spec, rv, R):
        ev = spec.evidence
        if not ev.training_sources:
            R["training_documented"] = CheckResult(PENDING, "no training sources recorded",
                                                   "record evidence.training_sources from the model card / paper")
        elif not (ev.training_evidence_url or ev.model_card_url):
            R["training_documented"] = CheckResult(PENDING, "training sources have no evidence URL",
                                                   "record evidence.training_evidence_url")
        else:
            enum = {True: "enumerated", False: "not enumerated", "partial": "partially enumerated"}.get(ev.enumerated_corpora, "enumeration unknown")
            R["training_documented"] = CheckResult(PASS if ev.enumerated_corpora is True else WARN,
                                                   f"[{enum}] " + " | ".join(ev.training_sources)[:260])
        R["provenance_review"] = self._review_result(spec, rv["provenance"], "provenance")
        R["overlap_review"] = self._review_result(spec, rv["overlap"], "overlap")
        if R["overlap_review"].status == PENDING and rv["overlap"].status == "PENDING":
            R["overlap_review"].detail = ("overlap assessment not recorded: a positive basis is required "
                                          "(absence of known overlap is not accepted)")
        R["overlap_restrictions"] = self._restr_result(spec, "overlap", rv)
        prov_open = self._open(spec, "provenance", rv)
        if prov_open and R["provenance_review"].status == PASS:
            R["provenance_review"] = CheckResult(PENDING, "open provenance restrictions: " + ",".join(prov_open),
                                                 self._review_action(spec, "provenance", " --resolves " + " ".join(prov_open)))

    def _competition(self, spec, rv, R, license_ok: bool):
        basis = self.policy.basis(spec.family)
        cls = self.lp.classify(spec.evidence.license_id)
        if basis == "no_statement":
            R["competition_policy"] = CheckResult(PENDING, "no competition policy statement recorded",
                                                  "record competition_policy.statement + source_url")
        elif basis == "not_permitted":
            R["competition_policy"] = CheckResult(FAIL, f"family '{spec.family}' not permitted "
                                                        f"({self.policy.permitted_families})", "model not permitted")
        elif rv["competition"].status == "REJECTED":
            R["competition_policy"] = CheckResult(FAIL, rv["competition"].detail, "")
        elif not self.policy.source_url:
            R["competition_policy"] = CheckResult(PENDING, f"family '{spec.family}' permitted by the recorded statement "
                                                           "but source_url is missing",
                                                  "paste the rule URL into competition_policy.source_url")
        elif basis == "general_rule" and cls != "permissive" and not license_ok:
            R["competition_policy"] = CheckResult(PENDING, f"family '{spec.family}' is allowed only by the general open-"
                                                           f"license rule and its license is {cls}; needs an approved "
                                                           "license review first", self._review_action(spec, "license"))
        else:
            R["competition_policy"] = CheckResult(PASS if basis == "named" else WARN,
                                                  f"family '{spec.family}' {'named by the organisers' if basis == 'named' else 'allowed by the general open-license rule'} "
                                                  f"({self.policy.source_url})")
        R["competition_restrictions"] = self._restr_result(spec, "competition", rv)

    # ------------------------------------------------------------------------ 7-8: checkpoint
    def _checkpoint(self, spec, snap, R, probe, verify_hashes):
        if not probe:
            for c in ("checkpoint_present", "revision_match", "manifest_files", "integrity"):
                R[c] = CheckResult(NOT_PROBED, "files not probed")
            return
        from .snapshots import download_commands, manifest_problems, manifest_status

        if snap.path is None:
            cmd = download_commands(spec)
            R["checkpoint_present"] = CheckResult(FAIL, snap.reason, cmd.get("project_cli") or cmd.get("note", ""))
            for c in ("revision_match", "manifest_files", "integrity"):
                R[c] = CheckResult(NOT_PROBED, "no snapshot")
            return
        R["checkpoint_present"] = CheckResult(PASS, f"{snap.origin} ({snap.kind}): {snap.path}")
        if spec.revision and snap.revision and snap.revision != spec.revision:
            R["revision_match"] = CheckResult(FAIL, f"snapshot revision {snap.revision} != pinned {spec.revision}",
                                              f"python -m wildspeech v3-install --model {spec.key} --execute")
        elif not snap.revision:
            R["revision_match"] = CheckResult(PENDING, "snapshot revision unknown (no manifest / metadata); reviews "
                                                       "cannot be tied to this checkpoint",
                                              f"python -m wildspeech v3-install --model {spec.key} --execute (writes the manifest)")
        else:
            R["revision_match"] = CheckResult(PASS, f"revision {snap.revision} ({snap.revision_source})")
        if spec.files:
            rows = manifest_status(snap.path, spec, verify_hashes, self.hash_cache)
            probs = manifest_problems(rows)
            sz = [r for r in rows if r["required"] and r["present"] and r["size_ok"] is not False]
            R["manifest_files"] = CheckResult(FAIL if any("missing" in p or "size" in p for p in probs) else PASS,
                                              "; ".join(p for p in probs if "sha256" not in p) or
                                              f"{len(sz)}/{len(rows)} manifest files present at the published size",
                                              f"python -m wildspeech v3-install --model {spec.key} --execute" if probs else "")
            hashed = [r for r in rows if r["expected_sha256"]]
            if not hashed:
                R["integrity"] = CheckResult(NA, "registry publishes no sha256 for this model")
            elif any(r["sha_ok"] is False for r in hashed):
                R["integrity"] = CheckResult(FAIL, "sha256 mismatch: " + ", ".join(r["file"] for r in hashed if r["sha_ok"] is False),
                                             "delete the corrupt file and re-install")
            elif all(r["sha_ok"] is True for r in hashed):
                R["integrity"] = CheckResult(PASS, "sha256 verified: " + ", ".join(r["substitute"] or r["file"] for r in hashed))
            else:
                R["integrity"] = CheckResult(WARN, "sha256 not verified in this check (size matches)",
                                             "python -m wildspeech v3-eligibility --verify-hashes")
        else:
            from .snapshots import detect_architecture as _a
            from .snapshots import missing_files

            arch = _a(snap.path)
            miss, rec = missing_files(snap.path, arch) if arch in ("hf_whisper", "hf_ctc", "hf_encoder", "qwen3_asr", "nemo") \
                else ([f"architecture '{arch}' not recognised from config.json"], [])
            R["manifest_files"] = CheckResult(FAIL if miss else (WARN if rec else PASS),
                                              ("missing " + "; ".join(miss)) if miss else
                                              (("recommended files missing: " + ", ".join(rec)) if rec else "all required files"),
                                              "re-download the snapshot (incomplete)" if miss else "")
            R["integrity"] = CheckResult(NA, "registry has no file manifest")

    # -------------------------------------------------------------------- 9-12: framework / GPU
    def _framework(self, spec, snap, arch, R, probe, deep):
        names = ("architecture_expected", "adapter_compatible", "remote_code", "components", "packages_importable",
                 "weights_loadable", "gpu_fit")
        if not probe or snap is None or snap.path is None:
            for c in names:
                R[c] = CheckResult(NOT_PROBED, "no local snapshot" if probe else "files not probed")
            return {}
        from .environment import component_check, framework_check, gpu_fit

        fr = framework_check(spec, arch, snap.path, self.cfg)
        rem = " | ".join(fr.remedies)
        exp = fr.checks.get("architecture_expected")
        R["architecture_expected"] = CheckResult(PASS if exp else (WARN if exp is None else FAIL),
                                                 f"detected {arch}; expected {spec.expected_architectures} "
                                                 f"({spec.expected_class or 'class not registered'})",
                                                 "" if exp is not False else "verify the checkpoint is the registered model")
        R["adapter_compatible"] = CheckResult(PASS if fr.checks.get("adapter_compatible") else FAIL,
                                              f"adapter {spec.adapter} / architecture {arch}")
        R["remote_code"] = CheckResult(PASS if fr.checks.get("remote_code") else FAIL,
                                       "no auto_map (no remote code)" if fr.checks.get("remote_code") else fr.detail,
                                       "" if fr.checks.get("remote_code") else rem)
        pk_ok = fr.checks.get("packages_importable")
        pk_detail = "; ".join(x for x in (fr.detail, "; ".join(fr.warnings)) if x)
        R["packages_importable"] = CheckResult(PASS if pk_ok and not fr.warnings else (WARN if pk_ok else FAIL),
                                               pk_detail or "ok", "" if pk_ok else (rem or "run scripts\\v3_install_asr_env.ps1"))
        wl = fr.checks.get("weights_loadable")
        R["weights_loadable"] = CheckResult(NA if wl is None else (PASS if wl else FAIL),
                                            "loaded by the adapter's backend" if wl is None else
                                            ("weights loadable" if wl else "weights not loadable: " + fr.detail),
                                            "" if wl in (None, True) else rem)
        if deep and pk_ok:
            cc = component_check(spec, snap.path, arch)
            R["components"] = CheckResult(PASS if cc["ok"] else FAIL, cc["problems"] or cc["notes"] or "ok",
                                          "" if cc["ok"] else "inspect the snapshot (wrong processor/tokenizer?)")
        elif deep:
            R["components"] = CheckResult(NOT_PROBED, "packages missing: processor/tokenizer not instantiated")
        else:
            R["components"] = CheckResult(NOT_PROBED, "deep check not requested",
                                          "python -m wildspeech v3-eligibility --deep")
        g = gpu_fit(spec, self.env, self.cfg)
        if g["cuda"] and g["fits"] is not False:
            R["gpu_fit"] = CheckResult(PASS, f"needs ~{g['estimated_need_gb']} GB of {g['usable_gb']} GB usable "
                                             f"({self.env.get('gpu_name')})")
        elif g["cuda"]:
            R["gpu_fit"] = CheckResult(FAIL, f"needs ~{g['estimated_need_gb']} GB > {g['usable_gb']} GB usable",
                                       "lower batch size / precision, or use a larger GPU")
        elif g["cpu_feasible"]:
            R["gpu_fit"] = CheckResult(WARN, "no CUDA: CPU inference feasible but slow")
        else:
            R["gpu_fit"] = CheckResult(FAIL, "no CUDA and the model is not CPU-feasible", "run on the GPU machine")
        return g

    def _smoke(self, spec, snap, R):
        st = (self.runtime.get(spec.key) or {}).get("smoke") or {}
        if not st:
            R["smoke_test"] = CheckResult(NOT_PROBED, "no smoke test recorded",
                                          f"python -m wildspeech v3-smoke --model {spec.key}")
        elif snap is not None and st.get("revision") not in (None, snap.revision):
            R["smoke_test"] = CheckResult(NOT_PROBED, f"smoke test was for revision {st.get('revision')}",
                                          f"python -m wildspeech v3-smoke --model {spec.key}")
        elif st.get("ok"):
            R["smoke_test"] = CheckResult(PASS, f"smoke {st.get('n_clips')} clips ok at {st.get('finished')} "
                                                f"({st.get('clips_per_s', 0):.2f} clips/s, {st.get('dtype')})")
        else:
            R["smoke_test"] = CheckResult(FAIL, f"smoke test failed: {st.get('error', '')}"[:300],
                                          "inspect cache/v3/asr_smoke/<model>/_smoke.json")

    # ---------------------------------------------------------------------------------- main
    def check(self, spec: ASRModelSpec, probe_files: bool = True, deep: bool = False,
              verify_hashes: bool = False) -> EligibilityRecord:
        from .reviews import assess_reviews
        from .snapshots import detect_architecture as _arch
        from .snapshots import resolve_snapshot

        snap = resolve_snapshot(spec, self.cfg.get_path("paths.hf_cache_dir")) if probe_files else None
        arch = _arch(snap.path) if (snap is not None and snap.path) else "unknown"
        rev_for_reviews = spec.revision or (snap.revision if snap is not None else None)
        rv, invalid = assess_reviews(spec, self.reviews, sorted(self.allowed), rev_for_reviews, self.base_dir,
                                     self.today, self.lp)
        R: dict[str, CheckResult] = {}
        self._registry(spec, R)
        self._license(spec, rv, R)
        self._provenance(spec, rv, R)
        lic_ok = R["license_review"].status == PASS and R["license_restrictions"].status in (PASS, NA)
        self._competition(spec, rv, R, lic_ok)
        self._checkpoint(spec, snap, R, probe_files, verify_hashes)
        gpu = self._framework(spec, snap, arch, R, probe_files, deep)
        self._smoke(spec, snap, R)
        dims = {d: _worst(R[c].status for c in cs if c in R) for d, cs in DIMENSIONS.items()}
        groups = {g: GROUP_STATUS[_worst(RANK_KEY[dims[d]] for d in ds)] for g, ds in GROUPS.items()}
        warnings = [f"{c}: {r.detail}" for c, r in R.items() if r.status == WARN]
        legacy = [k for k, v in spec.legacy_flags.items() if v is True]
        if legacy:
            warnings.append(f"legacy registry flags ignored ({', '.join(legacy)}); approvals come only from "
                            "configs/v3_asr_reviews.yaml")
        reasons = [f"[{d}] {c}: {R[c].detail}" for d, cs in DIMENSIONS.items() for c in cs
                   if d in GATING and c in R and R[c].status in (FAIL, PENDING, NOT_PROBED)]
        for iv in invalid:
            reasons.append(f"[review] invalid entry #{iv['index']} ignored: {iv['problems']}")
        actions = list(dict.fromkeys(R[c].action for d, cs in DIMENSIONS.items() for c in cs
                                     if c in R and R[c].action and R[c].status in (FAIL, PENDING, WARN, NOT_PROBED)))
        ok = all(dims[d] == "PASS" for d in GATING) and probe_files
        evid = ("registry_validity", "license_documentation", "license_approval", "training_provenance",
                "overlap_assessment", "competition_permission")
        dl_ok = all(dims[d] == "PASS" for d in evid)
        life = self._lifecycle(spec, dims, R, snap, probe_files)
        rec = EligibilityRecord(spec.key, v3.ELIGIBLE if ok else v3.NOT_EXECUTED, reasons,
                                {c: r.status for c, r in R.items()},
                                snap.path if snap is not None else None, arch,
                                snap.revision if snap is not None else None, R, dims, actions, warnings, invalid,
                                snap.origin if snap is not None else None, life, groups, dl_ok,
                                snap.kind if snap is not None else None, gpu or {})
        if not probe_files and ok is False and all(dims[d] == "PASS" for d in evid):
            rec.reasons.append("files not probed")
        return rec

    def _lifecycle(self, spec, dims, R, snap, probe) -> str:
        rt = self.runtime.get(spec.key) or {}
        fail_dims = ("registry_validity", "license_documentation", "license_approval", "training_provenance",
                     "overlap_assessment", "competition_permission")
        if any(dims[d] == "FAIL" for d in fail_dims):
            return "INELIGIBLE"
        if any(dims[d] != "PASS" for d in ("registry_validity", "license_documentation", "license_approval",
                                           "competition_permission")):
            return "LEGAL_REVIEW_PENDING"
        if any(dims[d] != "PASS" for d in ("training_provenance", "overlap_assessment")):
            return "PROVENANCE_REVIEW_PENDING"
        if rt.get("downloading"):
            return "DOWNLOADING"
        if not probe or snap is None or snap.path is None:
            return "READY_FOR_DOWNLOAD"
        if any(dims[d] != "PASS" for d in ("checkpoint_availability", "checkpoint_files")):
            return "CHECKPOINT_INCOMPLETE"
        if any(dims[d] == "FAIL" for d in ("architecture_compatibility", "processor_compatibility",
                                           "framework_dependencies", "gpu_readiness")):
            return "FRAMEWORK_INCOMPATIBLE"
        if rt.get("running"):
            return "INFERENCE_RUNNING"
        if rt.get("complete"):
            return "INFERENCE_COMPLETE"
        if rt.get("last_failed") or R.get("smoke_test", CheckResult(NOT_PROBED)).status == FAIL:
            return "FAILED"
        return "READY_FOR_INFERENCE"

    def check_all(self, specs: dict[str, ASRModelSpec], keys: list[str] | None = None, probe_files: bool = True,
                  deep: bool = False, verify_hashes: bool = False) -> dict[str, EligibilityRecord]:
        out = {}
        for k in list(keys or specs):
            if k not in specs:
                out[k] = EligibilityRecord(k, v3.NOT_EXECUTED, [f"{k} not in the V3 ASR registry"],
                                           {"registry_valid": FAIL}, dimensions={d: "FAIL" for d in DIMENSIONS},
                                           actions=[f"add {k} to configs/v3_asr_models.yaml or remove it from v3.asr.models"],
                                           lifecycle="INELIGIBLE", groups={g: "FAIL" for g in GROUPS})
                continue
            out[k] = self.check(specs[k], probe_files, deep, verify_hashes)
        return out


RANK_KEY = {"PASS": PASS, "NOT_PROBED": NOT_PROBED, "PENDING": PENDING, "FAIL": FAIL}


def checker_for(ctx, with_runtime: bool = True, env: dict | None = None) -> EligibilityChecker:
    from .model_registry import document_for, proposals_path, reviews_path
    from .reviews import load_proposals, load_reviews

    doc = document_for(ctx)
    runtime = {}
    hc = {}
    if with_runtime:
        from .run_state import load_hash_cache, runtime_states

        runtime = runtime_states(ctx, list(doc.specs))
        hc = load_hash_cache(ctx)
    return EligibilityChecker(ctx.cfg, doc.allowed_licenses, doc.policy, load_reviews(reviews_path(ctx)),
                              base_dir=ctx.paths.root, license_policy=doc.license_policy, runtime=runtime, env=env,
                              proposals=load_proposals(proposals_path(ctx)), hash_cache=hc)
