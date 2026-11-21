"""V3 speech-model registry (schema 3): configs/v3_asr_models.yaml -> ASRModelSpec + CompetitionPolicy + LicensePolicy.

The registry only DESCRIBES models: official card URLs, pinned commit, exact file manifest (size, LFS sha256, role),
license reference, training sources, expected architecture / class and open restrictions. It holds no approval
switches. Approvals are manual review records (asr.reviews); asr.eligibility decides from evidence + reviews +
local files whether a model may be downloaded and run. Nothing here touches the network.

Schemas 1 and 2 still load (tests, archived files). Schema-1 boolean flags (license_verified, provenance_verified,
overlap_checked, competition_eligible) are parsed but IGNORED by the eligibility checker; it reports them as
`legacy flag ignored`, so a stale YAML `true` can never approve a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..config import load_yaml

ROLES = ("child_phonetic", "child_word", "general_word", "general_phonetic", "general_char", "frozen_encoder")
ADAPTERS = ("pasketti_phonetic", "pasketti_word", "whisper", "ctc", "encoder", "qwen3_asr")
OUTPUTS = ("words", "phones", "chars", "embeddings")
FAMILIES = ("whisper", "wav2vec2", "pasketti", "hubert", "wavlm")
ARCHITECTURES = ("hf_whisper", "hf_ctc", "hf_encoder", "qwen3_asr", "nemo")
FILE_ROLES = ("weights", "config", "processor", "tokenizer", "other")
REVIEW_CHECKS = ("license", "provenance", "overlap", "competition")
LEGACY_FLAGS = ("license_verified", "provenance_verified", "overlap_checked", "competition_eligible")
TEXT_OUTPUTS = ("words", "phones", "chars")

# adapter -> architectures it can drive
ADAPTER_ARCHITECTURES = {"whisper": ("hf_whisper",), "ctc": ("hf_ctc",), "encoder": ("hf_encoder", "hf_ctc"),
                         "qwen3_asr": ("qwen3_asr",),
                         "pasketti_phonetic": ("hf_ctc", "hf_whisper"),
                         "pasketti_word": ("qwen3_asr", "hf_whisper", "hf_ctc", "nemo")}


@dataclass
class Restriction:
    id: str
    check: str
    text: str
    blocking: bool = True

    @classmethod
    def parse(cls, d: dict, model_key: str, i: int) -> "Restriction":
        d = dict(d or {})
        return cls(id=str(d.get("id") or f"{model_key}-R{i}"), check=str(d.get("check", "provenance")).lower(),
                   text=str(d.get("text", "")).strip(), blocking=d.get("blocking", True) is not False)


@dataclass
class FileSpec:
    """One file of the pinned snapshot. sha256 is the LFS hash published by the Hub (weights, large tokenizers);
    small git files are verified by size only (their Hub id is a git blob sha1, not a content sha256)."""
    name: str
    size: int | None = None
    role: str = "other"
    sha256: str | None = None
    required: bool = True

    @classmethod
    def parse(cls, d) -> "FileSpec":
        if isinstance(d, str):
            return cls(name=d)
        d = dict(d or {})
        return cls(name=str(d["name"]), size=int(d["size"]) if d.get("size") is not None else None,
                   role=str(d.get("role", "other")), sha256=(str(d["sha256"]).lower() if d.get("sha256") else None),
                   required=d.get("required", True) is not False)


@dataclass
class ModelEvidence:
    evidence_status: str = "missing"
    model_card_url: str | None = None
    license_id: str | None = None
    license_url: str | None = None
    license_evidence: str | None = None
    training_sources: list[str] = field(default_factory=list)
    training_evidence_url: str | None = None
    enumerated_corpora: object = None              # True | False | "partial" | None (unknown)
    release_date: str | None = None
    architecture_evidence: str | None = None
    output_format: str | None = None
    reported_metrics: str | None = None
    notes: str | None = None
    links: dict = field(default_factory=dict)      # api_url, paper_url, announcement_url, collection_url ...

    @classmethod
    def parse(cls, d: dict | None) -> "ModelEvidence":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "links"}
        ts = known.get("training_sources")
        if isinstance(ts, str):
            known["training_sources"] = [ts]
        elif ts is None:
            known["training_sources"] = []
        if known.get("license_id") is not None:
            known["license_id"] = str(known["license_id"]).strip().lower() or None
        if known.get("release_date") is not None:
            known["release_date"] = str(known["release_date"])
        return cls(**known, links={k: v for k, v in d.items() if v})


@dataclass
class ASRModelSpec:
    key: str
    hf_id: str | None = None
    local_path: str | None = None
    local_dir: str | None = None
    role: str = "general_word"
    adapter: str = "whisper"
    output: str = "words"
    family: str | None = None
    priority: int = 99
    child_speech: object = None
    hf_id_status: str = "unverified"
    expected_architectures: list[str] = field(default_factory=list)
    expected_model_type: str | None = None
    expected_class: str | None = None
    expected_params_millions: float | None = None
    revision: str | None = None
    revision_date: str | None = None
    precision: str | None = None
    sample_rate: int = 16000
    max_clip_seconds: float | None = None
    phone_inventory: str | None = None
    encoder_layers: list[int] = field(default_factory=list)
    runtime_backend: str | None = None
    runtime_requirements: dict = field(default_factory=dict)
    reviewed_remote_code: dict = field(default_factory=dict)     # {file: sha256} of explicitly reviewed remote code
    batch_size: int | None = None
    batch_seconds: float | None = None
    files: list[FileSpec] = field(default_factory=list)
    default_local_dir: str | None = None          # <repo>/<checkpoint_root>/<local_dir or key>, set by the loader
    evidence: ModelEvidence = field(default_factory=ModelEvidence)
    restrictions: list[Restriction] = field(default_factory=list)
    legacy_flags: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- compatibility (schema-1 callers)
    @property
    def license(self) -> str | None:
        return self.evidence.license_id

    @property
    def training_data(self) -> str | None:
        return "; ".join(self.evidence.training_sources) or None

    @property
    def is_child_model(self) -> bool:
        return self.role.startswith("child_")

    @property
    def is_encoder(self) -> bool:
        return self.output == "embeddings"

    # ---------------------------------------------------------------------------------------- helpers
    def blocking_restrictions(self, check: str | None = None) -> list[Restriction]:
        return [r for r in self.restrictions if r.blocking and (check is None or r.check == check)]

    def candidate_dirs(self) -> list[str]:
        return [p for p in (self.local_path, self.default_local_dir) if p]

    def file(self, name: str) -> FileSpec | None:
        return next((f for f in self.files if f.name == name), None)

    def weight_files(self) -> list[FileSpec]:
        return [f for f in self.files if f.role == "weights"]

    def bin_only(self) -> bool:
        w = self.weight_files()
        return bool(w) and all(f.name.endswith(".bin") for f in w)

    def manifest_bytes(self) -> int:
        return int(sum(f.size or 0 for f in self.files))

    def as_row(self) -> dict:
        e = self.evidence
        return {"hf_id": self.hf_id, "hf_id_status": self.hf_id_status, "family": self.family, "priority": self.priority,
                "role": self.role, "adapter": self.adapter, "output": self.output, "child_speech": self.child_speech,
                "expected_architectures": ",".join(self.expected_architectures), "expected_class": self.expected_class,
                "pinned_revision": self.revision, "local_path": self.local_path, "default_local_dir": self.default_local_dir,
                "license": e.license_id, "license_url": e.license_url, "model_card_url": e.model_card_url,
                "training_sources": " | ".join(e.training_sources), "training_evidence_url": e.training_evidence_url,
                "evidence_status": e.evidence_status, "output_format": e.output_format,
                "manifest_files": len(self.files), "manifest_gb": round(self.manifest_bytes() / 1e9, 3),
                "weights_format": ",".join(sorted({Path(f.name).suffix.lstrip('.') for f in self.weight_files()})),
                "restrictions_open": ",".join(r.id for r in self.blocking_restrictions()),
                "legacy_flags_set": ",".join(k for k, v in self.legacy_flags.items() if v is True)}

    def evidence_row(self) -> dict:
        d = asdict(self.evidence)
        links = d.pop("links", {}) or {}
        d["training_sources"] = " | ".join(d.get("training_sources") or [])
        return {"model_key": self.key, "hf_id": self.hf_id, "revision": self.revision, **d,
                **{f"link_{k}": v for k, v in links.items()}}

    def evidence_record(self) -> dict:
        """JSON-serialisable evidence for model_evidence.json (facts + sources + open restrictions)."""
        return {"model_key": self.key, "hf_id": self.hf_id, "hf_id_status": self.hf_id_status, "family": self.family,
                "role": self.role, "priority": self.priority, "revision": self.revision,
                "revision_date": self.revision_date, "expected_class": self.expected_class,
                "expected_architectures": self.expected_architectures, "runtime_backend": self.runtime_backend,
                "runtime_requirements": self.runtime_requirements,
                "files": [asdict(f) for f in self.files], "evidence": asdict(self.evidence),
                "restrictions": [asdict(r) for r in self.restrictions]}

    def validate(self) -> list[str]:
        errs = []
        if self.role not in ROLES:
            errs.append(f"unknown role {self.role}")
        if self.adapter not in ADAPTERS:
            errs.append(f"unknown adapter {self.adapter}")
        if self.output not in OUTPUTS:
            errs.append(f"unknown output {self.output}")
        if self.family not in FAMILIES:
            errs.append(f"unknown or missing family {self.family!r} (one of {FAMILIES})")
        if not (self.hf_id or self.local_path or self.default_local_dir):
            errs.append("no hf_id / local_path")
        bad = [a for a in self.expected_architectures if a not in ARCHITECTURES]
        if bad:
            errs.append(f"unknown expected_architectures {bad}")
        for r in self.restrictions:
            if r.check not in REVIEW_CHECKS:
                errs.append(f"restriction {r.id}: unknown check {r.check}")
        for f in self.files:
            if f.role not in FILE_ROLES:
                errs.append(f"file {f.name}: unknown role {f.role}")
        if self.files and not self.weight_files():
            errs.append("file manifest lists no weights")
        if (self.output == "embeddings") != (self.role == "frozen_encoder"):
            errs.append("output 'embeddings' is reserved for role 'frozen_encoder' (and vice versa)")
        return errs


@dataclass
class LicensePolicy:
    permissive: list[str] = field(default_factory=list)
    conditional: dict = field(default_factory=dict)      # license_id -> {class, license_url, conditions:[{id,text}]}
    prohibited: list[str] = field(default_factory=list)
    rule: str | None = None
    source_url: str | None = None

    @classmethod
    def parse(cls, d: dict | None, allowed_fallback: list[str]) -> "LicensePolicy":
        d = dict(d or {})
        if not d:                                         # schema 1/2: allowed_licenses are all permissive
            return cls(permissive=list(allowed_fallback))
        cond = {str(k).lower(): dict(v or {}) for k, v in (d.get("conditional") or {}).items()}
        return cls(permissive=[str(x).lower() for x in (d.get("permissive") or [])], conditional=cond,
                   prohibited=[str(x).lower() for x in (d.get("prohibited") or [])],
                   rule=(str(d["rule"]).strip() if d.get("rule") else None), source_url=d.get("source_url"))

    def classify(self, license_id: str | None) -> str:
        """permissive | conditional | prohibited | unknown | missing"""
        lic = str(license_id or "").lower()
        if not lic:
            return "missing"
        if lic in self.prohibited:
            return "prohibited"
        if lic in self.permissive:
            return "permissive"
        if lic in self.conditional:
            return "conditional"
        return "unknown"

    def conditions(self, license_id: str | None) -> list[dict]:
        return list((self.conditional.get(str(license_id or "").lower()) or {}).get("conditions") or [])

    def condition_ids(self, license_id: str | None) -> list[str]:
        return [str(c.get("id")) for c in self.conditions(license_id) if c.get("id")]

    def allowed(self) -> list[str]:
        return sorted(set(self.permissive) | set(self.conditional))


@dataclass
class CompetitionPolicy:
    statement: str | None = None
    recorded_by: str | None = None
    source_url: str | None = None
    permitted_families: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    named_families: list[str] = field(default_factory=list)
    general_rule_families: list[str] = field(default_factory=list)
    competition: str | None = None
    rules_url: str | None = None
    retrieved: str | None = None
    data_source: str | None = None

    @classmethod
    def parse(cls, d: dict | None) -> "CompetitionPolicy":
        d = dict(d or {})
        fams = [str(f).lower() for f in (d.get("permitted_families") or [])]
        named = [str(f).lower() for f in (d.get("named_families") or fams)]
        return cls(statement=(str(d["statement"]).strip() or None) if d.get("statement") else None,
                   recorded_by=d.get("recorded_by"), source_url=d.get("source_url"), permitted_families=fams,
                   conditions=[str(c) for c in (d.get("conditions") or [])], named_families=named,
                   general_rule_families=[str(f).lower() for f in (d.get("general_rule_families") or [])],
                   competition=d.get("competition"), rules_url=d.get("rules_url"),
                   retrieved=str(d["retrieved"]) if d.get("retrieved") else None,
                   data_source=(str(d["data_source"]).strip() if d.get("data_source") else None))

    def permits(self, family: str | None) -> bool:
        return bool(self.statement) and str(family or "").lower() in set(self.permitted_families)

    def basis(self, family: str | None) -> str:
        """named | general_rule | not_permitted | no_statement"""
        f = str(family or "").lower()
        if not self.statement:
            return "no_statement"
        if f in set(self.named_families) and f in set(self.permitted_families):
            return "named"
        if f in set(self.general_rule_families) or f in set(self.permitted_families):
            return "general_rule"
        return "not_permitted"

    def as_row(self) -> dict:
        return {"competition": self.competition, "statement": self.statement, "recorded_by": self.recorded_by,
                "source_url": self.source_url, "rules_url": self.rules_url, "retrieved": self.retrieved,
                "permitted_families": ",".join(self.permitted_families), "named_families": ",".join(self.named_families),
                "general_rule_families": ",".join(self.general_rule_families), "data_source": self.data_source,
                "conditions": " | ".join(self.conditions)}


@dataclass
class RegistryDocument:
    specs: dict[str, ASRModelSpec]
    allowed_licenses: list[str]
    policy: CompetitionPolicy
    schema_version: int
    path: Path | None
    root: Path | None
    license_policy: LicensePolicy = field(default_factory=LicensePolicy)
    checkpoint_root: str = "models"


KNOWN = {"hf_id", "local_path", "local_dir", "role", "adapter", "output", "family", "priority", "child_speech",
         "hf_id_status", "expected_architectures", "expected_model_type", "expected_class", "expected_params_millions",
         "revision", "revision_date", "precision", "sample_rate", "max_clip_seconds", "phone_inventory",
         "encoder_layers", "runtime_backend", "runtime_requirements", "reviewed_remote_code", "batch_size",
         "batch_seconds"}


def _infer_family(m: dict) -> str | None:
    """Schema-1 files had no family; infer it from the hf id / adapter so old registries still validate."""
    s = f"{m.get('hf_id', '')} {m.get('adapter', '')}".lower()
    for fam, needles in (("pasketti", ("pasketti",)), ("whisper", ("whisper",)), ("wav2vec2", ("wav2vec2",)),
                         ("hubert", ("hubert",)), ("wavlm", ("wavlm",))):
        if any(n in s for n in needles):
            return fam
    return None


def parse_model(key: str, m: dict, root: Path | None, checkpoint_root: str) -> ASRModelSpec:
    m = dict(m or {})
    legacy = {b: m.pop(b, None) is True for b in LEGACY_FLAGS}          # only an explicit YAML `true` is recorded
    ev_raw = m.pop("evidence", None)
    if ev_raw is None:                                                  # schema 1: license / training_data inline
        ev_raw = {"license_id": m.get("license"), "training_sources": m.get("training_data"),
                  "evidence_status": "schema1_inline"}
    m.pop("license", None)
    m.pop("training_data", None)
    m.pop("license_ok_reason", None)
    restr = [Restriction.parse(r, key, i) for i, r in enumerate(m.pop("restrictions", None) or [])]
    files = [FileSpec.parse(f) for f in (m.pop("files", None) or [])]
    known = {k: m.pop(k) for k in list(m) if k in KNOWN}
    known.setdefault("family", _infer_family({**known}))
    ea = known.get("expected_architectures")
    known["expected_architectures"] = [str(a) for a in ([ea] if isinstance(ea, str) else (ea or []))]
    known["encoder_layers"] = [int(x) for x in (known.get("encoder_layers") or [])]
    known["runtime_requirements"] = dict(known.get("runtime_requirements") or {})
    known["reviewed_remote_code"] = dict(known.get("reviewed_remote_code") or {})
    if known.get("priority") is None:
        known["priority"] = 99
    known["priority"] = int(known["priority"])
    known["sample_rate"] = int(known.get("sample_rate") or 16000)
    for k in ("hf_id", "local_path", "revision", "local_dir"):
        if known.get(k) is not None:
            known[k] = str(known[k]).strip() or None
    if known.get("revision_date") is not None:
        known["revision_date"] = str(known["revision_date"])
    sub = known.get("local_dir") or key
    default_dir = str((Path(root) / checkpoint_root / sub).resolve()) if root is not None else None
    lp = known.get("local_path")
    if lp and root is not None and not Path(lp).is_absolute():
        known["local_path"] = str((Path(root) / lp).resolve())
    return ASRModelSpec(key=key, default_local_dir=default_dir, evidence=ModelEvidence.parse(ev_raw), files=files,
                        restrictions=restr, legacy_flags=legacy, extra=m, **known)


def load_registry_document(config_dir: Path, filename: str = "v3_asr_models.yaml") -> RegistryDocument:
    p = Path(config_dir) / filename
    raw = (load_yaml(p) if p.exists() else {}) or {}
    root = Path(config_dir).resolve().parent
    allowed = [str(x).strip().lower() for x in (raw.get("allowed_licenses") or [])]
    lp = LicensePolicy.parse(raw.get("license_policy"), allowed)
    allowed = sorted(set(allowed) | set(lp.allowed()))
    schema = int(raw.get("schema_version") or 1)
    cr = str(raw.get("checkpoint_root") or ("models" if schema >= 3 else "models/asr"))
    specs = {str(k): parse_model(str(k), m, root, cr) for k, m in (raw.get("models") or {}).items()}
    return RegistryDocument(specs, allowed, CompetitionPolicy.parse(raw.get("competition_policy")), schema,
                            p if p.exists() else None, root, lp, cr)


def load_asr_registry(config_dir: Path, filename: str = "v3_asr_models.yaml") -> tuple[dict[str, ASRModelSpec], list[str]]:
    """-> ({key: spec}, allowed_licenses) (kept for schema-1 callers)."""
    doc = load_registry_document(config_dir, filename)
    return doc.specs, doc.allowed_licenses


def registry_file(ctx) -> str:
    return ctx.cfg.get_path("v3.asr.registry_file", "v3_asr_models.yaml")


def reviews_path(ctx) -> Path:
    return Path(ctx.paths.configs) / ctx.cfg.get_path("v3.asr.reviews_file", "v3_asr_reviews.yaml")


def proposals_path(ctx) -> Path:
    return Path(ctx.paths.configs) / ctx.cfg.get_path("v3.asr.proposals_file", "v3_asr_review_proposals.yaml")


def document_for(ctx) -> RegistryDocument:
    """Re-read on every call (a few KB): edits made while a notebook session is open are always picked up."""
    return load_registry_document(ctx.paths.configs, registry_file(ctx))


def registry_for(ctx) -> tuple[dict[str, ASRModelSpec], list[str]]:
    doc = document_for(ctx)
    return doc.specs, doc.allowed_licenses


def by_priority(keys, specs: dict[str, ASRModelSpec]) -> list[str]:
    return sorted(keys, key=lambda k: (getattr(specs.get(k), "priority", 99), k))


def preferred_models(specs: dict[str, ASRModelSpec], available: list[str]) -> dict[str, str | None]:
    """Primary phoneme / word recogniser and encoder among models WITH accepted outputs (priority order):
    phonemes: pasketti_phonetic first, then other phone-output models; words: whisper_small_en first, then others."""
    av = [k for k in by_priority(available, specs) if k in specs]
    ph = [k for k in av if specs[k].output == "phones"]
    wd = [k for k in av if specs[k].output == "words"]
    en = [k for k in av if specs[k].output == "embeddings"]
    return {"primary_phonetic": ph[0] if ph else None, "primary_word": wd[0] if wd else None,
            "primary_encoder": en[0] if en else None}


def registry_table(specs: dict[str, ASRModelSpec], eligibility: dict | None = None) -> pd.DataFrame:
    rows = []
    for k, s in specs.items():
        r = {"model_key": k, **s.as_row()}
        e = (eligibility or {}).get(k)
        if e is not None:
            r.update(e.as_row())
        rows.append(r)
    return pd.DataFrame(rows)


def evidence_table(specs: dict[str, ASRModelSpec]) -> pd.DataFrame:
    return pd.DataFrame([s.evidence_row() for s in specs.values()])


def files_table(specs: dict[str, ASRModelSpec]) -> pd.DataFrame:
    return pd.DataFrame([{"model_key": k, **asdict(f)} for k, s in specs.items() for f in s.files],
                        columns=["model_key", "name", "size", "role", "sha256", "required"])


def restrictions_table(specs: dict[str, ASRModelSpec]) -> pd.DataFrame:
    return pd.DataFrame([{"model_key": k, **asdict(r)} for k, s in specs.items() for r in s.restrictions],
                        columns=["model_key", "id", "check", "text", "blocking"])
