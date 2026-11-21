"""External (Colab-generated) ASR predictions -> canonical WildSpeech ASR schema, with an independent import gate.

The export's schema is NOT assumed. inspect_asr_schema() detects, from column names, dtypes and values:
    layout          long (one row per model x clip, a model column) | wide (one transcript column per model)
    roles           model id / version, clip id, audio filename / basename / member, split, transcript, phones,
                    confidence, token log-probabilities, inference status, error, timing, audio duration, sample
                    rate, task, provenance fields (timestamp, run id, device, ...)
    models          model ids + row counts, examples per role, null fractions
An explicit v4.asr_import.column_map / wide_model_columns overrides any detection. The report is written to
asr_import/asr_schema_report.json BEFORE anything else, so an ambiguous export can be mapped by hand.

canonicalize() returns one row per (model, source row) with
    model_id, model_version, model_key       identity (model_key = sanitized id[+version if ids repeat])
    clip_ref, audio_ref, filename_norm, split_src
    transcript (None = no prediction, "" = recognised silence), phones, has_phones
    asr_ok, status_raw, error, failure_kind  failed inference is KEPT (asr_ok=False), never silently dropped
    confidence, mean_token_logprob, min_token_logprob, has_token_probs, token_logprobs (JSON)
    decode_seconds, audio_seconds, sample_rate, task_src
    source_row, src__<original column>       every original field of the row (wide: shared + that model's)
resolve_duplicates() keeps one row per (model_key, split, filename_norm) by an explicit policy and returns every
discarded duplicate in a separate table (asr_import/duplicates.parquet).

Provenance: `external_colab_asr` + SHA-256 of the source file + model identity + import time + permission review.
These outputs are written to cache/v4/asr_external (never cache/v3/asr) and are never relabelled producer=P24;
asr.outputs._origin() recognises the provenance type and keeps them out of the P25-P29 gate.

validate_import() is the independent gate: permission review recorded, source hash present, one-to-one joins,
per-split coverage >= v4.asr_import.min_model_coverage, >= 1 non-empty transcript, test coverage when a submission
is requested. Models failing a check are reported and excluded; nothing is relabelled.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import FILE, ID
from . import EXTERNAL_PROVENANCE

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "model_id": ("model_id", "model", "model_name", "asr_model", "model_key", "recognizer", "recogniser", "engine",
                 "system", "asr_system", "checkpoint_name", "hf_model", "model_repo"),
    "model_version": ("model_version", "version", "revision", "model_revision", "commit", "checkpoint", "model_sha"),
    "clip_id": ("clip_id", "uid", "utt_id", "utterance_id", "id", "clip_uid", "row_id"),
    "filename": ("filename", "file_name", "audio_name", "audio_filename", "file", "audio", "wav", "flac"),
    "basename": ("basename", "base_name", "stem"),
    "member": ("member", "tar_member", "archive_member", "path", "audio_path", "filepath", "file_path", "relpath"),
    "split": ("split", "split_hint", "subset", "partition", "dataset_split"),
    "source": ("source", "archive", "tar", "tar_name", "source_archive"),
    "source_type": ("source_type", "origin_type"),
    "transcript": ("transcript", "transcription", "text", "prediction", "pred_text", "predicted_text", "hypothesis",
                   "hyp", "asr_text", "pred", "output_text", "decoded_text", "result_text"),
    "phones": ("phones", "phonemes", "phoneme", "phone_sequence", "pred_phones", "phoneme_transcript", "ipa",
               "phonetic", "phone_string", "phonemes_pred"),
    "confidence": ("confidence", "avg_logprob", "mean_logprob", "asr_confidence", "score", "logprob", "conf",
                   "avg_log_prob", "sequence_score"),
    "token_logprobs": ("token_logprobs", "token_log_probs", "token_probs", "tokens_logprob", "token_scores",
                       "word_confidences", "token_confidences"),
    "status": ("status", "inference_status", "success", "ok", "state", "result_status"),
    "error": ("error", "error_message", "exception", "failure_reason", "err", "error_msg"),
    "decode_seconds": ("decode_seconds", "inference_time", "inference_seconds", "latency", "latency_s", "elapsed",
                       "elapsed_s", "runtime_s", "time_s", "processing_time", "duration_ms", "inference_ms", "rtf_time"),
    "audio_seconds": ("audio_seconds", "audio_duration", "audio_duration_s", "duration_s", "duration_sec", "duration",
                      "audio_len_s", "length_s"),
    "sample_rate": ("sample_rate", "sr", "sampling_rate", "audio_sr", "file_sr"),
    "task": ("task", "task_name", "subtest"),
    "expected_text": ("expected_text", "reference", "ref_text", "prompt"),
    "timestamp": ("timestamp", "created_at", "time", "datetime", "inferred_at", "run_time"),
    "run_id": ("run_id", "job_id", "session_id", "batch_id"),
    "device": ("device", "gpu", "hardware", "accelerator"),
}
PROVENANCE_ROLES = ("timestamp", "run_id", "device", "source", "source_type")
TRANSCRIPT_TOKENS = ("transcript", "transcription", "text", "prediction", "pred", "hypothesis", "hyp", "output")
AUX_TOKENS = {"phones": ("phones", "phonemes", "phoneme", "ipa"), "confidence": ("confidence", "logprob", "score", "conf"),
              "token_logprobs": ("token_logprobs", "token_probs", "token_scores"),
              "status": ("status", "success", "ok"), "error": ("error", "exception"),
              "decode_seconds": ("time", "latency", "elapsed", "seconds", "_ms")}
LABEL_FIELDS = ("outcome_score", "item_score", "score_status", "school_id")


def sanitize_key(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()
    return re.sub(r"_+", "_", s) or "model"


def normalize_clip_name(name, exts=(".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a")) -> str:
    """Archive-relative path / basename / with-or-without extension -> lowercase stem.

    'train_audio/ABC_01.FLAC' -> 'abc_01'; 'C:\\x\\abc_01.wav' -> 'abc_01'; 'abc_01' -> 'abc_01'."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).strip().replace("\\", "/")
    s = s.split("?")[0].rstrip("/")
    s = s.split("/")[-1]
    low = s.lower()
    for e in sorted(exts, key=len, reverse=True):
        if low.endswith(e):
            s = s[: -len(e)]
            break
    return s.strip().lower()


def normalize_split(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if re.search(r"(^|[^a-z])(train|training|trn)([^a-z]|$)", s) or s.startswith("train"):
        return "train"
    if re.search(r"(^|[^a-z])(test|testing|tst|eval)([^a-z]|$)", s) or s.startswith("test"):
        return "test"
    return None


# ================================================================================ inspection
def _find(cols_lower: dict[str, str], aliases) -> str | None:
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None


def _example(s: pd.Series, k: int = 3) -> list:
    try:
        v = s.dropna()
    except Exception:  # noqa: BLE001 - array-valued cells
        v = s
    return [str(x)[:120] for x in v.head(k).tolist()]


def _nunique(s: pd.Series) -> int | None:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:                                  # list / array cells are unhashable
        try:
            return int(s.map(lambda x: str(x)).nunique())
        except Exception:  # noqa: BLE001
            return None


def inspect_asr_schema(df: pd.DataFrame, column_map: dict | None = None, wide_model_columns: dict | None = None) -> dict:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    roles = {}
    for role, aliases in ROLE_ALIASES.items():
        hit = _find(cols_lower, aliases)
        if hit is not None:
            roles[role] = hit
    for role, col in (column_map or {}).items():                 # explicit mapping wins
        if col in df.columns:
            roles[role] = col
        else:
            raise KeyError(f"v4.asr_import.column_map[{role!r}] = {col!r} is not a column of the export")
    # a numeric/boolean 'score' is a confidence only if it is not also the item score
    if roles.get("confidence") and not pd.api.types.is_numeric_dtype(df[roles["confidence"]]):
        roles.pop("confidence")
    layout, wide = "long", {}
    if wide_model_columns:
        layout, wide = "wide", {str(m): c for m, c in wide_model_columns.items()}
    elif "model_id" not in roles:
        wide = detect_wide_models(df, roles)
        if len(wide) >= 1 and "transcript" not in roles:
            layout = "wide"
        elif len(wide) >= 2:
            layout = "wide"
        else:
            layout = "long_single_model"
    if layout == "wide":
        roles.pop("transcript", None)
    models = []
    if layout == "long" and "model_id" in roles:
        vc = df[roles["model_id"]].astype(str).value_counts()
        models = [{"model_id": k, "rows": int(v)} for k, v in vc.items()]
    elif layout == "wide":
        models = [{"model_id": m, "rows": int(df[c].notna().sum()), "transcript_column": c} for m, c in wide.items()]
    else:
        models = [{"model_id": "single_model", "rows": int(len(df))}]
    report = {
        "n_rows": int(len(df)), "n_columns": int(df.shape[1]), "layout": layout, "roles": roles,
        "wide_model_columns": wide, "models": models,
        "columns": [{"column": str(c), "dtype": str(df[c].dtype), "null_frac": float(df[c].isna().mean()),
                     "n_unique": _nunique(df[c]), "examples": _example(df[c])} for c in df.columns],
        "label_fields_present": [c for c in df.columns if str(c).lower() in LABEL_FIELDS],
        "has_phones": "phones" in roles or any(_wide_aux(df, m, c, "phones") for m, c in wide.items()),
        "has_confidence": "confidence" in roles or "token_logprobs" in roles,
        "has_status": "status" in roles, "has_error": "error" in roles,
        "identity_roles": [r for r in ("clip_id", "filename", "basename", "member") if r in roles],
        "warnings": [],
    }
    if not report["identity_roles"]:
        report["warnings"].append("no clip identity column (filename/basename/member/clip_id) detected: set "
                                  "v4.asr_import.column_map")
    if layout == "long" and "transcript" not in roles and "phones" not in roles:
        report["warnings"].append("no transcript or phone column detected: set v4.asr_import.column_map")
    if report["label_fields_present"]:
        report["warnings"].append(f"export contains label/train-only fields {report['label_fields_present']}: they are "
                                  "ignored (never used as ASR features)")
    if "clip_id" in roles and not any(r in roles for r in ("filename", "basename", "member")):
        report["warnings"].append("only a clip_id column identifies clips: it is joined to the audio manifest's "
                                  "clip_id (never assumed to equal a metadata filename or subject_id)")
    return report


def detect_wide_models(df: pd.DataFrame, roles: dict) -> dict[str, str]:
    """Columns like '<model>_text' / 'text_<model>' / '<model>__transcript' (string dtype) -> {model: column}."""
    used = set(roles.values())
    out = {}
    for c in df.columns:
        if c in used or not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        name = str(c).lower()
        for tok in TRANSCRIPT_TOKENS:
            m = re.match(rf"^(.+?)[_\-.:]+{tok}s?$", name) or re.match(rf"^{tok}s?[_\-.:]+(.+)$", name)
            if m:
                model = m.group(1).strip("_-.: ")
                if model and model not in ("expected", "reference", "ref", "target", "item", "prompt"):
                    out[model] = c
                break
    return out


def _wide_aux(df: pd.DataFrame, model: str, transcript_col: str, role: str) -> str | None:
    m = model.lower()
    for c in df.columns:
        if c == transcript_col:
            continue
        n = str(c).lower()
        if m in n and any(t in n.replace(m, "") for t in AUX_TOKENS.get(role, ())):
            return c
    return None


# ================================================================================ canonical form
def _status_ok(raw: pd.Series, success_values, failure_values) -> pd.Series:
    """True / False / NaN(unknown) from a status column (strings, booleans or 0/1)."""
    s = raw.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.map(lambda v: 1.0 if v == 1 else (0.0 if v == 0 else np.nan)).astype(float)
    t = s.astype(str).str.strip().str.lower()
    ok = t.isin([str(v).lower() for v in success_values])
    bad = t.isin([str(v).lower() for v in failure_values])
    out = pd.Series(np.nan, index=s.index)
    out[ok] = 1.0
    out[bad] = 0.0
    return out


def _token_stats(v) -> tuple[float, float, bool, str]:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return np.nan, np.nan, False, ""
    arr = v
    if isinstance(v, str):
        try:
            arr = json.loads(v)
        except Exception:  # noqa: BLE001 - not JSON: "a b c" of floats
            try:
                arr = [float(x) for x in re.split(r"[\s,;]+", v.strip()) if x]
            except ValueError:
                return np.nan, np.nan, False, ""
    try:
        a = np.asarray(list(arr), float).ravel()
    except Exception:  # noqa: BLE001 - nested / non-numeric structure
        return np.nan, np.nan, False, ""
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, False, ""
    if a.min() >= 0 and a.max() <= 1:          # probabilities -> log-probabilities
        a = np.log(np.clip(a, 1e-12, 1.0))
    return float(a.mean()), float(a.min()), True, json.dumps([round(float(x), 5) for x in a])


def _str_or_none(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(map(str, v))
    return str(v)


def to_long(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Uniform long frame: one row per (model, source row); every original field kept as src__<col>."""
    roles = dict(schema["roles"])
    base = df.reset_index(drop=True)
    if schema["layout"] == "wide":
        wide = schema["wide_model_columns"]
        model_cols = set(wide.values())
        aux_all = {}
        for m, c in wide.items():
            aux_all[m] = {r: _wide_aux(base, m, c, r) for r in AUX_TOKENS}
            model_cols |= {v for v in aux_all[m].values() if v}
        shared = [c for c in base.columns if c not in model_cols]
        parts = []
        for m, c in wide.items():
            d = pd.DataFrame({"model_id": m, "source_row": np.arange(len(base))})
            # shared roles first (identity, split, task, shared status ...), then this model's own columns win
            for r, col in roles.items():
                if r not in ("transcript", "model_id") and col in base.columns:
                    d[r] = base[col].to_numpy()
            d["transcript"] = base[c].map(_str_or_none).to_numpy()
            for r, col in aux_all[m].items():
                if col:
                    d[r] = base[col].to_numpy()
            for col in shared:
                d[f"src__{col}"] = base[col].to_numpy()
            d[f"src__{c}"] = base[c].to_numpy()
            for col in {v for v in aux_all[m].values() if v}:
                d[f"src__{col}"] = base[col].to_numpy()
            parts.append(d)
        long = pd.concat(parts, ignore_index=True)
    else:
        long = pd.DataFrame({"source_row": np.arange(len(base))})
        long["model_id"] = base[roles["model_id"]].astype(str).to_numpy() if "model_id" in roles else "single_model"
        for r, col in roles.items():
            if r == "model_id":
                continue
            long[r] = base[col].to_numpy()
        for col in base.columns:
            long[f"src__{col}"] = base[col].to_numpy()
    return long


def canonicalize(long: pd.DataFrame, cfg4: dict) -> pd.DataFrame:
    imp = cfg4.get("asr_import", {}) or {}
    exts = tuple((cfg4.get("joins", {}) or {}).get("audio_extensions", (".flac", ".wav")))
    c = long.copy()
    c["model_id"] = c["model_id"].astype(str)
    c["model_version"] = c["model_version"].astype(str) if "model_version" in c else "unknown"
    c["model_version"] = c["model_version"].replace({"nan": "unknown", "None": "unknown", "": "unknown"})
    n_ver = c.groupby("model_id")["model_version"].nunique()
    multi = set(n_ver[n_ver > 1].index)
    c["model_key"] = [sanitize_key(f"{m}_{v}" if m in multi else m) for m, v in zip(c["model_id"], c["model_version"])]
    # identity
    ref_cols = [r for r in ("filename", "basename", "member") if r in c]
    c["audio_ref"] = None
    for r in ref_cols:
        c["audio_ref"] = c["audio_ref"].where(c["audio_ref"].notna(), c[r].map(_str_or_none))
    c["filename_norm"] = c["audio_ref"].map(lambda s: normalize_clip_name(s, exts))
    c["clip_ref"] = c["clip_id"].map(_str_or_none) if "clip_id" in c else None
    split = pd.Series(None, index=c.index, dtype=object)
    for r in ("split", "source", "member", "source_type"):
        if r in c:
            split = split.where(split.notna(), c[r].map(normalize_split))
    c["split_src"] = split
    # outputs
    c["transcript"] = c["transcript"].map(_str_or_none) if "transcript" in c else None
    c["phones"] = c["phones"].map(_str_or_none) if "phones" in c else None
    c["has_phones"] = c["phones"].fillna("").astype(str).str.strip().ne("") if "phones" in c else False
    st = _status_ok(c["status"], imp.get("success_values", ["ok"]), imp.get("failure_values", ["error"])) \
        if "status" in c else pd.Series(np.nan, index=c.index)
    err = c["error"].map(_str_or_none).fillna("") if "error" in c else pd.Series("", index=c.index)
    has_out = c["transcript"].notna() | c["has_phones"].astype(bool)
    ok = st.copy()
    ok[ok.isna()] = (has_out & err.str.strip().eq(""))[ok.isna()].astype(float)
    ok = ok.astype(bool) & has_out                                 # a "success" without any output is a failure
    c["asr_ok"] = ok
    c["status_raw"] = c["status"].map(_str_or_none) if "status" in c else None
    c["error"] = err
    c["failure_kind"] = np.select([ok, ~has_out & err.str.strip().ne(""), ~has_out, st.eq(0.0)],
                                  ["", "error", "no_output", "status_failed"], default="status_failed")
    # numeric fields
    for r in ("confidence", "audio_seconds", "sample_rate"):
        c[r] = pd.to_numeric(c[r], errors="coerce") if r in c else np.nan
    if "decode_seconds" in c:
        c["decode_seconds"] = pd.to_numeric(c["decode_seconds"], errors="coerce")
    else:
        c["decode_seconds"] = np.nan
    ts = c["token_logprobs"].map(_token_stats) if "token_logprobs" in c else pd.Series([(np.nan, np.nan, False, "")] * len(c), index=c.index)
    c["mean_token_logprob"] = [t[0] for t in ts]
    c["min_token_logprob"] = [t[1] for t in ts]
    c["has_token_probs"] = [t[2] for t in ts]
    c["token_logprobs"] = [t[3] for t in ts]
    c["task_src"] = c["task"].astype(str).str.strip().str.lower() if "task" in c else None
    c["provenance_type"] = EXTERNAL_PROVENANCE
    keep = ["model_key", "model_id", "model_version", "source_row", "clip_ref", "audio_ref", "filename_norm",
            "split_src", "transcript", "phones", "has_phones", "asr_ok", "status_raw", "error", "failure_kind",
            "confidence", "mean_token_logprob", "min_token_logprob", "has_token_probs", "token_logprobs",
            "decode_seconds", "audio_seconds", "sample_rate", "task_src", "provenance_type"]
    keep += [r for r in PROVENANCE_ROLES if r in c and r not in keep]
    src = [k for k in c.columns if str(k).startswith("src__")]
    return c[keep + src].reset_index(drop=True)


def ms_to_seconds(canon: pd.DataFrame, schema: dict) -> pd.DataFrame:
    col = str(schema["roles"].get("decode_seconds", "")).lower()
    if col.endswith("ms") or "_ms" in col or "millis" in col:
        canon = canon.copy()
        canon["decode_seconds"] = canon["decode_seconds"] / 1000.0
    return canon


def resolve_duplicates(canon: pd.DataFrame, key_cols=("model_key", "split", FILE),
                       policy: str = "prefer_success_then_latest") -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per key; discarded duplicates are returned (never silently dropped)."""
    key_cols = [k for k in key_cols if k in canon.columns]
    if canon.empty or not key_cols:
        return canon, canon.iloc[0:0]
    c = canon.copy()
    c["_ok"] = c["asr_ok"].astype(int)
    c["_nonempty"] = c["transcript"].fillna("").astype(str).str.strip().ne("").astype(int)
    ts = pd.to_datetime(c["timestamp"], errors="coerce", utc=True) if "timestamp" in c else pd.Series(pd.NaT, index=c.index)
    c["_ts"] = [int(t.value) if pd.notna(t) else 0 for t in ts]
    if policy == "prefer_success_then_latest":
        order = ["_ok", "_nonempty", "_ts", "source_row"]
    elif policy == "latest":
        order = ["_ts", "source_row"]
    elif policy == "first":
        c["_neg"] = -c["source_row"]
        order = ["_neg"]
    else:
        raise ValueError(f"unknown duplicate_policy {policy}")
    c = c.sort_values(key_cols + order, ascending=[True] * len(key_cols) + [False] * len(order), kind="mergesort")
    dup = c.duplicated(key_cols, keep="first")
    helpers = [h for h in ("_ok", "_nonempty", "_ts", "_neg") if h in c]
    kept = c[~dup].drop(columns=helpers).sort_values("source_row", kind="mergesort").reset_index(drop=True)
    gone = c[dup].drop(columns=helpers).reset_index(drop=True)
    if not gone.empty:
        gone["duplicate_reason"] = "duplicate model x clip prediction; kept row by policy " + policy
    return kept, gone


# ================================================================================ import
@dataclass
class ImportResult:
    canonical: pd.DataFrame
    schema: dict
    provenance: dict
    notes: list = field(default_factory=list)


def read_export(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    if path.suffix.lower() in (".jsonl", ".json"):
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise ValueError(f"unsupported ASR export format {path.suffix}")


def import_external_asr(path: Path, cfg4: dict, logger=None) -> ImportResult:
    from .artifacts import file_sha256

    imp = cfg4.get("asr_import", {}) or {}
    raw = read_export(path)
    schema = inspect_asr_schema(raw, imp.get("column_map") or {}, imp.get("wide_model_columns") or {})
    long = to_long(raw, schema)
    canon = ms_to_seconds(canonicalize(long, cfg4), schema)
    st = Path(path).stat()
    prov = {"provenance_type": EXTERNAL_PROVENANCE, "producer": EXTERNAL_PROVENANCE, "source_file": str(path),
            "source_name": Path(path).name, "source_bytes": int(st.st_size), "source_mtime": int(st.st_mtime),
            "source_sha256": file_sha256(path), "imported": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "layout": schema["layout"], "roles": schema["roles"], "n_source_rows": int(len(raw)),
            "models": sorted(canon["model_key"].unique().tolist()),
            "model_identity": canon.groupby("model_key")[["model_id", "model_version"]].first().reset_index().to_dict("records"),
            "permission_review": imp.get("permission_review", {}),
            "statement": "Externally generated ASR predictions imported by WildSpeech V4. Not produced by local P24 "
                         "inference and never labelled as such."}
    if logger:
        logger.info("ASR import: %d source rows, layout=%s, models=%s", len(raw), schema["layout"], prov["models"])
    return ImportResult(canon, schema, prov)


# ================================================================================ gate
def permission_status(cfg4: dict) -> tuple[bool, str]:
    pr = (cfg4.get("asr_import", {}) or {}).get("permission_review", {}) or {}
    if not pr.get("reviewed"):
        return False, "v4.asr_import.permission_review.reviewed is false"
    miss = [k for k in ("reviewer", "date", "statement") if not str(pr.get(k, "")).strip()]
    if miss:
        return False, f"permission review incomplete: missing {miss}"
    return True, f"reviewed by {pr.get('reviewer')} on {pr.get('date')}"


def validate_import(joined: pd.DataFrame, coverage: pd.DataFrame, provenance: dict, cfg4: dict,
                    splits=("train", "test"), need_test: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """-> (gate table: model_key, check, status PASS/FAIL/WARN, detail; usable model keys)."""
    imp = cfg4.get("asr_import", {}) or {}
    min_cov = float(imp.get("min_model_coverage", 0.5))
    allowed = set(map(str, (imp.get("permission_review", {}) or {}).get("models") or []))
    perm_ok, perm_detail = permission_status(cfg4)
    rows = []
    models = sorted(joined["model_key"].unique()) if not joined.empty else []
    for m in models:
        def add(check, ok, detail, warn=False):
            rows.append({"model_key": m, "check": check, "status": "PASS" if ok else ("WARN" if warn else "FAIL"),
                         "detail": detail})

        mid = str(joined.loc[joined["model_key"] == m, "model_id"].iloc[0])
        add("permission_review", perm_ok and (not allowed or m in allowed or mid in allowed),
            perm_detail if perm_ok else perm_detail + " (ASR families are NOT_EXECUTED until reviewed)")
        add("provenance_external", provenance.get("provenance_type") == EXTERNAL_PROVENANCE and bool(provenance.get("source_sha256")),
            f"sha256={str(provenance.get('source_sha256'))[:16]}")
        d = joined[joined["model_key"] == m]
        dups = int(d.duplicated(["split", FILE]).sum())
        add("one_prediction_per_clip", dups == 0, f"{dups} duplicate (split, clip) rows after resolution")
        for s in splits:
            cv = coverage[(coverage["model_key"] == m) & (coverage["split"] == s)]
            if cv.empty:
                add(f"coverage_{s}", s == "train" and False, f"no {s} predictions", warn=(s == "test" and not need_test))
                continue
            r = cv.iloc[0]
            frac = float(r["usable_frac"])
            add(f"coverage_{s}", frac >= min_cov, f"usable {int(r['n_usable'])}/{int(r['n_metadata_clips'])} = {frac:.3f} "
                                                 f"(min {min_cov}); failures={int(r['n_failed'])}",
                warn=(s == "test" and not need_test))
        add("nonempty_output", bool((d["transcript"].fillna("").str.strip().ne("") | d["has_phones"]).any()),
            f"{int(d['transcript'].fillna('').str.strip().ne('').sum())} non-empty transcripts, "
            f"{int(d['has_phones'].sum())} phone strings")
    gate = pd.DataFrame(rows, columns=["model_key", "check", "status", "detail"])
    if gate.empty:
        return gate, []
    bad = set(gate.loc[gate["status"] == "FAIL", "model_key"])
    return gate, [m for m in models if m not in bad]


# ================================================================================ store
def write_external_store(root: Path, joined: pd.DataFrame, provenance: dict, gate: pd.DataFrame) -> pd.DataFrame:
    """cache/v4/asr_external/<model_key>/<split>/part-00000.parquet (FILE + existing ASR_COLUMNS + identity)
    + _provenance.json (external_colab_asr). Written fresh for each import fingerprint."""
    from ..acoustic.store import FeatureStore
    from ..asr.base import ASR_COLUMNS
    from ..io_utils import stable_hash, write_json

    store = FeatureStore(Path(root))
    fp = stable_hash({"sha": provenance.get("source_sha256"), "roles": provenance.get("roles")})
    out = []
    for (m, s), d in joined.groupby(["model_key", "split"]):
        t = pd.DataFrame({FILE: d[FILE].astype(str).to_numpy()})
        t["transcript"] = d["transcript"].fillna("").to_numpy()
        t["phones"] = d["phones"].fillna("").to_numpy() if "phones" in d else ""
        t["raw_tokens"] = ""
        t["token_logprobs"] = d["token_logprobs"].fillna("").to_numpy()
        t["has_token_probs"] = d["has_token_probs"].astype(bool).to_numpy()
        t["mean_token_logprob"] = d["mean_token_logprob"].to_numpy(float)
        t["min_token_logprob"] = d["min_token_logprob"].to_numpy(float)
        for col in ASR_COLUMNS:
            if col not in t:
                t[col] = np.nan
        t["audio_seconds"] = d["audio_seconds"].to_numpy(float)
        t["decode_seconds"] = d["decode_seconds"].to_numpy(float)
        t["error"] = d["error"].fillna("").to_numpy()
        t["asr_ok"] = d["asr_ok"].astype(bool).to_numpy()
        t["confidence"] = d["confidence"].to_numpy(float)
        t["model_key"], t["model_version"] = m, str(d["model_version"].iloc[0])
        t["provenance_type"] = EXTERNAL_PROVENANCE
        store.invalidate(m, s)
        store.set_fingerprint(m, s, fp, {"source_sha256": provenance.get("source_sha256")})
        store.append(m, s, t)
        g = gate[gate["model_key"] == m] if not gate.empty else gate
        write_json(store.dir(m, s) / "_provenance.json",
                   {**provenance, "model_key": m, "split": s, "rows": int(len(t)),
                    "failures": int((~t["asr_ok"]).sum()), "fingerprint": fp,
                    "gate": g.to_dict("records") if not g.empty else []})
        out.append({"model_key": m, "split": s, "rows": len(t), "path": str(store.dir(m, s))})
    return pd.DataFrame(out)
