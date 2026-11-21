"""Section 01 - data integrity and hierarchical data reconstruction.

Scientific question: can the dataset be represented reliably at the student,
task, item, clip and acoustic-frame levels?

All schemas are discovered at runtime. A schema-resolution layer maps actual
columns to canonical names; ambiguous mappings are never guessed silently -
they must be resolved through ``CONFIG['schema_overrides']``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_manifest as am
from .families import sanitize_token

FILE_ROLES = {
    "train_student_metadata.csv": "student_train",
    "test_student_metadata.csv": "student_test",
    "train_clip_metadata.csv": "clip_train",
    "test_clip_metadata.csv": "clip_test",
    "train_labels.csv": "labels",
    "submission_format.csv": "submission",
}
ROLE_GROUP = {"student_train": "student", "student_test": "student", "clip_train": "clip",
              "clip_test": "clip", "labels": "labels", "submission": "submission"}

READ_CSV_PARAMS = {
    "sep", "delimiter", "header", "names", "index_col", "usecols", "dtype", "engine", "converters",
    "true_values", "false_values", "skipinitialspace", "skiprows", "skipfooter", "nrows", "na_values",
    "keep_default_na", "na_filter", "parse_dates", "date_format", "dayfirst", "thousands", "decimal",
    "lineterminator", "quotechar", "quoting", "doublequote", "escapechar", "comment", "encoding",
    "encoding_errors", "dialect", "on_bad_lines", "low_memory", "memory_map", "float_precision",
    "dtype_backend",
}

FIELD_SPECS: dict = {
    "student": {
        "student_id": [r"^(student|child|participant|subject|learner|pupil|kid)(_?(id|key|uid|identifier|code))?$", r"^s_?id$"],
        "school_id": [r"^(school|site|campus)(_?(id|code|name|key))?$"],
        "instrument": [r"(instrument|assessment_?name|assessment_?type|measure|test_?name|test_?type|outcome_?test|eoy_?test|outcome_?measure|score_?type|metric_?name)"],
        "grade": [r"^grade(_?level)?$"],
    },
    "clip": {
        "student_id": [r"^(student|child|participant|subject|learner|pupil|kid)(_?(id|key|uid|identifier|code))?$", r"^s_?id$"],
        "clip_id": [r"^(clip|recording|audio|utterance|response|trial)_?(id|key|uid|identifier)$", r"^clip$", r"^(file|filename|file_name)_?id$"],
        "audio_file": [r"^(file|filename|file_name|filepath|file_path|path|audio|audio_?file|audio_?path|audio_?filename|wav|wav_?file|wav_?path|recording|recording_?file|clip_?file|clip_?path|clip_?filename)$"],
        "school_id": [r"^(school|site|campus)(_?(id|code|name|key))?$"],
        "task": [r"^(task|task_?name|task_?type|subtest|sub_?test|activity|task_?id|assessment_?task|skill)$"],
        "item_id": [r"^(item|item_?id|item_?name|item_?code|prompt_?id|question_?id|stimulus_?id|word_?id|item_?number)$"],
        "expected_text": [r"^(expected|expected_?text|expected_?response|expected_?answer|prompt|prompt_?text|target_?text|stimulus|stimulus_?text|text|word|target_?word|answer|correct_?answer|correct_?response|sentence|transcript)$"],
        "correct": [r"^(correct|is_?correct|correctness|accuracy|response_?correct|item_?score|score|correct_?flag|accurate|result)$"],
        "order": [r"^(order|item_?order|sequence|seq|seq_?num|sequence_?number|position|item_?position|trial|trial_?num|trial_?number|item_?index|presentation_?order|item_?seq)$"],
        "timestamp": [r"(timestamp|start_?time|datetime|recorded_?at|created_?at|time_?stamp)$"],
        "response_duration": [r"^(duration|response_?time|rt|response_?duration|latency|duration_?s|duration_?sec|length_?s)$"],
        "orig_sample_rate": [r"^(sample_?rate|sampling_?rate|sr|orig_?sr|original_?sample_?rate|samplerate)$"],
        "deletion_source_text": [r"^(source_?word|original_?word|base_?word|full_?word)$"],
        "deletion_target_sound": [r"^(deleted_?sound|delete_?sound|target_?sound|removed_?sound|sound_?to_?delete|deleted_?phoneme)$"],
    },
    "labels": {
        "student_id": [r"^(student|child|participant|subject|learner|pupil|kid)(_?(id|key|uid|identifier|code))?$", r"^s_?id$"],
        "instrument": [r"(instrument|assessment_?name|assessment_?type|measure|test_?name|test_?type|outcome_?test|eoy_?test|outcome_?measure|score_?type|metric_?name)"],
        "school_id": [r"^(school|site|campus)(_?(id|code|name|key))?$"],
        "target": [r"(score|target|label|outcome|eoy|reading|^y)$"],
    },
    "submission": {
        "student_id": [r"^(student|child|participant|subject|learner|pupil|kid)(_?(id|key|uid|identifier|code))?$", r"^s_?id$"],
        "target": [r"(score|target|label|outcome|eoy|reading|prediction|^y)$"],
    },
}
REQUIRED_FIELDS = {"student": ["student_id"], "clip": ["student_id"], "labels": ["student_id", "target"],
                   "submission": ["student_id"]}
SKILL_PATTERNS = {"blending": r"blend", "deletion": r"(delet|elision|elid)", "repetition": r"(repet|sentence|srep|imitat)"}
SE_PATTERN = r"(^|_)(se|sem|stderr|std_?err|error|var|variance|sd)($|_)"
ABILITY_PATTERN = r"(theta|ability|irt|est|score|boy|screen|eap|mle|level|rasch)"
TASK_PATTERNS = SKILL_PATTERNS
TRUE_STRINGS = {"true", "t", "correct", "yes", "y", "1", "1.0", "c", "right"}
FALSE_STRINGS = {"false", "f", "incorrect", "no", "n", "0", "0.0", "i", "wrong", "x"}


class SchemaResolutionError(RuntimeError):
    pass


def normalize_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


# --------------------------------------------------------------------------- CSV
def load_read_kwargs(path: Path, files: list) -> tuple[dict, dict]:
    """Interpret pandas_read_csv_kwargs.json and decide which files it applies to."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    report: dict = {"raw_keys": list(raw) if isinstance(raw, dict) else str(type(raw)), "mode": None}
    per_file: dict = {f: {} for f in files}
    if not isinstance(raw, dict):
        report["mode"] = "UNRECOGNISED_IGNORED"
        return per_file, report
    file_keys = [k for k in raw if k in files or Path(k).name in files or f"{k}.csv" in files]
    if file_keys and len(file_keys) == len(raw):
        report["mode"] = "PER_FILE"
        for k, v in raw.items():
            name = k if k in files else (Path(k).name if Path(k).name in files else f"{k}.csv")
            per_file[name] = {p: val for p, val in (v or {}).items() if p in READ_CSV_PARAMS}
        return per_file, report
    params = {k: v for k, v in raw.items() if k in READ_CSV_PARAMS}
    report["ignored_keys"] = [k for k in raw if k not in READ_CSV_PARAMS]
    report["mode"] = "GLOBAL"
    for f in files:
        per_file[f] = dict(params)
    return per_file, report


def _adapt_kwargs(path: Path, kwargs: dict) -> tuple[dict, dict]:
    """Restrict column-specific arguments to columns that exist in this file."""
    probe_kwargs = {k: v for k, v in kwargs.items() if k in {"sep", "delimiter", "encoding", "header", "quotechar", "comment"}}
    header = pd.read_csv(path, nrows=0, **probe_kwargs).columns.tolist()
    applied = dict(kwargs)
    notes: dict = {}
    if isinstance(applied.get("dtype"), dict):
        dtypes = {c: t for c, t in applied["dtype"].items() if c in header}
        notes["dtype_columns_applied"] = sorted(dtypes)
        notes["dtype_columns_not_present"] = sorted(set(applied["dtype"]) - set(dtypes))
        applied["dtype"] = dtypes
    if isinstance(applied.get("parse_dates"), list):
        applied["parse_dates"] = [c for c in applied["parse_dates"] if c in header]
    if isinstance(applied.get("usecols"), list):
        applied["usecols"] = [c for c in applied["usecols"] if c in header]
    if isinstance(applied.get("converters"), dict):
        applied.pop("converters")
        notes["converters_ignored"] = True
    return applied, notes


def read_tables(data_root: Path, logger) -> tuple[dict, dict]:
    files = [f for f in FILE_ROLES]
    per_file, kw_report = load_read_kwargs(data_root / "pandas_read_csv_kwargs.json", files)
    tables: dict = {}
    report: dict = {"read_csv_kwargs": kw_report, "files": {}}
    for fname, role in FILE_ROLES.items():
        path = data_root / fname
        applied, notes = _adapt_kwargs(path, per_file.get(fname, {}))
        df = pd.read_csv(path, low_memory=False, **applied)
        tables[role] = df
        report["files"][fname] = {"role": role, "shape": list(df.shape), "kwargs_applied": applied, **notes}
        logger.info("Read %s: %s rows x %s cols", fname, df.shape[0], df.shape[1])
    return tables, report


def profile_table(df: pd.DataFrame, table: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        rows.append({
            "table": table, "column": col, "normalized": normalize_name(col), "dtype": str(s.dtype),
            "n_rows": n, "n_missing": int(s.isna().sum()), "missing_rate": float(s.isna().mean()) if n else np.nan,
            "n_unique": int(s.nunique(dropna=True)), "unique_ratio": float(s.nunique(dropna=True) / max(n, 1)),
            "numeric_fraction": float(num.notna().sum() / max(s.notna().sum(), 1)),
            "min": float(num.min()) if num.notna().any() else np.nan,
            "max": float(num.max()) if num.notna().any() else np.nan,
            "mean": float(num.mean()) if num.notna().any() else np.nan,
            "examples": "; ".join(map(str, s.dropna().unique()[:3])),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ schema resolution
def resolve_fields(df: pd.DataFrame, group: str, overrides: dict) -> dict:
    specs = FIELD_SPECS[group]
    norm = {c: normalize_name(c) for c in df.columns}
    assigned: dict = {}
    ambiguous: dict = {}
    used: set = set()
    over = overrides.get(group, {}) or {}
    for field, col in over.items():
        if col not in df.columns:
            raise SchemaResolutionError(f"schema_overrides[{group}][{field}]={col!r} not in columns {list(df.columns)}")
        assigned[field] = col
        used.add(col)
    for field, patterns in specs.items():
        if field in assigned:
            continue
        matches = [c for c in df.columns if c not in used and any(re.search(p, norm[c]) for p in patterns)]
        if field == "target":
            matches = [c for c in matches if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.95]
        if len(matches) == 1:
            assigned[field] = matches[0]
            used.add(matches[0])
        elif len(matches) > 1:
            exact = [c for c in matches if norm[c] == field]
            if len(exact) == 1:
                assigned[field] = exact[0]
                used.add(exact[0])
            else:
                ambiguous[field] = matches
    if group in {"labels", "submission"} and "target" not in assigned and "target" not in ambiguous:
        numeric = [c for c in df.columns if c not in used and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.95]
        if len(numeric) == 1:
            assigned["target"] = numeric[0]
            used.add(numeric[0])
        elif len(numeric) > 1:
            ambiguous["target"] = numeric
    missing = [f for f in REQUIRED_FIELDS.get(group, []) if f not in assigned]
    return {"assigned": assigned, "ambiguous": ambiguous, "missing_required": missing,
            "unassigned_columns": [c for c in df.columns if c not in used]}


def resolve_thetas(df: pd.DataFrame, exclude: set, overrides: dict) -> dict:
    over = (overrides.get("student", {}) or {})
    result: dict = {"theta": {}, "theta_se": {}, "ambiguous": {}, "other": {}}
    for skill in SKILL_PATTERNS:
        key = f"theta_{skill}"
        if key in over:
            result["theta"][skill] = over[key]
    numeric_cols = [c for c in df.columns if c not in exclude
                    and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    claimed = set(result["theta"].values())
    for skill, rx in SKILL_PATTERNS.items():
        cands = [c for c in numeric_cols if c not in claimed and re.search(rx, normalize_name(c))]
        se = [c for c in cands if re.search(SE_PATTERN, normalize_name(c))]
        est = [c for c in cands if c not in se]
        if se:
            result["theta_se"][skill] = se[0] if len(se) == 1 else se
        if skill in result["theta"]:
            continue
        if len(est) > 1:
            preferred = [c for c in est if re.search(ABILITY_PATTERN, normalize_name(c))]
            est = preferred if len(preferred) >= 1 else est
        if len(est) == 1:
            result["theta"][skill] = est[0]
            claimed.add(est[0])
        elif len(est) > 1:
            result["ambiguous"][skill] = est
    for c in numeric_cols:
        if c in claimed or any(c == v or (isinstance(v, list) and c in v) for v in result["theta_se"].values()):
            continue
        if re.search(r"(theta|ability|irt|rasch)", normalize_name(c)):
            result["other"][c] = f"theta_other_{sanitize_token(c)}"
    return result


def normalize_correct(series: pd.Series, threshold=None) -> tuple[pd.Series, pd.Series, dict]:
    s = series.copy()
    info: dict = {"original_dtype": str(series.dtype)}
    if s.dtype == bool:
        out = s.astype(float)
        info["mode"] = "BOOLEAN"
        return out, out, info
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 0.95 * s.notna().sum() and s.notna().any():
        vals = set(np.unique(num.dropna()))
        if vals <= {0.0, 1.0}:
            info["mode"] = "BINARY_NUMERIC"
            return num, num, info
        info["mode"] = "PARTIAL_SCORE"
        info["unique_values"] = sorted(map(float, list(vals)[:20]))
        if threshold is not None:
            info["binarized_at"] = float(threshold)
            return (num >= threshold).astype(float).where(num.notna()), num, info
        info["note"] = "Not binary; correct left missing. Set correct_binarize_threshold to binarise."
        return pd.Series(np.nan, index=s.index), num, info
    low = s.astype(str).str.strip().str.lower()
    mapped = pd.Series(np.where(low.isin(TRUE_STRINGS), 1.0, np.where(low.isin(FALSE_STRINGS), 0.0, np.nan)), index=s.index)
    mapped[s.isna()] = np.nan
    info["mode"] = "STRING_MAPPED"
    info["unmapped_values"] = sorted(set(low[mapped.isna() & s.notna()].unique()))[:20]
    return mapped, mapped, info


def task_family(value, overrides: dict) -> str:
    if pd.isna(value):
        return "unknown"
    if str(value) in overrides:
        return overrides[str(value)]
    low = normalize_name(value)
    matches = [fam for fam, rx in TASK_PATTERNS.items() if re.search(rx, low)]
    return matches[0] if len(matches) == 1 else f"other_{sanitize_token(value)}"


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ------------------------------------------------------------------- join keys
def candidate_join_keys(tables: dict) -> pd.DataFrame:
    rows = []
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            na = {normalize_name(c): c for c in tables[a].columns}
            nb = {normalize_name(c): c for c in tables[b].columns}
            for key in sorted(set(na) & set(nb)):
                va = set(tables[a][na[key]].dropna().astype(str))
                vb = set(tables[b][nb[key]].dropna().astype(str))
                if not va or not vb:
                    continue
                inter = len(va & vb)
                rows.append({"table_a": a, "table_b": b, "column_a": na[key], "column_b": nb[key],
                             "unique_a": len(va), "unique_b": len(vb), "overlap": inter,
                             "overlap_fraction_of_smaller": inter / min(len(va), len(vb))})
    return pd.DataFrame(rows)


def _canonical(df: pd.DataFrame, assigned: dict) -> pd.DataFrame:
    rename = {col: field for field, col in assigned.items()}
    out = df.rename(columns=rename)
    collisions = [c for c in out.columns[out.columns.duplicated()]]
    if collisions:
        raise SchemaResolutionError(f"canonical name collision: {collisions}")
    return out


def _id_str(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.astype(object).where(out.notna(), np.nan)


def _dedupe_students(df: pd.DataFrame, report: dict, label: str) -> pd.DataFrame:
    dup = df["student_id"].duplicated(keep=False)
    if not dup.any():
        report[f"{label}_duplicate_rows"] = 0
        return df
    exact = df[dup].duplicated(keep="first")
    conflicting_ids = df[dup].drop_duplicates().groupby("student_id").size()
    conflicting_ids = conflicting_ids[conflicting_ids > 1].index.tolist()
    report[f"{label}_duplicate_rows"] = int(dup.sum())
    report[f"{label}_exact_duplicates_removed"] = int(exact.sum())
    report[f"{label}_conflicting_student_ids"] = conflicting_ids[:50]
    report[f"{label}_n_conflicting_student_ids"] = len(conflicting_ids)
    out = df.drop_duplicates().drop_duplicates("student_id", keep="first").copy()
    out["student_record_conflict"] = out["student_id"].isin(conflicting_ids)
    return out


def sample_students(student_ids: pd.Series, clip_counts: pd.Series, cfg: dict) -> list:
    """Student-level sampling for SMOKE/DEBUG profiles (never samples clips independently)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    ids = np.array(sorted(student_ids.unique()))
    rng.shuffle(ids)
    chosen, total = [], 0
    for sid in ids:
        c = int(clip_counts.get(sid, 0))
        if len(chosen) >= int(cfg["debug_max_students"]):
            break
        if chosen and total + c > int(cfg["debug_max_clips"]):
            continue
        chosen.append(sid)
        total += c
    return chosen


# ------------------------------------------------------------------------ stage
def stage_data_audit(ctx) -> dict:
    cfg = ctx.config
    logger = ctx.logger
    data_root = ctx.data_root()
    tables, read_report = read_tables(data_root, logger)
    overrides = cfg.get("schema_overrides") or {}
    profiles = pd.concat([profile_table(df, role) for role, df in tables.items()], ignore_index=True)
    ctx.save_df(profiles, "column_profiles", fmt="csv")

    schema: dict = {"read_report": read_report, "groups": {}, "warnings": []}
    for group, roles in {"student": ["student_train", "student_test"], "clip": ["clip_train", "clip_test"],
                         "labels": ["labels"], "submission": ["submission"]}.items():
        primary = tables[roles[0]]
        res = resolve_fields(primary, group, overrides)
        if len(roles) > 1:
            other = tables[roles[1]]
            res["columns_only_in_train"] = sorted(set(primary.columns) - set(other.columns))
            res["columns_only_in_test"] = sorted(set(other.columns) - set(primary.columns))
            res["test_missing_assigned"] = [c for c in res["assigned"].values() if c not in other.columns]
        schema["groups"][group] = res
    st = schema["groups"]["student"]
    thetas = resolve_thetas(tables["student_train"], set(st["assigned"].values()), overrides)
    schema["thetas"] = thetas
    problems = {g: {"missing_required": r["missing_required"], "ambiguous": r["ambiguous"]}
                for g, r in schema["groups"].items() if r["missing_required"] or (
                    r["ambiguous"] and set(r["ambiguous"]) & set(REQUIRED_FIELDS.get(g, []) + ["clip_id", "audio_file", "task", "correct"]))}
    if thetas["ambiguous"]:
        problems["thetas"] = thetas["ambiguous"]
    ctx.save_json(schema, "schema_registry")
    if problems:
        raise SchemaResolutionError(
            "Schema could not be resolved unambiguously. Inspect schema_registry.json and set "
            f"CONFIG['schema_overrides']. Problems: {json.dumps(problems, default=str)}")

    quality: dict = {"tables": read_report["files"]}
    # ---- students -------------------------------------------------------------
    sa = st["assigned"]
    theta_map = {col: f"theta_{skill}" for skill, col in thetas["theta"].items()}
    for skill, col in thetas["theta_se"].items():
        if isinstance(col, str):
            theta_map[col] = f"theta_se_{skill}"
    theta_map.update(thetas["other"])
    stud = []
    for role, split in (("student_train", "train"), ("student_test", "test")):
        df = _canonical(tables[role], {k: v for k, v in sa.items() if v in tables[role].columns}).rename(columns=theta_map)
        df["student_id"] = _id_str(df["student_id"])
        df["split"] = split
        for c in theta_map.values():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        stud.append(_dedupe_students(df, quality, f"students_{split}"))
    students = pd.concat(stud, ignore_index=True)
    overlap_students = set(stud[0]["student_id"]) & set(stud[1]["student_id"])
    quality["train_test_student_overlap"] = len(overlap_students)
    # ---- labels -----------------------------------------------------------------
    la = schema["groups"]["labels"]["assigned"]
    labels = _canonical(tables["labels"], la)
    labels["student_id"] = _id_str(labels["student_id"])
    labels["target"] = pd.to_numeric(labels["target"], errors="coerce")
    quality["labels_duplicate_students"] = int(labels["student_id"].duplicated().sum())
    if quality["labels_duplicate_students"]:
        conflict = labels.groupby("student_id")["target"].nunique()
        quality["labels_conflicting_students"] = int((conflict > 1).sum())
        labels = labels.drop_duplicates("student_id", keep="first")
    label_cols = ["student_id", "target"] + [c for c in ("instrument", "school_id") if c in labels.columns]
    train_ids = set(stud[0]["student_id"])
    quality["labels_not_in_train_students"] = int((~labels["student_id"].isin(train_ids)).sum())
    quality["train_students_without_label"] = int(len(train_ids - set(labels["student_id"])))
    quality["labels_in_test_students"] = int(labels["student_id"].isin(set(stud[1]["student_id"])).sum())
    lab = labels[label_cols].rename(columns={"instrument": "instrument_label", "school_id": "school_label"})
    students = students.merge(lab, on="student_id", how="left")
    students.loc[students["split"] != "train", "target"] = np.nan
    for base, extra in (("instrument", "instrument_label"), ("school_id", "school_label")):
        if extra in students.columns:
            if base in students.columns:
                both = students[base].notna() & students[extra].notna()
                quality[f"{base}_label_conflicts"] = int((students.loc[both, base].astype(str) != students.loc[both, extra].astype(str)).sum())
                students[base] = students[base].where(students[base].notna(), students[extra])
            else:
                students[base] = students[extra]
            students = students.drop(columns=[extra])
    for col in ("school_id", "instrument"):
        if col not in students.columns:
            students[col] = np.nan
            schema["warnings"].append(f"{col} unavailable: grouped analyses will report fallbacks")
    students["school_id"] = students["school_id"].astype(object).where(students["school_id"].notna(), np.nan)
    if students["student_id"].duplicated().any():
        raise SchemaResolutionError("student_master would contain duplicate student ids")

    # ---- clips --------------------------------------------------------------
    ca = schema["groups"]["clip"]["assigned"]
    clip_parts = []
    correct_info = {}
    for role, split in (("clip_train", "train"), ("clip_test", "test")):
        df = _canonical(tables[role], {k: v for k, v in ca.items() if v in tables[role].columns})
        df["student_id"] = _id_str(df["student_id"])
        if "clip_id" not in df.columns:
            if "audio_file" in df.columns:
                df["clip_id"] = df["audio_file"].astype(str).map(lambda s: Path(s.replace("\\", "/")).stem)
                schema["warnings"].append(f"{role}: clip_id derived from audio_file stem")
            else:
                raise SchemaResolutionError(f"{role}: neither clip_id nor audio_file could be resolved")
        df["clip_id"] = _id_str(df["clip_id"])
        df["split"] = split
        if "correct" in df.columns:
            df["correct"], df["correct_score"], correct_info[split] = normalize_correct(
                df["correct"], cfg.get("correct_binarize_threshold"))
        for col in ("order", "response_duration", "orig_sample_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        clip_parts.append(df)
    clips = pd.concat(clip_parts, ignore_index=True)
    quality["correctness_normalization"] = correct_info
    dup_clip = clips.duplicated(["split", "clip_id"], keep=False)
    quality["duplicate_clip_ids"] = int(dup_clip.sum())
    if dup_clip.any():
        clips["clip_id_original"] = clips["clip_id"]
        clips.loc[dup_clip, "clip_id"] = clips.loc[dup_clip, "clip_id"] + "__dup" + clips[dup_clip].groupby(["split", "clip_id"]).cumcount().astype(str)
    quality["train_test_clip_id_overlap"] = int(len(set(clips.loc[clips.split == "train", "clip_id"]) & set(clips.loc[clips.split == "test", "clip_id"])))
    known = set(students["student_id"])
    quality["clips_with_unknown_student"] = int((~clips["student_id"].isin(known)).sum())
    split_check = clips.merge(students[["student_id", "split"]].rename(columns={"split": "student_split"}), on="student_id", how="left")
    quality["clips_split_mismatch"] = int((split_check["student_split"].notna() & (split_check["student_split"] != split_check["split"])).sum())
    if "task" not in clips.columns:
        clips["task"] = "unknown_task"
        schema["warnings"].append("task unavailable: all clips treated as one task")
    tf_over = cfg.get("task_family_overrides") or {}
    clips["task_family"] = clips["task"].map(lambda v: task_family(v, tf_over))
    if "item_id" in clips.columns:
        clips["item_key"] = clips["task_family"] + "||" + clips["item_id"].astype(str)
        item_source = "item_id"
    elif "expected_text" in clips.columns:
        clips["item_key"] = clips["task_family"] + "||" + clips["expected_text"].map(normalize_text)
        item_source = "task_family+expected_text"
    else:
        clips["item_key"] = np.nan
        item_source = "UNAVAILABLE"
    schema["item_key_source"] = item_source
    if "school_id" in clips.columns:
        clips = clips.rename(columns={"school_id": "clip_school_id"})

    # ---- audio inventory -----------------------------------------------------
    manifests, match_reports = [], {}
    clips["audio_archive"] = np.where(clips["split"] == "train", "train_audio.tar", "test_audio.tar")
    for split, archive in (("train", "train_audio.tar"), ("test", "test_audio.tar")):
        tar_path = data_root / archive
        members = am.inspect_archive(tar_path)
        quality[f"{split}_archive_members"] = int(len(members))
        quality[f"{split}_unsafe_members"] = members.loc[~members["safe"], ["member", "safety_reason"]].head(50).to_dict("records") if len(members) else []
        quality[f"{split}_non_audio_members"] = int((~members["is_audio_ext"]).sum()) if len(members) else 0
        sub = clips[clips["split"] == split]
        matched, rep = am.match_clips_to_members(sub, members, split)
        match_reports[split] = rep
        clips.loc[matched.index, "audio_member"] = matched["audio_member"]
        clips.loc[matched.index, "audio_match_key"] = matched["audio_match_key"]
        decode = bool(cfg.get("audio_inventory_decode")) and (split == "train" or cfg.get("allow_test_inference") or cfg.get("extract_test_audio"))
        inv = am.audio_inventory(tar_path, members, decode=decode, hash_bytes=bool(cfg.get("audio_inventory_hash")), logger=logger)
        inv["split"] = split
        manifests.append(inv)
    manifest = pd.concat(manifests, ignore_index=True)
    clip_links = (clips.dropna(subset=["audio_member"]).groupby(["audio_archive", "audio_member"])["clip_id"]
                  .agg(lambda s: ";".join(map(str, s))).reset_index()
                  .rename(columns={"audio_archive": "archive", "audio_member": "member", "clip_id": "linked_clip_ids"}))
    manifest = manifest.merge(clip_links, on=["archive", "member"], how="left")
    if "sha1" in manifest.columns:
        hashed = manifest.dropna(subset=["sha1"])
        quality["duplicate_audio_within_split"] = {s: int(g["sha1"].duplicated(keep=False).sum()) for s, g in hashed.groupby("split")}
        tr = set(hashed.loc[hashed.split == "train", "sha1"])
        te = set(hashed.loc[hashed.split == "test", "sha1"])
        quality["identical_audio_train_test"] = len(tr & te)
    quality["audio_matching"] = match_reports
    quality["decode_status_counts"] = manifest.groupby(["split", "decode_status"]).size().reset_index(name="n").to_dict("records")
    if "sample_rate" in manifest.columns:
        quality["sample_rate_counts"] = manifest.groupby(["split", "sample_rate"]).size().reset_index(name="n").to_dict("records")
        quality["channel_counts"] = manifest.groupby(["split", "channels"]).size().reset_index(name="n").to_dict("records")
        meta_cols = manifest[["archive", "member", "sample_rate", "channels", "duration_s", "size_bytes", "decode_status"]]
        meta_cols = meta_cols.drop_duplicates(["archive", "member"]).rename(
            columns={"archive": "audio_archive", "member": "audio_member", "sample_rate": "audio_sample_rate",
                     "channels": "audio_channels", "duration_s": "audio_duration_s", "size_bytes": "audio_size_bytes",
                     "decode_status": "audio_decode_status"})
        clips = clips.merge(meta_cols, on=["audio_archive", "audio_member"], how="left")
        if "response_duration" in clips.columns:
            both = clips[["response_duration", "audio_duration_s"]].dropna()
            if len(both) > 10:
                quality["metadata_vs_audio_duration_spearman"] = float(both.corr(method="spearman").iloc[0, 1])

    # ---- sampling (student level) ----------------------------------------------
    if cfg["run_mode"] in {"SMOKE", "DEBUG"} or cfg.get("resumed_profile") in {"SMOKE", "DEBUG"}:
        tr_students = students[(students.split == "train") & students["target"].notna()]["student_id"]
        counts = clips[clips.split == "train"].groupby("student_id").size()
        keep = set(sample_students(tr_students, counts, cfg))
        test_keep = set(students[students.split == "test"]["student_id"].head(int(cfg["debug_max_students"])))
        students = students[students["student_id"].isin(keep | test_keep)].reset_index(drop=True)
        clips = clips[clips["student_id"].isin(keep | test_keep)].reset_index(drop=True)
        quality["sampling"] = {"mode": "STUDENT_LEVEL", "train_students": len(keep), "test_students": len(test_keep)}

    # ---- hierarchical tables --------------------------------------------------------
    train_clips = clips[clips.split == "train"]
    n_clips = clips.groupby("student_id").size()
    students["n_clips"] = students["student_id"].map(n_clips).fillna(0).astype(int)
    agg = {"n_clips_train": ("clip_id", lambda s: int((clips.loc[s.index, "split"] == "train").sum())),
           "n_clips_test": ("clip_id", lambda s: int((clips.loc[s.index, "split"] == "test").sum())),
           "n_students": ("student_id", "nunique")}
    task_master = clips.groupby(["task", "task_family"], dropna=False).agg(**agg).reset_index()
    if item_source != "UNAVAILABLE":
        item_cols = {"n_exposures": ("clip_id", "size"), "n_students": ("student_id", "nunique")}
        if "correct" in train_clips.columns:
            item_cols.update(n_scored=("correct", "count"), p_correct=("correct", "mean"))
        item_master = train_clips.groupby("item_key").agg(**item_cols).reset_index()
        firsts = train_clips.drop_duplicates("item_key").set_index("item_key")
        for c in ("task", "task_family", "item_id", "expected_text", "deletion_source_text", "deletion_target_sound"):
            if c in firsts.columns:
                item_master[c] = item_master["item_key"].map(firsts[c])
        test_items = set(clips.loc[clips.split == "test", "item_key"].dropna())
        item_master["seen_in_test"] = item_master["item_key"].isin(test_items)
        quality["test_items_unseen_in_train"] = int(len(test_items - set(item_master["item_key"])))
    else:
        item_master = pd.DataFrame(columns=["item_key"])
    inventory = pd.DataFrame([{"file": f, "role": FILE_ROLES.get(f, "archive" if f.endswith(".tar") else "config"),
                               "size_bytes": (data_root / f).stat().st_size,
                               "rows": read_report["files"].get(f, {}).get("shape", [np.nan, np.nan])[0],
                               "cols": read_report["files"].get(f, {}).get("shape", [np.nan, np.nan])[1]}
                              for f in cfg["required_files"]])
    joins = candidate_join_keys(tables)
    quality.update({
        "n_students": {s: int((students.split == s).sum()) for s in ("train", "test")},
        "n_clips": {s: int((clips.split == s).sum()) for s in ("train", "test")},
        "n_tasks": int(clips["task"].nunique()), "task_families": sorted(clips["task_family"].unique().tolist()),
        "n_items": int(len(item_master)), "item_key_source": item_source,
        "n_schools": int(students["school_id"].nunique()), "n_instruments": int(students["instrument"].nunique()),
        "train_target_missing": int(students.loc[students.split == "train", "target"].isna().sum()),
        "clips_missing_audio": int(clips["audio_member"].isna().sum()),
        "warnings": schema["warnings"],
    })
    ctx.save_df(students, "student_master")
    ctx.save_df(clips, "clip_master")
    ctx.save_df(task_master, "task_master")
    ctx.save_df(item_master, "item_master")
    ctx.save_df(manifest, "audio_manifest")
    ctx.save_df(inventory, "dataset_inventory", fmt="csv")
    ctx.save_df(joins, "candidate_join_keys", fmt="csv")
    ctx.save_json(schema, "schema_registry")
    ctx.save_json(quality, "data_quality_report")
    _audit_figures(ctx, students, clips, manifest)
    return {"outputs": ["student_master", "clip_master", "task_master", "item_master", "audio_manifest",
                        "dataset_inventory", "schema_registry", "data_quality_report", "column_profiles",
                        "candidate_join_keys"],
            "summary": {"n_students": quality["n_students"], "n_clips": quality["n_clips"],
                        "item_key_source": item_source, "warnings": len(schema["warnings"])}}


def _audit_figures(ctx, students: pd.DataFrame, clips: pd.DataFrame, manifest: pd.DataFrame) -> None:
    from . import visualization as viz

    viz.missingness_matrix(ctx, clips[clips.split == "train"], "clip_missingness", "Training clip metadata missingness", "clip_master")
    per_student = clips[clips.split == "train"].groupby("student_id").size()
    fig, axes = viz.new_figure(1, 2)
    viz.hist_kde(axes[0, 0], per_student.to_numpy(), "clips per training student", "clips")
    if "duration_s" in manifest.columns:
        viz.hist_kde(axes[0, 1], manifest.loc[manifest.split == "train", "duration_s"], "training recording duration", "s")
    ctx.figure(fig, "clips_and_durations", "Clips per student and recording durations",
               "How many observations does each student contribute?", "clip_master/audio_manifest", int(len(per_student)),
               f"Median clips per student {per_student.median():.1f} (IQR {per_student.quantile(.25):.1f}-{per_student.quantile(.75):.1f}).")
