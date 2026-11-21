"""Audio manifest <-> clip metadata <-> ASR prediction joins (split + normalized original filename).

Rules
  * the join key is (split, normalize_clip_name(original filename)); slashes, archive-relative paths, case and
    audio extensions are normalized. A hashed / generated clip_id is NEVER assumed to equal a metadata filename or
    a subject_id: it is only resolved through an audio manifest that lists both (clip_id -> audio_name/member)
  * metadata keys must be unique per split (one-to-one); a collision is a FAIL in data_integrity
  * an ASR row without a split is assigned the split of the metadata clip it matches; a name present in both
    splits is ambiguous and stays unmatched (reported)
  * unmatched rows, failed inferences, task / split / sample-rate mismatches are reported with examples and never
    silently discarded (joined rows keep asr_ok=False for failures)
  * student identity comes from metadata only; train/test separation is asserted (no student in both splits)
  * school_id and score_status are train-only: they appear in the masters as descriptive columns flagged
    `*_train_only` and are never prediction features
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from .. import FILE, ID, TARGET
from .asr_import import normalize_clip_name, normalize_split

DEFAULT_EXPECTED = {"train_clips": 14704, "test_clips": 9814, "train_students": 541}


# ============================================================================== metadata index
def metadata_index(bundle, exts) -> pd.DataFrame:
    parts = []
    for split, clips in (("train", bundle.train_clips), ("test", bundle.test_clips)):
        c = clips.copy()
        c["split"] = split
        parts.append(c)
    m = pd.concat(parts, ignore_index=True, sort=False)
    m["filename_norm"] = m[FILE].map(lambda s: normalize_clip_name(s, exts))
    keep = ["split", FILE, "filename_norm", ID, "task", "item_id", "expected_text", "expected_text_norm", "item_score",
            "is_scored", "duration_sec"]
    out = m[[k for k in keep if k in m.columns]].copy()
    if "score_status" in m:
        out["score_status_train_only"] = m["score_status"]
    return out


# ============================================================================== manifests
def normalize_manifest(man: pd.DataFrame, split: str | None, exts, source: str) -> pd.DataFrame:
    """Local TAR manifest (filename, member, offset, size, ...) or an external manifest (clip_id, source, member,
    source_type, audio_name, basename, split_hint) -> split, filename_norm, member, audio_name, clip_id, ..."""
    if man is None or man.empty:
        return pd.DataFrame(columns=["split", "filename_norm", "member", "audio_name", "clip_id", "manifest_source"])
    d = man.copy()
    lower = {c.lower(): c for c in d.columns}
    name_col = next((lower[c] for c in ("audio_name", "filename", "basename", "member") if c in lower), None)
    out = pd.DataFrame(index=d.index)
    out["audio_name"] = d[name_col].astype(str) if name_col else None
    out["member"] = d[lower["member"]].astype(str) if "member" in lower else out["audio_name"]
    out["filename_norm"] = out["audio_name"].map(lambda s: normalize_clip_name(s, exts))
    out["clip_id"] = d[lower["clip_id"]].astype(str) if "clip_id" in lower else None
    for c in ("source", "source_type", "split_hint", "offset", "size", "safe"):
        if c in lower:
            out[c] = d[lower[c]]
    sp = pd.Series(split, index=d.index, dtype=object) if split else pd.Series(None, index=d.index, dtype=object)
    for c in ("split_hint", "source", "member", "source_type"):
        if c in out:
            sp = sp.where(sp.notna(), out[c].map(normalize_split))
    out["split"] = sp
    out["manifest_source"] = source
    return out.reset_index(drop=True)


def find_external_manifests(asr_root: Path, globs) -> list[Path]:
    root = Path(asr_root)
    if not root.exists():
        return []
    found = []
    for p in sorted(root.iterdir()):
        if p.is_file() and any(fnmatch.fnmatch(p.name.lower(), g.lower()) for g in globs):
            found.append(p)
    return found


def load_external_manifest(paths: list[Path], exts) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
        parts.append(normalize_manifest(df, None, exts, f"external:{p.name}"))
    return pd.concat(parts, ignore_index=True) if parts else normalize_manifest(None, None, exts, "external")


# ============================================================================== ASR join
def _examples(df: pd.DataFrame, cols, k: int) -> str:
    cols = [c for c in cols if c in df.columns]
    return "; ".join(" | ".join(str(v) for v in r) for r in df[cols].head(k).itertuples(index=False)) if not df.empty else ""


def join_asr(canon: pd.DataFrame, meta: pd.DataFrame, ext_manifest: pd.DataFrame | None, max_examples: int = 25
             ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """-> (joined rows incl. failures, unmatched rows, audit rows)."""
    audit: list[dict] = []
    c = canon.copy()
    # 1) clip_id -> audio name through an external manifest (never clip_id == filename)
    if "clip_ref" in c and c["clip_ref"].notna().any() and ext_manifest is not None and not ext_manifest.empty \
            and ext_manifest["clip_id"].notna().any():
        mm = ext_manifest.dropna(subset=["clip_id"]).drop_duplicates("clip_id")
        dup_ids = int(ext_manifest.dropna(subset=["clip_id"])["clip_id"].duplicated().sum())
        audit.append({"check": "external manifest clip_id unique", "status": "PASS" if dup_ids == 0 else "FAIL",
                      "value": dup_ids, "detail": "duplicate clip_id rows in the external audio manifest"})
        by_id = mm.set_index("clip_id")
        need = c["filename_norm"].eq("") & c["clip_ref"].notna()
        c.loc[need, "filename_norm"] = c.loc[need, "clip_ref"].map(by_id["filename_norm"]).fillna("")
        sp = c.loc[need, "clip_ref"].map(by_id["split"])
        c.loc[need, "split_src"] = c.loc[need, "split_src"].where(c.loc[need, "split_src"].notna(), sp)
        audit.append({"check": "ASR rows resolved through manifest clip_id", "status": "PASS", "value": int(need.sum()),
                      "detail": "rows whose only identity was a clip_id"})
    # 2) split inference by metadata membership
    tr_names = set(meta.loc[meta["split"] == "train", "filename_norm"])
    te_names = set(meta.loc[meta["split"] == "test", "filename_norm"])
    both = tr_names & te_names
    inferred = np.where(c["filename_norm"].isin(tr_names) & ~c["filename_norm"].isin(both), "train",
                        np.where(c["filename_norm"].isin(te_names) & ~c["filename_norm"].isin(both), "test", None))
    c["split_inferred"] = inferred
    conflict = c["split_src"].notna() & c["split_inferred"].notna() & (c["split_src"] != c["split_inferred"])
    audit.append({"check": "ASR split label agrees with metadata", "status": "PASS" if not conflict.any() else "WARN",
                  "value": int(conflict.sum()), "detail": "split label differs from the split of the matching clip "
                  "(metadata split used); examples: " + _examples(c[conflict], ["model_key", "audio_ref", "split_src"], max_examples)})
    c["split"] = np.where(c["split_inferred"].notna(), c["split_inferred"], c["split_src"])
    audit.append({"check": "clip names present in both splits", "status": "PASS" if not both else "FAIL",
                  "value": len(both), "detail": "ambiguous normalized names (left unmatched): " + ", ".join(sorted(both)[:max_examples])})
    # 3) join to metadata on (split, filename_norm)
    mcols = ["split", "filename_norm", FILE, ID, "task", "expected_text", "item_score", "duration_sec"]
    j = c.merge(meta[[k for k in mcols if k in meta.columns]], on=["split", "filename_norm"], how="left",
                validate="many_to_one")
    matched = j[FILE].notna()
    unmatched = j[~matched].copy()
    joined = j[matched].copy()
    for m, d in c.groupby("model_key"):
        um = unmatched[unmatched["model_key"] == m]
        audit.append({"check": "ASR rows matched to metadata", "model_key": m,
                      "status": "PASS" if um.empty else "WARN", "value": int(len(um)),
                      "detail": f"{len(um)}/{len(d)} unmatched; examples: " + _examples(um, ["audio_ref", "clip_ref", "split"], max_examples)})
    if "task_src" in joined and joined["task_src"].notna().any():
        tm = joined["task_src"].notna() & joined["task_src"].ne("none") & (joined["task_src"] != joined["task"])
        audit.append({"check": "ASR task agrees with metadata task", "status": "PASS" if not tm.any() else "FAIL",
                      "value": int(tm.sum()), "detail": _examples(joined[tm], ["model_key", FILE, "task_src", "task"], max_examples)})
    if "audio_seconds" in joined and joined["audio_seconds"].notna().any():
        dd = (joined["audio_seconds"] - joined["duration_sec"]).abs()
        bad = dd > 0.5 + 0.05 * joined["duration_sec"].abs()
        audit.append({"check": "ASR audio duration agrees with metadata duration", "status": "PASS" if not bad.any() else "WARN",
                      "value": int(bad.sum()), "detail": "|ASR audio_seconds - duration_sec| > 0.5 s + 5 %; examples: "
                      + _examples(joined[bad], ["model_key", FILE, "audio_seconds", "duration_sec"], max_examples)})
    return joined.reset_index(drop=True), unmatched.reset_index(drop=True), audit


def sample_rate_audit(joined: pd.DataFrame, file_sr: pd.Series | None, students: pd.DataFrame) -> list[dict]:
    """ASR-reported sample rate vs decoded file rate (META::file_sr of the acoustic cache) and vs metadata
    original_sample_rate (information: files are distributed resampled, the original rate is a domain variable)."""
    rows = []
    if "sample_rate" not in joined or not joined["sample_rate"].notna().any():
        rows.append({"check": "ASR sample rate reported", "status": "WARN", "value": 0,
                     "detail": "export carries no sample-rate field: mismatch check skipped"})
        return rows
    if file_sr is not None and len(file_sr):
        fs = joined[FILE].map(file_sr)
        bad = fs.notna() & (joined["sample_rate"] != fs)
        rows.append({"check": "ASR sample rate == decoded file rate", "status": "PASS" if not bad.any() else "FAIL",
                     "value": int(bad.sum()), "detail": _examples(joined[bad], ["model_key", FILE, "sample_rate"], 10)})
    st = students.drop_duplicates(ID).set_index(ID)
    if "original_sample_rate" in st:
        osr = joined[ID].map(st["original_sample_rate"]).astype(float)
        diff = (joined["sample_rate"].astype(float) != osr) & osr.notna()
        rows.append({"check": "ASR sample rate vs original_sample_rate (informational)", "status": "PASS",
                     "value": int(diff.sum()), "detail": "differences are expected when audio was resampled before "
                     "distribution; original_sample_rate is modelled as a recording domain"})
    return rows


# ============================================================================== coverage
def coverage_table(joined: pd.DataFrame, unmatched: pd.DataFrame, duplicates: pd.DataFrame, meta: pd.DataFrame
                   ) -> pd.DataFrame:
    rows = []
    models = sorted(set(joined["model_key"]) | set(unmatched.get("model_key", pd.Series(dtype=str))))
    for m in models:
        for s in ("train", "test"):
            mc = meta[meta["split"] == s]
            d = joined[(joined["model_key"] == m) & (joined["split"] == s)]
            usable = d["asr_ok"] & (d["transcript"].notna() | d["has_phones"].astype(bool))
            nonempty = d["transcript"].fillna("").astype(str).str.strip().ne("")
            um = unmatched[(unmatched["model_key"] == m) & (unmatched["split"].fillna("?") == s)] if not unmatched.empty else unmatched
            du = duplicates[(duplicates["model_key"] == m) & (duplicates["split"] == s)] if not duplicates.empty else duplicates
            n_meta = int(len(mc))
            stud = mc[ID].nunique()
            rows.append({"model_key": m, "model_id": d["model_id"].iloc[0] if len(d) else None, "split": s,
                         "n_metadata_clips": n_meta, "n_matched_rows": int(len(d)), "n_usable": int(usable.sum()),
                         "n_failed": int((~d["asr_ok"]).sum()), "n_nonempty_transcript": int(nonempty.sum()),
                         "n_empty_transcript": int((d["transcript"].notna() & ~nonempty).sum()),
                         "n_phone_strings": int(d["has_phones"].astype(bool).sum()),
                         "n_with_confidence": int(d["confidence"].notna().sum() + d["has_token_probs"].astype(bool).sum()),
                         "n_unmatched_rows": int(len(um)), "n_duplicates_discarded": int(len(du)),
                         "n_missing_predictions": int(n_meta - d[FILE].nunique()),
                         "usable_frac": float(usable.sum() / n_meta) if n_meta else np.nan,
                         "students_covered_frac": float(d.loc[usable, ID].nunique() / stud) if stud else np.nan,
                         "decode_seconds_mean": float(d["decode_seconds"].mean()) if d["decode_seconds"].notna().any() else np.nan})
    return pd.DataFrame(rows)


# ============================================================================== masters
def build_clip_master(meta: pd.DataFrame, manifests: pd.DataFrame, joined: pd.DataFrame, usable_models: list[str],
                      acoustic_presence: pd.DataFrame | None) -> pd.DataFrame:
    cm = meta.copy()
    if manifests is not None and not manifests.empty:
        mm = manifests.dropna(subset=["split"]).drop_duplicates(["split", "filename_norm"])
        keep = [c for c in ("split", "filename_norm", "member", "clip_id", "manifest_source") if c in mm]
        cm = cm.merge(mm[keep], on=["split", "filename_norm"], how="left", validate="one_to_one")
        cm["in_audio_manifest"] = cm["member"].notna()
    else:
        cm["in_audio_manifest"] = np.nan
    if not joined.empty:
        ok = joined[joined["asr_ok"]].copy()
        for m in sorted(joined["model_key"].unique()):
            d = joined[joined["model_key"] == m].drop_duplicates(FILE).set_index(FILE)
            cm[f"asr_present__{m}"] = cm[FILE].map(d["asr_ok"]).map({True: 1.0, False: 0.0})
        cm["asr_n_models_ok"] = cm[FILE].map(ok.groupby(FILE)["model_key"].nunique()).fillna(0).astype(int)
        cm["asr_n_usable_models_ok"] = cm[FILE].map(ok[ok["model_key"].isin(usable_models)].groupby(FILE)["model_key"].nunique()).fillna(0).astype(int)
    else:
        cm["asr_n_models_ok"] = 0
        cm["asr_n_usable_models_ok"] = 0
    if acoustic_presence is not None and not acoustic_presence.empty:
        cm = cm.merge(acoustic_presence, on=FILE, how="left")
    return cm


def build_student_master(bundle, clip_master: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for split, st in (("train", bundle.train_students), ("test", bundle.test_students)):
        s = st.copy()
        s["split"] = split
        parts.append(s)
    sm = pd.concat(parts, ignore_index=True, sort=False)
    if "school_id" in sm:
        sm = sm.rename(columns={"school_id": "school_id_train_only"})
    lab = bundle.labels.set_index(ID)[TARGET]
    sm[TARGET] = sm[ID].map(lab)
    sm.loc[sm["split"] == "test", TARGET] = np.nan                 # never a test label
    g = clip_master.groupby(ID)
    sm["n_clips"] = sm[ID].map(g.size()).fillna(0).astype(int)
    for t, d in clip_master.groupby("task"):
        sm[f"n_clips__{t}"] = sm[ID].map(d.groupby(ID).size()).fillna(0).astype(int)
    if "asr_n_usable_models_ok" in clip_master:
        sm["asr_clip_coverage"] = sm[ID].map(g["asr_n_usable_models_ok"].apply(lambda v: float((v > 0).mean())))
    if "in_audio_manifest" in clip_master:
        sm["audio_manifest_coverage"] = sm[ID].map(g["in_audio_manifest"].mean())
    ac = [c for c in clip_master.columns if c.startswith("acoustic_present__")]
    for c in ac:
        sm[c.replace("acoustic_present__", "acoustic_coverage__")] = sm[ID].map(g[c].mean())
    return sm


# ============================================================================== integrity
def data_integrity(bundle, meta: pd.DataFrame, manifests: pd.DataFrame, joined: pd.DataFrame, expected: dict | None
                   ) -> tuple[dict, list[dict]]:
    exp = {**DEFAULT_EXPECTED, **(expected or {})}
    rows: list[dict] = []

    def add(check, ok, value, detail, warn=False):
        rows.append({"check": check, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "value": value,
                     "detail": detail})

    tr, te = bundle.train_clips, bundle.test_clips
    add("train clip count", len(tr) == exp["train_clips"], len(tr), f"expected {exp['train_clips']}", warn=True)
    add("test clip count", len(te) == exp["test_clips"], len(te), f"expected {exp['test_clips']}", warn=True)
    add("total clip count", len(tr) + len(te) == exp["train_clips"] + exp["test_clips"], len(tr) + len(te),
        f"expected {exp['train_clips'] + exp['test_clips']}", warn=True)
    add("train students", bundle.train_students[ID].nunique() == exp["train_students"],
        int(bundle.train_students[ID].nunique()), f"expected ~{exp['train_students']}", warn=True)
    for s in ("train", "test"):
        d = meta[meta["split"] == s]
        dup = int(d["filename_norm"].duplicated().sum())
        add(f"{s} normalized clip names unique (one-to-one key)", dup == 0, dup,
            "collisions: " + ", ".join(d.loc[d["filename_norm"].duplicated(keep=False), FILE].astype(str).head(10)))
        add(f"{s} metadata filenames unique", d[FILE].is_unique, int(d[FILE].duplicated().sum()), "")
    ov = set(bundle.train_students[ID]) & set(bundle.test_students[ID])
    add("train/test students disjoint", not ov, len(ov), "students in both splits")
    clip_ov = set(tr[ID]) & set(te[ID])
    add("train/test clip owners disjoint", not clip_ov, len(clip_ov), "clip subject_ids in both splits")
    lab = set(bundle.labels[ID])
    add("labels only for train students", lab <= set(bundle.train_students[ID]), len(lab - set(bundle.train_students[ID])), "")
    add("every train student labelled", set(bundle.train_students[ID]) <= lab, len(set(bundle.train_students[ID]) - lab), "")
    if manifests is not None and not manifests.empty:
        for s in ("train", "test"):
            ms = manifests[manifests["split"] == s]
            if ms.empty:
                add(f"{s} audio manifest present", False, 0, "no manifest rows for this split", warn=True)
                continue
            md = set(meta.loc[meta["split"] == s, "filename_norm"])
            mn = set(ms["filename_norm"])
            add(f"{s} metadata clips present in audio manifest", md <= mn, len(md - mn),
                "missing: " + ", ".join(sorted(md - mn)[:10]), warn=True)
            add(f"{s} audio manifest has no unlisted audio", mn <= md, len(mn - md),
                "unlisted: " + ", ".join(sorted(mn - md)[:10]), warn=True)
            src = ms["manifest_source"] if "manifest_source" in ms else pd.Series("all", index=ms.index)
            dup = int(ms.assign(_src=src.to_numpy()).duplicated(["_src", "filename_norm"]).sum())
            add(f"{s} audio manifest names unique within each manifest", dup == 0, dup,
                "duplicate normalized names inside one manifest (first occurrence used)", warn=True)
    if joined is not None and not joined.empty:
        d = int(joined.duplicated(["model_key", "split", FILE]).sum())
        add("one ASR prediction per model x clip after resolution", d == 0, d, "")
        leak = joined[ID].isin(set(bundle.test_students[ID])) & (joined["split"] == "train")
        add("no test student in train ASR rows", not leak.any(), int(leak.sum()), "")
    add("school_id absent from test", "school_id" not in bundle.test_students.columns, None,
        "train-only: grouping / robustness only", warn=True)
    add("score_status absent from test", "score_status" not in bundle.test_clips.columns, None,
        "train-only: never a feature", warn=True)
    summary = {"n_train_clips": int(len(tr)), "n_test_clips": int(len(te)), "n_train_students": int(bundle.train_students[ID].nunique()),
               "n_test_students": int(bundle.test_students[ID].nunique()), "n_labels": int(len(bundle.labels)),
               "tasks": sorted(bundle.tasks), "theta_cols": list(bundle.theta_cols), "expected": exp,
               "n_fail": int(sum(r["status"] == "FAIL" for r in rows)), "n_warn": int(sum(r["status"] == "WARN" for r in rows)),
               "checks": rows}
    return summary, rows
