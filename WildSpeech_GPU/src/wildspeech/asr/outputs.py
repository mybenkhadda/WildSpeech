"""Detection of REAL ASR outputs and per-model execution state. This is the gate used by P25-P29.

An ASR output table (cache/v3/asr/<model>/<split>) is ACCEPTED only if all of these hold:
  * its _provenance.json says it was produced by P24 inference (producer == "P24") from an ELIGIBLE record
    (schema-1 caches without `producer` but with eligibility.eligible == true are accepted as legacy P24 output)
  * it has >= 1 row with a non-empty transcript (word/char models) or non-empty phone string (phone models)
Test fixtures (provenance.fixture == true) are accepted ONLY when v3.asr.accept_fixture_outputs is true, which is a
test-only switch. They are always labelled origin=fixture.

execution_state() reports never_run | partial | complete | failed_only | rejected per model and split (rows,
failures, candidates). The state is information only and never makes a model eligible.
output_state_token() fingerprints the accepted outputs, so P25 re-runs when real outputs appear or grow.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import FILE
from ..io_utils import read_json, stable_hash

ORIGINS = ("p24_inference", "p24_inference_legacy", "fixture", "external_colab_asr", "unverified", "none")
EXTERNAL_PROVENANCE = "external_colab_asr"


def _origin(prov: dict, accept_fixture: bool) -> tuple[str, bool, str]:
    if not prov:
        return "none", False, "no _provenance.json"
    if prov.get("provenance_type") == EXTERNAL_PROVENANCE or prov.get("producer") == EXTERNAL_PROVENANCE:
        # externally generated (Colab) predictions are NEVER P24 outputs: they are validated by the independent
        # V4 import gate (wildspeech.v4.asr_import.validate_import) and live in cache/v4, not here
        return EXTERNAL_PROVENANCE, False, ("external Colab ASR import: validated by wildspeech.v4.asr_import, "
                                            "not eligible as P24 output")
    if prov.get("fixture") is True:
        return "fixture", bool(accept_fixture), ("test fixture accepted (v3.asr.accept_fixture_outputs)"
                                                 if accept_fixture else "test fixture output rejected")
    el = prov.get("eligibility") or {}
    eligible = el.get("eligible") is True or el.get("eligibility") == "ELIGIBLE"
    if prov.get("producer") == "P24" and eligible:
        return "p24_inference", True, "P24 inference from an ELIGIBLE record"
    if "producer" not in prov and eligible:
        return "p24_inference_legacy", True, "schema-1 P24 output with an ELIGIBLE record"
    return "unverified", False, "provenance lacks an ELIGIBLE P24 record"


def _count(df: pd.DataFrame, col: str) -> int:
    if col not in df or df.empty:
        return 0
    return int(df[col].fillna("").astype(str).str.strip().ne("").sum())


def _emb_rows(root: Path, key: str, split: str) -> int:
    """Rows of pooled embeddings (parquet metadata only; nothing is loaded)."""
    d = Path(root) / "asr_emb" / key.lower() / split
    if not d.is_dir():
        return 0
    n = 0
    try:
        import pyarrow.parquet as pq

        for f in d.glob("part-*.parquet"):
            n += int(pq.ParquetFile(f).metadata.num_rows)
    except Exception:  # noqa: BLE001
        return 0
    return n


def scan_outputs(ctx, models: list[str] | None = None, splits=None) -> pd.DataFrame:
    """One row per (model, split) cache directory found under cache/v3/asr."""
    from .checkpointing import ASRCheckpointStore

    base = Path(ctx.paths.v3_cache) / "asr"
    splits = tuple(splits or ctx.cfg.get_path("v3.asr.splits", ["train"]) or ["train"])
    keys = sorted(p.name for p in base.glob("*") if p.is_dir()) if base.exists() else []
    if models is not None:
        keys = [k for k in keys if k in set(models)]
    accept_fx = bool(ctx.cfg.get_path("v3.asr.accept_fixture_outputs", False))
    rows = []
    for k in keys:
        st = ASRCheckpointStore(ctx.paths.v3_cache, k)
        for s in splits:
            if not (base / k.lower() / s).is_dir():          # FeatureStore.dir() would create it: check the path
                continue
            prov = st.provenance(s)
            origin, ok, why = _origin(prov, accept_fx)
            df = st.load(s)
            nt, nph = _count(df, "transcript"), _count(df, "phones")
            out_kind = str(prov.get("output") or "")
            ne = _emb_rows(ctx.paths.v3_cache, k, s) if out_kind == "embeddings" else 0
            fails = len(st.failures(s))
            real = (nt + nph) > 0 or (out_kind == "embeddings" and ne > 0)
            rows.append({"model_key": k, "split": s, "origin": origin, "output": out_kind or None, "rows": len(df),
                         "nonempty_transcripts": nt, "nonempty_phones": nph, "embedding_rows": ne,
                         "failures": fails, "has_token_probs_frac":
                         float(df["has_token_probs"].mean()) if "has_token_probs" in df and len(df) else None,
                         "fingerprint": (st.store.fingerprint(k, s) or {}).get("fingerprint"),
                         "model_version": str(df["model_version"].iloc[-1]) if "model_version" in df and len(df) else None,
                         "accepted": bool(ok and real),
                         "reason": why if not ok else (why if real else "no non-empty transcript, phone string or embedding")})
    return pd.DataFrame(rows, columns=["model_key", "split", "origin", "output", "rows", "nonempty_transcripts",
                                       "nonempty_phones", "embedding_rows", "failures", "has_token_probs_frac",
                                       "fingerprint", "model_version", "accepted", "reason"])


def accepted_models(ctx, splits=("train",), need: str = "any") -> list[str]:
    """Models with accepted outputs on every split in `splits`.
    need: any | text (transcripts or phones; recognisers only) | phones | words | embeddings."""
    t = scan_outputs(ctx, None, splits)
    if t.empty:
        return []
    t = t[t["accepted"]]
    if need == "phones":
        t = t[t["nonempty_phones"] > 0]
    elif need == "words":
        t = t[t["nonempty_transcripts"] > 0]
    elif need == "text":
        t = t[(t["nonempty_phones"] + t["nonempty_transcripts"]) > 0]
    elif need == "embeddings":
        t = t[t["embedding_rows"] > 0]
    ok = t.groupby("model_key")["split"].nunique()
    return sorted(ok[ok >= len(set(splits))].index)


def revalidate(ctx, models: list[str], scan: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Drop P24 outputs of models whose CURRENT legal or provenance dimension is no longer PASS (e.g. a review was
    rejected after inference). Fixture outputs (tests only) are not revalidated. Filesystem/registry only."""
    from .eligibility import checker_for
    from .model_registry import document_for

    fx = set(scan.loc[scan["origin"] == "fixture", "model_key"]) if not scan.empty else set()
    check = [m for m in models if m not in fx]
    if not check:
        return list(models), {}
    doc = document_for(ctx)
    recs = checker_for(ctx, with_runtime=False, env={}).check_all(doc.specs, check, probe_files=False)
    dropped = {k: f"current legal={r.groups.get('legal')} provenance={r.groups.get('provenance')}"
               for k, r in recs.items() if not (r.groups.get("legal") == "PASS" and r.groups.get("provenance") == "PASS")}
    return [m for m in models if m not in dropped], dropped


def require_outputs(ctx, sid: str, need: str = "any", min_models: int = 1, splits=("train",)) -> tuple[bool, list[str], str]:
    """Gate for P25-P29. Saves <stage>/tables/asr_output_gate.csv. -> (ok, accepted models, reason)."""
    t = scan_outputs(ctx, None, splits)
    acc = accepted_models(ctx, splits, need)
    acc, dropped = revalidate(ctx, acc, t)
    if not t.empty:
        t["revalidated_out"] = t["model_key"].map(dropped).fillna("")
        t["used"] = t["model_key"].isin(acc) & t["accepted"]
    ctx.save_table(t, sid, "asr_output_gate")
    if len(acc) >= int(min_models):
        return True, acc, ""
    rej = t[~t["accepted"]] if not t.empty else t
    why = (f"needs >= {min_models} model(s) with real ASR {need if need != 'any' else 'transcript/phone'} outputs on "
           f"{list(splits)}; accepted: {acc or 'none'}")
    if dropped:
        why += "; no longer eligible: " + "; ".join(f"{k} ({v})" for k, v in dropped.items())
    if not rej.empty:
        why += "; rejected caches: " + "; ".join(f"{r.model_key}/{r.split} ({r.reason})" for r in rej.itertuples())
    elif t.empty:
        why += "; no ASR cache found (P24 has not produced inference output)"
    return False, acc, why


def output_state_token(ctx) -> str:
    """Fingerprint of accepted outputs (model, split, cache fingerprint, part files, failures) for P25."""
    from .checkpointing import ASRCheckpointStore

    base = Path(ctx.paths.v3_cache) / "asr"
    if not base.exists():
        return "none"
    accept_fx = bool(ctx.cfg.get_path("v3.asr.accept_fixture_outputs", False))
    items = []
    for k in sorted(p.name for p in base.glob("*") if p.is_dir()):
        st = ASRCheckpointStore(ctx.paths.v3_cache, k)
        for s in sorted(p.name for p in (base / k).glob("*") if p.is_dir()):
            origin, ok, _ = _origin(st.provenance(s), accept_fx)
            if not ok:
                continue
            d = st.store.dir(k, s)
            parts = sorted(f.name for f in d.glob("part-*.parquet"))
            fz = d / "_failures.parquet"
            items.append([k, s, origin, (st.store.fingerprint(k, s) or {}).get("fingerprint"), parts,
                          fz.stat().st_size if fz.exists() else 0])
    return stable_hash(items) if items else "none"


def usable_pairs(ctx, splits=("train", "test")) -> set[tuple[str, str]]:
    """{(model_key, split)} whose ASR output is accepted AND whose model still passes legal + provenance.
    Cached per context and output-state token (scanning reads the ASR parquet files)."""
    from .model_registry import document_for, reviews_path

    stamps = tuple((str(p), Path(p).stat().st_mtime_ns if p and Path(p).exists() else 0)
                   for p in (document_for(ctx).path, reviews_path(ctx)))      # a new review invalidates the cache
    key = ("v3_usable_asr_pairs", tuple(splits), output_state_token(ctx), stamps,
           bool(ctx.cfg.get_path("v3.asr.accept_fixture_outputs", False)))
    cache = getattr(ctx, "_cache", None)
    if cache is not None and key in cache:
        return cache[key]
    t = scan_outputs(ctx, None, splits)
    pairs: set[tuple[str, str]] = set()
    if not t.empty:
        acc = t[t["accepted"]]
        keep, _ = revalidate(ctx, sorted(set(acc["model_key"])), t)
        pairs = {(r.model_key, r.split) for r in acc.itertuples() if r.model_key in set(keep)}
    if cache is not None:
        cache[key] = pairs
    return pairs


def candidate_counts(ctx, splits) -> dict[str, int]:
    b = ctx.bundle()
    mc = ctx.cfg.get_path("v3.asr.max_clips")
    out = {}
    for s in splits:
        c = (b.train_clips if s == "train" else b.test_clips)[FILE].drop_duplicates()
        out[s] = int(min(len(c), int(mc))) if mc else int(len(c))
    return out


def execution_state(ctx, model_keys: list[str], splits=None) -> pd.DataFrame:
    splits = tuple(splits or ctx.cfg.get_path("v3.asr.splits", ["train"]) or ["train"])
    scan = scan_outputs(ctx, model_keys, splits)
    try:
        cands = candidate_counts(ctx, splits)
    except Exception:  # noqa: BLE001 - data not available: state without totals
        cands = {}
    rows = []
    for k in model_keys:
        for s in splits:
            r = scan[(scan["model_key"] == k) & (scan["split"] == s)] if not scan.empty else scan
            n_c = cands.get(s)
            if r.empty:
                rows.append({"model_key": k, "split": s, "state": "never_run", "rows": 0, "failures": 0,
                             "candidates": n_c, "coverage": 0.0, "origin": "none", "accepted": False})
                continue
            r = r.iloc[0]
            done = int(r["rows"]) + int(r["failures"])
            if r["origin"] == "none" and done == 0:
                # an empty cache directory (created by a store probe, or a run that never wrote a row) is not a
                # rejected output: nothing was ever produced for this model/split
                rows.append({"model_key": k, "split": s, "state": "never_run", "rows": 0, "failures": 0,
                             "candidates": n_c, "coverage": 0.0, "origin": "none", "accepted": False})
                continue
            if not r["accepted"] and r["origin"] not in ("p24_inference", "p24_inference_legacy"):
                state = "rejected"
            elif int(r["rows"]) == 0 and int(r["failures"]) > 0:
                state = "failed_only"
            elif n_c and done >= n_c:
                state = "complete"
            else:
                state = "partial"
            rows.append({"model_key": k, "split": s, "state": state, "rows": int(r["rows"]), "failures": int(r["failures"]),
                         "candidates": n_c, "coverage": (int(r["rows"]) / n_c) if n_c else None, "origin": r["origin"],
                         "accepted": bool(r["accepted"])})
    return pd.DataFrame(rows)


def load_accepted_asr(ctx, splits=("train",)) -> pd.DataFrame:
    """Long table (FILE, model_key, ASR columns, split) of accepted outputs only."""
    from .checkpointing import load_all_asr

    acc = accepted_models(ctx, splits)
    return load_all_asr(ctx.paths.v3_cache, splits, acc) if acc else pd.DataFrame(columns=[FILE, "model_key"])
