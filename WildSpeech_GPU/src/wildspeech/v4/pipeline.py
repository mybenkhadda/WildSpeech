"""V4Pipeline — resumable orchestration of the V4 research programme (Colab).

Modes (CONFIG["run_mode"]):
  PREPARE      metadata + audio manifests + external ASR import + joins -> manifests/, asr_import/
  FEATURES     acoustic reuse / missing extraction, recognized-text, disagreement, reliability, measured phonetic
               clip tables, providers, feature catalog -> clip_features/, student_features/
  EXPERIMENTS  M0-M8, ablations, controls, robustness, fallback audit, hypotheses, importance, champion, figures,
               report -> cv/, experiments/, ablations/, robustness/, diagnostics/, model_registry/, figures/, reports/
  SUBMISSION   champion refit + validated submission -> models/, submission/
  FULL         all of the above (submission only when generate_submission is True)
  SMOKE        PREPARE..EXPERIMENTS on a deterministic student subsample, 1 repeat (no submission)
Each step reads its inputs from the artefact store when they are not in memory, so a later session (or a later
mode) resumes without recomputation. Fingerprinted clip tables are reused when their inputs are unchanged; CV
results are cached by the V3Evaluator (content keys) and provider tables by CachedProvider.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .. import FILE, ID
from . import (FAM_ACOU, FAM_COR, FAM_DIS, FAM_PHON, FAM_REL, FAM_TASK, FAM_TEXT, FAM_UNC, NOT_EXECUTED,
               SCHEMA_VERSION, STAGE)
from .artifacts import ArtifactStore, env_report, fingerprint
from .config import build_v4_context, is_smoke, merged_config, resolve_asr_file


class V4IntegrityError(RuntimeError):
    pass


class V4Pipeline:
    def __init__(self, config: dict, project_root: Path | str | None = None, v4_overrides: dict | None = None):
        self.run_cfg = merged_config(config)
        self.ctx = build_v4_context(self.run_cfg, project_root, v4_overrides)
        self.c4 = self.ctx.cfg.get_path("v4", {}) or {}
        self.store = ArtifactStore(self.run_cfg["output_root"], str(self.c4.get("code_version", "v4.0.0")),
                                   resume=bool(self.run_cfg.get("resume", True)))
        self.log = self.ctx.logger
        self.S: dict = {}
        self.steps: list[dict] = []
        self.store.save_json({"config": self.run_cfg, "schema_version": SCHEMA_VERSION, "env": env_report(),
                              "paths": {k: str(v) for k, v in self.ctx.paths.files.items()},
                              "data_dir": str(self.ctx.paths.data_dir), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
                             "reports", "run_config.json")

    # ================================================================== helpers
    @property
    def exts(self):
        return tuple((self.c4.get("joins", {}) or {}).get("audio_extensions", (".flac", ".wav")))

    @property
    def splits(self) -> list[str]:
        return list(self.run_cfg.get("splits", ["train", "test"]))

    def _get(self, key: str, sub: str, name: str):
        if key not in self.S:
            t = self.store.load_json(sub, name) if name.endswith(".json") else self.store.load_table(sub, name)
            self.S[key] = t
        return self.S[key]

    def _step(self, name: str, fn):
        t0 = time.perf_counter()
        self.log.info("===== V4 %s", name)
        try:
            out = fn()
            self.steps.append({"step": name, "status": "ok", "seconds": time.perf_counter() - t0})
            return out
        except Exception as e:
            self.steps.append({"step": name, "status": "failed", "seconds": time.perf_counter() - t0,
                               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]})
            self.store.save_table(pd.DataFrame(self.steps), "reports", "run_steps.csv")
            raise

    # ================================================================== PREPARE
    def prepare(self) -> dict:
        from ..data.tar_manifest import build_tar_manifest
        from . import asr_import as AI
        from . import joins as J
        from .acoustic import acoustic_presence

        ctx, c4, b = self.ctx, self.c4, self.ctx.bundle()
        jc = c4.get("joins", {}) or {}
        meta = J.metadata_index(b, self.exts)
        # ---------------- audio manifests (local TAR headers + optional external manifest)
        mans = []
        if jc.get("build_tar_manifests", True):
            for s in ("train", "test"):
                key = f"{s}_audio"
                if key in ctx.paths.files:
                    m = build_tar_manifest(ctx.paths.files[key], ctx.paths.cache / "manifests", logger=self.log)
                    mans.append(J.normalize_manifest(m, s, self.exts, f"tar:{ctx.paths.files[key].name}"))
        ext_paths = J.find_external_manifests(Path(self.run_cfg["asr_root"]), jc.get("external_manifest_glob", ["*manifest*.parquet"]))
        ext = J.load_external_manifest(ext_paths, self.exts) if ext_paths else None
        if ext is not None and not ext.empty:
            mans.append(ext)
        man = pd.concat(mans, ignore_index=True) if mans else J.normalize_manifest(None, None, self.exts, "none")
        self.store.save_table(man, "manifests", "audio_manifest.parquet")
        # ---------------- external ASR import
        audit: list[dict] = []
        joined = pd.DataFrame(columns=["model_key", "split", FILE, ID, "asr_ok", "transcript", "has_phones"])
        coverage, usable, gate = pd.DataFrame(), [], pd.DataFrame()
        path, info = resolve_asr_file(self.run_cfg, self.log)
        report = {"resolution": info, "status": NOT_EXECUTED}
        if path is not None:
            res = AI.import_external_asr(path, c4, self.log)
            self.store.save_json(res.schema, "asr_import", "asr_schema_report.json")
            j, unmatched, aud = J.join_asr(res.canonical, meta, ext, int(jc.get("max_unmatched_examples", 25)))
            audit += aud
            joined, dups = AI.resolve_duplicates(j, ("model_key", "split", FILE),
                                                 (c4.get("asr_import", {}) or {}).get("duplicate_policy", "prefer_success_then_latest"))
            coverage = J.coverage_table(joined, unmatched, dups, meta)
            need_test = bool(self.run_cfg.get("generate_submission")) and "test" in self.splits
            gate, usable = AI.validate_import(joined, coverage, res.provenance, c4, tuple(self.splits), need_test)
            AI.write_external_store(ctx.paths.cache / "v4" / "asr_external", joined, res.provenance, gate)
            self.store.save_table(joined, "asr_import", "canonical_predictions.parquet", fingerprint=res.provenance["source_sha256"])
            self.store.save_table(unmatched, "asr_import", "unmatched_predictions.parquet")
            self.store.save_table(dups, "asr_import", "duplicates.parquet")
            self.store.save_table(joined[~joined["asr_ok"].astype(bool)], "asr_import", "failed_inference.parquet")
            self.store.save_table(gate, "asr_import", "import_gate.csv")
            self.store.save_table(coverage, "asr_import", "asr_coverage.csv")
            self.store.save_table(coverage, "manifests", "asr_model_coverage.csv")
            ok, why = AI.permission_status(c4)
            report = {**{k: v for k, v in res.provenance.items() if k != "roles"}, "roles": res.schema["roles"],
                      "resolution": info, "status": "IMPORTED", "usable_models": usable, "permission_status": why,
                      "n_joined": int(len(joined)), "n_unmatched": int(len(unmatched)), "n_duplicates": int(len(dups)),
                      "n_failed": int((~joined["asr_ok"].astype(bool)).sum()), "schema_warnings": res.schema["warnings"]}
        self.store.save_json(report, "asr_import", "asr_import_report.json")
        # ---------------- masters + integrity
        fams = list((c4.get("acoustic", {}) or {}).get("families", []))
        pres = acoustic_presence(ctx, fams)
        cm = J.build_clip_master(meta, man, joined, usable, pres)
        sm = J.build_student_master(b, cm)
        summary, rows = J.data_integrity(b, meta, man, joined, (c4.get("joins", {}) or {}).get("expected_counts"))
        ja = pd.DataFrame(audit + [{**r, "check": "integrity: " + r["check"]} for r in rows])
        self.store.save_table(cm, "manifests", "clip_master.parquet")
        self.store.save_table(sm, "manifests", "student_master.parquet")
        self.store.save_table(ja, "manifests", "join_audit.csv")
        self.store.save_json(summary, "manifests", "data_integrity.json")
        self.S.update({"joined": joined, "coverage": coverage, "usable": usable, "clip_master": cm})
        hard = [r for r in rows if r["status"] == "FAIL" and ("one-to-one" in r["check"] or "disjoint" in r["check"]
                                                              or "unique" in r["check"])]
        if hard:
            raise V4IntegrityError("data integrity FAIL: " + "; ".join(r["check"] for r in hard) +
                                   " (see manifests/data_integrity.json)")
        return {"n_models": int(joined["model_key"].nunique()) if not joined.empty else 0, "usable_models": usable,
                "integrity_fail": summary["n_fail"]}

    # ================================================================== FEATURES
    def _joined(self) -> pd.DataFrame:
        j = self._get("joined", "asr_import", "canonical_predictions.parquet")
        return j if j is not None else pd.DataFrame(columns=["model_key", "split", FILE, ID, "asr_ok", "transcript"])

    def _usable(self) -> list[str]:
        if "usable" not in self.S:
            rep = self.store.load_json("asr_import", "asr_import_report.json", {}) or {}
            self.S["usable"] = list(rep.get("usable_models") or [])
        return self.S["usable"]

    def features(self) -> dict:
        from ..phonetic.g2p import G2P
        from . import acoustic as AC
        from . import phonetic_features as PF
        from . import text_features as TF

        ctx, c4 = self.ctx, self.c4
        ac = c4.get("acoustic", {}) or {}
        fams = list(ac.get("families", []))
        # ---------------- acoustic
        imp = AC.import_cached_families(ctx, fams, list(ac.get("search_dirs") or []), self.splits) \
            if self.run_cfg.get("reuse_existing_features", True) else pd.DataFrame()
        self.store.save_table(imp, "clip_features", "acoustic_cache_reuse.csv")
        inv = AC.inventory(ctx, fams, self.splits)
        ext_log = []
        if self.run_cfg.get("compute_missing_acoustic_features", True) and not inv.empty and (inv["n_missing"] > 0).any():
            need = sorted(set(inv.loc[inv["n_missing"] > 0, "split"]))
            ext_log = AC.extract_missing(ctx, fams, need, Path(self.run_cfg["local_work_root"]))
            inv = AC.inventory(ctx, fams, self.splits)
        self.store.save_table(inv, "clip_features", "acoustic_inventory.csv")
        self.store.save_table(pd.DataFrame(ext_log), "clip_features", "acoustic_extraction_log.csv")
        acou = AC.load_acoustic_clip_table(ctx, fams, ac.get("variant"), tuple(self.splits))
        acou_cols = AC.speech_columns(acou, ac.get("core_features") or {})
        self.store.save_json({"source": acou.attrs.get("source"), "columns": acou_cols, "n_clips": int(len(acou))},
                             "clip_features", "acoustic_columns.json")
        # ---------------- recognized text
        joined, usable = self._joined(), self._usable()
        cov = self._get("coverage", "asr_import", "asr_coverage.csv")
        tc = c4.get("text", {}) or {}
        g2p = G2P((c4.get("phonetic", {}) or {}).get("reference_g2p_backend", "auto"))
        sha = (self.store.load_json("asr_import", "asr_import_report.json", {}) or {}).get("source_sha256")
        fp_text = fingerprint("text", sha, usable, tc, c4.get("code_version"))
        notes = {}
        if usable and not joined.empty:
            if self.store.is_fresh("clip_features", "clip_text_long.parquet", fp_text):
                text_long = self.store.load_table("clip_features", "clip_text_long.parquet")
            else:
                text_long = TF.build_clip_text_table(joined, usable, tc, g2p)
                self.store.save_table(text_long, "clip_features", "clip_text_long.parquet", fingerprint=fp_text)
            feat_cols = list(tc.get("core_features", [])) + [f for v in (tc.get("task_features") or {}).values() for f in v] \
                + list(tc.get("uncertainty_features", []))
            if tc.get("include_g2p_derived"):
                feat_cols += [c for c in text_long.columns if c.startswith("g2pd_")]
            text_cons = TF.consensus_table(text_long, feat_cols, tc.get("primary_strategy", "consensus"), cov)
            dis = TF.disagreement_table(joined, text_long, usable)
            rel = TF.reliability_table(joined, usable, dis, text_long)
            ex = TF.transcript_examples(joined)
        else:
            text_long = text_cons = dis = rel = ex = pd.DataFrame(columns=[FILE])
            notes["text"] = "NOT_EXECUTED: no ASR model passed the import gate (see asr_import/import_gate.csv)"
        self.store.save_table(text_cons, "clip_features", "clip_text_consensus.parquet", fingerprint=fp_text)
        self.store.save_table(dis, "clip_features", "clip_disagreement.parquet", fingerprint=fp_text)
        self.store.save_table(rel, "clip_features", "clip_reliability.parquet", fingerprint=fp_text)
        self.store.save_table(ex, "clip_features", "transcript_examples.csv")
        # ---------------- measured phonetics
        pc = c4.get("phonetic", {}) or {}
        pm, avail = PF.availability(cov if cov is not None else pd.DataFrame(), usable, pc, g2p) if usable \
            else ([], pd.DataFrame([{"model_key": "*", "status": "UNAVAILABLE", "reason": "no usable ASR model"}]))
        self.store.save_table(avail, "clip_features", "phonetic_availability.csv")
        phone_long = phone_cons = pd.DataFrame(columns=[FILE])
        if pm:
            phone_long = PF.build_clip_phone_table(joined, pm, g2p, int(pc.get("position_bins", 3)))
            phone_cons = PF.consensus_phone_table(phone_long, list(pc.get("core_features", [])))
            conf = PF.confusion_matrix(phone_long)
            if not conf.empty:
                conf.index.name = "ref"
                self.store.save_table(conf.reset_index(), "clip_features", "phoneme_confusion.csv")
        self.store.save_table(phone_long, "clip_features", "clip_phone_long.parquet")
        self.store.save_table(phone_cons, "clip_features", "clip_phone_consensus.parquet")
        self.S.update({"acou": acou, "acou_cols": acou_cols, "text_long": text_long, "text_cons": text_cons, "dis": dis,
                       "rel": rel, "phone_long": phone_long, "phone_cons": phone_cons, "phone_models": pm,
                       "examples": ex, "acou_inventory": inv})
        # ---------------- providers + catalog
        P = self.providers(rebuild=True)
        cat, desc = self._catalog(P["providers"])
        self.store.save_table(cat, "student_features", "feature_catalog.parquet", csv=True)
        self.store.save_table(desc.reset_index(), "student_features", "student_features_descriptive_train.parquet")
        self.store.save_json(notes, "student_features", "feature_notes.json")
        # refresh the clip master with acoustic presence
        from . import joins as J

        cm = self._get("clip_master", "manifests", "clip_master.parquet")
        if cm is not None:
            cm = cm[[c for c in cm.columns if not c.startswith("acoustic_present__")]]
            cm = cm.merge(AC.acoustic_presence(ctx, fams), on=FILE, how="left")
            self.store.save_table(cm, "manifests", "clip_master.parquet")
            self.store.save_table(J.build_student_master(ctx.bundle(), cm), "manifests", "student_master.parquet")
            sr = J.sample_rate_audit(joined, AC.file_sr_series(acou), ctx.bundle().all_students) if not joined.empty else []
            if sr:
                ja = self.store.load_table("manifests", "join_audit.csv")
                ja = pd.concat([ja if ja is not None else pd.DataFrame(), pd.DataFrame(sr)], ignore_index=True)
                if "check" in ja:
                    key = ["check"] + (["model_key"] if "model_key" in ja else [])
                    ja = ja.drop_duplicates(subset=key, keep="last")
                self.store.save_table(ja, "manifests", "join_audit.csv")
        return {"text_models": usable, "phone_models": pm, "acoustic_columns": len(acou_cols),
                "providers": sorted(P["providers"])}

    def _load_feature_state(self) -> None:
        from . import acoustic as AC

        for k, name in (("text_long", "clip_text_long.parquet"), ("text_cons", "clip_text_consensus.parquet"),
                        ("dis", "clip_disagreement.parquet"), ("rel", "clip_reliability.parquet"),
                        ("phone_long", "clip_phone_long.parquet"), ("phone_cons", "clip_phone_consensus.parquet"),
                        ("examples", "transcript_examples.csv")):
            if k not in self.S:
                t = self.store.load_table("clip_features", name)
                self.S[k] = t if t is not None else pd.DataFrame(columns=[FILE])
        if "acou" not in self.S:
            ac = self.c4.get("acoustic", {}) or {}
            meta = self.store.load_json("clip_features", "acoustic_columns.json", {}) or {}
            self.S["acou"] = AC.load_acoustic_clip_table(self.ctx, list(ac.get("families", [])), ac.get("variant"),
                                                         tuple(self.splits))
            self.S["acou_cols"] = meta.get("columns") or AC.speech_columns(self.S["acou"], ac.get("core_features") or {})
        if "phone_models" not in self.S:
            av = self.store.load_table("clip_features", "phonetic_availability.csv")
            self.S["phone_models"] = sorted(av.loc[av["status"] == "AVAILABLE", "model_key"]) if av is not None and "status" in av else []

    # ================================================================== providers
    def providers(self, rebuild: bool = False) -> dict:
        if "P" in self.S and not rebuild:
            return self.S["P"]
        from ..evaluation.robustness import recording_provider
        from .aggregation import ReliabilityProvider, V4ClipAggregateProvider, cached
        from .text_features import TASK_PREFIX

        self._load_feature_state()
        ctx, c4, S = self.ctx, self.c4, self.S
        ag = c4.get("aggregation", {}) or {}
        tc = c4.get("text", {}) or {}
        controls = tuple(ag.get("controls", ["item", "domain", "duration", "complexity"]))
        stats, cstats = tuple(ag.get("stats", [])), tuple(ag.get("compact_stats", ["mean", "sd", "coverage"]))
        kw = dict(controls=controls, shrinkage_k=float(ag.get("shrinkage_k", 10.0)), min_clips=int(ag.get("min_clips", 2)),
                  max_columns=int(ag.get("max_raw_features_per_family", 400)))
        sha = (self.store.load_json("asr_import", "asr_import_report.json", {}) or {}).get("source_sha256")
        ver = fingerprint(sha, self._usable(), c4.get("code_version"))
        tcons, rel = S["text_cons"], S["rel"]
        relw = rel.set_index(FILE)["rel_agree"] if (rel is not None and "rel_agree" in rel and rel["rel_agree"].notna().any()) else \
            (rel.set_index(FILE)["rel_frac_ok"] if rel is not None and "rel_frac_ok" in rel else None)
        core = [c for c in tc.get("core_features", []) if c in tcons.columns]
        task_feats = [f for v in (tc.get("task_features") or {}).values() for f in v if f in tcons.columns]
        unc = [c for c in tc.get("uncertainty_features", []) if c in tcons.columns and tcons[c].notna().any()]
        P: dict = {}

        def add(name, prov):
            if prov is not None and prov.available():
                P[name] = cached(ctx, prov)

        if core:
            add(FAM_TEXT, V4ClipAggregateProvider(ctx, FAM_TEXT, FAM_TEXT, tcons, core, stats=stats, reliability=relw,
                                                  contrast_n=int(ag.get("contrast_features", 3)), version=ver,
                                                  description="recognized-text deviation (consensus)", **kw))
        if task_feats:
            add(FAM_TASK, V4ClipAggregateProvider(ctx, FAM_TASK, FAM_TASK, tcons, task_feats, stats=cstats, reliability=relw,
                                                  contrast_n=0, version=ver, description="task-specific text deviation", **kw))
        if unc:
            add(FAM_UNC, V4ClipAggregateProvider(ctx, FAM_UNC, FAM_UNC, tcons, unc, stats=cstats, contrast_n=0, version=ver,
                                                 description="recognizer uncertainty", **kw))
        dis = S["dis"]
        dcols = [c for c in tc.get("disagreement_features", []) if c in dis.columns]
        if dcols and len(dis):
            add(FAM_DIS, V4ClipAggregateProvider(ctx, FAM_DIS, FAM_DIS, dis, dcols, stats=cstats, contrast_n=0, version=ver,
                                                 description="cross-recognizer disagreement", **kw))
        if rel is not None and len(rel) and "rel_frac_ok" in rel:
            add(FAM_REL, ReliabilityProvider(ctx, FAM_REL, rel, version=ver))
        pc = c4.get("phonetic", {}) or {}
        pcons = S["phone_cons"]
        pcols = [c for c in pc.get("core_features", []) if c in pcons.columns]
        if S.get("phone_models") and pcols:
            add(FAM_PHON, V4ClipAggregateProvider(ctx, FAM_PHON, FAM_PHON, pcons, pcols, stats=stats, version=ver,
                                                  description="MEASURED phonetic errors", **kw))
        acou, acols = S["acou"], S["acou_cols"]
        aver = fingerprint("acou", acols, len(acou))
        if acols:
            add(FAM_ACOU, V4ClipAggregateProvider(ctx, FAM_ACOU, FAM_ACOU, acou, acols, stats=cstats, contrast_n=2,
                                                  version=aver, description="item-conditioned, domain-controlled acoustics", **kw))
        # correct-response-only (H3)
        cor_parts = []
        if core:
            cor_parts.append(tcons[[FILE] + core[:6]])
        if acols:
            cor_parts.append(acou[[FILE] + acols[:6]])
        correct_names = []
        if cor_parts:
            cor = cor_parts[0]
            for p in cor_parts[1:]:
                cor = cor.merge(p, on=FILE, how="outer")
            ccols = [c for c in cor.columns if c != FILE]
            add(FAM_COR, V4ClipAggregateProvider(ctx, FAM_COR, FAM_COR, cor, ccols, stats=("mean", "sd", "coverage"),
                                                 correct_only=True, contrast_n=0, version=fingerprint(ver, aver),
                                                 description="officially correct responses only", **kw))
            correct_names = [FAM_COR] if FAM_COR in P else []
        # per-task (H2)
        task_providers: dict[str, list[str]] = {}
        for t, pre in TASK_PREFIX.items():
            names = []
            tf = [f for f in (tc.get("task_features") or {}).get(t, []) if f in tcons.columns]
            if core:
                n = f"TXT{pre.strip('_').upper()}"
                add(n, V4ClipAggregateProvider(ctx, n, n, tcons, core + tf, stats=cstats, tasks=[t], contrast_n=0,
                                               version=ver, description=f"recognized text, {t} only", **kw))
                names.append(n)
            if acols:
                n = f"AC{pre.strip('_').upper()}"
                add(n, V4ClipAggregateProvider(ctx, n, n, acou, acols, stats=cstats, tasks=[t], contrast_n=0,
                                               version=aver, description=f"acoustics, {t} only", **kw))
                names.append(n)
            task_providers[t] = [x for x in names if x in P]
        # per ASR model and leave-one-model-out (ASR-model ablations)
        model_providers, loo_providers = {}, {}
        tl = S["text_long"]
        models = sorted(tl["model_key"].unique()) if (tl is not None and "model_key" in tl and len(tl)) else []
        for i, m in enumerate(models):
            d = tl[(tl["model_key"] == m) & tl["asr_ok"].astype(bool)][[FILE] + [c for c in core if c in tl]]
            n = f"TXTM{i}"
            add(n, V4ClipAggregateProvider(ctx, n, n, d, core, stats=cstats, contrast_n=0, version=fingerprint(ver, m),
                                           description=f"recognized text, model {m} only", **kw))
            if n in P:
                model_providers[m] = n
            if len(models) >= 3:
                o = tl[(tl["model_key"] != m) & tl["asr_ok"].astype(bool)].groupby(FILE)[core].mean().reset_index()
                n2 = f"TXTL{i}"
                add(n2, V4ClipAggregateProvider(ctx, n2, n2, o, core, stats=cstats, contrast_n=0,
                                                version=fingerprint(ver, "loo", m),
                                                description=f"recognized text, consensus without {m}", **kw))
                if n2 in P:
                    loo_providers[m] = n2
        rec = recording_provider(ctx)
        if rec is not None and rec.available():
            P["REC"] = rec
        out = {"providers": P, "task_providers": task_providers, "model_providers": model_providers,
               "loo_providers": loo_providers, "correct_providers": correct_names,
               "model_index": {f"TXTM{i}": m for i, m in enumerate(models)}}
        self.S["P"] = out
        return out

    def _catalog(self, P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Feature catalog (family, provider, modality, measured/derived) from provider tables fitted on ALL
        training students — used for description only; CV always refits providers per fold."""
        ids = sorted(self.ctx.bundle().y.index.astype(str))
        modality = {FAM_TEXT: "asr_text", FAM_TASK: "asr_text", FAM_UNC: "asr_uncertainty", FAM_DIS: "asr_disagreement",
                    FAM_REL: "asr_reliability", FAM_PHON: "phonetic", FAM_ACOU: "acoustic", FAM_COR: "text+acoustic",
                    "REC": "recording_control"}
        measurement = {FAM_PHON: "measured phone recognizer output", FAM_ACOU: "measured acoustics",
                       "REC": "measured recording conditions"}
        rows, tabs = [], []
        for name, p in P.items():
            try:
                T = p.table(ids, ids)
            except Exception as e:  # noqa: BLE001 - catalogued as failed delivery
                rows.append({"provider": name, "feature": None, "family": p.family, "modality": "?", "status": f"failed: {e}"})
                continue
            tabs.append(T)
            for c in T.columns:
                fam = c.split("__", 1)[0]
                rows.append({"provider": name, "feature": c, "family": fam,
                             "modality": modality.get(name, "asr_text" if name.startswith("TXT") else
                                                      ("acoustic" if name.startswith("AC") else "other")),
                             "measurement": measurement.get(name, "recognized-text deviation (NOT a pronunciation measure)"
                                                            if name.startswith(("ASRTXT", "ASRTASK", "TXT", "ASRDIS", "ASRUNC")) else "derived"),
                             "scope": c.split("__")[1] if c.count("__") >= 2 else "all",
                             "stat": c.split("__")[-1], "non_missing_frac": float(T[c].notna().mean()),
                             "label_free": True, "status": "ok", "description": getattr(p.p, "description", "")})
        desc = pd.concat(tabs, axis=1) if tabs else pd.DataFrame(index=pd.Index(ids, name=ID))
        desc = desc.loc[:, ~desc.columns.duplicated()]
        return pd.DataFrame(rows), desc

    # ================================================================== EXPERIMENTS
    def evaluator(self):
        if "ev" not in self.S:
            from .models import V4Evaluator

            P = self.providers()
            self.S["ev"] = V4Evaluator(self.ctx, STAGE, P["providers"])
        return self.S["ev"]

    def submittable_providers(self) -> set[str]:
        P = self.providers()["providers"]
        ok = set()
        inv = self.store.load_table("clip_features", "acoustic_inventory.csv")
        min_cov = float((self.c4.get("submission", {}) or {}).get("min_test_coverage", 0.8))
        acou_ok = inv is not None and not inv.empty and \
            bool((inv[(inv["split"] == "test") & (inv["n_cached"] > 0)]["coverage"] >= min_cov).any())
        rep = self.store.load_json("asr_import", "asr_import_report.json", {}) or {}
        asr_ok = bool(rep.get("usable_models"))
        cov = self._get("coverage", "asr_import", "asr_coverage.csv")
        if asr_ok and cov is not None and not cov.empty:
            t = cov[(cov["split"] == "test") & cov["model_key"].isin(rep["usable_models"])]
            asr_ok = bool((t["usable_frac"] >= min_cov).any())
        for n in P:
            if n == "REC":
                continue                                           # train-only control provider
            if n.startswith(("ACOU", "AC")) and not n.startswith("ASR"):
                if acou_ok:
                    ok.add(n)
            elif n == FAM_COR:
                if acou_ok and asr_ok:
                    ok.add(n)
            elif asr_ok:
                ok.add(n)
        return ok

    def experiments(self) -> dict:
        from ..v3.common import available_schemes
        from .experiments import (ExperimentRunner, SpecFactory, fallback_report, finalize, paired_fold_deltas,
                                  select_champion, stability_table)

        ctx, c4 = self.ctx, self.c4
        vc = c4.get("validation", {}) or {}
        alpha = float(vc.get("fdr_alpha", 0.05))
        n_boot = int(vc.get("bootstrap", 1000))
        primary = str(vc.get("primary", "repeated_kfold"))
        ev = self.evaluator()
        P = self.providers()
        fac = SpecFactory(ctx.cfg, P["providers"], ev.baseline)
        base_specs = fac.baselines()
        ref = next(s for s in base_specs if s.group == "baseline")
        m0 = {s.model: s for s in base_specs if s.base_families == ref.base_families}
        runner = ExperimentRunner(ev, self.log, n_boot)
        self.S["runner"] = runner
        main = finalize(runner.run(base_specs + fac.ladder(), primary, ref, m0), alpha)
        h = c4.get("hier_net", {}) or {}
        if h.get("enabled"):
            gained = main is not None and not main.empty and bool((main.get("result_class") == "GENUINE_GAIN").any())
            if gained or not h.get("require_gain_gate", True):
                main = pd.concat([main, finalize(runner.run(fac.hier_net(), primary, ref, m0), alpha)], ignore_index=True)
            else:
                main = pd.concat([main, pd.DataFrame([{"name": "M8_HIER_NET", "experiment": "M8", "group": "multimodal",
                                                       "scheme": primary, "status": "not_run",
                                                       "reason": "gate: no M1-M7 row is a GENUINE_GAIN"}])], ignore_index=True)
        abl = finalize(runner.run(fac.ablations(P["task_providers"], P["model_providers"], P["loo_providers"],
                                                P["correct_providers"]) + fac.controls(), primary, ref, m0), alpha)
        # ---------------- robustness on the other schemes (identical specs, baseline refitted per scheme)
        schemes = available_schemes(ctx, list(vc.get("robustness_schemes", [])))
        key_names = {ref.name}
        for pre in ("M1_TEXT_", "M2_ACOU_", "M3_PHON_", "M4_TEXT_ACOU_", "M5_RESID_ALL", "M7_GATED_MOE"):
            r = main[main["name"].astype(str).str.startswith(pre)] if not main.empty else main
            if not r.empty:
                key_names.add(r.sort_values("r2_mean", ascending=False)["name"].iloc[0])
        all_specs = {s.name: s for s in base_specs + fac.ladder() + fac.hier_net() + fac.controls()
                     + fac.ablations(P["task_providers"], P["model_providers"], P["loo_providers"], P["correct_providers"])}
        fam_single = [n for n in all_specs if n.startswith("ABL_FAMILY_")]
        rob_specs = [all_specs[n] for n in sorted(key_names | set(fam_single)) if n in all_specs and n != ref.name]
        rob_parts = [finalize(runner.run(rob_specs, sc, ref, None), alpha) for sc in schemes]
        rob = pd.concat([r for r in rob_parts if r is not None and not r.empty], ignore_index=True) if rob_parts else pd.DataFrame()
        # ---------------- exports
        allrows = pd.concat([x for x in (main, abl, rob) if x is not None and not x.empty], ignore_index=True)
        oof = runner.oof_long()
        traces = runner.traces()
        fb = fallback_report(allrows, traces)
        st = stability_table(oof, sorted(key_names), primary) if not oof.empty else pd.DataFrame()
        pdlt = paired_fold_deltas(oof, ref.name, sorted(key_names - {ref.name}), primary) if not oof.empty else pd.DataFrame()
        self.store.save_table(allrows, "cv", "cv_results.parquet", csv=True)
        self.store.save_table(oof, "cv", "oof_predictions.parquet")
        self.store.save_table(traces, "cv", "fold_traces.parquet")
        self.store.save_table(st, "cv", "fold_stability.parquet", csv=True)
        self.store.save_table(pdlt, "cv", "paired_fold_deltas.parquet", csv=True)
        self.store.save_table(main, "experiments", "experiment_results.csv")
        self.store.save_table(abl, "ablations", "ablation_results.csv")
        self.store.save_table(rob, "robustness", "robustness_results.csv")
        if not rob.empty:
            from ..evaluation.robustness import transfer_table

            self.store.save_table(transfer_table(pd.concat([main, rob], ignore_index=True)), "robustness", "transfer_table.csv")
        self.store.save_table(fb, "diagnostics", "fallback_report.csv")
        self.store.save_table(ev.reg.summary(), "cv", "fold_registry_summary.csv")
        # ---------------- champion
        sub_ok = self.submittable_providers()
        champ = select_champion(allrows, primary, ref.name, sub_ok,
                                bool((c4.get("submission", {}) or {}).get("require_validated_gain", True)))
        cspec = all_specs.get(champ["name"], ref)
        champ.update({"spec": cspec.as_dict(), "baseline": ev.baseline, "primary_scheme": primary,
                      "submittable_providers": sorted(sub_ok), "provider_fingerprints":
                      {p: P["providers"][p].fingerprint() for p in cspec.providers if p in P["providers"]},
                      "fold_registry_hash": ev.reg.data_hash(), "schema_version": SCHEMA_VERSION})
        self.store.save_json(champ, "model_registry", "champion_spec.json")
        spec_rows = [{k: str(v) for k, v in s.as_dict().items()} for s in all_specs.values()]
        self.store.save_table(pd.DataFrame(spec_rows), "model_registry", "experiment_specs.csv")
        self.S.update({"main": main, "abl": abl, "rob": rob, "allrows": allrows, "oof": oof, "traces": traces,
                       "champion": champ, "ref": ref, "all_specs": all_specs})
        # ---------------- hypotheses + importance
        self._step("hypotheses", lambda: self.hypotheses(primary, schemes, n_boot))
        self._step("importance", lambda: self.importance(primary))
        return {"n_rows": int(len(allrows)), "champion": champ["name"], "champion_is_baseline": champ["is_baseline"]}

    def hypotheses(self, primary: str, schemes: list[str], n_boot: int) -> pd.DataFrame:
        from ..discovery.asr_disagreement import variance_decomposition
        from ..discovery.item_conditioning import v3_clip_frame
        from ..evaluation.robustness import confounding_table, selection_stability
        from ..v3.common import student_meta
        from ..validation.metrics import mean_oof
        from . import hypotheses as H

        ctx, S, ev = self.ctx, self.S, self.evaluator()
        runner, ref, oof = S["runner"], S["ref"], S["oof"]
        P = self.providers()["providers"]
        clipf = v3_clip_frame(ctx)
        extra: dict = {"phonetic_available": bool(S.get("phone_models"))}
        dis = S.get("dis")
        if dis is not None and len(dis) and "dis_word_mean" in dis:
            extra["disagreement_variance"] = variance_decomposition(dis, clipf, "dis_word_mean")
            extra["disagreement_split_half"] = H.split_half(dis, clipf, "dis_word_mean")
            self.store.save_table(extra["disagreement_variance"], "diagnostics", "disagreement_variance_decomposition.csv")
        if extra["phonetic_available"]:
            extra["phonetic_split_half"] = H.split_half(S["phone_cons"], clipf, "ph_er")
        rr = runner.results
        if ("CTRL_RECORDING_ONLY", primary) in rr and ("H6_TEXT_PLUS_REC", primary) in rr:
            extra["h6_attenuation"] = ev.compare(rr[("CTRL_RECORDING_ONLY", primary)], rr[("H6_TEXT_PLUS_REC", primary)], n_boot)
        base_oof = runner.base[primary].oof
        if not base_oof.empty:
            m = mean_oof(base_oof)
            m.index = m.index.astype(str)
            resid = (m["y_true"] - m["y_pred"]).rename("resid")
            fams = {k: v for k, v in P.items() if k in (FAM_TEXT, FAM_TASK, FAM_DIS, FAM_ACOU, FAM_PHON, "REC")}
            conf = confounding_table(ev, fams, resid, student_meta(ctx))
            self.store.save_table(conf, "diagnostics", "confounding.csv")
            S["confounding"] = conf
            extra["confounding_attenuation"] = float(conf["attenuation"].median()) if not conf.empty else None
        extra["residual_r2_explained"] = H.residual_explained(oof, ref.name, "M5_RESID_ALL", primary) if not oof.empty else None
        tr = S["traces"]
        if tr is not None and not tr.empty and "alpha" in tr:
            t5 = tr[(tr["name"] == "M5_RESID_ALL") & (tr["scheme"] == primary)]
            extra["m5_alpha_zero_folds"] = int((pd.to_numeric(t5["alpha"], errors="coerce") == 0).sum())
        champ = S["champion"]
        mm = champ["name"]
        if champ["is_baseline"]:
            cand = S["main"][S["main"]["group"] == "multimodal"].dropna(subset=["delta_r2"]) if not S["main"].empty else pd.DataFrame()
            mm = cand.sort_values("delta_r2", ascending=False)["name"].iloc[0] if not cand.empty else ref.name
        h8, tab = H.failure_phenotypes(oof, ref.name, mm, primary, student_meta(ctx), n_boot)
        extra["h8"] = h8
        if not tab.empty:
            self.store.save_table(tab.assign(model=mm), "diagnostics", "failure_phenotypes.csv")
            S["phenotype_profiles"] = tab.attrs.get("profiles")
            S["phenotype_table"] = tab
        if ("H9_INTERACT_RIDGE", primary) in rr and ("H9_INTERACT_CATBOOST", primary) in rr:
            extra["h9"] = {**ev.compare(rr[("H9_INTERACT_RIDGE", primary)], rr[("H9_INTERACT_CATBOOST", primary)], n_boot),
                           "compared": "CATBOOST vs RIDGE on identical inputs"}
        stab = {}
        for n, sp in S["all_specs"].items():
            if n.startswith("ABL_FAMILY_"):
                try:
                    _, s = selection_stability(ev, sp, primary, int(ctx.cfg.get_path("v3.ablation.stability_top_k", 20)))
                    stab[n.replace("ABL_FAMILY_", "")] = s.get("jaccard_mean")
                except Exception as e:  # noqa: BLE001 - recorded
                    self.log.warning("selection stability %s failed: %s", n, e)
        extra["h10"] = H.family_generalization(S["allrows"], [primary] + list(schemes), stab)
        n_st = H.n_students(oof, primary)
        hyp = H.evaluate(S["allrows"], primary, list(schemes), n_st, extra)
        self.store.save_table(hyp, "reports", "hypotheses.csv")
        self.store.save_table(hyp, "experiments", "hypothesis_results.csv")
        if isinstance(extra["h10"], pd.DataFrame):
            self.store.save_table(extra["h10"], "robustness", "family_generalization.csv")
        S["hyp"] = hyp
        return hyp

    def importance(self, primary: str) -> None:
        from .importance import grouped_permutation_importance, model_importance, summarize_grouped

        S, ev, c4 = self.S, self.evaluator(), self.c4
        vc = c4.get("validation", {}) or {}
        champ = S["champion"]
        name = champ["name"]
        if champ["is_baseline"]:
            cand = S["main"][(S["main"]["group"] == "multimodal") & (S["main"]["status"] == "ok")] if not S["main"].empty else pd.DataFrame()
            name = cand.sort_values("r2_mean", ascending=False)["name"].iloc[0] if not cand.empty else name
        spec = S["all_specs"].get(name, S["ref"])
        imp = grouped_permutation_importance(ev, spec, primary, int(vc.get("importance_repeats", 1)),
                                             int(vc.get("importance_permutations", 5)), self.ctx.seed)
        summ = summarize_grouped(imp)
        fmod, Xtr, _ = ev.fit_full_v3(spec, ev.all_train)
        mi = model_importance(fmod, Xtr)
        out = pd.concat([imp.assign(kind="grouped_permutation_fold"), mi.assign(kind="champion_refit", name=spec.name)],
                        ignore_index=True)
        self.store.save_table(out, "diagnostics", "feature_importance.parquet")
        self.store.save_table(out, "models", "feature_importance.parquet")
        self.store.save_table(summ.assign(name=spec.name), "diagnostics", "feature_importance_grouped.csv")
        S["imp_grouped"], S["imp_model"] = summ, mi

    # ================================================================== FIGURES
    def figures(self) -> dict:
        from ..discovery.item_conditioning import v3_clip_frame
        from .experiments import stability_table
        from .figures import FigureMaker, outcome_frame, student_descriptives

        self._load_feature_state()
        S, st = self.S, self.store
        fm = FigureMaker(st, int(self.ctx.cfg.get_path("v3.project.figure_dpi", 150)))
        b = self.ctx.bundle()
        clipf = v3_clip_frame(self.ctx)
        cov = self._get("coverage", "asr_import", "asr_coverage.csv")
        cm = self._get("clip_master", "manifests", "clip_master.parquet")
        main = S.get("main") if S.get("main") is not None else st.load_table("experiments", "experiment_results.csv")
        abl = S.get("abl") if S.get("abl") is not None else st.load_table("ablations", "ablation_results.csv")
        rob = S.get("rob") if S.get("rob") is not None else st.load_table("robustness", "robustness_results.csv")
        oof = S.get("oof") if S.get("oof") is not None else st.load_table("cv", "oof_predictions.parquet")
        champ = S.get("champion") or st.load_json("model_registry", "champion_spec.json", {}) or {}
        conf = S.get("confounding") if S.get("confounding") is not None else st.load_table("diagnostics", "confounding.csv")
        y = b.y.copy()
        y.index = y.index.astype(str)
        fm.run("01_asr_model_coverage", fm.asr_coverage, cov)
        fm.run("02_audio_text_alignment", fm.join_alignment, cm)
        fm.run("03_wer_cer_by_task", fm.wer_cer_by_task, S["text_long"])
        fm.run("04_transcript_examples", fm.transcript_examples, S["examples"])
        fm.run("05_phoneme_confusion", fm.phoneme_confusion, st.load_table("clip_features", "phoneme_confusion.csv"),
               bool(S.get("phone_models")))
        P = self.providers()["providers"]
        ids = sorted(y.index)
        desc = student_descriptives(P, ids, [n for n in (FAM_TEXT, FAM_ACOU) if n in P]).join(outcome_frame(b), how="left")
        wer_col = next((c for c in desc.columns if c.startswith(f"{FAM_TEXT}__all__w_er") and c.endswith("__mean")), None)
        sp_col = next((c for c in desc.columns if c.startswith(f"{FAM_ACOU}__all__") and "speech_ratio" in c and c.endswith("__mean")), None)
        fm.run("06_screening_vs_linguistic", fm.scatter_pair, desc, "theta_mean", wer_col or "missing",
               "Screening theta vs recognized-text WER (item-conditioned z)", "mean screening theta",
               "student mean WER z (item-conditioned)", "instrument",
               "Descriptive: provider fitted on all training students.")
        fm.run("07_speech_vs_outcome", fm.scatter_pair, desc, wer_col or "missing", "outcome_score",
               "Recognized-text WER vs end-of-year outcome", "student mean WER z", "outcome_score", "instrument",
               "Descriptive association (not causal).")
        fm.run("07b_acoustic_vs_outcome", fm.scatter_pair, desc, sp_col or "missing", "outcome_score",
               "Speech ratio vs end-of-year outcome", "student mean speech-ratio z", "outcome_score", "instrument",
               "Descriptive association (not causal).")
        fm.run("08_residual_correlations", fm.residual_correlations, conf)
        fm.run("09_task_conditioned_distributions", fm.task_conditioned, S["text_cons"], clipf, "w_er")
        fm.run("10_correct_response_comparisons", fm.correct_response, S["text_cons"], S["acou"], clipf, y,
               ["w_er", "c_er", "TEMPORAL::speech_ratio", "TEMPORAL::onset_latency_s"])
        vd = st.load_table("diagnostics", "disagreement_variance_decomposition.csv")
        fm.run("11_asr_disagreement", fm.disagreement, S["dis"], clipf, vd)
        fm.run("12_domain_confounding", fm.domain_confounding, conf)
        if abl is not None and not abl.empty:
            fm.run("13_model_ablations", fm.forest_rows, abl[abl["group"].isin(["family", "leave_one_out", "task", "asr_model",
                                                                               "asr_model_loo", "control", "missing_modality"])],
                   "Ablations and negative controls: ΔR² vs reference baseline", "Paired-bootstrap ΔR² of every ablation.")
        if main is not None and not main.empty:
            fm.run("14_incremental_r2", fm.forest_rows, main[main["group"] == "multimodal"],
                   "Incremental R² of the multimodal ladder M1-M8",
                   "ΔR² vs the reference baseline on identical folds; fallback rows are labelled BASELINE_FALLBACK.")
        if oof is not None and not oof.empty:
            names = list(dict.fromkeys([n for n in main["name"]] if main is not None else []))[:12]
            fm.run("15_fold_stability", fm.fold_stability, stability_table(oof, names))
            fm.run("16_predicted_vs_actual", fm.predicted_vs_actual, oof, champ.get("name", ""),
                   champ.get("primary_scheme", "repeated_kfold"))
        g = S.get("imp_grouped") if S.get("imp_grouped") is not None else st.load_table("diagnostics", "feature_importance_grouped.csv")
        fm.run("17_grouped_importance", fm.importance, g, S.get("imp_model"))
        tab = S.get("phenotype_table") if S.get("phenotype_table") is not None else st.load_table("diagnostics", "failure_phenotypes.csv")
        fm.run("18_error_archetypes", fm.error_archetypes, tab)
        fm.run("19_multimodal_failure_recovery", fm.failure_recovery, S.get("phenotype_profiles"))
        fm.run("20_robustness_transfer", fm.robustness, rob)
        fm.write_index()
        return {"n_figures": sum(1 for r in fm.index if r.get("status") == "ok"),
                "n_failed": sum(1 for r in fm.index if r.get("status") == "failed")}

    # ================================================================== SUBMISSION
    def submission(self) -> dict:
        from .submission import make_submission, spec_from_dict

        champ = self.store.load_json("model_registry", "champion_spec.json", {}) or {}
        if not champ:
            raise RuntimeError("model_registry/champion_spec.json missing: run EXPERIMENTS first")
        ev = self.evaluator()
        if ev.reg.data_hash() != champ.get("fold_registry_hash"):
            self.log.warning("fold registry hash differs from the one the champion was selected on")
        spec = spec_from_dict(champ["spec"])
        missing = [p for p in spec.providers if p not in ev.providers]
        if missing:
            raise RuntimeError(f"champion providers {missing} cannot be rebuilt from the stored clip tables")
        cur = {p: ev.providers[p].fingerprint() for p in spec.providers}
        if cur != champ.get("provider_fingerprints", cur):
            self.log.warning("provider fingerprints changed since champion selection: %s", cur)
        return make_submission(self.ctx, ev, spec, self.store, self.c4, champ)

    # ================================================================== REPORT
    def report(self) -> str:
        from .report import write

        self.store.save_table(pd.DataFrame(self.steps), "reports", "run_steps.csv")
        self.store.save_table(self.store.inventory(), "reports", "artifact_inventory.csv")
        return write(self.store, self.c4, self.run_cfg)

    # ================================================================== RUN
    def plan(self, mode: str | None = None) -> list[str]:
        """Ordered steps for a mode (missing upstream artefacts are prepended; report always runs last)."""
        mode = str(mode or self.run_cfg.get("run_mode", "FULL")).upper()
        plan = {"PREPARE": ["prepare"], "FEATURES": ["features"], "EXPERIMENTS": ["experiments", "figures"],
                "SUBMISSION": ["submission"], "FULL": ["prepare", "features", "experiments", "figures", "submission"],
                "SMOKE": ["prepare", "features", "experiments", "figures"]}[mode]
        if not self.run_cfg.get("train_models", True):
            plan = [p for p in plan if p not in ("experiments", "figures", "submission")]
        if not self.run_cfg.get("generate_submission", True) or is_smoke(self.run_cfg):
            plan = [p for p in plan if p != "submission"]
        prepared = self.store.path("manifests", "clip_master.parquet").exists()
        featured = self.store.path("clip_features", "acoustic_columns.json").exists()
        if mode in ("FEATURES", "EXPERIMENTS") and not featured and "features" not in plan:
            plan = ["features"] + plan
        if mode in ("FEATURES", "EXPERIMENTS") and not prepared and "prepare" not in plan:
            plan = ["prepare"] + plan
        return plan

    def run(self, mode: str | None = None) -> dict:
        mode = str(mode or self.run_cfg.get("run_mode", "FULL")).upper()
        out = {"mode": mode}
        for step in self.plan(mode):
            out[step] = self._step(step, getattr(self, step))
        out["report"] = self._step("report", self.report)
        return out
