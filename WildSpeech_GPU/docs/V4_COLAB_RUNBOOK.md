# WildSpeech V4 — Colab runbook

V4 integrates the completed external ASR export with the existing WildSpeech feature, validation and modelling
code. It is driven by `notebooks/WILDSPEECH_V4_MULTIMODAL_MASTER.ipynb` and implemented in
`src/wildspeech/v4/`. Nothing has been executed yet: every table, figure and score appears only after you run the
notebook.

## 1. Inputs on Google Drive

| item | default location |
|---|---|
| competition CSVs + `train_audio.tar` / `test_audio.tar` | `/content/drive/MyDrive/WildSpeech` (searched up to depth 3) |
| combined ASR export | `/content/drive/MyDrive/WildSpeech/WildSpeech_ASR_Exports/all_models_predictions.parquet` |
| fallback (only if the mounted file is missing) | Drive file id `12SzHSmkNbeoS_mO1L80uEd6A2k4jH-8V` downloaded with `gdown` into `local_work_root` |
| optional external audio manifest | `*manifest*.parquet` / `*manifest*.csv` in the ASR export folder (`clip_id, source, member, source_type, audio_name, basename, split_hint`) |
| optional compatible acoustic cache | any folder with `<family>/<split>/part-*.parquet` + `_fingerprint.json`, listed in `V4_OVERRIDES['acoustic']['search_dirs']` |

## 2. Steps

1. Open the notebook in Colab (GPU runtime recommended for acoustic extraction; modelling runs on CPU).
2. Cell 1: set `CONFIG` and fill `V4_OVERRIDES['asr_import']['permission_review']`
   (`reviewed: True`, reviewer, date, statement). Without it the ASR families are `NOT_EXECUTED`.
3. First run `run_mode = 'PREPARE'`, then run cells 2–6 and 11. Read:
   * `asr_import/asr_schema_report.json`: detected layout (long / wide), roles, warnings.
     If a role is missing or wrong, set `V4_OVERRIDES['asr_import']['column_map']`
     (or `wide_model_columns`) and re-run PREPARE.
   * `manifests/join_audit.csv`, `manifests/data_integrity.json`, `asr_import/asr_coverage.csv`, `asr_import/import_gate.csv`.
4. `run_mode = 'FEATURES'`: acoustic cache reuse and extraction of the missing clips (TARs are staged to local
   disk first), then the recognized-text, disagreement, reliability and (if phone strings exist) measured-phonetic
   clip tables, the providers and the feature catalog.
5. `run_mode = 'EXPERIMENTS'`: M0–M8, ablations, negative controls, robustness schemes, fallback audit, H1–H10,
   importance, champion selection and figures.
6. `run_mode = 'SUBMISSION'`: champion refit on all labelled students, model reloaded from disk, validated
   `submission/submission.csv`.
7. `run_mode = 'FULL'` runs 3–6 in one go; `SMOKE` checks the whole chain on a 120-student subsample (1 repeat, no
   submission).
8. For a reproducible run, set `CONFIG['pinned_commit']` to the commit printed by cell 3.

Every step resumes from what already exists under `output_root`: fingerprinted clip tables are reused, CV results
are cached by content key (`cache/v4/eval`), and provider tables by training-id set (`cache/v4/fold_providers`).

## 3. Execution order and dependencies

```
prepare     DataBundle (data.schema) -> metadata index -> TAR manifests (data.tar_manifest) + external manifest
            -> ASR export: inspect_asr_schema -> to_long -> canonicalize -> join_asr (split + normalized filename)
            -> resolve_duplicates -> coverage_table -> validate_import (independent gate) -> write_external_store
            -> clip_master / student_master / join_audit / data_integrity
features    acoustic: import_cached_families -> inventory -> AcousticExtractionRunner.run (missing clips only)
            text: build_clip_text_table (phonetic.error_features) -> consensus / disagreement / reliability
            phonetics: availability -> build_clip_phone_table (measured phones only)
            providers: V4ClipAggregateProvider (ItemConditionedResidualizer + ItemReference, fold-fitted)
experiments V4Evaluator (V3Evaluator + V4FoldModel) on the frozen V2 validation registry
            -> SpecFactory (M0-M8, ablations, controls) -> ExperimentRunner -> classify_results -> fallback_report
            -> robustness schemes -> hypotheses -> grouped importance -> select_champion
figures     FigureMaker (v3.viz style) -> PNG + SVG + data
submission  fit_full_v3 -> joblib dump -> reload -> predict -> validate_submission + extra checks
report      reports/WILDSPEECH_V4_RESEARCH_REPORT.md from artefacts only
```

## 4. Output tree (`output_root`)

```
manifests/        audio_manifest.parquet, clip_master.parquet, student_master.parquet, asr_model_coverage.csv,
                  join_audit.csv, data_integrity.json
asr_import/       asr_schema_report.json, asr_import_report.json, canonical_predictions.parquet, asr_coverage.csv,
                  import_gate.csv, unmatched_predictions.parquet, duplicates.parquet, failed_inference.parquet
clip_features/    acoustic_inventory.csv, acoustic_cache_reuse.csv, clip_text_long.parquet, clip_text_consensus.parquet,
                  clip_disagreement.parquet, clip_reliability.parquet, phonetic_availability.csv, clip_phone_*.parquet
student_features/ feature_catalog.parquet, student_features_descriptive_train.parquet
cv/               cv_results.parquet, oof_predictions.parquet, fold_traces.parquet, fold_stability.parquet,
                  paired_fold_deltas.parquet, fold_registry_summary.csv
experiments/      experiment_results.csv, hypothesis_results.csv
ablations/        ablation_results.csv
robustness/       robustness_results.csv, transfer_table.csv, family_generalization.csv
diagnostics/      fallback_report.csv, confounding.csv, disagreement_variance_decomposition.csv,
                  failure_phenotypes.csv, feature_importance.parquet, feature_importance_grouped.csv
model_registry/   champion_spec.json, experiment_specs.csv, run logs
models/           champion_fold_model.joblib, feature_importance.parquet
figures/          *.png, *.svg, data/*.csv|parquet, figure_index.csv
submission/       submission.csv, test_predictions.parquet, submission_validation.csv, submission_meta.json
reports/          WILDSPEECH_V4_RESEARCH_REPORT.md, hypotheses.csv, run_config.json, run_steps.csv, artifact_inventory.csv
cache/            V4 caches (acoustic features, external ASR store, fold providers, CV results)
```

## 5. Safeguards

* Student-disjoint folds shared by every comparison (frozen registry); nested inner CV for every reducer, shrinkage,
  fusion weight and gate; no feature selection on validation labels; no fusion weight tuned on evaluation folds.
* Fallbacks (`alpha = 0`, zero fusion / gate weight, no effective features, identical predictions) are traced per
  fold and labelled `BASELINE_FALLBACK`; they are never reported as improvements and never become the champion.
* `school_id` / `score_status` are train-only and never features; the submission checks the design for them.
* External ASR predictions keep provenance `external_colab_asr` + source SHA-256 and are never relabelled as P24.

## 6. Packaging incident and guard (src/wildspeech/models)

Commit `17222a0` lacked the whole `src/wildspeech/models/` package: the unanchored `models/` rule in
`WildSpeech_GPU/.gitignore` (meant for the model-weights folder) also matched the source package, so Colab failed with
`ModuleNotFoundError: No module named 'wildspeech.models'`. Repairs:

* artefact patterns are anchored to the project root (`/models/`, `/cache/`, `/outputs/`, ...) and `src/`, `tests/`,
  `configs/`, `scripts/`, `notebooks/`, `docs/` are explicitly re-included (bytecode excluded);
* the package is restored unchanged from the local implementation;
* `wildspeech.v4.source_check` (stdlib only) parses every module and verifies that every imported `wildspeech`
  module exists; notebook cell 3 runs it right after the clone, cell 4 imports the training modules before any step;
* the notebook follows `main` (no pin to the broken commit); `tests/test_v4_source_completeness.py` guards the
  ignore rules and the checkout.
