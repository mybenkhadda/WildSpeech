# WildSpeech — local GPU-accelerated literacy research pipeline

Discovers which acoustic, phonetic, linguistic and behavioural characteristics of children's assessment speech
predict end-of-year reading outcomes **beyond the screening thetas**, then turns the stable findings into
validated predictive models. Everything runs locally on Windows + Conda + RTX 3070 (8 GB). No Colab/Drive.

> P00–P15 were authored as code and then executed by you (`notebooks/WILDSPEECH_FULL_EXPERIMENT.ipynb`, FULL
> profile). The **V2 research line P16–P22** (section 9 below) is authored as code only — none of its stages,
> tests or notebook cells has been executed; its tables and figures appear only after you run it.

## 1. Data facts the design relies on (read from the CSV headers)
| file | columns | note |
|---|---|---|
| train/test_clip_metadata | filename, subject_id, task, expected_text, item_score, duration_sec (+ score_status **train only**) | 3 tasks: blending, deletion, sentence_repetition |
| train/test_student_metadata | subject_id, instrument (wj4/ktea3), original_sample_rate (16000/11025), 3 × `<task>_theta` (+ school_id **train only**) | deletion_theta can be missing |
| train_labels / submission_format | subject_id, outcome_score | official metric: R² |

`school_id` and `score_status` are absent from test, so they are used only for grouping/robustness/description,
never as features. Unscored items are detected from a blank `item_score`, which exists in both splits.

## 2. Install into the existing `MITSUI` environment
```powershell
conda activate MITSUI
# PyTorch with CUDA 12.4 kernels (driver 581.80 / CUDA 13.0 runs any CUDA 12.x build)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
cd C:\Users\myben\Downloads\WildSpeech_P01\WildSpeech_GPU
pip install -r requirements.txt
pip install -e .                     # optional: makes `python -m wildspeech` work from any directory
# optional extras: umap-learn pygam transformers g2p_en markdown
python -m ipykernel install --user --name MITSUI --display-name "Python (MITSUI)"
```
If `ipykernel` installation fails on Python 3.11, pin `conda install "jupyter_client<8.10"` first.
A full reference spec is in `environment.yml`.

Check the environment and data discovery (prints only, runs no stage):
```powershell
python scripts\check_environment.py
```

## 3. Run
**CLI** (from the repository root; `$env:PYTHONPATH="src"` if you skipped `pip install -e .`):
```powershell
python -m wildspeech plan   --profile FAST                 # what would run (dry run)
python -m wildspeech run    --profile SMOKE                # ~minutes, technical verification
python -m wildspeech run    --profile FAST                 # tabular + GPU acoustic + baselines + champion + report
python -m wildspeech run    --profile FULL                 # + pitch, wavelets, robustness/LOSO, fusion
python -m wildspeech run    --profile RESEARCH             # + local ASR/alignment, embeddings, neural MIL
python -m wildspeech run    --profile RESUME               # continue the last run
python -m wildspeech run    --profile SUBMISSION           # P14 only, with the champion's frozen config
python -m wildspeech run    --profile FAST --stages P07 --force P07
python -m wildspeech run    --profile FAST --set gpu.batch_size=32 --set audio.enable_pitch=true
python -m wildspeech status | cache | clear-cache --family pitch | invalidate P07
.\scripts\run_pipeline.ps1 -RunProfile FAST                 # same via conda run -n MITSUI
```
**Notebooks:** `notebooks/WILDSPEECH_MASTER_PIPELINE.ipynb` (choose `PROFILE`, plan, run, inspect), or the per-stage
notebooks `P00_…` to `P15_…`. All are thin layers over the package; outputs are empty until you run them.
Stick to one profile per research line: switching profiles changes stage fingerprints and recomputes the dependants.

**Tests** (synthetic data, CPU; GPU-only checks auto-skip): `.\scripts\run_tests.ps1` or `python -m pytest tests`.

## 4. Stages
| id | stage | key outputs (`outputs/<id>_<name>/tables`, `figures`) |
|---|---|---|
| P00 | infrastructure | hardware_report.json, gpu_benchmark, numerical_parity, integrity_checks, batch_recommendation |
| P01 | tabular EDA | target/screening/item/IRT/adaptive tables, tabular_baselines |
| P02 | acoustic extraction + EDA | per-family caches, correctness_contrasts_item_centred, group effects, bandlimit vs rate |
| P03 | phonetic (opt-in) | ASR/ALIGN families, deviation & GOP contrasts |
| P04 | embeddings (opt-in) | EMB_<model> families |
| P05 | features | student_features_crossfit.parquet, feature_catalog, feature registry |
| P06 | phenotypes | PCA/UMAP, k selection, bootstrap ARI, profiles |
| P07 | residual discovery | residual_associations (FDR, CI, permutation, replication), matched comparisons, HC3 |
| P08 | robustness | school effects, artifact-adjusted associations, perturbation stability, domain generalisation |
| P09 | literacy indices | incremental R² CIs, α, split-half reliability (experimental constructs only) |
| P10 | benchmark | model × family grid, budgeted ablations with paired ΔR² CIs |
| P11 | multimodal | late fusion, task ensembles, residual stacking, MoE, attention-MIL, transformer-MIL |
| P12 | failure analysis | failure clusters, recovery per model, failure registry |
| P13 | champion | robust ranking, CV-stacking, conformal coverage, models/champion_spec.json |
| P14 | submission | independent re-load inference, validated CSV in submissions/ |
| P15 | report | reports/WILDSPEECH_RESEARCH_REPORT.md/.html + registry CSVs |

Dependencies: P01→P00; P02/P03/P04→P00; P05→P01 (+P02–P04 if present); P06–P10→P05; P11–P13→P10; P14→P13.
Fingerprints = hash(config sections a stage declares + upstream fingerprints); unchanged stages are skipped.

## 5. GPU/IO architecture (replaces the Joblib CPU pipeline)
* **TAR offsets:** one header pass builds `cache/manifests/*_manifest.parquet` (member → byte offset); audio is read by
  `seek+read` with one file handle per thread. Nothing is extracted to disk; the archives are opened read-only.
* **Threaded decoding:** `soundfile` (libsndfile releases the GIL) on a small thread pool; at most
  `prefetch_batches + 1` batches live in RAM, with a soft RSS limit. **No multiprocessing** anywhere.
* **Duration-aware GPU batches:** clips sorted by duration and packed under a padded-seconds budget; valid-length
  frame masks make every statistic independent of padding (verified by `test_batch_invariance_masking`).
* **One STFT per batch** → power, mel, MFCC(+Δ), spectral shape, VAD (adaptive threshold + morphological cleanup),
  quality (SNR, bandwidth, clipping), YIN pitch on GPU; run-length/peak bookkeeping on CPU numpy; wavelets on CPU.
* **Adaptive batch size** starts at 24 (config `gpu.batch_size`, or `auto` = P00 recommendation), grows when peak VRAM
  is low, halves on CUDA OOM; a batch that OOMs is split recursively; a single clip falls back to CPU.
* **Checkpoints:** every `audio.checkpoint_every` clips → atomic Parquet part per family; re-running resumes.
  Each family has its own fingerprint, so changing pitch settings recomputes only `pitch`.
* **Progress:** tqdm with %, ETA, clips/s, process RAM, VRAM reserved/peak, failures, last checkpoint, batch size.

## 6. Validation & leakage safeguards
* Student-disjoint repeated stratified K-fold (primary), school-grouped K-fold, leave-one-school-out, instrument and
  original-sample-rate transfer; R², RMSE, MAE, incremental R² with paired bootstrap CIs, subgroup diagnostics.
* Inside each training fold: item p-correct/difficulty (leave-one-student-out for training rows), item reference
  statistics for normalisation, Rasch item parameters, embedding PCA, imputation, scaling, supervised selection,
  index signs, fusion/stacking weights (inner CV), residual-learning shrinkage, neural early stopping.
* Discovery stages use cross-fitted student features (P05); predictive stages rebuild features per fold.
* Test data: never used to fit anything; test features are per-recording transforms plus train-fitted statistics;
  no pseudo-labelling; submissions refused for subsampled runs; bundle is re-loaded from disk before inference.
* Foundation models: `configs/foundation_models.yaml` gating (`license_verified`, `overlap_checked`), `local_files_only`,
  no downloads unless `runtime.allow_model_download: true`. No external APIs, no audio uploads; ASR text stays local.
* Indices are labelled experimental measurement constructs, not diagnoses.

## 7. Foundation models (RESEARCH)
Download the snapshots yourself, outside the pipeline, e.g.
`huggingface-cli download facebook/wav2vec2-base-960h`, then review each model card and set `license_verified: true`
and `overlap_checked: true`. For an eligible child-speech checkpoint, fill `child_ssl_custom` (`hf_id` or `local_path`).

## 8. Layout
```
WildSpeech_GPU/
  configs/        default.yaml, profiles.yaml, hypotheses.yaml, foundation_models.yaml
  data/           README only (raw data discovered in place)
  cache/          manifests/, features/<family>/<split>/, fold_features/, runtime/
  notebooks/      WILDSPEECH_MASTER_PIPELINE.ipynb, P00_…P15_…
  src/wildspeech/ data/ acoustic/ phonetic/ features/ validation/ models/ analysis/ stages/ + core modules
  outputs/ models/ experiments/ (registries, logs) submissions/ reports/
  scripts/        run_pipeline.ps1, run_tests.ps1, check_environment.py
  tests/          synthetic-data unit tests
```
The earlier CPU/Colab implementation in `../WildSpeech/` is left untouched.

## 9. V2 — Robust Acoustic & Phonetic Intelligence (P16–P22)
**Question:** can robust, phonetic and expected-text-conditioned speech representations explain future reading
outcomes beyond the *strongest validated tabular baseline* (picked from P10/P01 artefacts, e.g. GAM on
SCREENING+BEHAVIOR+ITEM), not just beyond the screening-only ridge?

### 9.1 Run (Windows, conda env MITSUI, from the repository root)
```powershell
conda activate MITSUI
cd C:\Users\myben\Downloads\WildSpeech_P01\WildSpeech_GPU
$env:PYTHONPATH = "src"                                  # unless `pip install -e .` was done
python -m wildspeech plan --profile V2_CORE              # dry run: P00-P15 show `cached`, V2 stages `would_run`
python -m wildspeech run  --profile V2_SMOKE             # technical check (first repeat, few clips, small grids)
python -m wildspeech run  --profile V2_CORE              # P16 P17 P19 P21 P22
python -m wildspeech run  --profile V2_PHONETIC          # + P18 (eligible local ASR / CTC / phoneme models)
python -m wildspeech run  --profile V2_FOUNDATION        # + P20 (eligible local encoders)
python -m wildspeech run  --profile V2_FULL              # everything
python -m wildspeech run  --profile V2_FULL --stages P21 --force P21
python -m wildspeech run  --profile V2_FULL --set v2.evaluation.bootstrap=500 --set "v2.multimodal.budgets=[5,10,20]"
python -m wildspeech v2-inventory                        # stage statuses / freshness + V1 and V2 caches
python -m wildspeech v2-registry                         # frozen validation registry summary
python -m wildspeech v2-audit                            # first-experiment findings re-derived from artefacts
python -m wildspeech v2-report                           # rebuild outputs/V2/scientific_report.html
python -m wildspeech run  --profile V2_SUBMISSION        # ONLY after reviewing P22: validated test submissions
.\scripts\run_v2.ps1 -RunProfile V2_FULL                 # same through `conda run -n MITSUI`
```
**Notebook:** `notebooks/WILDSPEECH_V2_ROBUST_PHONETIC_EXPERIMENT.ipynb` (kernel *Python (MITSUI)*). `RUN_PIPELINE = False`
by default (display only); set it to `True`, choose `PROFILE` and the per-stage `RUN` switches; `RUN_SUBMISSION`
gates test inference. Sections: configuration & hardware → artefact inventory → previous-findings audit & baseline
verification → data/validation sanity → P16 … P22 → statistical validation → ablations → figure atlas →
submissions → conclusions & recommendations (all generated from the run's tables).

**Tests** (synthetic signals/fixtures, CPU; GPU parity tests skip without CUDA; never executed while authoring):
```powershell
python -m pytest tests -m "not integration"              # unit tests (V1 + V2)
python -m pytest tests -m integration                    # end-to-end V2 on a synthetic project (minutes)
python -m pytest tests -m gpu                            # CPU/GPU numerical parity on the RTX 3070
```

### 9.2 Guarantees
* V2 settings live only under `v2:` (`configs/v2.yaml`); V2 profiles override nothing else, so P00–P15 fingerprints
  are unchanged and their results/caches are reused (never recomputed or overwritten).
* V2 caches: `cache/v2/{features,sequences,embeddings,fold_providers,eval,soft_meta,perturbation,reference_pairs}` —
  separate from `cache/features`, so V1 builders never pick up V2 families.
* Frozen validation registry `outputs/V2/validation_registry.parquet` (+ hash): repeated student-disjoint K-fold
  (identical to V1 folds), school K-fold, LOSO, instrument / sample-rate transfer (both directions) and
  **equal-training-size** transfer controls (same validation students, same n, source vs target domain).
* Nested student-disjoint cross-fitting for every supervised clip-level feature (soft correctness, reference
  templates, item centroids, supervised bottlenecks); residualizers, PCA, selection, calibration, stacking and
  residual shrinkage are fitted inside training folds.
* Stale artefacts: V2 stages re-plan stale V2 soft dependencies and refuse stale upstream artefacts
  (`StaleUpstreamError`); every V2 status records the upstream fingerprints it consumed.
* Deployability checks + quarantine; submissions only for validated candidates, disabled by default.
* P08's perturbation analysis now reuses cached base/perturbed features (`cache/v2/perturbation`, GPU-batched).

### 9.3 Stages
| id | stage | key outputs (`outputs/V2/<id>_<NAME>/tables`, `figures`) |
|---|---|---|
| P16 | harmonization | variant_summary, student_results, distribution_shift, perturbation_stability, feature_robustness_ranking, equal_n_transfer, selection.json |
| P18 | phonetics (opt-in) | model_eligibility, correct_vs_incorrect, within_correct_profile/association, phoneme confusion, error positions, alignments |
| P20 | foundation models (opt-in) | extraction_performance, layer_metrics, layer CKA, compression_comparison, student_results, selection.json |
| P17 | correctness | clip_metrics(+by_group), audio_specific_gain, domain_transfer_auc, student_soft_results, ROC/PR/calibration/confusion |
| P19 | reference | reference_clip_auc, within_correct_association, adjusted_associations, item_reference_atlas, DTW alignments |
| P21 | compact multimodal | baseline_verification, family_value, model_comparison_p21, ablations, budget curve, index_components, specs.json |
| P22 | failure recovery & champion | deployability, task_ensemble_diagnosis/repair, recovery_matrix, candidates, submissions, scientific_report.html |

Execution order P16 → P18 → P20 → P17 → P19 → P21 → P22 (phonetic and embedding caches exist before the stages that
consume them). Central files in `outputs/V2/`: feature_registry.parquet, hypothesis_registry.csv,
experiment_registry.csv, model_comparison.csv, ablation_results.csv, failure_registry.parquet, figure_index.csv,
scientific_report.html. The V1 `embeddings.py` module keeps its name; the V2 foundation-model package is
`wildspeech/foundation/` and the V2 report modules live in `wildspeech/v2/` (`visualization_v2.py`,
`research_report_v2.py`) because `reporting.py` / `embeddings.py` already exist as V1 modules.

### 9.4 Models for P18 / P20
Download snapshots yourself (outside the pipeline), e.g. `huggingface-cli download facebook/wav2vec2-lv-60-espeak-cv-ft`,
read each model card and set `license_verified: true` and `overlap_checked: true` in `configs/foundation_models.yaml`
(`wav2vec2_phoneme_espeak`, `whisper_small_en`, `wav2vec2_base_960h`, `wav2vec2_base`, `hubert_base`, `wavlm_base_plus`,
`whisper_base`, `child_asr_custom`, `child_ssl_custom`). Phoneme metrics need `g2p_en` (or nltk CMUdict); with only
the rule-based proxy they are skipped unless `v2.phonetic.allow_approximate_g2p: true`.

## 10. V3 — Child ASR, Phonetic Intelligence & Reading Prediction (P23–P29)
Setup, model snapshots, eligibility and the exact manual commands: **`docs/V3_WINDOWS_SETUP.md`**.
Notebook: `notebooks/WILDSPEECH_V3_ASR_PHONETIC_INTELLIGENCE_MASTER.ipynb` (`RUN_PIPELINE = False` by default).

| id | stage | key outputs (`outputs/V3/<id>_<NAME>/tables`) |
|---|---|---|
| P23 | fusion audit | v2_result_classification, v2_budget_cap, baseline_reproduction, fusion_audit(_folds), fusion_verdicts, task_ensemble_*, lineage_audit, v3_repairs |
| P24 | local ASR | model_eligibility, inference_runs, throughput, batch_log, asr_outputs_inventory, transcription_examples |
| P25 | phonetic features | feature_tables, correctness_validation_auc, child_vs_general_auc, confusion_long_*, manual_review_sheet |
| P26 | correct-response science | correct_vs_incorrect_profiles, clip_correctness_increment, residual_associations, student_incremental_r2 |
| P27 | ASR disagreement | pairwise_disagreement, consensus, variance_decomposition, student_disagreement_consistency |
| P28 | compact representations | representation_results, exploratory_best_config, fallback_share_by_family |
| P29 | reading ablation | ablation_results (ASR00–ASR12), transfer_robustness, feature_selection_stability, confounding, champion_comparison |

* V3 settings live only under `v3:` (`configs/v3.yaml`, `configs/v3_asr_models.yaml`, `configs/hypotheses_v3.yaml`).
  V3 profiles layer the `V2_FULL` overrides (`base_profile`), so V1/V2 stages stay `cached`; V3 never re-plans them.
* Packages: `asr/`, `phonetics/`, `discovery/`, `representations/`, `evaluation/`, `v3/`; stages `stages/p23_*` … `p29_*`.
* `evaluation.nested_asr_cv.V3Evaluator` traces every fold; a model whose predictions equal the baseline is
  `fallback_baseline` (never reported as ΔR² = 0 "ok"); failures are `failed` with a traceback, never replaced.
* Tests: `python -m pytest tests -k v3 -m "not integration"` and `python -m pytest tests/test_v3_integration.py`.
