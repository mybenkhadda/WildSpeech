# WildSpeech M8.1 — Enhanced Score-Aware Behavioral & Speech Discovery

Research-oriented, fold-safe feature engineering, acoustic feature bank, optional ASR reference diagnostics, supervised residual grouping, nested CV, school-transfer audit, visual atlas, and submission writer.

## Project contents

- `WILDSPEECH_M8_MASTER.ipynb`: Colab workflow, configurations, figures, and analysis.
- `m8_audio.py`: CPU-first tar-streaming extraction, quality metrics, temporal / spectral statistics, task-aware student aggregation; `tqdm` progress.
- `m8_speech.py`: **optional** local ASR-export integration (prompt-conditioned similarity, **not** verified child phonetic correctness).
- `m8_pipeline.py`: feature assembly, transformations, nested outcome-aware clustering, cross-fitted corrections, CV, leave-one-school-out, figures, exports.
- `tests/`: reproducible unit and synthetic integration smoke tests.

## Important practical prerequisite

The most comparable results require an **M7/M7.1 student-profile export** (`student_profiles_all_covariates.csv`), plus the six official WildSpeech CSV files. If absent, the pipeline uses only the available official metadata-derived features **with a warning**. That fallback is **not comparable** with the M7.3 full-feature baseline and is not a replacement for rebuilding the original M7 feature pipeline.

Official layout:

```text
WildSpeech/
  train_student_metadata.csv
  test_student_metadata.csv
  train_clip_metadata.csv
  test_clip_metadata.csv
  train_labels.csv
  submission_format.csv
  train_audio.tar
  test_audio.tar
```

## Running (Colab / local)

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_m8.py
python m8_audio.py --data-root /path/WildSpeech --output-root /path/WildSpeech_M8
# If you have an existing local ASR export with filename, transcript and optionally split:
python m8_speech.py --data-root /path/WildSpeech --asr-path /path/asr.csv --output-root /path/WildSpeech_M8
python m8_pipeline.py --data-root /path/WildSpeech --profile-csv /path/student_profiles_all_covariates.csv --output-root /path/WildSpeech_M8
```

For a **short infrastructure check** (not a valid score):

```bash
python m8_audio.py --data-root /path/WildSpeech --output-root /path/WildSpeech_M8 --limit 50
python m8_pipeline.py --data-root /path/WildSpeech --profile-csv /path/profile.csv --output-root /path/WildSpeech_M8 --quick --skip-loso
```

Clear or replace partial audio caches before full research. The audio module automatically replaces a partial extraction when invoked with `--limit 0`.

## Methodological safeguards

- Outcome labels are held separately; never included as predictive covariates.
- School IDs are used for diagnostics/LOSO only, never model features.
- Fold preprocessing and supervised latent projections are trained only on training students.
- The residual correction is trained against **cross-fitted** baseline residuals, never in-sample fit residuals.
- Candidate selection uses **inner CV**; the `nested_selected` row is the nested selection estimate. Individual candidate rows are exploratory comparisons.
- Cluster outcome differences on descriptive training fits are **not held-out evidence**.
- School-transfer requires fresh training for each held-out school.
- Unsupervised/observed acoustic quality metrics are **not speaker diarization**. Child/adult attribution, validated phonetic alignment, or clinical inference are not claimed.
- The generated submission uses the official `subject_id` order from `submission_format.csv`; always verify the competition's current column schema before submitting.

## Interpretation priority

1. Check `reports/run_summary.json` and `tables/feature_manifest.csv`; ensure the intended profile and modality features actually entered the model.
2. Check `tables/cv_metrics.csv` for `nested_selected` against `global` on **complete repeats**.
3. Check `tables/paired_bootstrap.csv` and school-level MAE/RMSE; small score differences may be noise.
4. Inspect `figures/01..12`, descriptive groups and permutation controls. Interpret descriptive associations as exploratory.
5. If local models do not improve robustly, use the global baseline; keep cluster analysis as research.

## Not included / explicit limits

The core package does **not** automatically download ASR/diarization/foundation-model checkpoints; such models have different licensing, consent, memory and accuracy constraints. It processes available ASR hypotheses if explicitly supplied. No claim of validated child-only segmentation is made. Full raw-audio processing can take time and disk space. The notebook can run on either local or Colab with mounted Google Drive.

---

## M8.1 enhanced experimental suite (new)

New files:
- `EXPERIMENT_MATRIX.md` — hypothesis-to-validation matrix, decision rules and guide to every new figure.
- `m8_research.py` — matched-fold feature family experiments, shuffle controls, double cross-fitted screening-adjusted feature/score associations, BH FDR, cluster stability and confounding, ability-matched pairs, train/test shift, instrument checks, student and school bootstrap, 13-figure scientific atlas, and `RESEARCH_SUMMARY.md`.
- `m8_phonetic.py` — **optional** local pretrained Pasketti-phonetic inference, CTC frame handling, checkpoint reuse, audio-tar streaming, GPU batches and cached phonetic comparisons to dictionary references. This is not validated child speech transcription or child/adult diarization.
- `optional-requirements.txt` — Torch, Transformers and IPA dictionary dependencies for phonetic inference only.

### Recommended full run

```bash
pip install -r requirements.txt
pytest -q tests/
# Full acoustic extraction (skip if a complete bank exists)
python m8_audio.py --data-root /path/WildSpeech --output-root /path/WildSpeech_M8
# Optional, after checking pretrained model weights and license:
pip install -r optional-requirements.txt
python m8_phonetic.py --data-root /path/WildSpeech --output-root /path/WildSpeech_M8 --batch-size 3
# Train and evaluate prediction models; outputs valid submission file in FULL mode
python m8_pipeline.py --data-root /path/WildSpeech --profile-csv /path/student_profiles_all_covariates.csv --output-root /path/WildSpeech_M8
# Independent enhanced experiment atlas (more CV fits, so longer runtime)
python m8_research.py --data-root /path/WildSpeech --profile-csv /path/student_profiles_all_covariates.csv --output-root /path/WildSpeech_M8 --folds 5 --repeats 2 --permutations 3
```

Outputs: `RESEARCH_ATLAS.html`, `RESEARCH_SUMMARY.md`, `tables/ablation_*`,
`tables/conditional_feature_score_associations.csv`, `tables/paired_*_bootstrap_ablation.csv`,
`tables/descriptive_group_stability.csv`, `tables/feature_distribution_shift.csv`,
`tables/ability_matched_pairs.csv`, `figures/E01*` through `E13*` (where supported by input features).

### Scientific boundaries and guardrails

- The **feature ablation model fits each feature set on the same student folds** and reports positive or negative deltas; separate modalities absent in input are not silently filled with invented features.
- **Same-fold shuffle controls** break acoustic/speech student alignment in training and validation separately; they are not a substitute for an external confirmation set.
- **Conditional associations** use cross-fitted screening predictions and feature residualization; adjusted p-values are explorations, not a causal claim or independently validated discovery after inspecting many experiments.
- **School-cluster bootstrap** resamples schools as clusters while preserving all of each sampled school's students; school sizes remain unequal. Student-bootstrap intervals are reported separately.
- **Group stability** is descriptive and should not be interpreted as cross-validated phenotype validity.
- **The current evaluation does not automatically treat the best research ablation as the submission model**; use nested model selection and independent verification.
- The phonetic encoder uses a pretrained model to generate imperfect hypotheses; it does not isolate adult/child voices, and expected-text IPA references can have dictionary/OOV errors.
- Official train/test labels are never mixed or used for pseudo-labeling. Any test features are used only for predictions and unlabeled distribution shift.

### Runtime guidance

Set `MODE='SMOKE'` and `RUN_PHONETIC=False` for environment validation. The full
M8 training pipeline plus M8.1 ablations trains many CV models and may take
significantly longer than feature extraction alone. Cache acoustic and phonetic
features once, and reuse for every CV/model run. Always verify the actual
feature coverage in `feature_manifest.csv` before interpreting missing or
negative acoustic findings.