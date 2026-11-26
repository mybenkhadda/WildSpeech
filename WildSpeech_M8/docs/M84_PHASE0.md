# WildSpeech M8.4 — Phase 0: Reproducible Evaluation Harness

**Implementation target:** 26 November–2 December 2026.  
**Actual official-data status:** NOT YET EXECUTED BY THIS COMMIT.  
**Source:** https://blog.drivendata.org/blog/literacy-speech-modeling-reference  
**Colab:** [WILDSPEECH_M84_PHASE0_MASTER.ipynb](../notebooks/WILDSPEECH_M84_PHASE0_MASTER.ipynb)

## What this package does

M8.4 is a new, separately namespaced Python package under src/wildspeech_m84. Its Phase 0 scope is reproducible evaluation, *not new speech model development*:

- Strictly validates the six official CSVs, student/label/clip identity joins and submission schema.
- Uses train_student_metadata.csv original row order (the reference's order); records whether train_labels.csv order matches.
- Builds the **exact 14 reference features** (3 task thetas, 3 pct_correct, 3 n_clips, 3 mean_duration, is_wj4, is_11khz).
- Uses SimpleImputer(strategy="median", add_indicator=True), StandardScaler, RidgeCV(alphas=np.logspace(-2,5,50)).
- Freezes 50 student folds with RepeatedKFold(5,10,random_state=42), and 50 seed-varied, balanced held-out-school folds.
- Creates a hash for canonical student order, group assignments, and fold positions. Fails rather than overwrite a changed hash.
- Checkpoints each of the 100 reference folds to private Drive, supports safe resume and validates complete OOF coverage.
- Computes mean-of-50-fold R² for reference parity and 10-repeat mean pooled OOF R² as the primary metric, with instrument and school robustness.
- Preregisters all model comparisons before fitting and uses an append-only event ledger.
- Fits the reference on **training rows only**; verifies single-test-student and batch/permutation prediction equality; emits a deterministic CSV only if parity passes.
- Generates M84_BASELINE_PARITY.md and a machine-readable M84_RESULTS.json under the private Drive output path.

The pipeline performs **no GPU inference**, no raw-audio read, no competition submission, no test pseudo-labeling, and no test-sample-fitted model or normalizer.

## Colab

1. Open [the GitHub Colab notebook](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M84_PHASE0_MASTER.ipynb).
2. Execute Drive mount and GitHub clone/install. Dataset expected under MyDrive/WildSpeech (auto discovery also supported).
3. Keep RUN_LEGACY_RESEARCH_ONLY=False for strict, compliant reference parity.
4. Run all cells. Reference OOF, folds, ledger, submission checksum and report are saved to MyDrive/WildSpeech_M84/phase0.
5. Upload the generated reference_submission.csv yourself if approved. **No online submission or public leaderboard result is claimed by this code.**
6. If reference parity fails, consult the generated report, verify original metadata order, item_score averaging, sample-rate flags, library versions and RidgeCV internals before attempting new models.

For Windows/local Jupyter set WILDSPEECH_DRIVE_ROOT to your mounted Google Drive folder. The standalone CLI is:

    python -m pip install -e "./WildSpeech_M8[test]"
    python -m wildspeech_m84.runner --drive-root "/content/drive/MyDrive"
    python -m pytest -q WildSpeech_M8/tests/test_m84_*.py

### Output contract (all private)

    MyDrive/WildSpeech_M84/phase0/
      manifests/data_manifest.json
      manifests/environment.json
      manifests/fold_manifest.json
      folds/folds.parquet
      folds/fold_assignment.sha256
      features/registry.csv
      ledger/planned_comparisons.json
      ledger/events.jsonl
      ledger/experiments.parquet
      runs/<run_id>/folds/*.parquet
      runs/<run_id>/oof_predictions_PRIVATE.parquet
      runs/<run_id>/fold_metrics.parquet
      runs/<run_id>/pooled_metrics.parquet
      runs/<run_id>/instrument_metrics.parquet
      runs/<run_id>/run_manifest.json
      parity/M84_BASELINE_PARITY.md
      parity/M84_RESULTS.json
      submissions/reference_submission.csv   (only after parity PASS)
      submissions/reference_submission.sha256
      submissions/reference_ridge14.joblib
      figures/*.png                            (notebook visualization cells)

Do not commit any raw child speech, original subject IDs, school identifiers, training outcomes, fitted private model, or OOF predictions.

## Parity and promotion rules

- **Reference parity:** mean of 50 student-fold R² in [0.188, 0.198], the published reference's R²=0.193 ±0.005 tolerance. Published fold-to-fold SD≈0.043 is NOT a confidence interval or acceptance tolerance.
- **Primary measure:** mean of 10 per-repeat pooled OOF R² scores. This differs mathematically from mean fold R².
- **For challengers:** paired mean Δ fold R²≥0.010, Nadeau–Bengio corrected t≥2, win fraction≥0.60, pooled school ΔR²≥−0.005, no instrument ΔR² below −0.010. Positive but subthreshold -> HOLD, nonpositive or hard robustness veto -> DROP.
- Multiple fold comparisons are correlated. The corrected t is approximate and does not fix experiment-selection bias. Preregistration and future untouched evaluation are still needed.
- School held-out folds are robustness, not necessarily the official public/private test split; avoid mean-fold R² across small schools as school gate.

## Historical parity, scientific attribution and limitations

The **original M7.3 global model source is not present in the current main repository**. It is explicitly NOT_REPRODUCED, not approximated by another algorithm.

The M8.3 code is present, but its feature-family naming can classify correctness counts as audio, and its legacy feature-construction stages may condition feature transformations on combined train/test values. Therefore the optional --legacy-research-only analysis is **not promotion-eligible**. Restoring exact M7.3 source and independently auditing/repairing M8.3 train-only feature formation is required before declaring historical parity or attributing its gain to child speech.

M8.4 reference features explicitly carry source and modeling role separately; is_11khz is a reference predictor *and* a nuisance proxy, while m8__audio__n_correct_items is correctness-derived despite its name. True child-only speech signal has **not** been established by this milestone.

## Tests

The GitHub Actions workflow runs tests covering:

- reference feature schema, train metadata ordering and official ID validation;
- deterministic folds (100 partitions), school-disjointness, repeat coverage, hash freeze and corruption rejection;
- equality of pooled R² under shuffled OOF storage rows, correct fold R², paired delta tests, bad hash rejection;
- provenance and exact duplicated-feature discovery;
- test singleton prediction == full test batch/permutation;
- ledger preregistration and append-only transitions;
- the actual .ipynb Python syntax and Drive-first execution order;
- synthetic full Phase 0 execution and rerun using frozen checkpoint artifacts.

All scores from these tests are synthetic and do not establish competition performance.

## Next step (3–16 December)

After parity PASS, preregister M0+ model arms:

- P1-A reference feature set plus regularized model stack;
- P1-B S2–S5 screening bundle (within-fold item difficulty, theta splines, instrument interactions, missing-task patterns);
- P1-C predeclared leave-one-block-out ablations.

Do not run GPU extraction, item normalization, target-guided feature selection or any test-derived fitted transformation inside Phase 0.
