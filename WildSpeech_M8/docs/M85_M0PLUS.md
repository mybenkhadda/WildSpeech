# WildSpeech M8.5 — M0+ Strong Screening Baseline

**Phase 1 implementation · target 3–16 December 2026**  
**Official-data status:** Source and synthetic tests only until you execute the notebook on your private Drive.

[Open the M8.5 Colab notebook](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M85_M0PLUS_MASTER.ipynb)

## Task

The official competition predicts end-of-year literacy from beginning-of-year screening and speech. M8.5 asks whether a compact, fold-safe **screening-only** model improves R² before introducing audio. It inherits the exact M8.4 reference results (mean 50-fold R² 0.193227; mean ten-repeat pooled OOF R² 0.199113), original 50 student folds, 50 school-disjoint folds, and frozen fold SHA-256:

    6ee0f59c8cf5ea3d8f20f5fd500353419587660e00949ac6f1e6a16e092c9ce1

The reference OOF predictions are read and hash-checked; they are never refitted, overwritten or copied into public GitHub. The same source CSVs, canonical student order, feature hash and immutable fold hash must match Phase 0 before a challenger is fitted.

The executed Phase 0 indicates KTEA-3 has substantially worse generalization across held-out schools (R² around 0.116) than WJ IV (around 0.184). Deletion theta is absent in roughly 39% of train students. These are **descriptive hypotheses**, not causal explanations.

## Package modules

| File | Responsibility |
|---|---|
| contracts.py | Data contract, NFC text normalization, feature blacklist and hashes |
| s2_items.py | Train-only smoothed item difficulty; student-level leave-one-out item statistics |
| s3_splines.py | Theta splines with training-fold-only quantile knots |
| s4_instrument.py | Instrument × theta and task completion |
| s5_task_pattern.py | Three-task participation mask, unscored and theta disagreement |
| features.py | Raw student feature fit/transform and singleton test |
| models.py | Regularized Ridge, fixed CatBoost seed ensemble and convex blend |
| nested_cv.py | Inner-fold re-fitting of S2–S5 and train-only tuning |
| diagnostics.py | School/instrument, task-availability and item-schema research atlas |
| experiments.py | E00–E09 preregistration, original-fold evaluation, ledger and promotion |
| submission.py | Raw single-student inference and deterministic export |
| reporting.py | Figures, scorecard, privacy-safe analysis, decision report |

Item difficulty uses the training-only smoothed success probability

    (correct_prompt + kappa * task_mean_correct) / (n_scored_prompt + kappa)

with fixed kappa=10. For each training student, exclude that student's **all scored attempts** from both item and task statistics. Held-out or test students only see fitted training statistics. S2 accepts unknown prompts via task-average fallback and has no learned test distribution. No school ID, end-of-year score, score_status or filename becomes a predictive covariate.

**Chronology features are intentionally disabled:** opaque filenames and CSV order do not establish item presentation order.

## Preregistered E00–E09 experiments

| Arm | Predictors | Model | Promotion eligible |
|---|---|---|---|
| E00 | Exact original 14-feature screening | Frozen reference RidgeCV | Reference only |
| E01 | S0 + nonlinear theta basis S3 | 3-inner-fold selected Ridge | Yes |
| E02 | S0 | Fixed shallow multi-seed CatBoost | Yes |
| E03 | S0 | Inner-OOF convex Ridge/CatBoost blend | Yes |
| E04 | S0 + S2 + S3 + S4 + S5 | Selected Ridge | Yes, primary bundle |
| E05 | E04 minus S2 | Selected Ridge | Diagnostic |
| E06 | E04 minus S3 | Selected Ridge | Diagnostic |
| E07 | E04 minus S4 | Selected Ridge | Diagnostic |
| E08 | E04 minus S5 | Selected Ridge | Diagnostic |
| E09 | E04 without duration/sample-rate reference fields | Selected Ridge | Nuisance audit |

Model priority E04 → E03 → E02 → E01 is declared before fitting. E05–E09 are explanatory ablations; they cannot be promoted by post-hoc outer-fold selection. The main bundle E04 is tested before interpreting the individual ablations.

Ridge alpha is selected in **three inner student CV folds where the item and spline transformations are refitted** each time, not from globally assembled features. E03 blend weights come from inner-only OOF predictions. Five-seed CatBoost FULL uses fixed seeds 42–46 and 600 iterations, depth 2, learning rate 0.03, L2=20 and RMSE loss. SCREEN uses one seed/250 trees and cannot promote any model.

## How to run

Run the [Colab notebook](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M85_M0PLUS_MASTER.ipynb) using a mounted Drive and existing Phase 0 output. Default SCREEN evaluates first repeat of both schemes and generates descriptive figures. Edit MODE to FULL to run all 100 frozen folds, all E01–E09 comparisons and decision gates.

Local commands from repository root:

    python -m pip install -e './WildSpeech_M8[test]'
    python -m wildspeech_m85.experiments --drive-root /content/drive/MyDrive --mode SCREEN
    python -m wildspeech_m85.experiments --drive-root /content/drive/MyDrive --mode FULL
    python -m pytest -q WildSpeech_M8/tests/test_m85_*.py

## Mandatory promotion gate

A candidate must satisfy **all** conditions using the frozen 50 student fold comparisons:

1. mean paired fold ΔR² at least +0.010;
2. Nadeau–Bengio corrected t at least 2;
3. wins at least 60% of folds;
4. mean pooled school-disjoint ΔR² at least −0.005;
5. both instrument ΔR² at least −0.010.

Primary performance is the ten-repeat mean of pooled student OOF R², which must not be confused with the reference-comparable mean fold R². Corrected t is approximate and cannot account for arbitrary experiment selection.

Only completed FULL E00–E09 and a valid PROMOTE gate allow new candidate submission export. That export re-fits only on **train students**, checks raw student metadata + clips → features → model predictions independently for every test student and in shuffled order, validates official submission row order, and writes a SHA-256. No leaderboard submission is made by the code. If no candidate passes, the original reference remains M0+.

## Private outputs

Located under MyDrive/WildSpeech_M85/m0plus:

- manifest.json; ledger/planned_comparisons.json; ledger/events.jsonl; ledger/experiments.parquet
- diagnostics/item_contract.json; private school/instrument and missingness tables; figures/01–05.png
- runs/<run_id>/folds/*.parquet; private OOF; fold and pooled metrics; JSON per-fold model settings
- tables/paired_model_results.csv; tables/paired_comparison_audit.csv
- reports/M85_M0PLUS_DECISION.md; M85_SCIENTIFIC_ATLAS.md; M85_RESULTS.json
- submission/m0plus_submission.csv; .sha256; m0plus_final.joblib **only when PROMOTED**

No student rows, school identifiers, outcomes, speech/audio, hidden-test predictions or raw private models are committed to GitHub.

## Interpretation and limits

The school/instrument errors may reflect assessment composition and administration differences; they do not prove KTEA-3 causes worse transfer. Task missingness may relate to stopping rules and is partly already in the reference via clip counts and imputer indicators. These must show *incremental* value on frozen paired folds. This milestone does **not** measure genuine child acoustics. That belongs to the later frozen speech-extraction and acoustic/phonetic experiment phases.

## Status

- Source, Colab, screening feature blocks, nested models and synthetic tests implemented.
- The official Drive SCREEN, FULL scores, promotion decision and optional public submission remain to be executed.
