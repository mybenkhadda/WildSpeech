# WildSpeech M8.6 — M0++ Item-Response Intelligence

**Target milestone:** 27 November–12 December 2026.  
**Evidence status:** Source implementation and synthetic tests; no M8.6 official R² score is claimed until the full Drive run.

[Open M8.6 in Colab](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M86_M0PP_MASTER.ipynb)

## Immutable inputs

- Original six official CSVs from MyDrive/WildSpeech.
- Original M8.4 reference OOF and 5×10 student + 5×10 held-out-school folds from MyDrive/WildSpeech_M84/phase0.
- M8.5 E04 **FULL** completed run, checksum-verified OOF and HOLD_REFERENCE decision from MyDrive/WildSpeech_M85/m0plus.
- Original M8.4 fold SHA-256:
  6ee0f59c8cf5ea3d8f20f5fd500353419587660e00949ac6f1e6a16e092c9ce1.

The runner verifies the exact M8.4 CSV file hashes, reference OOF, M8.5 screening-source hash, E04 ledger status and prediction checksum. It never changes the original baseline or its decision.

## How to execute

Open the linked notebook and run SCREEN first. SCREEN evaluates only the frozen repeat zero and cannot promote any model. Then change MODE to FULL to evaluate all 11 preregistered experiments on the complete frozen 100 folds, with inner-CV transformations and per-fold checkpointing.

Local commands from the WildSpeech repository:

~~~bash
pip install -e './WildSpeech_M8[test]'
python -m wildspeech_m86.experiments --drive-root /content/drive/MyDrive --mode SCREEN
python -m wildspeech_m86.experiments --drive-root /content/drive/MyDrive --mode FULL
pytest -q WildSpeech_M8/tests/test_m86_*.py
~~~

The full experiment is compute intensive (inner folds and cross-fitted Rasch features). Completed per-model folds are reused by run hashes. A change to fitted feature code creates a new run identity, not an overwrite of an earlier successful run.

## Preregistered experiment matrix

| Arm | Predictors | Learner | Hypothesis |
|---|---|---|---|
| B1 | M8.5 full S0+S2+S3+S4+S5 | Inner-OOF Ridge/CatBoost blend | Previous E03 fusion omitted E04 features |
| B2 | E04 minus free theta splines and theta×instrument effects, plus monotone hinges | Constrained additive GAM | Stable monotone nonlinear theta relationships |
| B3 | S0+S2+S3 | Ridge | Reduce redundant interactions/missingness |
| I1 | E04 + task×prompt matrix | Ridge | Observed item identity beyond task averages |
| I2 | E04 + Rasch ability and uncertainty | Ridge | Newly calibrated item difficulty captures latent ability |
| I3 | E04 + orthographic prompt complexity | Ridge | Target item complexity explains response topology |
| I4 | E04 + easy/medium/hard correctness profiles | Ridge | Conditional success adds information |
| I5 | E04 + I1–I4 | Ridge | Complementarity of item representations |
| D2 | E04 | Shared Ridge + penalized instrument deviations | Partial pooling improves instrument transfer |
| D3 | E04 | Shared Ridge + penalized deletion-availability deviations | Deletion protocol heterogeneity |
| T1 | E04 + item×correctness normalized clip durations | Ridge | Clip duration or recording protocol contributes information |

Primary predeclared scientific hypotheses: B1, B2, I5. Other arms are controlled diagnostics.

## Detailed fit contracts

- E04's original raw features and screening estimator source are reused unmodified.
- Item ID = task plus Unicode-normalized expected prompt text; no inferred item chronology.
- I1 chooses prompts using **training-student support only**. Each training row uses a student-leave-one-out item prior and explicit response-observed indicators. Unseen prompts receive zero item-matrix entries.
- I2 uses strongly regularized Rasch 1PL (fixed discrimination). Fit item difficulties only from training students; training representations are cross-fitted by student; new students are scored independently with frozen item parameters.
- I3 uses deterministic text character, word, consonant-cluster and vowel-group proxy counts, **not validated phoneme counts or word frequency**. External lexicons are not silently downloaded or used.
- I4 conditions item correctness on training-only item success probability; training students are excluded from their own prompt/task reference.
- T1 uses log(1+clip duration) and training-only item×score-category moments with student exclusion in training. Clip duration is **not** child response latency.
- B2 constrains all fitted raw theta hinges and pairwise nonnegative hinge interactions. Unrestricted theta splines and theta×instrument terms are excluded, so the raw-theta partial relationship is monotone conditional on the other predictors.
- D2/D3 use common regression coefficients plus strongly penalized subgroup deviations. This is not two independent high-variance models.
- Outcome labels are only supplied to outcome-model training and metrics. No student, school, examiner, filename, score_status or future reading outcome is a model predictor.
- There is NO audio processing or external model download in M8.6; child pronunciation belongs to M8.7.
- First error, last-three errors and stopping order are **disabled** until an authoritative item presentation sequence is available.

Inner CV recreates S2 and all added item features from inner-training students; outer-validation subjects and labels are never fitted into a feature transformer. Test rows are processed independently through the complete raw-metadata/clip-to-prediction pipeline.

## Decision rules

**Engineering submission eligibility** (declared before experiments):
- ten-repeat mean student pooled OOF ΔR² vs saved E04 strictly positive;
- ≥65% wins on the 50 student folds;
- ten-repeat school-disjoint pooled ΔR² ≥−0.005;
- no instrument-specific pooled ΔR² <−0.010.

**Scientific evidence threshold**: additionally require mean fold ΔR² ≥+0.010, Nadeau–Bengio corrected t ≥2, and ≥60% fold wins. The school/instrument vetoes still apply. The corrected t is approximate under overlapping folds, and model selection among eleven arms creates additional selection bias. Interpret any resulting claims conservatively.

Engineering selection picks the eligible model with highest pooled student CV R², with a predeclared tie-breaker. If none qualifies, do not generate a new candidate submission. E04 retains its historical HOLD_REFERENCE status and M8.4 stays the formally approved fallback.

## E04 diagnostic submission

The optional FULL-only flag --diagnostic-e04-export creates a **separate** refit and CSV from the existing M8.5 E04 code, with singleton/batch-invariance tests and a manifest stating not_formal_promotion. It does not change M8.5's ledger, CV score, or status. The user must manually upload and record actual leaderboard receipts and scores.

## Private output files

~~~text
MyDrive/WildSpeech_M86/m0pp/
  screen/
    manifest.json
    ledger/planned_comparisons.json
    ledger/events.jsonl
    ledger/experiments.parquet
    runs/<runid>/folds/*.parquet
    runs/<runid>/oof_predictions_PRIVATE.parquet
    runs/<runid>/fold_metrics.parquet
    runs/<runid>/pooled_metrics.parquet
    runs/<runid>/instrument_metrics.parquet
    reports/M86_DECISION.md
    reports/M86_DIAGNOSTICS.md
    reports/M86_RESULTS.json
    tables/*.csv
    figures/*.png
  full/
    [identical structure, original 100 frozen folds]
    submission/m0pp_submission.csv [only engineering eligible]
    submission/m0pp_submission.sha256
    submission/m0pp_model.joblib
  diagnostic_e04/ [only explicit opt-in]
    E04_DIAGNOSTIC_MANIFEST.json
~~~

Private school/student records, labels, audio, checkpoint models and OOF predictions must never be committed to GitHub. The CI suite uses synthetic data and cannot establish real competition R².

## M8.7 entry criteria

Verify permissive model licenses and hashes first. Run locally without third-party retention services. Independently validate child-answer localization on permitted **training** clips; do not listen to hidden-test clips. Prioritize sentence-repetition partial credit and blending/deletion near-misses; compare all resulting audio gains to recording-condition-only controls.
