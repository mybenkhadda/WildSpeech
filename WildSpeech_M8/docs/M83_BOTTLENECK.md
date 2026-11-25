# WildSpeech M8.3 — Performance Bottleneck Audit

This is a **drop-in addition** to [WildSpeech M8](https://github.com/mybenkhadda/WildSpeech/tree/main/WildSpeech_M8), not an independent replacement for the repository. It reuses `m8_pipeline.prepare_dataset` and the cached student-level features. **No raw audio is processed or redownloaded.**

## Install locally

1. Copy `src/wildspeech_m8/m83_bottleneck.py` into your checkout at `WildSpeech_M8/src/wildspeech_m8/m83_bottleneck.py`.
2. From `WildSpeech_M8` run:

```bash
python -m pip install -e '.[test]'
python -m wildspeech_m8.m83_bottleneck \
  --data-root /your/official/WildSpeech \
  --output-root /your/WildSpeech_M8 \
  --profile-csv /your/student_profiles_all_covariates.csv \
  --folds 5 --repeats 2
```

`--profile-csv` is optional; M8 attempts to discover it. Set `--quick --folds 3 --repeats 1 --bootstrap 50` for an initial run. A Colab launch notebook is included. On Windows use one command line or PowerShell backticks instead of the shell line continuation above.

## Outputs

`<output_root>/m83_bottleneck/`:

- `M83_BOTTLENECK_REPORT.md` main text report
- `reports/diagnostic_manifest.json` frozen configuration / feature taxonomy
- `tables/fold_metrics.csv` matched train/validation metrics for all feature sets
- `tables/oof_predictions_PRIVATE.csv` de-identified row IDs and school groups, **do not publish**
- `tables/paired_bootstrap.csv` paired student and school bootstrap intervals (positive MAE improvement means the new model is better)
- `tables/subgroup_errors.csv` score, prediction, school, and feature-quantile error slices
- `tables/pair_gap_diagnostics.csv` ranking accuracy by true score-gap size (descriptive only)
- `tables/feature_clusters.csv` unsupervised *training-fitted* acoustic/behavioral-feature geometry, linked to OOF errors for descriptive analysis
- `tables/residual_probe.csv` outer-validation prediction correction using *inner training-only OOF residuals*
- `tables/negative_control.csv` genuine prediction vs shuffled training labels
- `tables/learning_curve.csv` train-size sensitivity
- `figures/` seven automatic diagnostic figures (subject to available columns)

## What makes this different from M8.2?

Rather than finding more correlated student pairs, it directly tests whether the added feature families provide generalizable predictive signal, and whether errors are due to data quantity, domain shift, target noise, underfitting or missing representation. A narrow held-out-school gap and negligible incremental feature value would argue for changing the task representation or label strategy, not adding trees.

## Important cautions

- No official-data score can be inferred from synthetic unit tests.
- School generalization uses group-disjoint CV only when >=3 schools exist. With only a few schools, uncertainty remains high.
- Score and feature subgroup analysis are descriptive and should not be used for model selection without another independent validation layer.
- Inferred label-noise ceilings are *not* identifiable without repeat annotations or additional assumptions.
- The pair ranking numbers are calculated from dependent student pairs, not independent trials.
- Feature-family assignment follows the M8 naming taxonomy, so inspect `diagnostic_manifest.json` carefully.
- `oof_predictions_PRIVATE.csv` contains group indicators and outcomes; keep all outputs in private storage and **do not commit**.
- The student bootstrap captures uncertainty on the observed OOF sample; it does not make repeated CV folds independent.