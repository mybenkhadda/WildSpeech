# WildSpeech M8.3 — Exhaustive Performance Bottleneck Laboratory

[**Open the Colab notebook**](https://colab.research.google.com/github/mybenkhadda/WildSpeech/blob/main/WildSpeech_M8/notebooks/WILDSPEECH_M83_BOTTLENECK_MASTER.ipynb)

## What gets analyzed

This release makes the **notebook** the complete research interface, with a stable **M8.3 engine** in `src/wildspeech_m8/m83_bottleneck.py`.

The 60 cells cover: (1) Drive mount; (2) GitHub clone/install; (3) experiment profile; (4) official CSV discovery; (5) feature cache/profile detection; (6) official ID and label checks; (7) target + anonymized school atlases; (8) student feature assembly; (9) family missingness; (10) score correlations, descriptive; (11) input/config/code fingerprint; (12) matched-fold diagnostic execution; (13) OOF and school isolation assertions; (14) feature-family ablations; (15) paired student and school bootstraps; (16) random/group generalization; (17) OOF calibration; (18) subgroup errors; (19) score-gap pair ordering; (20) residual correlations; (21) target-blind screening neighbors; (22) nested residual correction; (23) learning curves + shuffled label control; (24) training-fold feature clusters; (25) numeric train/test drift; (26) train/test classifier; (27) bottleneck scorecard; and (28) full HTML report export.

## Colab usage

1. Open the GitHub notebook and execute the **Drive-mount cell first**. Authorize Drive.
2. `DATA_ROOT_OVERRIDE=""` automatically discovers the folder with all six CSVs:
   `train_student_metadata.csv`, `test_student_metadata.csv`, `train_labels.csv`, `submission_format.csv`, `train_clip_metadata.csv`, and `test_clip_metadata.csv`.
3. `OUTPUT_ROOT_OVERRIDE=""` looks for M8 caches in `tables/audio_student_features.csv`, `tables/speech_student_features.csv`, and `tables/phonetic_student_features.csv`.
4. `PROFILE_CSV_OVERRIDE=""` tries previous M7 `student_profiles_all_covariates.csv` exports. The notebook warns if absent. Choose the explicit path when more than one profile exists.
5. `MODE="SMOKE"` uses 3 folds, 1 repeat, limited feature count and a smaller bootstrap to check integration. This is NOT a leaderboard score.
6. Set `MODE="FULL"`, rerun all cells for 5 folds/2 repeats, 450 features and 400 bootstrap resamples. A complete audio feature cache is required by default; completeness flags are validated.
7. Review `<OUTPUT_ROOT>/m83_bottleneck/M83_ATLAS.html`, `reports/M83_DEEP_DIVE.md`, and `tables/notebook_figures.csv`. Never commit files containing student-level OOF scores or individual groups.

For local Jupyter, set `WILDSPEECH_DRIVE_ROOT` to the local mounted Drive folder and use the notebook. No ZIP extraction needed.

## Runtime, checkpointing and safety

The notebook keeps downloaded source in temporary Colab storage, but official data and every report remain in Google Drive. The experiment fingerprint includes feature values, target, school grouping, config, and Git commit so stale runs are not silently reused. The engine prints progress per CV fold and per research phase; it does **not** reprocess raw audio. A full run can be CPU-intensive: run SMOKE first, then FULL with the needed caches.

OOF sample coverage and grouped-CV isolation are checked before diagnostics. All feature transforms and residual correction models are fitted on training folds. Correlations/clusters/neighborhood findings are descriptive. Leave-school-out and randomized folds are different validation estimands; dependent pairs never get naïve IID confidence intervals. Without replicated annotation labels there is **no justified estimate of an irreducible label-noise ceiling**.

## Produced files

- `M83_BOTTLENECK_REPORT.md` and `reports/diagnostic_manifest.json`: core diagnostic evidence
- `M83_ATLAS.html`: extended visual atlas (links to `extended_figures/`)
- `reports/M83_DEEP_DIVE.md`, `reports/M83_NOTEBOOK_MANIFEST.json`: narrative & provenance
- `tables/fold_metrics.csv`, `tables/paired_bootstrap.csv`, `tables/residual_probe.csv`, `tables/negative_control.csv`, `tables/learning_curve.csv`: held-out predictive evidence
- `tables/oof_predictions_PRIVATE.csv`, `tables/notebook_nearest_neighbor_pairs_PRIVATE.csv`: **do not publish**
- `tables/notebook_score_correlations_DESCRIPTIVE.csv`, `tables/notebook_residual_correlations_DESCRIPTIVE.csv`, `tables/notebook_train_test_drift.csv`: descriptive analyses
- `tables/notebook_figures.csv`: figure index
- `extended_figures/`: figure atlas
- `figures/`: engine-generated core figures

### CLI

```bash
pip install -e './WildSpeech_M8[test]'
python -m wildspeech_m8.m83_bottleneck --data-root /path/WildSpeech --output-root /path/WildSpeech_M8 --quick --folds 3 --repeats 1 --bootstrap 60
python -m wildspeech_m8.m83_bottleneck --data-root /path/WildSpeech --output-root /path/WildSpeech_M8 --folds 5 --repeats 2
pytest -q WildSpeech_M8/tests/test_m83.py WildSpeech_M8/tests/test_m83_notebook.py
```

**Evidence status:** Local synthetic integration tests validate code behavior, not actual competition performance. The full official-data run has not been executed by this release.
