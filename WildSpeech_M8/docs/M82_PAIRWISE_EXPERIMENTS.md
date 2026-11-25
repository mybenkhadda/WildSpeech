# M8.2 — Multi-view pairwise student discovery

## Design and evidence policy

This project tests whether student behavioral relationships reproduce across **independent feature views** (screening, task behavior, timing, spectral acoustics, phonetic/speech and screening-adjusted audio). It does not assume discrete student phenotypes. Nuisance proxies such as sampling rate, reference text length, microphone or signal-quality indicators are explicitly reserved for auditing and excluded from the behavioral consensus.

**No outcome labels enter clustering.** The 146,070 student pairs (for 541 training students) are constructed without `outcome_score`; outcomes and cross-fitted screening residuals are attached after the partitions have been generated. Every output involving actual outcomes is descriptive unless separately held out.

## Experiment families

| ID | Experiment | Required output | Critical interpretation |
| --- | --- | --- | --- |
| P0 | M8.1 feature-bank integrity, nuisance registry and missingness | `view_manifest.json` | Check no target, ID or school enters views |
| P1 | k-means, diagonal Gaussian mixture and Ward, k=2/3/4/6/8, random rows/columns | `clustering_run_log.csv` | Repeated feature-only partitions |
| P2 | Pair co-membership conditional on valid observed overlap | `all_student_pairs.parquet`, `consensus_matrices.npz` | Imputation is not pairwise evidence |
| P3 | Cluster-size chance control and equal-view consensus | `pair_category_summary.csv` | Low coassignment may simply arise from large k |
| P4 | Robust pairs and disagreement | `stable_but_outcome_discordant.csv`, `view_disagreement_pairs.csv` | Never include self-pairs |
| P5 | Reading score gaps and screening OOF residual gaps | `ability_matched_pair_summary.csv` | Pairs are statistically dependent |
| P6 | Nuisance, same-school and task-view confounding | `school_confound_summary.csv` | Invariant does not mean literacy-specific |
| P7 | Connected components of stable-pair graph | `student_graph_components.csv` | Singletons remain unassigned |
| P8 | Optional, inductive prototype-based local correction with nested OOF residuals | `pairwise_predictive_fold_metrics.csv` | Test on exactly the same outer folds as global CatBoost |
| P9 | Full analytical atlas and evidence report | `M82_RESEARCH_ATLAS.html` | Descriptive figures are not leaderboard evidence |

## Why pairwise consensus is not equal to full outcome-aware metric learning

Pairwise consensus measures structural reliability. An outcome-aware metric would learn from actual score gaps. Such learning requires its own nested fitting loop and must never see the validation student's outcomes, including inside pair selection. M8.2 provides a conservative comparison: a regularized, fixed-shrinkage local correction trained from **training-only out-of-fold residuals** and **inductive view prototypes**. This is deliberately not a transductive graph prediction method.

## Confidence and computational controls

- A pair's denominator counts **only runs containing both students with sufficient jointly observed features**.
- Correlated subviews are averaged inside four domains (screening, task behavior, acoustic, phonetic) before an equal-domain vote; derived conditional speech does not earn a new independent vote. More acoustic configurations cannot inflate acoustic weight.
- Expected chance co-membership is computed from **the realized cluster sizes of each run**, not fixed at 1/k.
- `stable_apart` requires a meaningful negative deviation from chance; a low raw co-membership probability alone is not enough.
- All random seeds, sampled views, failures, actual k, and run counts are recorded.
- Expensive acoustic and pretrained-model extraction is reused from M8.1; M8.2 itself never opens raw audio.
- Full feature and outcome data remain on your Drive, not in GitHub.
- Pairwise-predictive results use student/school bootstrap uncertainty, not a naïve independent-pairs test.

## Typical execution

```bash
pip install -e '.[test]'
python -m wildspeech_m8.m82_pipeline --config configs/m82_experiments.yaml --mode SMOKE
python -m wildspeech_m8.m82_pipeline --config configs/m82_experiments.yaml --mode FULL
pytest -q tests/test_m82.py
```

If Parquet support is unavailable, the complete pair table is exported to `all_student_pairs.csv.gz` rather than lost. This alternate format is explicitly supported.

## Known limitations and research next steps

- No new phonetic embeddings are produced until optional, validated speech extraction is separately executed.
- Feature view taxonomy is based on column naming and must be audited using `view_manifest.json`.
- Consensus is transductive for descriptive training-pair discovery; new students do not have those relationships by default. The predictive comparator uses a separate **inductive** prototype representation trained inside each outer fold.
- Null expected coassignment is **descriptive**: do not quote it as a permutation p-value. Consider negative-control permutation experiments and held-out school shifts before declaring invariant student phenotypes.
- No claim of a leaderboard gain is made until the full official run and independent validation are available.