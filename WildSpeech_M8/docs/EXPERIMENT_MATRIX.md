# M8.1 Experiment Matrix & Interpretation Protocol

## Central scientific question

**Which early-screening speech and behavioral features add robust out-of-sample information about end-of-year reading scores beyond item correctness and task ability?**

Do not treat group visual separation, training correlations, or class purity as success.

## Experiments

| ID | Comparison | Controlled variable | Output | What counts as evidence |
|---|---|---|---|---|
| A01 | Screening only vs screening+context | Non-screening metadata | `ablation_fold_metrics.csv` | Paired held-out R² and MAE improvements across folds |
| A02 | Baseline vs +temporal | Duration/onset/pausing features | `ablation_summary.csv` | Incremental predictive value beyond ability |
| A03 | Baseline vs +audio | Signal-level acoustic features | Same | Consistent gain across student and school splits |
| A04 | Baseline vs +speech | ASR/phonetic-based features | Same | Consistent gain and meaningful negative control |
| A05 | Baseline vs +audio+speech | Fusion | Same | Complementarity rather than redundancy |
| A06 | Full vs full-minus-family | Leave-one-family-out | Same | Whether a family actually matters to a full model |
| N01 | Full vs shuffled audio | Student alignment of audio features | `ablation_oof.csv` | Observed fusion should beat randomized audio robustly |
| N02 | Full vs shuffled speech | Student alignment of speech hypotheses | Same | Observed fusion should beat randomized speech robustly |
| C01 | Cross-fitted screening residual vs feature residual | Ability screening covariates | `conditional_feature_score_associations.csv` | Association effect size, uncertainty, BH q-values; descriptive only |
| C02 | Matched screening-neighbor pairs | Screening similarity | `ability_matched_pairs.csv` | Examples of residual heterogeneity, not causality |
| G01 | Supervised projection + soft GMM | 80% student subsampling | `descriptive_group_stability.csv` | ARI, minimum group support, school association |
| G02 | Shallow rules on OOF screening residual | Interpretable feature thresholds | `descriptive_residual_rules.txt` | Hypothesis generation only; held-out rule evidence through nested CV |
| G03 | Global vs residual soft grouping vs rules | Outer student CV + LOSO | Existing M8 CV/LOSO tables | Nested-selection performance and replication across schools |
| T01 | Full train vs official test features | Unlabeled feature distribution | `feature_distribution_shift.csv` | Identify covariate-shift risks; do not train on test outcomes |
| R01 | Student bootstrap | Correlation of repeated OOF predictions | `paired_student_bootstrap_ablation.csv` | Within-student replication handling |
| R02 | School-cluster bootstrap | Shared school environment | `paired_school_bootstrap_ablation.csv` | Stable gains after school-level resampling |
| R03 | Paired LOSO school comparison | Unseen school | `loso_school_model_deltas.csv` | Local-model improvement on held-out schools |
| R04 | Instrument-specific prediction bias | KTEA-3 vs WJ IV | `instrument_oof.csv` | Robustness and calibration across outcome instruments |

## Interpretation rules

1. **Performance criterion:** The result must be supported on predeclared evaluation folds, not by searching hundreds of configurations and reporting the best outer-CV result without further confirmation.
2. **Uncertainty criterion:** The paired bootstrap interval should ideally exclude zero and the direction should hold under school resampling. Neither is a substitute for an untouched holdout.
3. **Generalization criterion:** Assess absolute performance, negative transfer and errors school-by-school, including small schools, rather than relying on pooled R² alone.
4. **Speech validity criterion:** A lower model/dictionary edit distance does not prove child speech accuracy. Adult voice, anonymization, phonetic OOV and ASR artifacts may dominate.
5. **Group criterion:** High within-train silhouette/ARI is insufficient. Groups should have interpretable, stable feature signatures and reproducible out-of-sample score differences at matched ability.
6. **Causal criterion:** All associations are observational. Strong correlation or conditional evidence does not establish a causal impact of speech mechanics on reading.
7. **Selection criterion:** The enhanced ablation atlas does **not** automatically alter the official submission; candidate selection remains in the training-only nested CV code.
8. **Leakage criterion:** All outcome-informed transforms, feature selection, residualization, and models are fitted only on the proper training partitions. Frozen audio representations can be extracted from unlabeled clips once.
9. **Missing modality criterion:** Check coverage of each added acoustic/phonetic family in train *and* test. SMOKE partial banks must not contaminate FULL research.
10. **Compute criterion:** Cache clip-level acoustic and phonetic information; reuse in repeated CV. Avoid repeatedly rerunning GPU encoders per fold.

## Which charts answer which question?

- **E01–E02:** What actually adds predictive value?
- **E03–E04:** Which features retain score associations conditional on screening?
- **E05–E06:** How uncertain is the improvement, accounting for both students and schools?
- **E07:** Are train and test feature distributions drifting?
- **E08–E09:** Do outcome-informed descriptive groups persist under perturbations and what residuals do they distinguish?
- **E10:** Are students close on screening still heterogeneous in later outcome?
- **E11:** Is there instrument-specific prediction bias?
- **E12:** Is the final model systematically over-/under-predicting certain performance levels?
- **E13:** Does a local model help or hurt on individual unseen schools?

## Prerequisites and honest scope

- A student-profile CSV from M7/M7.1 is necessary for a comparable full-feature screening baseline; without it, a metadata-derived fallback is a weaker experiment.
- Optional `m8_phonetic.py` runs the published pretrained `drivendata/pasketti-phonetic` model locally; it does not provide validated child/adult diarization or gold phoneme error rates.
- The full official challenge dataset was not bundled with this code. The synthetic demonstration is only an end-to-end integration fixture.
- The first publicly documented reference implementation from DrivenData (24 November 2026) found essentially no average 50-fold CV benefit from simple per-task phonetic edit distance but a small difference on the public leaderboard. This motivates rigorous family-level ablations rather than trusting a single random split. Reference: https://blog.drivendata.org/blog/literacy-speech-modeling-reference