"""Build the master notebook (no cell is executed; all outputs are empty)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "WILDSPEECH_P01_LITERACY_SIGNAL_DISCOVERY.ipynb"

SECTIONS = [
    ("00", "Environment initialization", "Is the runtime authorised, correctly configured and able to locate the data?",
     "Reproducibility and data-use compliance.", "CONFIG, Drive folder", "Environment detection, Drive mount (Colab, authorised only), recursive data discovery with duplicate detection, hardware/GPU/package diagnostics, seeds.",
     "environment, cloud_storage_authorized, data_root, use_gpu, n_jobs, seed", "environment_report.json, required_file_inventory.csv", "Confirms where data was found and which optional modules can run.", "Package availability ≠ functional correctness."),
    ("01", "Data integrity and hierarchical reconstruction", "Can the dataset be represented reliably at student, task, item, clip and frame level?",
     "Every later analysis depends on correct joins.", "All CSVs, read kwargs JSON, TAR archives (inspected, not extracted)", "Runtime schema resolution with ambiguity detection, ID uniqueness, join-key inference, duplicates/conflicts, train-test overlap, archive safety, clip-audio matching, audio metadata, student-level sampling.",
     "schema_overrides, task_family_overrides, audio_inventory_decode", "student_master, clip_master, task_master, item_master, audio_manifest, schema_registry, data_quality_report", "Review warnings and ambiguity before proceeding.", "Test audio headers decoded only when test inference is allowed."),
    ("02", "Target variable intelligence", "What determines the distribution and predictability of the outcome?", "Instrument and school effects can masquerade as signal.", "student_master", "Robust summaries, outliers, floor/ceiling, GMM modality, instrument/school comparisons, variance decomposition, heteroscedasticity.", "min_group_size", "target_summary.json, target_* tables, figures", "Instruments are compared, never assumed interchangeable.", "Descriptive only."),
    ("03", "Screening intelligence", "How much of the outcome does conventional screening explain?", "Defines the baseline every new signal must beat.", "student_master", "Linear/rank/MI/distance correlations, spline vs linear CV R², partial correlations, conditional distributions, group-conditioned (shrunk) correlations, PCA, missingness, derived features.", "min_n_correlation", "screening_features, screening_associations, figures", "theta_common_factor is refitted inside folds for prediction.", "Profiles are descriptive."),
    ("04", "Item intelligence and IRT", "Are some items substantially more informative?", "Adaptive exposure makes raw accuracy misleading.", "clip_master correctness", "Exposure, smoothed difficulty, rest-score discrimination, outcome association (raw and given theta), redundancy, Rasch/1PL/2PL with BIC selection and SEs.", "enable_irt, irt_*", "item_registry, irt_item_params, irt_student_features", "All-training parameters are EDA only.", "2PL only when identifiable."),
    ("05", "Adaptive behaviour trajectories", "Do response patterns add information beyond the final score?", "Streaks, recovery and stopping difficulty may reflect learning dynamics.", "clip_master order/timestamp", "Order validity check; streaks, recovery, early/late accuracy, difficulty and duration slopes, entropy, volatility.", "—", "order_validity, trajectory_features", "Disabled with a reason if no reliable order exists.", "Dataframe order is never used."),
    ("06", "Acoustic quality atlas and unified extraction", "How reliable is the acoustic information, and what does it contain?", "Quality defines which signals can be trusted.", "TAR archives", "Safe extraction (or streaming), single decode per clip, shared STFT/mel/MFCC/VAD bundle, quality/temporal/spectral/wavelet/prosody features, chunked resumable parallel extraction, stratified atlas.", "sample_rate, n_fft, hop_length, vad_*, extraction_chunk_size, atlas_*", "clip_features, vad_segments, audio_quality_*", "VAD marks speech-like regions, not the child.", "SNR and noise floor are approximations."),
    ("07", "Temporal and fluency intelligence", "Does response organisation predict later reading?", "Fluency is a core reading component.", "clip_features", "Within-item correctness associations, item/student ICC, complexity and difficulty relations, student-level raw/residual/partial associations, complexity-duration surfaces, shrunk complexity sensitivity.", "short_pause_s, long_pause_s", "temporal_family_report, complexity_sensitivity_estimates", "Initial silence is not a verified reaction time.", "Observational."),
    ("08", "Spectral and spectrotemporal intelligence", "Do frequency-domain properties carry literacy signal?", "Articulation precision may be visible spectrally.", "clip_features", "Same family engine for speech-active, full and background regions plus MFCC dynamics, controlling for quality.", "spectral_robust_stats, enable_hpss, enable_chroma", "spectral_family_report", "Background-region features are negative controls.", "Voice conversion may alter spectra."),
    ("09", "Wavelet multiscale discovery", "Do multiscale transients add signal?", "Transitions may be localised in time-frequency.", "clip_features", "DWT/WPD/CWT summaries with adaptive depth; correct vs incorrect, same-item ICC, ability, outcome and matched-residual comparisons.", "wavelet_*, cwt_*", "wavelet_family_report, multiscale_comparisons", "Continuous and discrete families are kept separate.", "High computational cost."),
    ("10", "Prosody and anonymisation robustness", "Which features survive voice conversion?", "Anonymisation changes physiological voice measures.", "clip_features", "Prosody family analysis plus sensitivity classification of all clip features.", "enable_prosody, f0_*, sensitivity_*", "prosody_family_report, anonymization_sensitivity", "No identity, age, sex or medical inference.", "Classification is heuristic."),
    ("11", "Linguistic and phonetic intelligence", "Do expected-vs-produced speech deviations matter?", "Phonological processing is central to literacy.", "item_master, audio (ASR optional)", "Local G2P complexity features; optional licensed local ASR with word/char/phone alignment, confidences, CTC expected likelihood, confusions.", "enable_linguistic_features, enable_phonetic_analysis, enable_asr, asr_models", "item_linguistic_features, clip_phonetic_features", "ASR disagreement is an uncertain proxy.", "Child-speech ASR error rates are high."),
    ("12", "Task-specific intelligence", "Which tasks carry the strongest information?", "Blending, deletion and repetition tap different processes.", "student_features, residuals", "Task relevance matrix and task-specific analyses (deletion categories only with known source and operation).", "—", "task_relevance_matrix, task_specific_report", "Signals are |rho| summaries.", "Association level."),
    ("13", "Student-level aggregation", "How should clips become a student representation?", "The target is student-level; clips are not independent examples.", "clip tables, tabular features", "Global/core robust stats, conditional (task, correctness, difficulty, complexity, quality, length, activity, stage, confidence), cross-task, contrasts, shrunk slopes, item-adjusted features, missingness indicators.", "aggregation_*, item_shrinkage_k, slope_shrinkage_k", "student_features, aggregation_report", "__diff= columns are EDA only.", "Large feature space; selection must be nested."),
    ("14", "Validation framework and screening baselines", "How much variance does screening explain under honest CV?", "Defines residuals and the reference for all deltas.", "screening_features", "Five CV schemes, M00–M06 models with in-pipeline preprocessing, pooled and mean-fold metrics, bootstrap CIs, school metrics, OOF residuals.", "cv_folds, cv_repeats, primary_baseline", "cv_assignments, baseline_results, baseline_residuals", "Residual = observed − OOF screening prediction.", "Unavailable schemes reported."),
    ("15", "Student phenotype atlas", "Are there distinct multimodal profiles?", "Similar screening scores may hide different profiles.", "student_features", "C01–C10 spaces, PCA, k-means/GMM/Ward/spectral/HDBSCAN, internal metrics, bootstrap ARI, composition, characterisation.", "cluster_*", "cluster_assignments, cluster_evaluation, cluster_profiles", "Clusters never use the target.", "Descriptive."),
    ("16", "Residual signal discovery", "What predicts outcome beyond screening?", "Central scientific question.", "student_features, baseline_residuals", "Raw/residual/partial correlations, adjusted cluster-robust regressions, FDR, cluster bootstrap, within-school permutations, held-out half replication, matched comparisons.", "fdr_alpha, discovery_*, matching_calipers, permutation_*", "residual_associations, residual_family_ranking, matched_*", "Associations are not causal.", "Selection optimism outside replication."),
    ("17", "Confounding and robustness", "Are signals genuine or technical shortcuts?", "Recording environment can leak outcome information.", "student_features, residuals", "Negative controls NC01–NC07 (grouped shuffles), region comparisons, school separability, confound matrix, sign consistency, subgroup performance, generalisation gap.", "min_group_size", "negative_controls, school_separability, confound_matrix", "Compare speech vs technical deltas.", "Speaker roles are not identified."),
    ("18", "Feature interactions", "Do features matter only for particular profiles?", "Signals may be conditional on ability or task.", "student_features", "Predefined candidates, fold-fitted representatives, OOF delta R², MI, conditional associations, Friedman H, partial dependence, optional SHAP.", "interaction_candidates", "interaction_results", "No unrestricted all-pairs search.", "Multiplicity across candidates."),
    ("19", "Experimental literacy indices", "Do interpretable composites help?", "Composites may be more stable and interpretable.", "student_features", "L01–L15 with declared directions, train-fitted robust scaling, reliability, residual association, incremental CV, cross-school stability.", "index_min_components", "literacy_indices, literacy_index_report", "Experimental constructs.", "irt_/ia_ components use EDA versions here (A14 is fold-fitted)."),
    ("20", "Predictive ablation", "Which modalities add predictive value?", "Controlled comparison of feature families.", "student_features, cv_assignments", "A00–A23 under student and school-grouped CV, paired bootstrap deltas, leave-one-family-out, pairwise complementarity, nested forward selection, redundancy.", "ablation_*", "ablation_results, ablation_oof_predictions", "Deltas with CIs, not single scores.", "Expensive in FULL mode."),
    ("21", "Nested residual learning", "Does two-stage residual learning beat direct regression?", "Residual correction can isolate incremental acoustic value.", "student_features", "Nested inner-OOF residual targets, inner-selected alpha, refit, outer evaluation; compared with direct and nested blending.", "residual_fusion_alphas, inner_cv_folds", "residual_learning_results", "Alpha never sees outer outcomes.", "—"),
    ("22", "Failure phenotype atlas", "Which students are hard to predict, and why?", "Failures point to missing information.", "OOF predictions", "Failure registry, rule-based F01–F12 flags, empirical failure clusters, recovery by phenotype.", "—", "failure_registry, failure_summary", "Descriptive, not diagnoses.", "Regression to the mean."),
    ("23", "Foundation-model exploration (optional)", "Do pretrained representations add information?", "Frozen encoders may capture complementary structure.", "audio, vad_segments", "Licensed local frozen encoders, layer selection, mean/std/quantile/speech-active pooling, PCA, task/student pooling, OOM recovery.", "enable_embeddings, embedding_models", "embedding_student_features", "Requires verified licence and overlap checks.", "Disabled by default."),
    ("24", "Feature discovery registry", "Which features carry robust signal?", "Separate evidence rankings instead of one importance score.", "all features and evidence", "Per-feature provenance, associations, FDR, CIs, CV gains, stability, redundancy, confound risk, sensitivity, classification.", "feature_registry_*", "feature_registry, feature_ranking_*", "Statuses follow explicit rules.", "Top-K CV only."),
    ("25", "Hypothesis registry", "Which hypotheses are supported?", "Explicit scientific accountability.", "registries and ablations", "H01–H20 evaluated programmatically with conservative status rules.", "—", "hypothesis_registry", "SUPPORTED requires OOF CI > 0 and grouped confirmation.", "NOT_TESTED when artifacts are missing."),
    ("26", "Visual atlas", "Where are all figures?", "Navigation of results.", "figure_index.csv", "Dataset diagrams and an HTML gallery grouped into five atlases.", "save_figures", "P01_VISUAL_ATLAS.html", "Local file only.", "—"),
    ("27", "Submission (disabled by default)", "Can a validated submission be produced safely?", "Competition deliverable.", "train/test features", "Train-only fitting, per-student test features, order-preserving format validation, provenance.", "enable_submission, allow_test_inference, submission_*", "submissions/*.csv", "Never uploaded automatically.", "Requires test processing."),
    ("28", "Final scientific report", "What did P01 find?", "Programmatic synthesis.", "all artifacts", "Answers 18 questions from artifacts, writes P01_* deliverables and HTML report.", "save_html_report", "P01_EXECUTIVE_SUMMARY.md, P01_SCIENTIFIC_REPORT.html, …", "Skipped/failed sections are stated.", "No fabricated findings."),
]

CONFIG_CELL = '''# ===================== CONFIGURATION =====================
CONFIG = {
    "project_name": "WildSpeech",
    "experiment_name": "P01_LITERACY_SIGNAL_DISCOVERY",
    "drive_folder_id": "10hOoRyDO2VwZ9rgU1grTe0tb97_MQjnf",
    "drive_folder_url": (
        "https://drive.google.com/drive/folders/"
        "10hOoRyDO2VwZ9rgU1grTe0tb97_MQjnf"
    ),

    "environment": "COLAB",
    "cloud_storage_authorized": False,

    "drive_root": None,
    "data_root": None,
    "local_root": None,
    "output_root": None,

    "run_mode": "FULL",
    "seed": 42,
    "use_gpu": True,
    "n_jobs": -1,

    "sample_rate": 16000,
    "audio_batch_size": 64,
    "debug_max_students": 30,
    "debug_max_clips": 300,

    "extract_train_audio": True,
    "extract_test_audio": False,

    "train_only_eda": True,
    "allow_test_inference": False,
    "allow_external_apis": False,

    "enable_tabular_eda": True,
    "enable_item_analysis": True,
    "enable_irt": True,
    "enable_acoustic_features": True,
    "enable_vad": True,
    "enable_wavelets": True,
    "enable_prosody": True,
    "enable_phonetic_analysis": False,
    "enable_asr": False,
    "enable_embeddings": False,
    "enable_student_clustering": True,
    "enable_residual_analysis": True,
    "enable_confound_analysis": True,
    "enable_feature_interactions": True,
    "enable_baseline_models": True,
    "enable_failure_analysis": True,
    "enable_submission": False,

    "cv_folds": 5,
    "cv_repeats": 3,
    "bootstrap_iterations": 1000,
    "fdr_alpha": 0.05,

    "resume": True,
    "overwrite_cache": False,

    "save_figures": True,
    "save_tables": True,
    "save_html_report": True,
    "figure_dpi": 160,
    "max_features_for_heatmap": 60,
}

# Optional advanced overrides (see configs/p01_config.yaml for every key), e.g.
# CONFIG["schema_overrides"] = {"clip": {"task": "task_name"}}
# CONFIG["asr_models"] = [{"alias": "w", "type": "whisper", "checkpoint": "/path", "license_verified": True, "eligible_for_competition": True}]
PROFILE = CONFIG["run_mode"]   # SMOKE | DEBUG | FAST | FULL | RESEARCH | RESUME
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def build():
    cells = [md("# WILDSPEECH P01 — Literacy Signal Discovery\n\n"
                "Problem-oriented discovery of literacy-related signal in anonymised child speech, from data integrity to a "
                "validated feature-discovery and hypothesis registry.\n\n"
                "**Execution policy.** Nothing runs automatically. Every expensive step is guarded by a flag that defaults to "
                "`False`. Cloud processing stays disabled until you set `cloud_storage_authorized=True` after confirming that your "
                "environment complies with the competition data-use agreement. No data is sent to external APIs.\n\n"
                "**Order of use:** (1) write/locate the package, (2) optional installs, (3) configure, (4) initialise context, "
                "(5) run sections individually or the full pipeline at the end.")]
    cells.append(md("## A. Package bootstrap\n\nThe cells in this section write the `wildspeech` package to `CODE_ROOT`. "
                    "If you already copied the repository (zip or Git), set `CODE_ROOT` to it and **skip the `%%writefile` cells** "
                    "(they only overwrite source files; they never touch data)."))
    cells.append(code('import os, sys\nfrom pathlib import Path\n\n'
                      'CODE_ROOT = Path("/content/WildSpeech_code") if "google.colab" in sys.modules else Path.cwd() / "WildSpeech_code"\n'
                      'for sub in ("src/wildspeech", "configs", "tests", "tools"):\n    (CODE_ROOT / sub).mkdir(parents=True, exist_ok=True)\n'
                      'if str(CODE_ROOT / "src") not in sys.path:\n    sys.path.insert(0, str(CODE_ROOT / "src"))\nprint("CODE_ROOT =", CODE_ROOT)\n'))
    files = sorted((ROOT / "src" / "wildspeech").glob("*.py")) + [ROOT / "configs" / "p01_config.yaml", ROOT / "requirements.txt",
                                                                  ROOT / "requirements-optional.txt", ROOT / "pytest.ini"] + sorted((ROOT / "tests").glob("*.py"))
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        cells.append(code(f"%%writefile {{CODE_ROOT}}/{rel}\n" + f.read_text()))
    cells.append(md("## B. Dependencies (optional)\n\nColab ships most core packages. Set `INSTALL_DEPENDENCIES=True` only if imports fail."))
    cells.append(code('INSTALL_DEPENDENCIES = False\nINSTALL_OPTIONAL = False\nif INSTALL_DEPENDENCIES:\n    import subprocess\n'
                      '    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements.txt")], check=True)\n'
                      '    if INSTALL_OPTIONAL:\n        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_ROOT / "requirements-optional.txt")], check=False)\n'))
    cells.append(md("## C. Configuration"))
    cells.append(code(CONFIG_CELL))
    cells.append(md("## D. Execution controls and context\n\n`INITIALIZE_CONTEXT` mounts Drive (Colab, authorised only) and creates the project tree. "
                    "`RUN_SECTIONS` controls each section below; dependencies are resolved automatically."))
    cells.append(code('INITIALIZE_CONTEXT = False\nINCLUDE_DEPENDENCIES = True\nRUN_SECTIONS = {f"{i:02d}": False for i in range(29)}\n\nctx = pipe = None\n'
                      'if INITIALIZE_CONTEXT:\n    from wildspeech.pipeline import Pipeline, STAGE_REGISTRY, build_context\n'
                      '    ctx = build_context(CONFIG, profile=PROFILE, yaml_path=CODE_ROOT / "configs" / "p01_config.yaml")\n'
                      '    pipe = Pipeline(ctx)\n    display(pipe.status_table())\n\n\n'
                      'def run_section(stage_id):\n    if not RUN_SECTIONS.get(stage_id):\n        print(f"Section {stage_id}: not requested (set RUN_SECTIONS[\'{stage_id}\'] = True)")\n        return None\n'
                      '    if pipe is None:\n        raise RuntimeError("Set INITIALIZE_CONTEXT = True and run the context cell first")\n'
                      '    report = pipe.run([stage_id], include_dependencies=INCLUDE_DEPENDENCIES)\n    display(report)\n    return report\n'))
    for sid, title, q, mot, inputs, meth, opts, outs, interp, lim in SECTIONS:
        cells.append(md(f"## Section {sid} — {title}\n\n**Research question.** {q}\n\n**Motivation.** {mot}\n\n**Inputs.** {inputs}\n\n"
                        f"**Methodology.** {meth}\n\n**Configuration.** `{opts}`\n\n**Outputs.** {outs}\n\n**Interpretation.** {interp}\n\n"
                        f"**Limitations.** {lim}"))
        cells.append(code(f'section_{sid}_report = run_section("{sid}")\n'))
    cells.append(md("## Unit tests (optional)\n\nSynthetic fixtures only; never uses competition data."))
    cells.append(code('RUN_TESTS = False\nif RUN_TESTS:\n    import subprocess\n    subprocess.run([sys.executable, "-m", "pytest", "-q", str(CODE_ROOT / "tests")], cwd=CODE_ROOT, check=False)\n'))
    cells.append(md("## Full pipeline\n\nRuns every enabled stage of the selected profile in dependency order, reusing validated caches."))
    cells.append(code('RUN_PIPELINE = False\n\nif RUN_PIPELINE:\n    if pipe is None:\n        raise RuntimeError("Initialise the context first (INITIALIZE_CONTEXT = True)")\n'
                      '    full_report = pipe.run()\n    display(full_report)\n    display(pipe.status_table())\n'))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    NB.parent.mkdir(parents=True, exist_ok=True)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {NB} with {len(cells)} cells")


if __name__ == "__main__":
    build()
