# WildSpeech P01 — Literacy Signal Discovery

Research codebase for problem-oriented discovery of literacy-related signal in anonymised child speech
(end-of-year reading prediction). Nothing runs on import; every stage is executed explicitly.

## Layout
```
WildSpeech/
  notebooks/WILDSPEECH_P01_LITERACY_SIGNAL_DISCOVERY.ipynb   master notebook (self-contained: embeds the package)
  src/wildspeech/                                            modules (one per section + infrastructure)
  configs/p01_config.yaml                                    full configuration
  tests/                                                     pytest unit tests (synthetic data only)
  tools/build_notebook.py, tools/static_check.py
  requirements.txt, requirements-optional.txt
```
Runtime tree (created by `build_context`): `data/raw` (pointer only — originals are read in place),
`data/extracted/{train,test}` (local fast storage), `cache/*`, `outputs/P01_LITERACY_SIGNAL_DISCOVERY/00_…21_…`,
`models/`, `submissions/`, `reports/`.

## Usage
```python
import sys; sys.path.insert(0, "src")
from wildspeech.pipeline import build_context, Pipeline
ctx = build_context(CONFIG, profile="DEBUG")      # SMOKE | DEBUG | FAST | FULL | RESEARCH | RESUME
pipe = Pipeline(ctx)
pipe.run(["03"])                                  # resolves and runs required dependencies (00, 01)
pipe.status_table()
```
Colab requires `cloud_storage_authorized=True` after you confirm compliance with the data-use agreement.
Schemas are discovered at runtime; ambiguous mappings stop Section 01 and must be set via
`CONFIG["schema_overrides"]` (see `outputs/.../01_data_audit/schema_registry.json`).

## Stage → output directory
00 environment · 01 data_audit · 02 target · 03 screening · 04 item_intelligence · 05 adaptive_testing ·
06 audio_quality (unified extraction + atlas) · 07 temporal · 08 spectral · 09 wavelets ·
10 audio_quality/prosody_anonymization · 11 phonetic · 12 task_intelligence · 13 aggregation ·
14 baselines · 15 student_phenotypes · 16 residual_discovery · 17 confound_analysis · 18 feature_interactions ·
19 literacy_indices · 20 baselines/ablation · 21 baselines/residual_learning · 22 failure_phenotypes ·
23 aggregation/foundation_models · 24 feature_registry · 25 feature_registry/hypotheses ·
26 final_report/visual_atlas · 27 submissions/ · 28 final_report

## Guarantees
Student-disjoint folds; all fitted transforms (common factor, IRT, item references, index scaling, clustering,
imputation, scaling, selection) fitted inside training folds; nested two-stage residual learning; test data
untouched unless `allow_test_inference`; no external APIs; safe TAR handling; atomic, resumable, cached stages
with dependency-aware invalidation; optional dependencies reported as unavailable instead of failing.

## Notes
* `official_metric` defaults to `r2` and is flagged unverified — confirm on the competition page.
* Tests: `pytest` (not executed by the author of this code).
