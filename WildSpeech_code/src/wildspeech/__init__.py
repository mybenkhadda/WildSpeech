"""WildSpeech P01: problem-oriented literacy speech signal discovery.

The package is organised as reusable modules, one per scientific section, plus
infrastructure (configuration, paths, logging, artifact storage and a
dependency-aware stage orchestrator). Importing the package never triggers
data access, downloads, extraction, model training or any other expensive
operation; work only happens when a stage is executed explicitly.
"""

__version__ = "0.1.0"
PACKAGE_NAME = "wildspeech"
