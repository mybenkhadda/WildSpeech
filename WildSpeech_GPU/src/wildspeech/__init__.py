"""WildSpeech — local GPU-accelerated research pipeline for literacy prediction from child speech.

Nothing executes on import. Entry points:
    from wildspeech.context import build_context
    from wildspeech.orchestrator import Pipeline
    python -m wildspeech run --profile FAST
"""
__version__ = "1.0.0"

ID = "subject_id"
TARGET = "outcome_score"
FILE = "filename"
TASKS_DEFAULT = ("blending", "deletion", "sentence_repetition")
