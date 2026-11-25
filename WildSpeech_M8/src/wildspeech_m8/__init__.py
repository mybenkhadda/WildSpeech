"""WildSpeech M8.1 — score-aware multimodal literacy research."""
from .m8_pipeline import Config, run, read_official, discover_profile
from .m8_research import ResearchConfig, run_research
__all__ = ["Config", "run", "read_official", "discover_profile", "ResearchConfig", "run_research"]