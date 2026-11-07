"""Expressive speech-to-text: transcripts that keep how something was said."""

from .pipeline import Options, Pipeline, load_audio
from .schema import Event, Result, Segment, Tag, Word

__all__ = ["Options", "Pipeline", "load_audio", "Event", "Result", "Segment", "Tag", "Word"]
__version__ = "0.1.0"
