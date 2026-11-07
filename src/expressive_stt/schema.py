"""Plain data classes for the pipeline output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Word:
    start: float
    end: float
    word: str
    score: float | None = None


@dataclass
class Tag:
    """A label attached to a segment or event, with the model's confidence (0-1)."""

    label: str
    score: float
    source: str  # "emotion2vec", "audioset", "prosody", "rule"


@dataclass
class Prosody:
    pitch_hz: float | None  # median F0 of voiced frames
    pitch_semitones: float | None  # relative to the file's baseline
    energy_db: float  # mean RMS level, dBFS
    energy_rel_db: float | None  # relative to the file's baseline
    rate_wps: float | None  # words per second
    rate_rel: float | None  # ratio to the file's baseline
    voiced_fraction: float  # share of frames with a detected pitch
    pitch_range_st: float | None = None  # spread of pitch inside the segment (semitones, std)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    expression: Tag | None = None  # final label: both voice models + prosody, via a small fitted head
    expression_scores: dict[str, float] = field(default_factory=dict)
    emotion: Tag | None = None  # emotion2vec on the voice alone
    emotion_scores: dict[str, float] = field(default_factory=dict)
    voice: dict[str, float] = field(default_factory=dict)  # arousal, dominance, valence, 0-1
    prosody: Prosody | None = None
    tags: list[Tag] = field(default_factory=list)
    sound_scores: dict[str, float] = field(default_factory=dict)  # raw AudioSet group scores


@dataclass
class Event:
    """Something heard between or over words: laughter, a pause, singing."""

    start: float
    end: float
    type: str  # "laughter", "pause", "singing", ...
    score: float | None = None


@dataclass
class Result:
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
