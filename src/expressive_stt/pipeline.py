"""Transcript -> per-segment emotion, prosody cues and vocal events."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from . import prosody
from .schema import Event, Result, Segment, Tag

SR = 16_000

# Decision thresholds. Laughter/whisper values were picked on Expresso speakers
# ex01-ex02 and tested on ex03-ex04 (see eval/README.md). AST gives laughing speech
# very low AudioSet "Laughter" scores, hence the low laughter threshold.
LAUGH_T = 0.01
WHISPER_T = 0.01
WHISPER_MAX_VOICED = 0.0  # whispered speech has no pitch at all
OTHER_SOUND_T = 0.40  # singing/humming/sigh/crying: not tuned or evaluated
MIN_GAP = 0.4  # shortest non-speech stretch worth checking for laughter
MIN_EMOTION_SEC = 0.2
EMOTION_PAD = 0.5  # seconds of context on each side; chosen on dev speakers


def load_audio(path: str) -> np.ndarray:
    """Decode any file ffmpeg understands to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", path, "-f", "s16le",
           "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        import librosa  # no ffmpeg on PATH: fall back to soundfile/audioread

        return librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


@dataclass
class Options:
    whisper_model: str = "large-v3-turbo"
    language: str | None = None
    device: str = "auto"
    emotion: bool = True
    sounds: bool = True


class Pipeline:
    def __init__(self, opts: Options | None = None):
        from .models import pick_device

        self.opts = opts or Options()
        self.device = pick_device(self.opts.device)
        self._asr = self._emo = self._snd = None

    # Models load on first use so `--no-emotion` etc. skip the download.
    @property
    def asr(self):
        if self._asr is None:
            from .models import Transcriber

            self._asr = Transcriber(self.opts.whisper_model, self.device, self.opts.language)
        return self._asr

    @property
    def emo(self):
        if self._emo is None:
            from .models import EmotionModel

            self._emo = EmotionModel(device=self.device)
        return self._emo

    @property
    def snd(self):
        if self._snd is None:
            from .models import SoundTagger

            self._snd = SoundTagger(device=self.device)
        return self._snd

    def run(self, audio: np.ndarray | str) -> Result:
        if isinstance(audio, str):
            audio = load_audio(audio)
        segments, lang = self.asr(audio)
        return self.annotate(audio, segments, lang)

    def annotate(self, audio: np.ndarray, segments: list[Segment], lang: str) -> Result:
        duration = round(len(audio) / SR, 2)
        clip = lambda a, b: audio[int(a * SR):int(b * SR)]  # noqa: E731

        track = prosody.analyse(audio)
        for s in segments:
            s.prosody = prosody.segment_prosody(track, s)
        prosody.add_relative(segments)

        if self.opts.emotion:
            for s in segments:
                if s.end - s.start >= MIN_EMOTION_SEC:
                    scores = self.emo(clip(max(0, s.start - EMOTION_PAD), s.end + EMOTION_PAD))
                    label = max(scores, key=scores.get)
                    s.emotion_scores = {k: round(v, 3) for k, v in scores.items()}
                    s.emotion = Tag(label, round(scores[label], 3), "emotion2vec")

        events = prosody.find_pauses(segments)
        if self.opts.sounds:
            seg_scores = self.snd([clip(s.start, s.end) for s in segments]) if segments else []
            for s, sc in zip(segments, seg_scores):
                s.sound_scores = {k: round(v, 3) for k, v in sc.items()}
                s.tags += sound_tags(sc, s.prosody.voiced_fraction if s.prosody else 1.0)
            gaps = speech_gaps(segments, duration)
            for (a, b), sc in zip(gaps, self.snd([clip(a, b) for a, b in gaps]) if gaps else []):
                if sc["laughter"] >= LAUGH_T:
                    events.append(Event(a, b, "laughter", round(sc["laughter"], 3)))
            laughs = [e for e in events if e.type == "laughter"]
            events = [e for e in events if not (e.type == "pause" and _overlaps(e, laughs))]

        for s in segments:
            if s.prosody:
                s.tags += prosody.cue_tags(s.prosody)

        events.sort(key=lambda e: e.start)
        return Result(lang, duration, segments, events, models=self.model_names())

    def model_names(self) -> dict[str, str]:
        names = {"asr": f"whisperx/{self.opts.whisper_model}"}
        if self.opts.emotion:
            names["emotion"] = "emotion2vec/emotion2vec_plus_large"
        if self.opts.sounds:
            names["sounds"] = "MIT/ast-finetuned-audioset-10-10-0.4593"
        return names


def sound_tags(scores: dict[str, float], voiced_fraction: float) -> list[Tag]:
    """Tags for speech that is laughed, whispered, sung... over the words themselves."""
    tags = []
    if scores["laughter"] >= LAUGH_T:
        tags.append(Tag("laughing", round(scores["laughter"], 3), "audioset"))
    if scores["whisper"] >= WHISPER_T:
        tags.append(Tag("whisper", round(scores["whisper"], 3), "audioset"))
    elif voiced_fraction <= WHISPER_MAX_VOICED:
        tags.append(Tag("whisper", round(1 - voiced_fraction, 3), "rule"))
    for name in ("singing", "humming", "sigh", "crying"):
        if scores[name] >= OTHER_SOUND_T:
            tags.append(Tag(name, round(scores[name], 3), "audioset"))
    return tags


def speech_gaps(segments: list[Segment], duration: float, min_gap: float = MIN_GAP):
    """Stretches with no transcribed words, where a laugh might hide."""
    edges = [0.0] + [t for s in segments for t in (s.start, s.end)] + [duration]
    gaps = [(round(a, 2), round(b, 2)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_gap]
    return gaps


def _overlaps(e: Event, others: list[Event]) -> bool:
    return any(e.start < o.end and o.start < e.end for o in others)
