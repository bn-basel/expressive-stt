"""Pitch, loudness, speaking rate and pauses, from librosa and word timings.

Cues are relative: each segment is compared with the speaker's own baseline
(the median over the file), so "pitch up" means higher than this speaker usually is.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .schema import Event, Prosody, Segment, Tag

SR = 16_000
HOP = 320  # 20 ms

# Thresholds for turning numbers into cue words (relative to the file baseline).
PITCH_ST = 2.0  # semitones
LOUD_DB = 4.0
QUIET_DB = -6.0
FAST = 1.25
SLOW = 0.8
MIN_PAUSE = 0.6  # seconds of silence between words


@dataclass
class Track:
    """Frame-level features for the whole file, computed once."""

    f0: np.ndarray  # Hz, NaN where unvoiced
    voiced: np.ndarray  # bool
    rms_db: np.ndarray

    def frames(self, start: float, end: float) -> slice:
        a = int(start * SR / HOP)
        b = max(a + 1, int(end * SR / HOP))
        return slice(a, min(b, len(self.f0)))


def analyse(audio: np.ndarray) -> Track:
    f0, voiced, _ = librosa.pyin(
        audio, fmin=65, fmax=600, sr=SR, frame_length=1024, hop_length=HOP
    )
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    rms_db = 20 * np.log10(np.maximum(rms[:n], 1e-5))
    return Track(f0=f0[:n], voiced=voiced[:n].astype(bool), rms_db=rms_db)


def segment_prosody(track: Track, seg: Segment) -> Prosody:
    sl = track.frames(seg.start, seg.end)
    f0, voiced, db = track.f0[sl], track.voiced[sl], track.rms_db[sl]
    # Ignore near-silent frames so pauses inside a segment don't drag the level down.
    active = db > (db.max() - 30) if len(db) else db.astype(bool)
    pitch = float(np.nanmedian(f0[voiced])) if voiced.any() else None
    energy = float(db[active].mean()) if active.any() else -100.0
    voiced_frac = float(voiced[active].mean()) if active.any() else 0.0
    rate = None
    if seg.words:
        spoken = sum(max(w.end - w.start, 0.05) for w in seg.words)
        rate = len(seg.words) / max(spoken, 0.3)
    return Prosody(pitch, None, energy, None, rate, None, voiced_frac)


def add_relative(segments: list[Segment]) -> None:
    """Fill the *_rel fields against the file's median. Needs 3+ segments to mean anything."""
    ps = [s.prosody for s in segments if s.prosody]
    if len(ps) < 3:
        return
    pitches = [p.pitch_hz for p in ps if p.pitch_hz]
    rates = [p.rate_wps for p in ps if p.rate_wps]
    base_pitch = float(np.median(pitches)) if pitches else None
    base_db = float(np.median([p.energy_db for p in ps]))
    base_rate = float(np.median(rates)) if rates else None
    for p in ps:
        if base_pitch and p.pitch_hz:
            p.pitch_semitones = round(12 * np.log2(p.pitch_hz / base_pitch), 2)
        p.energy_rel_db = round(p.energy_db - base_db, 2)
        if base_rate and p.rate_wps:
            p.rate_rel = round(p.rate_wps / base_rate, 2)


def cue_tags(p: Prosody) -> list[Tag]:
    tags = []
    if p.pitch_semitones is not None:
        if p.pitch_semitones >= PITCH_ST:
            tags.append(Tag("pitch ↑", 1.0, "prosody"))
        elif p.pitch_semitones <= -PITCH_ST:
            tags.append(Tag("pitch ↓", 1.0, "prosody"))
    if p.energy_rel_db is not None:
        if p.energy_rel_db >= LOUD_DB:
            tags.append(Tag("louder", 1.0, "prosody"))
        elif p.energy_rel_db <= QUIET_DB:
            tags.append(Tag("quiet", 1.0, "prosody"))
    if p.rate_rel is not None:
        if p.rate_rel >= FAST:
            tags.append(Tag("faster", 1.0, "prosody"))
        elif p.rate_rel <= SLOW:
            tags.append(Tag("slower", 1.0, "prosody"))
    return tags


def find_pauses(segments: list[Segment], min_pause: float = MIN_PAUSE) -> list[Event]:
    """Silences of at least `min_pause` seconds between consecutive words."""
    words = [w for s in segments for w in s.words]
    events = []
    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap >= min_pause:
            events.append(Event(round(a.end, 2), round(b.start, 2), "pause", None))
    return events
