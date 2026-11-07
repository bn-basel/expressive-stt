"""Expression: one label per segment, read from the voice.

A small linear classifier (multinomial logistic regression) reads:
  - arousal / dominance / valence and the pooled hidden state of a wav2vec2 model trained on
    natural podcast speech (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim),
  - the 9 emotion2vec+ scores,
  - pitch spread, speaking rate, voicing and length, and the laughter/whisper scores.
It was fitted with eval/train_expression.py; the weights live in data/expression_head.npz.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .schema import Segment

HEAD_FILE = Path(__file__).parent / "data" / "expression_head.npz"
E2V_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised"]
VOICE_LABELS = ["arousal", "dominance", "valence"]


def prosody_vector(seg: Segment) -> list[float]:
    """Speaker-independent prosody only. The relative cues (pitch vs. the file's median, ...)
    are left out: in a recording with one mood throughout they are ~0 whatever the mood."""
    p = seg.prosody
    dur = float(np.log(max(seg.end - seg.start, 0.05)))
    if p is None:
        return [0.0, 0.0, 1.0, dur]
    return [p.pitch_range_st or 0.0, p.rate_wps or 0.0, p.voiced_fraction, dur]


def features(seg: Segment, hidden: np.ndarray) -> np.ndarray:
    parts = [[seg.voice.get(k, 0.0) for k in VOICE_LABELS],
             hidden.astype(np.float32),
             [seg.emotion_scores.get(k, 0.0) for k in E2V_LABELS],
             prosody_vector(seg),
             [seg.sound_scores.get("laughter", 0.0), seg.sound_scores.get("whisper", 0.0)]]
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])


class Head:
    def __init__(self, z: dict):
        self.labels = [str(x) for x in z["labels"]]
        self.mean, self.scale, self.W, self.b = z["mean"], z["scale"], z["W"], z["b"]

    def __call__(self, x: np.ndarray) -> dict[str, float]:
        logits = self.W @ ((x - self.mean) / self.scale) + self.b
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(self.labels, map(float, p)))


@lru_cache(maxsize=1)
def load_head(path: str = str(HEAD_FILE)) -> Head | None:
    if not Path(path).exists():
        return None
    return Head(dict(np.load(path, allow_pickle=False)))
