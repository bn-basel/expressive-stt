"""Fast tests: no model downloads. Models are replaced with small fakes."""

import json

import numpy as np
import pytest

from expressive_stt import formats, prosody
from expressive_stt.models import _split
from expressive_stt.pipeline import Options, Pipeline, sound_tags, speech_gaps
from expressive_stt.schema import Event, Prosody, Result, Segment, Tag, Word

SR = 16_000


def tone(freq, sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def words(start, n, step=0.3):
    return [Word(round(start + i * step, 2), round(start + i * step + 0.25, 2), f"w{i}") for i in range(n)]


def test_pitch_of_a_pure_tone():
    track = prosody.analyse(tone(220, 1.0))
    p = prosody.segment_prosody(track, Segment(0.1, 0.9, "x", words(0.1, 2)))
    assert p.pitch_hz == pytest.approx(220, rel=0.03)
    assert p.voiced_fraction > 0.9


def test_noise_is_unvoiced():
    noise = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)
    p = prosody.segment_prosody(prosody.analyse(noise), Segment(0.1, 0.9, "x"))
    assert p.voiced_fraction < 0.2


def test_relative_cues():
    segs = [Segment(i, i + 1, "x") for i in range(4)]
    for s, hz, db, rate in zip(segs, [200, 200, 200, 260], [-20, -20, -20, -30], [3, 3, 3, 2]):
        s.prosody = Prosody(hz, None, db, None, rate, None, 1.0)
    prosody.add_relative(segs)
    labels = [t.label for t in prosody.cue_tags(segs[3].prosody)]
    assert labels == ["pitch ↑", "quiet", "slower"]
    assert prosody.cue_tags(segs[0].prosody) == []


def test_no_relative_cues_with_too_few_segments():
    segs = [Segment(0, 1, "x", prosody=Prosody(200, None, -20, None, 3, None, 1.0))]
    prosody.add_relative(segs)
    assert segs[0].prosody.pitch_semitones is None


def test_pauses_between_words():
    segs = [Segment(0, 1, "a", [Word(0, 0.5, "a"), Word(0.6, 1.0, "b")]),
            Segment(2, 3, "c", [Word(2.0, 3.0, "c")])]
    ev = prosody.find_pauses(segs)
    assert [(e.start, e.end, e.type) for e in ev] == [(1.0, 2.0, "pause")]


def test_split_at_long_silence_keeps_untimed_words():
    seg = {"start": 0, "end": 5, "text": "one two 3 four", "words": [
        {"word": "one", "start": 0.0, "end": 0.4, "score": 0.9},
        {"word": "two", "start": 0.5, "end": 0.9, "score": 0.9},
        {"word": "3"},  # WhisperX can't align digits; no timing
        {"word": "four", "start": 3.0, "end": 3.5, "score": 0.9},
    ]}
    out = _split(seg, split_pause=1.0)
    assert [s.text for s in out] == ["one two 3", "four"]
    assert (out[1].start, out[1].end) == (3.0, 3.5)


def test_gaps_between_segments():
    segs = [Segment(1.0, 2.0, "a"), Segment(2.2, 4.0, "b")]
    assert speech_gaps(segs, duration=5.0) == [(0.0, 1.0), (4.0, 5.0)]


def test_sound_tags_and_whisper_rule():
    base = {"laughter": 0.0, "whisper": 0.0, "singing": 0.0, "humming": 0.0, "sigh": 0.0, "crying": 0.0}
    assert sound_tags(base, voiced_fraction=0.8) == []
    assert [t.label for t in sound_tags({**base, "laughter": 0.9}, 0.8)] == ["laughing"]
    rule = sound_tags(base, voiced_fraction=0.0)
    assert rule[0].label == "whisper" and rule[0].source == "rule"


def test_sound_tagger_handles_tiny_and_ragged_clips():
    from expressive_stt.models import SoundTagger

    class FE:  # records what the real feature extractor would receive
        def __init__(self):
            self.lengths = []

        def __call__(self, pieces, **kw):
            self.lengths += [len(p) for p in pieces]
            import torch

            return type("B", (dict,), {"to": lambda self, d: self})({"x": torch.zeros(len(pieces), 1)})

    import torch

    t = SoundTagger.__new__(SoundTagger)
    t.torch, t.device, t.fe = torch, "cpu", FE()
    t.model = lambda **kw: type("O", (), {"logits": torch.zeros(len(kw["x"]), 3)})()
    t.groups = {"laughter": [0], "whisper": [1]}
    out = t([np.zeros(100, np.float32), np.zeros(10 * SR + 320, np.float32)])
    assert len(out) == 2
    assert min(t.fe.lengths) >= SR // 2  # nothing shorter than the extractor can take


class FakeEmotion:
    def __call__(self, audio):
        return {"happy": 0.8, "neutral": 0.2}


class FakeDims:
    """Loud clips get high arousal; the 'hidden state' is 4 ones."""

    def __call__(self, audio):
        a = 0.9 if np.abs(audio).max() > 0.4 else 0.2
        return {"arousal": a, "dominance": 0.5, "valence": 0.5}, np.ones(4, np.float32)


class FakeSounds:
    def __call__(self, clips):
        # Pretend every clip with no tone in it (the gap) is laughter.
        return [{"laughter": 0.9 if np.abs(c).max() < 0.2 else 0.0, "whisper": 0.0, "singing": 0.0,
                 "humming": 0.0, "sigh": 0.0, "crying": 0.0} for c in clips]


def fake_pipeline():
    p = Pipeline(Options(device="cpu"))
    p._emo, p._snd, p._dim = FakeEmotion(), FakeSounds(), FakeDims()
    return p


def fake_head_file(path):
    """A tiny head over 4-d fake hidden states: 'surprised' wins when arousal is high."""
    from expressive_stt import expression

    n = len(expression.VOICE_LABELS) + 4 + len(expression.E2V_LABELS) + 4 + 2
    W = np.zeros((2, n), np.float32)
    W[1, expression.VOICE_LABELS.index("arousal")] = 20.0
    np.savez(path, labels=np.array(["neutral", "surprised"]), W=W, b=np.array([5.0, 0.0], np.float32),
             mean=np.zeros(n, np.float32), scale=np.ones(n, np.float32))


def test_annotate_end_to_end_with_fakes(tmp_path, monkeypatch):
    from expressive_stt import expression

    fake_head_file(tmp_path / "head.npz")
    head = expression.load_head.__wrapped__(str(tmp_path / "head.npz"))
    monkeypatch.setattr(expression, "load_head", lambda: head)
    audio = np.concatenate([tone(200, 1.0), np.zeros(SR, np.float32) + 0.01, tone(200, 1.0, amp=0.5)])
    segs = [Segment(0.0, 1.0, "hello there", words(0.0, 2)), Segment(2.0, 3.0, "no way!", words(2.0, 1))]
    r = fake_pipeline().annotate(audio, segs, "en")
    assert r.segments[0].emotion.label == "happy"
    assert [s.voice["arousal"] for s in r.segments] == [0.2, 0.9]
    assert [s.expression.label for s in r.segments] == ["neutral", "surprised"]
    assert [t.label for t in r.segments[0].tags][:1] == ["low energy"]
    assert "high energy" in [t.label for t in r.segments[1].tags]
    assert [e.type for e in r.events] == ["laughter"]  # pause replaced by the laugh it contains
    text = formats.to_text(r)
    assert '"hello there"' in text and "(laughs)" in text
    assert "surprised 1.00" in text and "happy" not in text  # expression wins over raw emotion
    assert len(r.segments[1].expression_scores) == 2


def sample_result():
    s = Segment(12.0, 14.0, "Wait, all of week 9 is on the exam?", words(12.0, 3),
                emotion=Tag("surprised", 0.71, "emotion2vec"), tags=[Tag("pitch ↑", 1.0, "prosody")])
    return Result("en", 20.0, [s], [Event(15.0, 16.0, "laughter", 0.88)])


def test_formats():
    r = sample_result()
    text = formats.render(r, "text")
    assert text.splitlines()[0].startswith(" 0:12")
    assert "[surprised 0.71, pitch ↑]" in text
    assert "(laughs)" in text.splitlines()[1]
    assert json.loads(formats.render(r, "json"))["segments"][0]["emotion"]["label"] == "surprised"
    llm = formats.render(r, "llm")
    assert "<transcript" in llm and "can be wrong" in llm
    with pytest.raises(ValueError):
        formats.render(r, "xml")


def test_neutral_is_hidden_in_text():
    r = sample_result()
    r.segments[0].emotion = Tag("neutral", 0.9, "emotion2vec")
    assert "neutral" not in formats.to_text(r)


def test_api(tmp_path):
    from fastapi.testclient import TestClient

    from expressive_stt import api

    class P:
        def run(self, path):
            return sample_result()

    api.app.dependency_overrides[api.get_pipeline] = lambda: P()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    files = {"file": ("a.wav", b"RIFF....", "audio/wav")}
    assert c.post("/transcribe", files=files).json()["segments"][0]["text"].startswith("Wait")
    assert "(laughs)" in c.post("/transcribe?format=text", files=files).text
    assert c.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")}).status_code == 400
    api.app.dependency_overrides.clear()
