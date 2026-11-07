"""Render a Result as JSON, readable text, or a prompt-ready block for an LLM."""

from __future__ import annotations

import json

from .schema import Event, Result, Segment

# Emotions not worth printing in the short formats (still in JSON).
QUIET_EMOTIONS = {"neutral", "other"}
MIN_EMOTION_SCORE = 0.4

EVENT_TEXT = {"laughter": "(laughs)", "pause": "(pause {d:.1f}s)", "singing": "(sings)"}


def ts(t: float) -> str:
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def segment_tags(s: Segment) -> list[str]:
    """The expression label if there is one, else the voice-only emotion; then the cue tags."""
    out = []
    main = s.expression or s.emotion
    if main and main.label not in QUIET_EMOTIONS and main.score >= MIN_EMOTION_SCORE:
        out.append(f"{main.label} {main.score:.2f}")
    out += [t.label for t in s.tags]
    return out


def timeline(r: Result) -> list[tuple[float, Segment | Event]]:
    items: list[tuple[float, Segment | Event]] = [(s.start, s) for s in r.segments]
    items += [(e.start, e) for e in r.events]
    return sorted(items, key=lambda x: x[0])


def to_json(r: Result) -> str:
    return json.dumps(r.to_dict(), ensure_ascii=False, indent=2)


def to_text(r: Result, width: int = 48) -> str:
    lines = []
    for t, item in timeline(r):
        if isinstance(item, Segment):
            body = f'"{item.text}"'
            tags = segment_tags(item)
        else:
            body = EVENT_TEXT.get(item.type, f"({item.type})").format(d=item.end - item.start)
            tags = [item.type] if item.type != "pause" else []
        tag_str = f"[{', '.join(tags)}]" if tags else ""
        lines.append(f"{ts(t):>5}  {body:<{width}} {tag_str}".rstrip())
    return "\n".join(lines)


LLM_HEADER = """\
Below is a transcript of what I said out loud, with cues about HOW I said it.
The cues were estimated automatically from my voice and can be wrong; treat them as hints.
[...] after a line = how I seemed to feel (e.g. happy, amused, surprised, confused, sad, bored,
angry) with the model's confidence (0-1), then voice cues: high/low energy (happy + high energy
often means excited), pitch/loudness/pace compared with my usual voice, whisper, laughing.
(laughs) and (pause) mark sounds and silences between words.
Please respond to what I said and take how I seemed to feel into account."""


def to_llm(r: Result) -> str:
    body = to_text(r, width=0).replace("  ", " ")
    return f"{LLM_HEADER}\n\n<transcript language=\"{r.language}\">\n{body}\n</transcript>"


FORMATS = {"json": to_json, "text": to_text, "llm": to_llm}


def render(r: Result, fmt: str) -> str:
    try:
        return FORMATS[fmt](r)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; choose from {', '.join(FORMATS)}") from None
