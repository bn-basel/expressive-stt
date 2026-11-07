"""Run the full pipeline on every Expresso clip and cache the raw scores.

Writes eval/cache/expresso.jsonl (one line per clip). Thresholds are NOT applied here:
the laughter threshold is set to 0 so every non-speech gap keeps its raw score, which
lets score_expresso.py tune thresholds on dev speakers and test on held-out ones.

    uv run python eval/run_expresso.py
"""

import csv
import json
from pathlib import Path

from expressive_stt import pipeline
from expressive_stt.pipeline import Options, Pipeline, load_audio

DATA = Path(__file__).parent / "data" / "expresso"
CACHE = Path(__file__).parent / "cache" / "expresso.jsonl"


def main():
    pipeline.LAUGH_T = 0.0  # keep every gap's laughter score
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        done = {json.loads(line)["id"] for line in CACHE.open(encoding="utf-8")}
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    p = Pipeline(Options(language="en"))
    with CACHE.open("a", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            audio = load_audio(str(DATA / f"{r['id']}.flac"))
            res = p.run(audio)
            out.write(json.dumps({
                "id": r["id"], "speaker": r["speaker"], "style": r["style"], "ref": r["text"],
                "duration": res.duration,
                "clip_emotion": p.emo(audio),  # emotion2vec on the whole clip, for reference
                "segments": [{
                    "start": s.start, "end": s.end, "text": s.text,
                    "emotion_scores": s.emotion_scores, "sound_scores": s.sound_scores,
                    "voiced_fraction": s.prosody.voiced_fraction if s.prosody else None,
                } for s in res.segments],
                "gaps": [{"start": e.start, "end": e.end, "laughter": e.score}
                         for e in res.events if e.type == "laughter"],
            }) + "\n")
            if i % 50 == 0:
                print(i, "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
