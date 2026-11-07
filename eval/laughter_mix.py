"""Can the pipeline find a laugh *between* two sentences?

Builds mixes "sentence A + inserted sound + sentence B":
  - positives: the 40 ESC-50 "laughing" clips
  - negatives: 5 clips from each of 9 other human/indoor ESC-50 classes, and 40 plain silences
Sentences: Expresso "default" style, test speakers ex03-ex04 only.
A positive counts as found if a (laughs) event, or a segment tagged "laughing", overlaps the
inserted sound. Any laughter output in a negative mix is a false alarm.

Data: ESC-50 (Piczak 2015, CC BY-NC 3.0), Expresso (CC BY-NC 4.0). Nothing is redistributed.
    uv run python eval/laughter_mix.py
"""

import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from expressive_stt.pipeline import Pipeline, Options, load_audio

HERE = Path(__file__).parent
ZIP = HERE / "data" / "esc50_dl" / "esc50.zip"
NEG_CLASSES = ["crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
               "snoring", "drinking_sipping", "door_wood_knock"]
SR = 16_000


def esc_clips():
    z = zipfile.ZipFile(ZIP)
    rows = list(csv.DictReader(io.TextIOWrapper(z.open("ESC-50-master/meta/esc50.csv"), encoding="utf-8")))
    rng = random.Random(0)
    pos = [r for r in rows if r["category"] == "laughing"]
    neg = [r for c in NEG_CLASSES for r in rng.sample([r for r in rows if r["category"] == c], 5)]

    def read(r):
        y, sr = sf.read(io.BytesIO(z.read(f"ESC-50-master/audio/{r['filename']}")), dtype="float32")
        y = librosa.resample(y if y.ndim == 1 else y.mean(1), orig_sr=sr, target_sr=SR)
        y, _ = librosa.effects.trim(y, top_db=40)
        return y[: 3 * SR]

    return [(r["category"], read(r)) for r in pos], [(r["category"], read(r)) for r in neg]


def rms(y):
    return float(np.sqrt(np.mean(y ** 2)) + 1e-8)


def main():
    rows = [r for r in csv.DictReader(open(HERE / "data" / "expresso" / "manifest.csv", encoding="utf-8"))
            if r["style"] == "default" and r["speaker"] in {"ex03", "ex04"}]
    speech = [load_audio(str(HERE / "data" / "expresso" / f"{r['id']}.flac")) for r in rows]
    pos, neg = esc_clips()
    neg += [("silence", np.zeros(int(1.5 * SR), np.float32)) for _ in range(40)]
    gap = np.zeros(int(0.3 * SR), np.float32)
    p = Pipeline(Options(language="en"))
    rng = random.Random(1)
    results = []
    for label, (cat, snd) in [(1, x) for x in pos] + [(0, x) for x in neg]:
        a, b = rng.sample(speech, 2)
        if cat != "silence":
            snd = snd * (0.7 * rms(a) / rms(snd))  # roughly speech level
        mix = np.concatenate([a, gap, snd, gap, b])
        t0, t1 = (len(a) + len(gap)) / SR, (len(a) + len(gap) + len(snd)) / SR
        r = p.run(mix)
        laughs = [(e.start, e.end) for e in r.events if e.type == "laughter"]
        laughs += [(s.start, s.end) for s in r.segments if any(t.label == "laughing" for t in s.tags)]
        hit = any(s < t1 and t0 < e for s, e in laughs)
        results.append({"label": label, "category": cat, "found": hit, "any_laugh": bool(laughs)})
        print(cat, label, hit, file=sys.stderr, flush=True)
    tp = sum(r["found"] for r in results if r["label"])
    fa = [r for r in results if not r["label"] and r["any_laugh"]]
    out = {
        "n_pos": len(pos), "n_neg": len(neg),
        "recall": round(tp / len(pos), 3),
        "false_alarm_rate": round(len(fa) / len(neg), 3),
        "false_alarms_by_category": {c: sum(r["category"] == c for r in fa) for c in sorted({r["category"] for r in fa})},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "laughter_mix.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
