"""Build the README demo clip from Expresso (CC BY-NC 4.0) and print the pipeline output.

One speaker (ex04), five read sentences in different styles, joined with short silences
and one 1.2 s gap. The clip stays local (eval/data/demo.wav); only the text output is shown.

    uv run python eval/make_demo.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_stt.formats import render
from expressive_stt.pipeline import Options, Pipeline, load_audio

HERE = Path(__file__).parent
ORDER = ["default", "happy", "laughing", "whisper", "sad"]
SR = 16_000


def main(speaker="ex04", pick=7):
    sys.stdout.reconfigure(encoding="utf-8")
    rows = list(csv.DictReader(open(HERE / "data" / "expresso" / "manifest.csv", encoding="utf-8")))
    parts, used = [], []
    for style in ORDER:
        r = [r for r in rows if r["speaker"] == speaker and r["style"] == style][pick]
        used.append(r["id"])
        parts += [load_audio(str(HERE / "data" / "expresso" / f"{r['id']}.flac")),
                  np.zeros(int((1.2 if style == "happy" else 0.4) * SR), np.float32)]
    audio = np.concatenate(parts)
    sf.write(HERE / "data" / "demo.wav", audio, SR)
    res = Pipeline(Options(language="en")).run(audio)
    print("clips:", ", ".join(used))
    print(render(res, "text"))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "demo_output.txt").write_text(
        "Source: Expresso " + ", ".join(used) + "\n\n" + render(res, "text") + "\n", encoding="utf-8")
    (HERE / "results" / "demo_output.json").write_text(render(res, "json"), encoding="utf-8")


if __name__ == "__main__":
    main()
