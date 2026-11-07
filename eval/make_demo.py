"""Build the README demo clips from Expresso (CC BY-NC 4.0) and print the pipeline output.

Both demos use speaker ex04, and the expression head is the one fitted WITHOUT ex04
(eval/cache/head_without_ex04.npz, written by train_expression.py), so the voice is unseen.
  1. read: the same sentence read in six styles (clip index 7 of each style, not hand-picked).
  2. conversation: one line from each of six improvised conversations (awe, happy, sad, angry,
     bored, default): the first segment of 2.5 s or more that starts after 20 s.
The audio stays local (eval/data/demo_*.wav); only the text output is shown.

    uv run python eval/make_demo.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from expressive_stt import expression
from expressive_stt.formats import render
from expressive_stt.pipeline import Options, Pipeline, load_audio

HERE = Path(__file__).parent
SR = 16_000
SPEAKER = "ex04"
READ_STYLES = ["default", "happy", "laughing", "whisper", "sad", "confused"]
CONV_STYLES = ["awe", "happy", "sad", "angry", "bored", "default"]


def gap(sec):
    return np.zeros(int(sec * SR), np.float32)


def read_demo(pick=7):
    rows = list(csv.DictReader(open(HERE / "data" / "expresso" / "manifest.csv", encoding="utf-8")))
    parts, used = [], []
    for style in READ_STYLES:
        r = [r for r in rows if r["speaker"] == SPEAKER and r["style"] == style][pick]
        used.append(r["id"])
        parts += [load_audio(str(HERE / "data" / "expresso" / f"{r['id']}.flac")),
                  gap(1.2 if style == "happy" else 0.4)]
    return np.concatenate(parts), used


def conv_demo():
    d = HERE / "data" / "expresso_conv"
    man = json.loads((d / "manifest.json").read_text())
    feats = [json.loads(line) for line in open(HERE / "cache" / "feat_conv.jsonl", encoding="utf-8")]
    parts, used = [], []
    for style in CONV_STYLES:
        clip = sorted(m["file"] for m in man if m["speaker"] == SPEAKER and m["style"] == style)[0]
        cid = Path(clip).stem
        seg = next(f for f in feats if f["id"] == cid and f["row"] is not None
                   and f["start"] >= 20 and f["end"] - f["start"] >= 2.5)
        audio = load_audio(str(d / clip))
        parts += [audio[int((seg["start"] - 0.3) * SR):int((seg["end"] + 0.3) * SR)], gap(0.6)]
        used.append(f"{cid} {seg['start']:.1f}-{seg['end']:.1f}s ({style})")
    return np.concatenate(parts), used


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    head = expression.load_head.__wrapped__(str(HERE / "cache" / f"head_without_{SPEAKER}.npz"))
    expression.load_head = lambda: head
    p = Pipeline(Options(language="en"))
    out, js, llm = [], {}, ""
    for name, (audio, used) in {"read": read_demo(), "conversation": conv_demo()}.items():
        sf.write(HERE / "data" / f"demo_{name}.wav", audio, SR)
        res = p.run(audio)
        text = render(res, "text")
        print(name, "|", "; ".join(used))
        print(text)
        out.append(f"[{name}] Source: Expresso {'; '.join(used)}\n\n{text}\n")
        js[name] = {"source": used, "result": res.to_dict()}
        if name == "conversation":
            llm = render(res, "llm")
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "demo_output.txt").write_text("\n".join(out), encoding="utf-8")
    (res_dir / "demo_output.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    (res_dir / "demo_output_llm.txt").write_text(llm + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
