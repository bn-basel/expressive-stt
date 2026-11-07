"""Run the pipeline on Expresso clips and cache every segment's signals for training and scoring.

Sources:
  read  eval/data/expresso/manifest.csv         read sentences, one style per clip
  conv  eval/data/expresso_conv/manifest.json   improvised dialogue, one speaker's channel

Writes eval/cache/feat_<source>.jsonl (one line per segment) and eval/cache/dim/<source>/<id>.npy
(each segment's arousal/dominance/valence + the voice model's 1024-d hidden state, in the same
order). Re-running skips finished clips.

    uv run python eval/extract_features.py read
    uv run python eval/extract_features.py conv
"""

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from expressive_stt.pipeline import Options, Pipeline, load_audio

HERE = Path(__file__).parent
CACHE = HERE / "cache"


def clips(source):
    if source == "read":
        d = HERE / "data" / "expresso"
        for r in csv.DictReader(open(d / "manifest.csv", encoding="utf-8")):
            yield r["id"], r["speaker"], r["style"], d / f"{r['id']}.flac"
    else:
        d = HERE / "data" / "expresso_conv"
        for r in json.loads((d / "manifest.json").read_text()):
            yield Path(r["file"]).stem, r["speaker"], r["style"], d / r["file"]


def main(source):
    out_path = CACHE / f"feat_{source}.jsonl"
    emb_dir = CACHE / "dim" / source
    emb_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open(encoding="utf-8")}
    p = Pipeline(Options(language="en"))
    todo = [c for c in clips(source) if c[0] not in done]
    print(len(todo), "clips to do", flush=True)
    with out_path.open("a", encoding="utf-8") as out:
        for n, (cid, spk, style, path) in enumerate(todo):
            res = p.run(load_audio(str(path)))
            keep = [(i, s) for i, s in enumerate(res.segments) if p.last_features[i] is not None]
            if keep:
                np.save(emb_dir / f"{cid}.npy", np.stack([p.last_features[i] for i, _ in keep]))
            for row, (i, s) in enumerate(keep):
                out.write(json.dumps({
                    "id": cid, "row": row, "speaker": spk, "style": style,
                    "start": s.start, "end": s.end, "text": s.text,
                    "emotion_scores": s.emotion_scores, "voice": s.voice,
                    "sound_scores": s.sound_scores, "prosody": asdict(s.prosody) if s.prosody else None,
                }, ensure_ascii=False) + "\n")
            if not keep:  # remember empty clips so they are not retried
                out.write(json.dumps({"id": cid, "row": None, "speaker": spk, "style": style}) + "\n")
            out.flush()
            if n % 25 == 0:
                print(n, "/", len(todo), cid, len(keep), "segments", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
