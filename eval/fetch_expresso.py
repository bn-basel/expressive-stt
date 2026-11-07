"""Download a balanced subset of Expresso (read speech) for evaluation.

Expresso: Nguyen et al. 2023, https://huggingface.co/datasets/ylacombe/expresso, CC BY-NC 4.0.
Four speakers (ex01-ex04) read the same kind of script in different styles. We take the first
N utterances per (speaker, style) for the styles we can score: default, happy, sad, whisper,
laughing. Only the parquet row groups that contain wanted rows are fetched.
Audio is saved as 16 kHz FLAC under eval/data/expresso/ (git-ignored, not redistributed).

    uv run python eval/fetch_expresso.py --per-style 100
"""

import argparse
import csv
import io
from collections import Counter
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

STYLES = {"default", "happy", "sad", "whisper", "laughing"}
OUT = Path(__file__).parent / "data" / "expresso"
REPO = "datasets/ylacombe/expresso/read"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-style", type=int, default=100)
    n = ap.parse_args().per_style
    OUT.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    taken: Counter = Counter()
    rows = []
    for shard in sorted(fs.ls(REPO, detail=False)):
        pf = pq.ParquetFile(fs.open(shard))
        for g in range(pf.metadata.num_row_groups):
            meta = pf.read_row_group(g, columns=["speaker_id", "style"]).to_pylist()
            if not any(m["style"] in STYLES and taken[m["speaker_id"], m["style"]] < n for m in meta):
                continue
            t = pf.read_row_group(g, columns=["audio", "id", "text", "speaker_id", "style"]).to_pylist()
            for r in t:
                key = (r["speaker_id"], r["style"])
                if r["style"] not in STYLES or taken[key] >= n:
                    continue
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                y = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=16000)
                sf.write(OUT / f"{r['id']}.flac", y, 16000)
                rows.append({"id": r["id"], "speaker": r["speaker_id"], "style": r["style"],
                             "seconds": round(len(y) / 16000, 2), "text": r["text"]})
                taken[key] += 1
        print(shard.rsplit("/", 1)[-1], sum(taken.values()), "clips so far", flush=True)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(sorted(taken.items()))


if __name__ == "__main__":
    main()
