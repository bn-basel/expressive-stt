"""Fetch part of the Expresso *conversational* speech (CC BY-NC 4.0) without the full 38 GB archive.

The official release is one uncompressed tar, so we read its headers with HTTP range requests
to build an index, then download only the files we need. Each conversation is a stereo WAV,
one speaker per channel; the folder name gives each speaker's style, e.g. ex01-ex02/awe-sad
means ex01 acts "awe" and ex02 acts "sad".

    uv run python eval/fetch_expresso_conv.py index      # ~10 min, writes index.json
    uv run python eval/fetch_expresso_conv.py fetch      # downloads the chosen styles
"""

import json
import sys
import tarfile
from pathlib import Path

import librosa
import requests
import soundfile as sf

URL = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar"
OUT = Path(__file__).parent / "data" / "expresso_conv"
PREFIX = "expresso/audio_48khz/conversational/"


def index():
    OUT.mkdir(parents=True, exist_ok=True)
    s, off, items = requests.Session(), 0, []
    while True:
        h = s.get(URL, headers={"Range": f"bytes={off}-{off + 511}"}, timeout=60).content
        if len(h) < 512 or not h.strip(b"\0"):
            break
        ti = tarfile.TarInfo.frombuf(h, "utf-8", "surrogateescape")
        if ti.isfile():
            if not ti.name.startswith(PREFIX):
                break  # conversational part is first in the archive
            items.append({"name": ti.name, "offset": off + 512, "size": ti.size})
            if len(items) % 25 == 0:
                print(len(items), ti.name, flush=True)
        off += 512 + (ti.size + 511) // 512 * 512
    (OUT / "index.json").write_text(json.dumps(items, indent=1))
    print("files:", len(items), "GB:", round(sum(i["size"] for i in items) / 1e9, 1))


def parse(name):
    """.../ex01-ex02/awe-sad/ex01-ex02_awe-sad_003.wav -> [(ex01, awe, 0), (ex02, sad, 1)]"""
    pair, styles = name.split("/")[-3:-1]
    spk, sty = pair.split("-"), styles.split("-")
    if len(sty) == 1:
        sty = sty * 2
    return [(spk[c], sty[c], c) for c in range(2)]


def fetch(quota, seconds=180):
    """Download the first `seconds` of up to quota[style] conversations per speaker x style and keep
    that speaker's channel. Resumes: clips already in manifest.json are kept and counted."""
    items = json.loads((OUT / "index.json").read_text())
    man = OUT / "manifest.json"
    rows = json.loads(man.read_text()) if man.exists() else []
    done, have, s = {r["source"] for r in rows}, {}, requests.Session()
    for r in rows:
        have[(r["speaker"], r["style"])] = have.get((r["speaker"], r["style"]), 0) + 1
    for it in sorted(items, key=lambda i: i["name"]):
        if Path(it["name"]).stem in done:
            continue
        wanted = [(spk, sty, ch) for spk, sty, ch in parse(it["name"])
                  if sty in quota and have.get((spk, sty), 0) < quota[sty]]
        if not wanted:
            continue
        # 48 kHz, 2 channels, 24-bit; the header is ~1 kB. soundfile reads a cut-off file fine.
        n = min(it["size"], 4096 + seconds * 48_000 * 2 * 3)
        raw = s.get(URL, headers={"Range": f"bytes={it['offset']}-{it['offset'] + n - 1}"}, timeout=300).content
        tmp = OUT / "tmp.wav"
        tmp.write_bytes(raw)
        audio, sr = sf.read(tmp, dtype="float32", always_2d=True)
        stem = Path(it["name"]).stem
        for spk, sty, ch in wanted:
            out = OUT / f"{stem}_{spk}.flac"
            sf.write(out, librosa.resample(audio[:, ch], orig_sr=sr, target_sr=16_000), 16_000)
            have[(spk, sty)] = have.get((spk, sty), 0) + 1
            rows.append({"file": out.name, "speaker": spk, "style": sty, "source": stem})
            print(out.name, flush=True)
        man.write_text(json.dumps(rows, indent=1))
    (OUT / "tmp.wav").unlink(missing_ok=True)
    print("clips:", len(rows))


# Conversations per speaker x style (first 3 min each). Sad gets more: it was the weakest label.
# The first run used 2 of each (54 files, 2.7 GB transferred); this quota adds 60 files (3.1 GB).
QUOTA = {s: 4 for s in ("default", "calm", "happy", "awe", "confused", "angry", "fearful", "bored", "sleepy",
                        "laughing", "disgusted", "sarcastic", "sympathetic")} | {"sad": 6}

if __name__ == "__main__":
    index() if sys.argv[1:] == ["index"] else fetch(QUOTA)
