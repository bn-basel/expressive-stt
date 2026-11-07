"""Speed of the full pipeline (transcript + emotion + sound tags + prosody) on cuda and cpu.

Builds a ~60 s test clip from English FLEURS test clips (Google, CC BY 4.0,
https://huggingface.co/datasets/google/fleurs), then per device: times model load,
one untimed warm-up run, and one timed warm run of `Pipeline().run(path)`.
Real-time factor (RTF) = warm-run seconds / audio seconds (lower is faster).
Each device runs in its own subprocess with a time cap.

Usage: uv run --group eval python eval/speed.py [--cap 900]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
FLEURS_EN = HERE / "data" / "fleurs" / "en_us"
CLIP = HERE / "data" / "bench_60s.wav"
SR = 16_000


def build_clip(target_sec: float = 60.0) -> list[str]:
    """Concatenate whole FLEURS en_us test clips (in test.tsv order) until >= target."""
    from expressive_stt.pipeline import load_audio

    parts, ids, total = [], [], 0
    for line in (FLEURS_EN / "test.tsv").read_text(encoding="utf-8").splitlines():
        wav = line.split("\t")[1]
        a = load_audio(str(FLEURS_EN / "audio" / "test" / wav))
        parts.append(a)
        ids.append(wav)
        total += len(a)
        if total >= target_sec * SR:
            break
    sf.write(CLIP, np.concatenate(parts), SR, subtype="PCM_16")
    return ids


def measure(device: str) -> dict:
    """Runs in a child process; prints one JSON line."""
    import torch

    from expressive_stt.pipeline import Options, Pipeline

    t0 = time.perf_counter()
    p = Pipeline(Options(device=device))
    p.asr, p.emo, p.snd  # noqa: B018  (load all three models now)
    load = time.perf_counter() - t0

    t0 = time.perf_counter()
    p.run(str(CLIP))  # warm-up; also loads the word-alignment model
    first = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = p.run(str(CLIP))
    warm = time.perf_counter() - t0

    info = {"device": device, "torch_threads": torch.get_num_threads()}
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return {**info, "model_load_seconds": round(load, 2), "first_run_seconds": round(first, 2),
            "warm_run_seconds": round(warm, 2), "rtf": round(warm / res.duration, 4),
            "segments": len(res.segments)}


def scrub(text: str) -> str:
    """Keep local paths out of the results file."""
    for p in (str(HERE.parent.resolve()), str(Path.home())):
        text = text.replace(p, "<local>")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["cuda", "cpu"])
    ap.add_argument("--cap", type=int, default=900, help="seconds allowed per device")
    args = ap.parse_args()
    if args.child:
        print(json.dumps(measure(args.child)))
        return

    import torch
    import whisperx  # noqa: F401  (only for its version below)
    from importlib.metadata import version

    ids = build_clip()
    results = {
        "clip": {"seconds": round(sf.info(CLIP).duration, 2), "source": "FLEURS en_us test "
                 "(CC BY 4.0), whole clips concatenated", "fleurs_files": ids},
        "torch": torch.__version__, "whisperx": version("whisperx"),
        "ctranslate2": version("ctranslate2"), "python": platform.python_version(),
        "cpu": platform.processor(), "timing": "model load, then one untimed warm-up run, "
        "then one timed run of Pipeline().run(path) with all models on", "runs": {},
    }
    for device in ("cuda", "cpu"):
        try:
            out = subprocess.run([sys.executable, __file__, "--child", device], capture_output=True,
                                 text=True, timeout=args.cap, check=True)
            results["runs"][device] = json.loads(out.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            results["runs"][device] = {"device": device, "finished": False,
                                       "note": f"did not finish within {args.cap} s"}
        except subprocess.CalledProcessError as e:
            results["runs"][device] = {"device": device, "finished": False,
                                       "error": scrub(e.stderr.strip().splitlines()[-1][:300])}
        print(device, results["runs"][device])

    out = HERE / "results" / "speed.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
