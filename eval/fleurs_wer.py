"""ASR word error rate of the WhisperX transcriber on FLEURS (en_us, ar_eg).

Measures WER (and CER) of `expressive_stt.models.Transcriber` (large-v3-turbo,
language forced) on the first N utterances of each FLEURS test split.

Data: FLEURS, Google, CC BY 4.0 (https://huggingface.co/datasets/google/fleurs).
Expected layout (downloaded by hand, git-ignored):
    eval/data/fleurs/<lang>/test.tsv
    eval/data/fleurs/<lang>/audio/test/<id>.wav

Usage: uv run --group eval python eval/fleurs_wer.py [--n 100]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from expressive_stt.models import Transcriber
from expressive_stt.pipeline import load_audio

HERE = Path(__file__).parent
LANGS = {"en_us": "en", "ar_eg": "ar"}
MODEL = "large-v3-turbo"

AR_DIACRITICS = re.compile("[\u064B-\u0652\u0670\u0640]")  # harakat, dagger alef, tatweel
AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})

NORMALISATION = {
    "all": "lowercase; every Unicode punctuation character replaced by a space; spaces collapsed",
    "ar": "also: diacritics (U+064B-U+0652, U+0670) and tatweel removed; أ إ آ -> ا; ى -> ي; ة -> ه",
    "not_done": "no number normalisation (e.g. '100' vs 'one hundred' counts as an error)",
    "en_extra": "wer_whisper_normaliser: English WER with Whisper's EnglishTextNormalizer "
                "(numbers to digits, contractions; no spelling map), the usual way Whisper "
                "FLEURS numbers are reported",
}


def normalise(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "ar":
        text = AR_DIACRITICS.sub("", text).translate(AR_MAP)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def read_tsv(path: Path, n: int) -> list[tuple[str, str]]:
    """(wav file name, raw transcription) for the first n rows."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[:n]:
        cols = line.split("\t")
        rows.append((cols[1], cols[2]))
    return rows


def evaluate(folder: Path, lang: str, n: int) -> dict:
    asr = Transcriber(MODEL, device="cuda", language=lang)
    refs, hyps, raw, audio_sec = [], [], [], 0.0
    t0 = time.perf_counter()
    for wav, ref in read_tsv(folder / "test.tsv", n):
        audio = load_audio(str(folder / "audio" / "test" / wav))
        audio_sec += len(audio) / 16_000
        segments, _ = asr(audio)
        hyp = " ".join(s.text for s in segments)
        raw.append((ref, hyp))
        refs.append(normalise(ref, lang))
        hyps.append(normalise(hyp, lang))
    elapsed = time.perf_counter() - t0
    extra = {}
    if lang == "en":
        wn = EnglishTextNormalizer({})
        extra["wer_whisper_normaliser"] = round(
            jiwer.wer([wn(r) for r, _ in raw], [wn(h) for _, h in raw]), 4)
    return {
        "n": len(refs),
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        **extra,
        "empty_hypotheses": sum(1 for h in hyps if not h),
        "audio_seconds": round(audio_sec, 1),
        "transcribe_seconds": round(elapsed, 1),
        "examples": [{"reference": r, "hypothesis": h} for r, h in list(zip(refs, hyps))[:3]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--data", type=Path, default=HERE / "data" / "fleurs")
    args = ap.parse_args()

    results = {
        "dataset": "google/fleurs test split (CC BY 4.0), first N rows of test.tsv",
        "model": f"whisperx/{MODEL} (float16, cuda, language forced)",
        "normalisation": NORMALISATION,
        "languages": {},
    }
    for config, lang in LANGS.items():
        results["languages"][config] = evaluate(args.data / config, lang, args.n)
        r = results["languages"][config]
        print(f"{config}: WER {r['wer']:.3f}  CER {r['cer']:.3f}  n={r['n']}")

    out = HERE / "results" / "fleurs.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
