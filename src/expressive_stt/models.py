"""Thin wrappers around the three open models. Each loads once and is reused."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings

import numpy as np

from .schema import Segment, Word

SR = 16_000


def pick_device(device: str = "auto") -> str:
    _check_hf_symlinks()
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@contextlib.contextmanager
def quiet():
    """Keep third-party prints and logs off stdout, so `--format json` stays valid JSON."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    old = logging.root.manager.disable
    logging.disable(logging.WARNING)
    with warnings.catch_warnings(), contextlib.redirect_stdout(sys.stderr):
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            logging.disable(old)


def _check_hf_symlinks() -> None:
    """huggingface_hub tests symlink support lazily and racily; parallel downloads on
    Windows without Developer Mode can then crash with WinError 1314. Testing once,
    up front, makes every later download take the copy fallback."""
    from huggingface_hub.file_download import are_symlinks_supported

    with quiet():
        are_symlinks_supported()


class Transcriber:
    """WhisperX: faster-whisper transcript + wav2vec2 word alignment."""

    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda",
                 language: str | None = None, batch_size: int = 8):
        import whisperx

        self.wx = whisperx
        self.device = device
        self.language = language
        self.batch_size = batch_size
        compute = "float16" if device == "cuda" else "int8"
        with quiet():
            self.model = whisperx.load_model(model, device, compute_type=compute, language=language)
        self._align: dict[str, tuple] = {}

    def __call__(self, audio: np.ndarray, split_pause: float = 1.0) -> tuple[list[Segment], str]:
        with quiet():
            out = self.model.transcribe(audio, batch_size=self.batch_size, language=self.language)
            lang = out["language"]
            if lang not in self._align:
                self._align[lang] = self.wx.load_align_model(language_code=lang, device=self.device)
            model, meta = self._align[lang]
            aligned = self.wx.align(out["segments"], model, meta, audio, self.device)
        segments = []
        for s in aligned["segments"]:
            segments.extend(_split(s, split_pause))
        return segments, lang


def _split(seg: dict, split_pause: float) -> list[Segment]:
    """Turn one WhisperX segment into one or more Segments, cutting at long silences."""
    raw = seg.get("words") or []
    if not raw:
        return [Segment(round(seg["start"], 2), round(seg["end"], 2), seg["text"].strip())]
    chunks: list[list[dict]] = [[]]
    last_end = None
    for w in raw:
        if "start" in w and last_end is not None and w["start"] - last_end >= split_pause:
            chunks.append([])
        chunks[-1].append(w)
        if "end" in w:
            last_end = w["end"]
    out = []
    for ch in chunks:
        timed = [w for w in ch if "start" in w]
        if not timed:
            continue
        words = [Word(round(w["start"], 2), round(w["end"], 2), w["word"], round(w.get("score", 0), 3))
                 for w in timed]
        text = " ".join(w["word"] for w in ch).strip()
        out.append(Segment(words[0].start, words[-1].end, text, words))
    return out


class EmotionModel:
    """emotion2vec+ (via FunASR): 9 emotion classes per utterance."""

    def __init__(self, model: str = "emotion2vec/emotion2vec_plus_large", device: str = "cuda"):
        with contextlib.redirect_stderr(open(os.devnull, "w")), quiet():
            from funasr import AutoModel  # its import and load print hundreds of lines

            self.model = AutoModel(model=model, hub="hf", device=device, disable_update=True,
                                   disable_pbar=True, log_level="ERROR")

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        with quiet():
            res = self.model.generate(audio.astype(np.float32), granularity="utterance",
                                      extract_embedding=False, disable_pbar=True)[0]
        # Labels look like "开心/happy"; keep the English half.
        return {lab.split("/")[-1]: float(sc) for lab, sc in zip(res["labels"], res["scores"])
                if "unk" not in lab}


# AudioSet classes we care about, grouped. A group's score is the max over its classes.
SOUND_GROUPS = {
    "laughter": ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"],
    "whisper": ["Whispering"],
    "singing": ["Singing"],
    "humming": ["Humming"],
    "sigh": ["Sigh"],
    "crying": ["Crying, sobbing"],
}


class SoundTagger:
    """Audio Spectrogram Transformer fine-tuned on AudioSet (527 classes)."""

    def __init__(self, model: str = "MIT/ast-finetuned-audioset-10-10-0.4593", device: str = "cuda"):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.device = device
        with quiet():
            self.fe = ASTFeatureExtractor.from_pretrained(model)
            self.model = ASTForAudioClassification.from_pretrained(model).to(device).eval()
        names = self.model.config.id2label
        index = {v: int(k) for k, v in names.items()}
        self.groups = {g: [index[c] for c in classes] for g, classes in SOUND_GROUPS.items()}

    def __call__(self, clips: list[np.ndarray], batch: int = 16) -> list[dict[str, float]]:
        """Score each clip. Clips longer than 10 s are cut into pieces; the max is kept."""
        pieces, owner = [], []
        for i, c in enumerate(clips):
            for a in range(0, max(len(c), 1), 10 * SR):
                pieces.append(c[a:a + 10 * SR])
                owner.append(i)
        out = [dict.fromkeys(self.groups, 0.0) for _ in clips]
        for b in range(0, len(pieces), batch):
            feats = self.fe(pieces[b:b + batch], sampling_rate=SR, return_tensors="pt")
            with self.torch.no_grad():
                probs = self.torch.sigmoid(self.model(**feats.to(self.device)).logits).cpu().numpy()
            for row, i in zip(probs, owner[b:b + batch]):
                for g, idx in self.groups.items():
                    out[i][g] = max(out[i][g], float(row[idx].max()))
        return out
