# expressive-stt

Speech-to-text that keeps *how* something was said: emotion, pitch, pace, pauses, laughter and whispers,
next to the words and their timings. It runs locally, uses open models, and needs no API keys.

## Example (real output)

Here one speaker reads the same sentence five ways: plain, happy, laughing, whispered, sad. The audio is
from the [Expresso](https://huggingface.co/datasets/ylacombe/expresso) dataset (clips `ex04_*_00008`).
It was rebuilt with `eval/make_demo.py`, and the output below is unedited.

```
$ estt transcribe demo.wav
 0:00  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 1.00]
 0:06  (pause 0.9s)
 0:07  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 1.00]
 0:14  (pause 2.0s)
 0:16  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 1.00, pitch ↑]
 0:25  (pause 1.1s)
 0:26  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 0.96, whisper]
 0:34  (pause 1.0s)
 0:35  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [sad 0.88, pitch ↓]
```

The output gets the whisper, the sad reading and the pitch changes right. It also makes two mistakes,
which show the limits: the plain reading (line 1) is labelled happy, and the laughing reading (line 3) gets
no laughter tag.

## What it does

| Step | Tool |
|---|---|
| Transcript, word timings | [WhisperX](https://github.com/m-bain/whisperX) (Whisper large-v3-turbo + wav2vec2 alignment) |
| Emotion per segment (9 classes) | [emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| Laughter, whisper (also singing, humming, sigh, crying) | [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) AudioSet classifier + a voicing rule |
| Pitch, loudness, speaking rate, pauses | librosa (pYIN, RMS) and word timings |

Pitch, loudness and pace cues are relative to the speaker's own median in the same file. `pitch ↑` means
2+ semitones above their usual level, `quiet` means 6+ dB below, and `faster`/`slower` means ±25% words per
second. You need at least three segments for these cues.

## Install

You need [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12) and `ffmpeg` on your PATH.

```bash
git clone https://github.com/bn-basel/expressive-stt
cd expressive-stt
uv sync
```

On Windows and Linux this installs PyTorch built for CUDA 12.8, which RTX 50-series GPUs need. Without a
GPU it runs on CPU. The first run downloads about 4 GB of models from Hugging Face.
Tested on Windows 11 with an RTX 5080, and on CPU. Linux and macOS are untested.

## Usage

```bash
uv run estt transcribe talk.wav                 # readable text, as above
uv run estt transcribe talk.wav -f json         # segments, words, scores, prosody, events
uv run estt transcribe talk.wav -f llm          # text plus a short header, ready to paste into a chat
uv run estt transcribe talk.wav -l ar           # force the language (default: detect)
uv run estt serve                               # local API on http://127.0.0.1:8000
curl -F "file=@talk.wav" "http://127.0.0.1:8000/transcribe?format=text"
```

From Python:

```python
from expressive_stt import Pipeline
result = Pipeline().run("talk.wav")   # result.segments, result.events, result.to_dict()
```

The `llm` format puts a short header before the transcript. The header tells the model that the cues
are automatic estimates, so it should treat them as hints, and asks it to take your apparent mood into
account. This is the format for talking to an AI tutor.

## Results

Every number comes from the scripts in [`eval/`](eval/README.md). Detection thresholds and the emotion
context window were chosen on Expresso speakers ex01-ex02. All Expresso numbers below are measured on
speakers ex03-ex04, which were never used for tuning.

| Task | Data | Result |
|---|---|---|
| Transcript, English | FLEURS en_us test, 100 clips | WER 5.7% |
| Transcript, Arabic | FLEURS ar_eg test, 100 clips | WER 10.2%, CER 3.7% |
| Emotion: neutral / happy / sad | Expresso, 600 clips | accuracy 54.7% (chance 33%), macro-F1 0.54 |
| Emotion: exact label out of 9 | same | accuracy 43.7% |
| Whispered speech | Expresso, 200 whispered + 800 other | precision 0.97, recall 0.94, F1 0.95 |
| Laughing while speaking | Expresso, 200 laughing + 800 other | precision 0.88, recall 0.74, F1 0.80 |
| A laugh between two sentences | 40 ESC-50 laughs vs 85 other sounds/silences | recall 0.95, false alarms 9.4% |
| Speed, 63 s clip, all models | RTX 5080 / CPU (24 threads) | 1.9 s / 20.7 s (after model load, about 17 s) |

## Limitations

- **Emotion is the weakest part.** On Expresso, plain readings are often labelled happy (120 of 200).
  Expresso is acted, read speech from four speakers, so real conversation is probably harder. Use the
  emotion as a hint, not a fact.
- **Some whispers are lost entirely.** 13 of the 200 test whispered clips produced no transcript at all,
  because the speech detector missed them. Those clips count as misses above.
- **Thresholds were tuned on one dataset.** Two speakers in studio conditions picked them, and they may not
  carry over to other microphones or rooms.
- **Some tags are untested.** Singing, humming, sigh and crying are output, but they were not evaluated.
- **Arabic:** only the transcript was measured. Emotion and other cues in Arabic were not evaluated.
- **One speaker:** there is no speaker diarisation.

## Credits and licences

The code is MIT ([LICENSE](LICENSE)). The models are downloaded at run time under their own licences:

- WhisperX (BSD-2-Clause), faster-whisper and CTranslate2 (MIT). Whisper large-v3-turbo weights (MIT, OpenAI).
  The bundled pyannote VAD is MIT.
- Alignment: wav2vec2 base 960h (fairseq, MIT) for English, and
  [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic)
  (Apache-2.0) for Arabic.
- emotion2vec+ large by Ma et al., Alibaba DAMO / SJTU. Its weights use the
  [FunASR Model Open Source License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE), which
  requires attribution. FunASR itself is MIT.
- Audio Spectrogram Transformer by Gong et al. (MIT CSAIL), BSD-3-Clause.
- librosa (ISC), FastAPI (MIT).

Evaluation data is used locally and not redistributed:
- [Expresso](https://huggingface.co/datasets/ylacombe/expresso) (Nguyen et al. 2023, CC BY-NC 4.0)
- [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak 2015, CC BY-NC 3.0)
- [FLEURS](https://huggingface.co/datasets/google/fleurs) (Conneau et al. 2022, CC BY 4.0)
