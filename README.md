# expressive-stt

Speech-to-text that keeps *how* something was said: expression (happy, sad, surprised, angry, bored...),
energy, pitch, pace, pauses, laughter and whispers,
next to the words and their timings. It runs locally, uses open models, and needs no API keys.

## Example (real output)

Both examples use Expresso speaker ex04 ([dataset](https://huggingface.co/datasets/ylacombe/expresso)).
The expression model used here was fitted **without** ex04, so this voice is new to it. The clips were picked
by a fixed rule in `eval/make_demo.py`, not by hand, and the output is unedited.

**Read speech.** One sentence read six ways: plain, happy, laughing, whispered, sad, confused.

```
$ estt transcribe demo_read.wav
 0:00  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 0.58, high energy]
 0:06  (pause 0.9s)
 0:07  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [happy 0.85, high energy]
 0:14  (pause 2.0s)
 0:16  "In Mumbai, Thursday is going to be cloudy while Friday will be mostly clear with a high of 28 and a low of both days." [amused 0.82, high energy, pitch ↑]
 0:25  (pause 1.1s)
 0:26  "In Mumbai, Thursday is going to be cloudy, while Friday will be mostly clear with a high of 28 and a low of both days." [whisper]
 0:34  (pause 1.0s)
 0:35  "In Mumbai, Thursday is going to be cloudy, while Friday will be mostly clear with a high of 28 and a low of both days." [sad 0.46, pitch ↓]
 0:42  (pause 1.2s)
 0:43  "In Mumbai, Thursday is going to be cloudy, while Friday will be mostly clear with a high of 28 and a low of both days." [confused 0.94, pitch ↓]
 0:50  (pause 0.8s)
```

Five of the six are right: happy, laughing (shown as amused), whisper, sad and confused. The plain
reading (line 1) is called happy.

**Conversation.** One line from each of six improvised dialogues. The speaker was asked to sound
in awe, happy, sad, angry, bored, and plain.

```
$ estt transcribe demo_conversation.wav
 0:00  "I don't think I could have imagined anything quite like this." [surprised 0.82, pitch ↓]
 0:03  "It's... Let's go to the"                        [happy 0.55, high energy, pitch ↑]
 0:03  (pause 0.9s)
 0:05  (laughs)                                         [laughter]
 0:06  "ice cream shop, yeah?"                          [happy 0.65, high energy]
 0:07  "On..."                                          [high energy, slower]
 0:09  (pause 4.7s)
 0:13  "But see, I was already in motion, and you chose to speed up." [angry 0.93, high energy, pitch ↑]
 0:17  "That's not... Yeah, it's hard to rest when you're waiting for a long time, isn't it?" [confused 0.54, pitch ↓, quiet, slower]
 0:23  (pause 1.2s)
 0:24  "You're gonna want a jacket and a nice umbrella for Sunday just in case that drizzle doesn't go away like I said it would." [happy 0.42, high energy]
```

Right: in awe (shown as surprised), happy with a laugh, angry. Wrong: the sad line ("On...") is mostly
crying. Both voice models read it as high energy, and its top label (happy, 0.38) is too weak to show.
The bored line is called confused, but it gets `quiet, slower`, and the plain line is called happy.
Line 2 joins the end of the first clip with the start of the second.

## What it does

| Step | Tool |
|---|---|
| Transcript, word timings | [WhisperX](https://github.com/m-bain/whisperX) (Whisper large-v3-turbo + wav2vec2 alignment) |
| Emotion from the voice (9 classes) | [emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| Energy (arousal), dominance, valence | [wav2vec2 fine-tuned on MSP-Podcast](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) (audeering) |
| **Expression** (10 labels, below) | a small classifier over the two voice models, pitch spread, pace and laughter, fitted on Expresso (`eval/train_expression.py`) |
| Laughter, whisper (also singing, humming, sigh, crying) | [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) AudioSet classifier + a voicing rule |
| Pitch, loudness, speaking rate, pauses | librosa (pYIN, RMS) and word timings |

**Expression labels:** neutral, happy, amused (laughing), surprised (amazed, shocked), confused,
sad, angry, anxious, bored (or sleepy), disgusted. The text format shows the top label when its
score is at least 0.4. Neutral is not shown. JSON keeps all scores.

**Energy:** `high energy` when arousal is at least 0.6, `low energy` below 0.3. Enthusiasm shows up as
happy or amused **plus** `high energy`. There is no separate "excited" label, because there was no unseen
labelled data to test one on.

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
GPU it runs on CPU. The first run downloads about 5 GB of models from Hugging Face (2.4 GB more
the first time you transcribe Arabic).
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
account. This is the format for talking to an AI tutor. Example:
[`eval/results/demo_output_llm.txt`](eval/results/demo_output_llm.txt).

## Results

Every number comes from the scripts in [`eval/`](eval/README.md). Settings were chosen on Expresso
speakers ex01-ex02. The Expresso numbers below are for speakers ex03 and ex04. Each was scored by a
model fitted on the other three speakers, so the scored voice was never seen in training.

| Task | Data | Result |
|---|---|---|
| Transcript, English | FLEURS en_us test, 100 clips | WER 5.7% |
| Transcript, Arabic | FLEURS ar_eg test, 100 clips | WER 10.2%, CER 3.7% |
| Expression, improvised conversation, 10 labels | Expresso, 2,561 segments (86 clips of up to 3 min) | per segment: accuracy 54.7% (chance 10%), macro-F1 0.54; per clip: 82.6% |
| Expression, read sentences, 5 labels | Expresso, 1,000 clips | accuracy 77.2% (chance 20%), macro-F1 0.78 |
| Same segments, emotion2vec+ alone (on the 7 labels it has) | conversation / read | accuracy 36.2% / 43.4% (this model, same segments: 53.6% / 65.9%) |
| Whispered speech | Expresso, 200 whispered + 800 other | precision 0.97, recall 0.94, F1 0.95 |
| Laughing while speaking | Expresso, 200 laughing + 800 other | precision 0.88, recall 0.74, F1 0.80 |
| A laugh between two sentences | 40 ESC-50 laughs vs 85 other sounds/silences | recall 0.95, false alarms 9.4% |
| Speed, 63 s clip, all models | RTX 5080 / CPU (24 threads) | 2.0 s / 22.0 s (after loading the models, about 13 s) |

How often each expression is found in conversation (recall per segment, test speakers):

| amused | angry | surprised | anxious | happy | disgusted | bored | sad | confused | neutral |
|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.83 | 0.69 | 0.58 | 0.56 | 0.55 | 0.45 | 0.44 | 0.43 | 0.38 |

The `high energy` tag appears on 84% of happy and angry conversation segments and 93% of amused ones,
but only on 1% of bored and 9% of neutral ones.

## Limitations

- **Expression is a hint, not a fact.** In conversation it is right about half the time per segment.
  It gets better over longer stretches: 83% per 3-minute clip.
- **Few voices.** The classifier learned from four acted speakers in a studio. Real, unacted speech and other
  microphones will probably score lower. Results also vary by speaker: 50% for ex03 and 59% for ex04 in
  conversation.
- **Sad and neutral are the hardest.** One test speaker cries when acting sad, and the voice models read
  that as happy or high energy. Plain speech is often called happy, bored or confused.
- **Small labels:** anxious comes from one test speaker only, and surprised from 100 segments.
- **Some whispers are lost entirely.** 13 of the 200 test whispered clips produced no transcript at all,
  because the speech detector missed them. Those clips count as misses above.
- **Some tags are untested.** Singing, humming, sigh and crying are output, but they were not evaluated.
- **Arabic:** only the transcript was measured. Expression and other cues in Arabic were not evaluated.
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
- Arousal/dominance/valence model by Wagner et al., audeering
  ([paper](https://arxiv.org/abs/2203.07378)), **CC BY-NC-SA 4.0**. Its model card says it is for research
  use only.
- The bundled expression classifier (`src/expressive_stt/data/expression_head.npz`, 48 kB) was fitted on
  Expresso (CC BY-NC 4.0) and reads the model above, so treat expression output as **non-commercial**.
  `--no-emotion` turns it off.
- librosa (ISC), FastAPI (MIT).

Evaluation data is used locally and not redistributed:
- [Expresso](https://speechbot.github.io/expresso/) (Nguyen et al. 2023, CC BY-NC 4.0): read speech via
  [ylacombe/expresso](https://huggingface.co/datasets/ylacombe/expresso), and the first 3 minutes of 114
  improvised conversations (204 speaker channels), range-read from the official archive
- [ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak 2015, CC BY-NC 3.0)
- [FLEURS](https://huggingface.co/datasets/google/fleurs) (Conneau et al. 2022, CC BY 4.0)
