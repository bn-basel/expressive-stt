# Evaluation

Every number in the main README comes from the scripts here. Datasets are downloaded to
`eval/data/` (git-ignored) and are not redistributed.

| Script | Measures | Data (licence) |
|---|---|---|
| `fetch_expresso.py` | downloads 100 utterances per speaker x style | Expresso read speech (CC BY-NC 4.0) |
| `run_expresso.py` → `score_expresso.py` | emotion accuracy; whisper and laughing-speech detection | Expresso |
| `laughter_mix.py` | finding a laugh inserted between two sentences | ESC-50 (CC BY-NC 3.0) + Expresso |
| `fleurs_wer.py` | transcript WER, English and Arabic | FLEURS en_us, ar_eg (CC BY 4.0) |
| `speed.py` | real-time factor, GPU and CPU | FLEURS |
| `make_demo.py` | the README example output | Expresso |

Results are written to `eval/results/*.json`. Run order: `fetch_expresso.py`, `run_expresso.py`,
`score_expresso.py`, then the others in any order (install the eval tools with `uv sync --group eval`).

## Method notes
- **Why Expresso for emotion.** emotion2vec+ was fine-tuned on the EmoBox collection, which includes
  CREMA-D, RAVDESS, IEMOCAP, ESD and others, so those would give inflated scores. Expresso is not in EmoBox.
  Its read styles map to three emotion2vec classes: default → neutral, happy → happy, sad → sad.
  Expresso is acted read speech by four speakers, so it is easier than real conversation.
- **Emotion scoring.** "9-class" = the model's top label over all its classes must match.
  "3-class" = the top label among neutral/happy/sad only. Per clip, segment scores are averaged by duration.
- **Tuning.** Whisper and laughter thresholds, and the emotion context window (0.5 s each side of a
  segment), were picked on speakers ex01-ex02 (dev). Everything is reported on ex03-ex04 (test).
  For reference, on test speakers the emotion window raised 9-class accuracy from 38.3% to 43.7%.
- **Whisper.** A clip is "whispered" if at least half of its transcribed speech is tagged whisper.
  Positives: whisper style. Negatives: default, happy, sad, laughing.
- **Laughing speech.** Expresso "laughing" style = reading while laughing. Positives: laughing style.
  Negatives: default, happy, sad, whisper. Some "happy" clips contain real laughs, which counts against us.
