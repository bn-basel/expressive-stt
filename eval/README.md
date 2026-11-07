# Evaluation

Every number in the main README comes from the scripts here. Datasets are downloaded to
`eval/data/` (git-ignored) and are not redistributed.

| Script | Measures | Data (licence) |
|---|---|---|
| `fetch_expresso.py` | downloads 100 utterances per speaker x style | Expresso read speech (CC BY-NC 4.0) |
| `fetch_expresso_conv.py` | indexes the official 38 GB archive by HTTP range, then reads only the first 3 min of chosen conversations | Expresso conversations (CC BY-NC 4.0) |
| `extract_features.py` → `train_expression.py` | fits the expression classifier and scores it on unseen speakers | Expresso read + conversations |
| `run_expresso.py` → `score_expresso.py` | emotion2vec+ alone; whisper and laughing-speech detection | Expresso |
| `laughter_mix.py` | finding a laugh inserted between two sentences | ESC-50 (CC BY-NC 3.0) + Expresso |
| `fleurs_wer.py` | transcript WER, English and Arabic | FLEURS en_us, ar_eg (CC BY 4.0) |
| `speed.py` | real-time factor, GPU and CPU | FLEURS |
| `make_demo.py` | the README example output (uses the head fitted without ex04) | Expresso |

Results are written to `eval/results/*.json`. Run order: `fetch_expresso.py`, `run_expresso.py`,
`score_expresso.py`; for expression `fetch_expresso_conv.py index`, `fetch_expresso_conv.py fetch`,
`extract_features.py read`, `extract_features.py conv`, `train_expression.py`, then `make_demo.py`.
The rest run in any order. Install the eval tools with `uv sync --group eval`.

## Method notes
- **Why Expresso for emotion.** emotion2vec+ was fine-tuned on the EmoBox collection, which includes
  CREMA-D, RAVDESS, IEMOCAP, ESD and others, so those would give inflated scores. Expresso is not in EmoBox.
  Its read styles map to three emotion2vec classes: default → neutral, happy → happy, sad → sad.
  Expresso is acted speech by four speakers, so it is easier than real conversation. The audeering
  arousal/valence model was trained on MSP-Podcast, which does not include Expresso either.
- **Emotion scoring.** "9-class" = the model's top label over all its classes must match.
  "3-class" = the top label among neutral/happy/sad only. Per clip, segment scores are averaged by duration.
- **Tuning.** Whisper and laughter thresholds, and the emotion context window (0.5 s each side of a
  segment), were picked on speakers ex01-ex02 (dev). Everything is reported on ex03-ex04 (test).
  For reference, on test speakers the emotion window raised 9-class accuracy from 38.3% to 43.7%.
- **Whisper.** A clip is "whispered" if at least half of its transcribed speech is tagged whisper.
  Positives: whisper style. Negatives: default, happy, sad, laughing.
- **Laughing speech.** Expresso "laughing" style = reading while laughing. Positives: laughing style.
  Negatives: default, happy, sad, whisper. Some "happy" clips contain real laughs, which counts against us.

## Expression classifier
- **Labels** come from the style the speaker was asked to act: default/calm → neutral, happy, laughing →
  amused, awe → surprised, confused, sad, angry, fearful → anxious, bored/sleepy → bored, disgusted.
  Whisper, sarcastic, sympathetic, enunciated, emphasis and narration are left out. Read speech only has
  5 of these labels (neutral, happy, amused, confused, sad). Conversations have all 10.
- **Features:** arousal/dominance/valence and the pooled 1024-d hidden state of the audeering model,
  emotion2vec+'s 9 scores, pitch spread, words per second, voicing, log duration, and the laughter and
  whisper scores. Relative cues (vs. the file's median) are left out, because in a one-mood file they are
  near zero whatever the mood.
- **Classifier:** multinomial logistic regression on standardised features. Labels are balanced within
  each source, and read and conversation get equal total weight. Training uses segments of 0.5 s or more;
  scoring uses all segments.
- **Protocol:** C, the minimum length, the weighting and the feature set were chosen by training on ex01
  and testing on ex02 (and the reverse). The reported numbers are leave-one-speaker-out for ex03 and ex04.
  The shipped head is refit on all four speakers. During development the test numbers were also checked after
  each design round (about six). The choices followed the dev score, but this is still a small risk of
  optimism.
- **Tried and dropped** (no clear gain on dev):
  - a text-emotion model on the transcript (GoEmotions RoBERTa);
  - the emotion2vec+ embedding on top of the audeering features;
  - PCA;
  - smoothing over neighbouring segments. This one raised the test numbers only because each test file
    holds one mood, and it erased sad.
- **Per clip:** segment probabilities are averaged, weighted by duration. One clip is one speaker's
  channel of a conversation, up to 3 min long.
- **Data checks:** the conversation channel order was checked with a speaker-verification model. Only 9 of
  2,383 segments echo the other channel. In read speech no sentence is read by more than one speaker, so
  the words cannot give the speaker's style away across speakers.
