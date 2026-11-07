"""Score emotion, whisper and laughter detection from eval/cache/expresso.jsonl.

Speakers ex01-ex02 = dev (thresholds picked here), ex03-ex04 = test (reported).
The emotion context window was also chosen on dev, so emotion is reported on test speakers
(and, for reference, on all four).
Writes eval/results/expresso.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

HERE = Path(__file__).parent
DEV, TEST = {"ex01", "ex02"}, {"ex03", "ex04"}
EMO_MAP = {"default": "neutral", "happy": "happy", "sad": "sad"}
EMO3 = ["neutral", "happy", "sad"]


def load():
    return [json.loads(line) for line in open(HERE / "cache" / "expresso.jsonl", encoding="utf-8")]


def weighted(clip, key):
    """Duration-weighted mean of a per-segment score dict."""
    segs = [s for s in clip["segments"] if s[key]]
    if not segs:
        return {}
    w = np.array([s["end"] - s["start"] for s in segs])
    labels = segs[0][key].keys()
    return {k: float(np.dot(w, [s[key].get(k, 0) for s in segs]) / w.sum()) for k in labels}


def emotion(clips):
    out = {}
    clips = [c for c in clips if c["style"] in EMO_MAP]
    y = [EMO_MAP[c["style"]] for c in clips]
    for name, get in [("pipeline_segments", lambda c: weighted(c, "emotion_scores")),
                      ("whole_clip", lambda c: c["clip_emotion"])]:
        scores = [get(c) for c in clips]
        p9 = [max(s, key=s.get) if s else "none" for s in scores]
        p3 = [max(EMO3, key=lambda k: s.get(k, 0)) if s else "none" for s in scores]
        out[name] = {
            "n": len(y),
            "acc_9class": round(float(np.mean([a == b for a, b in zip(y, p9)])), 3),
            "acc_3class": round(float(np.mean([a == b for a, b in zip(y, p3)])), 3),
            "macro_f1_3class": round(f1_score(y, p3, labels=EMO3, average="macro"), 3),
            "confusion_3class": {"labels": EMO3,
                                 "matrix": confusion_matrix(y, p3, labels=EMO3).tolist()},
            "predicted_9class_counts": {k: p9.count(k) for k in sorted(set(p9))},
        }
    return out


def whisper_score(clip, tw, tv):
    """Share of speech time tagged as whisper."""
    segs = clip["segments"]
    if not segs:
        return 0.0
    dur = sum(s["end"] - s["start"] for s in segs)
    hit = sum(s["end"] - s["start"] for s in segs
              if s["sound_scores"].get("whisper", 0) >= tw
              or (1.0 if s["voiced_fraction"] is None else s["voiced_fraction"]) <= tv)
    return hit / max(dur, 1e-6)


def laugh_score(clip):
    vals = [s["sound_scores"].get("laughter", 0) for s in clip["segments"]]
    vals += [g["laughter"] or 0 for g in clip["gaps"]]
    return max(vals, default=0.0)


def prf(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
    return {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3),
            "n_pos": int(sum(y)), "n_neg": int(len(y) - sum(y))}


def per_style_rate(clips, pred):
    styles = sorted({c["style"] for c in clips})
    return {s: round(float(np.mean([pred(c) for c in clips if c["style"] == s])), 3) for s in styles}


def tune(clips, positive, grid, predict):
    dev = [c for c in clips if c["speaker"] in DEV]
    y = [c["style"] == positive for c in dev]
    best = max(grid, key=lambda g: f1_score(y, [predict(c, *g) for c in dev], zero_division=0))
    return best


def detection(clips, positive, grid, predict, names):
    best = tune(clips, positive, grid, predict)
    test = [c for c in clips if c["speaker"] in TEST]
    y = [c["style"] == positive for c in test]
    res = {"thresholds": dict(zip(names, best)), "test": prf(y, [predict(c, *best) for c in test]),
           "test_detection_rate_by_style": per_style_rate(test, lambda c: predict(c, *best))}
    return res


def main():
    clips = load()
    no_speech = {s: sum(1 for c in clips if c["style"] == s and not c["segments"])
                 for s in sorted({c["style"] for c in clips})}
    t = np.round(np.r_[[0.01, 0.02, 0.03], np.arange(0.05, 0.96, 0.05)], 2)
    whisper_full = detection(
        clips, "whisper", [(a, b) for a in t for b in [0.0, 0.1, 0.2, 0.3, 0.4]],
        lambda c, tw, tv: whisper_score(c, tw, tv) >= 0.5, ["audioset_whisper", "max_voiced_fraction"])
    whisper_ast = detection(clips, "whisper", [(a,) for a in t],
                            lambda c, tw: whisper_score(c, tw, -1) >= 0.5, ["audioset_whisper"])
    whisper_rule = detection(clips, "whisper", [(b,) for b in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]],
                             lambda c, tv: whisper_score(c, 2, tv) >= 0.5, ["max_voiced_fraction"])
    laugh = detection(clips, "laughing", [(a,) for a in t],
                      lambda c, tl: laugh_score(c) >= tl, ["audioset_laughter"])
    tl = laugh["thresholds"]["audioset_laughter"]
    marked = [c for c in clips if c["speaker"] in TEST and "<laugh>" in c["ref"]]
    laugh["test_recall_on_clips_with_a_marked_laugh"] = {
        "n": len(marked), "recall": round(float(np.mean([laugh_score(c) >= tl for c in marked])), 3)}
    results = {
        "dataset": "Expresso read speech (CC BY-NC 4.0), first 100 utterances per speaker x style",
        "n_clips": len(clips),
        "clips_with_no_transcribed_speech": no_speech,
        "emotion_test_speakers": emotion([c for c in clips if c["speaker"] in TEST]),
        "emotion_all_speakers": emotion(clips),
        "whisper": {"audioset_or_rule": whisper_full, "audioset_only": whisper_ast,
                    "voicing_rule_only": whisper_rule},
        "laughter": laugh,
        "split": {"dev": sorted(DEV), "test": sorted(TEST)},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "expresso.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
