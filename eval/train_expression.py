"""Fit the expression head and measure it on speakers it never saw.

Labels come from the style Expresso speakers were asked to act (read sentences and improvised
dialogue). Choices (features, regularisation) are made on speakers ex01-ex02 only, by training on
one and testing on the other (score = mean macro-F1 of read and conversation). Reported numbers then use leave-one-speaker-out on ex03 and ex04:
train on the other three speakers, test on the held-out one. The shipped head is refit on all four.

    uv run --group eval python eval/train_expression.py              # tune, report, save head
    uv run --group eval python eval/train_expression.py --no-save    # tune and report only
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from expressive_stt import expression
from expressive_stt.pipeline import HIGH_ENERGY, LOW_ENERGY
from expressive_stt.schema import Prosody, Segment

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"

# Acted style -> expression label. Styles left out: whisper (a tag of its own), enunciated,
# emphasis, sarcastic, sympathetic, narration and the other non-emotional styles.
STYLE_TO_LABEL = {
    "default": "neutral", "calm": "neutral",
    "happy": "happy", "laughing": "amused", "awe": "surprised", "confused": "confused",
    "sad": "sad", "angry": "angry", "fearful": "anxious", "bored": "bored", "sleepy": "bored",
    "disgusted": "disgusted",
}
DEV, TEST = ["ex01", "ex02"], ["ex03", "ex04"]
MIN_SEC = 0.5  # training segments only; scoring uses every segment (0.5 > 1.0 > 1.5 on dev)


def load(source):
    rows, X = [], defaultdict(dict)
    path = CACHE / f"feat_{source}.jsonl"
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["row"] is None or r["style"] not in STYLE_TO_LABEL:
            continue
        r["source"], r["label"] = source, STYLE_TO_LABEL[r["style"]]
        rows.append(r)
    for cid in {r["id"] for r in rows}:
        X[cid] = np.load(CACHE / "dim" / source / f"{cid}.npy")  # voice dims (3) + hidden (1024)
    return rows, X


def segment(r, dims):
    s = Segment(r["start"], r["end"], r["text"])
    s.emotion_scores, s.sound_scores = r["emotion_scores"], r["sound_scores"]
    s.voice = {k: round(float(v), 3) for k, v in zip(expression.VOICE_LABELS, dims[:3])}
    s.prosody = Prosody(**r["prosody"]) if r["prosody"] else None
    return s


def matrix(rows, feats):
    out = []
    for r in rows:
        d = feats[r["source"]][r["id"]][r["row"]]
        out.append(expression.features(segment(r, d), d[3:]))
    return np.stack(out)


def fit(X, y, src, C):
    """Labels are balanced within each source (read, conv), and each source gets equal total weight,
    so the many conversation segments don't swamp the read ones (better than pooled on dev)."""
    mean, scale = X.mean(0), X.std(0) + 1e-6
    w = np.zeros(len(y))
    for s in set(src):
        m = src == s
        w[m] = compute_sample_weight("balanced", y[m]) / m.sum()
    w *= len(y) / w.sum()
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit((X - mean) / scale, y, sample_weight=w)
    return clf, mean, scale


def predict(model, X):
    clf, mean, scale = model
    return clf.predict_proba((X - mean) / scale), list(clf.classes_)


def clip_votes(rows, proba, classes):
    """One label per clip: average the segment probabilities, weighted by duration."""
    acc = defaultdict(lambda: np.zeros(len(classes)))
    gold = {}
    for r, p in zip(rows, proba):
        acc[(r["source"], r["id"])] += p * (r["end"] - r["start"])
        gold[(r["source"], r["id"])] = r["label"]
    keys = sorted(acc)
    return [gold[k] for k in keys], [classes[int(np.argmax(acc[k]))] for k in keys], keys


def scores(y, pred, labels):
    return {"n": len(y), "accuracy": round(float(np.mean(np.array(y) == np.array(pred))), 3),
            "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 3)}


def save_head(path, model):
    clf, mean, scale = model
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=np.array(clf.classes_), W=clf.coef_.astype(np.float32),
                        b=clf.intercept_.astype(np.float32), mean=mean.astype(np.float32),
                        scale=scale.astype(np.float32))


def dev_score(X, y, spk, dur, src, C):
    """Train on one dev speaker, test on the other, both ways; mean macro-F1 over the sources."""
    f1s = []
    for a, b in ((DEV[0], DEV[1]), (DEV[1], DEV[0])):
        tr = (spk == a) & (dur >= MIN_SEC)
        model = fit(X[tr], y[tr], src[tr], C)
        for source in sorted(set(src)):
            te = (spk == b) & (src == source)
            proba, cl = predict(model, X[te])
            f1s.append(f1_score(y[te], np.array(cl)[proba.argmax(1)], labels=sorted(set(y[te])),
                                average="macro"))
    return float(np.mean(f1s))


def main(save=True):
    sources = [a for a in sys.argv[1:] if a in ("read", "conv")] or ["read", "conv"]
    data = {s: load(s) for s in sources if (CACHE / f"feat_{s}.jsonl").exists()}
    rows = [r for s in data for r in data[s][0]]
    feats = {s: data[s][1] for s in data}
    print(Counter((r["source"], r["label"]) for r in rows))

    X = matrix(rows, feats)
    y = np.array([r["label"] for r in rows])
    spk = np.array([r["speaker"] for r in rows])
    dur = np.array([r["end"] - r["start"] for r in rows])
    src = np.array([r["source"] for r in rows])

    # 1) choose C on dev speakers only
    res = {C: dev_score(X, y, spk, dur, src, C) for C in (0.003, 0.01, 0.03, 0.1)}
    C = max(res, key=res.get)
    print("dev macro-F1 by C:", {k: round(v, 3) for k, v in res.items()}, "-> C =", C)

    # 2) report: leave one test speaker out (train on the other three)
    report = {"C": C, "dev_macro_f1_by_C": {str(k): round(v, 3) for k, v in res.items()},
              "min_train_sec": MIN_SEC, "test_speakers": TEST}
    allidx, allp, cl = [], [], None
    for t in TEST:
        tr, te = (spk != t) & (dur >= MIN_SEC), np.where(spk == t)[0]
        model = fit(X[tr], y[tr], src[tr], C)
        if save:  # used by make_demo.py, so the demo voice is one the head never saw
            save_head(CACHE / f"head_without_{t}.npz", model)
        proba, cl = predict(model, X[te])
        allidx += list(te)
        allp.append(proba)
    allidx, proba = np.array(allidx), np.concatenate(allp)
    pred = np.array([cl[i] for i in proba.argmax(1)])
    gold = y[allidx]
    for source in sorted(set(src)):
        m = src[allidx] == source
        g, p = list(gold[m]), list(pred[m])
        labels = sorted(set(g))
        out = {"labels": labels, "chance_accuracy": round(1 / len(labels), 3),
               "segments": scores(g, p, labels)}
        cg, cp, _ = clip_votes([rows[i] for i in allidx[m]], proba[m], cl)
        out["clips"] = scores(cg, cp, labels)
        out["per_label_recall"] = {lab: round(float(np.mean([pp == lab for gg, pp in zip(g, p) if gg == lab])), 3)
                                   for lab in labels}
        out["per_speaker_accuracy"] = {t: scores(list(gold[m & (spk[allidx] == t)]),
                                                 list(pred[m & (spk[allidx] == t)]), labels)["accuracy"]
                                       for t in TEST}
        out["confusion"] = {"rows_gold_cols_pred": cl, "matrix": confusion_matrix(g, p, labels=cl).tolist()}
        # baseline: emotion2vec+ alone, on the labels it has
        e2v_map = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
                   "fearful": "anxious", "surprised": "surprised", "disgusted": "disgusted"}
        te = [rows[i] for i in allidx[m] if rows[i]["label"] in e2v_map.values()]
        bg = [r["label"] for r in te]
        bp = [e2v_map.get(max(r["emotion_scores"], key=r["emotion_scores"].get), "other") for r in te]
        hp = [pred[j] for j, i in enumerate(allidx) if m[j] and rows[i]["label"] in e2v_map.values()]
        out["on_emotion2vec_labels"] = {"labels": sorted(set(bg)), "emotion2vec_alone": scores(bg, bp, sorted(set(bg))),
                                        "this_head": scores(bg, hp, sorted(set(bg)))}
        # energy cues, per label (share of segments tagged)
        ar = np.array([feats[source][rows[i]["id"]][rows[i]["row"]][0] for i in allidx[m]])
        out["energy_tags"] = {lab: {"high": round(float(np.mean(ar[np.array(g) == lab] >= HIGH_ENERGY)), 2),
                                    "low": round(float(np.mean(ar[np.array(g) == lab] < LOW_ENERGY)), 2)}
                              for lab in labels}
        report[source] = out
        print(source, json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=1))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "expression.json").write_text(json.dumps(report, indent=1))

    # 3) ship: refit on all four speakers
    if save:
        tr = dur >= MIN_SEC
        save_head(expression.HEAD_FILE, fit(X[tr], y[tr], src[tr], C))
        print("saved", expression.HEAD_FILE.name, round(expression.HEAD_FILE.stat().st_size / 1e3), "kB")


if __name__ == "__main__":
    main(save="--no-save" not in sys.argv)
