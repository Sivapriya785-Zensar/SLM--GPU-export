"""
Train the hierarchical dispute reason-code classifier.

  Stage 1  network router      : text -> network (whichever networks the KB declares)
  Stage 2  per-network code    : text -> reason_code   (one model per network)

Both stages are word + char TF-IDF -> Logistic Regression; probabilities are used at
serve time to return a ranked top-k. Classic ML on purpose: trains in seconds on CPU,
no GPU, deterministic, easy to serve.

Trained purely on the labelled data in data/train.jsonl - no synthetic augmentation,
no seed phrases, no post-processing. The knowledge base is a separate lookup layer
used only to attach resolution guidance to whatever code the classifier predicts.

Usage:
  python ml/train.py                 # train on data/train.jsonl, evaluate on val + test
  python ml/train.py --quick         # smaller grid, faster
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from features import build_text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kb.kb import KnowledgeBase  # noqa: E402 — single source of truth for the network list

DATA = ROOT / "data"
MODELS = ROOT / "ml" / "models"
NETWORKS = KnowledgeBase().networks()


def load(split):
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    for r in rows:
        r["_text"] = build_text(r)
    return rows


def make_pipeline(quick=False):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, max_features=None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, max_features=200_000)
    feats = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(
        C=6.0 if not quick else 3.0, max_iter=2000, class_weight="balanced",
        solver="lbfgs", n_jobs=-1,
    )
    return Pipeline([("feats", feats), ("clf", clf)])


def topk_accuracy(pipe, X, y, k=3):
    proba = pipe.predict_proba(X)
    classes = pipe.classes_
    hits = 0
    for probs, gold in zip(proba, y):
        top = {classes[i] for i in probs.argsort()[-k:]}
        hits += gold in top
    return hits / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    train, val, test = load("train"), load("val"), load("test")
    print(f"[data] train={len(train)} val={len(val)} test={len(test)}")

    report = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}

    # ---- Stage 1: network router ---------------------------------------------
    t = time.time()
    net_pipe = make_pipeline(args.quick)
    net_pipe.fit([r["_text"] for r in train], [r["scheme"] for r in train])
    joblib.dump(net_pipe, MODELS / "network.joblib")
    for name, ds in (("val", val), ("test", test)):
        X = [r["_text"] for r in ds]
        yt = [r["scheme"] for r in ds]
        acc = accuracy_score(yt, net_pipe.predict(X))
        report["stages"].setdefault("network", {})[name + "_accuracy"] = round(acc, 4)
        print(f"[network] {name} accuracy = {acc:.4f}")
    print(f"[network] trained in {time.time()-t:.1f}s")

    # ---- Stage 2: per-network code classifiers -----------------------------
    for net in NETWORKS:
        t = time.time()
        tr = [r for r in train if r["scheme"] == net]
        n_codes = len(Counter(r["reason_code"] for r in tr))
        pipe = make_pipeline(args.quick)
        pipe.fit([r["_text"] for r in tr], [r["reason_code"] for r in tr])
        joblib.dump(pipe, MODELS / f"code_{net}.joblib")

        st = report["stages"].setdefault(f"code_{net}", {"n_codes": n_codes, "n_train": len(tr)})
        for name, ds in (("val", val), ("test", test)):
            sub = [r for r in ds if r["scheme"] == net]
            X = [r["_text"] for r in sub]
            yt = [r["reason_code"] for r in sub]
            pred = pipe.predict(X)
            st[f"{name}_accuracy"] = round(accuracy_score(yt, pred), 4)
            st[f"{name}_top3_accuracy"] = round(topk_accuracy(pipe, X, yt, 3), 4)
            st[f"{name}_macro_f1"] = round(f1_score(yt, pred, average="macro"), 4)
        print(f"[code_{net}] {n_codes} codes | val acc={st['val_accuracy']:.3f} "
              f"top3={st['val_top3_accuracy']:.3f} | test acc={st['test_accuracy']:.3f} "
              f"top3={st['test_top3_accuracy']:.3f} macroF1={st['test_macro_f1']:.3f} "
              f"| {time.time()-t:.1f}s")

    # ---- end-to-end (router + code) on test --------------------------------
    code_pipes = {n: joblib.load(MODELS / f"code_{n}.joblib") for n in NETWORKS}
    correct = correct_given_net = 0
    for r in test:
        pred_net = net_pipe.predict([r["_text"]])[0]
        pred_code = code_pipes[pred_net].predict([r["_text"]])[0]
        correct += pred_code == r["reason_code"]
        correct_given_net += code_pipes[r["scheme"]].predict([r["_text"]])[0] == r["reason_code"]
    report["end_to_end"] = {
        "test_accuracy_full_pipeline": round(correct / len(test), 4),
        "test_accuracy_given_true_network": round(correct_given_net / len(test), 4),
    }
    print(f"\n[end-to-end] full pipeline test accuracy      = {correct/len(test):.4f}")
    print(f"[end-to-end] with correct network given       = {correct_given_net/len(test):.4f}")

    (MODELS / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] models + report.json written to {MODELS}")


if __name__ == "__main__":
    main()
