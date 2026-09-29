"""Train the same hierarchical network-router + per-network code classifier, but on
nomic-embed-text embeddings instead of TF-IDF, and compare against the existing
TF-IDF baseline (ml/models/report.json) on the identical val/test split."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(r"D:\dispute-resolver")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
from features import build_text  # noqa: E402
from kb.kb import KnowledgeBase  # noqa: E402

DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
NETWORKS = KnowledgeBase().networks()


def load(split):
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    for r in rows:
        r["_text"] = build_text(r)
    emb = np.load(HERE / f"emb_{split}.npy")
    assert len(emb) == len(rows), f"{split}: {len(emb)} embeddings vs {len(rows)} rows"
    return rows, emb


def topk_accuracy(clf, X, y, k=3):
    proba = clf.predict_proba(X)
    classes = clf.classes_
    hits = 0
    for probs, gold in zip(proba, y):
        top = {classes[i] for i in probs.argsort()[-k:]}
        hits += gold in top
    return hits / len(y)


def main():
    train, Xtr = load("train")
    val, Xval = load("val")
    test, Xte = load("test")
    print(f"[data] train={len(train)} val={len(val)} test={len(test)}  dim={Xtr.shape[1]}")

    report = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "feature": "nomic-embed-text", "stages": {}}

    # ---- Stage 1: network router ----
    t = time.time()
    net_clf = LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
    net_clf.fit(Xtr, [r["scheme"] for r in train])
    for name, ds, X in (("val", val, Xval), ("test", test, Xte)):
        yt = [r["scheme"] for r in ds]
        acc = accuracy_score(yt, net_clf.predict(X))
        report["stages"].setdefault("network", {})[name + "_accuracy"] = round(acc, 4)
        print(f"[network] {name} accuracy = {acc:.4f}")
    print(f"[network] trained in {time.time()-t:.1f}s")

    # ---- Stage 2: per-network code classifiers ----
    code_clfs = {}
    idx = {"train": np.arange(len(train)), "val": np.arange(len(val)), "test": np.arange(len(test))}
    for net in NETWORKS:
        t = time.time()
        tr_mask = [i for i, r in enumerate(train) if r["scheme"] == net]
        tr = [train[i] for i in tr_mask]
        Xtr_net = Xtr[tr_mask]
        n_codes = len(Counter(r["reason_code"] for r in tr))
        clf = LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr_net, [r["reason_code"] for r in tr])
        code_clfs[net] = clf

        st = report["stages"].setdefault(f"code_{net}", {"n_codes": n_codes, "n_train": len(tr)})
        for name, ds, X in (("val", val, Xval), ("test", test, Xte)):
            mask = [i for i, r in enumerate(ds) if r["scheme"] == net]
            sub = [ds[i] for i in mask]
            Xsub = X[mask]
            yt = [r["reason_code"] for r in sub]
            pred = clf.predict(Xsub)
            st[f"{name}_accuracy"] = round(accuracy_score(yt, pred), 4)
            st[f"{name}_top3_accuracy"] = round(topk_accuracy(clf, Xsub, yt, 3), 4)
            st[f"{name}_macro_f1"] = round(f1_score(yt, pred, average="macro"), 4)
        print(f"[code_{net}] {n_codes} codes | val acc={st['val_accuracy']:.3f} "
              f"top3={st['val_top3_accuracy']:.3f} | test acc={st['test_accuracy']:.3f} "
              f"top3={st['test_top3_accuracy']:.3f} macroF1={st['test_macro_f1']:.3f} "
              f"| {time.time()-t:.1f}s")

    # ---- end-to-end on test ----
    correct = correct_given_net = 0
    for i, r in enumerate(test):
        pred_net = net_clf.predict(Xte[i:i+1])[0]
        pred_code = code_clfs[pred_net].predict(Xte[i:i+1])[0]
        correct += pred_code == r["reason_code"]
        correct_given_net += code_clfs[r["scheme"]].predict(Xte[i:i+1])[0] == r["reason_code"]
    report["end_to_end"] = {
        "test_accuracy_full_pipeline": round(correct / len(test), 4),
        "test_accuracy_given_true_network": round(correct_given_net / len(test), 4),
    }
    print(f"\n[end-to-end] full pipeline test accuracy = {correct/len(test):.4f}")
    print(f"[end-to-end] with correct network given    = {correct_given_net/len(test):.4f}")

    (HERE / "report_embed.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] report_embed.json written")


if __name__ == "__main__":
    main()
