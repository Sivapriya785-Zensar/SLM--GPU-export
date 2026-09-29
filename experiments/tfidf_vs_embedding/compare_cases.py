"""Qualitative side-by-side: TF-IDF (production) vs nomic-embed-text, on a handful
of hand-written narratives per network."""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(r"D:\dispute-resolver")
sys.path.insert(0, str(ROOT))
from ml.classifier import DisputeClassifier  # noqa: E402
from ml.features import build_text  # noqa: E402
from kb.kb import KnowledgeBase  # noqa: E402

HERE = Path(__file__).resolve().parent
NETWORKS = KnowledgeBase().networks()
HOST = "http://localhost:11434"
MODEL = "nomic-embed-text:latest"


def load(split):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{split}.jsonl", encoding="utf-8")]
    for r in rows:
        r["_text"] = build_text(r)
    emb = np.load(HERE / f"emb_{split}.npy")
    return rows, emb


def embed_batch(texts):
    payload = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(f"{HOST}/api/embed", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return np.array(json.load(r)["embeddings"], dtype=np.float32)


# retrain the embedding code-classifiers (fast - same as train_embed.py stage 2)
train, Xtr = load("train")
code_clfs = {}
for net in NETWORKS:
    idx = [i for i, r in enumerate(train) if r["scheme"] == net]
    clf = LogisticRegression(C=6.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr[idx], [train[i]["reason_code"] for i in idx])
    code_clfs[net] = clf

tfidf_clf = DisputeClassifier.load()

CASES = [
    ("Visa", "I was charged $89.99 by StreamPlus on my Visa card on Jan 12, but I "
             "cancelled my subscription on Dec 20 and have the cancellation confirmation email."),
    ("Visa", "My Visa card shows a $340 charge from an electronics store for a laptop "
             "that never arrived. Tracking shows it was never shipped."),
    ("Mastercard", "I paid $210 for a hotel booking on my Mastercard but never checked in "
                   "or cancelled - the hotel says it was a no-show and charged the full amount."),
    ("Mastercard", "There are two identical $56 charges from the same gas station on my "
                   "Mastercard statement on the same day, I only pumped gas once."),
    ("Amex", "My Amex was charged $1,200 for a laptop I returned three weeks ago; the "
             "merchant confirmed receipt of the return but hasn't issued a refund."),
    ("Amex", "I don't recognize a $45 charge on my Amex from a company I've never heard of, "
             "I still have my physical card in my possession."),
]

print(f"{'Network':<12}{'TF-IDF top1':<16}{'Embed top1':<16}{'Agree?':<8}")
for net, narrative in CASES:
    payload = {"dispute_narrative": narrative, "network": net}
    tfidf_pred = tfidf_clf.predict(payload, k=1)
    tfidf_code = tfidf_pred["predictions"][0]["code"]
    tfidf_prob = tfidf_pred["predictions"][0]["probability"]

    text = build_text(payload)
    emb = embed_batch([text])
    proba = code_clfs[net].predict_proba(emb)[0]
    classes = code_clfs[net].classes_
    best = proba.argsort()[-1]
    emb_code, emb_prob = classes[best], proba[best]

    agree = "yes" if tfidf_code == emb_code else "NO"
    print(f"{net:<12}{tfidf_code+' '+str(round(tfidf_prob,2)):<16}"
          f"{emb_code+' '+str(round(float(emb_prob),2)):<16}{agree:<8}")
    print(f"   narrative: {narrative[:90]}...")
