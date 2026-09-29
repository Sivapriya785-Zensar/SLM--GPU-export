"""Embed train/val/test dispute texts via Ollama's nomic-embed-text, cache to .npy."""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\dispute-resolver")
sys.path.insert(0, str(ROOT / "ml"))
from features import build_text  # noqa: E402

DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
HOST = "http://localhost:11434"
MODEL = "nomic-embed-text:latest"
BATCH = 64


def load(split):
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    for r in rows:
        r["_text"] = build_text(r)
    return rows


def embed_batch(texts):
    payload = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(f"{HOST}/api/embed", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["embeddings"]


def embed_all(rows, name):
    vecs = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = [r["_text"] for r in rows[i:i + BATCH]]
        vecs.extend(embed_batch(chunk))
        print(f"[{name}] {min(i+BATCH, len(rows))}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    arr = np.array(vecs, dtype=np.float32)
    np.save(OUT / f"emb_{name}.npy", arr)
    return arr


def main():
    for split in ("train", "val", "test"):
        rows = load(split)
        print(f"embedding {split}: {len(rows)} rows")
        embed_all(rows, split)
    print("done")


if __name__ == "__main__":
    main()
