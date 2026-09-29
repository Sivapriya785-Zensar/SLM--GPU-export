# Experiment: semantic embeddings vs. TF-IDF for the ML candidate classifier

**Question:** would swapping the ML classifier's TF-IDF features for a semantic
embedding model (to give the SLM a better prior / boost end-to-end accuracy) help?

**Answer: no — TF-IDF wins on every metric. Not adopted. Production code
(`ml/train.py`, `ml/classifier.py`, `ml/models/*.joblib`) was never touched.**

## Setup

- Embedding model: `nomic-embed-text:latest` (768-dim), run locally via Ollama —
  no new dependency, no internet call, same offline posture as the rest of the stack.
- Same hierarchical architecture as production (network router → per-network code
  classifier), same Logistic Regression settings, same train/val/test split
  (7,268 / 803 / 889 rows) — only the feature extractor changed (embeddings instead
  of TF-IDF word 1-2gram + char 3-5gram).

## Quantitative result (held-out test set)

| Stage | TF-IDF (production) | nomic-embed-text | Δ |
|---|---|---|---|
| Network router | 1.000 | 0.999 | −0.001 |
| Visa (21 codes) | 0.943 | 0.929 | −1.4 pts |
| Mastercard (20 codes) | 0.945 | 0.928 | −1.7 pts |
| Amex (24 codes) | 0.937 | 0.917 | −2.0 pts |
| **End-to-end** | **0.942** | **0.924** | **−1.8 pts** |

Consistently worse across every network and the router — not noise.

## Qualitative result (6 hand-written cases, 2 per network)

5/6 agreed between the two models. The one disagreement, on an Amex fraud case
("don't recognize a $45 charge... still have my card"), is informative: TF-IDF
picked `AM-F30` (EMV Counterfeit — implies a cloned card used in person), while
the embedding model picked the semantically correct `AM-F24` (No Card Member
Authorization — unrecognized/remote charge, card still in hand). So embeddings
aren't useless — they just don't win often enough in aggregate to close the gap.

## Why TF-IDF wins here

`ml/features.py`'s `build_text()` isn't pure prose — it appends discrete
`field=value` tokens (`channel=ecommerce`, `amt_25_100`, etc.) alongside the
narrative. TF-IDF's char n-grams are exact-match-friendly and pick up both these
structured tokens and this dataset's fairly templated synthetic phrasing very
precisely. A general-purpose semantic embedding model is tuned for
paraphrase/meaning similarity, which blurs exactly the fine, code-distinguishing
lexical cues this task depends on — consistent with the separately-documented
finding that the pipeline's main weakness is fine-grained disambiguation between
neighboring/confusable codes; a generic embedding model doesn't fix that, and
mildly worsens it in aggregate.

## Reproducing this

```bash
python embed_data.py     # embeds data/{train,val,test}.jsonl via Ollama, writes emb_*.npy (not committed - regenerate locally)
python train_embed.py    # trains + evaluates the embedding-based classifier, writes report_embed.json
python compare_cases.py  # side-by-side qualitative comparison against the production TF-IDF classifier
```

`emb_*.npy` caches are git-ignored (regenerable, and ~27MB combined) — rerun
`embed_data.py` to reproduce them locally before `train_embed.py` /
`compare_cases.py`.

## If revisited later

A domain-fine-tuned or larger embedding model is the more promising direction
than swapping the classifier family outright on a generic embedding — this was
a quick, honest test of the simplest version of the idea, not an exhaustive one.
