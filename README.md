# Dispute Assistant

An **SLM-first, classification-only** dispute assistant. The fine-tuned `dispute-phi3`
small language model classifies a card dispute into the correct reason code and holds
a **multi-turn conversation** about that classification. It does **not** give resolution
steps, evidence checklists, or any other advice — it classifies, nothing else. A
lightweight ML model supplies candidate codes and a curated knowledge base grounds the
SLM's *prompt only*; neither is consulted again after the SLM answers. The card networks
and their reason codes are not hardcoded anywhere — every component (training, serving,
API, UI) reads them from `data/reason_code_kb.json`, the single source of truth.

---

## Architecture (SLM primary, ML + KB support — KB used only *before* the SLM)

```
 narrative + REQUIRED network (chosen by the user, never auto-detected)
        │
        ▼
 ┌──────────────────────┐   SUPPORT 1 — ML candidate model (classification only)
 │ code_<network>.joblib│   TF-IDF → LogisticRegression, scoped to that network's codes
 └──────────────────────┘   → ranked candidates + probabilities
                                e.g. [VISA-13.2 .40, VISA-13.7 .29, VISA-13.6 .11]
        │
        ▼
 ┌──────────────────────┐   SUPPORT 2 — knowledge base (prompt grounding ONLY)
 │ reason_code_kb.json  │   pull the KB entry for each candidate, injected into the
 └──────────────────────┘   SLM's system prompt (definition + "does not apply when")
        │
        ▼
 ┌──────────────────────┐   PRIMARY, AND FINAL — dispute-phi3 SLM (via Ollama)
 │ slm/client.py        │   sees narrative + candidates + KB context,
 │ /api/classify         │  JSON-schema-constrained to the network's codes
 └──────────────────────┘   → returns reason_code + its OWN reason_label/category +
                               a short explanation. Nothing is looked up in the KB
                               again — no resolution steps, no evidence, no advice.
        │
        ▼
   FastAPI  →  chat UI (single message thread, "New chat" to reset)
        │
        ▼
 ┌──────────────────────┐   MULTI-TURN — /api/chat
 │ conversation store   │   full history (system + KB-grounded prompt + turns) replayed
 │ slm.follow_up()      │   to the SLM; free-text answers, stays consistent with
 └──────────────────────┘   its classification unless given new facts. Still no KB.
```

- **Network is required, not guessed.** The UI's network dropdown has no
  "auto-detect" option — it's populated live from `GET /api/networks` and the user
  must pick one before the first message can be sent. The ML router model
  (`network.joblib`) still exists and the API will fall back to it if a caller omits
  `network`, but the shipped UI never exercises that path.
- **Fallback**: if the SLM call fails twice, `/api/classify` returns the ML classifier's
  top candidate as the code (with `reason_label`/`category` left `null` — there is no
  KB fallback for those either) and `slm_used: false`; `/api/chat` returns 503 if Ollama
  is unreachable.
- Why keep the ML model at all: it gives the SLM a fast, well-calibrated prior (top-3
  ≈ 0.97 on the held-out set) and a graceful degradation path — it does not itself
  decide the final answer, and it plays no role after classification either.
- `GET /api/code/{code}` and `GET /api/networks` still read the full KB (definitions,
  resolution steps, etc.) — they're standalone lookup endpoints for browsing the code
  catalogue, not part of the `/api/classify` pipeline, and the shipped UI doesn't call
  them for classification results.
- **Nothing hardcoded**: network names / reason codes come from the KB
  (`KnowledgeBase.networks()`, read by `ml/train.py`, `ml/classifier.py`,
  `app/main.py`, and fetched live by the UI); the SLM model name, Ollama host, and
  generation settings are env vars with defaults (`slm/client.py`); the KB file path
  itself is overridable (`DISPUTE_KB_PATH`).

---

## Project layout

| Path | What |
|---|---|
| `data/train.jsonl` `val.jsonl` `test.jsonl` | labelled disputes (narrative + metadata + reason code) |
| `data/reason_code_kb.json` | the knowledge base — 65 codes, resolution steps included |
| `ml/features.py` | shared text builder (train + serve use the identical representation) |
| `ml/train.py` | trains the 4 models, writes `ml/models/*.joblib` + `report.json` |
| `ml/classifier.py` | `DisputeClassifier.load().predict(payload)` — candidate model |
| `kb/kb.py` | `KnowledgeBase` loader / lookup |
| `slm/client.py` | Ollama client for `dispute-phi3` — `classify()` + `follow_up()` |
| `app/main.py` | FastAPI service (`/api/classify`, `/api/chat`, …) + static UI |
| `app/static/` | frontend (`index.html`, `styles.css`, `app.js`) |
| `eval/confusions.py` | support-model per-network accuracy + confused pairs |
| `eval/queries.py` | support-model accuracy on terse plain-English queries |

Env vars (all optional, sensible defaults): `SLM_MODEL`, `OLLAMA_HOST`,
`SLM_TEMPERATURE`, `SLM_NUM_CTX`, `SLM_NUM_PREDICT`, `SLM_TIMEOUT_SECONDS`,
`DISPUTE_CHAT_HISTORY_MSGS`, `DISPUTE_KB_PATH` — see `slm/client.py` / `app/main.py`.

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # optional
pip install -r requirements.txt
```

## Train (≈45 s, CPU)

```bash
python ml/train.py
```

Writes `ml/models/network.joblib`, `code_Visa.joblib`, `code_Mastercard.joblib`,
`code_Amex.joblib` and `ml/models/report.json`.

## Run the app

**Windows, one click:** double-click **`run.bat`** — it starts the Ollama server,
installs deps and trains the models if needed, starts the FastAPI backend (which also
serves the frontend) on port 8010, and opens the browser. `stop.bat` shuts it down.

**Manual:**

```bash
python -m uvicorn app.main:app --port 8010
```

Open <http://localhost:8010>. (The frontend is static files served by the same
backend — there is no separate frontend server.)

## API

| Method | Path | Body / notes |
|---|---|---|
| `POST` | `/api/classify` | `{ "narrative": "...", "network": "Amex", "channel": "..."?, ... }` — `network` is **required**, must be one of `GET /api/networks` → classification only, no advice |
| `POST` | `/api/chat` | `{ "conversation_id": "...", "message": "..." }` → SLM free-text follow-up about the classification |
| `GET`  | `/api/code/{code}` | full KB entry for one reason code (standalone lookup, not used by `/api/classify`) |
| `GET`  | `/api/networks` | reason-code catalogue grouped by network (standalone lookup) |
| `GET`  | `/api/health` | SLM availability + KB / classifier status |

`/api/classify` response: `reason_code`, `reason_label` and `category` (straight from
the SLM's own JSON output — `null`/`null` if it fell back to ML-only), `slm_used`,
`slm_explanation`, `classifier_candidates` (the support model's ranked `{code,
probability}` list), and `conversation_id` for the follow-up chat. No resolution
steps, evidence, or advice fields — the KB is not queried after classification.

---

## Support-model test metrics (`ml/models/report.json`)

The ML candidate model — the SLM's prior — trained **only on `data/train.jsonl`**:

| stage | test accuracy | test top-3 |
|---|---|---|
| network router | 1.000 | — |
| Visa codes (21) | 0.943 | 0.970 |
| Mastercard codes (20) | 0.945 | 0.966 |
| Amex codes (24) | 0.937 | 0.957 |
| **end-to-end** | **0.942** | — |

On terse one-line queries (`eval/queries.py`) it scores ≈ **0.82 top-1 / 0.91 top-3**.
The SLM makes the final call, using this ranked list plus the KB definitions as prompt
context — the ML model and KB are never consulted again after that.

---

## Latest regression run (`eval/regression_test.py`)

17 classification queries (the original 14-query set + 3 previously-flagged "struggle"
cases) + a 2-turn follow-up, run against the live API:

- **classification: 16/17 (0.94)**. The one consistent miss: a Mastercard hotel
  no-show narrative gets `MA-4855` (Goods Not Provided) instead of `MA-4859`
  (Addendum/No-Show) — the SLM overrides a correct ML top-candidate here.
- **multi-turn bug found and fixed**: a follow-up turn could come back as raw JSON
  instead of prose (the stored first-turn assistant message was JSON, which primed the
  SLM to keep answering in JSON). Fixed by storing that turn as a natural-language
  sentence, plus a defensive reformatter in `slm/client.py` (`_as_prose`) in case a
  reply still slips into JSON. Retested clean on the same repro case.

## Notes / limitations

- **Ollama must be running** with the `dispute-phi3-4ep` model pulled. `run.bat` starts
  it; otherwise `ollama serve` + `ollama pull dispute-phi3-4ep`.
- The SLM runs locally on CPU — a classification or follow-up turn takes **~10–20 s**.
  The UI shows a typing indicator.
- The SLM's free-text `explanation` can still contain templated filler (e.g. a "Filed N
  days after…" clause). `reason_label`/`category` now come from the SLM's own JSON
  output, not a KB lookup — they're `null` if it falls back to the ML classifier only.
- **This service classifies; it does not advise.** No resolution steps, required
  evidence, merchant-rebuttal notes, filing windows, or "does this code apply" guidance
  are returned. That content still exists in `data/reason_code_kb.json` and is used to
  *ground the SLM's prompt*, and it's still browsable via `GET /api/code/{code}`, but
  `/api/classify` never surfaces it in the response.
- Conversations are held **in memory** — they are lost on server restart.
- Training data is synthetic-style (`label_quality: synthetic_reviewed_high`). Re-train
  / re-tune on real labelled disputes before production use.
- `time_limit_days` in the KB is a representative filing window — always confirm against
  the **current** network rulebook and any regional/domestic overrides before filing.
