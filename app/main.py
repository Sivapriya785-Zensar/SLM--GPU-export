"""
Dispute reason-code classifier service (classification only - no resolution advice).

Pipeline (SLM-primary):
  narrative --> ML classifier  -> ranked candidate reason codes  (SUPPORT: a prior)
            --> KB             -> definitions for those candidates (SUPPORT: grounding,
                                   injected into the SLM's prompt ONLY - never read again
                                   after the SLM answers)
            --> SLM            -> chooses the reason code, returns its own label/category
                                   + a short explanation (PRIMARY, and final)
  There is deliberately no KB lookup after the SLM responds: the service classifies a
  dispute into a reason code, it does not offer resolution steps or advice. reason_label
  and category in the response come straight from the SLM's own JSON output.
  Follow-up turns are answered by the SLM with the conversation history (also no KB).

Run:  uvicorn app.main:app --port 8010
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.classifier import DisputeClassifier  # noqa: E402
from kb.kb import KnowledgeBase  # noqa: E402
from slm import client as slm  # noqa: E402

app = FastAPI(title="Dispute Reason-Code Resolver", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CLASSIFIER: DisputeClassifier | None = None
KB: KnowledgeBase | None = None
CONVERSATIONS: dict[str, dict] = {}
MAX_HISTORY_MSGS = int(os.environ.get("DISPUTE_CHAT_HISTORY_MSGS", "10"))  # system + last N turns

NET_CODES: dict[str, list[str]] = {}  # network -> its codes; filled at startup from the KB


@app.on_event("startup")
def _load():
    global CLASSIFIER, KB
    CLASSIFIER = DisputeClassifier.load()
    KB = KnowledgeBase()
    for net in KB.networks():
        NET_CODES[net] = [e["code"] for e in KB.codes_for_network(net)]


# --------------------------------------------------------------------------- models
class ClassifyRequest(BaseModel):
    narrative: str = Field(..., min_length=10)
    network: str = Field(..., description="Must be one of the networks the KB declares — "
                                           "given by the caller, never guessed.")
    channel: str | None = None
    merchant_vertical: str | None = None
    currency: str | None = None
    transaction_amount: float | None = None
    days_since_transaction: int | None = None
    dispute_value_band: str | None = None
    chargeback_cycle: str | None = None
    merchant_country: str | None = None
    cardholder_region: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(..., min_length=1)


class SettingsRequest(BaseModel):
    model: str = Field(..., description="Must be one of the models GET /api/models lists "
                                         "as installed in Ollama.")


# --------------------------------------------------------------------------- helpers
META_LINE_FIELDS = [
    ("transaction_amount", "Transaction amount"),
    ("currency", "Currency"),
    ("channel", "Channel"),
    ("merchant_vertical", "Merchant vertical"),
    ("days_since_transaction", "Days since transaction"),
    ("dispute_value_band", "Dispute value band"),
    ("chargeback_cycle", "Chargeback cycle"),
    ("merchant_country", "Merchant country"),
    ("cardholder_region", "Cardholder region"),
]


# --- intake gate ------------------------------------------------------------------
# The SLM's output is JSON-schema-constrained to always contain a reason_code from
# the closed set — it structurally cannot say "this isn't a dispute", it can only
# ever pick a code. So off-topic input ("how's the weather?") must be screened out
# BEFORE the classify pipeline runs at all, not left for the SLM to refuse (it can't).
# Deliberately a cheap heuristic, not a model call — keeps this fast and avoids
# spending a generation on a decision that doesn't need one.
_CASE_KEYWORDS = (
    # stems, not whole words, so "charging"/"charged"/"cancelled"/"membership" etc.
    # all match via substring — an exact-word list missed real dispute phrasing like
    # "kept charging me... cancelled the membership" (charge != charging as a substring)
    "transaction", "charg", "dispute", "refund", "card", "merchant", "claim",
    "purchas", "chargeback", "payment", "order", "bill", "authoriz", "cancel",
    "subscri", "member", "recurring", "fraud", "unauthoris", "unauthoriz",
    "deliver", "receiv", "stolen", "declin", "amount", "approv", "terminal",
    "goods", "servic", "duplicat", "incorrect", "wrong", "bank", "statement",
)
_AMOUNT_RE = re.compile(r"\b\d+(\.\d{1,2})?\s*(usd|gbp|cad|aud|eur|sgd|inr|\$|£|€)\b", re.IGNORECASE)


def _looks_like_dispute(text: str) -> bool:
    if re.search(r"^-\s*\w[\w /]*:", text, re.MULTILINE):  # "- Field: value" metadata line
        return True
    if _AMOUNT_RE.search(text):
        return True
    # >=1, not >=2: real (if terse) dispute openers like "I don't recognize this
    # transaction" or "the amount I approved on the terminal" only hit 0-1 keywords
    # in testing. False-blocking a genuine dispute is worse than being permissive -
    # this only needs to catch input with literally zero dispute-domain signal.
    keyword_hits = sum(1 for k in _CASE_KEYWORDS if k in text.lower())
    return keyword_hits >= 1


def _kb_reference(codes: list[str]) -> str:
    lines = []
    for c in codes:
        e = KB.get(c)
        if not e:
            continue
        awn = "; ".join(e.get("applies_when", [])[:3])
        dna = "; ".join(e.get("does_not_apply_when", [])[:3])
        lines.append(
            f"- {c} ({e['label']}, {e['category']}): {e.get('summary', '')} "
            f"Applies when: {awn} | Does NOT apply when: {dna}"
        )
    return "\n".join(lines)


def _system_prompt(network: str, candidates: list[str]) -> str:
    codes = ", ".join(NET_CODES[network])
    return (
        "You are a dispute reason-code classification assistant for card chargebacks. "
        f"This dispute is on the {network} network. Choose exactly one reason code from "
        f"this closed set (no other codes exist):\n{codes}\n\n"
        "A statistical classifier has already ranked the most likely codes for this case "
        f"(most likely first): {', '.join(candidates)}. Treat that as a strong prior — "
        "usually the first candidate is correct — but override it if the narrative clearly "
        "points to a different code in the set.\n\n"
        "Reference — knowledge base entries for the candidate codes:\n"
        f"{_kb_reference(candidates)}\n\n"
        "Disambiguation rules:\n"
        "- Match the code to what actually happened to the goods, service, or charge. Do "
        "NOT pick an authorization-related code (approval/decline/amount-approved codes) "
        "for a dispute about the condition, delivery, or description of goods/services — "
        "those are goods/services dispute codes, not authorization codes.\n"
        "- If the case does not state the purchase channel (in-person vs online/phone/mail) "
        "and the narrative doesn't describe an in-person interaction (swiping a card, being "
        "at a store, chip-and-PIN), treat it as card-not-present (online) fraud rather than "
        "a card-present EMV/chip liability code.\n\n"
        'Respond to the classification request with exactly one JSON object:\n'
        '{"reason_code": "<code>", "reason_label": "<label>", "category": "<category>", '
        '"explanation": "<2-3 sentences citing the specific facts in THIS case that drove '
        'the choice>"}\n'
        "In later turns, answer in natural language and stay consistent with your "
        "classification unless the user gives genuinely new facts."
    )


def _user_message(req: ClassifyRequest) -> str:
    lines = [req.narrative.strip(), ""]
    for key, label in META_LINE_FIELDS:
        v = getattr(req, key)
        if v is not None and str(v).strip():
            lines.append(f"- {label}: {v}")
    lines.append("")
    lines.append("Classify this dispute.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "kb_codes": len(KB.all_codes()) if KB else 0,
        "slm_model": slm.get_model(),
        "slm_available": slm.available(),
        "classifier_report": CLASSIFIER.report.get("end_to_end") if CLASSIFIER else None,
        "active_conversations": len(CONVERSATIONS),
    }


@app.get("/api/models")
def list_models():
    """Every model installed in Ollama, plus which one is currently active - the
    Settings panel's model picker reads this. Architecture is unaffected either way:
    whichever model is active still plays the same PRIMARY role, still receives the
    same ML-candidates + KB-grounded prompt, still returns the same schema."""
    try:
        installed = slm.list_models()
    except Exception as exc:
        raise HTTPException(503, f"Could not reach Ollama to list models: {exc}")
    return {"active": slm.get_model(), "models": installed}


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    try:
        installed = {m["name"] for m in slm.list_models()}
    except Exception as exc:
        raise HTTPException(503, f"Could not reach Ollama to verify the model: {exc}")
    if req.model not in installed:
        raise HTTPException(400, f"{req.model!r} is not installed in Ollama. "
                                  f"Available: {', '.join(sorted(installed)) or '(none found)'}")
    slm.set_model(req.model)
    return {"active": slm.get_model()}


@app.get("/api/networks")
def networks():
    out = {}
    for net in KB.networks():
        out[net] = sorted(
            ({"code": e["code"], "label": e["label"], "category": e["category"]}
             for e in KB.codes_for_network(net)),
            key=lambda x: x["code"],
        )
    return out


@app.get("/api/code/{code}")
def code_detail(code: str):
    e = KB.get(code)
    if not e:
        raise HTTPException(404, f"Unknown reason code: {code}")
    return e


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    if req.network not in NET_CODES:
        raise HTTPException(400, f"Unknown network {req.network!r}. Choose one of: "
                                  f"{', '.join(NET_CODES)}")

    if not _looks_like_dispute(req.narrative):
        return {
            "conversation_id": None,
            "network": req.network,
            "network_source": "user",
            "not_a_dispute": True,
            "reason_code": None,
            "reason_label": None,
            "category": None,
            "slm_used": False,
            "slm_explanation": "This doesn't look like a card dispute. Describe what "
                                "happened - the transaction, merchant, amount, and what "
                                "you're disputing - so it can be classified into a reason code.",
            "slm_error": None,
            "classifier_candidates": [],
            "disclaimer": "Classification only (SLM-primary, ML-supported) — no resolution "
                          "guidance is provided. Confirm against the current network rulebook.",
        }

    payload = req.model_dump()
    payload["dispute_narrative"] = payload.pop("narrative")

    # 1. ML classifier — ranked candidates for the given network (support signal;
    #    the network itself is always supplied by the caller, never auto-detected here)
    #    k=5, not 3: the correct code was occasionally ranked #4/#5 by the ML model and
    #    so never reached the SLM's context at k=3.
    ml_out = CLASSIFIER.predict(payload, k=5)
    network = ml_out["network"]
    candidates = [p["code"] for p in ml_out["predictions"]]
    cand_probs = {p["code"]: p["probability"] for p in ml_out["predictions"]}

    # 2. + 3. SLM decides, grounded in the KB entries for those candidates.
    # No separate slm.available() pre-check: that ping has its own short timeout and
    # was going false-negative whenever Ollama was mid-generation on the previous call
    # (still up, just briefly slow to answer an unrelated endpoint) — those false
    # negatives showed up as needless ML-only fallbacks. Attempt the real call directly,
    # with one retry, and only fall back if it genuinely fails twice.
    system = _system_prompt(network, candidates)
    user_msg = _user_message(req)
    slm_used, slm_explanation, slm_error = False, "", None
    chosen = candidates[0]
    chosen_label, chosen_category = None, None
    # 3 attempts, not 2: a truncated/repaired response (see slm.classify) already
    # recovers most failures, so a genuine "_error" here means the model produced
    # nothing usable at all — worth one more try before accepting the ML-only fallback.
    for attempt in range(3):
        try:
            res = slm.classify(system, user_msg, NET_CODES[network])
            if "_error" in res:
                slm_error = "slm returned non-JSON"
                continue
            cc = res.get("reason_code")
            if cc in NET_CODES[network]:
                chosen = cc
            chosen_label = (res.get("reason_label") or None)
            chosen_category = (res.get("category") or None)
            slm_explanation = (res.get("explanation") or "").strip()
            slm_used = True
            slm_error = None
            break
        except Exception as exc:  # timeout / connection
            slm_error = f"{type(exc).__name__}: {exc}"

    # No KB lookup here on purpose: the service classifies, it doesn't advise. Label and
    # category come straight from the SLM's own JSON (or are None on an ML-only fallback,
    # where only the bare code is known).
    #
    # But an empty answer is a bug, not "classification only" — if the SLM never
    # produced a usable response, say so plainly instead of leaving the explanation
    # blank. This is a transparency note about how the answer was reached, not advice.
    if not slm_used:
        top_prob = cand_probs.get(chosen)
        conf_bit = f" (confidence {round(top_prob * 100)}%)" if top_prob is not None else ""
        slm_explanation = (
            f"The classification model (SLM) did not return a usable response for this "
            f"request{' - ' + slm_error if slm_error else ''}, so this result comes from "
            f"the statistical candidate model alone{conf_bit}. Try again, or treat this as "
            f"lower-confidence than a normal SLM-classified answer."
        )

    # conversation state for follow-up turns. The stored assistant turn is prose,
    # not the raw JSON the classify call used — seeing JSON as its own prior turn
    # measurably nudges the SLM to keep answering in JSON on follow-ups instead of
    # the natural language the system prompt asks for.
    conv_id = uuid.uuid4().hex[:12]
    label_bit = f" ({chosen_label}, {chosen_category})" if chosen_label else ""
    assistant_summary = f"{chosen}{label_bit}. {slm_explanation}".strip()
    CONVERSATIONS[conv_id] = {
        "network": network,
        "code": chosen,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_summary},
        ],
    }

    return {
        "conversation_id": conv_id,
        "network": network,
        "network_source": ml_out["network_source"],
        "reason_code": chosen,
        "reason_label": chosen_label,
        "category": chosen_category,
        "slm_used": slm_used,
        "slm_explanation": slm_explanation,
        "slm_error": slm_error,
        "classifier_candidates": [
            {"code": c, "probability": cand_probs.get(c)} for c in candidates
        ],
        "disclaimer": "Classification only (SLM-primary, ML-supported) — no resolution "
                      "guidance is provided. Confirm against the current network rulebook.",
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    conv = CONVERSATIONS.get(req.conversation_id)
    if conv is None:
        raise HTTPException(404, "Unknown or expired conversation_id — run /api/classify first.")
    if not slm.available():
        raise HTTPException(503, "The SLM (Ollama) is not reachable, so follow-up chat is unavailable.")

    conv["messages"].append({"role": "user", "content": req.message.strip()})

    # keep system + the most recent turns within context
    msgs = conv["messages"]
    trimmed = [msgs[0]] + msgs[1:][-(MAX_HISTORY_MSGS - 1):] if len(msgs) > MAX_HISTORY_MSGS else msgs
    try:
        reply = slm.follow_up(trimmed)
    except Exception as exc:
        conv["messages"].pop()  # roll back the unanswered user turn
        raise HTTPException(504, f"SLM did not respond: {type(exc).__name__}")

    conv["messages"].append({"role": "assistant", "content": reply})
    return {"conversation_id": req.conversation_id, "reply": reply,
            "reason_code": conv["code"], "network": conv["network"]}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Browsers probe /favicon.ico directly regardless of the <link rel="icon"> data-URI
    # in index.html; answer with "no icon here" (204) instead of a noisy 404.
    from fastapi import Response
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe():
    # Chrome DevTools auto-probes this path on every site looking for a workspace
    # config; unrelated to this app. Harmless either way, quieted for a clean log.
    from fastapi import Response
    return Response(status_code=204)


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
