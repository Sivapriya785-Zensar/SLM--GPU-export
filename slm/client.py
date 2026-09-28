"""
Ollama client for the dispute SLM (dispute-phi3-4ep).

The SLM is the primary decision-maker: it reads the narrative, the ML classifier's
ranked candidates, and the KB entries for those candidates, then chooses the reason
code and explains it. It also handles all follow-up conversation turns.

Every model/endpoint/generation setting below has a default but reads from the
environment first — nothing about which model or server is baked into the code.

Env:
  SLM_MODEL           default "dispute-phi3-4ep:latest"
  OLLAMA_HOST         default "http://localhost:11434"
  SLM_TEMPERATURE     default 0.05
  SLM_NUM_CTX         default 2560   (context window, tokens — per spec)
  SLM_NUM_PREDICT     default 512    (max generated tokens - the free-text "explanation"
                                      field is generated last and can run long; too small
                                      a budget truncates it mid-sentence, which breaks the
                                      JSON and used to force a full fallback to ML-only)
  SLM_TIMEOUT_SECONDS default 180
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

MODEL = os.environ.get("SLM_MODEL", "dispute-phi3-4ep:latest")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
TEMPERATURE = float(os.environ.get("SLM_TEMPERATURE", "0.05"))
NUM_CTX = int(os.environ.get("SLM_NUM_CTX", "2560"))
NUM_PREDICT = int(os.environ.get("SLM_NUM_PREDICT", "512"))
DEFAULT_TIMEOUT = int(os.environ.get("SLM_TIMEOUT_SECONDS", "180"))

CLASSIFY_SCHEMA_FIELDS = ("reason_code", "reason_label", "category", "explanation")


def get_model() -> str:
    return MODEL


def set_model(name: str) -> None:
    """Switch which Ollama model this client talks to, at runtime - e.g. from the
    Settings panel in the UI, no restart needed. _chat()/available() read the module
    global MODEL fresh on every call, so this takes effect on the very next request.
    Everything else (ML classifier, KB, prompt structure, schema) is untouched -
    swapping the model name is the only thing this changes."""
    global MODEL
    MODEL = name


AVAILABLE_MODELS = [
    m.strip() for m in os.environ.get(
        "SLM_AVAILABLE_MODELS", "dispute-slm:latest,disputeslm-v2:latest",
    ).split(",") if m.strip()
]


def list_models() -> list[dict]:
    """Models the Settings picker offers - a static list from SLM_AVAILABLE_MODELS
    (comma-separated), not discovered live via Ollama's /api/tags. Switched from a
    live query because that call could hang indefinitely on some Windows hosts when
    Ollama's HTTP listener wedged, taking this endpoint (and the whole picker) down
    with it. Set SLM_AVAILABLE_MODELS to whatever `ollama list` shows on this machine;
    update it when you pull/create a new model there - no code change needed."""
    return [{"name": name} for name in AVAILABLE_MODELS]


def available() -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=4) as r:
            tags = json.load(r)
        names = {m.get("name", "") for m in tags.get("models", [])}
        return MODEL in names or any(n.split(":")[0] == MODEL.split(":")[0] for n in names)
    except Exception:
        return False


def _chat(messages: list[dict], schema: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
            # A mild nudge above Ollama's own default (1.1) — just enough to stop the
            # model looping the same sentence verbatim inside the free-text explanation
            # field. Tested at 1.3 first: that was too strong for this small fine-tune
            # and pushed it into fabricating garbage (invented codes, a made-up support
            # email) instead of looping — worse, not better. 1.15 stops the literal
            # loop without destabilizing the model. _dedupe_explanation below is the
            # real backstop regardless of how this is tuned.
            "repeat_penalty": 1.15, "repeat_last_n": 64,
        },
    }
    if schema is not None:
        payload["format"] = schema
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"]


_CODE_RE = re.compile(r'"reason_code"\s*:\s*"([^"]*)"')
_LABEL_RE = re.compile(r'"reason_label"\s*:\s*"([^"]*)"')
_CATEGORY_RE = re.compile(r'"category"\s*:\s*"([^"]*)"')
_EXPLANATION_RE = re.compile(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"?')


def _repair_partial_json(raw: str, code_enum: list[str]) -> dict | None:
    """The schema generates fields in order (reason_code, reason_label, category,
    explanation), so if generation got cut off (num_predict reached, or a connection
    hiccup mid-stream) before the JSON closed, the earlier fields are usually complete
    even though the document as a whole doesn't parse. Recover them with a regex scan
    rather than throwing the whole answer away — reason_code is what actually matters
    for classification, and it's the first (so most reliably complete) field. Returns
    None if even reason_code can't be recovered, so the caller can fall back cleanly."""
    m = _CODE_RE.search(raw)
    if not m or m.group(1) not in code_enum:
        return None
    out = {"reason_code": m.group(1), "_raw": raw, "_repaired": True}
    lm = _LABEL_RE.search(raw)
    if lm:
        out["reason_label"] = lm.group(1)
    cm = _CATEGORY_RE.search(raw)
    if cm:
        out["category"] = cm.group(1)
    em = _EXPLANATION_RE.search(raw)
    if em:
        expl = em.group(1).replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
        out["explanation"] = expl.strip()
    return out


def classify(system_prompt: str, user_content: str, code_enum: list[str],
             timeout: int = DEFAULT_TIMEOUT) -> dict:
    """One schema-constrained classification turn. Returns the parsed JSON dict
    (with '_raw'), a best-effort repaired dict (with '_raw' and '_repaired') if the
    response was truncated but reason_code was still recoverable, or {'_error': ...}
    if nothing usable came back at all."""
    schema = {
        "type": "object",
        "properties": {
            "reason_code": {"type": "string", "enum": sorted(code_enum)},
            "reason_label": {"type": "string"},
            "category": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": list(CLASSIFY_SCHEMA_FIELDS),
        "additionalProperties": False,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw = _chat(messages, schema=schema, timeout=timeout)
    try:
        obj = json.loads(raw)
        obj["_raw"] = raw
    except json.JSONDecodeError:
        repaired = _repair_partial_json(raw, code_enum)
        if repaired is None:
            return {"_error": "invalid_json", "_raw": raw}
        obj = repaired
    if "explanation" in obj:
        obj["explanation"] = _dedupe_explanation(_strip_filing_boilerplate(obj["explanation"]))
    return obj


# The model reliably pads every explanation with a templated filing-timeline clause
# ("Filed 3 days after the transaction via the app channel...") regardless of whether
# timing is actually relevant to the code — cosmetic noise, not part of the reasoning
# that actually matters. Strip it rather than show it.
_FILED_BOILERPLATE_RE = re.compile(
    r"\s*Filed\b[^.]*?\bvia\b[^.]*\.", re.IGNORECASE,
)


def _strip_filing_boilerplate(text: str) -> str:
    if not text:
        return text
    return _FILED_BOILERPLATE_RE.sub("", text).strip()


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_EXPLANATION_CHARS = 380


def _dedupe_explanation(text: str, max_sentences: int = 6) -> str:
    """Safety net for two related failure modes: the model looping the same sentence
    verbatim, or (seen when the repeat penalty was tuned too high) rambling into
    fabricated, off-topic content that never repeats but never stops either. The
    repeat penalty in _chat is the real fix for the first; this is the backstop for
    both — collapse consecutive duplicate sentences, then hard-cap the total length
    unconditionally so a degenerate response can never reach the user as a wall of
    text regardless of how sentence boundaries happen to fall in that response."""
    if not text:
        return text
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    out = []
    seen = set()
    for s in sentences:
        key = s.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_sentences:
            break
    result = " ".join(out).strip()
    if len(result) > _MAX_EXPLANATION_CHARS:
        cut = result[:_MAX_EXPLANATION_CHARS]
        last_boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        result = (cut[:last_boundary + 1] if last_boundary > 40 else cut).strip() + " …"
    return result


def _as_prose(text: str) -> str:
    """Defensive: despite the system prompt asking for natural language on follow-up
    turns, the model occasionally slips back into the classification JSON shape.
    Reformat that into a sentence rather than showing raw JSON to the user."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    code = obj.get("reason_code")
    if not code:
        return text
    label = obj.get("reason_label")
    explanation = obj.get("explanation", "")
    head = f"{code}" + (f" ({label})" if label else "")
    return f"{head}: {explanation}" if explanation else head


def follow_up(messages: list[dict], timeout: int = DEFAULT_TIMEOUT) -> str:
    """One free-text conversational turn. `messages` is the full history so far
    (system + prior user/assistant turns + the new user question)."""
    return _as_prose(_chat(messages, schema=None, timeout=timeout).strip())
