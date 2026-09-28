"""Knowledge-base loader and lookup for dispute reason codes.

This is the single source of truth for which networks and which reason codes
exist — nothing else in the codebase should hardcode a network name or a code
list; they all derive it from here (in turn read from data/reason_code_kb.json,
overridable via the DISPUTE_KB_PATH env var)."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = Path(os.environ.get("DISPUTE_KB_PATH", ROOT / "data" / "reason_code_kb.json"))


class KnowledgeBase:
    def __init__(self, path: Path | str = KB_PATH):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = raw.get("meta", {})
        self.entries = {e["code"]: e for e in raw["codes"]}

    def get(self, code: str) -> dict | None:
        return self.entries.get(code)

    def networks(self) -> list[str]:
        """Ordered list of card networks covered by the KB — from meta.networks
        if declared, else the distinct `network` values seen on the codes."""
        declared = self.meta.get("networks")
        if declared:
            return list(declared)
        seen = []
        for e in self.entries.values():
            if e["network"] not in seen:
                seen.append(e["network"])
        return seen

    def enrich(self, code: str, probability: float | None = None) -> dict:
        """Return the KB entry for a code plus the model's probability, or a
        graceful stub if the code is somehow not in the KB."""
        e = self.entries.get(code)
        if e is None:
            return {"code": code, "known": False, "probability": probability,
                    "summary": "No knowledge-base entry found for this code."}
        out = dict(e)
        out["known"] = True
        out["probability"] = probability
        return out

    def codes_for_network(self, network: str) -> list[dict]:
        return [e for e in self.entries.values() if e["network"] == network]

    def all_codes(self) -> list[str]:
        return list(self.entries.keys())
