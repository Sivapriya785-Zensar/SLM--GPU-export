"""
Serving wrapper for the hierarchical dispute classifier.

    from ml.classifier import DisputeClassifier
    clf = DisputeClassifier.load()
    clf.predict({"dispute_narrative": "...", "channel": "ecommerce", ...})

Returns the predicted network and a ranked top-k of reason codes with probabilities.
Pure model output - no heuristic correction. If `network` is supplied in the payload
it constrains the prediction to that network's codes (the user picked it in the UI);
otherwise the router decides.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

try:
    from .features import build_text
except ImportError:  # run as a script / non-package context
    from features import build_text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kb.kb import KnowledgeBase  # noqa: E402 — the networks list comes from the KB, not a literal here

MODELS = ROOT / "ml" / "models"


class DisputeClassifier:
    def __init__(self, network_pipe, code_pipes, report=None):
        self.network_pipe = network_pipe
        self.code_pipes = code_pipes
        self.report = report or {}
        self.networks = list(code_pipes.keys())

    @classmethod
    def load(cls, models_dir: Path | str = MODELS, networks: list[str] | None = None):
        models_dir = Path(models_dir)
        networks = networks or KnowledgeBase().networks()
        net = joblib.load(models_dir / "network.joblib")
        codes = {n: joblib.load(models_dir / f"code_{n}.joblib") for n in networks}
        report = {}
        rp = models_dir / "report.json"
        if rp.exists():
            report = json.loads(rp.read_text(encoding="utf-8"))
        return cls(net, codes, report)

    def _rank(self, pipe, text, k):
        proba = pipe.predict_proba([text])[0]
        classes = pipe.classes_
        order = proba.argsort()[::-1][:k]
        return [{"code": str(classes[i]), "probability": round(float(proba[i]), 4)} for i in order]

    def predict(self, payload: dict, k: int = 3) -> dict:
        text = build_text(payload)

        forced = payload.get("network")
        if forced in self.networks:
            network, network_source = forced, "user"
            net_proba = None
        else:
            network = str(self.network_pipe.predict([text])[0])
            network_source = "model"
            np_ = self.network_pipe.predict_proba([text])[0]
            nc = self.network_pipe.classes_
            net_proba = {str(nc[i]): round(float(np_[i]), 4) for i in np_.argsort()[::-1]}

        ranked = self._rank(self.code_pipes[network], text, k)
        return {
            "network": network,
            "network_source": network_source,
            "network_probabilities": net_proba,
            "predictions": ranked,
            "top_code": ranked[0]["code"],
            "input_text": text,
        }
