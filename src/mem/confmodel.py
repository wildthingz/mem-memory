"""The conflict scorer's coefficients (architecture-v2 §3.1 A6): a logistic model over generic
text, embedding, rerank and store signals, plus the adaptive list cut (`top_k`, `delta`; MEM2-154).
It lives in the gate JSON next to `theta_conf`, with a fingerprint over its numbers, so a store
never scores with coefficients it cannot name. `list_cut` is the one list cut: `mem.conflicts.find`
and the bench's offline evaluator (`bench.synth.calibration`) both apply it.

Features are generic by construction (rule 4): nothing here may know the SYN generator's cue
lists, templates, relation ids or value lists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# name -> what it measures. New text vs a candidate memory; list = every candidate picked.
FEATURES: dict[str, str] = {
    "rerank": "reranker raw logit of (new text, candidate)",
    "rerank_gap": "best rerank logit in the list minus this candidate's",
    "cosine": "cosine of the document-prompt embeddings, new text vs candidate body",
    "cos_gap": "best cosine in the list minus this candidate's",
    "cos_rank": "0-based rank of the candidate by cosine in the list",
    "jaccard": "word Jaccard, new text vs candidate body",
    "new_in_old": "share of the new text's words found in the candidate",
    "old_in_new": "share of the candidate's words found in the new text",
    "open": "1 when the candidate is currently valid (valid_to is null)",
    "older": "1 when the candidate's valid_from precedes the new one's",
    "same_instant": "1 when the valid_from are equal",
    "shared_subjects": "share of the new memory's subjects the candidate is also about",
    "same_type": "1 when the candidate's memory type (nodes.type, set by the caller) is the new one's",
}
MAX_TOP_K = 5
VERSION = 2  # 2: the adaptive cut (`delta`) joined the fingerprint; v1 records are refused


def fingerprint(
    features: list[str], weights: list[float], bias: float, top_k: int, delta: float
) -> str:
    body = json.dumps([VERSION, features, weights, bias, top_k, delta], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def list_cut(
    scores: Sequence[float], theta: float, delta: float, top_k: int,
    keys: Sequence[Any] | None,
) -> list[int]:  # fmt: skip
    """Indices of the kept candidates, best first: a candidate stays when `score >= theta` AND
    `score >= (best score in the list) - delta`; then the best `min(top_k, MAX_TOP_K)` remain.
    Ties go to the smaller key (`keys`, e.g. node ids; None = list position), so the cut is
    deterministic. `delta` is measured from the list's top, not from theta."""
    if not (math.isfinite(delta) and delta > 0):
        raise ValueError(f"list_cut: delta must be finite and > 0, got {delta}")
    if top_k < 1:
        raise ValueError(f"list_cut: top_k must be >= 1, got {top_k}")
    if not scores:
        return []
    floor = max(theta, max(scores) - delta)
    tie = (lambda i: keys[i]) if keys is not None else (lambda i: i)
    kept = sorted(
        (i for i, s in enumerate(scores) if s >= floor), key=lambda i: (-scores[i], tie(i))
    )
    return kept[: min(top_k, MAX_TOP_K)]


@dataclass(frozen=True)
class ConfModel:
    features: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    top_k: int
    delta: float  # the adaptive cut: keep only scores within `delta` of the list's best

    @classmethod
    def build(
        cls, features: list[str], weights: list[float], bias: float, top_k: int, delta: float
    ) -> ConfModel:
        if not features or len(features) != len(weights):
            raise ValueError("conf_model: features and weights must be non-empty and equal length")
        unknown = [f for f in features if f not in FEATURES]
        if unknown or len(set(features)) != len(features):
            raise ValueError(f"conf_model: unknown or repeated features {unknown or features}")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"conf_model: top_k must be in 1..{MAX_TOP_K}, got {top_k}")
        if not all(math.isfinite(x) for x in [*weights, bias]):
            raise ValueError("conf_model: non-finite coefficient")
        if not isinstance(delta, int | float) or not (math.isfinite(delta) and delta > 0):
            raise ValueError(f"conf_model: delta must be a finite number > 0, got {delta!r}")
        return cls(tuple(features), tuple(float(w) for w in weights), float(bias), int(top_k),
                   float(delta))  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ConfModel:
        """A record missing `top_k` or `delta` is refused: the list cut has no silent default."""
        for key in ("top_k", "delta"):
            if d.get(key) is None:
                raise ValueError(f"conf_model: {key} is required (no default list cut)")
        m = cls.build(list(d.get("features") or []), list(d.get("weights") or []),
                      d.get("bias", math.nan), d["top_k"], d["delta"])  # fmt: skip
        if d.get("fingerprint") != m.fingerprint():
            raise ValueError("conf_model: fingerprint does not match its coefficients")
        return m

    def fingerprint(self) -> str:
        return fingerprint(list(self.features), list(self.weights), self.bias, self.top_k,
                           self.delta)  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "features": list(self.features),
                "weights": list(self.weights), "bias": self.bias, "top_k": self.top_k,
                "delta": self.delta, "fingerprint": self.fingerprint()}  # fmt: skip

    def score(self, x: Mapping[str, float]) -> float:
        return self.bias + sum(w * x[f] for f, w in zip(self.features, self.weights, strict=True))

    def cut(self, scores: Sequence[float], theta: float, keys: Sequence[Any] | None) -> list[int]:
        """The adaptive list cut on this model's scores (`list_cut` with its top_k and delta)."""
        return list_cut(scores, theta, self.delta, self.top_k, keys)
