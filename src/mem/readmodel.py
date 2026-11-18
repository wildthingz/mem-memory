"""The read gate's calibrated probability (architecture-v2 §3.3 step 3, MEM2-155).

A rerank logit alone cannot separate answerable from missing-slot queries on SYN (AUROC 0.90, one
threshold misses B-R2 (ii)). The read model is a small logistic over three generic signals of a
pool member: the reranker raw logit `rerank`, the embedding cosine of the query and the chunk
(`cosine`, query prompt vs document prompt, the signal the conflict scorer uses) and the pool's
`gap` (best raw logit minus second best: how clearly the top is a single answer). `p = σ(bias +
Σ wᵢ·xᵢ)` replaces `σ(a·z + b)`; `τ_keep`, `τ_high` and `g` then apply to this `p` unchanged.

The coefficients live in `meta.gate.read_model` with a fingerprint over their numbers, like the
conflict scorer's. A `read_model` that is incomplete, or whose fingerprint does not match, is
refused at load: there is no silent fallback to the Platt logit. No feature may know the SYN
generator's cue lists, templates, relation ids or value lists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mem import relphrase, subjectmask

FEATURES: dict[str, str] = {
    "rerank": "reranker raw logit of (query, chunk)",
    "cosine": "cosine of the query-prompt embedding and the chunk's document-prompt embedding",
    "gap": "best raw logit in the pool minus the second best (0 with one candidate)",
    "cosine_masked": "`cosine` with the subject's name masked in the query and in the chunk",
    "rerank_masked": "`rerank` on the pair with the subject's name masked (mem.subjectmask)",
    "cosine_masked_margin": "best `cosine_masked` in the pool minus the second best (0 with one)",
    "rerank_masked_margin": "best `rerank_masked` in the pool minus the second best (0 with one)",
    "phrase_cosine": "cosine of the question's and the chunk's predicate phrases, subject removed,"
    " function words dropped (mem.relphrase)",
    "phrase_overlap": "share of the question's predicate stems found in the chunk's (mem.relphrase)",
}
MASKED = ("cosine_masked", "rerank_masked")  # computed by mem.subjectmask, not from z / cosines
PHRASE = ("phrase_cosine", "phrase_overlap")  # computed by mem.relphrase (MEM2-171)
SIGNALS = (*MASKED, *PHRASE)  # every signal the store computes besides z and the cosine
MARGINS = {"cosine_masked_margin": "cosine_masked", "rerank_masked_margin": "rerank_masked"}
VERSION = 1


def signals_for(features: Sequence[str]) -> tuple[str, ...]:
    """The store-computed signals (`SIGNALS`) a model over `features` needs: a masked or phrase
    feature is its own signal, a margin needs the masked signal it is derived from."""
    want = {MARGINS.get(f, f) for f in features}
    return tuple(s for s in SIGNALS if s in want)


def fingerprint(features: Sequence[str], weights: Sequence[float], bias: float) -> str:
    body = [VERSION, list(features), list(weights), bias]
    if signals_for(features):  # the subject rule is part of what the numbers mean
        body.append(
            [
                subjectmask.MASK,
                subjectmask.MAX_NAME_WORDS,
                subjectmask.MIN_NAME_CHARS,
                subjectmask.MIN_TOKEN_CHARS,
            ]
        )
    if any(f in PHRASE for f in features):  # and so is the phrase rule
        body.append(relphrase.rule())
    body = json.dumps(body, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def feature_matrix(
    zs: Sequence[float],
    cosines: Sequence[float | None] | None,
    names: Sequence[str],
    masked: Mapping[str, Sequence[float | None]] | None = None,
) -> np.ndarray:
    """(items, len(names)) feature rows of one query's pool, in `names` order. `masked` holds the
    store-computed signals (`SIGNALS`) by name; a margin is derived from its masked signal."""
    n = len(zs)
    z = np.asarray(zs, dtype=float)
    cols = []
    for name in names:
        if name == "rerank":
            cols.append(z)
        elif name == "gap":
            cols.append(np.full(n, _margin(z)))
        elif name == "cosine":
            if cosines is None or any(c is None for c in cosines):
                raise ValueError("the read model needs the query-chunk cosine of every candidate")
            cols.append(np.asarray(cosines, dtype=float))
        elif name in SIGNALS or name in MARGINS:
            sig = MARGINS.get(name, name)
            col = None if masked is None else masked.get(sig)
            if col is None or len(col) != n or any(c is None for c in col):
                how = "subject-masked signal, mem.subjectmask" if sig in MASKED else (
                    "relation-phrase signal, mem.relphrase")  # fmt: skip
                raise ValueError(
                    f"the read model needs {sig!r} for every candidate ({how}; feature {name!r}):"
                    " re-score the records or pass it to probs()"
                )
            v = np.asarray(col, dtype=float)
            cols.append(np.full(n, _margin(v)) if name in MARGINS else v)
        else:
            raise ValueError(f"unknown read-model feature {name!r}")
    return np.stack(cols, axis=1) if cols else np.zeros((n, 0))


def _margin(v: np.ndarray) -> float:
    """Best minus second best (0 with fewer than two candidates)."""
    if v.size < 2:
        return 0.0
    top2 = np.sort(v)[::-1][:2]
    return float(top2[0] - top2[1])


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass(frozen=True)
class ReadModel:
    features: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float

    @classmethod
    def build(cls, features: Sequence[str], weights: Sequence[float], bias: float) -> ReadModel:
        if not features or len(features) != len(weights):
            raise ValueError("read_model: features and weights must be non-empty and equal length")
        unknown = [f for f in features if f not in FEATURES]
        if unknown or len(set(features)) != len(features):
            raise ValueError(f"read_model: unknown or repeated features {unknown or features}")
        if not all(isinstance(x, int | float) and math.isfinite(x) for x in [*weights, bias]):
            raise ValueError("read_model: missing or non-finite coefficient")
        return cls(tuple(features), tuple(float(w) for w in weights), float(bias))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReadModel:
        """A record without coefficients, or whose fingerprint disagrees with them, is refused."""
        if d.get("bias") is None or not d.get("weights") or not d.get("features"):
            raise ValueError(
                "read_model: features, weights and bias are required (no fallback to the Platt"
                " logit; refit: mem calibrate --objective p3)"
            )
        m = cls.build(list(d["features"]), list(d["weights"]), d["bias"])
        if d.get("fingerprint") != m.fingerprint():
            raise ValueError("read_model: fingerprint does not match its coefficients")
        return m

    def fingerprint(self) -> str:
        return fingerprint(self.features, self.weights, self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "features": list(self.features),
                "weights": list(self.weights), "bias": self.bias,
                "fingerprint": self.fingerprint()}  # fmt: skip

    @property
    def masked_features(self) -> tuple[str, ...]:
        """The store-computed signals this model needs (`SIGNALS`: subject-masked and phrase, the
        masked one under a margin), computed by `subjectmask.pool_masked`; usually none."""
        return signals_for(self.features)

    def probs(
        self,
        zs: Sequence[float],
        cosines: Sequence[float | None] | None,
        masked: Mapping[str, Sequence[float | None]] | None = None,
    ) -> list[float]:
        """p of every candidate of one query's pool (deterministic: float64, fixed order)."""
        if not zs:
            return []
        x = feature_matrix(zs, cosines, self.features, masked)
        return sigmoid(x @ np.asarray(self.weights) + self.bias).tolist()
