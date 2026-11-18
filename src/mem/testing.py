"""Deterministic CPU stand-ins for the embedder and reranker (the T0 `HashEmbed-256` stub).

Unit tests and T0 benchmarks use these; they never load a model or touch the network.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> list[float]: ...
    def embed_docs(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        """Raw logits, one per doc (no activation)."""
        ...


class HashEmbedder:
    """Bag-of-words hashed into `dim` buckets, L2-normalised: shared words mean cosine > 0."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _words(text):
            v[int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_docs(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OverlapReranker:
    """Logit = 6 * (fraction of query words found in the doc) - 3: monotone in overlap."""

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        q = set(_words(query))
        out = []
        for d in docs:
            frac = len(q & set(_words(d))) / len(q) if q else 0.0
            out.append(6.0 * frac - 3.0)
        return out


def stub_conf_model(top_k: int = 5, delta: float = 100.0) -> dict:
    """Conflict-scorer coefficients for unit tests: the reranker logit alone (score = rerank); the
    default `delta` is far wider than any logit gap, so only theta_conf and top_k cut."""
    from mem.confmodel import ConfModel

    return ConfModel.build(["rerank"], [1.0], 0.0, top_k, delta).to_dict()
