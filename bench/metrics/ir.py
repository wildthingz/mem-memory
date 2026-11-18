"""Ranking metrics (benchmarks-v2 §2.0).

`ndcg_std` and `ndcg_any_upstream` are different metrics and never share a threshold (A6): the
std form uses gain 2^rel - 1 and discount log2(i + 1); the upstream LongMemEval form
(`src/retrieval/eval_utils.py`) uses gain rel, no discount at rank 1 and 1/log2(i) after it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _dcg_std(rels: Sequence[int], k: int) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_std(ranked: Sequence[str], gold: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k (std). `gold` maps id -> rel in {1, 2}; ids absent from it have rel 0."""
    ideal = _dcg_std(sorted(gold.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return _dcg_std([gold.get(d, 0) for d in ranked], k) / ideal


def _dcg_upstream(rels: Sequence[int], k: int) -> float:
    rels = rels[:k]
    if not rels:
        return 0.0
    return rels[0] + sum(r / math.log2(i) for i, r in enumerate(rels[1:], start=2))


def ndcg_any_upstream(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    """ndcg_any@k as in upstream LongMemEval (binary relevance)."""
    ideal = _dcg_upstream([1] * min(len(gold), k), k)
    if ideal == 0:
        return 0.0
    return _dcg_upstream([int(d in gold) for d in ranked], k) / ideal


def recall_any(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return float(bool(gold & set(ranked[:k])))


def recall_all(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return float(bool(gold) and gold <= set(ranked[:k]))


def mrr(ranked: Sequence[str], gold: set[str], depth: int = 50) -> float:
    for i, d in enumerate(ranked[:depth], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0
