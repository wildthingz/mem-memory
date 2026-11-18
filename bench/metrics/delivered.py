"""Delivered-context metrics on S_q (benchmarks-v2 §2.0). Unit: one query = (S_q, G_q, answerable).

`G_q` is the set of ids with rel >= 1. No-answer queries have an empty G_q and answerable=False.
Empty denominators return `float('nan')`, never 0, so a missing class cannot pass a gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Delivered:
    returned: Sequence[str]
    gold: frozenset[str]  # rel >= 1
    answerable: bool
    required: frozenset[str] | None = None  # rel == 2; None = same as gold (binary sets)


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")


def coverage_any(qs: Sequence[Delivered]) -> float:
    a = [q for q in qs if q.answerable]
    return _ratio(sum(bool(set(q.returned) & q.gold) for q in a), len(a))


def delivered_precision(qs: Sequence[Delivered]) -> float:
    rows = sum(len(q.returned) for q in qs)
    return _ratio(sum(len(set(q.returned) & q.gold) for q in qs), rows)


def ira(qs: Sequence[Delivered]) -> float:
    nonempty = [q for q in qs if q.returned]
    return _ratio(sum(len(set(q.returned) - q.gold) for q in nonempty), len(nonempty))


def far(qs: Sequence[Delivered]) -> float:
    a = [q for q in qs if q.answerable]
    return _ratio(sum(not q.returned for q in a), len(a))


def abstention_recall(qs: Sequence[Delivered]) -> float:
    n = [q for q in qs if not q.answerable]
    return _ratio(sum(not q.returned for q in n), len(n))


def coverage_all(qs: Sequence[Delivered]) -> float:
    """Answerable queries whose every rel=2 id was delivered (binary sets: every gold id)."""
    a = [q for q in qs if q.answerable]
    return _ratio(
        sum((q.gold if q.required is None else q.required) <= set(q.returned) for q in a), len(a)
    )


def f05(precision: float, coverage: float) -> float:
    """Delivered F0.5 = 1.25 P Cov / (0.25 P + Cov); nan if either input is nan."""
    if math.isnan(precision) or math.isnan(coverage):
        return float("nan")
    den = 0.25 * precision + coverage
    return 1.25 * precision * coverage / den if den else 0.0
