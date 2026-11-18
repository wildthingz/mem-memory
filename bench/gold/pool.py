"""Pooling (benchmarks-v2 §3.1): top 5 of R_q from ours, BM25 and dense, per question, deduplicated,
minus the chunks already labelled. Budget: ≤ 1,800 pairs, ≈ 700-900 expected unjudged (~8 s each).

The pool order is not the systems' order: pairs of one question are shuffled by sha1(qid, id) so the
person is not anchored on which system found a chunk. The contributing systems are stored in the
pool row (never shown) so the baselines' unjudged share can be reported (pooling bias).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from bench.gold import pg
from bench.gold.session import Work

DEPTH = 5
MAX_PAIRS = 1_800
SYSTEMS = ("ours", "bm25", "dense")


class PoolBackend(Protocol):
    def systems(self, q: pg.Question) -> dict[str, list[str]]:
        """Ranked gold ids per system for q (scope filters applied), at least DEPTH each:
        `ours` = R_q, `bm25` and `dense` = the retrieval legs on the same eligible set."""
        ...


@dataclass
class PoolStats:
    questions: int
    pairs: int
    new_pairs: int
    over_budget: bool


def pool_for(
    q: pg.Question, backend: PoolBackend, labelled: set[tuple[str, str]]
) -> list[dict[str, Any]]:
    runs = backend.systems(q)
    missing = [s for s in SYSTEMS if s not in runs]
    if missing:
        raise ValueError(f"backend returned no run for {missing}")
    contrib: dict[str, list[str]] = {}
    for system in SYSTEMS:
        for gid in runs[system][:DEPTH]:
            contrib.setdefault(gid, []).append(system)
    rows = [{"qid": q.qid, "gold_id": g, "systems": s} for g, s in contrib.items()
            if (q.qid, g) not in labelled]  # fmt: skip
    return sorted(rows, key=lambda r: pg.sha1_key(q.qid + r["gold_id"]))


def build(work: Work, questions: Sequence[pg.Question], backend: PoolBackend) -> PoolStats:
    """Add the pairs of the accepted questions that are not pooled yet to pool.jsonl."""
    existing = work.pool()
    have_q = {p["qid"] for p in existing}
    labelled = set(work.labels())
    new: list[dict[str, Any]] = []
    for q in sorted(questions, key=lambda q: pg.sha1_key(q.qid)):
        if q.qid not in have_q:
            new += pool_for(q, backend, labelled)
    for row in new:
        work.log_event("pool.jsonl", row)
    total = len(existing) + len(new)
    return PoolStats(len({p["qid"] for p in existing + new}), total, len(new), total > MAX_PAIRS)
