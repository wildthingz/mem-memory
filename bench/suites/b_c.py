"""B-C1 and B-C2 recorders for P1 (no threshold, benchmarks-v2 §4 P1 row).

Both run the fixture corpus (10 memories, 8 queries + 1 query that matches nothing). They are
smoke recordings of the in-process path, not the B-C1 protocol: the real numbers come from the
day-1 spike (`bench/spike.py`, runbook p1-gate.md) and from B-R1/B-R3 at L5k/V5k (P2+).
B-C2 token counts are the server estimate (est.), not `count_tokens`.
"""

from __future__ import annotations

import platform
import statistics
import time

from bench.fixtures import streams as fx
from bench.metrics.tokens import est_tokens, summarize
from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import seed_corpus

QUERIES = [*fx.CORPUS_QUERIES, fx.NO_MATCH_QUERY]


def _responses(ctx: Context, tag: str) -> list[tuple[str, float, str]]:
    client = ctx.factory(ctx.store(f"{tag}.db"), scope="personal")
    try:
        seed_corpus(client, fx.CORPUS, "2026-10-01T09:00:00Z")
        out = []
        for q in QUERIES:
            t0 = time.perf_counter_ns()
            r = client.call("recall", {"query": q}, now="2026-10-22T09:00:00Z")
            out.append((q, (time.perf_counter_ns() - t0) / 1e6, r.text))
        return out
    finally:
        client.close()


def run_b_c2(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-C2", mode="fixture")
    rows = _responses(ctx, "b_c2")
    toks = [est_tokens(text) for _, _, text in rows]
    res.items = [{"q": q, "tokens_est": t} for (q, _, _), t in zip(rows, toks, strict=True)]
    s = summarize(toks)
    empty = [t for (q, _, text), t in zip(rows, toks, strict=True) if text.startswith("no results")]
    res.metrics |= {"tokens_mean_est": float(s["mean"]), "tokens_p95_est": float(s["p95"]),
                    "n": float(s["n"]), "empty_tokens_max_est": float(max(empty, default=0))}  # fmt: skip
    # P1 has no gate, so a corpus query is never empty (abstention arrives in P2): record how many
    # were, and require only that the responses are not all empty.
    res.metrics["n_empty"] = float(len(empty))
    res.check(
        "recorded_nonempty_cases", len(empty) < len(rows), f"{len(empty)} empty of {len(rows)}"
    )
    return res


def run_b_c1(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-C1", mode="fixture")
    rows = _responses(ctx, "b_c1")
    ms = [m for _, m, _ in rows]
    res.items = [{"q": q, "latency_ms": m} for q, m, _ in rows]
    res.metrics |= {"cold_ms": ms[0], "warm_median_ms": statistics.median(ms[1:]),
                    "warm_max_ms": max(ms[1:]), "n": float(len(ms))}  # fmt: skip
    res.notes.append(f"host {platform.platform()}, {platform.processor() or 'cpu'}; "
                     "wall time around the in-process call, first call = cold; not the B-C1 protocol")  # fmt: skip
    res.check("recorded", len(ms) == len(QUERIES))
    return res
