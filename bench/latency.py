"""B-C1 run protocol (benchmarks-v2 B-C): per-process latency runs and their aggregation.

One *process* run = fresh process, one load corpus, 20 warm-up + 300 queries with 1 client and
then with 4 concurrent clients (queueing included). The first call after process start is `cold`
and never enters warm statistics. A gate needs 3 independent process runs (`bench latency-run`
three times); the gate value is the median of the 3 runs' warm p95.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench.metrics.stats import percentile_nearest_rank

WARMUP, N_QUERIES, CLIENTS = 20, 300, 4
STAGES = ("eligible", "fts", "vec", "fusion", "rerank", "guard", "total")  # guard: ADR 0018
RUNS_REQUIRED = 3

# server(query) -> {"fts": ms, "vec": ms, "fusion": ms, "rerank": ms, "guard": ms, "total": ms}
# (server time; `guard` only when the read-time conflict guard ran on the final set)
Timed = Callable[[str], dict[str, float]]


def _summ(calls: Sequence[dict[str, float]]) -> dict[str, Any]:
    return {s: {"median": statistics.median(c[s] for c in calls if s in c),
                "p95": percentile_nearest_rank([c[s] for c in calls if s in c], 0.95)}
            for s in STAGES if any(s in c for c in calls)}  # fmt: skip


def schedule(questions: Sequence[str]) -> list[str]:
    """The 1 + WARMUP + N_QUERIES calls of one process run from `questions` (the B-R1 test set in
    sha1 order, 269 < 321): call i asks questions[i mod n]. Deterministic; with n = 269 the last
    52 measured calls repeat the cold, warm-up and first measured questions. The read path keeps
    no result cache, so a repeat costs the same as a first ask (only SQLite pages are warm, as
    they are for every warm call)."""
    if not questions:
        raise ValueError("no questions to schedule")
    return [questions[i % len(questions)] for i in range(1 + WARMUP + N_QUERIES)]


def run_process(server: Timed, queries: Sequence[str]) -> dict[str, Any]:
    """Measure one process. `queries` = 1 cold + WARMUP + N_QUERIES entries."""
    if len(queries) < 1 + WARMUP + N_QUERIES:
        raise ValueError(f"need {1 + WARMUP + N_QUERIES} queries, got {len(queries)}")
    cold = server(queries[0])  # first call after process start, reported separately
    for q in queries[1 : 1 + WARMUP]:
        server(q)
    measured = list(queries[1 + WARMUP : 1 + WARMUP + N_QUERIES])
    one = [server(q) for q in measured]

    def timed(q: str) -> dict[str, float]:
        t0 = time.perf_counter_ns()
        r = dict(server(q))
        r["total"] = (time.perf_counter_ns() - t0) / 1e6  # queueing included
        return r

    with ThreadPoolExecutor(CLIENTS) as pool:
        four = list(pool.map(timed, measured))
    return {"cold_ms": cold.get("total"), "n": len(measured),
            "one_client": _summ(one), "four_clients": _summ(four)}  # fmt: skip


def aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Gate values from >= 3 process runs: median of per-run warm p95 (1 client and 4 clients)."""
    if len(runs) < RUNS_REQUIRED:
        raise ValueError(f"B-C1 needs {RUNS_REQUIRED} independent process runs, got {len(runs)}")
    p1 = [r["one_client"]["total"]["p95"] for r in runs]
    p4 = [r["four_clients"]["total"]["p95"] for r in runs]
    return {"warm_p95_ms_median": statistics.median(p1), "warm_p95_ms_min": min(p1),
            "warm_p95_ms_max": max(p1), "warm_p95_4clients_ms_median": statistics.median(p4),
            "cold_ms_median": statistics.median(r["cold_ms"] for r in runs),
            "rerank_p95_ms_median": statistics.median(
                r["one_client"]["rerank"]["p95"] for r in runs if "rerank" in r["one_client"])
            if all("rerank" in r["one_client"] for r in runs) else float("nan"),
            "n_runs": float(len(runs))}  # fmt: skip


def regression(new_p95: Sequence[float], base_p95: Sequence[float]) -> bool:
    """True = regression: median of new > 1.2 x baseline median AND lowest new > highest baseline."""
    return statistics.median(new_p95) > 1.2 * statistics.median(base_p95) and min(new_p95) > max(
        base_p95
    )


def load_runs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]


def run_write_process(server: Timed, calls: Sequence[str]) -> dict[str, Any]:
    """The B-C3 form of `run_process`: distinct calls for every phase (a repeated `remember` would
    be an `exists` hit). `calls` = 1 cold + WARMUP + N_QUERIES (one client) + N_QUERIES (four
    clients, queueing included)."""
    need = 1 + WARMUP + 2 * N_QUERIES
    if len(calls) < need:
        raise ValueError(f"need {need} calls, got {len(calls)}")
    cold = server(calls[0])
    for q in calls[1 : 1 + WARMUP]:
        server(q)
    one_calls = list(calls[1 + WARMUP : 1 + WARMUP + N_QUERIES])
    four_calls = list(calls[1 + WARMUP + N_QUERIES : need])
    one = [server(q) for q in one_calls]

    def timed(q: str) -> dict[str, float]:
        t0 = time.perf_counter_ns()
        r = dict(server(q))
        r["total"] = (time.perf_counter_ns() - t0) / 1e6
        return r

    with ThreadPoolExecutor(CLIENTS) as pool:
        four = list(pool.map(timed, four_calls))
    return {"cold_ms": cold.get("total"), "n": N_QUERIES,
            "one_client": _summ(one), "four_clients": _summ(four)}  # fmt: skip
