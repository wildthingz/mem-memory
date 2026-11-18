"""B-H1's latency clause: the B-C1 p95 increase from auth/TLS must be ≤ 50 ms (benchmarks-v2).

B-C1 itself times recall inside the server; auth and TLS cost is on the way in. So this times the
whole MCP `recall` call from the client, sequentially, against two servers over the same stores
and models: `base` (plain HTTP, no token: the P1-P6 path) and `secure` (the P7 path: bearer
token, rate limit, audit log, and over `tailscale serve` https). The result is the difference
of the two p95s. Each leg runs WARMUP calls first (session, TLS and SQLite pages warm)."""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from bench.metrics.stats import percentile_nearest_rank

WARMUP, N_CALLS, BUDGET_MS = 5, 300, 50
QUERIES = ("what coffee do I like", "when is the dentist", "who reviews on Fridays",
           "project deadline", "favourite tea", "where did I park", "birthday plans",
           "which laptop do I use", "standing meeting time", "holiday dates")  # fmt: skip


def recall_times(url: str, *, token: str | None, n: int, store: str = "personal") -> list[float]:
    """ms per `recall` call (client wall time) on `url`/<store>/mcp, after WARMUP calls."""
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async def go() -> list[float]:
        headers = {"X-Mem-Host": "bench-auth"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        http = httpx2.AsyncClient(headers=headers, timeout=60, trust_env=False)
        out = []
        async with Client(streamable_http_client(f"{url.rstrip('/')}/{store}/mcp",
                                                 http_client=http), mode="legacy") as c:  # fmt: skip
            for i in range(WARMUP + n):
                t0 = time.perf_counter()
                res = await c.call_tool("recall", {"query": QUERIES[i % len(QUERIES)]})
                ms = (time.perf_counter() - t0) * 1000
                if res.is_error:
                    raise RuntimeError(f"recall on {url} failed: {res.content}")
                if i >= WARMUP:
                    out.append(ms)
        return out

    return asyncio.run(go())


def _summary(times: list[float]) -> dict[str, float]:
    return {"p95_ms": round(percentile_nearest_rank(times, 0.95), 2),
            "median_ms": round(statistics.median(times), 2)}  # fmt: skip


def measure(base: str, secure: str, *, token: str | None, base_token: str | None = None,
            n: int = N_CALLS, store: str = "personal") -> dict[str, Any]:  # fmt: skip
    b = _summary(recall_times(base, token=base_token, n=n, store=store))
    s = _summary(recall_times(secure, token=token, n=n, store=store))
    delta = round(s["p95_ms"] - b["p95_ms"], 2)
    return {"n": n, "store": store, "base_url": base, "secure_url": secure, "base": b,
            "secure": s, "delta_p95_ms": delta, "budget_ms": BUDGET_MS,
            "pass": delta <= BUDGET_MS}  # fmt: skip
