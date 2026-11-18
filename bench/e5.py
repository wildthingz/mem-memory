"""B-E5 client legs (benchmarks-v2 §2 B-E5): one machine's run against a `mem serve --http`.

`run_client` is what `bench e5-run` does on each machine: through the same MCP-over-HTTP shape
Claude Code uses (`/<store>/mcp`, `X-Mem-Host` header, one session per store) it

1. calls `recall` for the fixed queries on both stores, `runs` x (3 in the spec) and times each call
   with `time.perf_counter` around the MCP call (client round trip);
2. replays the fixed write ops on a *scratch* server (a copy of the stores; never the live one), and
   a remember -> recall -> get smoke round trip (B-S1's leg over this transport);
3. checks isolation: a canary remembered in one scratch store must not come back from the other.

The record (JSON) is compared across machines by `bench.suites.b_p5` (parity 1e-6, ids, +p95).
`run_reconnect` is the restart leg. Nothing here loads a model: only the server does.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import secrets
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

STORES = ("personal", "work")
VOLATILE = {
    "at",
    "ms",
    "took_ms",
    "elapsed_ms",
    "session",
    "host",
}  # never compared across machines

# The recall legs write `recall` events into the LIVE stores (300+ per run). They carry a bench-only
# host label so daily-use counts (P6 B-E4: recalls/day, days from work) can leave them out with
# `LIVE_EVENTS_FILTER` (MEM2-41 F7).
BENCH_HOST_PREFIX = "bench-"
LIVE_EVENTS_FILTER = f"host NOT LIKE '{BENCH_HOST_PREFIX}%'"


def bench_host_label(label: str) -> str:
    """The default `X-Mem-Host` of a bench run: `bench-<label>` (unchanged if already bench-)."""
    return label if label.startswith(BENCH_HOST_PREFIX) else BENCH_HOST_PREFIX + label


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def p95(values: list[float]) -> float:
    """Nearest-rank p95 (the same rule as `bench.metrics.tokens.summarize`)."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else float("nan")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_queries(path: Path) -> list[dict[str, str]]:
    """JSONL, one `{"qid": ..., "query": ...}` per line (`bench e5-queries` writes it from PG-v1)."""
    rows = read_jsonl(path)
    for r in rows:
        if not r.get("qid") or not r.get("query"):
            raise ValueError(f"{path}: every line needs qid and query, got {r}")
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate qid")
    return rows


def load_ops(path: Path) -> list[dict[str, Any]]:
    """JSONL, one `{"store": "personal"|"work", "tool": ..., "args": {...}}` per line."""
    rows = read_jsonl(path)
    for r in rows:
        if r.get("store") not in STORES or not r.get("tool") or not isinstance(r.get("args"), dict):
            raise ValueError(f"{path}: bad op line {r}")
    return rows


def clean(obj: Any) -> Any:
    """A tool result without the keys that legitimately differ per machine/call."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    return obj


def _mcp_client(base: str, store: str, host_label: str):
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    http = httpx2.AsyncClient(headers={"X-Mem-Host": host_label}, timeout=120)
    # "legacy" = the handshake protocol with `Mcp-Session-Id` (what Claude Code speaks today)
    return Client(streamable_http_client(f"{base.rstrip('/')}/{store}/mcp", http_client=http),
                  mode="legacy")  # fmt: skip


async def _timed(client: Any, tool: str, args: dict[str, Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    res = await client.call_tool(tool, args)
    ms = (time.perf_counter() - t0) * 1000
    if getattr(res, "is_error", False):
        text = " ".join(getattr(c, "text", "") for c in res.content)
        raise RuntimeError(f"{tool} returned an error: {text}")
    return res, ms


async def _recall_leg(base: str, store: str, host_label: str, queries: list[dict[str, str]],
                      runs: int) -> dict[str, Any]:  # fmt: skip
    results: dict[str, dict[str, Any]] = {}
    latencies: list[float] = []
    unstable: list[str] = []
    async with _mcp_client(base, store, host_label) as c:
        for _ in range(runs):
            for q in queries:
                res, ms = await _timed(c, "recall", {"query": q["query"]})
                latencies.append(ms)
                rows = (res.structured_content or {}).get("rows", [])
                got = {"ids": [r["id"] for r in rows], "p": [r.get("p") for r in rows]}
                first = results.setdefault(q["qid"], got)
                if first != got and q["qid"] not in unstable:
                    unstable.append(q["qid"])  # the same query, same machine, different answer
    return {"results": results, "latencies_ms": latencies, "unstable_qids": unstable}


async def _write_leg(base: str, host_label: str, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async with contextlib.AsyncExitStack() as stack:
        sessions = {s: await stack.enter_async_context(_mcp_client(base, s, host_label))
                    for s in STORES}  # fmt: skip
        for i, op in enumerate(ops):
            res, _ = await _timed(sessions[op["store"]], op["tool"], op["args"])
            out.append({"i": i, "store": op["store"], "tool": op["tool"],
                        "structured": clean(res.structured_content or {})})  # fmt: skip
    return out


async def _isolation_and_smoke(base: str, host_label: str) -> dict[str, Any]:
    """On the scratch server: a canary per store must stay in its store; remember -> recall -> get."""
    nonce = secrets.token_hex(4)
    canary = {s: f"zq{nonce}{s} canary memory for the isolation check" for s in STORES}
    seen: dict[str, dict[str, list[str]]] = {s: {} for s in STORES}
    problems: list[str] = []
    ids: dict[str, str] = {}
    async with _mcp_client(base, "personal", host_label) as p, \
            _mcp_client(base, "work", host_label) as w:  # fmt: skip
        sess = {"personal": p, "work": w}
        for s in STORES:
            res, _ = await _timed(sess[s], "remember", {"text": canary[s], "type": "fact",
                                  "about": [f"new:topic:Canary{s.title()}"]})  # fmt: skip
            st = res.structured_content or {}
            if st.get("status") != "saved" or not st.get("id"):
                problems.append(f"smoke: remember on {s} gave {st.get('status')!r}")
            ids[s] = st.get("id", "")
        cross: list[str] = []
        for s in STORES:
            for other in STORES:
                res, _ = await _timed(sess[other], "recall", {"query": canary[s]})
                rows = (res.structured_content or {}).get("rows", [])
                titles = " ".join(r.get("title", "") for r in rows)
                seen[other][s] = [r["id"] for r in rows]
                if other == s:
                    if ids[s] not in seen[s][s]:
                        problems.append(f"smoke: recall on {s} did not return {ids[s]}")
                    got, _ = await _timed(sess[s], "get", {"ids": [ids[s]]})
                    items = (got.structured_content or {}).get("items", [])
                    if not (len(items) == 1 and items[0].get("text") == canary[s]):
                        problems.append(f"smoke: get {ids[s]} on {s} not verbatim")
                else:
                    # the other store holds a different canary; `zq<nonce><s>` must never appear there
                    if f"zq{nonce}{s}" in titles:
                        cross.append(f"{s} canary came back from {other}")
                    for r in rows:
                        got, _ = await _timed(sess[other], "get", {"ids": [r["id"]]})
                        for it in (got.structured_content or {}).get("items", []):
                            if canary[s] in (it.get("text") or ""):
                                cross.append(f"{s} canary text in {other} item {r['id']}")
    return {"cross_store_ids": len(cross), "cross_store_detail": cross, "smoke_failures": len(problems),
            "smoke_detail": problems}  # fmt: skip


def run_client(
    base: str, *, label: str, role: str, host_label: str, queries: list[dict[str, str]],
    ops: list[dict[str, Any]], scratch: str | None, runs: int = 3, transport: str = "",
) -> dict[str, Any]:  # fmt: skip
    """One machine's B-E5 record. `role` is `local` (the home machine on 127.0.0.1: the baseline of
    the +p95 gate) or `remote`. `scratch` is the URL of a server on a scratch copy of the stores."""
    if role not in ("local", "remote"):
        raise ValueError("role must be local or remote")
    rec: dict[str, Any] = {
        "kind": "e5-run", "label": label, "role": role, "base": base, "scratch": scratch,
        "host_label": host_label, "transport": transport or urlparse(base).scheme,
        "runs": runs, "queries_sha256": sha256_text(json.dumps(queries, sort_keys=True)),
        "ops_sha256": sha256_text(json.dumps(ops, sort_keys=True)), "n_queries": len(queries),
        "stores": {}, "writes": [], "cross_store_ids": None, "smoke_failures": None,
    }  # fmt: skip
    for store in STORES:
        rec["stores"][store] = asyncio.run(_recall_leg(base, store, host_label, queries, runs))
    if scratch:
        rec["writes"] = asyncio.run(_write_leg(scratch, host_label, ops))
        rec.update(asyncio.run(_isolation_and_smoke(scratch, host_label)))
    return rec


# -- reconnect -----------------------------------------------------------------------------------


def listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_until(pred: Callable[[], bool], timeout: float, poll: float = 0.5) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(poll)
    return pred()


async def _reconnect_async(base: str, host_label: str, store: str, restart: Callable[[], None],
                           wait_s: float, query: str) -> dict[str, Any]:  # fmt: skip
    u = urlparse(base)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    rec: dict[str, Any] = {"before_ok": False, "retries": None, "ok": False, "detail": ""}
    c = _mcp_client(base, store, host_label)
    session = await c.__aenter__()
    try:
        await _timed(session, "recall", {"query": query})
        rec["before_ok"] = True
        restart()  # blocks until the restart command returns; the server may still be starting
        await asyncio.sleep(0.2)
        if not await asyncio.get_running_loop().run_in_executor(
                None, wait_until, lambda: listening(host, port), wait_s):  # fmt: skip
            rec["detail"] = f"server did not come back on {host}:{port} within {wait_s:g} s"
            return rec
        try:  # the old session: Claude Code sends its next call on it first (attempt 0, not a retry)
            await _timed(session, "recall", {"query": query})
            rec.update(retries=0, ok=True, detail="old session still worked")
            return rec
        except Exception as e:  # noqa: BLE001 - any failure of the stale session is expected
            rec["detail"] = f"old session failed ({type(e).__name__}); "
    finally:
        with contextlib.suppress(Exception):  # the connection is gone with the server
            await c.__aexit__(None, None, None)
    for retry in (1,):  # the gate allows exactly one retry, with a fresh session
        try:
            async with _mcp_client(base, store, host_label) as fresh:
                await _timed(fresh, "recall", {"query": query})
            rec.update(retries=retry, ok=True)
            rec["detail"] += f"fresh session worked on retry {retry}"
        except Exception as e:  # noqa: BLE001
            rec["detail"] += f"retry {retry} failed: {type(e).__name__}: {e}"
    return rec


def run_reconnect(base: str, *, label: str, host_label: str, restart_cmd: str | None,
                  store: str = "personal", query: str = "reconnect check",
                  wait_s: float = 180.0, ask: Callable[[str], Any] = input) -> dict[str, Any]:  # fmt: skip
    """Restart the server under an open session; pass if the next call works within 1 retry.
    Without `restart_cmd` the person restarts the server when asked."""

    def restart() -> None:
        if restart_cmd:
            subprocess.run(restart_cmd, shell=True, check=True)
        else:
            ask("Restart the server now (systemctl --user restart mem), wait until it is "
                "back up, then press Enter: ")  # fmt: skip

    rec = asyncio.run(_reconnect_async(base, host_label, store, restart, wait_s, query))
    return {"kind": "e5-reconnect", "label": label, "base": base, "store": store,
            "restart_cmd": restart_cmd, **rec}  # fmt: skip


# -- comparison ----------------------------------------------------------------------------------


def _diff(a: Any, b: Any, tol: float, path: str, out: list[str]) -> None:
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):  # fmt: skip
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    out.append(f"{path}.{k}: present on one side only")
                else:
                    _diff(a[k], b[k], tol, f"{path}.{k}", out)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out.append(f"{path}: length {len(a)} vs {len(b)}")
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                _diff(x, y, tol, f"{path}[{i}]", out)
        elif a != b:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif abs(a - b) > tol:
        out.append(f"{path}: {a!r} vs {b!r}")


def compare_runs(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-6) -> dict[str, Any]:
    """Parity of two e5-run records: ranked ids exactly equal, p within `tol`, write results equal."""
    id_mismatches, max_dp, notes = 0, 0.0, []
    for store in STORES:
        ra, rb = a["stores"][store]["results"], b["stores"][store]["results"]
        if set(ra) != set(rb):
            id_mismatches += len(set(ra) ^ set(rb))
            notes.append(f"{store}: query sets differ")
        for qid in sorted(set(ra) & set(rb)):
            if ra[qid]["ids"] != rb[qid]["ids"]:
                id_mismatches += 1
                notes.append(f"{store}/{qid}: ids differ")
                continue
            for x, y in zip(ra[qid]["p"], rb[qid]["p"], strict=True):
                if x is None or y is None:
                    if x is not y:
                        id_mismatches += 1
                        notes.append(f"{store}/{qid}: p None vs value")
                else:
                    max_dp = max(max_dp, abs(x - y))
    write_diffs: list[str] = []
    _diff(a["writes"], b["writes"], tol, "writes", write_diffs)
    return {"id_mismatches": id_mismatches, "max_p_diff": max_dp, "write_diffs": write_diffs,
            "notes": notes[:10]}  # fmt: skip


def added_p95(local: dict[str, Any], remote: dict[str, Any]) -> float:
    def all_ms(r: dict[str, Any]) -> list[float]:
        return [v for s in STORES for v in r["stores"][s]["latencies_ms"]]

    return p95(all_ms(remote)) - p95(all_ms(local))


def load_record(path: Path, kind: str) -> dict[str, Any]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("kind") != kind:
        raise ValueError(f"{path}: expected a {kind} record, got {rec.get('kind')!r}")
    return rec
