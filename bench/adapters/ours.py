"""Benchmark adapter for the production code (benchmarks-v2 §0.3). Private, not MCP.

It calls the in-process entry point that the MCP tools themselves call. The seam is ONE function,
assumed to be provided by `mem.serve` (p1-serve):

    mem.serve.open_inprocess(db_path, *, scope, embedder, tz, host, clock) -> server
    server.call_tool(name, args) -> (text, structured)      # same payloads as the MCP tool result
    server.close()
    mem.serve.list_tool_schemas() -> list[dict]             # the tools as advertised

`clock` is a zero-argument callable returning an aware UTC datetime (the injected op date).
`tz` is an IANA zone name, the "configured local time zone" of the schema comment on `nodes`.
If p1-serve names these differently, only this file changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.suites.base import BackendUnavailable, Client, ToolResult

FIXTURE_TZ = "Europe/Amsterdam"
FIXTURE_START = "2026-10-22T09:00:00Z"


def parse_utc(iso: str) -> datetime:
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


class _Clock:
    def __init__(self, now: str) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return parse_utc(self.now)


class OursClient:
    def __init__(self, server: Any, db_path: Path, clock: _Clock) -> None:
        self._server, self.db_path, self._clock = server, db_path, clock

    def call(self, tool: str, args: dict[str, Any], *, now: str | None = None) -> ToolResult:
        if now is not None:
            self._clock.now = now
        text, structured = self._server.call_tool(tool, args)
        return ToolResult(text=text, structured=structured)

    def close(self) -> None:
        self._server.close()


def open_client(
    db_path: Path, *, scope: str = "personal", tz: str = FIXTURE_TZ, now: str = FIXTURE_START
) -> Client:
    try:
        from mem.serve import open_inprocess
    except ImportError as e:
        raise BackendUnavailable(f"mem.serve.open_inprocess not available yet: {e}") from e
    from mem.testing import HashEmbedder

    clock = _Clock(now)
    server = open_inprocess(
        db_path, scope=scope, embedder=HashEmbedder(256), tz=tz, host="bench", clock=clock
    )
    return OursClient(server, Path(db_path), clock)


def list_tool_schemas() -> list[dict[str, Any]]:
    try:
        from mem.serve import list_tool_schemas as live
    except ImportError as e:
        raise BackendUnavailable(f"mem.serve.list_tool_schemas not available yet: {e}") from e
    return live()
