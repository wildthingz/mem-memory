"""mem serve (architecture-v2 §4, §5): in-process tool calls, stdio-less HTTP with two stores."""

import asyncio
import socket
import sqlite3
import threading
import time
from datetime import UTC, datetime

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from mem import contracts
from mem.common import MemError
from mem.serve import MemServer, build_app, open_inprocess
from mem.testing import HashEmbedder

REMEMBER = {"text": "Alice prefers espresso over filter coffee.", "type": "preference",
            "about": ["new:person:Alice"]}  # fmt: skip


def _clock():
    return datetime(2026, 10, 22, 9, 0, tzinfo=UTC)


@pytest.fixture
def server(tmp_path):
    s = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256),
                       tz="UTC", host="test", clock=_clock)  # fmt: skip
    yield s
    s.close()


def test_inprocess_remember_recall_get_round_trip(server):
    text, saved = server.call_tool("remember", REMEMBER)
    assert saved["status"] == "saved" and text.startswith("saved m1")
    text, found = server.call_tool("recall", {"query": "espresso coffee"})
    assert [r["id"] for r in found["rows"]] == ["m1"]
    text, got = server.call_tool("get", {"ids": ["m1"]})
    assert "espresso over filter" in text and got["items"][0]["found"]


def test_unknown_tool_and_bad_arguments_are_memerrors(server):
    with pytest.raises(MemError, match="unknown tool"):
        server.call_tool("nope", {})
    with pytest.raises(MemError, match="bad arguments"):
        server.call_tool("recall", {"query": "x", "bogus": 1})


# -- HTTP: one process, two stores, many sessions ------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    """uvicorn on an ephemeral port in its own thread. The stores are opened in that thread:
    a sqlite connection belongs to the thread (the event loop) that serves it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    servers: dict[str, MemServer] = {}

    def serve():
        for name in ("personal", "work"):
            servers[name] = open_inprocess(tmp_path / f"{name}.db", scope=name, tz="UTC",
                                           embedder=HashEmbedder(256), host="test", clock=_clock)  # fmt: skip
        config = uvicorn.Config(build_app(servers), host="127.0.0.1", port=port, log_level="error")
        uv.append(uvicorn.Server(config))
        uv[0].run()
        for s in servers.values():
            s.close()

    uv: list[uvicorn.Server] = []
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    for _ in range(100):
        if uv and uv[0].started:
            break
        time.sleep(0.05)
    assert uv and uv[0].started
    yield f"http://127.0.0.1:{port}", tmp_path
    uv[0].should_exit = True
    thread.join(5)


def _client(base: str, store: str, host_label: str) -> Client:
    http = httpx2.AsyncClient(headers={"X-Mem-Host": host_label}, timeout=30)
    # "legacy" = the handshake protocol with `Mcp-Session-Id`; the 2026-07-28 protocol has no sessions
    return Client(streamable_http_client(f"{base}/{store}/mcp", http_client=http), mode="legacy")


def test_two_sessions_share_one_server(http_server):
    base, tmp = http_server

    async def scenario():
        async with _client(base, "personal", "home") as a, _client(base, "personal", "laptop") as b:
            saved = await a.call_tool("remember", REMEMBER)
            seen = await b.call_tool("recall", {"query": "espresso"})
            return saved.structured_content, seen.structured_content

    saved, seen = asyncio.run(scenario())
    assert saved["status"] == "saved"
    assert [r["id"] for r in seen["rows"]] == ["m1"]  # B sees A's write: one process, one store
    db = sqlite3.connect(tmp / "personal.db")
    events = db.execute("SELECT tool, session, host FROM events ORDER BY id").fetchall()
    assert [e[0] for e in events] == ["remember", "recall"]
    assert events[0][1] and events[1][1] and events[0][1] != events[1][1]  # MCP session ids differ
    assert [e[2] for e in events] == ["home", "laptop"]  # X-Mem-Host header


def test_stores_are_mounted_separately(http_server):
    base, _ = http_server

    async def scenario():
        async with _client(base, "personal", "home") as p, _client(base, "work", "home") as w:
            await p.call_tool("remember", REMEMBER)
            return (await w.call_tool("recall", {"query": "espresso"})).structured_content

    assert asyncio.run(scenario())["rows"] == []


def test_wire_tool_list_equals_frozen_contract_and_errors_are_tool_errors(http_server):
    base, _ = http_server

    async def scenario():
        async with _client(base, "personal", "home") as c:
            tools = (await c.list_tools()).tools
            bad = await c.call_tool("remember", {"text": "", "type": "fact", "about": ["x"]})
            return tools, bad

    tools, bad = asyncio.run(scenario())
    frozen = contracts.mcp_tools()["tools"]
    assert [(t.name, t.input_schema, t.output_schema) for t in tools] == [
        (t["name"], t["inputSchema"], t["structuredContent"]) for t in frozen
    ]
    assert bad.is_error and "text is required" in bad.content[0].text
