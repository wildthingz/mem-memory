"""P5 (architecture-v2 §5, §6): the Tailscale bind, host labels, the work-machine client config,
and the contract suite over in-memory / localhost / Tailscale.

The Tailscale case runs against a server you started on the tailnet and is skipped unless
MEM_TAILSCALE_URL names it (e.g. http://home.tailnet.ts.net:8765). Point it at a THROWAWAY
`mem serve --tailscale --embedder hash` with empty stores: the suite writes to it.

P7: `localhost-auth` runs the suite with every P7 control on (bearer token, TLS-only, rate limit,
audit log); `tls` runs it through `tailscale serve` https with a token, skipped unless
MEM_TLS_URL (e.g. https://home.tailnet.ts.net:8443) and MEM_TOKEN are set (runbook p7-auth.md).
"""

import asyncio
import os
import socket
import sqlite3
import subprocess
import threading
import time
from datetime import UTC, datetime

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from typer.testing import CliRunner

from mem import auth, serve
from mem.cli import app
from mem.serve import build_app, open_inprocess
from mem.store import Store
from mem.testing import HashEmbedder

TAILSCALE_URL = os.environ.get("MEM_TAILSCALE_URL")
TLS_URL, TLS_TOKEN = os.environ.get("MEM_TLS_URL"), os.environ.get("MEM_TOKEN")


def _clock():
    return datetime(2026, 10, 22, 9, 0, tzinfo=UTC)


def _remember(text, name="Alice"):
    return {"text": text, "type": "preference", "about": [f"new:person:{name}"]}


# -- servers -------------------------------------------------------------------------------------


def _start_http(tmp_path, host="127.0.0.1", access=None):
    with socket.socket() as sock:
        sock.bind((host, 0))
        port = sock.getsockname()[1]
    uv: list[uvicorn.Server] = []

    def run():
        servers = {
            n: open_inprocess(tmp_path / f"{n}.db", scope=n, tz="UTC", embedder=HashEmbedder(256),
                              host="srv", clock=_clock)
            for n in ("personal", "work")
        }  # fmt: skip
        uv.append(uvicorn.Server(uvicorn.Config(build_app(servers, access=access), host=host,
                                                port=port, log_level="error")))  # fmt: skip
        uv[0].run()
        for s in servers.values():
            s.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv and uv[0].started:
            return f"http://{host}:{port}", lambda: (setattr(uv[0], "should_exit", True),
                                                      thread.join(5))  # fmt: skip
        time.sleep(0.05)
    raise AssertionError("server did not start")


class Backend:
    """`call(store, tool, args, host=)` -> structured result, however the server is reached."""

    def __init__(self, call):
        self.call = call


@pytest.fixture(params=["in-memory", "localhost", "tailscale", "localhost-auth", "tls"])
def backend(request, tmp_path):
    if request.param == "in-memory":
        servers = {n: open_inprocess(tmp_path / f"{n}.db", scope=n, embedder=HashEmbedder(256),
                                     tz="UTC", host="srv", clock=_clock)
                   for n in ("personal", "work")}  # fmt: skip
        yield Backend(lambda store, tool, args, host="home": servers[store].call_tool(
            tool, args, host=host)[1])  # fmt: skip
        for s in servers.values():
            s.close()
        return
    token = None
    if request.param == "localhost":
        base, stop = _start_http(tmp_path)
    elif request.param == "localhost-auth":
        tf = auth.TokenFile(tmp_path / "tokens")
        token = tf.add("contract")
        log = auth.AuditLog(tmp_path / "audit.jsonl")
        base, stop = _start_http(tmp_path, access=auth.Access(
            tokens=tf.load(), tls_only=True, limiter=auth.RateLimiter(600), audit=log))  # fmt: skip
    elif request.param == "tls":
        if not (TLS_URL and TLS_TOKEN):
            pytest.skip("MEM_TLS_URL / MEM_TOKEN not set: no tailscale serve endpoint to test")
        base, stop, token = TLS_URL.rstrip("/"), lambda: None, TLS_TOKEN
    else:
        if not TAILSCALE_URL:
            pytest.skip("MEM_TAILSCALE_URL not set: no Tailscale server to test against")
        base, stop = TAILSCALE_URL.rstrip("/"), lambda: None

    def call(store, tool, args, host="home"):
        async def go():
            headers = {"X-Mem-Host": host} | ({"Authorization": f"Bearer {token}"} if token else {})
            http = httpx2.AsyncClient(headers=headers, timeout=30)
            c = Client(streamable_http_client(f"{base}/{store}/mcp", http_client=http),
                       mode="legacy")  # fmt: skip
            async with c:
                return (await c.call_tool(tool, args)).structured_content

        return asyncio.run(go())

    yield Backend(call)
    stop()


# -- the contract suite, parametrised over the transport -----------------------------------------


def test_contract_remember_recall_get_round_trip(backend):
    unique = f"Zelda{time.time_ns()}"  # the Tailscale server is shared across runs
    saved = backend.call("personal", "remember", _remember(f"{unique} prefers espresso.", unique))
    assert saved["status"] == "saved"
    found = backend.call("personal", "recall", {"query": f"{unique} espresso"})
    assert saved["id"] in [r["id"] for r in found["rows"]]
    got = backend.call("personal", "get", {"ids": [saved["id"]]})
    assert got["items"][0]["found"] and unique in got["items"][0]["text"]


def test_contract_stores_isolated(backend):
    unique = f"Quux{time.time_ns()}"
    backend.call("personal", "remember", _remember(f"{unique} likes tea.", unique))
    assert backend.call("work", "recall", {"query": f"{unique} tea"})["rows"] == []


# -- test_stores_isolated / test_concurrent_writers_wal / test_host_label_from_header ------------


def test_stores_isolated(tmp_path):
    """personal and work are separate files: a write, a forget or a link in one is invisible
    to the other, and neither file holds the other's rows."""
    p = open_inprocess(tmp_path / "personal.db", scope="personal", embedder=HashEmbedder(256))
    w = open_inprocess(tmp_path / "work.db", scope="work", embedder=HashEmbedder(256))
    try:
        _, pm = p.call_tool("remember", _remember("Alice prefers espresso."))
        _, wm = w.call_tool("remember", _remember("Alice reviews on Fridays.", "Colleague"))
        assert pm["id"] == wm["id"] == "m1"  # independent id spaces
        w.call_tool("forget", {"id": "m1", "reason": "test"})
        assert p.call_tool("get", {"ids": ["m1"]})[1]["items"][0]["found"]
        assert [r["id"] for r in p.call_tool("recall", {"query": "espresso"})[1]["rows"]] == ["m1"]
        assert w.call_tool("recall", {"query": "espresso"})[1]["rows"] == []
    finally:
        p.close()
        w.close()
    for name, other in (("personal", "Fridays"), ("work", "espresso")):
        db = sqlite3.connect(tmp_path / f"{name}.db")
        assert db.execute("SELECT count(*) FROM chunks WHERE text LIKE ?",
                          (f"%{other}%",)).fetchone() == (0,)  # fmt: skip
        db.close()


def test_concurrent_writers_wal(tmp_path):
    """Four writers, each with its own connection to one store file, all succeed: WAL plus the
    busy timeout serialise them; no `database is locked`, no lost or duplicated memory."""
    path = tmp_path / "personal.db"
    Store.open(path, HashEmbedder(256)).close()
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    db.close()
    errors: list[BaseException] = []

    def writer(n):
        try:
            ms = open_inprocess(path, scope="personal", embedder=HashEmbedder(256),
                                host=f"w{n}", session=f"s{n}")  # fmt: skip
            for i in range(5):
                ms.call_tool("remember", _remember(f"Person{n} fact number {i}.", f"Person{n}"))
            ms.close()
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert errors == []
    db = sqlite3.connect(path)
    assert db.execute("SELECT count(*) FROM events WHERE tool='remember'").fetchone() == (20,)
    assert db.execute("SELECT count(DISTINCT host) FROM events").fetchone() == (4,)
    db.close()


def test_host_label_from_header(tmp_path):
    """The X-Mem-Host header names the host; an HTTP client without one is `unknown`, not the
    server's --host-label (MEM2-41 F6: an unlabeled peer is not the home machine)."""
    base, stop = _start_http(tmp_path)
    try:

        async def go():
            for text, headers in (("Alpha one.", {"X-Mem-Host": "work-laptop"}),
                                  ("Beta two.", {})):  # fmt: skip
                http = httpx2.AsyncClient(headers=headers, timeout=30)
                async with Client(streamable_http_client(f"{base}/work/mcp", http_client=http),
                                  mode="legacy") as c:  # fmt: skip
                    await c.call_tool("remember", _remember(text, text.split()[0]))

        asyncio.run(go())
    finally:
        stop()
    db = sqlite3.connect(tmp_path / "work.db")
    hosts = db.execute("SELECT host FROM events WHERE tool='remember' ORDER BY id").fetchall()
    assert hosts == [("work-laptop",), ("unknown",)]


@pytest.mark.parametrize(("label", "stored"), [
    ("work-laptop", "work-laptop"), ("a.b_c-9", "a.b_c-9"), ("x" * 64, "x" * 64),
    ("x" * 65, "unknown"), ("work laptop", "unknown"), ("<script>", "unknown"), ("", "unknown"),
    ("laptop\u00e9", "unknown"), (None, "unknown")])  # fmt: skip
def test_host_label_is_capped_and_restricted(label, stored):
    """X-Mem-Host is free client text: events.host keeps it only as 1-64 of [A-Za-z0-9._-],
    anything else is `unknown` (MEM2-41 F8)."""
    assert serve.clean_host(label) == stored


def test_a_bad_host_header_is_stored_as_unknown(tmp_path):
    base, stop = _start_http(tmp_path)
    try:

        async def go():
            for text, label in (("Alpha one.", "x" * 65), ("Beta two.", "a b;c"),
                                ("Gamma three.", "ok-1")):  # fmt: skip
                http = httpx2.AsyncClient(headers={"X-Mem-Host": label}, timeout=30)
                async with Client(streamable_http_client(f"{base}/work/mcp", http_client=http),
                                  mode="legacy") as c:  # fmt: skip
                    await c.call_tool("remember", _remember(text, text.split()[0]))

        asyncio.run(go())
    finally:
        stop()
    db = sqlite3.connect(tmp_path / "work.db")
    hosts = db.execute("SELECT host FROM events WHERE tool='remember' ORDER BY id").fetchall()
    assert hosts == [("unknown",), ("unknown",), ("ok-1",)]


def test_client_config_refuses_a_bad_host_label():
    r = CliRunner().invoke(app, ["client-config", "--server", "home.tailnet.ts.net",
                                 "--host-label", "work laptop"])  # fmt: skip
    assert r.exit_code != 0 and "--host-label" in r.output


def test_stdio_events_keep_the_host_label(tmp_path):
    """stdio has no request headers: `_origin` defers to the store's --host-label."""

    class Ctx:
        request = None

    assert serve._origin(Ctx()) == (None, None)
    ms = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256),
                        host="home")  # fmt: skip
    try:
        ms.call_tool("remember", _remember("Gamma three."), host=None)
        assert ms.store.conn.execute("SELECT host FROM events").fetchall() == [("home",)]
    finally:
        ms.close()


# -- the Tailscale bind --------------------------------------------------------------------------


def _fake_tailscale(stdout="100.101.102.103\n", code=0):
    def run(cmd, **kw):
        assert cmd == ["tailscale", "ip", "-4"]
        return subprocess.CompletedProcess(cmd, code, stdout, "")

    return run


def test_tailscale_ip_is_the_first_ipv4_line():
    assert serve.tailscale_ip(_fake_tailscale("100.101.102.103\n")) == "100.101.102.103"


@pytest.mark.parametrize("run", [_fake_tailscale("", 1), _fake_tailscale("not an ip\n")])
def test_tailscale_ip_errors_name_the_fix(run):
    with pytest.raises(serve.MemError, match="tailscale up"):
        serve.tailscale_ip(run)


def test_tailscale_ip_when_tailscale_is_not_installed():
    def run(cmd, **kw):
        raise FileNotFoundError("tailscale")

    with pytest.raises(serve.MemError, match="tailscale"):
        serve.tailscale_ip(run)


def test_bind_addresses_always_keep_loopback():
    assert serve.bind_addresses("127.0.0.1", None) == ["127.0.0.1"]
    assert serve.bind_addresses("127.0.0.1", "100.1.2.3") == ["127.0.0.1", "100.1.2.3"]
    assert serve.bind_addresses("192.168.1.5", "100.1.2.3") == ["192.168.1.5", "100.1.2.3"]


def test_models_routes_stay_local_on_the_tailnet_bind():
    """The /models routes 403 a non-loopback peer (a tailnet address is not loopback)."""
    from starlette.testclient import TestClient

    from mem.config import stub_config
    from mem.models_http import ModelHost
    from mem.testing import OverlapReranker

    host = ModelHost(HashEmbedder(256), OverlapReranker(), stub_config())
    c = TestClient(build_app({}, models=host), client=("100.64.0.9", 5000))
    assert c.get("/models").status_code == 403


# -- the work-machine client config --------------------------------------------------------------


def test_client_config_commands():
    cmds = serve.client_config("home.tailnet.ts.net", "work-laptop", port=8765)
    assert cmds == [
        (
            "claude mcp add --transport http --scope user mem-personal "
            'http://home.tailnet.ts.net:8765/personal/mcp --header "X-Mem-Host: work-laptop"'
        ),
        (
            "claude mcp add --transport http --scope user mem-work "
            'http://home.tailnet.ts.net:8765/work/mcp --header "X-Mem-Host: work-laptop"'
        ),
    ]


def test_client_config_cli():
    r = CliRunner().invoke(app, ["client-config", "--server", "home.tailnet.ts.net",
                                 "--host-label", "work-laptop"])  # fmt: skip
    assert r.exit_code == 0
    assert "http://home.tailnet.ts.net:8765/work/mcp" in r.output
    assert r.output.count("X-Mem-Host: work-laptop") == 2


def test_require_tailscale_fails_cleanly_without_tailscale(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no tailscale binary
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--require-tailscale", "--embedder", "hash",
                                 "--store", f"personal={tmp_path}/p.db"])  # fmt: skip
    assert r.exit_code == 1
    assert "tailscale" in r.output and "--require-tailscale" in r.output
    assert not (tmp_path / "p.db").exists()


def _serve_cli(tmp_path, monkeypatch, *flags):
    """`mem serve --http ...` up to run_http, whose arguments it returns (no models, no gate)."""
    calls = []
    monkeypatch.setattr(serve, "preflight", lambda stores, cfg: [])
    monkeypatch.setattr(serve, "run_http", lambda *a, **kw: calls.append(kw))
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--embedder", "hash",
                                 "--store", f"personal={tmp_path}/p.db", *flags])  # fmt: skip
    return r, calls


def test_tailscale_down_serves_loopback_only_with_a_warning(tmp_path, monkeypatch):
    """MEM2-41 F3: Tailscale not up (not installed here) must not take local use down."""
    monkeypatch.setenv("PATH", str(tmp_path))  # no tailscale binary
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tailscale")
    assert r.exit_code == 0, r.output
    assert r.output.count("WARNING") == 1 and "serving on 127.0.0.1 only" in r.output
    assert "tailscale up" in r.output  # the fix
    assert calls[0]["tailscale"] is None
    assert not serve.host_allowed("home.tail1234.ts.net", calls[0]["allow"])


def test_tailscale_up_allows_its_address_and_name(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "tailscale_ip", lambda: "100.101.102.103")
    monkeypatch.setattr(serve, "tailscale_dns_name", lambda: "home.tail1234.ts.net")
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tailscale", "--allow-host", "mem.lan")
    assert r.exit_code == 0, r.output
    assert calls[0]["tailscale"] == "100.101.102.103"
    assert calls[0]["allow"] == ["127.0.0.1", "localhost", "::1", "100.101.102.103",
                                 "home.tail1234.ts.net", "mem.lan"]  # fmt: skip


def test_tailscale_dns_name_from_status_json():
    def run(stdout, code=0):
        return lambda cmd, **kw: subprocess.CompletedProcess(cmd, code, stdout, "")

    assert serve.tailscale_dns_name(run('{"Self": {"DNSName": "home.tail1234.ts.net."}}')) == (
        "home.tail1234.ts.net"
    )
    assert serve.tailscale_dns_name(run("", 1)) is None
    assert serve.tailscale_dns_name(run("not json")) is None


def _status(ms, allowed, authority, origin=None):
    from starlette.testclient import TestClient

    app_ = build_app({"personal": ms}, allowed_hosts=allowed)  # one lifespan per app
    headers = {"Accept": "application/json, text/event-stream"}
    if origin:
        headers["Origin"] = origin
    with TestClient(app_, base_url=f"http://{authority}") as c:
        return c.post("/personal/mcp", json=_initialize(), headers=headers).status_code


def test_serve_without_tailscale_stays_loopback_only(tmp_path):
    """MEM2-41 F1: `mem serve`'s allow list without --tailscale is loopback (and --host) only."""
    ms = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256))
    try:
        local = serve.allowed_hosts("127.0.0.1")
        assert _status(ms, local, "127.0.0.1:8765") == 200
        assert _status(ms, local, "localhost:8765") == 200
        assert _status(ms, local, "[::1]:8765") == 200
        assert _status(ms, local, "localhost:8765", origin="http://localhost:3000") == 200
        assert _status(ms, local, "home.tail1234.ts.net:8765") == 421
        assert _status(ms, local, "100.101.102.103:8765") == 421
        assert _status(ms, local, "127.0.0.1:8765", origin="http://evil.example") == 403
        ts = serve.allowed_hosts("127.0.0.1", "100.101.102.103", "home.tail1234.ts.net")
        assert _status(ms, ts, "home.tail1234.ts.net:8765") == 200
        assert _status(ms, ts, "home.tail1234.ts.net.:8765") == 200  # FQDN form
        assert _status(ms, ts, "100.101.102.103:8765") == 200
        assert _status(ms, ts, "other.tail1234.ts.net:8765") == 421
        assert _status(ms, ts, "evil.example:8765") == 421
    finally:
        ms.close()


def test_one_server_answers_on_every_bound_address(tmp_path):
    """run_http's socket set (127.0.0.2 stands in for the tailnet address): both answer."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sockets = [serve._listen(a, port) for a in serve.bind_addresses("127.0.0.1", "127.0.0.2")]
    uv: list[uvicorn.Server] = []

    def run():
        ms = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256))
        uv.append(uvicorn.Server(uvicorn.Config(build_app({"personal": ms}), log_level="error")))
        uv[0].run(sockets=sockets)
        ms.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if uv and uv[0].started:
                break
            time.sleep(0.05)
        for addr in ("127.0.0.1", "127.0.0.2"):  # MEM2-41 F4: a handshake, not just "< 500"
            r = httpx2.post(f"http://{addr}:{port}/personal/mcp", json=_initialize(),
                            headers={"Accept": "application/json, text/event-stream"}, timeout=5,
                            trust_env=False)  # fmt: skip
            assert r.status_code == 200, f"{addr}: {r.status_code} {r.text[:80]}"
    finally:
        uv[0].should_exit = True
        thread.join(5)


# -- review proofs (MEM2-41): what the tailnet client actually sends ------------------------------


def _initialize():
    return {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "work-laptop", "version": "1"}}}  # fmt: skip


@pytest.mark.parametrize("authority", ["home.tail1234.ts.net:8765", "100.101.102.103:8765"])
def test_mcp_accepts_the_tailnet_host_header(tmp_path, authority):
    """MEM2-41 F1. The work machine connects to http://home.<tailnet>.ts.net:8765/<store>/mcp (or
    the 100.x address), so its requests carry that Host header. build_app leaves the MCP SDK's
    DNS-rebinding guard at its loopback-only default, which answers 421 Invalid Host header: no
    session can be opened over Tailscale, whatever the socket is bound to."""
    from starlette.testclient import TestClient

    ms = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256))
    try:
        with TestClient(build_app({"personal": ms}), base_url=f"http://{authority}") as c:
            r = c.post("/personal/mcp", json=_initialize(),
                       headers={"Accept": "application/json, text/event-stream"})  # fmt: skip
        assert r.status_code == 200, f"{r.status_code} {r.text[:80]}"
    finally:
        ms.close()


def test_bound_address_test_would_catch_a_rejected_host(tmp_path):
    """MEM2-41 F1 (the missing assertion of test_one_server_answers_on_every_bound_address): a
    client reaching the second bound address must complete an MCP handshake there, not merely
    get a non-5xx answer (421 and 403 are < 500)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sockets = [serve._listen(a, port) for a in serve.bind_addresses("127.0.0.1", "127.0.0.2")]
    uv: list[uvicorn.Server] = []

    def run():
        ms = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256))
        uv.append(uvicorn.Server(uvicorn.Config(build_app({"personal": ms}), log_level="error")))
        uv[0].run(sockets=sockets)
        ms.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if uv and uv[0].started:
                break
            time.sleep(0.05)
        # the same socket set; the Host header names the non-loopback address, as a peer's would
        r = httpx2.post(f"http://127.0.0.1:{port}/personal/mcp", json=_initialize(),
                        headers={"Accept": "application/json, text/event-stream",
                                 "Host": f"127.0.0.2:{port}"}, timeout=5)  # fmt: skip
        assert r.status_code == 200, f"{r.status_code} {r.text[:80]}"
    finally:
        uv[0].should_exit = True
        thread.join(5)
