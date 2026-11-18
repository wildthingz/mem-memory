"""P7 auth controls (architecture-v2 §6 P7, benchmarks-v2 B-H1): bearer tokens on the MCP routes,
the TLS-only endpoint behind `tailscale serve`, rate limits, the audit log, and the optional OAuth
endpoint a claude.ai connector uses. One test (at least) per control; the contract suite over the
authenticated transport is in tests/test_p5_net.py (`localhost-auth`, `tls`)."""

import asyncio
import base64
import hashlib
import json
import os
import stat
import time
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from starlette.testclient import TestClient
from typer.testing import CliRunner

from mem import auth, serve
from mem.cli import app
from mem.config import stub_config
from mem.models_http import ModelHost
from mem.serve import build_app, open_inprocess
from mem.testing import HashEmbedder, OverlapReranker
from tests.helpers import serve_thread

LOCAL = ("127.0.0.1", 41000)
TAILNET_PEER = ("100.64.0.9", 41000)
TS_NAME = "home.tail1234.ts.net"
ACCEPT = {"Accept": "application/json, text/event-stream"}


def _initialize():
    return {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}}}  # fmt: skip


def _recall(query="espresso"):
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"query": query}}}  # fmt: skip


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def ms(tmp_path):
    m = open_inprocess(tmp_path / "p.db", scope="personal", embedder=HashEmbedder(256))
    yield m
    m.close()


@pytest.fixture
def tokens(tmp_path):
    tf = auth.TokenFile(tmp_path / "tokens")
    return tf, tf.add("work-laptop"), tf.add("home")


def _client(ms, access, *, client=LOCAL, authority="127.0.0.1:8765", models=None,
            allowed=serve.DEFAULT_ALLOWED_HOSTS):  # fmt: skip
    a = build_app({"personal": ms}, models=models, allowed_hosts=allowed, access=access)
    return TestClient(a, base_url=f"http://{authority}", client=client)


def _post(c, token=None, body=None, **headers):
    h = dict(ACCEPT, **headers)
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    return c.post("/personal/mcp", json=body or _initialize(), headers=h)


# -- tokens ---------------------------------------------------------------------------------------


def test_token_file_stores_a_hash_not_the_token(tokens):
    tf, work, _ = tokens
    text = tf.path.read_text()
    assert work not in text
    assert hashlib.sha256(work.encode()).hexdigest() in text
    assert stat.S_IMODE(os.stat(tf.path).st_mode) == 0o600


def test_token_verify_names_the_label(tokens):
    tf, work, home = tokens
    t = tf.load()
    assert t.verify(work) == "work-laptop"
    assert t.verify(home) == "home"
    assert t.verify(work + "x") is None
    assert t.verify("") is None


def test_token_revoke_removes_only_that_label(tokens):
    tf, work, home = tokens
    assert tf.revoke("work-laptop") is True
    t = tf.load()
    assert t.verify(work) is None and t.verify(home) == "home"
    assert tf.revoke("work-laptop") is False


def test_a_loaded_token_set_sees_a_revoke_without_a_restart(tokens):
    tf, work, home = tokens
    live = tf.load()
    tf.revoke("work-laptop")
    assert live.verify(work) is None and live.verify(home) == "home"
    new = tf.add("phone")
    assert live.verify(new) == "phone"


def test_token_labels_are_unique_and_simple(tokens):
    tf, _, _ = tokens
    with pytest.raises(auth.MemError, match="already"):
        tf.add("home")
    with pytest.raises(auth.MemError, match="label"):
        tf.add("has space")


def test_token_cli_add_list_revoke(tmp_path):
    path = tmp_path / "tokens"
    r = CliRunner().invoke(app, ["token", "add", "work-laptop", "--file", str(path)])
    assert r.exit_code == 0, r.output
    token = r.output.split()[-1]
    assert token.startswith("mem_") and auth.TokenFile(path).load().verify(token) == "work-laptop"
    r = CliRunner().invoke(app, ["token", "list", "--file", str(path)])
    assert r.exit_code == 0 and "work-laptop" in r.output and token not in r.output
    r = CliRunner().invoke(app, ["token", "revoke", "work-laptop", "--file", str(path)])
    assert r.exit_code == 0
    assert auth.TokenFile(path).load().verify(token) is None


# -- control 1: a missing or bad token is rejected -------------------------------------------------


def test_missing_token_rejected(ms, tokens):
    tf, _, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load())) as c:
        r = _post(c)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


@pytest.mark.parametrize("header", ["Bearer mem_wrong", "Basic dXNlcjpwYXNz", "Bearer", "mem_x"])
def test_bad_token_rejected(ms, tokens, header):
    tf, _, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load())) as c:
        r = c.post("/personal/mcp", json=_initialize(), headers=dict(ACCEPT, Authorization=header))
    assert r.status_code == 401


def test_good_token_accepted_and_revoked_token_rejected(ms, tokens):
    tf, work, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load())) as c:
        assert _post(c, work).status_code == 200
    tf.revoke("work-laptop")
    with _client(ms, auth.Access(tokens=tf.load())) as c:
        assert _post(c, work).status_code == 401


def test_loopback_is_not_exempt_from_the_token(ms, tokens):
    """`tailscale serve` connects from loopback: exempting loopback would exempt the tailnet."""
    tf, _, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load()), client=LOCAL) as c:
        assert _post(c).status_code == 401


# -- control 2: TLS-only endpoint ------------------------------------------------------------------


def test_tls_only_rejects_a_plaintext_tailnet_peer(ms, tokens):
    tf, work, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load(), tls_only=True), client=TAILNET_PEER,
                 authority=f"{TS_NAME}:8765") as c:  # fmt: skip
        r = _post(c, work)
    assert r.status_code == 403 and "https" in r.text


def test_tls_only_rejects_a_plaintext_proxied_request(ms, tokens):
    """A `tailscale serve --http` (plaintext) proxy forwards without X-Forwarded-Proto: https."""
    tf, work, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load(), tls_only=True)) as c:
        assert _post(c, work, **{"X-Forwarded-For": "100.64.0.9"}).status_code == 403
        assert _post(c, work, **{"X-Forwarded-For": "100.64.0.9",
                                 "X-Forwarded-Proto": "http"}).status_code == 403  # fmt: skip


def test_tls_only_accepts_the_https_proxy_and_local_clients(ms, tokens):
    tf, work, _ = tokens
    with _client(ms, auth.Access(tokens=tf.load(), tls_only=True),
                 authority=TS_NAME) as c:  # fmt: skip
        r = _post(c, work, **{"X-Forwarded-For": "100.64.0.9", "X-Forwarded-Proto": "https"})
        assert r.status_code == 200
    with _client(ms, auth.Access(tokens=tf.load(), tls_only=True)) as c:
        assert _post(c, work).status_code == 200  # a local client on 127.0.0.1


def test_tls_only_accepts_the_https_proxy_through_uvicorn(tmp_path, tokens):
    """The real server, not TestClient: uvicorn must not rewrite the peer from X-Forwarded-For
    (its proxy_headers default), or `tailscale serve`'s requests look like tailnet peers."""
    tf, work, _ = tokens
    url, stop = serve_thread(tmp_path / "srv", auth.Access(tokens=tf.load(), tls_only=True))
    try:
        r = httpx2.post(f"{url}/personal/mcp", json=_initialize(), timeout=5, trust_env=False,
                        headers=dict(ACCEPT, Authorization=f"Bearer {work}",
                                     **{"X-Forwarded-For": "100.64.0.9",
                                        "X-Forwarded-Proto": "https"}))  # fmt: skip
        plain = httpx2.post(f"{url}/personal/mcp", json=_initialize(), timeout=5, trust_env=False,
                            headers=dict(ACCEPT, Authorization=f"Bearer {work}",
                                         **{"X-Forwarded-For": "100.64.0.9"}))  # fmt: skip
    finally:
        stop()
    assert (r.status_code, plain.status_code) == (200, 403), r.text


def test_models_routes_refuse_proxied_requests(ms):
    """Behind `tailscale serve` every request comes from loopback: /models must still be local."""
    host = ModelHost(HashEmbedder(256), OverlapReranker(), stub_config())
    with _client(ms, None, models=host) as c:
        assert c.get("/models").status_code == 200
        assert c.get("/models", headers={"X-Forwarded-For": "100.64.0.9"}).status_code == 403
        assert c.get("/models", headers={"Forwarded": "for=100.64.0.9"}).status_code == 403


def _serve_cli(tmp_path, monkeypatch, *flags):
    calls = []
    monkeypatch.setattr(serve, "preflight", lambda stores, cfg: [])
    monkeypatch.setattr(serve, "run_http", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(serve, "tailscale_dns_name", lambda: TS_NAME)
    monkeypatch.setattr(serve, "tailscale_ip", lambda: "100.101.102.103")
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--embedder", "hash",
                                 "--store", f"personal={tmp_path}/p.db", *flags])  # fmt: skip
    return r, calls


def test_tls_mode_binds_loopback_only_and_answers_the_tailnet_name(tmp_path, monkeypatch, tokens):
    tf, _, _ = tokens
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tls", "--token-file", str(tf.path))
    assert r.exit_code == 0, r.output
    kw = calls[0]
    assert kw["tailscale"] is None  # no plaintext tailnet socket
    assert kw["host"] == "127.0.0.1"
    assert serve.host_allowed(TS_NAME, kw["allow"])
    assert kw["access"].tls_only and kw["access"].tokens.verify(tokens[1]) == "work-laptop"


def test_tls_mode_needs_a_token_file(tmp_path, monkeypatch):
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tls")
    assert r.exit_code == 1 and "--token-file" in r.output and calls == []
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tls", "--token-file", str(tmp_path / "none"))
    assert r.exit_code == 1 and "mem token add" in r.output and calls == []


def test_tls_mode_refuses_the_plaintext_tailnet_bind(tmp_path, monkeypatch, tokens):
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tls", "--tailscale",
                          "--token-file", str(tokens[0].path))  # fmt: skip
    assert r.exit_code != 0 and "plaintext" in r.output and calls == []
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tls", "--host", "0.0.0.0",
                          "--token-file", str(tokens[0].path))  # fmt: skip
    assert r.exit_code != 0 and "loopback" in r.output and calls == []


def test_plaintext_tailscale_without_tokens_warns(tmp_path, monkeypatch):
    r, calls = _serve_cli(tmp_path, monkeypatch, "--tailscale")
    assert r.exit_code == 0
    assert "WARNING" in r.output and "--tls" in r.output
    assert calls[0]["access"].tokens is None


def test_token_file_alone_enables_auth_and_audit(tmp_path, monkeypatch, tokens):
    r, calls = _serve_cli(tmp_path, monkeypatch, "--token-file", str(tokens[0].path),
                          "--audit-log", str(tmp_path / "audit.jsonl"), "--rate-limit", "60")  # fmt: skip
    assert r.exit_code == 0, r.output
    a = calls[0]["access"]
    assert a.tokens.verify(tokens[2]) == "home" and not a.tls_only
    assert a.audit.path == tmp_path / "audit.jsonl" and a.limiter.per_minute == 60


# -- control 3: rate limit enforced ----------------------------------------------------------------


def test_rate_limiter_refills_at_the_set_rate():
    clock = Clock()
    rl = auth.RateLimiter(60, clock=clock)  # 1/s, burst 60
    assert all(rl.allow("a")[0] for _ in range(60))
    ok, retry = rl.allow("a")
    assert not ok and retry == pytest.approx(1.0)
    assert rl.allow("b")[0]  # per client
    clock.t += 1.0
    assert rl.allow("a")[0] and not rl.allow("a")[0]
    clock.t += 3600
    assert sum(rl.allow("a")[0] for _ in range(100)) == 60  # never above the burst


def test_rate_limit_enforced(ms, tokens):
    tf, work, home = tokens
    clock = Clock()
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(3, clock=clock))
    with _client(ms, access) as c:
        assert [_post(c, work).status_code for _ in range(4)] == [200, 200, 200, 429]
        r = _post(c, work)
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert _post(c, home).status_code == 200  # another client keeps its own budget
        clock.t += 20  # 3/min: one request back every 20 s
        assert _post(c, work).status_code == 200


def test_rate_limit_stops_token_guessing(ms, tokens):
    tf, _, _ = tokens
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(3, clock=Clock()))
    with _client(ms, access) as c:
        codes = [_post(c, f"mem_guess{i}").status_code for i in range(5)]
    assert codes == [401, 401, 401, 429, 429]


def test_rate_limit_keys_proxied_peers_by_forwarded_address(ms, tokens):
    tf, _, _ = tokens
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(1, clock=Clock()))
    with _client(ms, access) as c:
        assert _post(c, "bad", **{"X-Forwarded-For": "100.64.0.1"}).status_code == 401
        assert _post(c, "bad", **{"X-Forwarded-For": "100.64.0.1"}).status_code == 429
        assert _post(c, "bad", **{"X-Forwarded-For": "100.64.0.2"}).status_code == 401


def test_a_forged_forwarded_entry_changes_neither_the_bucket_nor_the_audit_peer(
    ms, tokens, tmp_path
):
    """`tailscale serve` appends the address it saw to whatever X-Forwarded-For the client sent:
    only that last entry is the peer (MEM2-57 F9)."""
    tf, _, _ = tokens
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(1, clock=Clock()), audit=log)
    with _client(ms, access) as c:
        real = "100.64.0.1"
        assert _post(c, "bad", **{"X-Forwarded-For": real}).status_code == 401
        for forged in ("100.64.0.2", "203.0.113.7, 10.0.0.1"):
            r = _post(c, "bad", **{"X-Forwarded-For": f"{forged}, {real}"})
            assert r.status_code == 429  # still the real peer's (spent) bucket
    peers = [json.loads(x)["peer"] for x in log.path.read_text().splitlines()]
    assert peers == [real] * 3


def test_a_forwarded_header_from_a_non_loopback_peer_is_ignored(ms, tokens):
    tf, _, _ = tokens
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(1, clock=Clock()))
    with _client(ms, access, client=TAILNET_PEER) as c:
        assert _post(c, "bad", **{"X-Forwarded-For": "100.64.0.1"}).status_code == 401
        assert _post(c, "bad", **{"X-Forwarded-For": "100.64.0.2"}).status_code == 429


# -- control 4: audit log complete -----------------------------------------------------------------


def test_audit_log_complete(ms, tokens, tmp_path):
    """Every request to the server gets exactly one line, whatever its outcome, naming who, what
    and the answer; no line holds a token or a tool argument."""
    tf, work, _ = tokens
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(3, clock=Clock()), audit=log,
                         tls_only=True)  # fmt: skip
    allowed = serve.allowed_hosts("127.0.0.1", dns_name=TS_NAME)
    with _client(ms, access, allowed=allowed) as c:
        assert _post(c, work).status_code == 200
        sid = _post(c, work).headers.get("mcp-session-id")
        r = _post(
            c,
            work,
            body=_recall("secret espresso plans"),
            **{"Mcp-Session-Id": sid or "", "MCP-Protocol-Version": "2025-06-18"},
        )
        assert r.status_code == 200
        assert _post(c, "mem_wrong").status_code == 401
        assert _post(c, work).status_code == 429
        assert c.post("/personal/mcp", json=_initialize(),
                      headers=dict(ACCEPT, Host="evil.example")).status_code == 421  # fmt: skip
    with _client(ms, access, client=TAILNET_PEER, allowed=allowed,
                 authority=f"{TS_NAME}:8765") as c:  # fmt: skip
        assert _post(c, work).status_code == 403
    text = log.path.read_text()
    lines = [json.loads(line) for line in text.splitlines()]
    assert [x["status"] for x in lines] == [200, 200, 200, 401, 429, 421, 403]
    for x in lines:
        assert {"ts", "peer", "client", "method", "path", "store", "rpc", "tool", "status",
                "ms"} <= set(x)  # fmt: skip
        assert x["method"] == "POST" and x["path"] == "/personal/mcp" and x["store"] == "personal"
    assert [x["client"] for x in lines] == ["work-laptop"] * 3 + [None, "work-laptop", None, None]
    assert lines[2]["rpc"] == "tools/call" and lines[2]["tool"] == "recall"
    assert lines[0]["rpc"] == "initialize" and lines[0]["tool"] is None
    assert lines[3]["reason"] == "bad token" and lines[4]["reason"] == "rate limited"
    assert lines[6]["peer"] == "100.64.0.9" and lines[6]["reason"] == "not https"
    assert work not in text and "mem_wrong" not in text and "espresso" not in text
    assert stat.S_IMODE(os.stat(log.path).st_mode) == 0o600


def test_audit_log_covers_every_route(ms, tmp_path):
    host = ModelHost(HashEmbedder(256), OverlapReranker(), stub_config())
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    with _client(ms, auth.Access(audit=log), models=host) as c:
        c.get("/models")
        c.get("/nothing-here")
    lines = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [(x["path"], x["status"], x["store"]) for x in lines] == [
        ("/models", 200, None), ("/nothing-here", 404, None)]  # fmt: skip


def test_audit_log_survives_a_non_json_body(ms, tmp_path):
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    with _client(ms, auth.Access(audit=log)) as c:
        c.post("/personal/mcp", content=b"\xff not json", headers=ACCEPT)
    (line,) = log.path.read_text().splitlines()
    assert json.loads(line)["rpc"] is None


def _asgi_post(app, path, headers, chunk=64 * 1024):
    """POST to `app` a 4 MiB chunked body; (status, chunks the app read, response headers)."""
    sent, out = [0], {}

    async def receive():
        sent[0] += 1
        return {"type": "http.request", "body": b"x" * chunk, "more_body": sent[0] < 64}

    async def send(m):
        if m["type"] == "http.response.start":
            out.update(status=m["status"], headers=dict(m["headers"]))

    scope = {"type": "http", "method": "POST", "path": path, "raw_path": path.encode(),
             "query_string": b"", "root_path": "", "scheme": "http", "http_version": "1.1",
             "server": ("127.0.0.1", 8765), "client": LOCAL,
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}  # fmt: skip
    asyncio.run(app(scope, receive, send))
    return out["status"], sent[0], out["headers"]


def test_an_unauthenticated_body_is_not_read(ms, tokens, tmp_path):
    """The token is checked before the body is read: a request without a valid token is refused
    having read nothing, however large its body (MEM2-57 F11)."""
    tf, _, _ = tokens
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    app = build_app({"personal": ms}, access=auth.Access(tokens=tf.load(), audit=log))
    h = {"host": "127.0.0.1:8765", "content-type": "application/json", **ACCEPT}
    assert _asgi_post(app, "/personal/mcp", h)[:2] == (401, 0)
    assert _asgi_post(app, "/personal/mcp", {**h, "authorization": "Bearer mem_x"})[:2] == (401, 0)
    assert [json.loads(x)["status"] for x in log.path.read_text().splitlines()] == [401, 401]


def test_an_authenticated_body_over_the_cap_is_refused(ms, tokens, tmp_path):
    """With a valid token the body is read up to `MAX_BODY`, then refused with 413; one with a
    Content-Length over the cap is refused unread."""
    tf, work, _ = tokens
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    app = build_app({"personal": ms}, access=auth.Access(tokens=tf.load(), audit=log))
    h = {"host": "127.0.0.1:8765", "content-type": "application/json",
         "authorization": f"Bearer {work}", **ACCEPT}  # fmt: skip
    status, read, _ = _asgi_post(app, "/personal/mcp", h)
    assert status == 413 and read == auth.MAX_BODY // (64 * 1024) + 1
    big = {**h, "content-length": str(auth.MAX_BODY + 1)}
    assert _asgi_post(app, "/personal/mcp", big)[:2] == (413, 0)
    lines = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [(x["status"], x["client"], x["reason"]) for x in lines] == [
        (413, "work-laptop", "body too large")] * 2  # fmt: skip


def test_an_oauth_body_over_the_cap_is_refused(ms, tokens, tmp_path):
    tf, _, _ = tokens
    oauth = auth.OwnerOAuth(ISSUER, tf.load(), tmp_path / "oauth.json")
    app = build_app({"personal": ms}, allowed_hosts=["127.0.0.1", TS_NAME],
                    access=auth.Access(tokens=tf.load(), oauth=oauth))  # fmt: skip
    h = {"host": TS_NAME, "content-type": "application/x-www-form-urlencoded"}
    status, read, _ = _asgi_post(app, "/token", h)
    assert status == 413 and read == auth.MAX_BODY // (64 * 1024) + 1


# -- optional: the OAuth endpoint a claude.ai connector uses --------------------------------------


ISSUER = f"https://{TS_NAME}"
CALLBACK = "https://claude.ai/api/mcp/auth_callback"


def _pkce():
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge.decode().rstrip("=")


def _oauth_client(ms, tf, tmp_path, **kw):
    oauth = auth.OwnerOAuth(ISSUER, tf.load(), tmp_path / "oauth.json")
    access = auth.Access(tokens=tf.load(), oauth=oauth, **kw)
    return _client(ms, access, authority=TS_NAME, allowed=["127.0.0.1", TS_NAME]), oauth


def _authorize(c, owner_token):
    reg = c.post("/register", json={"redirect_uris": [CALLBACK], "client_name": "claude.ai",
                                    "token_endpoint_auth_method": "none"})  # fmt: skip
    assert reg.status_code == 201, reg.text
    client_id = reg.json()["client_id"]
    verifier, challenge = _pkce()
    r = c.get("/authorize", params={"response_type": "code", "client_id": client_id,
                                    "redirect_uri": CALLBACK, "code_challenge": challenge,
                                    "code_challenge_method": "S256", "state": "st1"},
              follow_redirects=False)  # fmt: skip
    assert r.status_code == 302, r.text
    consent = urlsplit(r.headers["location"])
    assert consent.path == "/oauth/consent"
    page = c.get(f"{consent.path}?{consent.query}")
    assert page.status_code == 200 and "claude.ai" in page.text
    req = parse_qs(consent.query)["req"][0]
    r = c.post("/oauth/consent", data={"req": req, "owner_token": owner_token},
               follow_redirects=False)  # fmt: skip
    return r, client_id, verifier


def test_oauth_metadata_points_the_connector_at_this_server(ms, tokens, tmp_path):
    c, _ = _oauth_client(ms, tokens[0], tmp_path)
    with c:
        r = _post(c)
        assert r.status_code == 401
        assert "resource_metadata" in r.headers["www-authenticate"]
        prm = c.get("/.well-known/oauth-protected-resource/personal/mcp").json()
        assert prm["authorization_servers"] == [f"{ISSUER}/"] or prm["authorization_servers"] == [
            ISSUER]  # fmt: skip
        md = c.get("/.well-known/oauth-authorization-server").json()
        assert md["registration_endpoint"].endswith("/register")
        assert "S256" in md["code_challenge_methods_supported"]


def test_oauth_flow_issues_a_token_that_opens_mcp(ms, tokens, tmp_path):
    tf, _, home = tokens
    c, _ = _oauth_client(ms, tf, tmp_path)
    with c:
        r, client_id, verifier = _authorize(c, home)
        assert r.status_code == 302
        back = urlsplit(r.headers["location"])
        q = parse_qs(back.query)
        assert f"{back.scheme}://{back.netloc}{back.path}" == CALLBACK and q["state"] == ["st1"]
        tok = c.post("/token", data={"grant_type": "authorization_code", "code": q["code"][0],
                                     "redirect_uri": CALLBACK, "client_id": client_id,
                                     "code_verifier": verifier})  # fmt: skip
        assert tok.status_code == 200, tok.text
        access_token = tok.json()["access_token"]
        assert _post(c, access_token).status_code == 200
        refreshed = c.post("/token", data={"grant_type": "refresh_token", "client_id": client_id,
                                           "refresh_token": tok.json()["refresh_token"]})  # fmt: skip
        assert refreshed.status_code == 200, refreshed.text
        assert _post(c, refreshed.json()["access_token"]).status_code == 200
        assert _post(c, access_token).status_code == 401  # rotated out


def test_oauth_consent_needs_an_owner_token(ms, tokens, tmp_path):
    c, _ = _oauth_client(ms, tokens[0], tmp_path)
    with c:
        r, _, _ = _authorize(c, "mem_not_mine")
    assert r.status_code == 403


def test_oauth_tokens_survive_a_restart_and_are_stored_hashed(ms, tokens, tmp_path):
    tf, _, home = tokens
    c, _ = _oauth_client(ms, tf, tmp_path)
    with c:
        r, client_id, verifier = _authorize(c, home)
        code = parse_qs(urlsplit(r.headers["location"]).query)["code"][0]
        tok = c.post("/token", data={"grant_type": "authorization_code", "code": code,
                                     "redirect_uri": CALLBACK, "client_id": client_id,
                                     "code_verifier": verifier}).json()  # fmt: skip
    assert tok["access_token"] not in (tmp_path / "oauth.json").read_text()
    c2, _ = _oauth_client(ms, tf, tmp_path)
    with c2:
        assert _post(c2, tok["access_token"]).status_code == 200


def test_oauth_audit_names_the_connector(ms, tokens, tmp_path):
    tf, _, home = tokens
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    c, _ = _oauth_client(ms, tf, tmp_path, audit=log)
    with c:
        r, client_id, verifier = _authorize(c, home)
        code = parse_qs(urlsplit(r.headers["location"]).query)["code"][0]
        tok = c.post("/token", data={"grant_type": "authorization_code", "code": code,
                                     "redirect_uri": CALLBACK, "client_id": client_id,
                                     "code_verifier": verifier}).json()  # fmt: skip
        _post(c, tok["access_token"])
    lines = [json.loads(x) for x in log.path.read_text().splitlines()]
    consent = [x for x in lines if x["path"] == "/oauth/consent" and x["method"] == "POST"]
    assert [(x["status"], x["client"]) for x in consent] == [(302, "owner:home")]
    assert lines[-1]["client"] == "oauth:claude.ai" and lines[-1]["status"] == 200
    assert home not in log.path.read_text()


def test_oauth_registration_is_capped(ms, tokens, tmp_path, monkeypatch):
    """Over funnel anyone can call /register: the state file must not grow without bound."""
    monkeypatch.setattr(auth.OwnerOAuth, "MAX_CLIENTS", 2)
    c, _ = _oauth_client(ms, tokens[0], tmp_path)
    with c:
        codes = [c.post("/register", json={"redirect_uris": [CALLBACK]}).status_code
                 for _ in range(3)]  # fmt: skip
    assert codes == [201, 201, 400]


def test_oauth_consent_page_cannot_be_framed(ms, tokens, tmp_path):
    c, _ = _oauth_client(ms, tokens[0], tmp_path)
    with c:
        reg = c.post("/register", json={"redirect_uris": [CALLBACK], "client_name": "x",
                                        "token_endpoint_auth_method": "none"}).json()  # fmt: skip
        r = c.get("/authorize", params={"response_type": "code", "client_id": reg["client_id"],
                                        "redirect_uri": CALLBACK, "code_challenge": _pkce()[1],
                                        "code_challenge_method": "S256"},
                  follow_redirects=False)  # fmt: skip
        page = c.get(r.headers["location"].removeprefix(ISSUER))
    assert page.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_oauth_needs_tls_mode(tmp_path, monkeypatch, tokens):
    r, calls = _serve_cli(tmp_path, monkeypatch, "--oauth", "--token-file", str(tokens[0].path))
    assert r.exit_code != 0 and "--tls" in r.output and calls == []
    r, calls = _serve_cli(tmp_path, monkeypatch, "--oauth", "--tls",
                          "--token-file", str(tokens[0].path))  # fmt: skip
    assert r.exit_code == 0, r.output
    assert calls[0]["access"].oauth.issuer == ISSUER


# -- client config ---------------------------------------------------------------------------------


def test_client_config_over_tls_sends_the_token():
    cmds = serve.client_config(TS_NAME, "work-laptop", tls=True, token=True)
    assert cmds[0] == (
        "claude mcp add --transport http --scope user mem-personal "
        f'https://{TS_NAME}/personal/mcp --header "X-Mem-Host: work-laptop" '
        '--header "Authorization: Bearer $MEM_TOKEN"'
    )
    assert len(cmds) == 2 and f"https://{TS_NAME}/work/mcp" in cmds[1]


@pytest.mark.parametrize(("env", "flag", "sends"), [
    ({"MEM_TOKEN": "mem_secret"}, [], True), ({}, [], False), ({}, ["--token"], True),
    ({"MEM_TOKEN": "mem_secret"}, ["--no-token"], False)])  # fmt: skip
def test_client_config_never_prints_the_token(monkeypatch, env, flag, sends):
    """The lines name `$MEM_TOKEN` for the client's shell to expand: the token is never on a
    command line or in the output (MEM2-57 F12)."""
    monkeypatch.delenv("MEM_TOKEN", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = CliRunner().invoke(app, ["client-config", "--server", TS_NAME, "--host-label", "home",
                                 "--tls", *flag])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "mem_secret" not in r.output
    assert r.output.count('"Authorization: Bearer $MEM_TOKEN"') == (2 if sends else 0)


# -- B-H1: the auth overhead on B-C1 ---------------------------------------------------------------


def _recall_p95_ms(url, token, n=60):
    async def go():
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        http = httpx2.AsyncClient(headers=headers, timeout=30, trust_env=False)
        out = []
        async with Client(streamable_http_client(f"{url}/personal/mcp", http_client=http),
                          mode="legacy") as c:  # fmt: skip
            for i in range(n):
                t0 = time.perf_counter()
                res = await c.call_tool("recall", {"query": f"espresso {i}"})
                out.append((time.perf_counter() - t0) * 1000)
                assert not res.is_error
        return sorted(out)[int(0.95 * n) - 1]

    return asyncio.run(go())


def test_auth_overhead_on_recall_is_under_50ms(tokens, tmp_path):
    """Bearer + rate limit + audit add well under the 50 ms B-C1 budget (TLS is tailscaled's; the
    runbook measures the whole path on the host with `bench auth-overhead`)."""
    tf, work, _ = tokens
    access = auth.Access(tokens=tf.load(), limiter=auth.RateLimiter(100_000),
                         audit=auth.AuditLog(tmp_path / "a.jsonl"))  # fmt: skip
    plain_url, stop_plain = serve_thread(tmp_path / "plain", None)
    guarded_url, stop_guarded = serve_thread(tmp_path / "guarded", access)
    try:
        plain, guarded = _recall_p95_ms(plain_url, None), _recall_p95_ms(guarded_url, work)
    finally:
        stop_plain()
        stop_guarded()
    assert guarded - plain < 50, (plain, guarded)
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) >= 60
