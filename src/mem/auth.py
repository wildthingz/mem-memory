"""P7 access controls on `mem serve --http` (architecture-v2 §6 P7, benchmarks-v2 B-H1).

- **Bearer tokens.** `mem token add <label>` prints a token once and keeps only its SHA-256 in the
  token file (`TokenFile`, mode 0600). With `--token-file`, every MCP request needs
  `Authorization: Bearer <token>`, loopback included: `tailscale serve` connects from loopback,
  so exempting loopback would exempt the tailnet. The label names the client in the audit log.
  The file is re-read when it changes, so `mem token revoke` takes effect without a restart.
  The body is read only after the token check, and at most `MAX_BODY` bytes of it (else 413).
- **TLS only** (`--tls`). The tailnet reaches mem only through `tailscale serve`'s HTTPS proxy:
  mem binds loopback only, and `AccessGuard` refuses any request that is neither a direct
  loopback client nor forwarded by a loopback proxy with `X-Forwarded-Proto: https`.
- **Rate limits.** A token bucket per client (`RateLimiter`): `per_minute` requests a minute,
  bursts up to `per_minute`. A request without a valid token spends from its peer address's
  bucket first, so token guessing is limited too. Over the limit: 429 with Retry-After.
- **Audit log.** `Audited` writes one JSON line per HTTP request, on every route and whatever
  the answer (200, 401, 403, 421, 429, ...): time, peer, client label, method, path, store,
  JSON-RPC method and tool name, status, ms and the reason of a refusal. Never a token, a query
  string or a tool argument.
- **OAuth** (`--oauth`, optional, for a claude.ai connector). `OwnerOAuth` is an OAuth 2.1
  authorization server (dynamic client registration, PKCE, refresh rotation, revocation) on the
  MCP SDK's routes. Its consent page approves a client only when the person pastes one of their
  own mem tokens. Issued tokens are stored hashed in a state file and survive restarts.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import ipaddress
import json
import math
import os
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mem.common import MemError

TOKEN_PREFIX = "mem_"
DEFAULT_TOKEN_FILE = Path("~/.config/mem/tokens")
DEFAULT_AUDIT_LOG = Path("~/.local/state/mem/audit.jsonl")
DEFAULT_OAUTH_STATE = Path("~/.config/mem/oauth.json")
DEFAULT_RATE_LIMIT = 600  # requests per minute per client; Claude Code makes a few per turn
_LABEL = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FORWARDED = ("x-forwarded-for", "forwarded", "x-forwarded-proto")
_NOTE = "mem.audit"  # scope key: what the guards tell the audit log about a request


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _write_private(path: Path, text: str) -> None:
    """Replace `path` with `text`, mode 0600, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---- tokens -------------------------------------------------------------------------------------


class Tokens:
    """label -> SHA-256 of its token. With `source`, re-read whenever that file changes."""

    def __init__(self, entries: Mapping[str, str], source: TokenFile | None = None) -> None:
        self._entries, self._source = dict(entries), source
        self._stamp = source.stamp() if source else None

    def __len__(self) -> int:
        return len(self._refresh())

    def _refresh(self) -> dict[str, str]:
        if self._source is not None and (stamp := self._source.stamp()) != self._stamp:
            self._entries, self._stamp = self._source.read(), stamp
        return self._entries

    def verify(self, token: str) -> str | None:
        """The label of `token`, or None. Compares every entry in constant time."""
        if not token:
            return None
        d, found = _digest(token), None
        for label, h in self._refresh().items():
            if hmac.compare_digest(d, h):
                found = label
        return found


class TokenFile:
    """`<label> sha256:<hex>` per line; `#` comments. Written 0600."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def stamp(self) -> tuple[int, int] | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return st.st_mtime_ns, st.st_size

    def read(self) -> dict[str, str]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {}
        out = {}
        for n, line in enumerate(lines, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2 or not parts[1].startswith("sha256:"):
                raise MemError(f"{self.path}:{n}: expected `<label> sha256:<hex>`")
            out[parts[0]] = parts[1].removeprefix("sha256:")
        return out

    def load(self) -> Tokens:
        return Tokens(self.read(), source=self)

    def labels(self) -> list[str]:
        return list(self.read())

    def _save(self, entries: Mapping[str, str]) -> None:
        body = "".join(f"{label} sha256:{h}\n" for label, h in entries.items())
        _write_private(self.path, "# mem tokens: label sha256(token); `mem token add`\n" + body)

    def add(self, label: str) -> str:
        """A new token for `label`; only its hash is kept, so this is the one time it is shown."""
        if not _LABEL.match(label):
            raise MemError(f"token label {label!r}: use 1-64 of A-Z a-z 0-9 . _ -")
        entries = self.read()
        if label in entries:
            raise MemError(f"{label!r} already has a token: `mem token revoke {label}` first")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        self._save({**entries, label: _digest(token)})
        return token

    def revoke(self, label: str) -> bool:
        entries = self.read()
        if entries.pop(label, None) is None:
            return False
        self._save(entries)
        return True


# ---- rate limits --------------------------------------------------------------------------------


class RateLimiter:
    """A token bucket per key: `per_minute` a minute, bursts up to `per_minute`."""

    MAX_KEYS = 10_000

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.per_minute, self._rate, self._clock = per_minute, per_minute / 60.0, clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        """(allowed, seconds until the next request would be)."""
        now = self._clock()
        level, last = self._buckets.get(key, (float(self.per_minute), now))
        level = min(float(self.per_minute), level + (now - last) * self._rate)
        if len(self._buckets) >= self.MAX_KEYS and key not in self._buckets:
            self._prune(now)
        if level >= 1 - 1e-9:
            self._buckets[key] = (level - 1, now)
            return True, 0.0
        self._buckets[key] = (level, now)
        return False, (1 - level) / self._rate

    def _prune(self, now: float) -> None:
        full = 60.0  # a bucket idle for a minute is full again: forgetting it changes nothing
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < full}


# ---- audit log ----------------------------------------------------------------------------------


class AuditLog:
    """Append-only JSON lines, mode 0600. Opened on the first write."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._fd: int | None = None

    def write(self, record: Mapping[str, Any]) -> None:
        if self._fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        os.write(self._fd, line.encode())  # one O_APPEND write per line

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def note(scope: Scope, **fields: Any) -> None:
    """Tell the audit log about this request (`client`, `reason`)."""
    if (n := scope.get(_NOTE)) is not None:
        n.update(fields)


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def proxied(headers: Headers) -> bool:
    return any(h in headers for h in _FORWARDED)


def peer(scope: Scope) -> str | None:
    client = scope.get("client")
    return client[0] if client else None


def remote(scope: Scope, headers: Headers) -> str | None:
    """Who is asking: the peer, or for a loopback proxy (`tailscale serve`) the X-Forwarded-For
    entry that proxy appended, the last one; the entries before it are whatever the client sent."""
    p = peer(scope)
    if _is_loopback(p) and (xff := headers.get("x-forwarded-for")):
        return xff.split(",")[-1].strip() or p
    return p


MAX_BODY = 1 << 20  # an MCP or OAuth request body (remember's text is at most 1200 chars)


def _too_long(headers: Headers, limit: int) -> bool:
    try:
        return int(headers.get("content-length", "0")) > limit
    except ValueError:
        return False


async def read_capped(receive: Receive, limit: int = MAX_BODY) -> Receive | None:
    """Read the request body, stopping once it passes `limit` bytes (None: the rest is left
    unread); else a `receive` that replays it to the app."""
    messages: list[Message] = []
    size, more = 0, True
    while more:
        m = await receive()
        messages.append(m)
        if m["type"] != "http.request":
            break
        size += len(m.get("body", b""))
        if size > limit:
            return None
        more = m.get("more_body", False)
    pending = iter(messages)

    async def replay() -> Message:
        return next(pending, None) or await receive()

    return replay


async def capped(scope: Scope, headers: Headers, receive: Receive, send: Send) -> Receive | None:
    """`read_capped`, or answer 413 (None) for a body over `MAX_BODY`: by Content-Length without
    reading it, else as soon as the chunks read pass it."""
    if not _too_long(headers, MAX_BODY) and (replay := await read_capped(receive)) is not None:
        return replay
    note(scope, reason="body too large")
    await PlainTextResponse(f"request body over {MAX_BODY} bytes", 413,
                            headers={"Connection": "close"})(scope, receive, send)  # fmt: skip
    return None


def _rpc(body: bytes) -> tuple[str | None, str | None]:
    """(JSON-RPC method, tool name) of an MCP POST body; the first message of a batch."""
    try:
        msg = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None, None
    if isinstance(msg, list):
        msg = msg[0] if msg else None
    if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
        return None, None
    params = msg.get("params")
    tool = (
        params.get("name") if msg["method"] == "tools/call" and isinstance(params, dict) else None
    )
    return msg["method"], tool if isinstance(tool, str) else None


class Audited:
    """Outermost middleware: one `AuditLog` line per HTTP request, written when the response
    starts (a streaming answer is logged at once, not when it closes).

    It reads no body itself: it keeps a copy (up to `MAX_PARSE`) of what the app reads, so a
    request the guard refuses before reading its body is logged without `rpc`/`tool`."""

    MAX_PARSE = 1 << 20

    def __init__(self, app: ASGIApp, log: AuditLog, stores: Iterable[str] = ()) -> None:
        self.app, self.log, self.stores = app, log, frozenset(stores)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0, headers, path = time.perf_counter(), Headers(scope=scope), scope["path"]
        scope[_NOTE] = info = {"client": None, "reason": None}
        seen: list[bytes] = []
        read = {"size": 0, "done": False}
        upstream = receive

        async def teed() -> Message:
            m = await upstream()
            if m["type"] == "http.request" and not read["done"]:
                body = m.get("body", b"")
                read["size"] += len(body)
                if read["size"] <= self.MAX_PARSE:
                    seen.append(body)
                read["done"] = not m.get("more_body", False)
            return m

        if scope["method"] == "POST":
            receive = teed
        seg = path.split("/")[1] if path.count("/") >= 2 else None
        record = {"ts": None, "peer": remote(scope, headers), "proxied": proxied(headers),
                  "host": headers.get("host"), "method": scope["method"], "path": path,
                  "store": seg if seg in self.stores else None, "rpc": None, "tool": None}  # fmt: skip
        written = False

        def write(status: int) -> None:
            nonlocal written
            written = True
            if read["done"] and read["size"] <= self.MAX_PARSE:
                record["rpc"], record["tool"] = _rpc(b"".join(seen))
            self.log.write({**record, **info, "status": status,
                            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                            "ms": round((time.perf_counter() - t0) * 1000, 2)})  # fmt: skip

        async def logged_send(message: Message) -> None:
            if message["type"] == "http.response.start" and not written:
                write(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, logged_send)
        finally:
            if not written:
                write(500)


# ---- the guard on the MCP routes ----------------------------------------------------------------


@dataclass
class Access:
    """The P7 controls `build_app` applies; None/False = that control is off (P1-P6 behaviour)."""

    tokens: Tokens | None = None
    limiter: RateLimiter | None = None
    audit: AuditLog | None = None
    tls_only: bool = False
    oauth: OwnerOAuth | None = None

    @property
    def auth_required(self) -> bool:
        return self.tokens is not None or self.oauth is not None

    def identify(self, token: str) -> str | None:
        if self.tokens is not None and (label := self.tokens.verify(token)):
            return label
        if self.oauth is not None:
            return self.oauth.verify(token)
        return None

    def close(self) -> None:
        if self.audit is not None:
            self.audit.close()


def via_tls(scope: Scope, headers: Headers) -> bool:
    """A direct loopback client, or a loopback proxy that says the client used https."""
    if not _is_loopback(peer(scope)):
        return False
    return not proxied(headers) or headers.get("x-forwarded-proto", "").lower() == "https"


def _bearer(headers: Headers) -> str | None:
    """The token of `Authorization: Bearer <token>`; None without the header, "" if malformed."""
    value = headers.get("authorization")
    if value is None:
        return None
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


class AccessGuard:
    """TLS-only, then bearer token, then rate limit, then the body size cap (`MAX_BODY`), in
    front of one store's MCP app."""

    def __init__(self, app: ASGIApp, access: Access, resource_path: str = "") -> None:
        self.app, self.access, self.resource_path = app, access, resource_path

    def _challenge(self, error: str | None) -> str:
        parts = ['Bearer realm="mem"']
        if error:
            parts.append(f'error="{error}"')
        if self.access.oauth is not None:
            parts.append(f'resource_metadata="{self.access.oauth.issuer}'
                         f'/.well-known/oauth-protected-resource{self.resource_path}"')  # fmt: skip
        return ", ".join(parts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        a, headers = self.access, Headers(scope=scope)

        async def refuse(status: int, reason: str, text: str, **extra: str) -> None:
            note(scope, reason=reason)
            await PlainTextResponse(text, status, headers=extra or None)(scope, receive, send)

        if a.tls_only and not via_tls(scope, headers):
            await refuse(403, "not https", "mem answers the tailnet only over https:// "
                         "(tailscale serve); plaintext is refused")  # fmt: skip
            return
        key = f"peer:{remote(scope, headers)}"
        if a.auth_required:
            token = _bearer(headers)
            label = a.identify(token) if token else None
            if label is None:
                if a.limiter is not None and not (allowed := a.limiter.allow(key))[0]:
                    await refuse(429, "rate limited", "too many requests",
                                 **{"Retry-After": str(math.ceil(allowed[1]))})  # fmt: skip
                    return
                missing = token is None
                await refuse(401, "no token" if missing else "bad token",
                             "a mem token is required: Authorization: Bearer <token>",
                             **{"WWW-Authenticate": self._challenge(None if missing
                                                                    else "invalid_token")})  # fmt: skip
                return
            note(scope, client=label)
            key = f"client:{label}"
        if a.limiter is not None:
            ok, retry = a.limiter.allow(key)
            if not ok:
                await refuse(429, "rate limited", "too many requests",
                             **{"Retry-After": str(math.ceil(retry))})  # fmt: skip
                return
        if (replay := await capped(scope, headers, receive, send)) is None:
            return  # the body is read only now, after the token and the rate limit
        await self.app(scope, replay, send)


class PeerLimited:
    """The rate limit, keyed by peer, then the body size cap: for the OAuth routes (consent
    guessing, /token)."""

    def __init__(self, app: ASGIApp, limiter: RateLimiter | None) -> None:
        self.app, self.limiter = app, limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.limiter is not None:
            ok, retry = self.limiter.allow(f"peer:{remote(scope, Headers(scope=scope))}")
            if not ok:
                note(scope, reason="rate limited")
                await PlainTextResponse("too many requests", 429,
                                        headers={"Retry-After": str(math.ceil(retry))})(
                    scope, receive, send)  # fmt: skip
                return
        if scope["type"] == "http":
            if (replay := await capped(scope, Headers(scope=scope), receive, send)) is None:
                return
            receive = replay
        await self.app(scope, receive, send)


# ---- OAuth for a claude.ai connector -------------------------------------------------------------


class OwnerOAuth:
    """An OAuth 2.1 authorization server whose only resource owner is the holder of a mem token.

    Implements the MCP SDK's `OAuthAuthorizationServerProvider`. `authorize` sends the browser to
    `/oauth/consent`, where pasting a mem token (any label in the token file) approves the client.
    Clients and the hashes of issued tokens persist in `state_path` (0600); codes and pending
    consents are in memory only (minutes)."""

    CONSENT_PATH = "/oauth/consent"
    CODE_TTL, CONSENT_TTL = 300, 600
    ACCESS_TTL, REFRESH_TTL = 3600, 30 * 86400
    MAX_CLIENTS, MAX_PENDING = 50, 1000  # over funnel anyone may call /register and /authorize

    def __init__(self, issuer: str, owner: Tokens, state_path: str | Path, *,
                 clock: Callable[[], float] = time.time) -> None:  # fmt: skip
        self.issuer, self.owner = issuer.rstrip("/"), owner
        self.state_path, self._clock = Path(state_path).expanduser(), clock
        self._codes: dict[str, Any] = {}
        self._pending: dict[str, tuple[str, Any, float]] = {}
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self._clients: dict[str, dict[str, Any]] = state.get("clients", {})
        self._access: dict[str, dict[str, Any]] = state.get("access", {})
        self._refresh: dict[str, dict[str, Any]] = state.get("refresh", {})

    def _save(self) -> None:
        _write_private(self.state_path, json.dumps(
            {"clients": self._clients, "access": self._access, "refresh": self._refresh},
            indent=1, sort_keys=True))  # fmt: skip

    def _label(self, client_id: str) -> str:
        name = self._clients.get(client_id, {}).get("client_name") or client_id
        return f"oauth:{name}"

    def verify(self, token: str) -> str | None:
        rec = self._access.get(_digest(token))
        if rec is None or rec["expires_at"] <= self._clock():
            return None
        return self._label(rec["client_id"])

    # -- the SDK provider protocol --

    async def get_client(self, client_id: str) -> Any:
        from mcp.shared.auth import OAuthClientInformationFull

        info = self._clients.get(client_id)
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def register_client(self, client_info: Any) -> None:
        from mcp.server.auth.provider import RegistrationError

        if len(self._clients) >= self.MAX_CLIENTS:
            raise RegistrationError("invalid_client_metadata",
                                    f"{self.MAX_CLIENTS} clients are registered already: remove "
                                    f"{self.state_path} and restart mem serve to start over")  # fmt: skip
        for uri in client_info.redirect_uris or []:
            if not (
                uri.scheme == "https" or (uri.scheme == "http" and _is_loopback_name(uri.host))
            ):
                raise RegistrationError("invalid_redirect_uri",
                                        f"redirect URIs must be https (or http on loopback): {uri}")  # fmt: skip
        self._clients[client_info.client_id] = client_info.model_dump(mode="json")
        self._save()

    async def authorize(self, client: Any, params: Any) -> str:
        from mcp.server.auth.provider import AuthorizeError

        self._expire()
        if len(self._pending) >= self.MAX_PENDING:
            raise AuthorizeError("temporarily_unavailable", "too many pending approvals")
        req = secrets.token_urlsafe(16)
        self._pending[req] = (client.client_id, params, self._clock() + self.CONSENT_TTL)
        return f"{self.issuer}{self.CONSENT_PATH}?req={req}"

    async def load_authorization_code(self, client: Any, authorization_code: str) -> Any:
        code = self._codes.get(authorization_code)
        if code is None or code.client_id != client.client_id or code.expires_at <= self._clock():
            return None
        return code

    async def exchange_authorization_code(self, client: Any, authorization_code: Any) -> Any:
        self._codes.pop(authorization_code.code, None)
        return self._issue(client.client_id, authorization_code.scopes,
                           authorization_code.resource, authorization_code.subject)  # fmt: skip

    async def load_refresh_token(self, client: Any, refresh_token: str) -> Any:
        from mcp.server.auth.provider import RefreshToken

        rec = self._refresh.get(_digest(refresh_token))
        if (
            rec is None
            or rec["client_id"] != client.client_id
            or rec["expires_at"] <= self._clock()
        ):
            return None
        return RefreshToken(token=refresh_token, client_id=rec["client_id"], scopes=rec["scopes"],
                            expires_at=int(rec["expires_at"]), resource=rec.get("resource"),
                            subject=rec.get("subject"))  # fmt: skip

    async def exchange_refresh_token(
        self, client: Any, refresh_token: Any, scopes: list[str]
    ) -> Any:
        rec = self._refresh.get(_digest(refresh_token.token), {})
        self._drop_pair(rec.get("pair"))
        return self._issue(client.client_id, scopes or refresh_token.scopes,
                           refresh_token.resource, refresh_token.subject)  # fmt: skip

    async def load_access_token(self, token: str) -> Any:
        from mcp.server.auth.provider import AccessToken

        rec = self._access.get(_digest(token))
        if rec is None or rec["expires_at"] <= self._clock():
            return None
        return AccessToken(token=token, client_id=rec["client_id"], scopes=rec["scopes"],
                           expires_at=int(rec["expires_at"]), resource=rec.get("resource"),
                           subject=rec.get("subject"))  # fmt: skip

    async def revoke_token(self, token: Any) -> None:
        d = _digest(token.token)
        rec = self._access.get(d) or self._refresh.get(d)
        if rec is not None:
            self._drop_pair(rec["pair"])

    async def exchange_identity_assertion(self, *a: Any, **kw: Any) -> Any:
        raise NotImplementedError("identity assertion grants are not offered")

    # -- internals --

    def _issue(self, client_id: str, scopes: list[str], resource: str | None,
               subject: str | None) -> Any:  # fmt: skip
        from mcp.shared.auth import OAuthToken

        now, pair = self._clock(), secrets.token_hex(8)
        access, refresh = (
            "mem_oa_" + secrets.token_urlsafe(32),
            "mem_or_" + secrets.token_urlsafe(32),
        )
        base = {"client_id": client_id, "scopes": list(scopes), "resource": resource,
                "subject": subject, "pair": pair}  # fmt: skip
        self._access[_digest(access)] = {**base, "expires_at": now + self.ACCESS_TTL}
        self._refresh[_digest(refresh)] = {**base, "expires_at": now + self.REFRESH_TTL}
        self._expire()
        self._save()
        return OAuthToken(access_token=access, expires_in=self.ACCESS_TTL, refresh_token=refresh,
                          scope=" ".join(scopes) or None)  # fmt: skip

    def _drop_pair(self, pair: str | None) -> None:
        if pair is None:
            return
        self._access = {k: v for k, v in self._access.items() if v["pair"] != pair}
        self._refresh = {k: v for k, v in self._refresh.items() if v["pair"] != pair}
        self._save()

    def _expire(self) -> None:
        now = self._clock()
        self._pending = {k: v for k, v in self._pending.items() if v[2] > now}
        self._codes = {k: v for k, v in self._codes.items() if v.expires_at > now}
        self._access = {k: v for k, v in self._access.items() if v["expires_at"] > now}
        self._refresh = {k: v for k, v in self._refresh.items() if v["expires_at"] > now}

    # -- routes --

    async def _consent(self, request: Request) -> Response:
        from mcp.server.auth.provider import AuthorizationCode, construct_redirect_uri

        self._expire()
        if request.method == "GET":
            req = request.query_params.get("req", "")
        else:
            form = await request.form()
            req = str(form.get("req", ""))
        pending = self._pending.get(req)
        if pending is None:
            return PlainTextResponse("this approval request is unknown or expired: "
                                     "start the connection again", 400)  # fmt: skip
        client_id, params, _ = pending
        name = self._clients.get(client_id, {}).get("client_name") or client_id
        if request.method == "GET":
            return HTMLResponse(_CONSENT_PAGE.format(
                name=html.escape(name), redirect=html.escape(str(params.redirect_uri)),
                req=html.escape(req)), headers=_NO_FRAMES)  # fmt: skip
        owner = self.owner.verify(str(form.get("owner_token", "")).strip())
        if owner is None:
            note(request.scope, reason="bad owner token")
            return PlainTextResponse("not a mem token of this server: nothing approved", 403)
        del self._pending[req]
        code = secrets.token_urlsafe(32)
        self._codes[code] = AuthorizationCode(
            code=code, scopes=params.scopes or [], expires_at=self._clock() + self.CODE_TTL,
            client_id=client_id, code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource, subject=owner)  # fmt: skip
        note(request.scope, client=f"owner:{owner}")
        return RedirectResponse(construct_redirect_uri(str(params.redirect_uri), code=code,
                                                       state=params.state), 302)  # fmt: skip

    def routes(self, resource_paths: Iterable[str]) -> list[Route]:
        """The authorization server (`/.well-known/oauth-authorization-server`, `/register`,
        `/authorize`, `/token`, `/revoke`), the consent page and one protected-resource
        metadata document per MCP path (RFC 9728)."""
        from mcp.server.auth.routes import create_auth_routes, create_protected_resource_routes
        from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
        from pydantic import AnyHttpUrl

        out = create_auth_routes(self, AnyHttpUrl(self.issuer),
                                 client_registration_options=ClientRegistrationOptions(enabled=True),
                                 revocation_options=RevocationOptions(enabled=True))  # fmt: skip
        for path in resource_paths:
            out += create_protected_resource_routes(AnyHttpUrl(f"{self.issuer}{path}"),
                                                    [AnyHttpUrl(self.issuer)],
                                                    resource_name=f"mem {path}")  # fmt: skip
        out.append(Route(self.CONSENT_PATH, self._consent, methods=["GET", "POST"]))
        return out


def _is_loopback_name(host: str | None) -> bool:
    return host in ("localhost", "127.0.0.1", "[::1]", "::1")


_NO_FRAMES = {"X-Frame-Options": "DENY", "Content-Security-Policy": "frame-ancestors 'none'",
              "Cache-Control": "no-store"}  # fmt: skip
_CONSENT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>mem: approve {name}</title></head>
<body style="font-family: sans-serif; max-width: 36em; margin: 3em auto">
<h1>Connect {name} to mem?</h1>
<p><b>{name}</b> asks to read and write your memories. After you approve, the browser returns to
<code>{redirect}</code>.</p>
<p>To approve, paste one of your mem tokens (<code>mem token add</code> prints one). It is checked
here and not stored or sent on.</p>
<form method="post" action="/oauth/consent">
<input type="hidden" name="req" value="{req}">
<input type="password" name="owner_token" autocomplete="off" size="50" autofocus>
<button type="submit">Approve</button>
</form>
<p>To refuse, close this page.</p>
</body></html>
"""
