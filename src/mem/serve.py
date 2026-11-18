"""`mem serve` (architecture-v2 §4, §5): MCP over local HTTP (both stores mounted) and stdio.

One `MemServer` wraps one store; each becomes its own MCP server entry (`mem-personal`,
`mem-work`) under `/<scope>/mcp`. Tool bodies live in `mem.read` / `mem.write`; this module
only validates arguments against the frozen contract, routes the call and shapes the reply.
"""

from __future__ import annotations

import contextlib
import fcntl
import ipaddress
import json
import logging
import re
import socket
import sqlite3
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import mcp_types as types
import uvicorn
from jsonschema import Draft202012Validator
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Router
from starlette.types import ASGIApp, Receive, Scope, Send

from mem import auth, contracts, shipped
from mem.common import MemError
from mem.config import Config
from mem.gate import NO_READ_MODEL, read_gate
from mem.hook import HookServer
from mem.listmode import NO_LIST_MODEL
from mem.models_http import ModelHost, routes
from mem.store import Store
from mem.testing import Embedder, Reranker

log = logging.getLogger("mem.serve")

STORE_NAMES = ("personal", "work")
DEFAULT_PORT = 8765
UNKNOWN_HOST = "unknown"  # events.host of an HTTP call that sent no X-Mem-Host (A12)
HOST_LABEL = re.compile(r"[A-Za-z0-9._-]{1,64}")


def clean_host(label: str | None) -> str:
    """`label` for events.host if it is 1-64 of [A-Za-z0-9._-], else `unknown`: X-Mem-Host is
    client text, kept as a short tag, never stored as sent (MEM2-41 F8)."""
    return label if label is not None and HOST_LABEL.fullmatch(label) else UNKNOWN_HOST


# Host/Origin names the MCP routes answer (DNS-rebinding guard, `HostGuard`). `mem serve` passes
# LOOPBACK_HOSTS, plus the tailnet address and MagicDNS name with --tailscale. The library default
# also takes the tailnet forms: IP literals and Tailscale-owned names cannot be rebound by a page.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_ALLOWED_HOSTS = ("127.0.0.0/8", "localhost", "::1", "100.64.0.0/10", "*.ts.net")


def list_tool_schemas() -> list[dict[str, Any]]:
    """The tools as advertised: the frozen snapshot (`mem.contracts`), unchanged."""
    return contracts.mcp_tools()["tools"]


class MemServer:
    """One store plus the routing of tool calls to it. Calls are serialised by the caller's
    event loop (sqlite3 connection, one user): a call never interleaves with another."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._validators = {
            t["name"]: Draft202012Validator(t["inputSchema"]) for t in list_tool_schemas()
        }

    @property
    def scope(self) -> str:
        return self.store.scope

    def call_tool(
        self, name: str, args: Mapping[str, Any] | None = None, *,
        session: str | None = None, host: str | None = None,
    ) -> tuple[str, dict[str, Any]]:  # fmt: skip
        args = dict(args or {})
        validator = self._validators.get(name)
        if validator is None:
            raise MemError(f"unknown tool {name!r}: tools are {', '.join(self._validators)}")
        errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
        if errors:
            e = errors[0]
            where = ".".join(str(p) for p in e.path) or "arguments"
            raise MemError(f"bad arguments for {name}: {where}: {e.message}")
        res = getattr(self.store, name)(**args, session=session, host=host)
        return res.text, res.structured

    def close(self) -> None:
        self.store.close()


def open_inprocess(
    db_path: str | Path, *, scope: str, embedder: Embedder, tz: str = "UTC",
    host: str | None = None, clock: Callable[[], datetime] | None = None,
    session: str = "inproc", reranker: Reranker | None = None, cfg: Config | None = None,
) -> MemServer:  # fmt: skip
    """The in-process entry point used by tests and `bench` (adapters/ours.py). `session` is the
    stand-in for the MCP session id (stdio passes a per-process id, HTTP supplies its own).
    With `cfg` the store must pass `preflight` (the bench runs the pinned config, §2.1 D5)."""
    fp = None
    if cfg is not None:
        problems = preflight({scope: Path(db_path)}, cfg)
        if problems:
            raise MemError("; ".join(problems))
        fp = cfg.fp_embed()
    store = Store.open(db_path, embedder, scope=scope, fp=fp, tz=tz, host=host, session=session,
                       now=clock or (lambda: datetime.now(UTC)), reranker=reranker,
                       retrieval=cfg.retrieval if cfg else None, read=cfg.read if cfg else None,
                       ingest=cfg.ingest if cfg else None,
                       fallback_gate=shipped.gate_for(cfg) if cfg else None)  # fmt: skip
    return MemServer(store)


def gate_warnings(stores: Mapping[str, Path]) -> list[str]:
    """Non-blocking: one line per store whose gate was fitted infeasible (architecture-v2 §3.3
    steps 4-5: no setting met the five P2 tune targets, the stored one is the flagged fallback).
    `mem serve` prints them and starts, so the person sees the numbers in use; the fix is on tune."""
    out = []
    for name, path in stores.items():
        if not Path(path).exists():
            continue
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            gate = read_gate(conn)
        except ValueError:
            continue  # preflight reports it
        finally:
            conn.close()
        if gate is not None and gate.tune.get("feasible") is False:
            unmet = [k for k, t in gate.tune.get("targets", {}).items() if not t.get("met")]
            out.append(
                f"{name} ({path}): the gate's fit is INFEASIBLE on tune ({gate.tune.get('rule')}"
                f"; unmet: {', '.join(unmet) or 'the five targets jointly'}): serving the flagged "
                "fallback; fix the cause on tune and re-run mem calibrate (architecture-v2 §3.3)"
            )
    return out


def preflight(stores: Mapping[str, Path], cfg: Config) -> list[str]:
    """Why `mem serve` must not start with `cfg` on these stores (§3.3, A2): one line per problem,
    each naming the fix. Read-only: a missing store is reported, never created."""
    problems = []
    try:
        ship = shipped.load()
    except MemError as err:
        return [str(err)]
    ship_ok = ship is not None and ship.gate.fingerprint_hash == cfg.fp_all()
    for name, path in stores.items():
        where = f"{name} ({path})"
        if not Path(path).exists():
            if ship is not None and not ship_ok:
                problems.append(
                    f"{where}: no store yet, and {shipped.mismatch(cfg, ship)}: "
                    f"restore the config, or run `{shipped.calibrate_fix(name, path)}`"
                )
            elif not ship_ok:  # a new store starts on the shipped gate when it fits
                problems.append(f"{where}: no store and no gate yet: fit one with `mem calibrate "
                                f"--records <tune records> --store {name}={path}`")  # fmt: skip
            continue
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            fp_store = json.loads(meta.get("fingerprint", "{}")).get("embed")
            if fp_store != cfg.fp_embed():
                problems.append(
                    f"{where}: embedder fingerprint mismatch: the store's vectors are "
                    f"{str(fp_store)[:8]}, the config's embedder is {cfg.fp_embed()[:8]}: restore "
                    f"the config, or move the store to it: `mem reembed --store {name}={path} "
                    "--config <new config.yaml>`, then `mem calibrate --records <tune records> "
                    "--config <new config.yaml> --out gate.json`, then `mem switch --gate gate.json "
                    f"--store {name}={path} --config <new config.yaml>`"
                )
                continue
            try:
                gate = read_gate(conn)
            except ValueError as err:
                problems.append(f"{where}: {err}")
                continue
        finally:
            conn.close()
        if gate is None and ship_ok:
            continue  # no local gate: the shipped one fits this config
        if gate is None and ship is not None:
            problems.append(
                f"{where}: no gate of its own, and {shipped.mismatch(cfg, ship)}: "
                f"restore the config, or run `{shipped.calibrate_fix(name, path)}`"
            )
        elif gate is None:
            problems.append(f"{where}: no gate: run `mem calibrate --records <tune records> "
                            f"--store {name}={path}`")  # fmt: skip
        elif gate.fingerprint_hash != cfg.fp_all():
            problems.append(
                f"{where}: gate fingerprint mismatch: the gate was fitted for "
                f"{gate.fingerprint_hash[:8]}, the config's embedder + reranker + retrieval are "
                f"{cfg.fp_all()[:8]}: restore the config, or run `{shipped.calibrate_fix(name, path)}`"
            )
        else:
            differ = [f"{k} config {v} ≠ store {getattr(gate, k)}" for k, v in cfg.gate.items()
                      if v is not None and getattr(gate, k) != v]  # fmt: skip
            if differ:
                problems.append(f"{where}: gate mismatch ({'; '.join(differ)}): remove the gate "
                                "values from config.yaml or re-run mem calibrate")  # fmt: skip
            if cfg.p3_model and gate.read_model is None:
                problems.append(f"{where}: {NO_READ_MODEL}")
            if cfg.list_mode and gate.list_model is None:
                problems.append(f"{where}: {NO_LIST_MODEL}")
            if cfg.conflict_guard and gate.needs_read_guard():
                problems.append(
                    f"{where}: the gate has a conflict model but no read_guard (theta_read, ADR"
                    " 0018) and read.conflict_guard is on: fit it (`bench syn-fit-read-guard"
                    " --gate gate.json`, `bench gate-add-read-guard`, `mem switch --gate`) or set"
                    " read.conflict_guard: false in config.yaml (a new fingerprint: re-calibrate)"
                )
    return problems


def _origin(ctx: Any) -> tuple[str | None, str | None]:
    """(MCP session id, X-Mem-Host label) of the request; None where the transport has none.

    The session id is the stateful Streamable HTTP `Mcp-Session-Id` (handshake-era clients). The
    2026-07-28 protocol has no sessions, so such a client's events carry no session id."""
    if ctx.request is None:  # stdio: the store's --host-label applies
        return None, None
    headers = ctx.request.headers
    return headers.get("mcp-session-id"), clean_host(headers.get("x-mem-host"))


def build_mcp_server(ms: MemServer) -> Server:
    tools = [
        types.Tool(name=t["name"], description=t["description"], input_schema=t["inputSchema"],
                   output_schema=t["structuredContent"])
        for t in list_tool_schemas()
    ]  # fmt: skip

    async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        session, host = _origin(ctx)
        try:
            text, structured = ms.call_tool(params.name, params.arguments, session=session,
                                            host=host)  # fmt: skip
        except MemError as err:
            return types.CallToolResult(content=[types.TextContent(text=str(err))], is_error=True)
        return types.CallToolResult(content=[types.TextContent(text=text)],
                                    structured_content=structured)  # fmt: skip

    return Server(f"mem-{ms.scope}", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def _host_name(authority: str) -> str:
    """`name` of a Host header `name[:port]` / `[v6]:port`, lowercased, no trailing dot."""
    if authority.startswith("["):
        name = authority[1:].partition("]")[0]
    else:
        name = authority.rpartition(":")[0] if ":" in authority else authority
    return name.lower().rstrip(".")


def host_allowed(name: str, allowed: Sequence[str]) -> bool:
    """`allowed` entries: an exact name or address (any port), `*.suffix`, or a CIDR network."""
    for pattern in allowed:
        pattern = pattern.lower().rstrip(".")
        if "/" in pattern:
            with contextlib.suppress(ValueError):
                if ipaddress.ip_address(name) in ipaddress.ip_network(pattern):
                    return True
        elif pattern.startswith("*."):
            if name.endswith(pattern[1:]):
                return True
        elif name == pattern:
            return True
    return False


class HostGuard:
    """DNS-rebinding guard for the MCP routes, in place of the SDK's (which only matches exact
    `host:port` entries and so cannot take a tailnet range or name): 421 for a Host header, 403
    for an Origin header, whose name is not in `allowed`."""

    def __init__(self, app: ASGIApp, allowed: Sequence[str]) -> None:
        self.app, self.allowed = app, tuple(allowed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if not host_allowed(_host_name(headers.get("host", "")), self.allowed):
                auth.note(scope, reason="bad host")
                await PlainTextResponse("Invalid Host header", 421)(scope, receive, send)
                return
            if origin and not host_allowed((urlsplit(origin).hostname or "").rstrip("."),
                                           self.allowed):  # fmt: skip
                auth.note(scope, reason="bad origin")
                await PlainTextResponse("Invalid Origin header", 403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(
    servers: Mapping[str, MemServer], models: ModelHost | None = None, *,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS, access: auth.Access | None = None,
) -> Starlette:  # fmt: skip
    """One Starlette app mounting `/<name>/mcp` for each store, and with `models` the local-only
    `/models` routes over them (`mem.models_http`, D5). The MCP routes answer only requests whose
    Host (and Origin, if sent) is in `allowed_hosts` (`HostGuard`). `access` adds the P7 controls
    (`mem.auth`): TLS-only, bearer tokens and rate limits on the MCP routes, the OAuth routes,
    and the audit log over every route."""
    no_sdk_guard = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    subs = {n: build_mcp_server(ms).streamable_http_app(streamable_http_path="/mcp",
                                                        transport_security=no_sdk_guard)
            for n, ms in servers.items()}  # fmt: skip

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for sub in subs.values():
                await stack.enter_async_context(sub.router.lifespan_context(sub))
            yield

    def guarded(name: str, sub: ASGIApp) -> ASGIApp:
        return auth.AccessGuard(sub, access, f"/{name}/mcp") if access else sub

    mounts: list[Any] = [Mount(f"/{n}", app=HostGuard(guarded(n, sub), allowed_hosts))
                         for n, sub in subs.items()]  # fmt: skip
    if access and access.oauth:
        oauth = Router(routes=access.oauth.routes(f"/{n}/mcp" for n in subs))
        mounts.append(Mount("", app=HostGuard(auth.PeerLimited(oauth, access.limiter),
                                              allowed_hosts)))  # fmt: skip
    middleware = ([Middleware(auth.Audited, log=access.audit, stores=STORE_NAMES)]
                  if access and access.audit else [])  # fmt: skip
    return Starlette(routes=[*(routes(models) if models else []), *mounts], lifespan=lifespan,
                     middleware=middleware)  # fmt: skip


def parse_stores(specs: Sequence[str]) -> dict[str, Path]:
    """`name=path` entries (name is a scope) → {name: path}."""
    out: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or name not in STORE_NAMES or not path or "=" in path or "," in path:
            raise ValueError(
                f"--store takes name=path with name one of {', '.join(STORE_NAMES)}, got {spec!r}"
            )
        if name in out:
            raise ValueError(f"--store {name} given twice")
        out[name] = Path(path).expanduser()
    return out


class StoreBusy(MemError):
    pass


@contextlib.contextmanager
def store_locks(paths: Iterable[Path], *, exclusive: bool = False) -> Iterator[None]:
    """Advisory locks on `<store>.serve.lock` (docs/decisions/0006): every `mem serve` holds a
    shared one on each store it mounts, `mem reembed` an exclusive one, so neither runs while the
    other does. Keyed on the store's real file, so every spelling of one database (symlink, `..`,
    relative) meets the same lock. Never blocks: a held lock raises `StoreBusy` naming the holder."""
    real = dict.fromkeys(Path(p).expanduser().resolve() for p in paths)
    with contextlib.ExitStack() as stack:
        for path in real:
            lock = Path(f"{path}.serve.lock")
            lock.parent.mkdir(parents=True, exist_ok=True)
            fh = stack.enter_context(lock.open("a"))
            try:
                fcntl.flock(fh, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
            except BlockingIOError as err:
                holder = "a running mem serve" if exclusive else "a running mem reembed"
                raise StoreBusy(f"{path} is in use by {holder}") from err
        yield


def _open_all(
    stores: Mapping[str, Path], embedder: Embedder, reranker: Reranker, cfg: Config,
    host_label: str, tz: str, session: str,
) -> dict[str, MemServer]:  # fmt: skip
    return {
        name: MemServer(
            Store.open(
                path,
                embedder,
                scope=name,
                fp=cfg.fp_embed(),
                tz=tz,
                host=clean_host(host_label),
                session=session,
                reranker=reranker,
                retrieval=cfg.retrieval,
                read=cfg.read,
                ingest=cfg.ingest,
                fallback_gate=shipped.gate_for(cfg),
            )
        )
        for name, path in stores.items()
    }


def tailscale_ip(run: Callable[..., Any] = subprocess.run) -> str:
    """This machine's tailnet IPv4 address, from `tailscale ip -4`."""
    fix = "is Tailscale installed and up on this machine? Try `tailscale up`"
    try:
        out = run(["tailscale", "ip", "-4"], capture_output=True, text=True, check=False)
    except OSError as err:
        raise MemError(f"cannot run tailscale ({err}): {fix}") from err
    lines = out.stdout.split()
    if out.returncode != 0 or not lines:
        raise MemError(f"`tailscale ip -4` gave no address: {fix}")
    try:
        ipaddress.IPv4Address(lines[0])
    except ValueError as err:
        raise MemError(
            f"`tailscale ip -4` printed {lines[0]!r}, not an IPv4 address: {fix}"
        ) from err
    return lines[0]


def tailscale_dns_name(run: Callable[..., Any] = subprocess.run) -> str | None:
    """This machine's MagicDNS name (`tailscale status --json` `.Self.DNSName`, no trailing dot);
    None when tailscale cannot say."""
    try:
        out = run(["tailscale", "status", "--json"], capture_output=True, text=True, check=False)
        name = json.loads(out.stdout)["Self"]["DNSName"] if out.returncode == 0 else ""
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return name.rstrip(".") or None


def allowed_hosts(host: str, tailscale: str | None = None, dns_name: str | None = None,
                  extra: Sequence[str] = ()) -> list[str]:  # fmt: skip
    """What `mem serve` lets the MCP routes answer: loopback and `host`; with `tailscale` also
    the tailnet address and MagicDNS name; plus `extra` (--allow-host)."""
    out = [*LOOPBACK_HOSTS, host, *(a for a in (tailscale, dns_name) if a), *extra]
    return list(dict.fromkeys(out))


def bind_addresses(host: str, tailscale: str | None) -> list[str]:
    """Where the one HTTP port listens: `host`, plus the tailnet address when given."""
    return [host, tailscale] if tailscale else [host]


def _listen(addr: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.set_inheritable(True)
    return sock


def client_config(server: str, host_label: str, *, port: int = DEFAULT_PORT, tls: bool = False,
                  token: bool = False) -> list[str]:  # fmt: skip
    """The `claude mcp add` commands for a client machine (architecture-v2 §6 P5): one MCP entry
    per store, each sending `host_label` as X-Mem-Host. `tls`: through `tailscale serve`
    (https://<server>/, port 443, P7); `token`: also send `$MEM_TOKEN` as the bearer token. The
    shell running the line expands it, so the token itself is never printed or put in history."""
    base = f"https://{server}" if tls else f"http://{server}:{port}"
    auth_header = ' --header "Authorization: Bearer $MEM_TOKEN"' if token else ""
    return [
        f"claude mcp add --transport http --scope user mem-{name} "
        f'{base}/{name}/mcp --header "X-Mem-Host: {host_label}"{auth_header}'
        for name in STORE_NAMES
    ]


def uvicorn_config(app: ASGIApp, *, host: str, port: int, log_level: str = "info") -> Any:
    """Never let uvicorn rewrite the peer and scheme from X-Forwarded-* (its default for a
    127.0.0.1 proxy): `mem.auth` reads them itself, and must see `tailscale serve` as loopback."""
    return uvicorn.Config(app, host=host, port=port, log_level=log_level, proxy_headers=False)


def run_http(
    stores: Mapping[str, Path], embedder: Embedder, reranker: Reranker, cfg: Config, *,
    host: str, port: int, host_label: str, tz: str, tailscale: str | None = None,
    allow: Sequence[str] | None = None, access: auth.Access | None = None,
    hook_socket: Path | None = None,
) -> None:  # fmt: skip
    """Serve every store under `/<name>/mcp`, and the models under `/models`, until interrupted.
    With `tailscale` (this machine's tailnet IP) the port is also bound there (§5, P5); `host`
    stays bound, so local clients keep working. `allow`: the Host names the MCP routes answer
    (default: loopback, `host` and `tailscale`; see `allowed_hosts`).
    The stores are opened here, in the thread that runs the event loop, because a sqlite
    connection belongs to one thread. No stores = `--models-only`. `access`: the P7 controls.
    `hook_socket`: where the Stop-hook endpoint listens (`mem.hook`, F4); None = no endpoint
    (the CLI passes it only when `mem install --hooks` registered the hook)."""
    servers = _open_all(stores, embedder, reranker, cfg, host_label, tz, session="http")
    hook_server = None
    try:
        if hook_socket is not None:
            hook_server = HookServer(hook_socket).start()
        app = build_app(servers, models=ModelHost(embedder, reranker, cfg),
                        allowed_hosts=allow or allowed_hosts(host, tailscale),
                        access=access)  # fmt: skip
        config = uvicorn_config(app, host=host, port=port)
        sockets = [_listen(addr, port) for addr in bind_addresses(host, tailscale)]
        uvicorn.Server(config).run(sockets=sockets)
    finally:
        if hook_server is not None:
            hook_server.close()
        for ms in servers.values():
            ms.close()
        if access is not None:
            access.close()


def run_stdio(
    stores: Mapping[str, Path], embedder: Embedder, reranker: Reranker, cfg: Config, *,
    host_label: str, tz: str,
) -> None:  # fmt: skip
    """One store over stdio (tests, and clients that cannot use HTTP). The session id is a
    per-process id: stdio has no `Mcp-Session-Id` (architecture-v2 §5)."""
    if len(stores) != 1:
        raise ValueError(f"stdio serves exactly one store, got {len(stores)}")
    session = f"stdio-{uuid.uuid4().hex[:8]}"
    (ms,) = _open_all(stores, embedder, reranker, cfg, host_label, tz, session).values()
    server = build_mcp_server(ms)

    async def main() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(main)
    finally:
        ms.close()
