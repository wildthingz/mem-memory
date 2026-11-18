"""Model access through `mem serve` (architecture-v2 §2.1 D5, docs/decisions/0006).

Only `mem serve` loads models. Tools that need them (`mem calibrate --labels`, the bench's LME
adapter) use the ones it holds through three local-only routes, mounted at `/models` next to
`/<store>/mcp` on the same HTTP port (default 127.0.0.1:8765):

    GET  /models         -> {"embedder": {...}, "reranker": {...}, "fp_embed": str, "dim": int}
    POST /models/embed   {"kind": "query" | "docs", "texts": [str, ...]}
                         -> {"vectors": [[float, ...], ...]}       one per text, in order
    POST /models/rerank  {"query": str, "docs": [str, ...]}
                         -> {"logits": [float, ...]}               raw logits, one per doc

- `embedder` / `reranker` are the config sections the served models were loaded for; a client
  compares them with its own config (`connect`) before scoring anything.
- Texts reach the models exactly as sent: callers apply the query / document prefixes and the
  rerank pair format (`Store.query_vector`, `doc_text`, `read.rerank`), as in-process.
- At most `MAX_ITEMS` texts or docs per request (`RemoteEmbedder` batches). Floats are JSON
  doubles, exact for the models' float32 outputs: remote scores equal in-process ones.
- A client that is not loopback, or comes through a proxy (`tailscale serve`, P7), gets 403; a bad body gets
  400 `{"error": "<reason>"}`.
- Calls run on the server's event loop, so they are serialised with MCP tool calls.

`mem serve --http --models-only` serves these routes with no store mounted: for calibrating a
config no store is gated for yet (first gate, or after `mem reembed`), and for the bench.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mem import catalog
from mem.config import Config
from mem.testing import Embedder, Reranker

MAX_ITEMS = 512
TIMEOUT_S = 600.0  # a CPU embed of MAX_ITEMS texts; the server answers 4xx at once
START_HINT = "mem serve --http --models-only --port {port} (with the same --config and --embedder)"


class ServeUnavailable(RuntimeError):
    """No usable `mem serve` for these models; the message says what to start."""


# ---- server side --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelHost:
    """The models `mem serve` loaded, and the config they were loaded for."""

    embedder: Embedder
    reranker: Reranker
    cfg: Config

    def info(self) -> dict[str, Any]:
        return {"embedder": self.cfg.embedder, "reranker": self.cfg.reranker,
                "fp_embed": self.cfg.fp_embed(), "dim": self.embedder.dim}  # fmt: skip


class _BadBody(ValueError):
    pass


def _strings(body: dict[str, Any], key: str) -> list[str]:
    value = body.get(key)
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        raise _BadBody(f"{key} must be a list of strings")
    if len(value) > MAX_ITEMS:
        raise _BadBody(f"{key}: {len(value)} items, at most {MAX_ITEMS} per request")
    return value


async def _body(request: Request, keys: set[str]) -> dict[str, Any]:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise _BadBody(f"body is not JSON: {err}") from err
    if not isinstance(body, dict):
        raise _BadBody("body must be a JSON object")
    if unknown := sorted(set(body) - keys):
        raise _BadBody(f"unknown field(s) {unknown}")
    return body


def _loopback(request: Request) -> bool:
    """A loopback peer that is not a proxy: behind `tailscale serve` (P7) every request comes
    from loopback, carrying X-Forwarded-For / Forwarded."""
    if any(h in request.headers for h in ("x-forwarded-for", "forwarded", "x-forwarded-proto")):
        return False
    try:
        return request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return False


def routes(host: ModelHost) -> list[Route]:
    """The `/models` routes over `host`'s models."""

    def guarded(handler):
        async def endpoint(request: Request) -> JSONResponse:
            if not _loopback(request):
                return JSONResponse({"error": "/models is local-only"}, status_code=403)
            try:
                return JSONResponse(await handler(request))
            except _BadBody as err:
                return JSONResponse({"error": str(err)}, status_code=400)

        return endpoint

    async def info(_: Request) -> dict[str, Any]:
        return host.info()

    async def embed(request: Request) -> dict[str, Any]:
        body = await _body(request, {"kind", "texts"})
        texts = _strings(body, "texts")
        if body.get("kind") == "query":
            return {"vectors": [host.embedder.embed_query(t) for t in texts]}
        if body.get("kind") == "docs":
            return {"vectors": host.embedder.embed_docs(texts) if texts else []}
        raise _BadBody('kind must be "query" or "docs"')

    async def rerank(request: Request) -> dict[str, Any]:
        body = await _body(request, {"query", "docs"})
        if not isinstance(body.get("query"), str):
            raise _BadBody("query must be a string")
        return {"logits": host.reranker.score(body["query"], _strings(body, "docs"))}

    return [Route("/models", guarded(info), methods=["GET"]),
            Route("/models/embed", guarded(embed), methods=["POST"]),
            Route("/models/rerank", guarded(rerank), methods=["POST"])]  # fmt: skip


# ---- client side --------------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never a proxy


def serve_listening(port: int, host: str = "127.0.0.1") -> bool:
    """Whether anything accepts connections on host:port (a `mem serve`, presumably)."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


class _Client:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self.base, self.port = f"http://{host}:{port}", port

    def call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None
                                     else "POST", headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with _OPENER.open(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:200]
            raise ServeUnavailable(f"{self.base}{path}: HTTP {err.code} {detail} (is it mem "
                                   "serve P2 or later?)") from err  # fmt: skip
        except (urllib.error.URLError, OSError) as err:
            reason = getattr(err, "reason", err)
            raise ServeUnavailable(
                f"no mem serve at {self.base} ({reason}): start it: "
                + START_HINT.format(port=self.port)
            ) from err


class RemoteEmbedder:
    """`mem.testing.Embedder` over `/models/embed`."""

    def __init__(self, client: _Client, dim: int, prompts: catalog.Prompts | None = None) -> None:
        self._client, self.dim = client, dim
        self.prompts = prompts or catalog.DEFAULT_PROMPTS

    def embed_query(self, text: str) -> list[float]:
        return self._client.call("/models/embed", {"kind": "query", "texts": [text]})["vectors"][0]

    def embed_docs(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), MAX_ITEMS):
            batch = list(texts[start : start + MAX_ITEMS])
            out += self._client.call("/models/embed", {"kind": "docs", "texts": batch})["vectors"]
        return out


class RemoteReranker:
    """`mem.testing.Reranker` over `/models/rerank` (pools are ≤ `pool` = 30 docs)."""

    def __init__(self, client: _Client) -> None:
        self._client = client

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        if not docs:
            return []
        return self._client.call("/models/rerank", {"query": query, "docs": list(docs)})["logits"]


def connect(
    port: int, cfg: Config, host: str = "127.0.0.1"
) -> tuple[RemoteEmbedder, RemoteReranker]:
    """The models of the `mem serve` on host:port, after checking they are `cfg`'s.

    Raises `ServeUnavailable` (one line, naming what to start) when nothing answers or the
    serve holds other models."""
    client = _Client(port, host)
    info = client.call("/models")
    want = json.loads(json.dumps({"embedder": cfg.embedder, "reranker": cfg.reranker}))
    differ = [k for k in ("embedder", "reranker") if info.get(k) != want[k]]
    if differ:
        raise ServeUnavailable(
            f"the mem serve at {client.base} holds other models ({' and '.join(differ)} differ "
            "from the config's): stop it, or start one for this config on another port and pass "
            "that --port: " + START_HINT.format(port="<port>")
        )
    return (RemoteEmbedder(client, int(info["dim"]), catalog.prompts_of(info["embedder"])),
            RemoteReranker(client))  # fmt: skip
