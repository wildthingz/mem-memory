"""A store: one SQLite file (a scope) plus the embedder that fills its vec table.

`Store` owns the connection and the events log; the tool logic lives in `mem.write` and `mem.read`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import sqlite_vec

from mem import catalog, link, read, schema, sessionvec, subjects, window, write
from mem.common import MemError, ToolResult, fmt_ts
from mem.config import default_config
from mem.gate import NO_READ_MODEL, Gate, read_gate
from mem.readmodel import ReadModel
from mem.testing import Embedder, Reranker


def doc_text(ctx: str, text: str, embedder: object | None = None) -> str:
    """The document-prompted text an embedder sees; the embedder's own documented format when it
    carries one (`mem.catalog.Prompts`), else EmbeddingGemma's."""
    return catalog.doc_prompted(ctx, text, embedder)


def init_store(path: str | Path, scope: str, fp: str, dim: int) -> sqlite3.Connection:
    """Open (creating if needed) a store file for `scope` whose vectors live in `vec_<fp8>`.
    Refuses a file created for another scope or another embedder fingerprint."""
    if scope not in ("personal", "work"):
        raise MemError(f"scope must be 'personal' or 'work', got {scope!r}")
    conn = schema.connect(path)
    conn.isolation_level = None  # explicit BEGIN IMMEDIATE in the write path
    table = schema.vec_table_name(fp)
    for key, value in (
        ("scope", scope),
        ("vec_table", table),
        ("fingerprint", json.dumps({"embed": fp})),
    ):
        conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value))
    stored = dict(conn.execute("SELECT key, value FROM meta WHERE key IN ('scope','vec_table')"))
    if stored != {"scope": scope, "vec_table": table}:
        conn.close()
        raise MemError(f"{path} was created for {stored}, opened as {scope}/{table}")
    schema.create_vec_table(conn, fp, dim)
    return conn


class Store:
    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: Embedder,
        *,
        scope: str,
        vec_table: str,
        now: Callable[[], datetime],
        tz: str,
        host: str | None,
        session: str | None,
        retrieval: dict[str, Any],
        reranker: Reranker | None = None,
        read: dict[str, Any] | None = None,
        ingest: dict[str, Any] | None = None,
        fallback_gate: Gate | None = None,
    ) -> None:
        self.conn, self.embedder, self.scope, self.vec_table = conn, embedder, scope, vec_table
        self.now, self.tz, self.host, self.session = now, tz, host, session
        self.retrieval, self.reranker = retrieval, reranker
        self.read_cfg = dict(read if read is not None else default_config().read)
        self.ingest_cfg = dict(ingest if ingest is not None else default_config().ingest)
        # a store's own gate wins; the shipped one (checked against the config by preflight) is used
        # only when it has none, and is never written into the store
        self.gate: Gate | None = read_gate(conn) or fallback_gate
        self._last_query: tuple[str, list[float]] | None = None

    @property
    def leg_k(self) -> int:
        return int(self.retrieval["leg_k"])

    @property
    def conflict_guard(self) -> bool:
        """`read.conflict_guard` (ADR 0018): the read-time conflict guard is in force."""
        return bool(self.read_cfg.get("conflict_guard", False))

    @property
    def list_mode(self) -> bool:
        """`read.list_mode` (ADR 0023): aggregation queries get the list-then-count answer."""
        return bool(self.read_cfg.get("list_mode", False))

    @property
    def list_paging(self) -> bool:
        """`read.list_paging` (MEM2-212): the list-mode answer is paged (needs `list_mode`)."""
        return bool(self.read_cfg.get("list_paging", False)) and self.list_mode

    @property
    def split_endpoints(self) -> bool:
        """`read.split_endpoints` (ADR 0026): a two-event question gets a pool and a gate per event."""
        return bool(self.read_cfg.get("split_endpoints", False))

    @property
    def list_fill(self) -> int:
        """`read.list_fill` (ADR 0030): the list-mode answer is filled to the top N rows by p after the
        `tau_list` test (needs `list_mode`); 0 is off."""
        return int(self.read_cfg.get("list_fill", 0) or 0) if self.list_mode else 0

    @property
    def window_boost(self) -> dict[str, float] | None:
        """`read.window_boost` (ADR 0031): `{delta, slack}` of the anchored-window boost, or None."""
        value = self.read_cfg.get("window_boost", False)
        return None if value is False else window.settings(value)

    @property
    def p3_model(self) -> bool:
        """`read.p3_model` (ADR 0020): p comes from the gate's read model, not the Platt logit."""
        return bool(self.read_cfg.get("p3_model", False))

    @property
    def session_leg(self) -> bool:
        """`retrieval.session_leg` (ADR 0022): one vector per note/session and a leg over them."""
        return bool(self.retrieval.get("session_leg", False))

    def refresh_session(self, node_id: int) -> None:
        """Recompute a note's session vector after its chunks changed (no-op with the leg off)."""
        if self.session_leg:
            sessionvec.refresh(self.conn, self.vec_table, node_id)

    @property
    def resolve_dates(self) -> bool:
        """`ingest.resolve_dates` (ADR 0021): chunks get a derived date annotation at ingest."""
        return bool(self.ingest_cfg.get("resolve_dates", False))

    @property
    def new_checks_existing(self) -> bool:
        """`ingest.new_checks_existing` (ADR 0027 "Switch"): `new:<type>:<Name>` first checks existing
        names of any type. On unless written false."""
        return bool(self.ingest_cfg.get("new_checks_existing", True))

    def ref_date(self, ts: str) -> date:
        """The calendar date of a UTC timestamp in the store's zone: what "today" meant to the
        person who wrote the note."""
        return datetime.fromisoformat(ts).astimezone(ZoneInfo(self.tz)).date()

    def read_model(self) -> ReadModel | None:
        """The read model in force: the gate's when `read.p3_model` is on (a gate without one is
        refused), else None: the P2 Platt gate, whatever the gate file carries."""
        if not self.p3_model or self.gate is None:
            return None
        rm = self.gate.reader
        if rm is None:
            raise ValueError(NO_READ_MODEL)
        return rm

    @classmethod
    def open(
        cls,
        path: str | Path,
        embedder: Embedder,
        *,
        scope: str = "personal",
        fp: str | None = None,
        now: Callable[[], datetime] | None = None,
        tz: str = "UTC",
        host: str | None = None,
        session: str | None = None,
        retrieval: dict[str, Any] | None = None,
        reranker: Reranker | None = None,
        read: dict[str, Any] | None = None,
        ingest: dict[str, Any] | None = None,
        fallback_gate: Gate | None = None,
    ) -> Store:
        """Open (creating if needed) a store. `fp` is the embedder fingerprint naming the vec table;
        without one the fake embedders get a fingerprint from their class and dimension.

        Without a `reranker` recall is the P1 slice (RRF top 5, `uncalibrated`); with one it needs
        the store's gate (`mem calibrate`). `mem serve` always passes a reranker (§3.2 P1 note)."""
        fp = fp or hashlib.sha256(f"{type(embedder).__name__}-{embedder.dim}".encode()).hexdigest()
        conn = init_store(path, scope, fp, embedder.dim)
        try:
            if retrieval and retrieval.get("session_leg"):  # catch up on what changed with it off
                conn.execute("BEGIN IMMEDIATE")
                sessionvec.sync(conn, schema.vec_table_name(fp))
                conn.execute("COMMIT")
            return cls(conn, embedder, scope=scope, vec_table=schema.vec_table_name(fp), tz=tz,
                       host=host, session=session, now=now or (lambda: datetime.now(UTC)),
                       retrieval=retrieval or default_config().retrieval, reranker=reranker,
                       read=read, ingest=ingest, fallback_gate=fallback_gate)  # fmt: skip
        except ValueError:
            conn.close()
            raise

    def close(self) -> None:
        self.conn.close()

    def now_ts(self) -> str:
        return fmt_ts(self.now())

    def add_vector(self, chunk_id: int, ctx: str, text: str) -> None:
        self.add_vectors([(chunk_id, ctx, text)])

    def embed_chunks(self, chunks: Sequence[tuple[int | None, str, str]]) -> list[list[float]]:
        """Document vectors for (chunk_id, ctx, text) rows, in one embedder call."""
        return self.embedder.embed_docs(
            [doc_text(ctx, text, self.embedder) for _, ctx, text in chunks]
        )

    def add_vectors(
        self, chunks: Sequence[tuple[int, str, str]], vecs: Sequence[Sequence[float]] | None = None
    ) -> None:
        """Store vectors for (chunk_id, ctx, text) rows; `vecs` are `embed_chunks` output made
        earlier (write path phase A), else the rows are embedded here."""
        if not chunks:
            return
        vecs = vecs if vecs is not None else self.embed_chunks(chunks)
        self.conn.executemany(
            f"INSERT INTO {self.vec_table}(chunk_id, embedding) VALUES (?, ?)",
            [(cid, sqlite_vec.serialize_float32(v)) for (cid, _, _), v in zip(chunks, vecs,
                                                                             strict=True)],
        )  # fmt: skip

    def query_vector(self, query: str) -> list[float]:
        """The query-prompt embedding; the last one is memoised so the read model's cosines (which
        need the same vector `candidate_pool` used) cost no second embedding call."""
        if self._last_query is None or self._last_query[0] != query:
            vec = self.embedder.embed_query(catalog.query_prompt(self.embedder) + query)
            self._last_query = (query, vec)
        return self._last_query[1]

    def log_event(
        self, tool: str, args: dict[str, Any], result: dict[str, Any], started: float,
        session: str | None, host: str | None,
    ) -> None:  # fmt: skip
        self.conn.execute(
            "INSERT INTO events(at, session, host, tool, args, result, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.now_ts(), session or self.session, host or self.host, tool,
             json.dumps(args, ensure_ascii=False), json.dumps(result, ensure_ascii=False),
             round((time.perf_counter() - started) * 1000)),
        )  # fmt: skip

    def remember(self, **kw: Any) -> ToolResult:
        return write.remember(self, **kw)

    def recall(self, **kw: Any) -> ToolResult:
        return read.recall(self, **kw)

    def get(self, ids: Sequence[str], max_tokens: int = 2000, **kw: Any) -> ToolResult:
        return read.get(self, ids, max_tokens, **kw)

    def link(self, **kw: Any) -> ToolResult:
        return link.link(self, **kw)

    def forget(self, **kw: Any) -> ToolResult:
        return link.forget(self, **kw)

    def subjects(self, **kw: Any) -> ToolResult:
        return subjects.subjects(self, **kw)
