"""The vault's chunks as the gold set sees them: id `path::chunk_key` <-> store chunk.

A gold id is `source_ref + "::" + chunk_key` (architecture-v2 §2.3). `source_ref` is the
vault-relative path the importer wrote, `chunk_key` the heading path + duplicate index + part. The
index reads the store with plain SQL on the schema (nodes, chunks), so it works on any store the
importer produced and on the stand-in importer of the T0 tests.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SEP = "::"
# LME filler chunks that V50k adds carry `lme:<...>` source refs: they are never gold.
FILLER_PREFIX = "lme:"


def gold_id(path: str, chunk_key: str) -> str:
    return f"{path}{SEP}{chunk_key}"


def split_gold_id(gid: str) -> tuple[str, str]:
    path, sep, key = gid.partition(SEP)
    if not sep or not path or not key:
        raise ValueError(f"not a gold id (path::chunk_key): {gid!r}")
    return path, key


@dataclass(frozen=True)
class Chunk:
    id: int
    gold_id: str
    path: str
    chunk_key: str
    heading_path: str
    ctx: str
    text: str
    content_hash: str
    node_name: str
    store: str = "personal"  # which store holds it (Q3 routing: personal.db or work.db)


class ChunkIndex:
    """Active imported note chunks, keyed by gold id and by store chunk id."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = sorted(chunks, key=lambda c: c.gold_id)
        self.by_gold = {c.gold_id: c for c in self.chunks}
        if len(self.by_gold) != len(self.chunks):
            raise ValueError("duplicate gold ids in the store: (source_ref, chunk_key) not unique")
        # chunk ids are per store: (store, id) -> chunk
        self.by_id = {(c.store, c.id): c for c in self.chunks}
        self.store_of_path = {c.path: c.store for c in self.chunks}

    @classmethod
    def from_conns(cls, conns: dict[str, sqlite3.Connection]) -> ChunkIndex:
        """The chunks of every store; a path lives in one store (gold ids are unique across them)."""
        return cls([c for name, conn in conns.items() for c in cls.from_conn(conn, name).chunks])

    @classmethod
    def from_conn(cls, conn: sqlite3.Connection, store: str = "personal") -> ChunkIndex:
        rows = conn.execute(
            "SELECT c.id, n.source_ref, c.chunk_key, coalesce(c.heading_path, ''),"
            " coalesce(c.ctx, ''), c.text, c.content_hash, n.name"
            " FROM chunks c JOIN nodes n ON n.id = c.node_id"
            " WHERE n.origin = 'import' AND n.status = 'active' AND n.kind IN ('note', 'entity')"
            " AND n.source_ref IS NOT NULL AND n.source_ref NOT LIKE ? AND c.key_kind = 'section'"
            " ORDER BY c.id",
            (FILLER_PREFIX + "%",),
        )
        return cls([Chunk(i, gold_id(p, k), p, k, h, x, t, ch, nm, store)
                    for i, p, k, h, x, t, ch, nm in rows])  # fmt: skip

    def paths(self) -> list[str]:
        return sorted({c.path for c in self.chunks})

    def of_path(self, path: str) -> list[Chunk]:
        return [c for c in self.chunks if c.path == path]

    def __len__(self) -> int:
        return len(self.chunks)
