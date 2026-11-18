"""Canonical store dump (benchmarks-v2 B-W6 item 1, B-I1): sorted JSONL, one row per line.

`nodes.session` is left out: an MCP session id is minted by the transport, not by the op stream,
so it cannot be part of a byte-identical dump. `events` is left out too (latency is wall time).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

TABLES: dict[str, tuple[str, str | None]] = {
    # table: (ORDER BY, column to drop)
    "nodes": ("id", "session"),
    "edges": ("id", None),
    "chunks": ("id", None),
    "aliases": ("alias_norm, node_id", None),
    "props": ("node_id, key, value", None),
    "link_occurrences": ("id", None),
}


def canonical_dump(db_path: Path) -> bytes:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        lines: list[str] = []
        for table, (order, drop) in TABLES.items():
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                d = {k: row[k] for k in row.keys() if k != drop}  # noqa: SIM118
                lines.append(json.dumps({"t": table, **d}, sort_keys=True, ensure_ascii=False))
        return ("\n".join(lines) + "\n").encode("utf-8")
    finally:
        conn.close()


def dump_sha256(db_path: Path) -> str:
    return hashlib.sha256(canonical_dump(db_path)).hexdigest()


def vectors_sha256(db_path: Path) -> str:
    """sha256 over (chunk_id, fp32 bytes) of the store's vec table, by chunk_id (B-I1)."""
    import sqlite_vec

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        (table,) = conn.execute("SELECT value FROM meta WHERE key = 'vec_table'").fetchone()
        h = hashlib.sha256()
        for cid, emb in conn.execute(f"SELECT chunk_id, embedding FROM {table} ORDER BY chunk_id"):
            h.update(f"{cid}:".encode() + emb)
        return h.hexdigest()
    finally:
        conn.close()
