"""One vector per note/session node and the vector leg over them (ADR 0022, `retrieval.session_leg`).

A session exceeds the embedder's 512 tokens, so its vector is the L2-normalised mean of its chunk
vectors (already stored; no second embedder pass, nothing truncated, rebuilt from the chunk table
alone). Memories are one chunk already and get none. Everything here is a no-op unless the switch
is on.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np
import sqlite_vec

from mem import schema

__all__ = ["rebuild", "refresh", "session_leg", "sync"]

_KEYS = "('body', 'section', 'fact')"


def refresh(conn: sqlite3.Connection, vec_table: str, node_id: int) -> None:
    """Recompute `node_id`'s session vector from its stored chunk vectors; drop it if none."""
    svec = schema.session_table_name(vec_table)
    blobs = [r[0] for r in conn.execute(
        f"SELECT v.embedding FROM chunks c JOIN {vec_table} v ON v.chunk_id = c.id"
        f" WHERE c.node_id = ? AND c.key_kind IN {_KEYS}", (node_id,))]  # fmt: skip
    conn.execute(f"DELETE FROM {svec} WHERE node_id = ?", (node_id,))
    conn.execute(f"DELETE FROM {svec}_dirty WHERE node_id = ?", (node_id,))
    if not blobs:
        return
    mean = np.mean([np.frombuffer(b, dtype=np.float32) for b in blobs], axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return
    conn.execute(f"INSERT INTO {svec}(node_id, embedding) VALUES (?, ?)",
                 (node_id, sqlite_vec.serialize_float32((mean / norm).tolist())))  # fmt: skip


def rebuild(conn: sqlite3.Connection, vec_table: str) -> int:
    """Create the session table and (re)compute the vector of every note node; return how many
    notes were (re)computed."""
    svec = schema.create_session_table(conn, vec_table, _dim(conn, vec_table))
    _track(conn, svec)
    n = 0
    for (node_id,) in conn.execute(
        "SELECT id FROM nodes WHERE kind = 'note' ORDER BY id"
    ).fetchall():
        refresh(conn, vec_table, node_id)
        n += 1
    conn.execute(f"DELETE FROM {svec}_dirty")
    return n


def sync(conn: sqlite3.Connection, vec_table: str) -> int:
    """Bring the session vectors up to date on a store opened with the leg on: the notes whose
    chunks changed since (with the leg off too: the triggers record them) and the notes without
    one; everything when the store has no record of changes yet. Returns how many were computed."""
    svec = schema.create_session_table(conn, vec_table, _dim(conn, vec_table))
    if _track(conn, svec):
        return rebuild(conn, vec_table)
    todo = [r[0] for r in conn.execute(
        f"SELECT id FROM nodes WHERE kind = 'note' AND (id IN (SELECT node_id FROM {svec}_dirty)"
        f" OR id NOT IN (SELECT node_id FROM {svec})) ORDER BY id")]  # fmt: skip
    for node_id in todo:
        refresh(conn, vec_table, node_id)
    conn.execute(f"DELETE FROM {svec}_dirty")
    return len(todo)


def _track(conn: sqlite3.Connection, svec: str) -> bool:
    """Record the node of every chunk inserted, updated or deleted in `<svec>_dirty`, whatever the
    switch (created only once the leg has been on). True when the tracking is new here."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (f"{svec}_dirty",)).fetchone():
        return False
    conn.execute(f"CREATE TABLE {svec}_dirty(node_id INTEGER PRIMARY KEY)")
    for event, row in (("INSERT", "NEW"), ("UPDATE", "OLD"), ("DELETE", "OLD")):
        conn.execute(
            f"CREATE TRIGGER {svec}_dirty_{event.lower()} AFTER {event} ON chunks BEGIN"
            f" INSERT OR IGNORE INTO {svec}_dirty(node_id) VALUES ({row}.node_id);"
            + (f" INSERT OR IGNORE INTO {svec}_dirty(node_id) VALUES (NEW.node_id);" if event == "UPDATE" else "")
            + " END"
        )  # fmt: skip
    return True


def _dim(conn: sqlite3.Connection, vec_table: str) -> int:
    row = conn.execute(f"SELECT embedding FROM {vec_table} LIMIT 1").fetchone()
    if row:
        return len(row[0]) // 4
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (vec_table,)).fetchone()[0]
    return int(sql.split("float[")[1].split("]")[0])


def session_leg(
    conn: sqlite3.Connection, vec_table: str, qvec: Sequence[float], k: int, per_node_cap: int,
) -> list[int]:  # fmt: skip
    """The extra RRF list: the top `k` sessions by cosine among those with an eligible chunk (the
    `allowed` table, applied inside the scan, before LIMIT), each contributing its eligible chunks
    nearest the query first, at most `per_node_cap` of them; cut to `k` chunks."""
    svec = schema.session_table_name(vec_table)
    q = sqlite_vec.serialize_float32(list(qvec))
    nodes = [r[0] for r in conn.execute(
        f"SELECT node_id FROM {svec} WHERE embedding MATCH ? AND k = ? AND node_id IN"
        " (SELECT c.node_id FROM chunks c JOIN allowed a ON a.chunk_id = c.id)"
        " ORDER BY distance", (q, k))]  # fmt: skip
    out: list[int] = []
    for node_id in nodes:
        out += [r[0] for r in conn.execute(
            f"SELECT c.id FROM chunks c JOIN allowed a ON a.chunk_id = c.id"
            f" JOIN {vec_table} v ON v.chunk_id = c.id WHERE c.node_id = ?"
            " ORDER BY vec_distance_cosine(v.embedding, ?), c.id LIMIT ?",
            (node_id, q, per_node_cap))]  # fmt: skip
        if len(out) >= k:
            break
    return out[:k]
