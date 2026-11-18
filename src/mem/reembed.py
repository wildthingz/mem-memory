"""Model change without mixing vector spaces (architecture-v2 §3.3, A21).

`reembed` fills the new fingerprint's vec table next to the old one; `mem calibrate` fits a gate
for the new `fp_all`; `switch` then moves `meta.vec_table`, `meta.fingerprint` and `meta.gate`
in one transaction. The old table stays until it is dropped later.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from mem import schema, sessionvec
from mem.config import Config
from mem.gate import Gate, gate_json
from mem.store import doc_text
from mem.testing import Embedder

BATCH = 64


def reembed(
    conn: sqlite3.Connection, embedder: Embedder, fp: str, *, session_leg: bool = False
) -> str:
    """Embed every chunk that has no vector yet in `vec_<fp8>` (resumable); return the table.
    With `session_leg` (ADR 0022) the note/session vectors of the new space are rebuilt from the
    chunk vectors at the end (all of them: they are cheap and a resumed run may have stale ones)."""
    table = schema.create_vec_table(conn, fp, embedder.dim)
    todo = conn.execute(
        f"SELECT id, coalesce(ctx, ''), text FROM chunks WHERE id NOT IN"
        f" (SELECT chunk_id FROM {table}) ORDER BY id"
    ).fetchall()
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vecs = embedder.embed_docs([doc_text(ctx, text, embedder) for _, ctx, text in batch])
        conn.executemany(
            f"INSERT INTO {table}(chunk_id, embedding) VALUES (?, ?)",
            [
                (cid, sqlite_vec.serialize_float32(v))
                for (cid, _, _), v in zip(batch, vecs, strict=True)
            ],
        )
        if conn.in_transaction:
            conn.commit()
    if session_leg:
        sessionvec.rebuild(conn, table)
        if conn.in_transaction:
            conn.commit()
    return table


def switch(path: str | Path, cfg: Config, gate: Gate) -> None:
    """Point the store at `cfg`'s vectors and gate, atomically. Refuses a gate fitted for another
    config, and a vec table that is missing or lacks vectors for some chunks."""
    if gate.fingerprint_hash != cfg.fp_all():
        raise ValueError(f"the gate was fitted for fingerprint {gate.fingerprint_hash[:8]}, the new "
                         f"config is {cfg.fp_all()[:8]}: run mem calibrate for the new config")  # fmt: skip
    conn = schema.connect(path)
    try:
        table = cfg.vec_table()
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (table,)).fetchone():
            raise ValueError(f"{path} has no {table}: run mem reembed with the new config first")
        if (
            cfg.session_leg
            and not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (schema.session_table_name(table),)
            ).fetchone()
        ):
            raise ValueError(f"{path} has no {schema.session_table_name(table)}: run mem reembed "
                             "with the new config first")  # fmt: skip
        (missing,) = conn.execute(
            f"SELECT count(*) FROM chunks WHERE id NOT IN (SELECT chunk_id FROM {table})"
        ).fetchone()
        if missing:
            raise ValueError(f"{missing} chunks have no vector in {table}: re-run mem reembed")
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        for key, value in (("vec_table", table), ("fingerprint", json.dumps({"embed": cfg.fp_embed()})),
                           ("gate", gate_json(gate))):  # fmt: skip
            conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))  # fmt: skip
        conn.execute("COMMIT")
    finally:
        conn.close()
