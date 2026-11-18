"""Entity resolver `lookup(name)` (architecture-v2 §3.1, A8): read-only, three outcomes.

An int is the entity to link, a list of (id, type, name) rows is the set to ask the caller about,
`None` means no candidate resembles the mention (create it in phase B).

Fuzzy matches are never merged silently: only a typo-level match auto-links. Token-subset names
("Ana Lopez" vs "Ana") make the name ambiguous.
"""

from __future__ import annotations

import re
import sqlite3
from typing import TYPE_CHECKING

import sqlite_vec

from mem.common import norm

if TYPE_CHECKING:
    from mem.store import Store

Row = tuple[int, str, str]
TRIGRAM_K = 10
NAME_K = 5
THETA_NAME = 0.85  # provisional (§3.1 step 6)
RATIO_MIN = 0.75
SHARED_TOKEN = 3
NEAR = 3  # no other candidate within this distance of a typo match
_WORD = re.compile(r"\w+", re.UNICODE)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def typo_limit(mention: str) -> int:
    """Largest edit distance that still counts as a typo for a mention of this length."""
    return 2 if len(mention) >= 12 else 1 if len(mention) >= 5 else 0


def exact(conn: sqlite3.Connection, alias_norm: str) -> list[Row]:
    return conn.execute(
        "SELECT n.id, n.type, n.name FROM aliases a JOIN nodes n ON n.id = a.node_id"
        " WHERE a.alias_norm = ? AND n.kind = 'entity' AND n.status = 'active' ORDER BY n.id",
        (alias_norm,),
    ).fetchall()


def _trigram_leg(conn: sqlite3.Connection, mention: str) -> list[int]:
    grams = dict.fromkeys(mention[i : i + 3] for i in range(len(mention) - 2))
    if not grams:
        return []
    match = " OR ".join('"' + g.replace('"', '""') + '"' for g in grams)
    rows = conn.execute(
        "SELECT a.node_id FROM aliases_tri t JOIN aliases a ON a.rowid = t.rowid"
        " JOIN nodes n ON n.id = a.node_id WHERE aliases_tri MATCH ?"
        " AND n.kind = 'entity' AND n.status = 'active' ORDER BY bm25(aliases_tri) LIMIT ?",
        (match, TRIGRAM_K),
    )
    return list(dict.fromkeys(r[0] for r in rows))


def _name_leg(store: Store, mention: str) -> dict[int, float]:
    """Node id → cosine for the nearest name chunks of active entities (independent of trigrams)."""
    conn = store.conn
    if not conn.execute("SELECT 1 FROM chunks WHERE key_kind = 'name' LIMIT 1").fetchone():
        return {}
    rows = conn.execute(
        f"SELECT c.node_id, v.distance FROM {store.vec_table} v JOIN chunks c ON c.id = v.chunk_id"
        " WHERE v.embedding MATCH ? AND v.k = ? AND v.chunk_id IN (SELECT c2.id FROM chunks c2"
        " JOIN nodes n ON n.id = c2.node_id WHERE c2.key_kind = 'name' AND n.kind = 'entity'"
        " AND n.status = 'active')",
        (sqlite_vec.serialize_float32(store.query_vector(mention)), NAME_K),
    ).fetchall()
    return {nid: 1.0 - dist for nid, dist in rows}


def lookup(store: Store, mention: str) -> int | list[Row] | None:
    conn, m = store.conn, norm(mention)
    found = exact(conn, m)
    if len(found) == 1:
        return found[0][0]
    if found:
        return found
    cosine = _name_leg(store, m)
    ids = list(dict.fromkeys([*_trigram_leg(conn, m), *cosine]))
    tokens = {w for w in _WORD.findall(m) if len(w) >= SHARED_TOKEN}
    dist: dict[int, int] = {}
    relevant: list[Row] = []
    for nid in sorted(ids):
        typ, name = conn.execute("SELECT type, name FROM nodes WHERE id = ?", (nid,)).fetchone()
        names = {norm(name), *(r[0] for r in conn.execute(
            "SELECT alias_norm FROM aliases WHERE node_id = ?", (nid,)))}  # fmt: skip
        dist[nid] = min(levenshtein(m, n) for n in names)
        ratio = max(1 - levenshtein(m, n) / max(len(m), len(n)) for n in names)
        shares = any(tokens & set(_WORD.findall(n)) for n in names)
        if shares or ratio >= RATIO_MIN or cosine.get(nid, 0.0) >= THETA_NAME:
            relevant.append((nid, typ, name))
    limit = typo_limit(m)
    if limit:
        near = [n for n, d in dist.items() if d <= limit]
        if len(near) == 1 and all(d > NEAR for n, d in dist.items() if n != near[0]):
            return near[0]
    return relevant or None
