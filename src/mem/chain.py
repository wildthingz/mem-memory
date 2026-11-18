"""Supersession chains (architecture-v2 §3.1 `link` semantics, A1, N1, N2).

A chain is the connected component of live `supersedes` edges, ordered by (`valid_from`,
`recorded_at`, `id`); its members are the active memories in it. A retracted or merged member
drops out but still connects its neighbours, so forgetting the middle of A → B → C leaves A closed
by C instead of reopening it (`forget` "re-derives the chain without it"). `valid_to` is derived: each member's `valid_to` is the next
member's `valid_from`, and the last member is open. Equal `valid_from` gives the empty interval
`[t, t)` (N1), which the half-open predicate never returns.
"""

from __future__ import annotations

import itertools
import sqlite3

from mem.common import MemError

Key = tuple[str, str, int]


def _key(conn: sqlite3.Connection, node_id: int) -> Key:
    return conn.execute(
        "SELECT valid_from, recorded_at, id FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()


def component(conn: sqlite3.Connection, node_id: int) -> list[int]:
    """The chain `node_id` belongs to, oldest first (just the node if it has no live edge). The
    walk crosses inactive memories; only `node_id` and active ones are returned."""
    seen, todo = {node_id}, [node_id]
    while todo:
        cur = todo.pop()
        for (other,) in conn.execute(
            "SELECT CASE WHEN src = ?1 THEN dst ELSE src END FROM edges"
            " WHERE rel = 'supersedes' AND removed_at IS NULL AND (src = ?1 OR dst = ?1)",
            (cur,),
        ):
            if other not in seen:
                seen.add(other)
                todo.append(other)
    marks = ", ".join("?" * len(seen))
    return [
        r[0]
        for r in conn.execute(
            f"SELECT id FROM nodes WHERE id IN ({marks}) AND (status = 'active' OR id = ?)"
            " ORDER BY valid_from, recorded_at, id",
            (*seen, node_id),
        )
    ]


def rederive(conn: sqlite3.Connection, node_id: int) -> list[str]:
    """Set `valid_to` along the chain of `node_id`; returns the side effects, e.g.
    `m302 valid_to → 2026-11-01`. Must run inside the caller's write transaction."""
    chain = component(conn, node_id)
    want = {a: _key(conn, b)[0] for a, b in itertools.pairwise(chain)}
    want[chain[-1]] = None
    effects, reopen = [], []
    for nid in chain:
        (cur,) = conn.execute("SELECT valid_to FROM nodes WHERE id = ?", (nid,)).fetchone()
        if cur == want[nid]:
            continue
        effects.append(f"m{nid} valid_to → {want[nid] or 'open'}")
        if want[nid] is None:
            reopen.append(nid)
        else:
            conn.execute("UPDATE nodes SET valid_to = ? WHERE id = ?", (want[nid], nid))
    for nid in reopen:  # close first, so a same-text pair never has two open members
        try:
            conn.execute("UPDATE nodes SET valid_to = NULL WHERE id = ?", (nid,))
        except sqlite3.IntegrityError as err:
            if "content_hash" not in str(err):
                raise
            (other,) = conn.execute(
                "SELECT n2.id FROM nodes n JOIN nodes n2 ON n2.content_hash = n.content_hash"
                " AND n2.id <> n.id AND n2.kind = 'memory' AND n2.status = 'active'"
                " AND n2.valid_to IS NULL WHERE n.id = ?",
                (nid,),
            ).fetchone()
            raise MemError(
                f"m{nid} would reopen while m{other} has the same text: "
                f"link m{other} supersedes m{nid} or duplicate_of first"
            ) from err
    return effects


def supersede(
    conn: sqlite3.Connection, new_id: int, old_id: int, origin: str, now: str
) -> tuple[list[str], bool]:
    """`link(new, 'supersedes', old)`: add the edge oriented newer → older, re-derive the chain.
    Returns (side effects, whether the call's direction was swapped)."""
    if new_id == old_id:
        raise MemError("a memory cannot supersede itself")
    for nid in (new_id, old_id):
        row = conn.execute("SELECT kind, status, origin, quarantine FROM nodes WHERE id = ?",
                           (nid,)).fetchone()  # fmt: skip
        if row is None or row[0] != "memory":
            raise MemError(f"m{nid} is not a memory: supersedes links memories (m<id>)")
        if row[1] != "active":
            raise MemError(f"m{nid} is not active ({row[1]})")
        if row[2] == "import":
            raise MemError(f"m{nid} is imported: edit the vault")
        if row[3] is not None:
            raise MemError(f"m{nid} is quarantined ({row[3]}): the person reviews it first")
    swapped = _key(conn, new_id) < _key(conn, old_id)
    src, dst = (old_id, new_id) if swapped else (new_id, old_id)
    conn.execute(
        "INSERT OR IGNORE INTO edges(src, dst, rel, origin, recorded_at)"
        " VALUES (?, ?, 'supersedes', ?, ?)",
        (src, dst, origin, now),
    )
    return rederive(conn, new_id), swapped
