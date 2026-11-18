"""`link` and `forget` (architecture-v2 §3.1 `link` semantics, A1, A11, N1, N2).

Both run in one `BEGIN IMMEDIATE` transaction: any error (a reopen collision, a rule broken half
way through a merge) rolls everything back. Errors are `MemError`s that say how to fix the call.
Imported (source-owned) nodes may be the `dst` of an edge, never mutated: "edit the vault".
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from mem import chain
from mem.common import PREFIX, MemError, ToolResult, norm, parse_id

if TYPE_CHECKING:
    from mem.store import Store

IMPORT_ONLY = ("part_of",)
MERGES = ("duplicate_of", "same_as")
_KINDS = {"m": "memory", "e": "entity", "n": "note", "t": "tag"}
VAULT = "{ref} is imported: edit the vault"


@contextlib.contextmanager
def _write(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _node(conn: sqlite3.Connection, ident: str) -> dict[str, Any]:
    parsed = parse_id(ident)
    if parsed is None or parsed[0] not in _KINDS:
        raise MemError(f"{ident!r} is not a node id: use m<id>, e<id>, n<id> or t<id>")
    row = conn.execute(
        "SELECT id, kind, status, origin, quarantine FROM nodes WHERE id = ?", (parsed[1],)
    ).fetchone()
    if row is None or row[1] != _KINDS[parsed[0]]:
        raise MemError(f"{ident} does not exist")
    return {"id": row[0], "kind": row[1], "status": row[2], "origin": row[3],
            "quarantine": row[4], "ref": f"{PREFIX[row[1]]}{row[0]}"}  # fmt: skip


def _valid_rels(conn: sqlite3.Connection) -> str:
    return ", ".join(
        r for (r,) in conn.execute("SELECT rel FROM rels ORDER BY rowid") if r not in IMPORT_ONLY
    )


def _edge_dict(conn: sqlite3.Connection, edge_id: int) -> dict[str, Any]:
    eid, src, dst, rel, origin, note, skind, dkind = conn.execute(
        "SELECT e.id, e.src, e.dst, e.rel, e.origin, e.note, s.kind, d.kind FROM edges e"
        " JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst WHERE e.id = ?",
        (edge_id,),
    ).fetchone()
    return {"id": eid, "src": f"{PREFIX[skind]}{src}", "rel": rel, "dst": f"{PREFIX[dkind]}{dst}",
            "origin": origin, "note": note}  # fmt: skip


def _live_edge(conn: sqlite3.Connection, src: int, dst: int, rel: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM edges WHERE src = ? AND dst = ? AND rel = ? AND removed_at IS NULL",
        (src, dst, rel),
    ).fetchone()
    return row[0] if row else None


def _rederive(conn: sqlite3.Connection, members: list[int]) -> list[str]:
    """Re-derive each distinct chain among `members` (the nodes that were in a chain before it
    changed); inactive members are skipped, they are out of the chain."""
    covered: set[int] = set()
    effects: list[str] = []
    for nid in members:
        if nid in covered or conn.execute(
            "SELECT status FROM nodes WHERE id = ?", (nid,)
        ).fetchone()[0] != "active":  # fmt: skip
            continue
        covered.update(chain.component(conn, nid))
        effects += chain.rederive(conn, nid)
    return effects


def _reply(ok_text: str, edge: dict[str, Any] | None, effects: list[str]) -> ToolResult:
    lines = [ok_text, *effects]
    return ToolResult("\n".join(lines),
                      {"ok": True, "edge": edge, "side_effects": effects, "error": None})  # fmt: skip


def link(
    store: Store, *, src: str, rel: str, dst: str, note: str | None = None, remove: bool = False,
    session: str | None = None, host: str | None = None,
) -> ToolResult:  # fmt: skip
    started = time.perf_counter()
    args = {"src": src, "rel": rel, "dst": dst, "note": note, "remove": remove}
    conn = store.conn
    with _write(conn):
        out = _link(store, src, rel, dst, note, remove)
    store.log_event("link", args, out.structured, started, session, host)
    return out


def _link(store: Store, src: str, rel: str, dst: str, note: str | None, remove: bool) -> ToolResult:
    conn, now = store.conn, store.now_ts()
    a, b = _node(conn, src), _node(conn, dst)
    spec = conn.execute(
        "SELECT src_kinds, dst_kinds FROM rels WHERE rel = ?", (rel,)
    ).fetchone()  # fmt: skip
    if spec is None or rel in IMPORT_ONLY:
        why = "is made by the vault importer only" if spec else "is not a relation"
        raise MemError(f"{rel!r} {why}: valid rels are {_valid_rels(conn)}")
    if a["id"] == b["id"]:
        raise MemError(f"{src}: a node cannot link to itself")
    if remove:
        return _remove(conn, a, b, rel, now)
    for n in (a, b):  # P7 (`mem.guard`): no rel may tie a quarantined memory into the graph
        if n["quarantine"] is not None:
            raise MemError(f"{n['ref']} is quarantined ({n['quarantine']}): the person reviews it"
                           " first (`mem quarantine`)")  # fmt: skip

    live = _live_edge(conn, a["id"], b["id"], rel) or (
        _live_edge(conn, b["id"], a["id"], rel) if rel == "supersedes" else None
    )
    if live:  # idempotent: the same edge again changes nothing
        return _reply(
            f"ok · {a['ref']} {rel} {b['ref']} (already linked)", _edge_dict(conn, live), []
        )

    swapped = False
    if rel == "same_as":
        if a["origin"] == "import" and b["origin"] == "import":
            raise MemError(f"{VAULT.format(ref=f'{a["ref"]} and {b["ref"]}')}: both are imported")
        if a["origin"] == "import":
            a, b, swapped = b, a, True
    elif rel in ("supersedes", "duplicate_of"):
        for n in (a, b):
            if n["origin"] == "import":
                raise MemError(VAULT.format(ref=n["ref"]))
    elif a["origin"] == "import":
        raise MemError(VAULT.format(ref=a["ref"]))
    for n, kinds in ((a, spec[0]), (b, spec[1])):
        if kinds != "*" and n["kind"] not in kinds.split(","):
            raise MemError(
                f"{rel} links {spec[0]} → {spec[1]}, but {n['ref']} is a {n['kind']}: "
                f"valid rels are {_valid_rels(conn)}"
            )
    for n in (a, b):
        if n["status"] != "active":
            raise MemError(f"{n['ref']} is not active ({n['status']})")

    effects: list[str] = []
    if rel == "supersedes":
        effects, swapped = chain.supersede(conn, a["id"], b["id"], "agent", now)
        newer, older = (b, a) if swapped else (a, b)
        edge_id = _live_edge(conn, newer["id"], older["id"], rel)
    elif rel == "duplicate_of":
        members = chain.component(conn, a["id"])
        conn.execute("UPDATE nodes SET status = 'duplicate', status_at = ? WHERE id = ?",
                     (now, a["id"]))  # fmt: skip
        conn.execute(
            "INSERT OR IGNORE INTO edges(src, dst, rel, origin, recorded_at)"
            " SELECT ?, dst, 'about', 'agent', ? FROM edges"
            " WHERE src = ? AND rel = 'about' AND removed_at IS NULL",
            (b["id"], now, a["id"]),
        )
        effects = _rederive(conn, members)
        edge_id = _add_edge(conn, a, b, rel, note, now)
    elif rel == "same_as":
        effects = _merge_entities(conn, a, b, now)
        edge_id = _add_edge(conn, a, b, rel, note, now)
    else:
        edge_id = _add_edge(conn, a, b, rel, note, now)
    assert edge_id is not None
    if swapped:
        effects.append("direction swapped")
    return _reply(f"ok · {a['ref']} {rel} {b['ref']}", _edge_dict(conn, edge_id), effects)


def _add_edge(
    conn: sqlite3.Connection, a: dict[str, Any], b: dict[str, Any], rel: str, note: str | None,
    now: str,
) -> int:  # fmt: skip
    cur = conn.execute(
        "INSERT INTO edges(src, dst, rel, origin, recorded_at, note) VALUES (?, ?, ?, 'agent', ?, ?)",
        (a["id"], b["id"], rel, now, note),
    )
    assert cur.lastrowid is not None
    return cur.lastrowid


def _merge_entities(
    conn: sqlite3.Connection, a: dict[str, Any], b: dict[str, Any], now: str
) -> list[str]:  # fmt: skip
    """Merge entity `a` into `b` without tripping a unique index (aliases, `edges_live`)."""
    s, d = a["id"], b["id"]
    (name,) = conn.execute("SELECT name FROM nodes WHERE id = ?", (s,)).fetchone()
    conn.execute(
        "INSERT OR IGNORE INTO aliases(alias, alias_norm, node_id, origin) VALUES (?, ?, ?, 'agent')",
        (name, norm(name), d),
    )
    conn.execute(
        "INSERT OR IGNORE INTO aliases(alias, alias_norm, node_id, origin)"
        " SELECT alias, alias_norm, ?, origin FROM aliases WHERE node_id = ?",
        (d, s),
    )
    conn.execute("DELETE FROM aliases WHERE node_id = ?", (s,))
    moved = dropped = 0
    for eid, esrc, edst, erel in conn.execute(
        "SELECT id, src, dst, rel FROM edges WHERE removed_at IS NULL AND (src = ?1 OR dst = ?1)",
        (s,),
    ).fetchall():
        nsrc, ndst = (d if esrc == s else esrc), (d if edst == s else edst)
        if nsrc == ndst or _live_edge(conn, nsrc, ndst, erel):  # between the two / already there
            conn.execute("UPDATE edges SET removed_at = ? WHERE id = ?", (now, eid))
            dropped += 1
        else:
            conn.execute("UPDATE edges SET src = ?, dst = ? WHERE id = ?", (nsrc, ndst, eid))
            moved += 1
    conn.execute("UPDATE nodes SET status = 'duplicate', status_at = ? WHERE id = ?", (now, s))
    return [f"{a['ref']} merged into {b['ref']}: {moved} edges re-pointed, {dropped} dropped"]


def _remove(
    conn: sqlite3.Connection, a: dict[str, Any], b: dict[str, Any], rel: str, now: str
) -> ToolResult:  # fmt: skip
    if rel in MERGES:
        raise MemError(f"a {rel} merge is not undone by remove: nothing was changed")
    edge_id = _live_edge(conn, a["id"], b["id"], rel)
    if edge_id is None and rel == "supersedes":
        edge_id = _live_edge(conn, b["id"], a["id"], rel)
    if edge_id is None:
        return _reply(
            f"ok · no live {a['ref']} {rel} {b['ref']} edge (nothing to remove)", None, []
        )
    (origin,) = conn.execute("SELECT origin FROM edges WHERE id = ?", (edge_id,)).fetchone()
    if origin == "import":
        raise MemError(f"{a['ref']} {rel} {b['ref']} comes from the vault: edit the vault")
    edge = _edge_dict(conn, edge_id)
    members = chain.component(conn, a["id"]) if rel == "supersedes" else []
    conn.execute("UPDATE edges SET removed_at = ? WHERE id = ?", (now, edge_id))
    effects = _rederive(conn, [a["id"], b["id"], *members])
    return _reply(f"ok · removed {a['ref']} {rel} {b['ref']}", edge, effects)


def forget(
    store: Store, *, id: str, reason: str, session: str | None = None, host: str | None = None
) -> ToolResult:  # fmt: skip
    started = time.perf_counter()
    conn = store.conn
    if not reason.strip():
        raise MemError("reason is required: say why the memory is wrong")
    with _write(conn):
        node = _node(conn, id)
        if node["origin"] == "import":
            raise MemError(VAULT.format(ref=node["ref"]))
        if node["kind"] != "memory":
            raise MemError(f"{id} is not a memory: forget retracts memories (m<id>)")
        effects: list[str] = []
        if node["status"] == "retracted":
            pass
        elif node["status"] != "active":
            raise MemError(f"{id} is not active ({node['status']})")
        else:
            members = chain.component(conn, node["id"])
            conn.execute(
                "UPDATE nodes SET status = 'retracted', status_at = ? WHERE id = ?",
                (store.now_ts(), node["id"]),
            )
            conn.execute(
                "INSERT OR REPLACE INTO props(node_id, key, value, origin)"
                " VALUES (?, 'retracted_reason', ?, 'agent')",
                (node["id"], reason.strip()),
            )
            effects = _rederive(conn, members)
    out = ToolResult(
        "\n".join([f"ok · {node['ref']} retracted", *effects]),
        {"ok": True, "id": node["ref"], "status": "retracted"},
    )
    store.log_event("forget", {"id": id, "reason": reason}, out.structured | {"effects": effects},
                    started, session, host)  # fmt: skip
    return out
