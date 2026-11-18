"""P6 daily-use tools (architecture-v2 §6, benchmarks-v2 B-E4): the `SessionStart` core index, the
2-minute evening review sample, the unresolved-conflict query, the pairs the read-time conflict
guard hid (ADR 0018) and the skill-tuning counts. All of it reads `nodes`, `edges` and `events` only."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from mem.common import PREFIX, estimate_tokens

REVIEW_N = 10
CONTEXT_N = 3
CORE_HEADER = "# mem core: pinned, then recent"


def core_index(conn: sqlite3.Connection, budget: int = 800) -> str:
    """Pinned memories, then the newest open ones, one row each, as many as fit in `budget`
    tokens (header included). Quarantined memories are left out (P7, `mem.guard`); a store the
    P7 migration has not reached yet has none."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)")}
    held = " AND quarantine IS NULL" if "quarantine" in cols else ""
    rows = conn.execute(
        "SELECT id, type, date, name FROM nodes"
        f" WHERE kind = 'memory' AND status = 'active' AND valid_to IS NULL{held}"
        " ORDER BY pinned DESC, recorded_at DESC, id DESC"
    )  # fmt: skip
    out = [CORE_HEADER]
    used = estimate_tokens(CORE_HEADER)
    for nid, typ, day, name in rows:
        line = f"{PREFIX['memory']}{nid} · {typ} · {day} · {name}"
        used += estimate_tokens(line + "\n")
        if used > budget:
            break
        out.append(line)
    return "\n".join(out)


@dataclass(frozen=True)
class ReviewEvent:
    id: int
    at: str
    session: str | None
    query: str
    rows: list[str]
    context: list[str] = field(default_factory=list)


def _query_of(tool: str, args: dict) -> str | None:
    return (
        args.get("query") if tool == "recall" else args.get("text") if tool == "remember" else None
    )


def event_ref(store: str, event_id: int) -> str:
    """The store-aware name of an event: the two stores number their events independently."""
    return f"{store}:{event_id}"


def sample_key(ref: str) -> str:
    """Sort key of the daily sample: sha1 of the event ref (`store:id`). `bench daily-review`
    uses this same key."""
    return hashlib.sha1(ref.encode()).hexdigest()


def store_name(conn: sqlite3.Connection) -> str:
    """The store a connection reads: its file name without the extension (`personal`, `work`)."""
    path = next((f for _, name, f in conn.execute("PRAGMA database_list") if name == "main"), "")
    return Path(path).stem


def inline_resolution(args: dict) -> bool:
    """A `remember` that carries `supersedes` already resolves its own conflicts."""
    return bool(args.get("supersedes"))


def closes_conflict(new_id: str, tool: str, args: dict) -> bool:
    """Whether a later `link` (naming `new_id` as src or dst) or `forget` (of it) resolves the
    conflicts of the `remember` that saved `new_id`."""
    if tool == "link":
        return new_id in (str(args.get("src")), str(args.get("dst")))
    return tool == "forget" and str(args.get("id")) == new_id


def daily_sample(
    conn: sqlite3.Connection, day: str, n: int = REVIEW_N, store: str | None = None
) -> list[ReviewEvent]:
    """Up to `n` of the day's `recall` events (UTC date `day`), picked by sha1 of `store:id` so
    the same events come back on every run; shown in time order."""
    store = store or store_name(conn)
    found = conn.execute(
        "SELECT id, at, session, args, result FROM events WHERE tool = 'recall' AND at LIKE ?",
        (day + "%",),
    ).fetchall()  # fmt: skip
    found.sort(key=lambda r: sample_key(event_ref(store, r[0])))
    out = []
    for eid, at, session, args, result in sorted(found[:n]):
        before = conn.execute(
            "SELECT tool, args FROM events WHERE session IS ? AND id < ?"
            " AND tool IN ('recall', 'remember') ORDER BY id DESC LIMIT ?",
            (session, eid, CONTEXT_N),
        ).fetchall()  # fmt: skip
        context = [q for t, a in reversed(before) if (q := _query_of(t, json.loads(a or "{}")))]
        ids = json.loads(result or "{}").get("ids", [])
        out.append(ReviewEvent(eid, at, session, json.loads(args or "{}").get("query", ""),
                               [_row(conn, i) for i in ids], context))  # fmt: skip
    return out


def _row(conn: sqlite3.Connection, ident: str) -> str:
    r = conn.execute(
        "SELECT type, date, name FROM nodes WHERE id = ?", (int(ident[1:]),)
    ).fetchone()  # fmt: skip
    return f"{ident} · {r[0]} · {r[1]} · {r[2]}" if r else ident


@dataclass(frozen=True)
class Unresolved:
    event_id: int
    new_id: str
    candidates: list[str]


def unresolved_conflicts(conn: sqlite3.Connection, since: str | None = None) -> list[Unresolved]:
    """B-E4: `remember` events that returned `possible_conflicts` with no `supersedes` argument and
    no later `link` or `forget` event naming the new id in the same session. A session-less event
    can never be resolved."""
    out = []
    saved = conn.execute(
        "SELECT id, session, args, result FROM events WHERE tool = 'remember' AND at >= ?"
        " ORDER BY id",
        (since or "",),
    ).fetchall()  # fmt: skip
    for eid, session, args, result in saved:
        res = json.loads(result or "{}")
        cands = [c["id"] for c in res.get("possible_conflicts") or []]
        if res.get("status") != "saved" or not cands:
            continue
        new = res["id"]
        if inline_resolution(json.loads(args or "{}")):
            continue
        acted = session is not None and any(
            closes_conflict(new, tool, json.loads(raw or "{}"))
            for tool, raw in conn.execute(
                "SELECT tool, args FROM events WHERE session = ? AND id > ?"
                " AND tool IN ('link', 'forget')",
                (session, eid),
            )
        )
        if not acted:
            out.append(Unresolved(eid, new, cands))
    return out


@dataclass(frozen=True)
class Hidden:
    """A pair the read-time conflict guard hid at least once: `earlier` was left out of a recall
    because `later` scored as its conflict. The review offers `link later supersedes earlier`
    (or `duplicate_of` / `relates_to`): the guard itself never writes."""

    earlier: str
    later: str
    times: int
    last_at: str
    last_query: str
    score: float


def read_guard_hidden(conn: sqlite3.Connection, since: str | None = None) -> list[Hidden]:
    """Pairs the read guard hid in `recall` events at/after `since`, still unresolved: no live
    `supersedes` / `duplicate_of` / `relates_to` edge between the two and both still active.
    Most-hidden first, then the latest."""
    seen: dict[tuple[str, str], list] = {}
    for at, args, result in conn.execute(
        "SELECT at, args, result FROM events WHERE tool = 'recall' AND at >= ? ORDER BY id",
        (since or "",),
    ):
        for h in json.loads(result or "{}").get("hidden") or []:
            key = (str(h["id"]), str(h["by"]))
            row = seen.setdefault(key, [0, at, "", float(h.get("score", 0.0))])
            row[0] += 1
            row[1], row[2] = at, json.loads(args or "{}").get("query", "")
            row[3] = max(row[3], float(h.get("score", 0.0)))
    out = []
    for (earlier, later), (n, at, query, score) in seen.items():
        a, b = int(earlier[1:]), int(later[1:])
        active = conn.execute(
            "SELECT count(*) FROM nodes WHERE id IN (?, ?) AND status = 'active'", (a, b)
        ).fetchone()[0]
        linked = conn.execute(
            "SELECT 1 FROM edges WHERE removed_at IS NULL"
            " AND rel IN ('supersedes', 'duplicate_of', 'relates_to')"
            " AND ((src = ? AND dst = ?) OR (src = ? AND dst = ?)) LIMIT 1",
            (a, b, b, a),
        ).fetchone()
        if active == 2 and linked is None:
            out.append(Hidden(earlier, later, n, at, query, score))
    return sorted(out, key=lambda h: (-h.times, h.last_at, h.earlier), reverse=False)


def tuning_counts(conn: sqlite3.Connection, since: str | None = None) -> dict[str, int]:
    """Skill-tuning signals from `events`: calls per tool, empty recalls, saves that came back
    with conflicts, how many of those are still unresolved, and the pairs the read guard hid that
    nobody has linked yet."""
    counts = dict(conn.execute(
        "SELECT tool, count(*) FROM events WHERE at >= ? GROUP BY tool", (since or "",)))  # fmt: skip
    empty = sum(
        1 for (r,) in conn.execute(
            "SELECT result FROM events WHERE tool = 'recall' AND at >= ?", (since or "",))
        if not json.loads(r or "{}").get("ids"))  # fmt: skip
    return {**counts, "empty_recalls": empty,
            "unresolved_conflicts": len(unresolved_conflicts(conn, since)),
            "read_guard_hidden": len(read_guard_hidden(conn, since))}  # fmt: skip
