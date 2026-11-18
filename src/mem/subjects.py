"""`subjects` (architecture-v2 §4.1): list or search entities and tags with their alias and
usage counts, so a caller can pick an `#e<id>` for `remember`."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mem.common import PREFIX, MemError, ToolResult, norm

if TYPE_CHECKING:
    from mem.store import Store

MAX_LIMIT = 100


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def subjects(
    store: Store, *, query: str | None = None, type: str | None = None, limit: int = 20,
    session: str | None = None, host: str | None = None,
) -> ToolResult:  # fmt: skip
    started = time.perf_counter()
    if not 1 <= limit <= MAX_LIMIT:
        raise MemError(f"limit must be 1–{MAX_LIMIT}")
    conn = store.conn
    where, params = ["n.kind IN ('entity', 'tag')", "n.status = 'active'"], []
    if type:
        where.append("n.type = ?")
        params.append(type)
    qn = norm(query) if query else ""
    if qn:
        like = "%" + qn.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append("(EXISTS (SELECT 1 FROM aliases a WHERE a.node_id = n.id"
                     " AND a.alias_norm LIKE ? ESCAPE '\\') OR n.name LIKE ? ESCAPE '\\')")  # fmt: skip
        params += [like, like]
    rows = conn.execute(
        "SELECT n.id, n.kind, n.type, n.name,"
        " (SELECT count(DISTINCT e.src) FROM edges e JOIN nodes m ON m.id = e.src"
        "   WHERE e.dst = n.id AND e.rel = 'about' AND e.removed_at IS NULL"
        "   AND m.kind = 'memory' AND m.status = 'active'),"
        " (SELECT count(DISTINCT e.src) FROM edges e JOIN nodes m ON m.id = e.src"
        "   WHERE e.dst = n.id AND e.removed_at IS NULL AND m.kind = 'note'"
        "   AND m.status = 'active')"
        f" FROM nodes n WHERE {' AND '.join(where)}",
        params,
    ).fetchall()
    items = []
    for nid, kind, typ, name, memories, notes in rows:
        aliases = [a for (a,) in conn.execute(
            "SELECT alias FROM aliases WHERE node_id = ? ORDER BY alias_norm", (nid,))
            if norm(a) != norm(name)]  # fmt: skip
        norms = {norm(name), *(norm(a) for a in aliases)}
        rank = 0 if qn in norms else 1 if any(x.startswith(qn) for x in norms) else 2
        items.append((rank if qn else 0, -memories, name.casefold(), nid, {
            "id": f"{PREFIX[kind]}{nid}", "type": typ or kind, "name": name, "aliases": aliases,
            "memories": memories, "notes": notes}))  # fmt: skip
    items.sort(key=lambda t: t[:4])
    found = [t[4] for t in items[:limit]]
    lines = [
        f"{s['id']} · {s['type']} · {s['name']}"
        + (f" (aka {', '.join(s['aliases'])})" if s["aliases"] else "")
        + f" · {_count(s['memories'], 'memory', 'memories')} · {_count(s['notes'], 'note', 'notes')}"
        for s in found
    ]
    out = ToolResult("\n".join(lines) or "no subjects", {"subjects": found})
    store.log_event("subjects", {"query": query, "type": type, "limit": limit},
                    {"ids": [s["id"] for s in found]}, started, session, host)  # fmt: skip
    return out
