"""BM25 top-k on an imported store with no model: the human half of the no-answer check.

The same eligible set and FTS leg `mem.read.recall` uses (`read.build_allowed`, `read.fts_leg`), so
what the person sees is what our own BM25 leg would return for the question under its scope
filters. `about` goes through the read path's exact lookup; an unknown name is dropped like there.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from bench.gold.chunks import ChunkIndex


def bm25_top(
    conns: dict[str, sqlite3.Connection], index: ChunkIndex, store: str, query: str,
    filters: dict[str, Any], k: int, t: str,
) -> list[str]:  # fmt: skip
    from mem import read

    conn = conns[store]
    about_ids, _ = read._resolve_about(conn, filters.get("about") or [])
    read.build_allowed(conn, t, types=filters.get("type") or None, since=filters.get("since"),
                       until=filters.get("until"), about_ids=about_ids or None)  # fmt: skip
    return [index.by_id[(store, c)].gold_id for c in read.fts_leg(conn, query, k)
            if (store, c) in index.by_id]  # fmt: skip
