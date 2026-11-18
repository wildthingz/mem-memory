"""Subject masking: the read model's "does this memory answer THIS question's relation" signal
(MEM2-165, B-R2 (ii)).

Topical relevance cannot tell a memory about the subject from a memory that answers the question's
relation: the subject's name dominates both the embedding cosine and the reranker logit. Masking
the subject's name in the query and in the chunk leaves what is asked and what is said about it.
The masked features are `cosine_masked` (query prompt vs document prompt, both masked) and
`rerank_masked` (the reranker on the masked pair); `mem.readmodel` names them.

The subject is known from the store, never from a cue list: the query's subjects are the active
entities one of whose aliases occurs in the query as a run of words; a chunk's subjects are the
entities its memory is `about` (a name chunk's node is its own subject). All their names and
aliases are masked. The same function runs at fit time (`calibrate.score_pool`) and at serve time
(`read._recall_gated`), so the feature cannot drift between them.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from mem import catalog
from mem.common import norm

if TYPE_CHECKING:
    from mem.store import Store

MASK = "someone"  # replaces every occurrence of a subject's name (part of the model fingerprint)
MAX_NAME_WORDS = 6  # longest alias looked for in a query
MIN_NAME_CHARS = 2
MIN_TOKEN_CHARS = 3  # shortest word of a multi-word name that is masked on its own
_WORD = re.compile(r"\w+", re.UNICODE)


def mask_text(text: str, names: Iterable[str]) -> str:
    """`text` with every whole-word, case-insensitive occurrence of any of `names` replaced by
    `MASK`. Longest names first (so "Ana Lopez" goes before "Ana"); no names, or none present, is
    the identity. Deterministic: depends on the set of names, not on their order."""
    uniq = sorted({n.strip() for n in names if len(n.strip()) >= MIN_NAME_CHARS},
                  key=lambda n: (-len(n), n.casefold(), n))  # fmt: skip
    if not uniq:
        return text
    body = "|".join(re.escape(n) for n in uniq)
    return re.sub(rf"(?<!\w)(?:{body})(?!\w)", MASK, text, flags=re.IGNORECASE)


def entity_names(conn: sqlite3.Connection, node_ids: Iterable[int]) -> list[str]:
    """Canonical names, aliases and the words of the multi-word ones, of the given nodes, sorted."""
    ids = sorted(set(node_ids))
    if not ids:
        return []
    marks = ", ".join("?" * len(ids))
    names = {r[0] for r in conn.execute(f"SELECT name FROM nodes WHERE id IN ({marks})", ids)}
    names |= {r[0] for r in conn.execute(
        f"SELECT alias FROM aliases WHERE node_id IN ({marks})", ids)}  # fmt: skip
    # a multi-word name is also used by its words ("Ana Lopez" is "Ana" in the next sentence)
    names |= {w for n in list(names) if len(_WORD.findall(n)) > 1
              for w in _WORD.findall(n) if len(w) >= MIN_TOKEN_CHARS}  # fmt: skip
    return sorted(n for n in names if n)


def query_subject_ids(conn: sqlite3.Connection, query: str) -> list[int]:
    """Active entities with an alias (or name) that occurs in `query` as a run of words."""
    words = _WORD.findall(norm(query))
    grams = {
        " ".join(words[i : i + n])
        for n in range(1, MAX_NAME_WORDS + 1)
        for i in range(len(words) - n + 1)
    }
    if not grams:
        return []
    marks = ", ".join("?" * len(grams))
    rows = conn.execute(
        "SELECT DISTINCT n.id FROM aliases a JOIN nodes n ON n.id = a.node_id"
        f" WHERE a.alias_norm IN ({marks}) AND n.kind = 'entity' AND n.status = 'active'"
        " ORDER BY n.id", sorted(grams))  # fmt: skip
    return [r[0] for r in rows]


def chunk_subject_ids(conn: sqlite3.Connection, node_id: int) -> list[int]:
    """The entities a chunk's node is about (live `about` edges); an entity is its own subject."""
    kind = conn.execute("SELECT kind FROM nodes WHERE id = ?", (node_id,)).fetchone()
    if kind and kind[0] == "entity":
        return [node_id]
    rows = conn.execute(
        "SELECT DISTINCT e.dst FROM edges e JOIN nodes n ON n.id = e.dst"
        " WHERE e.src = ? AND e.rel = 'about' AND e.removed_at IS NULL AND n.kind = 'entity'"
        " ORDER BY e.dst", (node_id,))  # fmt: skip
    return [r[0] for r in rows]


def masked_query(store: Store, query: str) -> str:
    return mask_text(query, entity_names(store.conn, query_subject_ids(store.conn, query)))


def masked_docs(store: Store, chunk_ids: Sequence[int]) -> list[tuple[str, str]]:
    """(ctx, text) of each chunk with its own subjects' names masked."""
    conn, out = store.conn, []
    cache: dict[int, list[str]] = {}
    for cid in chunk_ids:
        node_id, ctx, text = conn.execute(
            "SELECT node_id, coalesce(ctx, ''), text FROM chunks WHERE id = ?", (cid,)
        ).fetchone()  # fmt: skip
        if node_id not in cache:
            cache[node_id] = entity_names(conn, chunk_subject_ids(conn, node_id))
        out.append((mask_text(ctx, cache[node_id]), mask_text(text, cache[node_id])))
    return out


def masked_cosines(store: Store, query: str, chunk_ids: Sequence[int]) -> list[float]:
    """Cosine of the masked query's query-prompt vector and each masked chunk's document-prompt
    vector (embedded now: the stored vectors hold the unmasked text)."""
    from mem.store import doc_text

    if not chunk_ids:
        return []
    q = np.asarray(
        store.embedder.embed_query(catalog.query_prompt(store.embedder) + masked_query(store, query)),
        dtype=np.float64,
    )  # fmt: skip
    docs = masked_docs(store, chunk_ids)
    vecs = store.embedder.embed_docs([doc_text(c, t, store.embedder) for c, t in docs])
    out = []
    for v in vecs:
        v = np.asarray(v, dtype=np.float64)
        d = float(np.linalg.norm(q) * np.linalg.norm(v))
        out.append(float(np.dot(q, v)) / d if d else 0.0)
    return out


def masked_rerank(store: Store, query: str, chunk_ids: Sequence[int]) -> list[float]:
    """The reranker's raw logit for (masked query, masked ctx + "\\n" + masked text)."""
    if not chunk_ids:
        return []
    assert store.reranker is not None
    docs = [f"{c}\n{t}" for c, t in masked_docs(store, chunk_ids)]
    return [float(z) for z in store.reranker.score(masked_query(store, query), docs)]


def pool_masked(
    store: Store, query: str, chunk_ids: Sequence[int], want: Iterable[str]
) -> dict[str, list[float]]:
    """The store-computed signals named in `want` for one pool: the masked ones ("cosine_masked",
    "rerank_masked") and the relation-phrase ones ("phrase_cosine", "phrase_overlap",
    `mem.relphrase`, MEM2-171). One function for fit time and serve time."""
    from mem import relphrase

    want = set(want)
    out: dict[str, list[float]] = {}
    if "cosine_masked" in want:
        out["cosine_masked"] = masked_cosines(store, query, chunk_ids)
    if "rerank_masked" in want:
        out["rerank_masked"] = masked_rerank(store, query, chunk_ids)
    out.update(relphrase.pool_phrase(store, query, chunk_ids, want))
    return out
