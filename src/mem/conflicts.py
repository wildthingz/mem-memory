"""Conflict candidates for `remember` (architecture-v2 §3.1 A5-A6, A22): deterministic, advisory.

(a) The best 20 memories about the resolved existing subjects, ranked by RRF(BM25, dense) of the
new text *within that set*, so recency plays no part and an old attribute fact is never crowded
out by newer facts about the same person. (b) The global hybrid top 10 over the other active
memories. Every picked candidate gets generic features (reranker logit, embedding cosine, word
overlap, validity, subjects; `confmodel.FEATURES`) and the gate's small logistic model scores it.
The adaptive list cut (`confmodel.list_cut`, MEM2-154): a candidate stays when its score is at or
above `θ_conf` AND within `delta` of the list's best score; the best `top_k` (≤ 5) go out. `top_k`
and `delta` are part of `conf_model` (fingerprinted). A gate with `θ_conf` but no coefficients is
refused at load (`Gate.__post_init__`): there is no fallback scorer. The one run without
coefficients is the dev export (`bench syn-score-dev`, `θ_conf = -inf`, no `conf_model`): it cuts
nothing and returns every pick with its features, ordered by the raw rerank logit only so the
record order is stable; the fit re-scores from the recorded features.
"""

from __future__ import annotations

import math
import re
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mem import read

if TYPE_CHECKING:
    from mem.store import Store

SUBJECT_K = 20
GLOBAL_K = 10
MAX_OUT = 5  # == confmodel.MAX_TOP_K: the hard cap `list_cut` applies


@dataclass(frozen=True)
class Conflict:
    node_id: int
    score: float
    hint: str
    features: dict[str, float] | None = None  # only the unfiltered dev run (θ_conf = -inf)
    valid: tuple[str, str | None] | None = None  # (valid_from, valid_to) at pick time; dev run only


def _top_nodes(store: Store, text: str, k: int) -> list[tuple[int, int]]:
    """(node_id, best chunk_id) of the top k nodes over the current `allowed` set."""
    conn, cfg = store.conn, store.retrieval
    lexical = read.fts_leg(conn, text, cfg["leg_k"])
    dense = read.vec_leg(conn, store.vec_table, store.query_vector(text), cfg["leg_k"])
    out: dict[int, int] = {}
    for cid, _ in read.rrf([lexical, dense], k=cfg["rrf_k"]):
        (nid,) = conn.execute("SELECT node_id FROM chunks WHERE id = ?", (cid,)).fetchone()
        out.setdefault(nid, cid)
        if len(out) == k:
            break
    return list(out.items())


def _subjects(conn: sqlite3.Connection, node_id: int) -> dict[int, str]:
    return dict(conn.execute(
        "SELECT e.dst, n.name FROM edges e JOIN nodes n ON n.id = e.dst"
        " WHERE e.src = ? AND e.rel = 'about' AND e.removed_at IS NULL", (node_id,)))  # fmt: skip


def _hint(conn: sqlite3.Connection, node_id: int, subjects: set[int], valid_from: str) -> str:
    theirs = _subjects(conn, node_id)
    shared = sorted(theirs[i] for i in subjects & theirs.keys())
    parts = []
    if shared and set(theirs) == subjects:
        parts.append("same_subjects")
    elif shared:
        parts.append("shares:" + ",".join(shared))
    (vf,) = conn.execute("SELECT valid_from FROM nodes WHERE id = ?", (node_id,)).fetchone()
    parts.append("older" if vf < valid_from else "newer" if vf > valid_from else "same instant")
    return ", ".join(parts)


_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


def _features(
    store: Store, text: str, vec: Sequence[float] | None, subjects: set[int], valid_from: str,
    nodes: list[int], logits: list[float], mem_type: str | None = None,
) -> list[dict[str, float]]:  # fmt: skip
    """Feature rows of the picked nodes (order kept); the list-relative ones see every pick."""
    conn = store.conn
    if vec is None:
        vec = store.embed_chunks([(None, "", text)])[0]
    marks = ", ".join("?" * len(nodes))
    body = {nid: (cid, t) for cid, nid, t in conn.execute(
        f"SELECT id, node_id, text FROM chunks WHERE key_kind = 'body' AND node_id IN ({marks})",
        nodes)}  # fmt: skip
    vecs = {}
    for cid, blob in conn.execute(
        f"SELECT chunk_id, embedding FROM {store.vec_table} WHERE chunk_id IN"
        f" ({', '.join('?' * len(body))})", [c for c, _ in body.values()]):  # fmt: skip
        vecs[cid] = array("f", blob).tolist()
    meta = {nid: (vf, vt, ty) for nid, vf, vt, ty in conn.execute(
        f"SELECT id, valid_from, valid_to, type FROM nodes WHERE id IN ({marks})", nodes)}  # fmt: skip
    mine = _words(text)
    cos = []
    for nid in nodes:
        cid = body[nid][0] if nid in body else None
        cos.append(_cosine(vec, vecs[cid]) if cid in vecs else 0.0)
    best_cos, best_z = max(cos), max(logits)
    order = sorted(range(len(nodes)), key=lambda i: (-cos[i], nodes[i]))
    rank = {i: r for r, i in enumerate(order)}
    rows = []
    for i, nid in enumerate(nodes):
        theirs = _words(body[nid][1]) if nid in body else set()
        both = len(mine & theirs)
        vf, vt, ty = meta[nid]
        rows.append({
            "rerank": logits[i], "rerank_gap": best_z - logits[i], "cosine": cos[i],
            "cos_gap": best_cos - cos[i], "cos_rank": float(rank[i]),
            "jaccard": both / len(mine | theirs) if mine | theirs else 0.0,
            "new_in_old": both / len(mine) if mine else 0.0,
            "old_in_new": both / len(theirs) if theirs else 0.0,
            "open": float(vt is None), "older": float(vf < valid_from),
            "same_instant": float(vf == valid_from),
            "shared_subjects": len(subjects & _subjects(conn, nid).keys()) / len(subjects)
            if subjects else 0.0,
            # the store's own memory type (fact | preference | event | ...), as the caller set it:
            # a same-subject memory of another kind that shares surface words is not the value
            # the new one replaces (MEM2-156 round 5, ADR 0019)
            "same_type": float(mem_type is not None and ty == mem_type),
        })  # fmt: skip
    return rows


def find(
    store: Store, text: str, subject_ids: Sequence[int], valid_from: str, chash: str,
    vec: Sequence[float] | None = None, mem_type: str | None = None,
) -> list[Conflict]:  # fmt: skip
    """`vec`: the new memory's body document vector (write path phase A), else embedded here.
    `mem_type`: the new memory's type (`same_type` feature); None scores every candidate 0 there."""
    gate = store.gate
    if store.reranker is None or gate is None or gate.theta_conf is None:
        return []
    conn = store.conn
    picked: dict[int, int] = {}
    if subject_ids:
        read.build_allowed(conn, None, kinds=["memory"], about_ids=list(subject_ids))
        picked.update(_top_nodes(store, text, SUBJECT_K))
    read.build_allowed(conn, None, kinds=["memory"])
    if subject_ids:
        marks = ", ".join("?" * len(subject_ids))
        conn.execute(
            "DELETE FROM allowed WHERE chunk_id IN (SELECT c.id FROM chunks c WHERE c.node_id IN"
            f" (SELECT src FROM edges WHERE rel = 'about' AND removed_at IS NULL"
            f" AND dst IN ({marks})))",
            list(subject_ids),
        )
    for nid, cid in _top_nodes(store, text, GLOBAL_K):
        picked.setdefault(nid, cid)
    same_text = {r[0] for r in conn.execute(
        "SELECT id FROM nodes WHERE content_hash = ?", (chash,))}  # fmt: skip
    picked = {n: c for n, c in picked.items() if n not in same_text}
    nodes = list(picked)
    if not nodes:
        return []
    logits = read.rerank(store, text, list(picked.values()))
    subjects = set(subject_ids)
    feats = _features(store, text, vec, subjects, valid_from, nodes, logits, mem_type)
    model = gate.model
    if gate.theta_conf == -math.inf:
        # Dev export (bench syn-score-dev): no cut, every pick with its features. Without a
        # conf_model the order is the raw rerank logit; that is a record order, not a scorer: the
        # fit re-scores every candidate from the recorded features.
        scores = [model.score(f) if model else f["rerank"] for f in feats]
        ranked = sorted(range(len(nodes)), key=lambda i: (-scores[i], nodes[i]))
        # Each candidate's validity at pick time goes out with its features (MEM2-156 round 4), so
        # store-time signals the model does not have yet can be computed offline from the records.
        marks = ", ".join("?" * len(nodes))
        valid = {nid: (vf, vt) for nid, vf, vt in conn.execute(
            f"SELECT id, valid_from, valid_to FROM nodes WHERE id IN ({marks})", nodes)}  # fmt: skip
        return [Conflict(nodes[i], round(scores[i], 2), _hint(conn, nodes[i], subjects, valid_from),
                         feats[i], valid[nodes[i]]) for i in ranked]  # fmt: skip
    assert model is not None  # Gate refuses theta_conf without coefficients
    scores = [model.score(f) for f in feats]
    kept = model.cut(scores, gate.theta_conf, nodes)  # theta, then delta from the top, then top_k
    return [Conflict(nodes[i], round(scores[i], 2), _hint(conn, nodes[i], subjects, valid_from))
            for i in kept]  # fmt: skip
