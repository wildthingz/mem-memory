"""The read-time conflict guard (architecture-v2 §3.2 step 7b; ADR 0018, MEM2-159).

B-R5 leakage 0 means zero stale slot memories in a delivered set. The write path surfaces ~97% of
updates; a missed `supersedes` leaves the older value open and both come back on a current-value
read. The guard runs inside recall's final set, after the gate and the limit/budget cut (never
adding a row, A2): when two memories of the set share a subject and the gate's conflict model
(`meta.gate.conf_model`, the write path's scorer) scores the pair as a conflict at the strict read
cut `theta_read`, only the one later in its chain (later `valid_from`; tie: later `recorded_at`,
then id) stays. The earlier one is *hidden*, not touched: it stays in the store, in `get`, in
`include_history`; the response names it with `possibly_superseded_by`, and the recall event logs
the pair so the daily review (`mem.review.read_guard_hidden`) can offer a real `supersedes`.

The pair is scored exactly as the write path would have scored it when the later memory was saved:
the later memory's body is the "new text", the earlier one the candidate, with the generic features
of `mem.confmodel.FEATURES` (`mem.conflicts._features`); the list-relative features see the later
memory's same-subject predecessors in the set. Generic signals only (rule 4): no SYN cue list,
template, relation id or value list.

`theta_read` is fitted by nested leave-one-seed-out CV on SYN dev (`bench syn-fit-read-guard`) with
zero wrongly hidden co-valid memories on the training seeds; it lives in `meta.gate.read_guard`
with the fingerprint of the conflict model it was fitted against. A gate whose `conf_model` changed
is refused until `theta_read` is refitted; with `read.conflict_guard` on, a P3 gate without a
`read_guard` is refused (serve preflight, `bench gate P3`, and recall itself).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mem import conflicts, read

if TYPE_CHECKING:
    from mem.store import Store

VERSION = 1
MAX_SET = 8  # the set the guard sees is recall's final set: ≤ `limit` rows (8 by default)


def fingerprint(theta_read: float, conf_model_fingerprint: str) -> str:
    body = json.dumps([VERSION, theta_read, conf_model_fingerprint], separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReadGuard:
    theta_read: float
    conf_model_fingerprint: str  # the scorer the cut was fitted against

    @classmethod
    def build(cls, theta_read: float, conf_model_fingerprint: str) -> ReadGuard:
        if not isinstance(theta_read, int | float) or not math.isfinite(theta_read):
            raise ValueError(f"read_guard: theta_read must be a finite number, got {theta_read!r}")
        if not conf_model_fingerprint or not isinstance(conf_model_fingerprint, str):
            raise ValueError("read_guard: conf_model_fingerprint is required")
        return cls(float(theta_read), conf_model_fingerprint)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReadGuard:
        """A record without `theta_read`, or whose fingerprint disagrees with it, is refused."""
        if d.get("theta_read") is None:
            raise ValueError(
                "read_guard: theta_read is required (no default read cut; fit it: bench"
                " syn-fit-read-guard, then bench gate-add-read-guard)"
            )
        g = cls.build(d["theta_read"], d.get("conf_model_fingerprint") or "")
        if d.get("fingerprint") != g.fingerprint():
            raise ValueError("read_guard: fingerprint does not match theta_read")
        return g

    def fingerprint(self) -> str:
        return fingerprint(self.theta_read, self.conf_model_fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "theta_read": self.theta_read,
                "conf_model_fingerprint": self.conf_model_fingerprint,
                "fingerprint": self.fingerprint()}  # fmt: skip


@dataclass(frozen=True)
class Pair:
    later: int  # node id of the member later in its chain (kept)
    earlier: int  # node id of the predecessor (hidden when score >= theta_read)
    score: float  # conf_model score of (later as new text, earlier as candidate)
    features: dict[str, float]


@dataclass(frozen=True)
class Member:
    node_id: int
    body: str
    chunk_id: int | None  # the body chunk (the reranker's document)
    valid_from: str
    recorded_at: str
    subjects: frozenset[int]

    @property
    def order(self) -> tuple[str, str, int]:
        return (self.valid_from, self.recorded_at, self.node_id)


def members(conn: sqlite3.Connection, node_ids: Sequence[int]) -> list[Member]:
    """The memory members of a final set, in the set's order; non-memory nodes are left out."""
    if not node_ids:
        return []
    marks = ", ".join("?" * len(node_ids))
    rows = {nid: (body or "", vf, ra) for nid, body, vf, ra in conn.execute(
        f"SELECT id, body, valid_from, recorded_at FROM nodes WHERE kind = 'memory'"
        f" AND id IN ({marks})", list(node_ids))}  # fmt: skip
    chunk = {nid: cid for cid, nid in conn.execute(
        f"SELECT id, node_id FROM chunks WHERE key_kind = 'body' AND node_id IN ({marks})",
        list(node_ids))}  # fmt: skip
    about: dict[int, set[int]] = {nid: set() for nid in rows}
    for src, dst in conn.execute(
        f"SELECT src, dst FROM edges WHERE rel = 'about' AND removed_at IS NULL AND src IN ({marks})",
        list(node_ids)):  # fmt: skip
        if src in about:
            about[src].add(dst)
    return [Member(node_id=nid, body=rows[nid][0], chunk_id=chunk.get(nid),
                   valid_from=rows[nid][1], recorded_at=rows[nid][2],
                   subjects=frozenset(about[nid])) for nid in node_ids if nid in rows]  # fmt: skip


def _vector(store: Store, chunk_id: int | None) -> list[float] | None:
    if chunk_id is None:
        return None
    row = store.conn.execute(
        f"SELECT embedding FROM {store.vec_table} WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return array("f", row[0]).tolist() if row else None


def score_pairs(store: Store, node_ids: Sequence[int]) -> list[Pair]:
    """Every (later, earlier) pair of same-subject memories in `node_ids`, scored by the gate's
    conflict model as the write path scores a candidate of a new memory. Cost: one reranker call
    per member that has same-subject predecessors in the set (≤ `limit` − 1 pairs each)."""
    gate = store.gate
    if gate is None or store.reranker is None:
        return []
    model = gate.model
    if model is None:
        return []
    mem = members(store.conn, node_ids)
    out: list[Pair] = []
    for later in mem:
        if not later.subjects:
            continue
        cands = [m for m in mem if m is not later and m.subjects & later.subjects
                 and m.order < later.order and m.chunk_id is not None]  # fmt: skip
        if not cands:
            continue
        logits = read.rerank(store, later.body, [c.chunk_id for c in cands])
        feats = conflicts._features(store, later.body, _vector(store, later.chunk_id),
                                    set(later.subjects), later.valid_from,
                                    [c.node_id for c in cands], logits)  # fmt: skip
        for c, f in zip(cands, feats, strict=True):
            out.append(Pair(later.node_id, c.node_id, float(model.score(f)), f))
    return out


def hidden_of(pairs: Sequence[Pair], theta_read: float) -> dict[int, tuple[int, float]]:
    """earlier node id -> (the later member that hides it, the score), at `theta_read`. When several
    later members hide the same one, the best-scoring pair names it (ties: the lower node id)."""
    out: dict[int, tuple[int, float]] = {}
    for p in sorted(pairs, key=lambda p: (-p.score, p.later, p.earlier)):
        if p.score >= theta_read and p.earlier not in out:
            out[p.earlier] = (p.later, p.score)
    return out


def apply(store: Store, node_ids: Sequence[int]) -> dict[int, tuple[int, float]]:
    """The guard on a final set: which of `node_ids` to hide, and by whom. Empty when the gate has
    no `read_guard` (the caller refuses such a gate before calling, A2: no silent fallback)."""
    gate = store.gate
    if gate is None or gate.read_guard is None or len(node_ids) < 2:
        return {}
    guard = gate.guard
    assert guard is not None
    return hidden_of(score_pairs(store, node_ids), guard.theta_read)
