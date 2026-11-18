"""Two-endpoint query splitting (MEM2-213, ADR 0026): the patterns and the merge.

A question that names two events ("between A and B", "how long had I been A when B", "A or B") is
split into one sub-query per event; `mem.read` runs the pool, rerank and gate for each and
`merge_endpoints` combines what each kept. Patterns only, frozen from the dry run (MEM2-209): no
type labels, no dataset cue lists, no LLM; when unclear there is no split.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["MAX_SPLIT_CHARS", "Kept", "Split", "merge_endpoints", "split_endpoints"]


@dataclass(frozen=True)
class Split:
    pattern: str
    parts: list[str]


@dataclass(frozen=True)
class Kept:
    """A chunk a sub-query's gate kept: its p there, its node and the node's date (None: undated)."""

    chunk_id: int
    p: float
    node_id: int
    date: str | None


# Longer questions are not split: the patterns' cost is bounded and no natural two-event question is this long.
MAX_SPLIT_CHARS = 1000
_CLAUSE = re.compile(r"\b(before|after|when)\b", re.IGNORECASE)


def _clean(s: str) -> str:
    return s.strip(" ,.?!")


def _clause(q: str) -> tuple[str, str, str] | None:
    """The first `<left>[,] before|after|when <right>`: left has a word character and ends in whitespace
    (after an optional comma), right starts with whitespace. One scan over the keywords, no backtracking
    (the old single regex was cubic in runs of spaces, MEM2-222 F6)."""
    for m in _CLAUSE.finditer(q):
        left, right = q[: m.start()], q[m.end() :]
        if not left[-1:].isspace() or not right[:1].isspace():
            continue
        head = left.rstrip().removesuffix(",").rstrip()
        if "\n" in head or "\n" in right.strip():  # the old pattern's `.` did not cross a line
            continue
        if re.search(r"\w", head):
            return head, m[1], right.lstrip()
    return None


def _split(q: str) -> Split | None:
    if m := re.search(r"\bbetween (.+?) and (.+)$", q, re.IGNORECASE):
        return Split("between_and", [_clean(m[1]), _clean(m[2])])
    if m := re.search(r"\bhow (?:long|many \w+) (?:had|have|has|did) (?:I|we) been (.+?) (?:when|before|until|after) (.+)$", q, re.IGNORECASE):  # fmt: skip
        return Split("how_long_when", [_clean(m[1]), _clean(m[2])])
    if m := re.search(r"\bsince (.+?) (?:when|before|until) (.+)$", q, re.IGNORECASE):
        return Split("since_when", [_clean(m[1]), _clean(m[2])])
    if c := _clause(q):
        left, right = _clean(c[0]), _clean(c[2])
        if len(left.split()) >= 4 and len(right.split()) >= 3:
            return Split(f"{c[1].lower()}_clause", [left, right])
    if " or " in q:
        left, right = q.split(" or ", 1)
        left = left.rstrip(" ,")  # "…first, or B": the comma ends A, it does not start it (F2)
        a = left.rsplit(",", 1)[-1] if "," in left else " ".join(left.split()[-3:])
        return Split("a_or_b", [_clean(a), _clean(right)])
    return None


def split_endpoints(question: str) -> Split | None:
    """Two event sub-queries, or None. Patterns only: no type labels, no cue lists of dataset phrases.
    Never an empty sub-query: a split with an empty part is no split (MEM2-222 F2)."""
    if len(question) > MAX_SPLIT_CHARS:
        return None
    s = _split(_clean(question))
    return s if s is not None and all(p.strip() for p in s.parts) else None


def merge_endpoints(
    groups: Sequence[Sequence[Kept]], *, per_node_cap: int, limit: int
) -> list[Kept]:
    """What the sub-queries kept, merged: a chunk both kept counts once (the higher p), at most
    `per_node_cap` chunks per node, taken round-robin from each sub-query's list (p descending) up to
    `limit` so that each endpoint's best rows survive the cut, then ordered by date (undated last,
    ties by p descending, then chunk id)."""
    best: dict[int, Kept] = {}
    for g in groups:
        for k in g:
            if k.chunk_id not in best or k.p > best[k.chunk_id].p:
                best[k.chunk_id] = k
    taken: list[Kept] = []
    seen: set[int] = set()
    per_node: dict[int, int] = {}
    for rank in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if rank >= len(g) or len(taken) >= limit:
                continue
            k = best[g[rank].chunk_id]
            if k.chunk_id in seen or per_node.get(k.node_id, 0) >= per_node_cap:
                continue
            seen.add(k.chunk_id)
            per_node[k.node_id] = per_node.get(k.node_id, 0) + 1
            taken.append(k)
    return sorted(taken, key=lambda k: (k.date is None, k.date or "", -k.p, k.chunk_id))
