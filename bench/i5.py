"""B-I5 edge precision: the sample, the labelling sitting and the score (benchmarks-v2 B-I).

100 live import edges sampled from the store, stratified by rel, seed 7. The person labels each
against the note (≈ 30 min): does the edge mean what its rel claims? Gate: overall precision ≥ 0.98
and no rel below 0.95.

Allocation (decision record 0010): proportional to the rels' edge counts with a floor of
min(5, n_rel) per rel, the remainder by largest remainder, total exactly 100 (or every edge when
the store has fewer). With n = 5 in a small rel a single wrong edge is 0.80 < 0.95: the spec's
per-rel bar is that strict, the sheet says how many edges each rel got.
"""

from __future__ import annotations

import csv
import random
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

N_EDGES = 100
SEED = 7
FLOOR = 5
OVERALL_MIN, REL_MIN = 0.98, 0.95
COLS = ["edge_id", "rel", "src", "dst", "context", "label"]
CONTEXT = 160


def allocate(counts: dict[str, int], total: int = N_EDGES, floor: int = FLOOR) -> dict[str, int]:
    if sum(counts.values()) <= total:
        return dict(counts)
    alloc = {r: min(n, floor) for r, n in counts.items()}
    left = total - sum(alloc.values())
    room = {r: counts[r] - alloc[r] for r in counts}
    weight = sum(room.values())
    quota = {r: left * room[r] / weight for r in counts}
    for r in counts:
        alloc[r] += int(quota[r])
    rest = total - sum(alloc.values())
    for r in sorted(counts, key=lambda r: (-(quota[r] - int(quota[r])), r))[:rest]:
        alloc[r] += 1
    return alloc


def _context(conn: sqlite3.Connection, edge_id: int, rel: str, src_name: str, dst: str) -> str:
    occ = conn.execute(
        "SELECT o.chunk_id, o.field, o.pos, c.text FROM link_occurrences o"
        " LEFT JOIN chunks c ON c.id = o.chunk_id WHERE o.edge_id = ? ORDER BY o.id LIMIT 1",
        (edge_id,)).fetchone()  # fmt: skip
    if occ is None:
        return f"{rel}: {src_name} -> {dst}" if rel == "tagged" else f"{src_name} -> {dst}"
    _, field, pos, text = occ
    if field:
        return f"frontmatter `{field}` of {src_name}: {dst}"
    lo = max(pos - CONTEXT // 2, 0)
    return " ".join(text[lo : pos + CONTEXT].split())


def sample(conns: dict[str, sqlite3.Connection], seed: int = SEED) -> list[dict[str, str]]:
    """The seed-7 sample over every store's live import edges; edge ids are per store, so the
    sheet keys an edge as `<store>:<id>`."""
    by_rel: dict[str, list[tuple]] = defaultdict(list)
    for store, conn in sorted(conns.items()):
        for r in conn.execute(
                "SELECT e.id, e.rel, a.source_ref, a.name, b.source_ref, b.name, b.kind FROM edges e"
                " JOIN nodes a ON a.id = e.src JOIN nodes b ON b.id = e.dst"
                " WHERE e.origin = 'import' AND e.removed_at IS NULL ORDER BY e.id"):  # fmt: skip
            by_rel[r[1]].append((store, conn, *r))
    rng = random.Random(seed)
    alloc = allocate({rel: len(v) for rel, v in by_rel.items()})
    out = []
    for rel in sorted(by_rel):
        for store, conn, eid, _, src, src_name, dst_ref, dst_name, dst_kind in rng.sample(
                by_rel[rel], alloc[rel]):  # fmt: skip
            dst = dst_ref or (f"#{dst_name}" if dst_kind == "tag" else f"[stub] {dst_name}")
            out.append({"edge_id": f"{store}:{eid}", "rel": rel, "src": src or src_name, "dst": dst,
                        "context": _context(conn, eid, rel, src_name, dst), "label": ""})  # fmt: skip
    return out


def write_sheet(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


@dataclass
class I5Score:
    n: int
    precision: float
    by_rel: dict[str, tuple[int, int]]  # rel -> (correct, n)
    unlabelled: int
    wrong: list[str]

    @property
    def min_rel_precision(self) -> float:
        return min((c / n for c, n in self.by_rel.values() if n), default=float("nan"))


def score(rows: list[dict[str, str]]) -> I5Score:
    lab = [r for r in rows if r["label"].strip().lower() in ("y", "n")]
    by_rel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in lab:
        by_rel[r["rel"]][1] += 1
        by_rel[r["rel"]][0] += r["label"].strip().lower() == "y"
    correct = sum(c for c, _ in by_rel.values())
    return I5Score(len(lab), correct / len(lab) if lab else float("nan"),
                   {k: (v[0], v[1]) for k, v in sorted(by_rel.items())}, len(rows) - len(lab),
                   [f"{r['edge_id']} {r['rel']} {r['src']} -> {r['dst']}" for r in lab
                    if r["label"].strip().lower() == "n"])  # fmt: skip


def label_sitting(
    path: Path, *, minutes: float | None, ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic,
) -> tuple[int, int]:  # fmt: skip
    """Label the unlabelled rows of the sheet y / n (does the edge mean what its rel claims?),
    s skip, q quit; the sheet is rewritten after every answer. Returns (labelled now, left)."""
    rows = read_sheet(path)
    t0, done = clock(), 0
    for r in rows:
        if r["label"].strip():
            continue
        if minutes is not None and clock() - t0 >= minutes * 60:
            break
        out(f"\n[{r['edge_id']}] {r['rel']}: {r['src']}  ->  {r['dst']}\n  {r['context']}")
        k = ask("does this edge mean what its rel says?  [y]es  [n]o  [s]kip  [q]uit > ")
        k = k.strip().lower()[:1]
        if k == "q":
            break
        if k in ("y", "n"):
            r["label"] = k
            write_sheet(rows, path)
            done += 1
    return done, sum(not r["label"].strip() for r in rows)
