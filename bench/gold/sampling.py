"""Note sampling and the drafting slot plan (benchmarks-v2 §3.1 "Note sampling", "Authoring").

40 notes (seed 20261022) plus 10 linked note pairs, from `personal/` and `projects/` only and only
notes routed to the personal store (spec change 2026-10-24: PG-v1 holds no work-store content). The
40 notes are proportional to the number of eligible notes per folder (largest remainder). Each of
the 50 units gets `slots_per_unit` draft slots (4 by default: ≈ 200 drafts), dealt so every
category has about 5/3 of its quota in drafts (a third are rejected at ~25 s each) and the
pair-only categories (multi-note, relational) are drafted only from pairs. If 4 per unit cannot
fill a quota, raise `slots_per_unit`; a quota is never lowered.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from bench.gold import pg
from bench.gold.chunks import ChunkIndex

MIN_WORDS = 40  # a note needs a section of at least this many words to be worth a question
SLOTS_PER_UNIT = 4


@dataclass(frozen=True)
class Unit:
    """One drafting unit: a note or a linked pair of notes."""

    uid: str
    paths: tuple[str, ...]
    slots: tuple[str, ...] = ()  # categories to draft, one draft each
    store: str = "personal"  # the store of its notes (a pair is always inside one store)

    @property
    def is_pair(self) -> bool:
        return len(self.paths) == 2


def folder_of(path: str) -> str:
    return path.split("/", 1)[0].lower() if "/" in path else ""


def eligible_notes(index: ChunkIndex) -> dict[str, list[str]]:
    """folder -> sorted paths with at least one section of MIN_WORDS words, for the PG folders.
    Only notes held by the personal store: a folder that config.yaml routes to work (an override
    such as projects/client-x) is in work.db and is never eligible."""
    out: dict[str, list[str]] = {f: [] for f in pg.PG_FOLDERS}
    for path in index.paths():
        f = folder_of(path)
        if (f in out and index.store_of_path[path] == "personal"
                and any(len(c.text.split()) >= MIN_WORDS for c in index.of_path(path))):  # fmt: skip
            out[f].append(path)
    return out


def linked_pairs(conns: dict[str, sqlite3.Connection], notes: set[str]) -> list[tuple[str, str]]:
    """Distinct unordered pairs of notes joined by a live `links_to` edge, both in `notes`. Edges
    live inside a store, so a pair is never split across personal.db and work.db (one `recall` asks
    one store)."""
    pairs: set[tuple[str, str]] = set()
    for conn in conns.values():
        pairs |= _pairs_of(conn, notes)
    return sorted(pairs)


def _pairs_of(conn: sqlite3.Connection, notes: set[str]) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT a.source_ref, b.source_ref FROM edges e JOIN nodes a ON a.id = e.src"
        " JOIN nodes b ON b.id = e.dst WHERE e.rel = 'links_to' AND e.removed_at IS NULL"
        " AND a.origin = 'import' AND b.origin = 'import' AND a.id <> b.id"
    )
    return {tuple(sorted((x, y))) for x, y in rows if x in notes and y in notes}  # type: ignore[misc]


def sample_units(
    index: ChunkIndex,
    conns: dict[str, sqlite3.Connection],
    seed: int = pg.NOTE_SEED,
    slots_per_unit: int = SLOTS_PER_UNIT,
) -> list[Unit]:
    rng = random.Random(seed)
    pool = eligible_notes(index)
    singles: list[str] = []
    quotas = pg.folder_quotas({f: len(ps) for f, ps in pool.items()})
    for folder, quota in quotas.items():
        singles += rng.sample(pool[folder], quota)
    all_notes = {p for ps in pool.values() for p in ps}
    pairs = linked_pairs(conns, all_notes)
    fresh = [p for p in pairs if not set(p) & set(singles)]
    chosen = rng.sample(fresh, pg.N_PAIRS) if len(fresh) >= pg.N_PAIRS else None
    if chosen is None:  # not enough disjoint pairs: allow overlap with the single notes
        if len(pairs) < pg.N_PAIRS:
            raise ValueError(f"only {len(pairs)} linked note pairs, need {pg.N_PAIRS}")
        chosen = rng.sample(pairs, pg.N_PAIRS)
    st = index.store_of_path
    units = [Unit(f"n{i:02d}", (p,), store=st[p]) for i, p in enumerate(sorted(singles))]
    units += [Unit(f"p{i:02d}", p, store=st[p[0]]) for i, p in enumerate(sorted(chosen))]
    return plan_slots(units, slots_per_unit=slots_per_unit)


def slot_targets(total: int = (pg.N_NOTES + pg.N_PAIRS) * SLOTS_PER_UNIT) -> dict[str, int]:
    """Drafts wanted per category: the quota total in proportion, 5/3 of it at 200 drafts in all
    (largest remainder, so the targets sum to `total`)."""
    n = pg.N_QUESTIONS
    exact = {c: t * total / n for c, t in pg.TOTAL.items()}
    out = {c: int(x) for c, x in exact.items()}
    for c in sorted(exact, key=lambda c: (-(exact[c] - out[c]), c))[: total - sum(out.values())]:
        out[c] += 1
    return out


def plan_slots(units: list[Unit], shift: int = 0, slots_per_unit: int = SLOTS_PER_UNIT
               ) -> list[Unit]:  # fmt: skip
    """`shift` > 0 deals the same slots to other units (`bench gold draft --more`); the unit ids
    get a `.r<shift>` suffix so drafts never collide."""
    singles = [u for u in units if not u.is_pair]
    pairs = [u for u in units if u.is_pair]
    want = slot_targets(len(units) * slots_per_unit)
    pair_slots = [c for c in pg.PAIR_ONLY for _ in range(want[c])]
    spare = len(pairs) * slots_per_unit - len(pair_slots)
    if spare < 0:
        raise ValueError("not enough pairs for the pair-only categories")
    pair_slots += ["knowledge-update"] * spare
    single_left = dict(want)
    for c in pg.PAIR_ONLY:
        single_left[c] = 0
    single_left["knowledge-update"] -= spare
    single_slots = [c for c in pg.QUOTAS for _ in range(single_left[c])]

    def deal(us: list[Unit], slots: list[str]) -> list[Unit]:
        buckets: list[list[str]] = [[] for _ in us]
        for j, c in enumerate(slots):
            buckets[(j + shift) % len(us)].append(c)
        suffix = f".r{shift}" if shift else ""
        return [
            Unit(u.uid + suffix, u.paths, tuple(b), u.store)
            for u, b in zip(us, buckets, strict=True)
        ]

    return deal(singles, single_slots) + deal(pairs, pair_slots)
