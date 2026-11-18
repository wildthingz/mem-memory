"""Metrics of the P3 write benchmarks (benchmarks-v2 §2.0 definitions), from a replay log, the
replayed store and the SYN gold. Every function states its numerator and denominator.

Server "current" = `status = 'active' AND valid_to IS NULL` (chain-derived). A slot is a
(subject, relation) pair; its gold members and current set come from `expected_state.json`.
Ops that the server never saved (an error, a mutant) are never silently dropped: they stay in the
denominators and count against the metric that needed them.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bench.synth import data as D
from bench.synth.generator import Synth
from bench.synth.replay import OpLog

SCORED_STYLES = ("canonical", "alias", "typo", "ambiguous")


def rate(num: float, den: float) -> float:
    return num / den if den else math.nan


# ---- store reading ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MemRow:
    id: int
    status: str
    valid_from: str
    valid_to: str | None
    recorded_at: str
    body: str
    content_hash: str


def read_memories(db: Path) -> dict[int, MemRow]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {r[0]: MemRow(*r) for r in conn.execute(
            "SELECT id, status, valid_from, valid_to, recorded_at, body, content_hash "
            "FROM nodes WHERE kind = 'memory'")}  # fmt: skip
    finally:
        conn.close()


def read_components(db: Path) -> list[list[int]]:
    """Connected components of live `supersedes` edges (memory ids, sorted)."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        edges = conn.execute("SELECT src, dst FROM edges WHERE rel = 'supersedes' "
                             "AND removed_at IS NULL").fetchall()  # fmt: skip
    finally:
        conn.close()
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[int] = set()
    out = []
    for n in sorted(adj):
        if n in seen:
            continue
        comp, todo = {n}, [n]
        while todo:
            x = todo.pop()
            for y in adj[x] - comp:
                comp.add(y)
                todo.append(y)
        seen |= comp
        out.append(sorted(comp))
    return out


def is_current(m: MemRow | None) -> bool:
    return m is not None and m.status == "active" and m.valid_to is None


def saved_map(logs: list[OpLog]) -> dict[int, int]:
    """op index -> memory number, for every op that created a memory."""
    return {lg.op["i"]: int(lg.mid[1:]) for lg in logs
            if lg.status == "saved" and lg.mid and lg.op["tool"] == "remember"}  # fmt: skip


# ---- B-W1 ---------------------------------------------------------------------------------------


def w1_metrics(logs: list[OpLog], eid: dict[str, int]) -> dict[str, float]:
    """Scored mentions = resolver decisions (styles canonical, alias, typo, ambiguous); explicit
    `new:`, `#id` and introduction (`alias → #id`) entries are not decisions and not scored.

    wrong_merge_rate = first decision linked an existing entity other than gold / scored mentions
    split_rate       = first decision created a new entity / scored mentions (gold always existed)
    ask_back_recall  = asked on `ambiguous` mentions / ambiguous mentions
    ask_rate         = asked / scored mentions (introductions excluded)
    final_link_acc   = entity after the oracle's answer is gold / scored mentions
    """
    scored = [(lg, d) for lg in logs for d in lg.decisions if d.style in SCORED_STYLES]
    amb = [d for _, d in scored if d.style == "ambiguous"]
    n = len(scored)
    wrong = sum(d.first == "link" and d.first_id != eid.get(d.entity) for _, d in scored)
    asks = sum(d.first == "ask" for _, d in scored)
    return {
        "mentions_scored": float(n), "mentions_ambiguous": float(len(amb)),
        "wrong_merge_rate": rate(wrong, n),
        "ask_back_recall": rate(sum(d.first == "ask" for d in amb), len(amb)),
        "ask_rate": rate(asks, n),
        "split_rate": rate(sum(d.first == "new" for _, d in scored), n),
        "final_link_accuracy": rate(sum(d.final_id == eid.get(d.entity) for _, d in scored), n),
        "introductions": float(sum(d.style == "alias_intro" for lg in logs for d in lg.decisions)),
        "ask_offered_gold": rate(sum(bool(d.offered_gold) for d in amb if d.first == "ask"),
                                 sum(d.first == "ask" for d in amb)),
        "op_errors": float(sum(lg.error is not None for lg in logs)),
    }  # fmt: skip


# ---- shared gold views --------------------------------------------------------------------------


def gold_current_ops(syn: Synth, *, multi: bool | None = None) -> set[int]:
    out: set[int] = set()
    for s in syn.expected["slots"].values():
        if multi is None or s["multi"] == multi:
            out |= set(s["current"])
    return out


def wrong_invalidation(
    syn: Synth, mem: dict[int, MemRow], mids: dict[int, int], *, multi: bool | None = None
) -> tuple[float, list[int]]:  # fmt: skip
    """#memories that are not current at the end but gold says should be current / #memories gold
    says are current (A17: `duplicate` on a true duplicate is correct and is not in the gold set)."""
    gold = sorted(gold_current_ops(syn, multi=multi))
    bad = [op for op in gold if op in mids and not is_current(mem.get(mids[op]))]
    return rate(len(bad), len(gold)), bad


def slot_accuracy(syn: Synth, mem: dict[int, MemRow], mids: dict[int, int]) -> float:
    """#slots whose current value set at the end equals gold / #slots (every slot with any op)."""
    ok = 0
    for s in syn.expected["slots"].values():
        got = {m for op in s["members"] if (m := mids.get(op)) is not None
               and is_current(mem.get(m))}  # fmt: skip
        ok += got == {mids[op] for op in s["current"] if op in mids} and all(
            op in mids for op in s["current"])  # fmt: skip
    return rate(ok, len(syn.expected["slots"]))


def chain_correctness(syn: Synth, mem: dict[int, MemRow], mids: dict[int, int]) -> float:
    """#slots with >= 1 update whose final chain (members, order, valid_from/valid_to) equals gold
    / #slots that had >= 1 update. Single-valued slots; retracted and duplicate memories are not
    members."""
    total = ok = 0
    for s in syn.expected["slots"].values():
        if s["multi"] or not s["had_update"]:
            continue
        total += 1
        want = [(mids.get(op), syn.expected["memories"][str(op)]["valid_from"],
                 syn.expected["memories"][str(op)]["valid_to"]) for op in s["chain"]]  # fmt: skip
        ids = {mids[op] for op in s["members"] if op in mids}
        act = sorted((mem[i] for i in ids if i in mem and mem[i].status == "active"),
                     key=lambda m: (m.valid_from, m.recorded_at, m.id))  # fmt: skip
        got = [(m.id, m.valid_from, m.valid_to) for m in act]
        ok += got == want
    return rate(ok, total)


def graph_diffs(syn: Synth, mem: dict[int, MemRow], mids: dict[int, int]) -> list[str]:
    """Memory-by-memory difference between the replayed store and `expected_state.json`."""
    out = []
    for key, g in syn.expected["memories"].items():
        op = int(key)
        row = mem.get(mids[op]) if op in mids else None
        if row is None:
            out.append(f"op {op}: not saved")
            continue
        if row.status != g["status"]:
            out.append(f"op {op} m{row.id}: status {row.status!r}, want {g['status']!r}")
            continue
        if g["status"] != "active":
            continue
        if row.valid_from != g["valid_from"] or row.valid_to != g["valid_to"]:
            out.append(f"op {op} m{row.id}: [{row.valid_from}, {row.valid_to}), "
                       f"want [{g['valid_from']}, {g['valid_to']})")  # fmt: skip
    return out


def final_duplicate_rate(syn: Synth, mem: dict[int, MemRow]) -> float:
    """(#current memories - #distinct current gold facts) / #distinct current gold facts. Signed:
    a negative value means memories vanished (wrong invalidation reports that separately)."""
    gold = len(gold_current_ops(syn))
    current = sum(is_current(m) for m in mem.values())
    return rate(current - gold, gold)


# ---- B-W2 / B-W3 --------------------------------------------------------------------------------


def w2_metrics(syn: Synth, logs: list[OpLog], mem: dict[int, MemRow], mids: dict[int, int],
               ) -> dict[str, float]:  # fmt: skip
    ex = [lg for lg in logs if lg.op["kind"] == "dup_exact"]
    first_mid = {lg.op["i"]: lg.mid for lg in logs}
    exists_ok = sum(lg.status == "exists" and lg.mid == first_mid.get(lg.op["gold"]["target"])
                    for lg in ex)  # fmt: skip
    para = [lg for lg in logs if lg.op["kind"] == "dup_paraphrase"]
    wi, _ = wrong_invalidation(syn, mem, mids, multi=True)
    return {
        "exact_n": float(len(ex)), "exists_rate": rate(exists_ok, len(ex)),
        "paraphrase_n": float(len(para)),
        "paraphrase_surfacing": rate(sum(bool(lg.surfaced) for lg in para), len(para)),
        "final_duplicate_rate": final_duplicate_rate(syn, mem),
        "wrong_invalidation_compat": wi,
    }  # fmt: skip


def conflict_precision(syn: Synth, logs: list[OpLog], mids: dict[int, int],
                       kinds: tuple[str, ...]) -> tuple[float, int]:  # fmt: skip
    """#returned candidates sharing the gold subject AND relation / #returned candidates, over the
    saved ops of `kinds`. Returns (precision, #candidates)."""
    inv = {f"m{m}": op for op, m in mids.items()}
    hit = tot = 0
    for lg in logs:
        if lg.op["kind"] not in kinds or lg.status != "saved":
            continue
        slot = lg.op["gold"]["slot"]
        for c in lg.conflicts:
            tot += 1
            op = inv.get(c)
            gold = syn.expected["memories"].get(str(op))  # a stray memory is no match
            hit += gold is not None and gold["slot"] == slot
    return rate(hit, tot), tot


def w3_metrics(syn: Synth, logs: list[OpLog], mem: dict[int, MemRow], mids: dict[int, int],
               ) -> dict[str, float]:  # fmt: skip
    upd = [lg for lg in logs if lg.op["kind"] == "update"]
    prec, n_cand = conflict_precision(syn, logs, mids, ("update", "dup_paraphrase"))
    prec_all, _ = conflict_precision(syn, logs, mids, tuple(D.SHARES))
    wi, _ = wrong_invalidation(syn, mem, mids)
    by_sub: dict[str, list[bool]] = defaultdict(list)
    for lg in upd:
        by_sub[lg.op["gold"]["subtype"]].append(bool(lg.surfaced))
    out = {
        "updates_n": float(len(upd)),
        "surfacing_recall": rate(sum(bool(lg.surfaced) for lg in upd), len(upd)),
        "list_precision": prec,
        "list_candidates": float(n_cand),
        "list_precision_all_ops": prec_all,
        "chain_correctness": chain_correctness(syn, mem, mids),
        "wrong_invalidation": wi,
        "slot_accuracy": slot_accuracy(syn, mem, mids),
        "max_list_len": float(max((len(lg.conflicts) for lg in logs), default=0)),
        "link_errors": float(
            sum(lg.acted in ("supersedes", "duplicate_of") and lg.action_ok is False for lg in logs)
        ),
    }
    for sub, hits in sorted(by_sub.items()):
        out[f"surfacing_{sub}"] = rate(sum(hits), len(hits))
    return out
