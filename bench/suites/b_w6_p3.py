"""B-W6, all items (benchmarks-v2 B-W6): property tests on the fixture op streams and SYN-S seeds 1-3.

Items 1, 2, 3, 5, 6 are the P1 checks (`bench.suites.b_w6`) on the fixture streams, repeated here on
the SYN-S seeds; items 4 and 7-11 need `link`, `forget`, the resolver and two writers:
 4 `supersedes` components are chains: valid_to = the next member's valid_from (members ordered by
   valid_from, recorded_at, id), the last open
 7 `needs_choice` writes nothing, including other new subjects (canonical dump unchanged)
 8 repeated `link` is a no-op (canonical dump unchanged)
 9 `same_as` with a shared neighbour and a shared alias succeeds, leaving one live edge
10 `link` / `forget` cannot invalidate an imported node
11 two concurrent writers: no SQLITE_BUSY reaches the client
Gate: 0 violations.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from bench.dump import canonical_dump
from bench.suites import b_w6
from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import sql, sql_write
from bench.synth import data as D
from bench.synth import score
from bench.synth.generator import generate
from bench.synth.replay import Replay
from bench.synth.run import require_p3

Q = "2026-10-22T09:00:00Z"
SEEDS = (1, 2, 3)
ROWS = ("nodes", "edges", "chunks", "aliases")


def _snap(db: Path) -> dict[str, dict[Any, tuple]]:
    out = {}
    for t in ROWS:
        out[t] = {r[0]: r[1:] for r in sql(db, f"SELECT * FROM {t} ORDER BY 1")}
    return out


def _replay(ctx: Context, seed: int, name: str, hook=None):
    syn = generate(seed, "S")
    db = ctx.store(f"{name}.db")
    client = ctx.factory(db, scope="personal", tz=D.TZ, now=syn.ops[0]["now"])
    rp = Replay(client, syn, on_op=hook)
    try:
        rp.run()
    finally:
        client.close()
    return syn, rp, db


def syn_items_1235(ctx: Context, res: SuiteResult) -> None:
    for seed in SEEDS:
        tag = f"s{seed}"
        # 1: same stream twice -> byte-identical dump; recorded_at is the op date, as a column
        syn, rp, db_a = _replay(ctx, seed, f"w6_1a_{tag}")
        _, _, db_b = _replay(ctx, seed, f"w6_1b_{tag}")
        res.check(f"1_{tag}_dumps_byte_identical", canonical_dump(db_a) == canonical_dump(db_b))
        mem = score.read_memories(db_a)
        mids = score.saved_map(rp.logs)
        off = [(op, m) for op, m in mids.items() if mem[m].recorded_at != syn.ops[op]["now"]]
        res.check(f"1_{tag}_recorded_at_is_op_date", not off, str(off[:3]))
        # 2: nothing deleted or rewritten: ids stay, memory text/provenance never change
        prev: dict[str, dict] = {}
        problems: list[str] = []

        def hook(log, db=None, prev=prev, problems=problems):
            cur = _snap(db)
            for t in ROWS:
                lost = set(prev.get(t, {})) - set(cur[t])
                if lost and t != "aliases":
                    problems.append(f"op {log.op['i']}: {t} ids vanished {sorted(lost)[:3]}")
            for i, row in prev.get("nodes", {}).items():
                new = cur["nodes"].get(i)
                if new and (row[2], row[3], row[5], row[6], row[8]) != (
                    new[2],
                    new[3],
                    new[5],
                    new[6],
                    new[8],
                ):
                    problems.append(f"op {log.op['i']}: node {i} text or valid_from changed")
            prev.clear()
            prev.update(cur)

        db_c = ctx.store(f"w6_2_{tag}.db")
        client = ctx.factory(db_c, scope="personal", tz=D.TZ, now=syn.ops[0]["now"])
        try:
            Replay(client, syn, on_op=lambda lg, db=db_c, h=hook: h(lg, db)).run()
        finally:
            client.close()
        res.check(f"2_{tag}_nothing_deleted_or_rewritten", not problems, "; ".join(problems[:3]))
        # 3: every memory has a live about edge, session and host
        orphans = sql(db_a, "SELECT n.id FROM nodes n WHERE n.kind='memory' AND NOT EXISTS ("
                      "SELECT 1 FROM edges e WHERE e.src=n.id AND e.rel='about' "
                      "AND e.removed_at IS NULL)")  # fmt: skip
        res.check(f"3_{tag}_every_memory_has_live_about_edge", not orphans, str(orphans[:5]))
        miss = sql(db_a, "SELECT id FROM nodes WHERE kind='memory' AND (session IS NULL OR "
                   "session='' OR host IS NULL OR host='')")  # fmt: skip
        res.check(f"3_{tag}_session_and_host_set", not miss, str(miss[:5]))
        # 4: components are chains in valid time
        bad = []
        for comp in score.read_components(db_a):
            act = sorted((mem[i] for i in comp if i in mem and mem[i].status == "active"),
                         key=lambda m: (m.valid_from, m.recorded_at, m.id))  # fmt: skip
            for k, m in enumerate(act):
                want = act[k + 1].valid_from if k + 1 < len(act) else None
                if m.valid_to != want:
                    bad.append(f"m{m.id}: valid_to {m.valid_to}, want {want}")
        res.check(f"4_{tag}_supersedes_components_are_chains", not bad, "; ".join(bad[:3]))
        # 5: valid_to >= valid_from; equality only for a same-instant predecessor of a chain
        rev = sql(db_a, "SELECT id FROM nodes WHERE valid_to IS NOT NULL AND valid_to < valid_from")
        res.check(f"5_{tag}_valid_to_ge_valid_from", not rev, str(rev[:3]))
        comps = {m: c for c in score.read_components(db_a) for m in c}
        eq = [m.id for m in mem.values() if m.valid_to is not None and m.valid_to == m.valid_from
              and not any(mem[o].valid_from == m.valid_from and o != m.id
                          for o in comps.get(m.id, []))]  # fmt: skip
        res.check(f"5_{tag}_equality_only_same_instant_in_chain", not eq, str(eq[:3]))


def item7_needs_choice(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("w6_7.db"), scope="personal", tz=D.TZ)
    try:
        for n in ("Ana Silva", "Ana Lopez"):
            c.call("remember", {"text": f"{n} speaks Dutch.", "type": "fact",
                                "about": [f"new:person:{n}"]}, now=Q)  # fmt: skip
        before = canonical_dump(c.db_path)
        r = c.call("remember", {"text": "Ana likes tea.", "type": "preference",
                                "about": ["Ana", "new:topic:Fresh Subject"]}, now=Q)  # fmt: skip
        res.check("7_reply_is_needs_choice", r.structured["status"] == "needs_choice",
                  str(r.structured["status"]))  # fmt: skip
        res.check("7_needs_choice_writes_nothing_incl_new_subjects",
                  canonical_dump(c.db_path) == before)  # fmt: skip
    finally:
        c.close()


def item8_link_idempotent(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("w6_8.db"), scope="personal", tz=D.TZ)
    try:
        a = c.call("remember", {"text": "Ana Silva lives in Lisbon.", "type": "fact",
                                "about": ["new:person:Ana Silva"], "as_of": "2026-01-01"},
                   now=Q).structured["id"]  # fmt: skip
        b = c.call("remember", {"text": "Ana Silva lives in Porto.", "type": "fact",
                                "about": ["Ana Silva"], "as_of": "2026-06-01"},
                   now="2026-10-22T09:01:00Z").structured["id"]  # fmt: skip
        c.call("link", {"src": b, "rel": "supersedes", "dst": a}, now="2026-10-22T09:02:00Z")
        first = canonical_dump(c.db_path)
        try:
            c.call("link", {"src": b, "rel": "supersedes", "dst": a}, now="2026-10-22T09:03:00Z")
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = False
            detail = str(e)
        res.check("8_repeated_link_is_ok", ok, "" if ok else detail)
        res.check("8_repeated_link_changes_nothing", canonical_dump(c.db_path) == first)
    finally:
        c.close()


def item9_same_as(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("w6_9.db"), scope="personal", tz=D.TZ)
    try:
        m = c.call(
            "remember",
            {
                "text": "Ana Silva and A. Silva share a desk.",
                "type": "fact",
                "about": ["new:person:Ana Silva", "new:person:A. Silva"],
            },
            now=Q,
        )
        e = {r[0]: r[1] for r in sql(c.db_path, "SELECT name, id FROM nodes WHERE kind='entity'")}
        e1, e2 = e["Ana Silva"], e["A. Silva"]
        for ent in (e1, e2):  # one alias on both entities
            c.call("remember", {"text": f"Entity {ent} answers to Ana at the office.",
                                "type": "fact", "about": [f"Ana → #e{ent}"]}, now=Q)  # fmt: skip
        ok, detail = True, ""
        try:
            c.call("link", {"src": f"e{e2}", "rel": "same_as", "dst": f"e{e1}"}, now=Q)
        except Exception as err:  # noqa: BLE001
            ok, detail = False, f"{type(err).__name__}: {err}"
        res.check("9_same_as_shared_neighbour_and_alias_succeeds", ok, detail)
        if ok:
            mid = int(m.structured["id"][1:])
            live = sql(
                c.db_path,
                "SELECT count(*) FROM edges WHERE src=? AND dst=? AND "
                "rel='about' AND removed_at IS NULL",
                (mid, e1),
            )[0][0]
            res.check("9_one_live_edge_to_the_target", live == 1, f"{live} live edges")
            gone = sql(c.db_path, "SELECT status FROM nodes WHERE id=?", (e2,))[0][0]
            res.check("9_source_entity_is_duplicate", gone == "duplicate", gone)
    finally:
        c.close()


def item10_imported(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("w6_10.db"), scope="personal", tz=D.TZ)
    try:
        c.call("remember", {"text": "Ana Silva lives in Lisbon.", "type": "fact",
                            "about": ["new:person:Ana Silva"]}, now=Q)  # fmt: skip
        sql_write(c.db_path, "INSERT INTO nodes(kind,type,name,body,date,valid_from,recorded_at,"
                  "origin,source_ref,content_hash) VALUES ('memory','fact','Imported fact',"
                  "'Imported statement from the vault','2026-01-01','2026-01-01T00:00:00Z',"
                  "'2026-01-01T00:00:00Z','import','vault/people/ana.md','imported-hash')")  # fmt: skip
        imp = sql(c.db_path, "SELECT id FROM nodes WHERE origin='import'")[0][0]
        new = c.call("remember", {"text": "Ana Silva lives in Porto.", "type": "fact",
                                  "about": ["Ana Silva"], "as_of": "2026-06-01"},
                     now="2026-10-22T09:01:00Z").structured["id"]  # fmt: skip

        def refused(tool: str, args: dict[str, Any]) -> bool:
            try:
                c.call(tool, args, now="2026-10-22T09:02:00Z")
            except Exception:  # noqa: BLE001
                return True
            return False

        res.check(
            "10_supersedes_imported_is_refused",
            refused("link", {"src": new, "rel": "supersedes", "dst": f"m{imp}"}),
        )
        res.check("10_forget_imported_is_refused",
                  refused("forget", {"id": f"m{imp}", "reason": "test"}))  # fmt: skip
        row = sql(c.db_path, "SELECT status, valid_to FROM nodes WHERE id=?", (imp,))[0]
        res.check("10_imported_node_untouched", row == ("active", None), str(row))
    finally:
        c.close()


def item11_two_writers(ctx: Context, res: SuiteResult, n: int = 25) -> None:
    db = ctx.store("w6_11.db")
    c0 = ctx.factory(db, scope="personal", tz=D.TZ)
    c0.call("remember", {"text": "Seed memory for two writers.", "type": "fact",
                         "about": ["new:topic:Writers"]}, now=Q)  # fmt: skip
    c0.close()
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def writer(k: int) -> None:
        c = ctx.factory(db, scope="personal", tz=D.TZ)
        try:
            barrier.wait(5)
            for i in range(n):
                try:
                    c.call("remember", {"text": f"Writer {k} note number {i} about writers.",
                                        "type": "fact", "about": ["Writers"]},
                           now=f"2026-10-22T10:{k:02d}:{i:02d}Z")  # fmt: skip
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {e}")
        finally:
            c.close()

    ths = [threading.Thread(target=writer, args=(k,)) for k in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    busy = [e for e in errors if "lock" in e.lower() or "busy" in e.lower()]
    res.check("11_no_sqlite_busy_reaches_the_client", not busy, "; ".join(busy[:3]))
    res.check("11_no_write_errors", not errors, "; ".join(errors[:3]))
    rows = sql(db, "SELECT count(*) FROM nodes WHERE kind='memory'")[0][0]
    res.check("11_every_write_landed", rows == 1 + 2 * n - len(errors), f"{rows} memories")


def run(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-W6", mode="fixture")
    require_p3(ctx)
    for item in (
        b_w6.item1_determinism,
        b_w6.item2_nothing_deleted,
        b_w6.item3_provenance,
        b_w6.item5_intervals,
        b_w6.item6_two_stores,
    ):
        item(ctx, res)
    syn_items_1235(ctx, res)
    for item in (
        item7_needs_choice,
        item8_link_idempotent,
        item9_same_as,
        item10_imported,
        item11_two_writers,
    ):
        item(ctx, res)
    res.metrics["violations"] = float(len(res.violations))
    return res
