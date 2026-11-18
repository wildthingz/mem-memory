"""B-W6 items 1, 2, 3, 5, 6 on fixture op streams (P1 form, benchmarks-v2 B-W6).

1 determinism (canonical dump, `recorded_at` column = op date)  2 nothing deleted
3 every memory has a live `about` edge plus session/host       5 valid_to >= valid_from
6 work and personal stores never share ids
Items 4 and 7-11 need `link`/`forget`/the resolver (P3) and are not run here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from bench.dump import canonical_dump
from bench.fixtures import streams as fx
from bench.suites.base import Client, Context, SuiteResult
from bench.suites.helpers import run_ops, sql


def _state(db: Path) -> dict[str, Any]:
    nodes = {r[0]: r[1:] for r in sql(
        db, "SELECT id, name, body, content_hash, valid_from, recorded_at, origin FROM nodes")}  # fmt: skip
    edges = {r[0]: r[1:] for r in sql(db, "SELECT id, src, dst, rel, recorded_at FROM edges")}
    chunks = {r[0]: r[1:] for r in sql(db, "SELECT id, node_id, chunk_key, text FROM chunks")}
    return {"nodes": nodes, "edges": edges, "chunks": chunks}


def _run_stream(ctx: Context, name: str, ops: list[dict[str, Any]], scope: str = "personal"):
    client = ctx.factory(ctx.store(name), scope=scope)
    return client, run_ops(client, ops)


def item1_determinism(ctx: Context, res: SuiteResult) -> None:
    dumps, results = [], []
    for run in ("a", "b"):
        client, out = _run_stream(ctx, f"b_w6_1{run}.db", fx.OP_STREAM)
        client.close()
        dumps.append(canonical_dump(client.db_path))
        results.append(client.db_path)
    res.check("1_dumps_byte_identical", dumps[0] == dumps[1],
              f"{len(dumps[0])} vs {len(dumps[1])} bytes")  # fmt: skip
    res.check("1_dump_not_empty", b'"t": "nodes"' in dumps[0])
    op_dates = {op["now"] for op in fx.OP_STREAM}
    recorded = sql(results[0], "SELECT id, kind, recorded_at FROM nodes ORDER BY id")
    off = [(i, k, t) for i, k, t in recorded if t not in op_dates]
    res.check("1_recorded_at_is_op_date_column", not off, f"not an op date: {off}")
    # Each saved memory carries exactly the date of the op that created it.
    client, out = _run_stream(ctx, "b_w6_1c.db", fx.OP_STREAM)
    client.close()
    wrong = []
    for op in out:
        s = op["structured"]
        if op["tool"] == "remember" and s.get("status") == "saved":
            got = sql(client.db_path, "SELECT recorded_at FROM nodes WHERE id = ?",
                      (int(s["id"][1:]),))[0][0]  # fmt: skip
            if got != op["now"]:
                wrong.append((s["id"], got, op["now"]))
    res.check("1_memory_recorded_at_equals_its_op_date", not wrong, str(wrong))


def item2_nothing_deleted(ctx: Context, res: SuiteResult) -> None:
    client = ctx.factory(ctx.store("b_w6_2.db"), scope="personal")
    prev = _state(client.db_path)
    problems: list[str] = []
    try:
        for n, op in enumerate(fx.OP_STREAM):
            client.call(op["tool"], op["args"], now=op["now"])
            cur = _state(client.db_path)
            for table in prev:
                lost = set(prev[table]) - set(cur[table])
                if lost:
                    problems.append(f"op {n}: {table} ids vanished {sorted(lost)}")
                changed = [
                    k for k in prev[table] if k in cur[table] and cur[table][k] != prev[table][k]
                ]
                if changed:
                    problems.append(f"op {n}: {table} rows rewritten {sorted(changed)}")
            prev = cur
    finally:
        client.close()
    res.check("2_no_row_deleted_or_rewritten", not problems, "; ".join(problems))
    res.check("2_stream_wrote_rows", len(prev["nodes"]) >= 6, f"{len(prev['nodes'])} nodes")


def item3_provenance(ctx: Context, res: SuiteResult) -> None:
    client, _ = _run_stream(ctx, "b_w6_3.db", fx.OP_STREAM)
    client.close()
    db = client.db_path
    orphans = sql(db, "SELECT n.id FROM nodes n WHERE n.kind = 'memory' AND NOT EXISTS ("
                  "SELECT 1 FROM edges e WHERE e.src = n.id AND e.rel = 'about' "
                  "AND e.removed_at IS NULL)")  # fmt: skip
    res.check("3_every_memory_has_live_about_edge", not orphans, f"orphans {orphans}")
    missing = sql(db, "SELECT id FROM nodes WHERE kind = 'memory' AND "
                  "(session IS NULL OR session = '' OR host IS NULL OR host = '')")  # fmt: skip
    res.check("3_session_and_host_columns_set", not missing, f"missing on {missing}")
    hosts = {r[0] for r in sql(db, "SELECT DISTINCT host FROM nodes WHERE kind = 'memory'")}
    res.check("3_host_is_bench", hosts == {"bench"}, str(hosts))
    n_mem = sql(db, "SELECT count(*) FROM nodes WHERE kind = 'memory'")[0][0]
    res.check("3_stream_saved_memories", n_mem >= 5, f"{n_mem} memories")


def item5_intervals(ctx: Context, res: SuiteResult) -> None:
    client, _ = _run_stream(ctx, "b_w6_5.db", fx.OP_STREAM)
    client.close()
    db = client.db_path
    bad = sql(db, "SELECT id, valid_from, valid_to FROM nodes WHERE valid_to IS NOT NULL "
              "AND valid_to < valid_from")  # fmt: skip
    res.check("5_valid_to_ge_valid_from_after_stream", not bad, str(bad))
    equal = sql(db, "SELECT n.id FROM nodes n WHERE n.valid_to = n.valid_from AND NOT EXISTS ("
                "SELECT 1 FROM edges e JOIN nodes s ON s.id = e.src WHERE e.dst = n.id "
                "AND e.rel = 'supersedes' AND e.removed_at IS NULL AND s.valid_from = n.valid_from)")  # fmt: skip
    res.check("5_equality_only_for_same_instant_predecessor", not equal, str(equal))
    # The CHECK itself: a reversed interval cannot be written, even around the app.
    conn = sqlite3.connect(db)
    try:
        nid = conn.execute("SELECT id FROM nodes WHERE kind = 'memory' LIMIT 1").fetchone()[0]
        try:
            with conn:
                conn.execute("UPDATE nodes SET valid_to = '1999-01-01T00:00:00Z' WHERE id = ?",
                             (nid,))  # fmt: skip
            enforced = False
        except sqlite3.IntegrityError:
            enforced = True
    finally:
        conn.close()
    res.check("5_check_constraint_rejects_reversed_interval", enforced)
    # Future-dated memory from the stream is stored open and not before its valid_from.
    fut = sql(
        db, "SELECT valid_from, valid_to FROM nodes WHERE valid_from = '2026-12-01T00:00:00Z'"
    )
    res.check(
        "5_future_dated_memory_stored_open", fut == [("2026-12-01T00:00:00Z", None)], str(fut)
    )


def _shown_ids(client: Client, ops: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for op in run_ops(client, ops):
        s = op["structured"]
        if op["tool"] == "remember":
            ids |= {x for x in [s.get("id")] if x}
    return ids


def item6_two_stores(ctx: Context, res: SuiteResult) -> None:
    """Ids are integers per store (architecture-v2 §2.3), so isolation is what is checked: what
    one store holds is invisible to the other, by `get` and by `recall`."""
    personal = ctx.factory(ctx.store("b_w6_6p.db"), scope="personal")
    work = ctx.factory(ctx.store("b_w6_6w.db"), scope="work")
    try:
        p_ids = _shown_ids(personal, fx.TWO_STORE_STREAM)
        res.check("6_stream_wrote_ids", bool(p_ids), f"{p_ids}")
        got = work.call("get", {"ids": sorted(p_ids)}, now="2026-10-22T09:00:00Z")
        found = [i["id"] for i in got.structured["items"] if i.get("found")]
        res.check("6_id_from_other_store_is_not_found", not found, f"found {found}")
        seen = work.call("recall", {"query": "Ana Lisbon standup Atlas release"},
                         now="2026-10-22T09:00:00Z")  # fmt: skip
        res.check("6_recall_does_not_cross_stores", not seen.structured["rows"],
                  str(seen.structured["rows"]))  # fmt: skip
        w_ids = _shown_ids(work, fx.TWO_STORE_STREAM)
        own = work.call("get", {"ids": sorted(w_ids)}, now="2026-10-22T09:00:00Z")
        res.check("6_own_ids_resolve_in_own_store",
                  all(i.get("found") for i in own.structured["items"]), str(own.structured))  # fmt: skip
    finally:
        personal.close()
        work.close()


def run(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-W6", mode="fixture")
    for item in (
        item1_determinism,
        item2_nothing_deleted,
        item3_provenance,
        item5_intervals,
        item6_two_stores,
    ):
        item(ctx, res)
    res.metrics["violations"] = float(len(res.violations))
    return res
