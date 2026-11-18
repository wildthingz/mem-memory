"""B-W1 (entity resolution), B-W2 (duplicates) and B-W3 (conflict surfacing, supersession) on SYN.

Each metric is computed per seed (benchmarks-v2 §5.1: the pipeline is deterministic per seed, so no
bootstrap over dependent ops); the gate value is the WORST seed (min for rates that must be high,
max for rates that must be low), and the per-seed values are in `extra["per_seed"]`.
T0 runs seed 1, size S, on whatever server the factory opens; the held-out gate runs seeds 101-105.
"""

from __future__ import annotations

import math
import threading
from datetime import UTC, datetime, timedelta

from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import sql
from bench.synth import data as D
from bench.synth import score, w3_diagnose
from bench.synth.baselines import baseline_metrics
from bench.synth.run import SeedRun, per_seed, replay_seed, require_p3, setup_of

Q = "2026-10-22T09:00:00Z"


def _seed_loop(ctx: Context, res: SuiteResult, tag: str, fn) -> dict[int, dict[str, float]]:
    setup = setup_of(ctx)
    require_p3(ctx)
    out: dict[int, dict[str, float]] = {}
    for seed in setup.seeds:
        run = replay_seed(ctx, seed, setup.size, "syn")
        out[seed] = fn(run, res)
    res.extra["per_seed"] = {str(s): v for s, v in out.items()}
    res.extra["dataset"] = {"name": "SYN", "version": D.SYN_VERSION, "size": setup.size,
                            "seeds": list(setup.seeds), "n": len(setup.seeds)}  # fmt: skip
    res.tier = "t0" if setup.split == "fixture" else "t1"
    res.mode = "oracle_client"
    return out


def _agg(res: SuiteResult, ps: dict[int, dict[str, float]], worst: dict[str, str]) -> None:
    for metric, fn in worst.items():
        res.metrics[metric] = per_seed(ps, metric, fn)
    for metric in worst:
        vals = [v[metric] for v in ps.values() if not math.isnan(v[metric])]
        if vals:
            res.metrics[f"{metric}_median"] = sorted(vals)[len(vals) // 2]


# ---- B-W1 ---------------------------------------------------------------------------------------


def run_b_w1(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-W1")

    def per(run: SeedRun, res: SuiteResult) -> dict[str, float]:
        m = score.w1_metrics(run.logs, run.replay.eid)
        m |= baseline_metrics(run.logs, run.replay.eid)
        return m

    ps = _seed_loop(ctx, res, "w1", per)
    _agg(res, ps, {"wrong_merge_rate": "max", "ask_back_recall": "min", "ask_rate": "max",
                   "split_rate": "max", "final_link_accuracy": "min"})  # fmt: skip
    res.metrics["op_errors"] = sum(v["op_errors"] for v in ps.values())
    res.items = [{"seed": s, **v} for s, v in ps.items()]
    t0_table_cases(ctx, res)
    res.metrics["table_case_failures"] = float(
        sum(1 for i in res.items if "check" in i and not i["ok"])
    )
    return res


def t0_table_cases(ctx: Context, res: SuiteResult) -> None:
    """benchmarks-v2 B-W1 T0 table cases (exact)."""

    def fresh(name: str):
        c = ctx.factory(ctx.store(f"b_w1_{name}.db"), scope="personal", tz=D.TZ)
        return c

    def put(c, text, about, now=Q, **kw):
        return c.call("remember", {"text": text, "type": "fact", "about": about, **kw}, now=now)

    def ents(c) -> int:
        return sql(c.db_path, "SELECT count(*) FROM nodes WHERE kind='entity'")[0][0]

    def mems(c) -> int:
        return sql(c.db_path, "SELECT count(*) FROM nodes WHERE kind='memory'")[0][0]

    c = fresh("two_anas")
    try:
        put(c, "Ana Silva speaks Dutch.", ["new:person:Ana Silva"])
        put(c, "Ana Lopez speaks Polish.", ["new:person:Ana Lopez"])
        r = put(c, "Ana likes tea.", ["Ana"])
        res.check("table_two_anas_ask", r.structured["status"] == "needs_choice",
                  str(r.structured["status"]))  # fmt: skip
    finally:
        c.close()
    c = fresh("subset")
    try:
        put(c, "Ana speaks Dutch.", ["new:person:Ana"])
        r = put(c, "Ana Lopez speaks Polish.", ["Ana Lopez"])
        res.check("table_ana_lopez_with_only_ana_known_asks_not_merges",
                  r.structured["status"] == "needs_choice", str(r.structured["status"]))  # fmt: skip
    finally:
        c.close()
    c = fresh("typo")
    try:
        put(c, "Katarina Johansson speaks Dutch.", ["new:person:Katarina Johansson"])
        before = ents(c)
        r = put(c, "Katarina Johanssen speaks Polish.", ["Katarina Johanssen"])
        res.check("table_typo_links", r.structured["status"] == "saved" and ents(c) == before
                  and "(matched" in " ".join(r.structured["subjects"]), str(r.structured))  # fmt: skip
    finally:
        c.close()
    c = fresh("atlas")
    try:
        put(c, "Atlas launches at Bellhall.", ["new:project:Atlas"])
        # a second type under a taken name asks first (ADR 0027); the writer says it is distinct
        put(c, "Atlas mascot is Marmot.", ["new:org:Atlas"], distinct_new=True)
        r = put(c, "Atlas uses Kafka.", ["Atlas"])
        res.check("table_atlas_project_vs_org_asks", r.structured["status"] == "needs_choice",
                  str(r.structured["status"]))  # fmt: skip
    finally:
        c.close()
    c = fresh("unknown_recall")
    try:
        put(c, "Ana Silva speaks Dutch.", ["new:person:Ana Silva"])
        e0, m0 = ents(c), mems(c)
        r = c.call("recall", {"query": "Ana Silva speaks", "about": ["Unknown"]}, now=Q)
        res.check("table_recall_unknown_about_writes_nothing",
                  (ents(c), mems(c)) == (e0, m0), f"entities {e0}->{ents(c)}")  # fmt: skip
        res.check("table_recall_unknown_about_filter_dropped",
                  "Unknown" in " ".join(map(str, r.structured.get("filters_dropped", []))),
                  str(r.structured.get("filters_dropped")))  # fmt: skip
    finally:
        c.close()


# ---- B-W2 ---------------------------------------------------------------------------------------


def run_b_w2(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-W2")

    def per(run: SeedRun, res: SuiteResult) -> dict[str, float]:
        mem = score.read_memories(run.db)
        return score.w2_metrics(run.syn, run.logs, mem, run.mids)

    ps = _seed_loop(ctx, res, "w2", per)
    _agg(res, ps, {"exists_rate": "min", "paraphrase_surfacing": "min",
                   "final_duplicate_rate": "max", "wrong_invalidation_compat": "max"})  # fmt: skip
    res.items = [{"seed": s, **v} for s, v in ps.items()]
    concurrency_case(ctx, res)
    superseded_original_case(ctx, res)
    res.metrics["case_failures"] = float(sum(1 for i in res.items if "check" in i and not i["ok"]))
    return res


def concurrency_case(ctx: Context, res: SuiteResult, trials: int = 20) -> None:
    """Two clients `remember` the same text at the same time: exactly one row, the other `exists`."""
    bad = []
    for t in range(trials):
        db = ctx.store(f"b_w2_conc{t}.db")
        c0 = ctx.factory(db, scope="personal", tz=D.TZ)
        c0.call("remember", {"text": "Seed memory for the concurrency trial", "type": "fact",
                             "about": ["new:topic:Conc"]}, now=Q)  # fmt: skip
        c0.close()
        barrier = threading.Barrier(2)
        out: list = [None, None]

        def worker(k: int, db=db, barrier=barrier, out=out) -> None:
            c = ctx.factory(db, scope="personal", tz=D.TZ)
            try:
                barrier.wait(5)
                out[k] = c.call(
                    "remember",
                    {
                        "text": "Ana Silva speaks Dutch at the weekend.",
                        "type": "fact",
                        "about": ["Conc"],
                    },
                    now=Q,
                ).structured
            except Exception as e:  # noqa: BLE001
                out[k] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                c.close()

        ths = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(30)
        rows = sql(db, "SELECT count(*) FROM nodes WHERE kind='memory' AND body LIKE 'Ana Silva%'")[
            0
        ][0]
        statuses = sorted(str(o.get("status", o)) for o in out)
        if rows != 1 or statuses != ["exists", "saved"]:
            bad.append(f"trial {t}: rows {rows}, replies {statuses}")
    res.check("concurrent_same_text_one_row_other_exists", not bad, "; ".join(bad[:3]))


def superseded_original_case(ctx: Context, res: SuiteResult) -> None:
    """N2: the same text as a superseded memory is `saved` with a new id, never `exists`."""
    c = ctx.factory(ctx.store("b_w2_n2.db"), scope="personal", tz=D.TZ)
    try:

        def put(text, now):
            return c.call(
                "remember",
                {
                    "text": text,
                    "type": "fact",
                    "about": ["new:person:Ana Silva" if now == Q else "Ana Silva"],
                    "as_of": now[:10],
                },
                now=now,
            )

        a = put("Ana Silva lives in Lisbon.", Q).structured
        b = put("Ana Silva lives in Porto.", "2026-10-23T09:00:00Z").structured
        c.call("link", {"src": b["id"], "rel": "supersedes", "dst": a["id"]},
               now="2026-10-23T09:01:00Z")  # fmt: skip
        again = put("Ana Silva lives in Lisbon.", "2026-10-24T09:00:00Z").structured
        res.check("n2_resave_superseded_text_is_saved_not_exists",
                  again["status"] == "saved" and again["id"] not in (a["id"], b["id"]),
                  str(again["status"]))  # fmt: skip
        exact = put("Ana Silva lives in Lisbon.", "2026-10-24T10:00:00Z").structured
        res.check(
            "n2_exact_duplicate_of_open_memory_is_exists",
            exact["status"] == "exists" and exact["id"] == again["id"],
            str(exact["status"]),
        )
    finally:
        c.close()


# ---- B-W3 ---------------------------------------------------------------------------------------


def run_b_w3(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-W3")

    def per(run: SeedRun, res: SuiteResult) -> dict[str, float]:
        mem = score.read_memories(run.db)
        m = score.w3_metrics(run.syn, run.logs, mem, run.mids)
        m["history"] = 1.0 if not run.history else 0.0
        m["history_violations"] = float(len(run.history))
        diffs = score.graph_diffs(run.syn, mem, run.mids)
        m["graph_diffs"] = float(len(diffs))
        res.extra.setdefault("graph_diffs", {})[str(run.syn.seed)] = diffs[:20]
        res.extra.setdefault("history_violations", {})[str(run.syn.seed)] = run.history[:20]
        res.extra.setdefault("diagnosis", {})[str(run.syn.seed)] = {
            "missed_slots": w3_diagnose.missed_slots(run.logs),
            "chain": w3_diagnose.chain_report(run.syn, run.logs, mem, run.mids),
            "subtype_misses": {lg.op["i"]: lg.op["gold"].get("subtype") or lg.op["kind"]
                               for lg in run.logs if lg.op["kind"] in ("update", "dup_paraphrase")
                               and lg.status == "saved" and not lg.surfaced}}  # fmt: skip
        return m

    ps = _seed_loop(ctx, res, "w3", per)
    _agg(res, ps, {"surfacing_recall": "min", "list_precision": "min", "chain_correctness": "min",
                   "wrong_invalidation": "max", "slot_accuracy": "min", "history": "min",
                   "graph_diffs": "max", "max_list_len": "max"})  # fmt: skip
    res.items = [{"seed": s, **v} for s, v in ps.items()]
    setup = setup_of(ctx)
    if setup.split == "fixture":
        res.check("t0_seed1_graph_equals_expected_state", res.metrics["graph_diffs"] == 0,
                  f"{res.metrics['graph_diffs']:g} differences: {res.extra['graph_diffs']}")  # fmt: skip
    res.check("possible_conflicts_capped_at_5", res.metrics["max_list_len"] <= 5,
              f"longest list {res.metrics['max_list_len']:g}")  # fmt: skip
    old_attribute_case(ctx, res)
    res.metrics["case_failures"] = float(sum(1 for i in res.items if "check" in i and not i["ok"]))
    return res


def old_attribute_case(ctx: Context, res: SuiteResult) -> None:
    """A22 regression: an old attribute update after >= 60 intervening same-subject writes about
    other attributes must still surface the old memory."""
    c = ctx.factory(ctx.store("b_w3_a22.db"), scope="personal", tz=D.TZ)
    try:
        first = c.call(
            "remember",
            {
                "text": "Ana Silva lives in Lisbon. Said with a smile.",
                "type": "fact",
                "about": ["new:person:Ana Silva"],
                "as_of": "2026-01-05",
            },
            now="2026-01-05T09:00:00Z",
        ).structured
        langs = ["Kvorlan", "Mirzeti", "Dolvani", "Pesorin", "Tavilo", "Zerumi", "Bolnari"]
        n = 0
        for i in range(62):
            val = f"{langs[i % len(langs)]}{'x' * (i // len(langs))}{i}"
            at = (datetime(2026, 2, 1, 8, tzinfo=UTC) + timedelta(minutes=i)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            c.call(
                "remember",
                {
                    "text": f"Ana Silva speaks {val}.",
                    "type": "fact",
                    "about": ["Ana Silva"],
                    "as_of": "2026-02-01",
                },
                now=at,
            )
            n += 1
        r = c.call("remember", {"text": "Now Ana Silva lives in Porto. Added to the running notes.",
                                "type": "fact", "about": ["Ana Silva"], "as_of": "2026-06-01"},
                   now="2026-06-01T09:00:00Z").structured  # fmt: skip
        ids = [x["id"] if isinstance(x, dict) else str(x).split()[0]
               for x in r["possible_conflicts"]]  # fmt: skip
        res.check("a22_old_attribute_update_after_62_writes_surfaces", first["id"] in ids,
                  f"{n} intervening writes; conflicts {ids}")  # fmt: skip
    finally:
        c.close()
