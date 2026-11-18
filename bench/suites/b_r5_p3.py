"""B-R5 as gated at P3 (benchmarks-v2 B-R5) and B-R2 (ii) missing-slot abstention.

T0 cases (exact): (a) the November example with `link`-derived intervals, (b) a late-arriving older
fact re-derives the chain, (c) a `link` in the wrong direction is swapped and reported, (d) a
retracted member is excluded and the chain re-derives, (e) date-only input (the P1 check), (f) a
same-instant correction: every predecessor gets an empty interval, no error, only the last by
(recorded_at, id) is returned.

SYN read probes, per seed, after the oracle-client replay (`queries.jsonl`):
- leakage = #current-value queries whose S_q holds a slot memory not valid at t / #current-value
  queries (validity from the SYN gold intervals);
- as-of accuracy = #as-of queries whose top-ranked slot memory in S_q is the one valid at D / #as-of;
- current-value Recall@1 = #current-value queries whose FIRST row of S_q is the gold current memory /
  #current-value queries.
B-R2 (ii): abstention recall = #missing-slot queries with S_q = empty / #missing-slot queries.
Gate value of every SYN metric = the worst seed (benchmarks-v2 §5.1).
"""

from __future__ import annotations

import math
from typing import Any

from bench.fixtures import streams as fx
from bench.suites import b_r5
from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import node_num, sql
from bench.synth import data as D
from bench.synth.generator import local_midnight_utc as u
from bench.synth.run import per_seed, replay_seed, require_p3, setup_of

NOW = "2026-10-22T08:00:00Z"
TODAY = "2026-10-22T09:00:00Z"


def _ids(structured: dict[str, Any]) -> list[str]:
    return [r["id"] for r in structured["rows"]]


def _remember(c, text: str, as_of: str, about: str, now: str = NOW) -> str:
    return c.call("remember", {"text": text, "type": "fact", "about": [about], "as_of": as_of},
                  now=now).structured["id"]  # fmt: skip


def _link(c, src: str, dst: str, now: str = NOW) -> Any:
    return c.call("link", {"src": src, "rel": "supersedes", "dst": dst}, now=now)


def case_a_link(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("r5_a_link.db"), scope="personal", tz=D.TZ)
    try:
        m302 = _remember(c, fx.NOV_M302["text"], fx.NOV_M302["from"], "new:person:Ana")
        m451 = _remember(c, fx.NOV_M451["text"], fx.NOV_M451["from"], "Ana")
        _link(c, m451, m302)
        stored = {n: sql(c.db_path, "SELECT valid_from, valid_to FROM nodes WHERE id=?",
                         (node_num(i),))[0] for n, i in (("M302", m302), ("M451", m451))}  # fmt: skip
        res.check("a_link_derives_intervals", stored == {
            "M302": ("2026-06-03T00:00:00Z", "2026-11-01T00:00:00Z"),
            "M451": ("2026-11-01T00:00:00Z", None)}, str(stored))  # fmt: skip
        label = {"M302": m302, "M451": m451}
        for as_of, expect in fx.NOV_CASES:
            r = c.call("recall", {"query": fx.NOV_QUERY, "as_of": as_of, "limit": 8}, now=TODAY)
            res.check(f"a_link_as_of_{as_of}", set(_ids(r.structured)) == {label[x] for x in expect},
                      f"want {expect} got {_ids(r.structured)}")  # fmt: skip
        r = c.call("recall", {"query": fx.NOV_QUERY}, now=TODAY)
        res.check("a_link_default_t_is_now", set(_ids(r.structured)) == {m302},
                  str(_ids(r.structured)))  # fmt: skip
        r = c.call("recall", {"query": fx.NOV_QUERY}, now="2026-12-01T09:00:00Z")
        res.check("a_link_after_boundary", set(_ids(r.structured)) == {m451},
                  str(_ids(r.structured)))  # fmt: skip
    finally:
        c.close()


def _weekly(c, day: str, as_of: str, first: bool = False) -> str:
    return _remember(c, f"Ana weekly planning session is on {day}.", as_of,
                     "new:person:Ana" if first else "Ana")  # fmt: skip


def case_b(ctx: Context, res: SuiteResult) -> None:
    """m3 (from 08-01) is linked after m1 (06-03) and m2 (11-01)."""
    c = ctx.factory(ctx.store("r5_b.db"), scope="personal", tz=D.TZ)
    try:
        m1 = _weekly(c, "Monday", "2026-06-03", True)
        m2 = _weekly(c, "Thursday", "2026-11-01")
        _link(c, m2, m1)
        m3 = _weekly(c, "Wednesday", "2026-08-01")
        _link(c, m3, m1)
        _link(c, m2, m3)
        got = {n: sql(c.db_path, "SELECT valid_from, valid_to FROM nodes WHERE id=?",
                      (node_num(i),))[0] for n, i in (("m1", m1), ("m2", m2), ("m3", m3))}  # fmt: skip
        res.check("b_chain_rederives", got == {
            "m1": (u("2026-06-03"), u("2026-08-01")),
            "m3": (u("2026-08-01"), u("2026-11-01")),
            "m2": (u("2026-11-01"), None)}, str(got))  # fmt: skip
        for as_of, want in (("2026-07-01", m1), ("2026-09-15", m3), (None, m3)):
            args = {"query": "Ana weekly planning session"} | ({"as_of": as_of} if as_of else {})
            r = c.call("recall", args, now=TODAY)
            res.check(f"b_as_of_{as_of or 'today'}", _ids(r.structured) == [want],
                      f"want {want} got {_ids(r.structured)}")  # fmt: skip
    finally:
        c.close()


def case_c(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("r5_c.db"), scope="personal", tz=D.TZ)
    try:
        old = _weekly(c, "Monday", "2026-06-03", True)
        new = _weekly(c, "Thursday", "2026-11-01")
        r = _link(c, old, new)  # the "wrong" direction: the older fact supersedes the newer
        res.check("c_reply_reports_the_swap", "swap" in r.text.lower(), r.text[:120])
        got = sql(c.db_path, "SELECT valid_to FROM nodes WHERE id=?", (node_num(old),))[0][0]
        res.check("c_older_memory_closed_at_newer_start", got == u("2026-11-01"), str(got))
        open_new = sql(c.db_path, "SELECT valid_to FROM nodes WHERE id=?", (node_num(new),))[0][0]
        res.check("c_newer_memory_stays_open", open_new is None, str(open_new))
    finally:
        c.close()


def case_d(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("r5_d.db"), scope="personal", tz=D.TZ)
    try:
        m1 = _weekly(c, "Monday", "2026-06-03", True)
        m2 = _weekly(c, "Wednesday", "2026-08-01")
        m3 = _weekly(c, "Thursday", "2026-11-01")
        _link(c, m2, m1)
        _link(c, m3, m2)
        c.call("forget", {"id": m2, "reason": "entered by mistake"}, now=TODAY)
        got = sql(c.db_path, "SELECT valid_to FROM nodes WHERE id=?", (node_num(m1),))[0][0]
        res.check("d_chain_rederives_without_the_retracted", got == u("2026-11-01"), str(got))
        r = c.call("recall", {"query": "Ana weekly planning session", "as_of": "2026-09-15"},
                   now=TODAY)  # fmt: skip
        res.check(
            "d_retracted_member_excluded", _ids(r.structured) == [m1], str(_ids(r.structured))
        )
    finally:
        c.close()


def case_f(ctx: Context, res: SuiteResult) -> None:
    c = ctx.factory(ctx.store("r5_f.db"), scope="personal", tz=D.TZ)
    try:
        day = "2026-09-01"
        m1 = _remember(c, "Ana weekly planning session is on Monday.", day, "new:person:Ana",
                       now="2026-09-01T08:00:00Z")  # fmt: skip
        m2 = _remember(c, "Ana weekly planning session is on Tuesday.", day, "Ana",
                       now="2026-09-01T09:00:00Z")  # fmt: skip
        m3 = _remember(c, "Ana weekly planning session is on Friday.", day, "Ana",
                       now="2026-09-01T10:00:00Z")  # fmt: skip
        ok, detail = True, ""
        try:
            _link(c, m2, m1, "2026-09-01T09:01:00Z")
            _link(c, m3, m2, "2026-09-01T10:01:00Z")
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        res.check("f_no_error_on_same_instant_chain", ok, detail)
        iv = {n: sql(c.db_path, "SELECT valid_from, valid_to FROM nodes WHERE id=?",
                     (node_num(i),))[0] for n, i in (("m1", m1), ("m2", m2), ("m3", m3))}  # fmt: skip
        t0 = iv["m1"][0]
        res.check(
            "f_predecessors_get_empty_intervals",
            iv["m1"] == (t0, t0) and iv["m2"] == (t0, t0) and iv["m3"] == (t0, None),
            str(iv),
        )
        for as_of in (day, "2026-09-02", "2026-12-01"):
            r = c.call(
                "recall", {"query": "Ana weekly planning session", "as_of": as_of}, now=TODAY
            )
            res.check(f"f_only_the_last_is_returned_{as_of}", _ids(r.structured) == [m3],
                      str(_ids(r.structured)))  # fmt: skip
    finally:
        c.close()


def _cases(ctx: Context, res: SuiteResult) -> None:
    case_a_link(ctx, res)
    case_b(ctx, res)
    case_c(ctx, res)
    case_d(ctx, res)
    b_r5.case_e(ctx, res)
    case_f(ctx, res)


# ---- SYN probes ---------------------------------------------------------------------------------


def _valid_at(mem: dict[str, Any], t: str) -> bool:
    if mem["status"] != "active":
        return False
    return mem["valid_from"] <= t and (mem["valid_to"] is None or t < mem["valid_to"])


def probe_seed(ctx: Context, seed: int, size: str) -> dict[str, float]:
    run = replay_seed(ctx, seed, size, "syn")
    syn, inv = run.syn, {f"m{m}": op for op, m in run.mids.items()}
    mems = syn.expected["memories"]
    c = ctx.factory(run.db, scope="personal", tz=D.TZ, now=syn.expected["end_now"])
    cur_n = cur_leak = cur_hit = asof_n = asof_hit = miss_n = miss_abstain = 0
    detail: list[dict[str, Any]] = []
    try:
        for q in syn.queries:
            args = {"query": q["query"]} | ({"as_of": q["as_of"]} if q.get("as_of") else {})
            r = c.call("recall", args, now=q["now"])
            rows = _ids(r.structured)
            t = q.get("as_of") or q["now"]
            slot_rows = [inv[i] for i in rows if inv.get(i) in set(q.get("slot_ops", []))]
            if q["type"] == "current":
                cur_n += 1
                leak = any(not _valid_at(mems[str(op)], t) for op in slot_rows)
                cur_leak += leak
                hit = bool(rows) and inv.get(rows[0]) == q["gold"][0]
                cur_hit += hit
                detail.append(
                    {"id": q["id"], "rows": rows, "ok": hit, "leaked": leak, "slot_ops": slot_rows}
                )
            elif q["type"] == "as_of":
                asof_n += 1
                hit = bool(slot_rows) and slot_rows[0] == q["gold"][0]
                asof_hit += hit
                detail.append({"id": q["id"], "rows": rows, "ok": hit})
            else:
                miss_n += 1
                abstain = not rows
                miss_abstain += abstain
                detail.append({"id": q["id"], "rows": rows, "ok": abstain})
    finally:
        c.close()
    return {"leakage": cur_leak / cur_n if cur_n else math.nan,
            "as_of_accuracy": asof_hit / asof_n if asof_n else math.nan,
            "current_recall_at_1": cur_hit / cur_n if cur_n else math.nan,
            "missing_slot_abstention": miss_abstain / miss_n if miss_n else math.nan,
            "current_n": float(cur_n), "as_of_n": float(asof_n), "missing_n": float(miss_n),
            "_detail": detail}  # fmt: skip


def _syn_probes(ctx: Context) -> dict[int, dict[str, float]]:
    setup = setup_of(ctx)
    require_p3(ctx)
    return {s: probe_seed(ctx, s, setup.size) for s in setup.seeds}


def run_b_r5(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-R5", mode="oracle_client")
    require_p3(ctx)
    _cases(ctx, res)
    res.metrics["case_failures"] = float(len(res.violations))
    per = _syn_probes(ctx)
    setup = setup_of(ctx)
    res.tier = "t0" if setup.split == "fixture" else "t1"
    res.metrics["leakage"] = per_seed(per, "leakage", "max")
    res.metrics["as_of_accuracy"] = per_seed(per, "as_of_accuracy", "min")
    res.metrics["current_recall_at_1"] = per_seed(per, "current_recall_at_1", "min")
    res.extra["per_seed"] = {str(s): {k: v for k, v in m.items() if k != "_detail"}
                             for s, m in per.items()}  # fmt: skip
    res.extra["leaked_queries"] = {str(s): [d for d in m["_detail"] if d.get("leaked")]
                                   for s, m in per.items()}  # fmt: skip
    res.extra["failed_probes"] = {str(s): [d for d in m["_detail"] if not d["ok"]][:20]
                                  for s, m in per.items()}  # fmt: skip
    res.extra["dataset"] = {"name": "SYN", "version": D.SYN_VERSION, "size": setup.size,
                            "seeds": list(setup.seeds), "n": len(setup.seeds)}  # fmt: skip
    return res


def run_b_r2_ii(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-R2", mode="oracle_client")
    per = _syn_probes(ctx)
    setup = setup_of(ctx)
    res.tier = "t0" if setup.split == "fixture" else "t1"
    res.metrics["missing_slot_abstention"] = per_seed(per, "missing_slot_abstention", "min")
    res.metrics["missing_n_min"] = min(m["missing_n"] for m in per.values())
    res.extra["per_seed"] = {str(s): {k: v for k, v in m.items() if k != "_detail"}
                             for s, m in per.items()}  # fmt: skip
    res.extra["dataset"] = {"name": "SYN missing-slot", "version": D.SYN_VERSION,
                            "size": setup.size, "seeds": list(setup.seeds), "n": len(setup.seeds)}  # fmt: skip
    res.items = [{"seed": s, **{k: v for k, v in m.items() if k != "_detail"}}
                 for s, m in per.items()]  # fmt: skip
    return res


def run_b_r2_p3(ctx: Context) -> SuiteResult:
    """B-R2 at P3: (i) evidence-removed abstention must not regress after re-calibration with SYN
    dev (the P2 suite on the new gate), and (ii) SYN missing-slot abstention."""
    from bench.suites import b_r2

    res = b_r2.run(ctx)  # (i): needs the LME setup, UNAVAILABLE without it
    ii = run_b_r2_ii(ctx)
    res.metrics |= ii.metrics
    res.extra["syn_missing_slot"] = ii.extra
    res.notes.append("B-R2 (ii): " + ", ".join(f"{k}={v:g}" for k, v in ii.metrics.items()))
    return res
