"""B-R5 T0 cases (a) and (e) on fixture intervals (P1 form, benchmarks-v2 B-R5).

(a) the November example: half-open eligibility, current reads exclude not-yet-valid and
    no-longer-valid memories. The fixture writes the intervals (valid_to set directly in SQL).
(e) date-only input is read in the configured zone, converted to UTC: on the write side
    (`remember as_of`) it is local midnight, the start of the day; on the read side (`recall as_of`)
    it is the end of that local day (ADR 0028).
Cases (b)-(d), (f) need `link` and belong to P3.
"""

from __future__ import annotations

from bench.fixtures import streams as fx
from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import node_num, sql, sql_write


def _ids(structured: dict) -> set[str]:
    return {r["id"] for r in structured["rows"]}


def case_a(ctx: Context, res: SuiteResult) -> None:
    client = ctx.factory(ctx.store("b_r5_a.db"), scope="personal")
    try:
        label: dict[str, str] = {}
        for name, spec in (("M302", fx.NOV_M302), ("M451", fx.NOV_M451)):
            r = client.call(
                "remember",
                {"text": spec["text"], "type": spec["type"], "about": ["new:person:Ana"]
                 if name == "M302" else ["Ana"], "as_of": spec["from"]},
                now="2026-10-22T08:00:00Z",
            )  # fmt: skip
            label[name] = r.structured["id"]
            if spec["to"]:
                sql_write(client.db_path, "UPDATE nodes SET valid_to = ? WHERE id = ?",
                          (spec["to"], node_num(r.structured["id"])))  # fmt: skip
        stored = {
            n: sql(client.db_path, "SELECT valid_from, valid_to FROM nodes WHERE id = ?",
                   (node_num(i),))[0]
            for n, i in label.items()
        }  # fmt: skip
        res.check("a_fixture_intervals_stored", stored == {
            "M302": ("2026-06-03T00:00:00Z", "2026-11-01T00:00:00Z"),
            "M451": ("2026-11-01T00:00:00Z", None)}, str(stored))  # fmt: skip
        for as_of, expect in fx.NOV_CASES:
            r = client.call("recall", {"query": fx.NOV_QUERY, "as_of": as_of, "limit": 8},
                            now="2026-10-22T09:00:00Z")  # fmt: skip
            want = {label[x] for x in expect}
            res.check(f"a_as_of_{as_of}", _ids(r.structured) == want,
                      f"want {sorted(want)} got {sorted(_ids(r.structured))}")  # fmt: skip
        # No as_of: "now" is the op date (2026-10-22): only Monday is current.
        r = client.call("recall", {"query": fx.NOV_QUERY}, now="2026-10-22T09:00:00Z")
        res.check("a_default_t_is_now", _ids(r.structured) == {label["M302"]},
                  str(sorted(_ids(r.structured))))  # fmt: skip
        # Same query on 2026-12-01 as the clock, no as_of.
        r = client.call("recall", {"query": fx.NOV_QUERY}, now="2026-12-01T09:00:00Z")
        res.check("a_default_t_after_boundary", _ids(r.structured) == {label["M451"]},
                  str(sorted(_ids(r.structured))))  # fmt: skip
    finally:
        client.close()


def case_e(ctx: Context, res: SuiteResult) -> None:
    client = ctx.factory(ctx.store("b_r5_e.db"), scope="personal")
    try:
        # Write side: valid_from column of a memory saved with a date-only as_of.
        for i, (date_only, utc) in enumerate(fx.DATE_ONLY_CASES):
            r = client.call("remember", {"text": f"Date-only probe number {i} on {date_only}",
                            "type": "event", "about": ["new:topic:Probe" if i == 0 else "Probe"],
                            "as_of": date_only}, now="2026-10-22T08:00:00Z")  # fmt: skip
            got = sql(client.db_path, "SELECT valid_from FROM nodes WHERE id = ?",
                      (node_num(r.structured["id"]),))[0][0]  # fmt: skip
            res.check(f"e_write_{date_only}", got == utc, f"valid_from {got!r}, want {utc!r}")
        # Read side (ADR 0028): date-only 2026-11-01 reads through the END of that local day,
        # 2026-11-01T22:59:59Z in Europe/Amsterdam. A memory from 22:30Z (23:30 local, same day) is
        # visible; one from 23:30Z (00:30 local on 11-02) is not. Read as local midnight the first
        # would be hidden; read as the end of the UTC day the second would be visible.
        r = client.call("remember", {"text": "Probe schedule is Monday", "type": "fact",
                        "about": ["Probe"], "as_of": "2026-11-01T22:30:00Z"},
                        now="2026-10-22T08:00:00Z")  # fmt: skip
        late = r.structured["id"]
        r = client.call("remember", {"text": "Probe lunch is at noon", "type": "fact",
                        "about": ["Probe"], "as_of": "2026-11-01T23:30:00Z"},
                        now="2026-10-22T08:00:00Z")  # fmt: skip
        nxt = r.structured["id"]
        r = client.call("recall", {"query": "Probe schedule Monday", "as_of": "2026-11-01"},
                        now="2026-10-22T08:00:00Z")  # fmt: skip
        res.check(
            "e_read_date_only_covers_the_local_day",
            late in _ids(r.structured),
            "valid_from 11-01T22:30Z is 23:30 local on 11-01, before the day's end 11-01T22:59:59Z",
        )
        r = client.call("recall", {"query": "Probe lunch noon", "as_of": "2026-11-01"},
                        now="2026-10-22T08:00:00Z")  # fmt: skip
        res.check(
            "e_read_date_only_ends_at_local_midnight",
            nxt not in _ids(r.structured),
            "valid_from 11-01T23:30Z is 00:30 local on 11-02, after the day's end 11-01T22:59:59Z",
        )
        r = client.call("recall", {"query": "Probe lunch noon", "as_of": "2026-11-02"},
                        now="2026-10-22T08:00:00Z")  # fmt: skip
        res.check(
            "e_read_next_day_sees_it",
            nxt in _ids(r.structured),
            "the end of 11-02 local = 11-02T22:59:59Z is after valid_from",
        )
    finally:
        client.close()


def run(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-R5", mode="fixture")
    case_a(ctx, res)
    case_e(ctx, res)
    res.metrics["case_failures"] = float(len(res.violations))
    return res
