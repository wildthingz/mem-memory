"""B-E4 metrics, daily-review sampling, duplicate scan, PG-v2 build and the weekly runner, on a
synthetic events log (MEM2-46). Minimal tables stand in for the stores: the code under test reads
only `events`, `nodes`, `edges` and `chunks`."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import gates, p6_use, pgv2
from bench.cli import app
from bench.gold import pg
from bench.suites.base import SuiteResult

DAYS = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]  # ten working days


def make_db(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE events (id INTEGER PRIMARY KEY, at TEXT, session TEXT, host TEXT, tool TEXT,
                         args TEXT, result TEXT, latency_ms INTEGER);
    CREATE TABLE nodes (id INTEGER PRIMARY KEY, kind TEXT, status TEXT DEFAULT 'active', type TEXT,
                        date TEXT, name TEXT, body TEXT, valid_to TEXT, source_ref TEXT, origin TEXT);
    CREATE TABLE edges (id INTEGER PRIMARY KEY, src INT, dst INT, rel TEXT, removed_at TEXT);
    CREATE TABLE chunks (id INTEGER PRIMARY KEY, node_id INT, chunk_key TEXT, text TEXT);
    """)
    return c


def ev(c, at, tool, args, result, *, host="home", session="s1", lat=100):
    c.execute(
        "INSERT INTO events(at, session, host, tool, args, result, latency_ms)"
        " VALUES (?,?,?,?,?,?,?)",
        (at, session, host, tool, json.dumps(args), json.dumps(result), lat),
    )


@pytest.fixture()
def stores(tmp_path: Path) -> Path:
    p, w = make_db(tmp_path / "personal.db"), make_db(tmp_path / "work.db")
    for k, day in enumerate(DAYS):
        # 4 recalls a day at home (one empty), the first 3 days also 2 from work
        for j in range(4):
            ev(
                p,
                f"{day}T09:0{j}:00Z",
                "recall",
                {"query": f"home q{k}-{j}"},
                {"ids": [] if j == 3 else [f"m{j + 1}"], "tokens": 100 + j, "abstained": j == 3},
                session=f"h{k}",
                lat=100 + j,
            )
        if k < 3:
            for j in range(2):
                ev(
                    w,
                    f"{day}T14:0{j}:00Z",
                    "recall",
                    {"query": f"work q{k}-{j}"},
                    {"ids": ["m9"], "tokens": 90},
                    host="work-laptop",
                    session=f"w{k}",
                    lat=900 + 100 * j,
                )
    # bench traffic must be ignored
    for j in range(50):
        ev(
            p,
            "2026-11-03T03:00:00Z",
            "recall",
            {"query": f"b{j}"},
            {"ids": ["m1"]},
            host="bench-home",
        )
    # conflicts: m10 resolved by link, m11 by supersedes arg, m12 unresolved, m13 unresolved (no session)
    ev(
        p,
        "2026-11-04T10:00:00Z",
        "remember",
        {"text": "a"},
        {"status": "saved", "id": "m10", "possible_conflicts": [{"id": "m1"}]},
        session="c1",
    )
    ev(
        p,
        "2026-11-04T10:01:00Z",
        "link",
        {"src": "m10", "rel": "supersedes", "dst": "m1"},
        {"ok": True},
        session="c1",
    )
    ev(
        p,
        "2026-11-04T11:00:00Z",
        "remember",
        {"text": "b", "supersedes": "m2"},
        {"status": "saved", "id": "m11", "possible_conflicts": [{"id": "m2"}]},
        session="c2",
    )
    ev(
        p,
        "2026-11-04T12:00:00Z",
        "remember",
        {"text": "c"},
        {"status": "saved", "id": "m12", "possible_conflicts": [{"id": "m3"}]},
        session="c3",
    )
    ev(
        p,
        "2026-11-04T12:05:00Z",
        "link",
        {"src": "m3", "rel": "about", "dst": "e1"},
        {"ok": True},
        session="c3",
    )  # a link, but not of the new id
    ev(
        p,
        "2026-11-04T13:00:00Z",
        "remember",
        {"text": "d"},
        {"status": "saved", "id": "m13", "possible_conflicts": [{"id": "m3"}]},
        session=None,
    )
    for t in "efgh":  # four more new memories without conflicts
        ev(
            p,
            "2026-11-05T10:00:00Z",
            "remember",
            {"text": t},
            {"status": "saved", "id": "m20", "possible_conflicts": []},
            session="n",
        )
    p.commit(), w.commit()
    p.close(), w.close()
    return tmp_path


def reviews_for(stores: Path, marks: dict[str, tuple[str, str | None]]) -> dict:
    return {r: p6_use.Review(r, m, f) for r, (m, f) in marks.items()}


def test_live_events_exclude_bench_hosts(stores):
    live = p6_use.load_events(stores)
    assert not any((e.host or "").startswith("bench-") for e in live)
    assert len(p6_use.load_events(stores, include_bench=True)) - len(live) == 50


def test_review_sample_is_deterministic_and_capped(stores):
    events = p6_use.load_events(stores)
    a = p6_use.review_sample(events, "2026-11-03", 3)
    b = p6_use.review_sample(list(reversed(events)), "2026-11-03", 3)
    assert [e.ref for e in a] == [e.ref for e in b] and len(a) == 3
    assert all(e.tool == "recall" for e in a)
    assert [e.ref for e in a] == sorted((e.ref for e in a), key=p6_use.sha1_key)
    assert (
        len(p6_use.review_sample(events, "2026-11-03", 10)) == 6
    )  # 4 home + 2 work, fewer than 10


def test_review_marks_validation_and_latest_wins(tmp_path):
    f = tmp_path / "r.jsonl"
    p6_use.append_review(f, p6_use.Review("personal:1", "irrelevant"))
    p6_use.append_review(f, p6_use.Review("personal:1", "useful"))
    assert p6_use.load_reviews(f)["personal:1"].mark == "useful"
    with pytest.raises(ValueError):
        p6_use.append_review(f, p6_use.Review("personal:2", "should-have-found"))  # needs an id
    with pytest.raises(ValueError):
        p6_use.append_review(f, p6_use.Review("personal:2", "great"))


def test_unresolved_conflict_query(stores):
    n, left = p6_use.unresolved_conflicts(p6_use.load_events(stores))
    # m10 resolved by a link naming it, m11 by `supersedes`; m12's session links only m3; m13 has no session
    rows = {e.ref: e.result["id"] for e in p6_use.load_events(stores) if e.tool == "remember"}
    assert n == 4 and sorted(rows[r] for r in left) == ["m12", "m13"]


def test_e4_metrics_known_values(stores):
    events = p6_use.load_events(stores)
    # review every personal recall of the first two days
    day1 = [
        e
        for e in events
        if e.tool == "recall" and e.store == "personal" and e.at.startswith("2026-11-02")
    ]
    day2 = [
        e
        for e in events
        if e.tool == "recall" and e.store == "personal" and e.at.startswith("2026-11-03")
    ]
    marks = {}
    for e in day1 + day2:
        if e.ids:
            marks[e.ref] = ("useful", None)
        else:
            marks[e.ref] = ("should-have-found", "m77")
    marks[day1[0].ref] = ("irrelevant", None)
    rep = p6_use.compute_e4(events, reviews_for(stores, marks), work_overhead_ms=50.0)
    m = rep.metrics
    assert m["working_days"] == 10 and m["work_days"] == 3
    assert m["recalls"] == 40 + 6
    assert m["recalls_per_day_median"] == 4
    assert m["empty_rate"] == pytest.approx(10 / 46)
    # reviewed: 8 (6 non-empty: 5 useful, 1 irrelevant; 2 empty should-have-found)
    assert m["reviewed"] == 8 and m["reviewed_nonempty"] == 6
    assert m["useful_hit"] == pytest.approx(5 / 6)
    assert m["miss"] == pytest.approx(2 / 8)
    assert m["work_server_p95_ms"] == 1000 and m["work_p95_ms"] == 1050
    assert m["home_server_p95_ms"] == 103
    assert m["tokens_per_recall_mean"] == pytest.approx(
        (sum(100 + j for j in range(4)) * 10 + 90 * 6) / 46
    )
    assert m["blocking_incidents"] == 0


def test_duplicates_and_incidents_feed_the_gate(stores):
    events = p6_use.load_events(stores)
    scans = [
        {
            "date": "2026-11-05",
            "store": "personal",
            "theta": 1.0,
            "n_current": 9,
            "pairs": [["m1", "m2", 3.0], ["m2", "m1", 2.5]],
        },  # one pair, seen twice
        {
            "date": "2026-11-06",
            "store": "personal",
            "theta": 1.0,
            "n_current": 9,
            "pairs": [["m3", "m4", 2.0]],
        },
    ]
    inc = [
        {"date": "2026-11-06", "kind": "crash", "store": "", "note": "x"},
        {"date": "2026-11-09", "kind": "wrong-store", "store": "work", "note": "y"},
    ]
    m = p6_use.compute_e4(events, {}, scans, inc).metrics
    new = m["new_memories"]
    assert (
        new == 8
        and m["duplicate_pairs"] == 2
        and m["duplicates_per_100"] == pytest.approx(2 / new * 100)
    )
    assert m["blocking_incidents"] == 1 and m["crashes"] == 1
    assert math.isnan(m["useful_hit"]) and math.isnan(m["miss"])  # nothing reviewed


def test_window_uses_last_ten_working_days(stores):
    events = p6_use.load_events(stores)
    days = sorted({p6_use.day_of(e.at) for e in events})
    assert p6_use.window_start(days) == days[-10] and p6_use.window_start(days[:9]) is None
    m = p6_use.compute_e4(events, {}, since="2026-11-06").metrics
    assert m["working_days"] == 6


def passing_metrics() -> dict[str, float]:
    return {
        "working_days": 10,
        "work_days": 3,
        "recalls_per_day_median": 3,
        "useful_hit": 0.70,
        "miss": 0.10,
        "unresolved_conflicts": 0.10,
        "duplicates_per_100": 5,
        "blocking_incidents": 0,
        "work_p95_ms": 2000,
    }


def verdicts(metrics: dict[str, float]) -> dict[str, bool]:
    res = SuiteResult("B-E4", metrics=metrics)
    return {s.metric: gates.evaluate(s, res)[0] for s in gates.load_gates()["P6"] if s.id == "B-E4"}


def test_gate_thresholds_are_the_spec_values_at_the_boundary():
    assert all(verdicts(passing_metrics()).values())
    bad = {
        "working_days": 9,
        "work_days": 2,
        "recalls_per_day_median": 2.5,
        "useful_hit": 0.69,
        "miss": 0.11,
        "unresolved_conflicts": 0.11,
        "duplicates_per_100": 5.01,
        "blocking_incidents": 1,
        "work_p95_ms": 2001,
    }
    for k, v in bad.items():
        got = verdicts({**passing_metrics(), k: v})
        assert not got[k] and sum(not x for x in got.values()) == 1, k
    assert not verdicts({**passing_metrics(), "useful_hit": math.nan})[
        "useful_hit"
    ]  # unreviewed != pass


def test_duplicate_pairs_respects_subject_links_and_theta():
    c = make_db(":memory:")
    for i, body in [
        (1, "likes tea"),
        (2, "likes green tea"),
        (3, "likes coffee"),
        (4, "other subject"),
        (5, "likes tea!"),
    ]:
        c.execute("INSERT INTO nodes(id, kind, body) VALUES (?, 'memory', ?)", (i, body))
    c.execute(
        "INSERT INTO nodes(id, kind, status, body, valid_to) VALUES (6, 'memory', 'active', 'old tea', '2026-01-01')"
    )
    for m in (1, 2, 3, 5, 6):
        c.execute("INSERT INTO edges(src, dst, rel) VALUES (?, 100, 'about')", (m,))
    c.execute("INSERT INTO edges(src, dst, rel) VALUES (4, 200, 'about')")
    c.execute(
        "INSERT INTO edges(src, dst, rel) VALUES (5, 1, 'duplicate_of')"
    )  # 1~5 already linked
    logits = {
        frozenset(p): z
        for p, z in [
            (("likes tea", "likes green tea"), 4.0),
            (("likes tea", "likes coffee"), -2.0),
            (("likes green tea", "likes coffee"), 0.5),
            (("likes green tea", "likes tea!"), 3.0),
            (("likes coffee", "likes tea!"), -1.0),
            (("likes tea", "likes tea!"), 9.0),
        ]
    }
    calls = []

    def score(q, docs):
        calls.append((q, tuple(docs)))
        return [logits.get(frozenset((q, d)), -9.0) for d in docs]

    assert p6_use.duplicate_pairs(c, score, 1.0) == [("m1", "m2", 4.0), ("m2", "m5", 3.0)]
    assert p6_use.duplicate_pairs(c, score, 5.0) == []
    assert not any("old tea" in (q, *d) for q, d in calls)  # closed interval: not current


def test_pgv2_sample_sheet_freeze_load_roundtrip(stores, tmp_path):
    events = p6_use.load_events(stores)
    revs = reviews_for(stores, {"personal:1": ("should-have-found", "m5")})
    sample = pgv2.sample_events(events, revs, 12)
    assert len(sample) == 12 and sample[0].ref == "personal:1"  # reviewed first
    assert sample == pgv2.sample_events(list(reversed(events)), revs, 12)
    conns = {n: sqlite3.connect(stores / f"{n}.db") for n in ("personal", "work")}
    for c in conns.values():
        c.execute(
            "INSERT INTO nodes(id, kind, type, date, name, body) VALUES (1,'memory','fact','2026-11-01','tea','likes tea')"
        )
        c.execute(
            "INSERT INTO nodes(id, kind, source_ref, origin, name, body) VALUES (50,'note','p/a.md','import','a','')"
        )
        c.execute(
            "INSERT INTO chunks(id, node_id, chunk_key, text) VALUES (7, 50, 'intro', 'imported text')"
        )
    pooled = lambda q: {"ours": ["m1", "c7"], "bm25": ["m1"]}
    qs, lines = pgv2.make_sheet(sample, revs, conns, pool=pooled, depth=2)
    out = tmp_path / "sheet"
    pgv2.write_sheet(out, qs, lines)
    assert any(ln["id"] == "p/a.md::intro" for ln in lines)  # c7 -> gold id
    assert any(ln["id"] == "m5" and ln["source"] == "found" for ln in lines)
    with pytest.raises(ValueError, match="rel ''"):
        pgv2.freeze(out, tmp_path / "v2", built_at="2026-11-14T00:00:00Z", window={})
    rows = (out / "labels.tsv").read_text().splitlines()
    filled = [rows[0]]
    for i, ln in enumerate(rows[1:]):
        cols = ln.split("\t")
        cols[2] = "2" if cols[1] == "m1" and i % 2 == 0 else "0"
        filled.append("\t".join(cols))
    (out / "labels.tsv").write_text("\n".join(filled) + "\n")
    hashes = pgv2.freeze(
        out, tmp_path / "v2", built_at="2026-11-14T00:00:00Z", window={"first": "a", "last": "b"}
    )
    assert set(hashes) == {"questions.yaml", "qrels.tsv"}
    with pytest.raises(pg.Frozen):
        pgv2.freeze(out, tmp_path / "v2", built_at="x", window={})
    qs2 = pgv2.load(tmp_path / "v2")
    assert len(qs2) == 12 and {q.category for q in qs2} <= {pgv2.ANSWERABLE, pgv2.NO_ANSWER}
    assert all(q.answerable == bool(q.gold) for q in qs2)
    (tmp_path / "v2" / "qrels.tsv").write_text("tampered")
    with pytest.raises(ValueError, match="sha256"):
        pgv2.load(tmp_path / "v2")


def test_pgv2_scores_through_the_b_r3_code(tmp_path):
    from bench import pg_eval
    from bench.lme_eval import Retrieved

    qs = [
        pg.Question("a", pgv2.ANSWERABLE, "q1", gold={"m1": 2}, store="personal"),
        pg.Question("b", pgv2.NO_ANSWER, "q2", store="work"),
    ]

    class Be:
        def retrieve(self, q):
            hit = ["m1"] if q.qid == "a" else []
            return Retrieved(hit, hit, hit, "text", 5.0, [0.9] * len(hit), {})

    sc = pg_eval.score_pg(pg_eval.run_pg(Be(), qs), b=200)
    assert (
        sc.metrics["coverage_any"] == 1.0
        and sc.metrics["far"] == 0.0
        and sc.metrics["abstention_recall"] == 1.0
    )


def test_cli_e4_report_end_to_end(stores, tmp_path):
    r = CliRunner()
    rv = tmp_path / "reviews.jsonl"
    for e in p6_use.load_events(stores):
        if e.tool == "recall" and e.ids and e.store == "personal":
            p6_use.append_review(rv, p6_use.Review(e.ref, "useful"))
    res = r.invoke(
        app,
        [
            "e4-report",
            "--stores",
            str(stores),
            "--reviews",
            str(rv),
            "--scans",
            str(tmp_path / "none.jsonl"),
            "--incidents",
            str(tmp_path / "none2.jsonl"),
            "--out",
            str(tmp_path / "rep.md"),
        ],
    )
    assert "PASS  working_days=10" in res.output and "PASS  useful_hit=1" in res.output
    assert "FAIL  unresolved_conflicts" in res.output  # 1 of 4 left: 0.25 > 0.10
    assert res.exit_code == 1
    assert json.loads((tmp_path / "rep.json").read_text())["metrics"]["working_days"] == 10
    inc = r.invoke(
        app, ["incident", "data-loss", "lost a memory", "--incidents", str(tmp_path / "i.jsonl")]
    )
    assert (
        inc.exit_code == 0
        and r.invoke(
            app, ["incident", "oops", "x", "--incidents", str(tmp_path / "i.jsonl")]
        ).exit_code
        != 0
    )


def test_cli_daily_review_marks_and_skips_done(stores, tmp_path):
    rv = tmp_path / "rv.jsonl"
    args = [
        "daily-review",
        "--stores",
        str(stores),
        "--reviews",
        str(rv),
        "--day",
        "2026-11-02",
        "--n",
        "2",
    ]
    res = CliRunner().invoke(app, args, input="u\ns\nm44\n")
    assert res.exit_code == 0, res.output
    got = list(p6_use.load_reviews(rv).values())
    assert sorted(r.mark for r in got) == ["should-have-found", "useful"]
    assert [r.found for r in got if r.mark == "should-have-found"] == ["m44"]
    again = CliRunner().invoke(app, args, input="")
    assert "0 left unmarked" in again.output


def test_weekly_runs_steps_reports_unset_vars_and_regressions(tmp_path):
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "vars: {x: ok, empty: ''}\nsteps:\n"
        "  - {name: good, argv: [python, -c, 'import sys; print(sys.argv[1])', '{x}']}\n"
        "  - {name: needs, argv: [python, -c, 'pass', '{empty}']}\n"
        "  - {name: bad, argv: [python, -c, 'raise SystemExit(3)']}\n"
    )
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "weekly-20260101T000000Z.json").write_text(
        json.dumps(
            {"outcomes": [{"step": "bad", "status": "PASS"}, {"step": "good", "status": "PASS"}]}
        )
    )
    r = CliRunner().invoke(app, ["weekly", "--plan", str(plan), "--results-dir", str(res_dir)])
    assert r.exit_code == 1
    assert "PASS good" in r.output and "UNAVAILABLE needs" in r.output and "FAIL bad" in r.output
    assert (
        "REGRESSION since the last weekly run: bad" in r.output
        and "1/3 weekly steps pass" in r.output
    )
    out = max(res_dir.glob("weekly-2*.json"))
    assert {o["step"]: o["status"] for o in json.loads(out.read_text())["outcomes"]} == {
        "good": "PASS",
        "needs": "UNAVAILABLE",
        "bad": "FAIL",
    }
