"""MEM2-47 review proofs for the P6 step (bosmang/p6-tools + bosmang/p6-bench). Each test is red
on the reviewed tip and states one finding; it passes once the finding is fixed."""

from __future__ import annotations

from bench import gates, p6_use
from bench.suites import suite_for
from mem import review
from tests.test_p6_use import ev, make_db


def test_every_p6_gate_row_has_a_suite():
    """F1: `bench gate P6` crashes (`KeyError: 'B-E4'`): gates.yaml gained a P6 block but no suite
    is registered for its ids."""
    for spec in gates.load_gates()["P6"]:
        suite_for("P6", spec.id)  # raises KeyError on the reviewed tip


def test_mem_review_and_bench_daily_review_pick_the_same_sample(tmp_path):
    """F2a: the two evening-review commands sample the same day differently (`mem review` hashes
    the bare event id, `bench daily-review` hashes `store:id`), so "today's 10" depends on which
    command they run. One definition is needed (benchmarks-v2 B-E4: sha1 of the event id)."""
    c = make_db(tmp_path / "personal.db")
    for i in range(1, 26):
        ev(c, f"2026-11-02T09:{i:02d}:00Z", "recall", {"query": f"q{i}"}, {"ids": ["m1"]})
    c.commit()
    mem_ids = sorted(e.id for e in review.daily_sample(c, "2026-11-02"))
    bench_ids = sorted(
        e.id for e in p6_use.review_sample(p6_use.load_events(tmp_path), "2026-11-02")
    )
    assert mem_ids == bench_ids


def test_mem_conflicts_and_bench_e4_agree_on_inline_supersedes(tmp_path):
    """F2b: the gate metric `unresolved_conflicts` has two definitions: `bench e4-report` counts an
    inline `supersedes` argument as a resolution, `mem conflicts` does not."""
    c = make_db(tmp_path / "personal.db")
    ev(
        c,
        "2026-11-04T11:00:00Z",
        "remember",
        {"text": "b", "supersedes": "m2"},
        {"status": "saved", "id": "m11", "possible_conflicts": [{"id": "m2"}]},
        session="c2",
    )
    c.commit()
    mem_left = [u.new_id for u in review.unresolved_conflicts(c)]
    _, bench_left = p6_use.unresolved_conflicts(p6_use.load_events(tmp_path))
    assert (mem_left == []) == (bench_left == []), (mem_left, bench_left)


def test_e4_report_without_events_exits_cleanly(tmp_path):
    """F5 (MEM2-108, a regression of MEM2-107): `bench e4-report` on a stores directory with no
    events used to print the report with its FAIL rows (exit 1); it now dies with an uncaught
    `BackendUnavailable` traceback. The CLI should say so in a line and exit 2 like its siblings."""
    from typer.testing import CliRunner

    from bench.cli import app

    r = CliRunner().invoke(app, ["e4-report", "--stores", str(tmp_path)])
    assert r.exception is None or isinstance(r.exception, SystemExit), repr(r.exception)
    assert r.exit_code in (1, 2)
    assert "no live events" in r.output or "B-E4" in r.output
