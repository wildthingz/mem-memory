"""The P6 gate suites (benchmarks-v2 §4 P6 row): B-E4 (daily use, over the live stores) and the
PG-v2 record. Inputs are the live stores and the human-kept files of `bench.p6_cli` (reviews, nightly
scans, incidents), passed in `ctx.options["p6"]` (`P6Setup`, defaults: the same as `bench e4-report`).
No live events, or no frozen PG-v2, is UNAVAILABLE (not run), never a pass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bench import p6_use, pgv2
from bench.suites.base import BackendUnavailable, Context, SuiteResult

OPT = "p6"
DATA = Path("data/p6")
GOLD_V2 = Path(__file__).resolve().parent.parent / "gold" / "personal" / "v2"


@dataclass
class P6Setup:
    stores: Path = field(default_factory=lambda: Path.home() / ".local" / "share" / "mem")
    reviews: Path = DATA / "reviews.jsonl"
    scans: Path = DATA / "dupscan.jsonl"
    incidents: Path = DATA / "incidents.jsonl"
    since: str | None = None
    until: str | None = None
    tz: str = "UTC"
    work_hosts: list[str] = field(default_factory=lambda: ["work"])
    work_overhead_ms: float | None = None
    gold_dir: Path = GOLD_V2


def setup_of(ctx: Context) -> P6Setup:
    return ctx.options.get(OPT) or P6Setup()


def e4_result(rep: p6_use.E4Report) -> SuiteResult:
    res = SuiteResult("B-E4", tier="t1", mode="live")
    res.metrics |= rep.metrics
    res.notes += rep.notes
    return res


def e4_report(setup: P6Setup) -> tuple[p6_use.E4Report, str | None]:
    """(report, first day of its window): the last 10 working days unless `since` is given."""
    events = p6_use.load_events(setup.stores)
    if not events:
        raise BackendUnavailable(f"B-E4: no live events in {setup.stores}")
    tz, until = setup.tz, setup.until
    days = sorted({p6_use.day_of(e.at, tz) for e in events
                   if not until or p6_use.day_of(e.at, tz) <= until})  # fmt: skip
    start = setup.since or p6_use.window_start(days) or (days[0] if days else None)
    rep = p6_use.compute_e4(
        events, p6_use.load_reviews(setup.reviews), p6_use.load_scans(setup.scans),
        p6_use.load_incidents(setup.incidents), since=start, until=until, tz=tz,
        work_hosts=setup.work_hosts, work_overhead_ms=setup.work_overhead_ms)  # fmt: skip
    return rep, start


def run_b_e4(ctx: Context) -> SuiteResult:
    return e4_result(e4_report(setup_of(ctx))[0])


def run_pgv2(ctx: Context) -> SuiteResult:
    """The size of the frozen PG-v2 (the B-R3 numbers on it are `bench pgv2-run`'s record)."""
    gold = setup_of(ctx).gold_dir
    if not (gold / "SNAPSHOT").exists():
        raise BackendUnavailable(f"PG-v2 not built: no {gold / 'SNAPSHOT'} (`bench pgv2-sample`)")
    res = SuiteResult("B-R3-PGv2", tier="t1", mode="recorded")
    questions = pgv2.load(gold)
    res.metrics["n_answerable"] = float(sum(q.answerable for q in questions))
    return res
