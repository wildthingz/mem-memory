"""ADR 0028: a date-only `as_of` on a read means the END of that day in the store's zone.

A memory saved later on the same day is visible; one saved on the next day is not. Timestamped
`as_of` keeps the half-open predicate unchanged. The write side is unchanged: a date-only
`remember(as_of=)` is still local midnight (the start of the day the fact holds from).
"""

import pytest

from mem import calibrate
from mem.common import normalize_as_of, normalize_ts
from mem.gate import Gate
from mem.testing import OverlapReranker

ANA = "new:person:Ana Silva"


def _remember(store, text, about=(ANA,), **kw):
    return store.remember(text=text, type="event", about=list(about), **kw).structured["id"]


def _ids(res):
    return {r["id"] for r in res.structured["rows"]}


@pytest.mark.parametrize(
    ("value", "tz", "want"),
    [
        ("2026-11-01", "UTC", "2026-11-01T23:59:59Z"),
        ("2026-11-01", "Europe/Amsterdam", "2026-11-01T22:59:59Z"),  # CET, UTC+1
        ("2026-06-03", "Europe/Amsterdam", "2026-06-03T21:59:59Z"),  # CEST, UTC+2
        ("2026-11-01T08:00:00Z", "Europe/Amsterdam", "2026-11-01T08:00:00Z"),  # timestamps as-is
        ("2026-11-01T08:00:00", "UTC", "2026-11-01T08:00:00Z"),
    ],
)
def test_normalize_as_of_reads_a_date_only_value_as_the_end_of_that_local_day(value, tz, want):
    assert normalize_as_of(value, tz) == want


def test_write_side_date_only_is_still_local_midnight():
    assert normalize_ts("2026-11-01", "Europe/Amsterdam") == "2026-10-31T23:00:00Z"


def test_a_same_day_later_memory_is_visible_with_a_date_only_as_of(make_store):
    store = make_store(tz="UTC")
    later = _remember(store, "Ana Silva ran the Paris marathon.", as_of="2026-11-01T15:00:00Z")
    assert later in _ids(store.recall(query="Ana Silva Paris marathon", as_of="2026-11-01"))


def test_a_next_day_memory_is_not_visible_with_a_date_only_as_of(make_store):
    store = make_store(tz="UTC")
    nxt = _remember(store, "Ana Silva ran the Rome marathon.", as_of="2026-11-02T00:00:00Z")
    assert nxt not in _ids(store.recall(query="Ana Silva Rome marathon", as_of="2026-11-01"))
    assert nxt in _ids(store.recall(query="Ana Silva Rome marathon", as_of="2026-11-02"))


def test_the_day_is_the_stores_configured_zone(make_store):
    store = make_store(tz="Europe/Amsterdam")  # 2026-11-01 local = [10-31T23:00Z, 11-01T23:00Z)
    late = _remember(store, "Ana Silva ran the Paris marathon.", as_of="2026-11-01T22:30:00Z")
    nxt = _remember(store, "Ana Silva ran the Rome marathon.", as_of="2026-11-01T23:30:00Z")
    assert late in _ids(store.recall(query="Ana Silva Paris marathon", as_of="2026-11-01"))
    assert nxt not in _ids(store.recall(query="Ana Silva Rome marathon", as_of="2026-11-01"))


def test_a_timestamped_as_of_keeps_the_half_open_predicate(make_store):
    store = make_store(tz="UTC")
    later = _remember(store, "Ana Silva ran the Paris marathon.", as_of="2026-11-01T15:00:00Z")
    q = "Ana Silva Paris marathon"
    assert later not in _ids(store.recall(query=q, as_of="2026-11-01T14:59:59Z"))
    assert later in _ids(store.recall(query=q, as_of="2026-11-01T15:00:00Z"))


def test_calibration_scoring_uses_the_same_day_rule(make_store):
    """`calibrate.score_pool` is the read path with the gate off: same eligibility as recall."""
    store = make_store(tz="UTC", reranker=OverlapReranker())
    store.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0)
    _remember(store, "Ana Silva ran the Paris marathon.", as_of="2026-11-01T15:00:00Z")
    _remember(store, "Ana Silva ran the Rome marathon.", as_of="2026-11-02T00:00:00Z")
    pool = calibrate.score_pool(store, "Ana Silva marathon", "2026-11-01")
    texts = {store.conn.execute("SELECT text FROM chunks WHERE id = ?",
                                (int(i.key[1:]),)).fetchone()[0] for i in pool}  # fmt: skip
    assert any("Paris" in t for t in texts) and not any("Rome" in t for t in texts)
