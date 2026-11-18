"""P6 daily-use tools (architecture-v2 §6): core index, daily review, unresolved conflicts."""

import json

from typer.testing import CliRunner

from mem import review
from mem.cli import app
from mem.common import estimate_tokens


def _ev(store, at, tool, args, result, session="s1"):
    store.conn.execute(
        "INSERT INTO events(at, session, host, tool, args, result, latency_ms)"
        " VALUES (?, ?, 'home', ?, ?, ?, 5)",
        (at, session, tool, json.dumps(args), json.dumps(result)),
    )


def _mem(store, text, **kw):
    return store.remember(text=text, type="fact", about=["new:topic:Core"], **kw).structured["id"]


# ---- core index ------------------------------------------------------------------------------
def test_core_index_within_budget(make_store, clock):
    s = make_store()
    pinned = _mem(s, "pinned rule alpha", pin=True)
    for i in range(40):
        clock.set(f"2026-10-22T10:{i:02d}:00Z")
        _mem(s, f"recent fact number {i} with some padding words")
    text = review.core_index(s.conn, budget=200)
    assert 0 < estimate_tokens(text) <= 200
    lines = text.splitlines()
    assert lines[1].startswith(pinned + " ")  # pinned first, after the header
    assert "recent fact number 39" in lines[2]  # then newest first
    assert "recent fact number 0 " not in text  # budget cut the oldest


def test_core_index_skips_retracted_and_superseded(make_store):
    s = make_store()
    gone = _mem(s, "retracted pinned", pin=True)
    s.forget(id=gone, reason="wrong")
    kept = _mem(s, "kept fact")
    text = review.core_index(s.conn, budget=800)
    assert gone not in text.replace(kept, "") and "retracted pinned" not in text
    assert "kept fact" in text


def test_core_index_empty_store_is_header_only(make_store):
    assert review.core_index(make_store().conn, budget=800).count("\n") == 0


# ---- daily review ----------------------------------------------------------------------------
def _seed_recalls(s, n=25):
    for i in range(1, n + 1):
        _ev(s, f"2026-10-22T09:{i:02d}:00Z", "recall", {"query": f"q{i}"}, {"ids": ["m1"]})
    _ev(s, "2026-10-23T09:00:00Z", "recall", {"query": "other day"}, {"ids": []})
    _ev(s, "2026-10-22T09:30:00Z", "remember", {"text": "x"}, {"status": "saved"})


def test_daily_review_sample_deterministic(make_store):
    s = make_store()
    _seed_recalls(s)
    a = review.daily_sample(s.conn, "2026-10-22")
    b = review.daily_sample(s.conn, "2026-10-22")
    assert [e.id for e in a] == [e.id for e in b]
    assert len(a) == 10
    ids = [
        r[0]
        for r in s.conn.execute(
            "SELECT id FROM events WHERE tool='recall' AND at LIKE '2026-10-22%'"
        )
    ]
    store = review.store_name(s.conn)
    want = sorted(ids, key=lambda i: review.sample_key(review.event_ref(store, i)))[:10]
    assert sorted(e.id for e in a) == sorted(want)
    assert all(e.query != "other day" for e in a)


def test_daily_review_sample_is_stable_when_later_events_arrive(make_store):
    s = make_store()
    _seed_recalls(s, 12)
    first = {e.id for e in review.daily_sample(s.conn, "2026-10-22", n=3)}
    _ev(s, "2026-10-22T20:00:00Z", "remember", {"text": "y"}, {"status": "saved"})
    assert {e.id for e in review.daily_sample(s.conn, "2026-10-22", n=3)} == first


def test_daily_review_context_is_preceding_queries_of_same_session(make_store):
    s = make_store()
    _ev(s, "2026-10-22T09:00:00Z", "recall", {"query": "earlier"}, {"ids": []})
    _ev(s, "2026-10-22T09:01:00Z", "remember", {"text": "saved text"}, {"status": "saved"})
    _ev(s, "2026-10-22T09:02:00Z", "recall", {"query": "target"}, {"ids": []})
    _ev(s, "2026-10-22T09:00:30Z", "recall", {"query": "foreign"}, {"ids": []}, session="s2")
    (e,) = [e for e in review.daily_sample(s.conn, "2026-10-22") if e.query == "target"]
    assert e.context == ["earlier", "saved text"]


# ---- unresolved conflicts --------------------------------------------------------------------
def _saved(s, at, mid, conflicts, session="s1"):
    _ev(
        s,
        at,
        "remember",
        {"text": "t"},
        {"status": "saved", "id": mid, "possible_conflicts": [{"id": c} for c in conflicts]},
        session,
    )


def test_unresolved_conflict_query(make_store):
    s = make_store()
    _saved(s, "2026-10-22T09:00:00Z", "m10", ["m3"])  # resolved by link
    _ev(s, "2026-10-22T09:01:00Z", "link", {"src": "m10", "rel": "supersedes", "dst": "m3"}, {})
    _saved(s, "2026-10-22T09:02:00Z", "m11", ["m4", "m5"])  # unresolved
    _saved(s, "2026-10-22T09:03:00Z", "m12", ["m6"])  # resolved by forget
    _ev(s, "2026-10-22T09:04:00Z", "forget", {"id": "m12"}, {})
    _saved(s, "2026-10-22T09:05:00Z", "m13", ["m7"])  # link was in another session
    _ev(
        s,
        "2026-10-22T09:06:00Z",
        "link",
        {"src": "m13", "rel": "relates_to", "dst": "m7"},
        {},
        session="s2",
    )
    _saved(s, "2026-10-22T09:07:00Z", "m14", [])  # no conflicts, never counted
    _saved(s, "2026-10-22T09:08:00Z", "m15", ["m8"])  # link to it came before the remember
    got = review.unresolved_conflicts(s.conn)
    assert [(u.new_id, u.candidates) for u in got] == [
        ("m11", ["m4", "m5"]),
        ("m13", ["m7"]),
        ("m15", ["m8"]),
    ]


def test_unresolved_conflict_dst_side_link_resolves(make_store):
    s = make_store()
    _saved(s, "2026-10-22T09:00:00Z", "m20", ["m1"])
    _ev(s, "2026-10-22T09:01:00Z", "link", {"src": "m1", "rel": "supersedes", "dst": "m20"}, {})
    assert review.unresolved_conflicts(s.conn) == []


# ---- CLI -------------------------------------------------------------------------------------
def test_cli_core_and_conflicts(tmp_path, make_store):
    s = make_store("personal")
    _mem(s, "cli visible fact", pin=True)
    _saved(s, "2026-10-22T09:02:00Z", "m99", ["m1"])
    s.conn.commit()
    path = str(tmp_path / "personal.db")
    r = CliRunner().invoke(app, ["core", "--store", path, "--budget", "800"])
    assert r.exit_code == 0 and "cli visible fact" in r.output
    r = CliRunner().invoke(app, ["conflicts", "--store", path])
    assert r.exit_code == 0 and "m99" in r.output


def test_cli_review_prints_sample_and_tuning(tmp_path, make_store):
    s = make_store("personal")
    _ev(s, "2026-10-22T09:00:00Z", "recall", {"query": "where is x"}, {"ids": []})
    s.conn.commit()
    r = CliRunner().invoke(
        app, ["review", "--store", str(tmp_path / "personal.db"), "--day", "2026-10-22"]
    )
    assert r.exit_code == 0 and "q: where is x" in r.output and "(empty)" in r.output
    assert "'empty_recalls': 1" in r.output
