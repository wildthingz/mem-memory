"""The read-time conflict guard (ADR 0018, MEM2-159; architecture-v2 §3.2 step 7b).

Stub models: HashEmbedder + OverlapReranker (logit 6·overlap − 3), the stub conflict model
(score = rerank logit). Two memories of one slot with a missed `supersedes` share almost every word
(score ≈ 2.1-3.0); a co-valid pair about the same person with different relations shares few
(score < 0). θ_read = 1.5 separates them here."""

import json
import math

import pytest

from mem import config
from mem.common import MemError
from mem.confmodel import ConfModel
from mem.gate import Gate, read_gate, write_gate
from mem.readguard import ReadGuard, hidden_of, score_pairs
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker, stub_conf_model

NOW = "2026-10-22T09:00:00Z"
CM = stub_conf_model()
CM_FP = ConfModel.from_dict(CM).fingerprint()


def guard_dict(theta: float = 1.5, cm: dict = CM) -> dict:
    return ReadGuard.build(theta, ConfModel.from_dict(cm).fingerprint()).to_dict()


def p3_gate(theta_read: float | None = 1.5, fingerprint_hash: str = "0" * 64) -> Gate:
    return Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5, conf_model=CM,
                fingerprint_hash=fingerprint_hash,
                read_guard=None if theta_read is None else guard_dict(theta_read))  # fmt: skip


@pytest.fixture
def gstore(tmp_path, clock):
    opened = []

    def _make(gate: Gate | None = None, read: dict | None = None, name="personal"):
        gate = gate or p3_gate()
        read = {"conflict_guard": True} if read is None else read  # the default is off (ADR 0018)
        s = Store.open(tmp_path / f"{name}.db", HashEmbedder(), scope=name, now=clock,
                       reranker=OverlapReranker(), read=read)  # fmt: skip
        write_gate(s.conn, gate)
        s.gate = read_gate(s.conn)
        opened.append(s)
        return s

    yield _make
    for s in opened:
        s.close()


def _rem(store, text, about, as_of, now=None, clock=None):
    if clock is not None and now is not None:
        clock.set(now)
    return store.remember(text=text, type="fact", about=[about], as_of=as_of).structured["id"]


def _ids(res):
    return [r["id"] for r in res.structured["rows"]]


def missed_update(store, clock):
    """Ana's weekly session: Monday (June), then Thursday (September) saved WITHOUT the
    `supersedes` link: both stay open (the write path missed the update)."""
    clock.set("2026-06-03T09:00:00Z")
    old = _rem(store, "Ana weekly planning session is on Monday.", "new:person:Ana", "2026-06-03")
    clock.set("2026-09-01T09:00:00Z")
    new = _rem(store, "Ana weekly planning session is on Thursday.", "Ana", "2026-09-01")
    clock.set(NOW)
    return old, new


# ---- the ReadGuard record ----------------------------------------------------------------------


def test_read_guard_record_round_trips_and_refuses_without_theta_or_with_a_bad_fingerprint():
    g = ReadGuard.build(1.5, CM_FP)
    assert ReadGuard.from_dict(g.to_dict()) == g
    with pytest.raises(ValueError, match="theta_read is required"):
        ReadGuard.from_dict({"conf_model_fingerprint": CM_FP, "fingerprint": "x"})
    with pytest.raises(ValueError, match="fingerprint does not match"):
        ReadGuard.from_dict({**g.to_dict(), "theta_read": 1.4})
    with pytest.raises(ValueError, match="finite"):
        ReadGuard.build(math.nan, CM_FP)


def test_gate_refuses_a_read_guard_without_conf_model_or_against_another_scorer():
    with pytest.raises(ValueError, match="no conf_model"):
        Gate(1.0, 0.0, 0.5, 0.5, 0.0, read_guard=guard_dict())
    other = ConfModel.build(["rerank", "cosine"], [1.0, 1.0], 0.0, 5, 100.0).to_dict()
    with pytest.raises(ValueError, match="another conf_model"):
        Gate(
            1.0,
            0.0,
            0.5,
            0.5,
            0.0,
            theta_conf=0.5,
            conf_model=CM,
            read_guard=guard_dict(1.5, other),
        )
    with pytest.raises(ValueError, match="theta_read is required"):
        Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5, conf_model=CM,
             read_guard={"fingerprint": "x", "conf_model_fingerprint": CM_FP})  # fmt: skip
    g = p3_gate()
    assert g.guard is not None and g.guard.theta_read == 1.5 and not g.needs_read_guard()
    assert p3_gate(None).needs_read_guard()
    assert not Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=-math.inf).needs_read_guard()  # dev export
    assert not Gate(1.0, 0.0, 0.5, 0.5, 0.0).needs_read_guard()  # a P2 gate: no scorer in force


def test_gate_json_round_trips_the_read_guard(tmp_path, clock):
    s = Store.open(tmp_path / "g.db", HashEmbedder(), now=clock)
    write_gate(s.conn, p3_gate())
    assert read_gate(s.conn) == p3_gate()
    s.close()


# ---- the guard in recall ----------------------------------------------------------------------


def test_hides_the_older_of_a_missed_update_pair_at_now_and_at_as_of(gstore, clock):
    s = gstore()
    old, new = missed_update(s, clock)
    r = s.recall(query="Ana weekly planning session")
    assert _ids(r) == [new]
    assert r.structured["hidden"] == [{"id": old, "possibly_superseded_by": new,
                                       "score": pytest.approx(2.14, abs=0.01)}]  # fmt: skip
    assert f"hidden {old} · possibly superseded by {new} · get {old} to see it" in r.text
    assert r.text.startswith("1 results (of 2 checked)")
    # as_of after both valid_from: the same pair, the same decision
    r = s.recall(query="Ana weekly planning session", as_of="2026-10-01")
    assert _ids(r) == [new] and r.structured["hidden"][0]["id"] == old
    # as_of before the newer one's valid_from: only the older is eligible, nothing to hide
    r = s.recall(query="Ana weekly planning session", as_of="2026-07-01")
    assert _ids(r) == [old] and "hidden" not in r.structured
    # the store is untouched: both open, no edge, no status change
    rows = s.conn.execute("SELECT valid_to, status FROM nodes WHERE kind = 'memory'").fetchall()
    assert rows == [(None, "active"), (None, "active")]
    assert s.conn.execute("SELECT count(*) FROM edges WHERE rel = 'supersedes'").fetchone()[0] == 0


def test_never_hides_a_co_valid_pair_below_theta_read(gstore, clock):
    s = gstore()
    clock.set("2026-06-03T09:00:00Z")
    a = _rem(s, "Ana weekly planning session is on Monday.", "new:person:Ana", "2026-06-03")
    clock.set("2026-07-01T09:00:00Z")
    b = _rem(s, "Ana is fluent in Portuguese and Dutch.", "Ana", "2026-07-01")
    clock.set(NOW)
    pairs = score_pairs(s, [int(a[1:]), int(b[1:])])
    assert len(pairs) == 1 and pairs[0].later == int(b[1:]) and pairs[0].score < 1.5
    r = s.recall(query="Ana weekly planning session Portuguese Dutch")
    assert set(_ids(r)) == {a, b} and "hidden" not in r.structured


def test_only_the_later_in_chain_survives_and_ties_go_by_recorded_at(gstore, clock):
    s = gstore()
    day = "2026-09-01"
    clock.set("2026-09-01T08:00:00Z")
    m1 = _rem(s, "Ana weekly planning session is on Monday.", "new:person:Ana", day)
    clock.set("2026-09-01T09:00:00Z")
    m2 = _rem(s, "Ana weekly planning session is on Tuesday.", "Ana", day)
    clock.set(NOW)
    r = s.recall(query="Ana weekly planning session")
    assert (
        _ids(r) == [m2] and r.structured["hidden"][0]["id"] == m1
    )  # same valid_from: later record


def test_never_adds_a_row_and_hidden_rows_stay_in_get_and_history(gstore, clock):
    s = gstore()
    old, new = missed_update(s, clock)
    clock.set("2026-09-02T09:00:00Z")
    third = _rem(s, "Ana weekly planning notes are kept in the shared folder.", "Ana", "2026-09-02")
    clock.set(NOW)
    # limit 2: the set before the guard is the two slot members; after hiding the older one the
    # third memory is NOT pulled in (A2: the guard removes, never adds)
    full = s.recall(query="Ana weekly planning session", limit=8)
    assert _ids(full)[:2] == [new, third] and full.structured["hidden"][0]["id"] == old
    r = s.recall(query="Ana weekly planning session", limit=2)
    assert _ids(r) == [new] and r.structured["hidden"][0]["id"] == old
    # get and include_history still return the hidden row
    g = s.get([old])
    assert g.structured["items"][0]["found"] and "Monday" in g.text
    h = s.recall(query="Ana weekly planning session", include_history=True)
    assert set(_ids(h)) >= {old, new} and "hidden" not in h.structured


def test_hidden_pairs_are_logged_and_the_review_offers_a_supersede(gstore, clock):
    from mem import review

    s = gstore()
    old, new = missed_update(s, clock)
    s.recall(query="Ana weekly planning session")
    s.recall(query="Ana weekly planning session")
    (res,) = s.conn.execute(
        "SELECT result FROM events WHERE tool = 'recall' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    logged = json.loads(res)
    assert logged["ids"] == [new] and logged["hidden"][0]["id"] == old
    assert logged["hidden"][0]["by"] == new
    hid = review.read_guard_hidden(s.conn)
    assert [(h.earlier, h.later, h.times) for h in hid] == [(old, new, 2)]
    assert review.tuning_counts(s.conn)["read_guard_hidden"] == 1
    # a real supersede resolves it, and the pair leaves the review; recall no longer needs the guard
    s.link(src=new, rel="supersedes", dst=old)
    assert review.read_guard_hidden(s.conn) == []
    r = s.recall(query="Ana weekly planning session")
    assert _ids(r) == [new] and "hidden" not in r.structured


def test_guard_runs_only_on_memory_rows_and_records_its_stage(gstore, clock):
    s = gstore()
    missed_update(s, clock)
    from mem import read

    t: dict[str, float] = {}
    read.recall(s, query="Ana weekly planning session", timings=t)
    assert "guard" in t and t["guard"] >= 0.0 and t["total"] >= t["guard"]


# ---- the switch and the refusals ----------------------------------------------------------------


def test_config_switch_off_disables_the_guard(gstore, clock):
    s = gstore(read={"conflict_guard": False})
    old, new = missed_update(s, clock)
    r = s.recall(query="Ana weekly planning session")
    assert set(_ids(r)) == {old, new} and "hidden" not in r.structured
    # and with the switch off a gate without read_guard is not refused
    s2 = gstore(gate=p3_gate(None), read={"conflict_guard": False}, name="work")
    missed_update(s2, clock)
    assert len(_ids(s2.recall(query="Ana weekly planning session"))) == 2


def test_recall_refuses_a_p3_gate_without_theta_read_when_the_guard_is_on(gstore, clock):
    s = gstore(gate=p3_gate(None))
    missed_update(s, clock)
    with pytest.raises(MemError, match="no read_guard"):
        s.recall(query="Ana weekly planning session")


def test_serve_preflight_refuses_a_p3_gate_without_theta_read(tmp_path):
    from mem.serve import preflight
    from tests.helpers import STUB

    path = tmp_path / "personal.db"
    s = Store.open(path, HashEmbedder(), fp=STUB.fp_embed())
    write_gate(s.conn, p3_gate(None, fingerprint_hash=STUB.fp_all()))
    s.close()
    on = STUB.replace("read", conflict_guard=True)
    s = Store.open(path, HashEmbedder(), fp=on.fp_embed())
    write_gate(s.conn, p3_gate(None, fingerprint_hash=on.fp_all()))
    s.close()
    (problem,) = preflight({"personal": path}, on)
    assert "no read_guard" in problem and "bench syn-fit-read-guard" in problem
    s = Store.open(path, HashEmbedder(), fp=on.fp_embed())
    write_gate(s.conn, p3_gate(1.5, fingerprint_hash=on.fp_all()))
    s.close()
    assert preflight({"personal": path}, on) == []
    # off (the default) the same gate shape is not refused, under that config's own fp_all
    off = STUB
    assert off.conflict_guard is False
    path2 = tmp_path / "work.db"
    s = Store.open(path2, HashEmbedder(), scope="work", fp=off.fp_embed())
    write_gate(s.conn, p3_gate(None, fingerprint_hash=off.fp_all()))
    s.close()
    assert preflight({"work": path2}, off) == []
    (mismatch,) = preflight({"work": path2}, on)  # the switch is in fp_all
    assert "fingerprint mismatch" in mismatch


def test_bench_gate_p3_refuses_a_gate_file_without_theta_read(tmp_path):
    from dataclasses import asdict, replace

    from typer.testing import CliRunner

    from bench.adapters.lme_ours import load_config
    from bench.cli import app
    from tests.bench_t0.test_p3_real import stub_gate

    cfg = tmp_path / "on.yaml"
    cfg.write_text("read:\n  conflict_guard: true\n")

    def run(config_path):
        g = tmp_path / "gate.json"
        gate = replace(stub_gate(load_config(config_path, "hash")), read_guard=None)
        g.write_text(json.dumps(asdict(gate)))
        args = ["gate", "P3", "--embedder", "hash", "--gate", str(g), "--syn-size", "S",
                "--syn-seeds", "1", "--only", "B-W6"]  # fmt: skip
        return CliRunner().invoke(
            app, [*args, *(["--config", str(config_path)] if config_path else [])]
        )

    assert (
        "read_guard" not in run(None).output
    )  # the guard is off by default: no guard section needed
    r = run(cfg)
    assert r.exit_code != 0 and "read_guard" in r.output and "syn-fit-read-guard" in r.output


def test_the_switch_is_in_the_fingerprint():
    off, on = config.stub_config(), config.stub_config().replace("read", conflict_guard=True)
    assert on.fp_all() != off.fp_all()


# ---- leakage on a fixture with a missed supersede ----------------------------------------------


def _leakage(store, queries):
    """B-R5's leakage over `queries` = [(query, gold id, stale ids)]: a current read that still
    holds a stale slot memory leaks."""
    leaks = 0
    for q, _gold, stale in queries:
        rows = set(_ids(store.recall(query=q)))
        leaks += bool(rows & set(stale))
    return leaks / len(queries)


def test_leakage_on_a_fixture_with_missed_supersedes_goes_to_zero(gstore, clock):
    """Three slots, two with a missed update (both members open), one correctly linked."""
    s_off = gstore(read={"conflict_guard": False})
    s_on = gstore(read={"conflict_guard": True}, name="work")
    fixture = []
    for s in (s_off, s_on):
        fixture = []
        old, new = missed_update(s, clock)
        fixture.append(("Ana weekly planning session", new, [old]))
        clock.set("2026-05-01T09:00:00Z")
        c1 = _rem(s, "Bram commutes to work by ferry.", "new:person:Bram", "2026-05-01")
        clock.set("2026-08-01T09:00:00Z")
        c2 = _rem(s, "Bram commutes to work by tram.", "Bram", "2026-08-01")
        fixture.append(("Bram commutes to work", c2, [c1]))
        clock.set("2026-05-02T09:00:00Z")
        d1 = _rem(s, "Clara drives a Volvo.", "new:person:Clara", "2026-05-02")
        clock.set("2026-08-02T09:00:00Z")
        d2 = _rem(s, "Clara drives a Skoda.", "Clara", "2026-08-02")
        s.link(src=d2, rel="supersedes", dst=d1)  # the write path caught this one
        fixture.append(("Clara drives a", d2, [d1]))
        clock.set(NOW)
    assert _leakage(s_off, fixture) == pytest.approx(2 / 3)
    assert _leakage(s_on, fixture) == 0.0
    # current@1 is the gold current in every case, hidden rows are named
    for q, gold, stale in fixture:
        r = s_on.recall(query=q)
        assert _ids(r)[0] == gold
        if q.startswith("Clara"):
            assert "hidden" not in r.structured
        else:
            assert [h["id"] for h in r.structured["hidden"]] == stale


def test_hidden_of_names_the_best_scoring_later_member_once():
    from mem.readguard import Pair

    pairs = [Pair(3, 1, 2.0, {}), Pair(2, 1, 2.5, {}), Pair(3, 2, 1.0, {})]
    assert hidden_of(pairs, 1.5) == {1: (2, 2.5)}
    assert hidden_of(pairs, 0.5) == {1: (2, 2.5), 2: (3, 1.0)}
    assert hidden_of(pairs, 3.0) == {}
