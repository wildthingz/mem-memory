"""MEM2-156 round 4: the store signal behind the conflict scorer's `open` feature.

Under SYN_VERSION 3 an update's gold counterpart is the slot's latest active member by valid time
(the latest in its supersedes chain). In a correctly linked chain that member is the one with
`valid_to` NULL, so `open` is exactly the "latest in its chain" store signal: no SYN knowledge, read
off the chain. These tests pin that in the two cases the owner named (a future-dated update, an earlier
miss), and the dev export carries each candidate's validity so store-time signals can be computed
offline from the records.
"""

from __future__ import annotations

import math

from mem.gate import Gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker


def _dev_store(tmp_path, clock) -> Store:
    """Unfiltered dev run (theta_conf = -inf): every pick comes back with its features."""
    s = Store.open(tmp_path / "s.db", HashEmbedder(), now=clock, reranker=OverlapReranker())
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0, theta_conf=-math.inf)
    return s


def _rem(store, text, about, **kw):
    return store.remember(text=text, type="fact", about=about, **kw).structured


def _by_id(out):
    return {c["id"]: c for c in out["possible_conflicts"]}


def _intervals(store):
    return {f"m{i}": (vf[:10], vt and vt[:10]) for i, vf, vt in store.conn.execute(
        "SELECT id, valid_from, valid_to FROM nodes WHERE kind='memory' ORDER BY id")}  # fmt: skip


def test_after_a_future_dated_update_the_chain_latest_member_is_the_open_one(tmp_path, clock):
    """A (Jan) superseded by a future-dated B (Sep) closes A at Sep. A normal update C (Mar) finds
    A closed yet valid at C's valid_from, and B open. The gold counterpart (latest by valid time,
    SYN_VERSION 3) is B, the open one; a "valid at the new valid_from" signal would name A instead.
    Linking C to B gives the right chain either way: A Jan-Mar, C Mar-Sep, B open."""
    s = _dev_store(tmp_path, clock)
    a = _rem(s, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")["id"]
    b = _rem(s, "Ana desk is on floor 7.", ["Ana"], as_of="2026-09-01", supersedes=a)["id"]
    assert _intervals(s) == {a: ("2026-01-01", "2026-09-01"), b: ("2026-09-01", None)}
    out = _rem(s, "Ana desk is on floor 5.", ["Ana"], as_of="2026-03-01")
    rows = _by_id(out)
    assert rows[a]["features"]["open"] == 0.0 and rows[a]["features"]["older"] == 1.0
    assert rows[b]["features"]["open"] == 1.0 and rows[b]["features"]["older"] == 0.0
    assert (rows[a]["valid_from"][:10], rows[a]["valid_to"][:10]) == ("2026-01-01", "2026-09-01")
    assert (rows[b]["valid_from"][:10], rows[b]["valid_to"]) == ("2026-09-01", None)
    c = out["id"]
    s.link(src=c, rel="supersedes", dst=b)
    assert _intervals(s) == {a: ("2026-01-01", "2026-03-01"), c: ("2026-03-01", "2026-09-01"),
                             b: ("2026-09-01", None)}  # fmt: skip
    s.close()


def test_after_a_missed_update_the_stale_member_stays_open_beside_the_counterpart(tmp_path, clock):
    """A (Jan), then B (Feb) written without a link (a missed conflict): both stay open. For C (Mar)
    the gold counterpart is B (latest by valid time), open; the stale A is open too, so `open` does
    not separate them (that is what the gated-records fit sees). Linking C to B closes B only: A
    stays open and current, the B-R5 leak of a missed conflict."""
    s = _dev_store(tmp_path, clock)
    a = _rem(s, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")["id"]
    b = _rem(s, "Ana desk is on floor 7.", ["Ana"], as_of="2026-02-01")["id"]
    assert _intervals(s) == {a: ("2026-01-01", None), b: ("2026-02-01", None)}
    out = _rem(s, "Ana desk is on floor 5.", ["Ana"], as_of="2026-03-01")
    rows = _by_id(out)
    assert rows[a]["features"]["open"] == rows[b]["features"]["open"] == 1.0
    assert rows[a]["features"]["older"] == rows[b]["features"]["older"] == 1.0
    assert rows[a]["valid_to"] is None and rows[b]["valid_to"] is None
    s.link(src=out["id"], rel="supersedes", dst=b)
    assert _intervals(s)[a] == ("2026-01-01", None)  # the leak: the stale member stays current
    assert _intervals(s)[b] == ("2026-02-01", "2026-03-01")
    s.close()


def test_dev_export_carries_each_candidate_validity_beside_its_features(tmp_path, clock):
    """Only the unfiltered dev run returns features, and with them the candidate's valid_from and
    valid_to, so store-time signals (valid at the new valid_from, chain position) can be computed
    offline from the records. A gated store returns neither."""
    s = _dev_store(tmp_path, clock)
    _rem(s, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")
    out = _rem(s, "Ana desk is on floor 5.", ["Ana"], as_of="2026-03-01")
    (row,) = out["possible_conflicts"]
    assert set(row) >= {"features", "valid_from", "valid_to"}
    assert row["valid_from"].startswith("2026-01-01") and row["valid_to"] is None
    s.close()


def test_same_type_is_one_for_candidates_of_the_new_memory_type(tmp_path, clock):
    """MEM2-156 round 5: `same_type`, 1 when the candidate's memory type (the store's `nodes.type`,
    set by the caller of `remember`) equals the new memory's. On the dev records the list's best
    for the remaining misses is a same-entity memory of another kind (an event against a fact,
    a preference against a fact) that shares surface words with the new text; the counterpart,
    being the same kind of fact about the same subject, is of the new memory's type."""
    from mem.confmodel import FEATURES

    assert "same_type" in FEATURES
    s = _dev_store(tmp_path, clock)
    fact = _rem(s, "Ana works at Acme as an engineer.", ["new:person:Ana"], as_of="2026-01-01")
    event = s.remember(text="Ana gave a talk at PyCon on Tuesday.", type="event", about=["Ana"],
                       as_of="2026-02-01").structured  # fmt: skip
    out = s.remember(text="Ana presented at FOSDEM on Saturday.", type="event", about=["Ana"],
                     as_of="2026-03-01").structured  # fmt: skip
    rows = _by_id(out)
    assert rows[event["id"]]["features"]["same_type"] == 1.0
    assert rows[fact["id"]]["features"]["same_type"] == 0.0
    s.close()
