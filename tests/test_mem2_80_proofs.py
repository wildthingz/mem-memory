"""MEM2-80 regression tests for P3 review F1: retracting or `duplicate_of`-merging the middle of a
supersedes chain keeps the chain connected (architecture-v2 §3.1, A1, N1; B-R5 T0 case (d))."""

import pytest

from mem.serve import MemServer

ABOUT = ["new:person:Ana"]


def _chain(store):
    srv = MemServer(store)

    def rem(text, as_of, about=ABOUT):
        args = {"text": text, "type": "fact", "about": about, "as_of": as_of}
        return srv.call_tool("remember", args)[1]["id"]

    m1 = rem("Ana weekly planning session is on Monday.", "2026-06-03")
    m2 = rem("Ana weekly planning session is on Wednesday.", "2026-08-01")
    m3 = rem("Ana weekly planning session is on Thursday.", "2026-11-01")
    srv.call_tool("link", {"src": m2, "rel": "supersedes", "dst": m1})
    srv.call_tool("link", {"src": m3, "rel": "supersedes", "dst": m2})
    return srv, rem, m1, m2, m3


def _valid_to(store, ref):
    return store.conn.execute(
        "SELECT valid_to FROM nodes WHERE id = ?", (int(ref[1:]),)
    ).fetchone()[0]  # fmt: skip


def _ids(srv, **kw):
    rows = srv.call_tool("recall", {"query": "Ana weekly planning session", **kw})[1]["rows"]
    return [r["id"] for r in rows]


def _retract(srv, rem, mid):
    srv.call_tool("forget", {"id": mid, "reason": "entered by mistake"})


def _merge(srv, rem, mid):
    other = rem(
        "Ana's weekly planning session happens on Wednesday.", "2026-08-01", ["new:topic:X"]
    )
    srv.call_tool("link", {"src": mid, "rel": "duplicate_of", "dst": other})


@pytest.mark.parametrize("drop", [_retract, _merge], ids=["forget", "duplicate_of"])
def test_dropping_the_middle_member_keeps_the_chain_connected(store, drop):
    srv, rem, m1, m2, m3 = _chain(store)
    drop(srv, rem, m2)
    assert _valid_to(store, m1) == "2026-11-01T00:00:00Z"  # closed by m3, never reopened
    assert _valid_to(store, m3) is None
    assert m1 in _ids(srv, as_of="2026-09-15") and m3 not in _ids(srv, as_of="2026-09-15")
    assert m3 in _ids(srv, as_of="2026-12-01") and m1 not in _ids(srv, as_of="2026-12-01")


def test_retracting_the_middle_then_the_newest_reopens_the_oldest(store):
    srv, rem, m1, m2, m3 = _chain(store)
    _retract(srv, rem, m2)
    _retract(srv, rem, m3)
    assert _valid_to(store, m1) is None
    assert m1 in _ids(srv)
