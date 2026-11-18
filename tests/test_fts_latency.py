"""F-P2a (MEM2-82): the BM25 leg stays fast at L5k and its results are unchanged by the fix.

Root cause: `rowid IN (SELECT chunk_id FROM allowed)` was offered to FTS5 as a `rowid = ?`
constraint, so FTS5 ran the whole MATCH + bm25 once per eligible chunk (O(allowed x matches)).
"""

import random
import time

import pytest

from mem import read

T = "2026-10-22T09:00:00Z"
COMMON = ["meeting", "project", "budget", "ana", "plan", "thursday", "notes", "team", "decided"]
QUERY = "what did ana decide about the project budget at the thursday team meeting"
# The pre-fix query, kept as the oracle for "same ids, same order": MATCH, scope, bm25, LIMIT.
ORACLE = (
    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
    " AND rowid IN (SELECT chunk_id FROM allowed) ORDER BY bm25(chunks_fts, 2.0, 1.0) LIMIT ?"
)


def _fill(conn, nodes: int, per_node: int, words: int, seed: int = 1, future_every: int = 0):
    """`nodes` memories of `per_node` section chunks of `words` words; ~8% are common query terms.
    Every `future_every`-th node is valid only from 2027 (outside the eligible set at T)."""
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(4000)]
    for n in range(nodes):
        future = future_every and n % future_every == 0
        valid = "2027-01-01T00:00:00Z" if future else "2026-01-01T00:00:00Z"
        nid = conn.execute(
            "INSERT INTO nodes(kind,type,name,body,valid_from,recorded_at,origin,content_hash)"
            " VALUES ('memory','fact',?,'',?,?,'agent',?)",
            (f"n{n}", valid, valid, f"h{n}"),
        ).lastrowid
        for j in range(per_node):
            text = " ".join(
                rnd.choice(COMMON) if rnd.random() < 0.08 else rnd.choice(vocab)
                for _ in range(words)
            )
            conn.execute(
                "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,ctx,text,tokens,content_hash)"
                " VALUES (?,?,?,'section',?,?,?,?)",
                (nid, f"k{j}", j, f"n{n} › s{j}", text, words, text),
            )
    conn.commit()


@pytest.fixture(scope="module")
def l5k(tmp_path_factory):
    """5,004 chunks / 834 nodes (the P2 host gate's L5k shape), all eligible at T."""
    from mem.store import Store
    from mem.testing import HashEmbedder

    s = Store.open(tmp_path_factory.mktemp("l5k") / "l5k.db", HashEmbedder(), scope="personal")
    _fill(s.conn, nodes=834, per_node=6, words=120)
    yield s
    s.close()


def test_fts_leg_p95_at_l5k_is_at_most_50ms(l5k):
    conn = l5k.conn
    assert len(read.build_allowed(conn, T)) == 5004
    read.fts_leg(conn, QUERY, 50)  # warm the page cache
    runs = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = read.fts_leg(conn, QUERY, 50)
        runs.append((time.perf_counter() - t0) * 1000)
    assert len(out) == 50
    p95 = sorted(runs)[18]  # 19th of 20
    assert p95 <= 50, f"fts leg p95 {p95:.0f} ms at L5k"


def test_fts_leg_does_not_push_scope_into_fts5_as_a_rowid_lookup(l5k):
    # FTS5's idxStr marks a rowid equality constraint with '='; with it, MATCH runs once per row.
    conn = l5k.conn
    read.build_allowed(conn, T)
    sent: list[str] = []
    conn.set_trace_callback(sent.append)
    try:
        read.fts_leg(conn, "ana", 50)
    finally:
        conn.set_trace_callback(None)
    (sql,) = [q for q in sent if "chunks_fts MATCH" in q]
    plan = " ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert "VIRTUAL TABLE INDEX" in plan
    idx = plan.split("VIRTUAL TABLE INDEX ")[1].split()[0]
    assert "=" not in idx, plan


@pytest.mark.parametrize(
    "query", [QUERY, "ana", "budget thursday w17", "w1 w2 w3 meeting", "notes"]
)
@pytest.mark.parametrize("k", [3, 50, 500])
def test_fts_leg_ids_and_order_match_the_pre_fix_query(make_store, query, k):
    s = make_store()
    # a third of the nodes are future-dated, so scope must still apply before LIMIT
    _fill(s.conn, nodes=120, per_node=3, words=60, seed=7, future_every=3)
    allowed = set(read.build_allowed(s.conn, T))
    expected = [r[0] for r in s.conn.execute(ORACLE, (read.fts_match(query), k))]
    got = read.fts_leg(s.conn, query, k)
    assert got == expected
    assert got and set(got) <= allowed
