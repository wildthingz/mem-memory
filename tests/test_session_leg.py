"""ADR 0022: `retrieval.session_leg` adds one vector per note/session (the L2-normalised mean of its
chunk vectors) and a vector leg over them that feeds RRF as an extra list. Off, nothing changes."""

import math

import numpy as np
import pytest

from mem import config, read, reembed, schema, write
from mem.importer import run_import
from mem.store import Store, doc_text
from mem.testing import HashEmbedder

ON = {**config.default_config().retrieval, "session_leg": True}
ROUNDS = [("alpha beta gamma", "delta"), ("epsilon zeta", "eta theta"), ("iota kappa", "lambda")]


def _expected_mean(texts, ctx):
    vecs = HashEmbedder().embed_docs([doc_text(ctx, t) for t in texts])
    mean = [sum(col) / len(vecs) for col in zip(*vecs, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean))
    return [x / norm for x in mean]


def _session_vecs(store):
    table = schema.session_table_name(store.vec_table)
    return {
        nid: list(np.frombuffer(blob, dtype=np.float32))
        for nid, blob in store.conn.execute(f"SELECT node_id, embedding FROM {table}")
    }


def _approx(a, b):
    return a == pytest.approx(b, abs=1e-6)


def _session(store, ref, date="2023-03-10", rounds=ROUNDS):
    return write.ingest_session(store, source_ref=ref, date=date, rounds=rounds)


def _allowed_all(store):
    store.conn.execute("DROP TABLE IF EXISTS temp.allowed")
    store.conn.execute("CREATE TEMP TABLE allowed(chunk_id INTEGER PRIMARY KEY)")
    store.conn.execute("INSERT INTO allowed SELECT id FROM chunks")


# ---- config ----------------------------------------------------------------------------------


def test_default_retrieval_has_no_session_leg_key_so_the_fingerprint_is_the_old_one():
    cfg = config.default_config()
    assert "session_leg" not in cfg.retrieval and cfg.session_leg is False


def test_session_leg_on_is_a_new_fingerprint_and_load_checks_the_type(tmp_path):
    base = config.default_config()
    on = base.replace("retrieval", session_leg=True)
    assert on.session_leg is True and on.fp_all() != base.fp_all()
    assert on.fp_embed() == base.fp_embed()
    p = tmp_path / "c.yaml"
    p.write_text("retrieval:\n  session_leg: true\n")
    assert config.load(p).fp_all() == on.fp_all()
    p.write_text("retrieval:\n  session_leg: maybe\n")
    with pytest.raises(config.ConfigError, match="true or false"):
        config.load(p)
    p.write_text("retrieval:\n  nonsense: 1\n")
    with pytest.raises(config.ConfigError, match="unknown field"):
        config.load(p)


# ---- the session vector ----------------------------------------------------------------------


def test_the_session_vector_is_the_normalised_mean_of_its_chunk_vectors(make_store):
    store = make_store(retrieval=ON)
    nid = _session(store, "s1")
    texts = ["\n".join(r) for r in ROUNDS]
    vecs = _session_vecs(store)
    assert list(vecs) == [nid]
    assert _approx(vecs[nid], _expected_mean(texts, "chat session 2023-03-10"))
    assert math.sqrt(sum(x * x for x in vecs[nid])) == pytest.approx(1.0, abs=1e-6)


def test_off_there_is_no_session_table_and_no_vectors(make_store):
    store = make_store()
    _session(store, "s1")
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master")}
    assert schema.session_table_name(store.vec_table) not in names


def test_memories_get_no_session_vector(make_store):
    store = make_store(retrieval=ON)
    store.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    assert _session_vecs(store) == {}


def test_a_changed_chunk_changes_the_session_vector_on_re_import(make_store, tmp_path):
    store = make_store(retrieval=ON)
    vault = tmp_path / "vault"
    (vault / "personal").mkdir(parents=True)
    note = vault / "personal" / "Trip.md"
    note.write_text("# Trip\n\nalpha beta\n\n## Plans\n\ngamma delta\n", encoding="utf-8")
    run_import(vault, {"personal": store}, routes={})
    (before,) = _session_vecs(store).values()
    note.write_text("# Trip\n\nalpha beta\n\n## Plans\n\nomega sigma\n", encoding="utf-8")
    run_import(vault, {"personal": store}, routes={})
    (after,) = _session_vecs(store).values()
    rows = store.conn.execute("SELECT ctx, text FROM chunks ORDER BY ord").fetchall()
    assert not _approx(before, after)
    assert _approx(after, _expected_all(rows))


def _expected_all(rows):
    vecs = HashEmbedder().embed_docs([doc_text(c, t) for c, t in rows])
    mean = [sum(col) / len(vecs) for col in zip(*vecs, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean))
    return [x / norm for x in mean]


def test_opening_a_store_with_the_leg_newly_on_backfills_missing_vectors(tmp_path, clock):
    path = tmp_path / "p.db"
    off = Store.open(path, HashEmbedder(), now=clock)
    nid = _session(off, "s1")
    off.close()
    on = Store.open(path, HashEmbedder(), now=clock, retrieval=ON)
    assert list(_session_vecs(on)) == [nid]
    on.close()


# ---- the leg -----------------------------------------------------------------------------------


def _leg(store, query, k=10):
    qvec = store.query_vector(query)
    return read.session_leg(store.conn, store.vec_table, qvec, k, store.retrieval["per_node_cap"])


def test_the_leg_ranks_sessions_and_a_hit_contributes_its_chunks(make_store):
    store = make_store(retrieval=ON)
    a = _session(store, "a", rounds=[("apple pear plum", "fig"), ("apple grape", "lime")])
    b = _session(store, "b", rounds=[("steel bolt nut", "screw"), ("bolt washer", "rivet")])
    _allowed_all(store)
    by_node = {}
    for cid, nid in store.conn.execute("SELECT id, node_id FROM chunks ORDER BY id"):
        by_node.setdefault(nid, []).append(cid)
    got = _leg(store, "apple pear plum fig grape lime")
    assert set(got[:2]) == set(by_node[a]) and set(got[2:]) == set(by_node[b])
    assert _leg(store, "steel bolt nut screw washer rivet")[:2] != got[:2]


def test_the_leg_contributes_at_most_per_node_cap_chunks_per_session(make_store):
    store = make_store(retrieval={**ON, "per_node_cap": 2})
    _session(store, "a", rounds=[(f"apple w{i}", "x") for i in range(6)])
    _allowed_all(store)
    assert len(_leg(store, "apple")) == 2


def test_the_leg_applies_the_eligible_set_before_its_limit(make_store):
    store = make_store(retrieval=ON)
    dead = _session(store, "dead", date="2023-01-01", rounds=[("apple apple apple", "apple")])
    live = [
        _session(store, f"l{i}", date="2023-03-10", rounds=[(f"apple w{i}", "z")]) for i in range(3)
    ]
    read.build_allowed(store.conn, store.now_ts(), since="2023-02-01")
    got = _leg(store, "apple apple apple", k=1)  # k=1: an unfiltered scan would return the dead one
    owners = {store.conn.execute("SELECT node_id FROM chunks WHERE id = ?", (c,)).fetchone()[0]
              for c in got}  # fmt: skip
    assert len(owners) == 1 and owners <= set(live) and dead not in owners


def test_a_retracted_session_is_not_in_the_leg(make_store):
    store = make_store(retrieval=ON)
    gone = _session(store, "gone", rounds=[("apple apple", "apple")])
    keep = _session(store, "keep", rounds=[("pear", "plum")])
    store.conn.execute("UPDATE nodes SET status = 'retracted' WHERE id = ?", (gone,))
    read.build_allowed(store.conn, None)
    owners = {store.conn.execute("SELECT node_id FROM chunks WHERE id = ?", (c,)).fetchone()[0]
              for c in _leg(store, "apple")}  # fmt: skip
    assert owners == {keep}


# ---- fusion ------------------------------------------------------------------------------------


def test_rrf_with_a_third_hand_ranked_list():
    # k=60: c2 is 2nd,1st,3rd; c1 is 1st,absent,1st; c3 is absent,2nd,2nd.
    lists = [[1, 2], [2, 3], [1, 3, 2]]
    got = dict(read.rrf(lists))
    assert got[1] == pytest.approx(1 / 61 + 1 / 61)
    assert got[2] == pytest.approx(1 / 62 + 1 / 61 + 1 / 63)
    assert got[3] == pytest.approx(1 / 62 + 1 / 62)
    assert [c for c, _ in read.rrf(lists)] == [2, 1, 3]


def test_the_pool_is_the_rrf_of_three_lists_with_the_leg_on(make_store):
    store = make_store(retrieval=ON)
    for i in range(4):
        _session(store, f"s{i}", rounds=[(f"apple fruit w{i}", "x"), (f"pear stone w{i}", "y")])
    pool, _ = read.candidate_pool(store, "apple pear", t=store.now_ts(), size=100)
    cfg = store.retrieval
    lists = [read.fts_leg(store.conn, "apple pear", cfg["leg_k"]),
             read.vec_leg(store.conn, store.vec_table, store.query_vector("apple pear"),
                          cfg["leg_k"]),
             _leg(store, "apple pear", cfg["leg_k"])]  # fmt: skip
    want = [c for c, _ in read.rrf(lists, k=cfg["rrf_k"])]
    assert [c for c, _ in pool] == want[: len(pool)] and len(pool) > 0


def test_off_the_pool_and_recall_are_what_they_were(make_store):
    plain, off = make_store("personal"), make_store("work", retrieval={**ON, "session_leg": False})
    for s in (plain, off):
        for i in range(3):
            _session(s, f"s{i}", rounds=[(f"apple w{i}", "x"), (f"pear w{i}", "y")])
    a = plain.recall(query="apple pear")
    b = off.recall(query="apple pear")
    assert a.text == b.text.replace("work", "personal") and a.structured == b.structured
    pa, _ = read.candidate_pool(plain, "apple", t=plain.now_ts())
    pb, _ = read.candidate_pool(off, "apple", t=off.now_ts())
    assert pa == pb


# ---- reembed -----------------------------------------------------------------------------------


class Wide(HashEmbedder):
    def __init__(self):
        super().__init__(dim=64)


def test_reembed_rebuilds_the_session_vectors_in_the_new_space(make_store):
    store = make_store(retrieval=ON)
    nid = _session(store, "s1")
    new_fp = "ab" * 32
    table = reembed.reembed(store.conn, Wide(), new_fp, session_leg=True)
    svec = schema.session_table_name(table)
    rows = dict(store.conn.execute(f"SELECT node_id, embedding FROM {svec}").fetchall())
    assert list(rows) == [nid]
    got = list(np.frombuffer(rows[nid], dtype=np.float32))
    vecs = Wide().embed_docs([doc_text("chat session 2023-03-10", "\n".join(r)) for r in ROUNDS])
    mean = [sum(col) / 3 for col in zip(*vecs, strict=True)]
    norm = math.sqrt(sum(x * x for x in mean))
    assert len(got) == 64 and _approx(got, [x / norm for x in mean])


def test_reembed_without_the_leg_creates_no_session_table(make_store):
    store = make_store()
    _session(store, "s1")
    table = reembed.reembed(store.conn, Wide(), "cd" * 32)
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master")}
    assert schema.session_table_name(table) not in names
