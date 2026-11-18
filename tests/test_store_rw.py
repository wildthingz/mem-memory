import json
import math

from mem import read

ANA = "new:person:Ana Silva"


def _remember(store, text, about=(ANA,), type="fact", **kw):
    return store.remember(text=text, type=type, about=list(about), **kw).structured


def _ids(res):
    return [r["id"] for r in res.structured["rows"]]


def test_remember_then_recall_returns_row(store):
    saved = _remember(
        store, "Ana Silva prefers async standups over daily meetings.", title="Ana async"
    )
    assert saved["status"] == "saved" and saved["id"].startswith("m")

    res = store.recall(query="async standups")
    assert _ids(res) == [saved["id"]]
    row = res.structured["rows"][0]
    assert (row["kind"], row["type"], row["title"], row["about"]) == (
        "memory", "fact", "Ana async", ["Ana Silva"],
    )  # fmt: skip
    assert (
        res.text.splitlines()[0] == "1 results (of 1 checked) · personal · uncalibrated (P1 build)"
    )
    assert res.text.splitlines()[1].startswith(f"{saved['id']} · fact · 2026-10-22 · Ana async")


def test_recall_on_empty_store_is_one_short_line(store):
    res = store.recall(query="anything at all")
    assert res.text == "no results · personal"
    assert res.structured["rows"] == []


def test_exact_duplicate_returns_exists(store):
    first = _remember(store, "Ana Silva owns the billing migration.")
    again = _remember(
        store, "  ana SILVA owns the  billing migration. ", about=["Ana Silva", "new:topic:Billing"]
    )
    assert (again["status"], again["id"]) == ("exists", first["id"])
    assert store.conn.execute("SELECT count(*) FROM nodes WHERE kind='memory'").fetchone() == (1,)
    # the existing memory gained an about edge to the subject that already existed
    n_about = store.conn.execute("SELECT count(*) FROM edges WHERE rel='about'").fetchone()
    assert n_about == (1,)


def test_ambiguous_about_writes_nothing(store):
    store.conn.executescript(
        """
        INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin)
          VALUES ('entity','person','Ana Silva','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','agent'),
                 ('entity','person','Ana Lopez','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','agent');
        INSERT INTO aliases(alias,alias_norm,node_id,origin) VALUES ('Ana','ana',1,'agent'),('Ana','ana',2,'agent');
        """
    )
    r = _remember(store, "Ana likes tea.", about=["Ana", "new:topic:Tea"])
    assert r["status"] == "needs_choice" and r["id"] is None
    assert sorted(r["candidates"]) == ["e1 · person · Ana Silva", "e2 · person · Ana Lopez"]
    assert store.conn.execute("SELECT count(*) FROM nodes").fetchone() == (2,)


def test_about_forms_alias_id_and_mention(store):
    a = _remember(store, "Ana Silva joined in 2024.")
    ana_id = a["subjects"][0].split()[0]
    assert ana_id.startswith("e")
    r = _remember(store, "Silva leads platform.", about=[f"Silva → #{ana_id}", "ana silva"])
    assert r["status"] == "saved"
    n = store.conn.execute("SELECT count(*) FROM edges WHERE rel='about' AND dst=?",
                           (int(ana_id[1:]),)).fetchone()  # fmt: skip
    assert n == (2,)
    # the mention is learned as an alias (origin agent)
    assert store.conn.execute("SELECT origin FROM aliases WHERE alias_norm='silva'").fetchone() == (
        "agent",
    )


def test_eligibility_future_dated_fact_hidden_until_valid_from(store, clock):
    old = _remember(store, "Weekly 1:1s with Ana Silva are on Mondays.", as_of="2026-06-03")
    new = _remember(store, "Weekly 1:1s with Ana Silva move to Thursdays.", as_of="2026-11-01")
    q = "weekly 1:1s with Ana"

    assert _ids(store.recall(query=q)) == [old["id"]]  # now = 2026-10-22
    assert _ids(store.recall(query=q, as_of="2026-10-31T23:59:59Z")) == [old["id"]]
    assert set(_ids(store.recall(query=q, as_of="2026-11-01T00:00:00Z"))) == {
        old["id"],
        new["id"],
    }  # valid_from is inclusive (no supersedes edge in P1, so both are open)
    assert _ids(store.recall(query=q, as_of="2026-05-01")) == []


def test_closed_interval_is_half_open_and_include_history_drops_it(store):
    m = _remember(store, "Ana Silva sits in Berlin.", as_of="2026-01-01")
    mid = int(m["id"][1:])
    store.conn.execute("UPDATE nodes SET valid_to='2026-03-01T00:00:00Z' WHERE id=?", (mid,))
    q = "Ana sits Berlin"
    assert _ids(store.recall(query=q, as_of="2026-02-28T23:59:59Z")) == [m["id"]]
    assert _ids(store.recall(query=q, as_of="2026-03-01T00:00:00Z")) == []
    assert _ids(store.recall(query=q, as_of="2026-03-05", include_history=True)) == [m["id"]]


def test_retracted_and_other_filters_hide_rows(store):
    a = _remember(store, "Ana Silva decided to adopt sqlite.", type="decision")
    b = _remember(store, "Ana Silva dislikes sqlite locks.", type="preference")
    assert set(_ids(store.recall(query="sqlite", type=["decision"]))) == {a["id"]}
    store.conn.execute("UPDATE nodes SET status='retracted' WHERE id=?", (int(b["id"][1:]),))
    assert _ids(store.recall(query="sqlite")) == [a["id"]]


def test_unknown_about_filter_is_dropped_and_reported(store):
    a = _remember(store, "Ana Silva uses vim.")
    res = store.recall(query="vim", about=["Nobody"])
    assert _ids(res) == [a["id"]]
    assert res.structured["filters_dropped"] == ["about 'Nobody': unknown, not applied"]
    assert _ids(store.recall(query="vim", about=["ana silva"])) == [a["id"]]


def _bulk_memory(store, text, valid_from="2026-01-01T00:00:00Z", kind="memory", type="fact"):
    """Insert a memory (+ body chunk + vector) directly, bypassing remember."""
    cur = store.conn.execute(
        "INSERT INTO nodes(kind,type,name,body,valid_from,recorded_at,origin,content_hash)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (kind, type, text[:20], text, valid_from, valid_from, "agent", text),
    )
    nid = cur.lastrowid
    cur = store.conn.execute(
        "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,ctx,text,tokens,content_hash)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (nid, "body", 0, "body", text[:20], text, 5, text),
    )
    store.add_vector(cur.lastrowid, text[:20], text)
    return nid


def test_fts_and_vec_legs_apply_scope_before_limit(store):
    # 6 future-dated rows match the query far better than the one eligible row, and leg_k is 3:
    # scoping after LIMIT would return nothing.
    for i in range(6):
        _bulk_memory(store, f"ana thursday meeting plan {i}", valid_from="2027-01-01T00:00:00Z")
    ok = _bulk_memory(store, "meeting notes with ana about budget")
    store.conn.commit()

    allowed = read.build_allowed(store.conn, t="2026-10-22T09:00:00Z")
    q = "ana thursday meeting plan"
    fts = read.fts_leg(store.conn, q, k=3)
    assert [cid for (cid,) in store.conn.execute("SELECT chunk_id FROM allowed")] == [
        cid for (cid,) in store.conn.execute("SELECT id FROM chunks WHERE node_id=?", (ok,))
    ]
    assert len(allowed) == 1 and fts == allowed
    vec = read.vec_leg(store.conn, store.vec_table, store.embedder.embed_query(q), k=3)
    assert vec == allowed

    ids = store.recall(query=q).structured["rows"]
    assert [r["id"] for r in ids] == [f"m{ok}"]


def test_rrf_fuses_hand_ranked_lists():
    fused = read.rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused] == [1, 3, 2, 4]
    assert math.isclose(fused[0][1], 1 / 61 + 1 / 62)
    assert math.isclose(fused[1][1], 1 / 63 + 1 / 61)
    assert math.isclose(fused[2][1], 1 / 62)
    assert math.isclose(fused[3][1], 1 / 63)


def test_recall_returns_at_most_five_rows_and_caps_chunks_per_node(store):
    for i in range(8):
        _remember(store, f"Ana Silva note about gardening tip number {i}.")
    res = store.recall(query="gardening tip")
    assert len(res.structured["rows"]) == 5
    assert res.structured["checked"] == 8


def test_events_log_every_tool_call(store):
    saved = _remember(store, "Ana Silva keeps a gratitude journal.")
    store.recall(query="gratitude")
    store.get(ids=[saved["id"]])
    rows = store.conn.execute(
        "SELECT tool, args, result, latency_ms FROM events ORDER BY id"
    ).fetchall()
    assert [r[0] for r in rows] == ["remember", "recall", "get"]
    assert json.loads(rows[0][1])["type"] == "fact"
    assert json.loads(rows[1][2])["ids"] == [saved["id"]]
    assert all(r[3] is not None for r in rows)


def test_stores_are_separate_files(make_store):
    personal, work = make_store("personal"), make_store("work")
    _remember(personal, "Ana Silva plays chess.")
    assert work.recall(query="chess").text == "no results · work"


def test_reopening_with_another_embedder_dimension_is_refused(tmp_path):
    import pytest

    from mem.common import MemError
    from mem.store import Store
    from mem.testing import HashEmbedder

    Store.open(tmp_path / "x.db", HashEmbedder(256)).close()
    with pytest.raises(MemError, match="created for"):
        Store.open(tmp_path / "x.db", HashEmbedder(64))
