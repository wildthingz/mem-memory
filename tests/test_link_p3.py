"""P3 `link`, `forget`, `subjects` (architecture-v2 §3.1 link semantics, §4.1): idempotency, the
endpoint rules, chronology, collision-aware merges and the imported-node protection."""

import pytest

from mem.common import MemError
from mem.serve import MemServer

NOW = "2026-10-22T09:00:00Z"


def _rem(store, text, about=("Topic",), **kw):
    return store.remember(
        text=text, type=kw.pop("type", "fact"), about=[f"new:topic:{a}" for a in about], **kw
    ).structured


def _mid(res):
    return int(res["id"][1:])


def _entity(store, name, type="person", *aliases, origin="agent", source_ref=None):
    nid = store.conn.execute(
        "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin,source_ref)"
        " VALUES ('entity',?,?,?,?,?,?)", (type, name, NOW, NOW, origin, source_ref),
    ).lastrowid  # fmt: skip
    for a in (name, *aliases):
        store.conn.execute(
            "INSERT INTO aliases(alias,alias_norm,node_id,origin) VALUES (?,?,?,?)",
            (a, a.casefold(), nid, origin),
        )
    return nid


def _imported_note(store, name="Atlas notes"):
    return store.conn.execute(
        "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin,source_ref)"
        " VALUES ('note','note',?,?,?,'import',?)", (name, NOW, NOW, f"{name}.md"),
    ).lastrowid  # fmt: skip


def _imported_memory(store, text="Imported fact."):
    return store.conn.execute(
        "INSERT INTO nodes(kind,type,name,body,valid_from,recorded_at,origin,content_hash)"
        " VALUES ('memory','fact',?,?,?,?,'import','h-imported')", (text, text, NOW, NOW),
    ).lastrowid  # fmt: skip


def _edge(store, src, dst, rel, origin="agent"):
    return store.conn.execute(
        "INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES (?,?,?,?,?)",
        (src, dst, rel, origin, NOW),
    ).lastrowid  # fmt: skip


def _live(store, rel=None):
    q = "SELECT src, dst, rel FROM edges WHERE removed_at IS NULL"
    q += " AND rel = ?" if rel else ""
    return sorted(store.conn.execute(q + " ORDER BY src, dst, rel", (rel,) if rel else ()))


def _node(store, nid):
    return store.conn.execute(
        "SELECT status, valid_to FROM nodes WHERE id = ?", (nid,)
    ).fetchone()  # fmt: skip


def _link(store, src, rel, dst, **kw):
    return store.link(src=src, rel=rel, dst=dst, **kw).structured


# ---- link: generic rules ---------------------------------------------------------------------


def test_repeated_link_noop(store):
    a, b = _mid(_rem(store, "Atlas ships in March.")), _mid(_rem(store, "Atlas uses Postgres."))
    first = _link(store, f"m{a}", "relates_to", f"m{b}", note="same project")
    assert first["ok"] and first["error"] is None
    assert first["edge"]["src"] == f"m{a}" and first["edge"]["note"] == "same project"
    events = store.conn.execute("SELECT count(*) FROM events").fetchone()
    again = _link(store, f"m{a}", "relates_to", f"m{b}")
    assert again["ok"] and again["edge"] == first["edge"] and again["side_effects"] == []
    assert _live(store, "relates_to") == [(a, b, "relates_to")]
    assert store.conn.execute("SELECT count(*) FROM edges WHERE rel='relates_to'").fetchone() == (
        1,
    )
    assert store.conn.execute("SELECT count(*) FROM events").fetchone()[0] == events[0] + 1


def test_repeated_supersedes_in_either_direction_is_a_noop(store, clock):
    old = _mid(_rem(store, "1:1s are on Mondays.", as_of="2026-06-03"))
    new = _mid(_rem(store, "1:1s are on Thursdays.", as_of="2026-11-01"))
    r = _link(store, f"m{new}", "supersedes", f"m{old}")
    assert r["side_effects"] == [f"m{old} valid_to → 2026-11-01T00:00:00Z"]
    assert _link(store, f"m{new}", "supersedes", f"m{old}")["side_effects"] == []
    assert _link(store, f"m{old}", "supersedes", f"m{new}")["side_effects"] == []
    assert _live(store, "supersedes") == [(new, old, "supersedes")]


def test_self_link_is_an_error(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    with pytest.raises(MemError, match="itself"):
        _link(store, f"m{a}", "relates_to", f"m{a}")


@pytest.mark.parametrize(
    ("rel", "src", "dst", "needle"),
    [
        ("befriends", "m1", "m3", "supersedes"),  # unknown rel lists the valid ones
        ("same_as", "m1", "m3", "entity"),  # wrong endpoint kinds
        ("supersedes", "e2", "m1", "memory"),
        ("part_of", "m1", "m3", "import"),  # importer-only rel
    ],
)
def test_unknown_rel_or_wrong_kinds_list_the_valid_rels(store, rel, src, dst, needle):
    _rem(store, "Atlas ships in March.")
    _rem(store, "Atlas uses Postgres.")
    with pytest.raises(MemError, match=needle):
        _link(store, src, rel, dst)


def test_bad_ids_and_missing_nodes_are_errors(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    with pytest.raises(MemError, match="x9"):
        _link(store, "x9", "relates_to", f"m{a}")
    with pytest.raises(MemError, match="m99"):
        _link(store, f"m{a}", "relates_to", "m99")


def test_both_ends_must_be_active(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    b = _mid(_rem(store, "Atlas uses Postgres."))
    store.forget(id=f"m{b}", reason="wrong")
    with pytest.raises(MemError, match=rf"m{b} is not active"):
        _link(store, f"m{a}", "relates_to", f"m{b}")


def test_imported_node_can_be_the_dst_of_a_rel(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    n = _imported_note(store)
    assert _link(store, f"m{a}", "relates_to", f"n{n}")["ok"]
    assert _live(store, "relates_to") == [(a, n, "relates_to")]


def test_link_cannot_invalidate_imported_node(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    im = _imported_memory(store)
    note = _imported_note(store)
    before = [_node(store, n) for n in (a, im, note)]
    for src, rel, dst in [
        (f"m{a}", "supersedes", f"m{im}"),
        (f"m{im}", "supersedes", f"m{a}"),
        (f"m{im}", "duplicate_of", f"m{a}"),
        (f"m{a}", "supersedes", f"n{note}"),
        (f"n{note}", "relates_to", f"m{a}"),
    ]:
        with pytest.raises(MemError, match="edit the vault"):
            _link(store, src, rel, dst)
    with pytest.raises(MemError, match="edit the vault"):
        store.forget(id=f"m{im}", reason="x")
    with pytest.raises(MemError, match="edit the vault"):
        store.forget(id=f"n{note}", reason="x")
    assert [_node(store, n) for n in (a, im, note)] == before
    assert _live(store) == [(a, store.conn.execute(
        "SELECT dst FROM edges WHERE src=?", (a,)).fetchone()[0], "about")]  # fmt: skip


def test_remove_deletes_an_agent_edge_and_is_repeatable(store):
    a, b = _mid(_rem(store, "Atlas ships in March.")), _mid(_rem(store, "Atlas uses Postgres."))
    _link(store, f"m{a}", "relates_to", f"m{b}")
    r = _link(store, f"m{a}", "relates_to", f"m{b}", remove=True)
    assert r["ok"] and _live(store, "relates_to") == []
    assert store.conn.execute("SELECT removed_at FROM edges WHERE rel='relates_to'").fetchone() == (
        NOW,
    )  # soft: the row stays
    assert _link(store, f"m{a}", "relates_to", f"m{b}", remove=True)["ok"]
    assert _link(store, f"m{a}", "relates_to", f"m{b}")["ok"]  # can be re-added
    assert _live(store, "relates_to") == [(a, b, "relates_to")]


def test_remove_refuses_an_imported_edge(store):
    a = _mid(_rem(store, "Atlas ships in March."))
    n = _imported_note(store)
    _edge(store, n, a, "links_to", origin="import")
    with pytest.raises(MemError, match="edit the vault"):
        _link(store, f"n{n}", "links_to", f"m{a}", remove=True)
    assert _live(store, "links_to") == [(n, a, "links_to")]


# ---- link: supersedes chronology -------------------------------------------------------------


def test_link_supersedes_swaps_a_backwards_call_and_says_so(store):
    old = _mid(_rem(store, "1:1s are on Mondays.", as_of="2026-06-03"))
    new = _mid(_rem(store, "1:1s are on Thursdays.", as_of="2026-11-01"))
    r = _link(store, f"m{old}", "supersedes", f"m{new}")  # the call has it backwards
    assert "direction swapped" in r["side_effects"]
    assert r["edge"]["src"] == f"m{new}" and r["edge"]["dst"] == f"m{old}"
    assert _node(store, old) == ("active", "2026-11-01T00:00:00Z")
    assert _node(store, new) == ("active", None)


def test_remove_supersedes_reopens_the_predecessor(store):
    old = _mid(_rem(store, "1:1s are on Mondays.", as_of="2026-06-03"))
    new = _mid(_rem(store, "1:1s are on Thursdays.", as_of="2026-11-01"))
    _link(store, f"m{new}", "supersedes", f"m{old}")
    r = _link(store, f"m{new}", "supersedes", f"m{old}", remove=True)
    assert r["side_effects"] == [f"m{old} valid_to → open"]
    assert _node(store, old)[1] is None


def test_reopen_collision_rolls_back(store):
    m1 = _mid(_rem(store, "Atlas ships in March."))
    m2 = _mid(_rem(store, "Atlas ships in April."))
    _link(store, f"m{m2}", "supersedes", f"m{m1}")
    m3 = _rem(store, "Atlas ships in March.")  # the superseded text again: saved, not exists
    assert m3["status"] == "saved"
    m3 = _mid(m3)
    snapshot = store.conn.execute("SELECT id, status, valid_to FROM nodes ORDER BY id").fetchall()
    with pytest.raises(MemError) as err:
        store.forget(id=f"m{m2}", reason="wrong")  # m1 would reopen beside m3
    assert f"m{m1} would reopen while m{m3} has the same text" in str(err.value)
    assert store.conn.execute("SELECT id, status, valid_to FROM nodes ORDER BY id").fetchall() == (
        snapshot
    )
    assert not store.conn.in_transaction
    with pytest.raises(MemError, match=rf"m{m1} would reopen while m{m3}"):
        _link(store, f"m{m2}", "supersedes", f"m{m1}", remove=True)
    assert _live(store, "supersedes") == [(m2, m1, "supersedes")]
    assert store.conn.execute("SELECT count(*) FROM events WHERE tool='forget'").fetchone() == (0,)


# ---- link: duplicate_of ----------------------------------------------------------------------


def test_duplicate_of_hides_src_and_copies_about_edges(store):
    a = _mid(_rem(store, "Ana runs the guild.", about=("Guild", "Ana")))
    b = _mid(_rem(store, "Ana leads the guild.", about=("Guild",)))
    r = _link(store, f"m{b}", "duplicate_of", f"m{a}")
    assert r["ok"]
    assert _node(store, b)[0] == "duplicate"
    about = lambda m: sorted(
        d for (d,) in store.conn.execute(
            "SELECT dst FROM edges WHERE src=? AND rel='about' AND removed_at IS NULL", (m,))
    )  # fmt: skip
    assert len(about(a)) == 2 and set(about(b)) <= set(about(a))
    assert _link(store, f"m{b}", "duplicate_of", f"m{a}")["side_effects"] == []  # repeat: no-op


def test_duplicate_of_rederives_the_chain_it_leaves(store):
    old = _mid(_rem(store, "1:1s are on Mondays.", as_of="2026-06-03"))
    new = _mid(_rem(store, "1:1s are on Thursdays.", as_of="2026-11-01"))
    other = _mid(_rem(store, "1:1s are on Thursdays!", as_of="2026-11-01", about=("Other",)))
    _link(store, f"m{new}", "supersedes", f"m{old}")
    r = _link(store, f"m{new}", "duplicate_of", f"m{other}")
    assert f"m{old} valid_to → open" in r["side_effects"]
    assert _node(store, old) == ("active", None)


def test_duplicate_of_unmerge_is_refused(store):
    a, b = _mid(_rem(store, "Ana runs the guild.")), _mid(_rem(store, "Ana leads the guild."))
    _link(store, f"m{b}", "duplicate_of", f"m{a}")
    with pytest.raises(MemError, match="not undone"):
        _link(store, f"m{b}", "duplicate_of", f"m{a}", remove=True)
    assert _node(store, b)[0] == "duplicate"


# ---- link: same_as ---------------------------------------------------------------------------


def test_same_as_shared_neighbour(store):
    e1 = _entity(store, "Ana Silva", "person", "ana", "AS")
    e2 = _entity(store, "Ana S.", "person", "ana", "Anita")  # shares the alias "ana"
    shared = _entity(store, "Guild", "topic")
    only1 = _entity(store, "Berlin", "place")
    m = _mid(_rem(store, "Ana runs the guild.", about=("Guild",)))
    for e in (e1, e2):
        _edge(store, e, shared, "relates_to")  # a shared neighbour: would collide on re-point
        _edge(store, m, e, "about")  # the memory is about both
    _edge(store, e1, only1, "relates_to")
    _edge(store, e1, e2, "relates_to")  # between the two: would become a self-loop
    r = _link(store, f"e{e1}", "same_as", f"e{e2}")
    assert r["ok"] and r["error"] is None
    assert _node(store, e1)[0] == "duplicate" and _node(store, e2)[0] == "active"
    assert store.conn.execute(
        "SELECT alias_norm FROM aliases WHERE node_id=? ORDER BY 1", (e2,)
    ).fetchall() == [("ana",), ("ana s.",), ("ana silva",), ("anita",), ("as",)]
    assert store.conn.execute("SELECT count(*) FROM aliases WHERE node_id=?", (e1,)).fetchone() == (
        0,
    )
    live = _live(store)
    assert (e2, shared, "relates_to") in live and (e2, only1, "relates_to") in live  # re-pointed
    assert (m, e2, "about") in live
    assert not any(e1 in (s, d) and rel != "same_as" for s, d, rel in live)  # nothing left on e1
    assert not any(s == d for s, d, _ in live)
    assert (e1, e2, "same_as") in live  # the merge itself is recorded
    assert len(live) == len(set(live))


def test_same_as_is_idempotent(store):
    e1, e2 = _entity(store, "Ana S."), _entity(store, "Ana Silva")
    _link(store, f"e{e1}", "same_as", f"e{e2}")
    assert _link(store, f"e{e1}", "same_as", f"e{e2}")["side_effects"] == []
    assert _live(store, "same_as") == [(e1, e2, "same_as")]


def test_same_as_swaps_when_only_src_is_imported(store):
    imp = _entity(store, "Ana Silva", "person", origin="import", source_ref="people/Ana.md")
    mine = _entity(store, "Ana S.", "person")
    r = _link(store, f"e{imp}", "same_as", f"e{mine}")
    assert "direction swapped" in r["side_effects"]
    assert _node(store, mine)[0] == "duplicate" and _node(store, imp)[0] == "active"
    assert store.conn.execute(
        "SELECT alias_norm FROM aliases WHERE node_id=? ORDER BY 1", (imp,)
    ).fetchall() == [("ana s.",), ("ana silva",)]


def test_same_as_two_imported_entities_is_refused(store):
    a = _entity(store, "Ana", "person", origin="import", source_ref="a.md")
    b = _entity(store, "Ana Silva", "person", origin="import", source_ref="b.md")
    with pytest.raises(MemError, match="edit the vault"):
        _link(store, f"e{a}", "same_as", f"e{b}")
    assert _node(store, a)[0] == _node(store, b)[0] == "active"


def test_same_as_repoints_imported_edges(store):
    e1, e2 = _entity(store, "Ana S.", "person", "anita"), _entity(store, "Ana Silva")
    note = _imported_note(store)
    _edge(store, note, e1, "links_to", origin="import")
    r = _link(store, f"e{e1}", "same_as", f"e{e2}")  # imported edges are re-pointed, not dropped
    assert (note, e2, "links_to") in _live(store) and r["ok"]


# ---- forget ----------------------------------------------------------------------------------


def test_forget_retracts_and_never_deletes(store):
    a = _mid(_rem(store, "Ana runs the guild."))
    r = store.forget(id=f"m{a}", reason="wrong person")
    assert r.structured == {"ok": True, "id": f"m{a}", "status": "retracted"}
    assert r.text.startswith(f"ok · m{a} retracted")
    row = store.conn.execute("SELECT status, status_at FROM nodes WHERE id=?", (a,)).fetchone()
    assert row == ("retracted", NOW)
    assert store.conn.execute(
        "SELECT value FROM props WHERE node_id=? AND key='retracted_reason'", (a,)
    ).fetchone() == ("wrong person",)
    assert store.conn.execute("SELECT count(*) FROM chunks WHERE node_id=?", (a,)).fetchone()[0] > 0
    assert store.recall(query="Ana runs the guild", about=None).structured["rows"] == []
    assert store.forget(id=f"m{a}", reason="again").structured["status"] == "retracted"  # repeat


def test_forget_rederives_the_chain_without_it(store):
    old = _mid(_rem(store, "1:1s are on Mondays.", as_of="2026-06-03"))
    new = _mid(_rem(store, "1:1s are on Thursdays.", as_of="2026-11-01"))
    _link(store, f"m{new}", "supersedes", f"m{old}")
    r = store.forget(id=f"m{new}", reason="not decided yet")
    assert f"m{old} valid_to → open" in r.text
    assert _node(store, old) == ("active", None)


@pytest.mark.parametrize("ident", ["e1", "x1", "m99", "banana"])
def test_forget_only_memories_that_exist(store, ident):
    _entity(store, "Ana")
    with pytest.raises(MemError):
        store.forget(id=ident, reason="x")


def test_forget_needs_a_reason(store):
    a = _mid(_rem(store, "Ana runs the guild."))
    with pytest.raises(MemError, match="reason"):
        store.forget(id=f"m{a}", reason="  ")


# ---- subjects --------------------------------------------------------------------------------


def test_subjects_row_format_and_counts(store):
    ana = _entity(store, "Ana Silva", "person", "Ana", "AS")
    for text in ("Ana likes tea.", "Ana runs the guild."):
        store.remember(text=text, type="fact", about=[f"#e{ana}"])
    note = _imported_note(store)
    _edge(store, note, ana, "links_to", origin="import")
    res = store.subjects().structured["subjects"]
    assert res == [{"id": f"e{ana}", "type": "person", "name": "Ana Silva",
                    "aliases": ["Ana", "AS"], "memories": 2, "notes": 1}]  # fmt: skip
    assert (
        store.subjects().text == f"e{ana} · person · Ana Silva (aka Ana, AS) · 2 memories · 1 note"
    )


def test_subjects_filters_by_query_type_and_limit(store):
    ana = _entity(store, "Ana Silva", "person", "Ana")
    _entity(store, "Atlas", "project")
    _entity(store, "Anaconda", "thing")
    ids = lambda **kw: [s["id"] for s in store.subjects(**kw).structured["subjects"]]
    assert ids(query="ana silva") == [f"e{ana}"]
    assert ids(query="ANA")[0] == f"e{ana}" and len(ids(query="ana")) == 2  # exact alias first
    assert len(ids(type="person")) == 1 and len(ids(limit=2)) == 2
    assert store.subjects(query="zzz").text == "no subjects"
    with pytest.raises(MemError, match="limit"):
        store.subjects(limit=0)


def test_subjects_hides_merged_and_retracted_counts(store):
    e1, e2 = _entity(store, "Ana S."), _entity(store, "Ana Silva")
    m = _mid(store.remember(text="Ana runs the guild.", type="fact", about=[f"#e{e1}"]).structured)
    _link(store, f"e{e1}", "same_as", f"e{e2}")
    assert [s["id"] for s in store.subjects().structured["subjects"]] == [f"e{e2}"]
    assert store.subjects().structured["subjects"][0]["memories"] == 1
    store.forget(id=f"m{m}", reason="x")
    assert store.subjects().structured["subjects"][0]["memories"] == 0


# ---- MCP surface -----------------------------------------------------------------------------


def test_tools_are_served_over_the_contract(store):
    ms = MemServer(store)
    ms.call_tool("remember", {"text": "Atlas ships in March.", "type": "fact",
                              "about": ["new:topic:Atlas"]})  # fmt: skip
    ms.call_tool("remember", {"text": "Atlas ships in April.", "type": "fact",
                              "about": ["#e2"]})  # fmt: skip
    text, structured = ms.call_tool("link", {"src": "m3", "rel": "supersedes", "dst": "m1"})
    assert structured["ok"] and "valid_to" in text
    _, s = ms.call_tool("subjects", {"query": "atlas"})
    assert s["subjects"][0]["name"] == "Atlas" and s["subjects"][0]["memories"] == 2
    text, s = ms.call_tool("forget", {"id": "m3", "reason": "wrong"})
    assert s == {"ok": True, "id": "m3", "status": "retracted"}
    with pytest.raises(MemError, match="bad arguments for forget"):
        ms.call_tool("forget", {"id": "m3"})
    with pytest.raises(MemError, match="bad arguments for link"):
        ms.call_tool("link", {"src": "m3", "rel": "supersedes"})
