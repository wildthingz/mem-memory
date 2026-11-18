"""P3 write path (architecture-v2 §3.1): resolver, conflict candidates, two-phase write,
inline `supersedes` and chain re-derivation."""

import threading

import pytest

from mem.common import MemError
from mem.gate import Gate, read_gate, write_gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker, stub_conf_model

NOW = "2026-10-22T09:00:00Z"


@pytest.fixture
def cstore(tmp_path, clock):
    """A store with a reranker and a gate whose θ_conf is 0.5 (OverlapReranker logit
    6·overlap − 3, so a candidate needs ≥ 58% of the new text's words)."""
    s = Store.open(tmp_path / "c.db", HashEmbedder(), now=clock, reranker=OverlapReranker())
    write_gate(
        s.conn,
        Gate(
            1.0,
            0.0,
            0.5,
            0.5,
            0.0,
            theta_conf=0.5,
            conf_model=stub_conf_model(),
            fingerprint_hash="0" * 64,
        ),
    )
    s.gate = read_gate(s.conn)
    yield s
    s.close()


def _rem(store, text, about, **kw):
    return store.remember(text=text, type=kw.pop("type", "fact"), about=about, **kw).structured


def _entity(store, name, type="person", *aliases):
    nid = store.conn.execute(
        "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin)"
        " VALUES ('entity',?,?,?,?,'agent')", (type, name, NOW, NOW),
    ).lastrowid  # fmt: skip
    for a in (name, *aliases):
        store.conn.execute(
            "INSERT INTO aliases(alias,alias_norm,node_id,origin) VALUES (?,?,?,'agent')",
            (a, a.casefold(), nid),
        )
    return nid


def _counts(store):
    return [
        store.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        for t in ("nodes", "edges", "chunks", "aliases")
    ]


def _intervals(store):
    return {
        f"m{i}": (vf[:10], vt and vt[:10])
        for i, vf, vt in store.conn.execute(
            "SELECT id, valid_from, valid_to FROM nodes WHERE kind='memory' ORDER BY id"
        )
    }


# ---- resolver --------------------------------------------------------------------------------


def test_two_anas_ask(store):
    a, b = (
        _entity(store, "Ana Silva", "person", "ana"),
        _entity(store, "Ana Lopez", "person", "ana"),
    )
    r = _rem(store, "Ana likes tea.", ["Ana"])
    assert r["status"] == "needs_choice"
    assert sorted(r["candidates"]) == [f"e{a} · person · Ana Silva", f"e{b} · person · Ana Lopez"]


def test_new_longer_name_while_short_exists_asks(store):
    ana = _entity(store, "Ana", "person")
    before = _counts(store)
    r = _rem(store, "Ana Lopez leads infra.", ["Ana Lopez", "new:topic:Infra"])
    assert r["status"] == "needs_choice"
    assert r["candidates"] == [f"e{ana} · person · Ana"]
    assert _counts(store) == before  # not even the other new subject


def test_short_name_vs_longer_existing_asks(store):
    _entity(store, "Ana Silva", "person")
    assert _rem(store, "Ana runs the guild.", ["Ana"])["status"] == "needs_choice"


def test_typo_links_without_creating(store):
    ana = _entity(store, "Ana Silva", "person")
    r = _rem(store, "Ana Silva is on leave.", ["Ana Silvaa"])
    assert r["status"] == "saved"
    assert r["subjects"] == [f"e{ana} Ana Silva (matched)"]
    assert store.conn.execute("SELECT count(*) FROM nodes WHERE kind='entity'").fetchone() == (1,)


def test_typo_with_a_second_near_candidate_asks(store):
    _entity(store, "Ana Silva", "person")
    _entity(store, "Ana Silvo", "person")
    assert _rem(store, "Ana is on leave.", ["Ana Silvaa"])["status"] == "needs_choice"


def test_short_mention_is_never_a_typo_match(store):
    _entity(store, "Ana", "person")
    assert _rem(store, "Anas is on leave.", ["Anas"])["status"] == "needs_choice"


def test_cross_kind_collision_asks(store):
    p, o = _entity(store, "Atlas", "project"), _entity(store, "Atlas", "org")
    r = _rem(store, "Atlas ships in March.", ["Atlas"])
    assert r["status"] == "needs_choice"
    assert sorted(r["candidates"]) == [f"e{p} · project · Atlas", f"e{o} · org · Atlas"]


def test_unrelated_name_creates_a_topic(store):
    _entity(store, "Ana Silva", "person")
    r = _rem(store, "Zebra crossings are striped.", ["Zebra"])
    assert r["status"] == "saved" and r["subjects"][0].endswith("Zebra (new)")
    assert store.conn.execute("SELECT type FROM nodes WHERE name='Zebra'").fetchone() == ("topic",)


def test_new_entities_get_a_name_chunk(store):
    _rem(store, "Zebra crossings are striped.", ["new:topic:Zebra"])
    row = store.conn.execute(
        "SELECT c.key_kind, c.chunk_key, c.text FROM chunks c JOIN nodes n ON n.id=c.node_id"
        " WHERE n.name='Zebra'"
    ).fetchone()  # fmt: skip
    assert row == ("name", "name", "Zebra")


def test_needs_choice_writes_nothing(store):
    a = _entity(store, "Ana Silva", "person", "ana")
    _entity(store, "Ana Lopez", "person", "ana")
    before = _counts(store)
    r = _rem(
        store, "Ana likes tea.", [f"Anita → #e{a}", "ana", "new:topic:Tea"], as_of="2026-01-01"
    )
    assert r["status"] == "needs_choice" and r["id"] is None
    assert _counts(store) == before  # no node, edge, chunk, vector row or learned alias
    assert store.conn.execute(
        "SELECT count(*) FROM aliases WHERE alias_norm='anita'"
    ).fetchone() == (0,)


def test_mention_alias_is_learned_and_resolves_next_time(store):
    ana = _entity(store, "Ana Silva", "person")
    _rem(store, "Ana Silva is on leave.", [f"Anushka → #e{ana}"])
    r = _rem(store, "Anushka is back.", ["Anushka"])
    assert r["subjects"] == [f"e{ana} Ana Silva (matched)"]


# ---- `new:<type>:<Name>` against existing names (ADR 0027) -----------------------------------


def _entities(store):
    return store.conn.execute("SELECT count(*) FROM nodes WHERE kind='entity'").fetchone()[0]


def test_new_name_of_another_type_asks_naming_the_entity(store):
    juniper = _entity(store, "Juniper", "place")
    before = _counts(store)
    r = _rem(store, "The offsite is in Juniper.", ["new:topic:Juniper", "new:topic:Offsite"])
    assert r["status"] == "needs_choice" and r["id"] is None
    assert r["candidates"] == [f"e{juniper} · place · Juniper"]
    assert _counts(store) == before  # not even the other new subject
    text = store.remember(text="The offsite is in Juniper.", type="fact",
                          about=["new:topic:Juniper"]).text  # fmt: skip
    assert f"e{juniper} · place · Juniper" in text and "distinct_new" in text


def test_resend_with_the_id_links_the_existing_entity(store):
    juniper = _entity(store, "Juniper", "place")
    _rem(store, "The offsite is in Juniper.", ["new:topic:Juniper"])
    r = _rem(store, "The offsite is in Juniper.", [f"#e{juniper}"])
    assert r["status"] == "saved" and r["subjects"] == [f"e{juniper} Juniper (matched)"]
    assert _entities(store) == 1


def test_new_name_of_another_type_matches_case_insensitively(store):
    juniper = _entity(store, "Juniper", "place")
    r = _rem(store, "Juniper is cold.", ["new:thing:  JUNIPER "])
    assert r["candidates"] == [f"e{juniper} · place · Juniper"]


def test_new_same_type_reuses_the_entity(store):
    juniper = _entity(store, "Juniper", "place")
    r = _rem(store, "The offsite is in Juniper.", ["new:place:juniper"])
    assert r["status"] == "saved" and r["subjects"] == [f"e{juniper} Juniper (matched)"]
    assert _entities(store) == 1


def test_new_alias_hit_same_type_reuses_other_type_asks(store):
    juniper = _entity(store, "Juniper", "place", "Juni")
    r = _rem(store, "Juni is cold.", ["new:place:Juni"])
    assert r["status"] == "saved" and r["subjects"] == [f"e{juniper} Juniper (matched)"]
    r = _rem(store, "Juni is far.", ["new:topic:Juni"])
    assert r["status"] == "needs_choice"
    assert r["candidates"] == [f"e{juniper} · place · Juniper"]
    assert _entities(store) == 1


def test_new_name_with_two_hits_asks_with_both(store):
    p, o = _entity(store, "Atlas", "project"), _entity(store, "Atlas", "org")
    r = _rem(store, "Atlas ships in March.", ["new:topic:Atlas"])
    assert r["status"] == "needs_choice"
    assert sorted(r["candidates"]) == sorted([f"e{p} · project · Atlas", f"e{o} · org · Atlas"])


def test_new_name_with_one_same_type_hit_among_several_reuses_it(store):
    p, _ = _entity(store, "Atlas", "project"), _entity(store, "Atlas", "org")
    r = _rem(store, "Atlas ships in March.", ["new:project:Atlas"])
    assert r["status"] == "saved" and r["subjects"] == [f"e{p} Atlas (matched)"]


def test_new_name_without_a_hit_is_created(store):
    _entity(store, "Juniper", "place")
    r = _rem(store, "Fenove is a board game.", ["new:thing:Fenove"])
    assert r["status"] == "saved" and r["subjects"][0].endswith("Fenove (new)")
    assert store.conn.execute(
        "SELECT type FROM nodes WHERE kind='entity' AND name='Fenove'"
    ).fetchone() == ("thing",)


def test_distinct_new_creates_next_to_another_type_but_still_reuses_same_type(store):
    juniper = _entity(store, "Juniper", "place")
    r = _rem(store, "Juniper the band split up.", ["new:topic:Juniper"], distinct_new=True)
    assert r["status"] == "saved" and r["subjects"][0].endswith("Juniper (new)")
    topic = int(r["subjects"][0].split()[0][1:])
    assert topic != juniper and _entities(store) == 2
    r = _rem(store, "Juniper the town has a lake.", ["new:place:Juniper"], distinct_new=True)
    assert r["subjects"] == [f"e{juniper} Juniper (matched)"]
    # with both now present, a third type asks with both
    r = _rem(store, "Juniper again.", ["new:thing:Juniper"])
    assert sorted(r["candidates"]) == sorted([f"e{juniper} · place · Juniper",
                                              f"e{topic} · topic · Juniper"])  # fmt: skip


def test_distinct_new_does_not_skip_an_ambiguous_bare_name(store):
    _entity(store, "Ana Silva", "person", "ana")
    _entity(store, "Ana Lopez", "person", "ana")
    assert _rem(store, "Ana likes tea.", ["Ana"], distinct_new=True)["status"] == "needs_choice"


# ---- conflict candidates ---------------------------------------------------------------------


def test_same_subject_conflict_surfaces_with_hints(cstore):
    old = _rem(cstore, "Alice weekly 1:1s with Ana are on Mondays.", ["new:person:Ana"],
               type="preference", as_of="2026-06-03")  # fmt: skip
    _rem(cstore, "Ana likes green tea.", ["Ana"])
    new = _rem(cstore, "Alice weekly 1:1s with Ana move to Thursdays.", ["Ana"],
               type="preference", as_of="2026-11-01")  # fmt: skip
    assert [c["id"] for c in new["possible_conflicts"]] == [old["id"]]
    c = new["possible_conflicts"][0]
    assert c["score"] == pytest.approx(1.0) and c["hint"] == "same_subjects, older"
    assert c["row"].startswith(f"{old['id']} · preference · 2026-06-03 · ")
    assert (
        "possible_conflicts:"
        in cstore.remember(
            text="Alice weekly 1:1s with Ana move to Fridays.", type="preference", about=["Ana"]
        ).text
    )


def test_conflicts_below_theta_conf_are_dropped(cstore):
    _rem(cstore, "Ana likes green tea.", ["new:person:Ana"])
    assert _rem(cstore, "Ana owns the billing migration.", ["Ana"])["possible_conflicts"] == []


def test_global_candidates_cover_other_subjects(cstore):
    other = _rem(
        cstore, "The billing migration ships in March.", ["new:project:Billing"], as_of="2026-12-01"
    )
    new = _rem(cstore, "The billing migration ships in April.", ["new:person:Ana"])
    c = new["possible_conflicts"][0]
    assert (c["id"], c["hint"]) == (other["id"], "newer")


def test_possible_conflicts_capped_at_five_best_first(cstore):
    for i in range(7):
        _rem(cstore, f"Ana desk is on floor {i}.", ["new:person:Ana"] if i == 0 else ["Ana"])
    new = _rem(cstore, "Ana desk is on floor 9.", ["Ana"])
    scores = [c["score"] for c in new["possible_conflicts"]]
    assert len(scores) == 5 and scores == sorted(scores, reverse=True)


def test_old_attribute_update_after_60_same_subject_writes_surfaces(cstore):
    old = _rem(cstore, "Ana Silva's 1:1 happens on Mondays at ten.", ["new:person:Ana Silva"])
    for i in range(60):
        _rem(cstore, f"Ana Silva reviewed ticket {1000 + i} today.", ["Ana Silva"])
    new = _rem(cstore, "Ana Silva's 1:1 moved to Thursdays at ten.", ["Ana Silva"])
    assert old["id"] in [c["id"] for c in new["possible_conflicts"]]


def test_no_conflicts_without_a_reranker_or_theta(store):
    _rem(store, "Ana desk is on floor 1.", ["new:person:Ana"])
    assert _rem(store, "Ana desk is on floor 2.", ["Ana"])["possible_conflicts"] == []


# ---- two-phase write -------------------------------------------------------------------------


def test_two_clients_same_memory_one_row(tmp_path):
    path = tmp_path / "shared.db"
    Store.open(path, HashEmbedder()).close()
    barrier, out, errors = threading.Barrier(2), [], []

    def client():
        try:
            s = Store.open(path, HashEmbedder())
            barrier.wait()
            out.append(_rem(s, "The vault lives on the NAS.", ["new:topic:Vault"]))
            s.close()
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=client) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sorted(r["status"] for r in out) == ["exists", "saved"]
    assert out[0]["id"] == out[1]["id"]
    s = Store.open(path, HashEmbedder())
    assert s.conn.execute("SELECT count(*) FROM nodes WHERE kind='memory'").fetchone() == (1,)
    assert s.conn.execute("SELECT count(*) FROM nodes WHERE kind='entity'").fetchone() == (1,)
    s.close()


# ---- inline supersedes and chains ------------------------------------------------------------


def test_inline_supersedes_closes_the_old_interval(store):
    old = _rem(store, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")
    new = _rem(store, "Ana desk is on floor 5.", ["Ana"], as_of="2026-06-01", supersedes=old["id"])
    assert _intervals(store) == {old["id"]: ("2026-01-01", "2026-06-01"),
                                 new["id"]: ("2026-06-01", None)}  # fmt: skip
    assert store.conn.execute("SELECT src, dst FROM edges WHERE rel='supersedes'").fetchone() == (
        int(new["id"][1:]), int(old["id"][1:]),
    )  # fmt: skip


def test_inline_supersedes_unknown_target_rolls_everything_back(store):
    before = _counts(store)
    with pytest.raises(MemError, match="m99"):
        _rem(store, "Ana desk is on floor 5.", ["new:person:Ana"], supersedes="m99")
    assert _counts(store) == before


def test_inline_supersedes_needs_a_memory(store):
    ana = _rem(store, "Ana desk is on floor 2.", ["new:person:Ana"])["subjects"][0].split()[0]
    with pytest.raises(MemError, match="memory"):
        _rem(store, "Ana desk is on floor 5.", ["Ana"], supersedes=ana)


def test_late_arriving_older_fact_rederives_chain(store):
    m1 = _rem(store, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")["id"]
    m2 = _rem(store, "Ana desk is on floor 5.", ["Ana"], as_of="2026-06-01", supersedes=m1)["id"]
    m3 = _rem(store, "Ana desk is on floor 3.", ["Ana"], as_of="2026-03-01", supersedes=m1)["id"]
    assert _intervals(store) == {m1: ("2026-01-01", "2026-03-01"),
                                 m2: ("2026-06-01", None),
                                 m3: ("2026-03-01", "2026-06-01")}  # fmt: skip


def test_supersedes_with_an_older_new_fact_swaps_direction(store):
    m1 = _rem(store, "Ana desk is on floor 5.", ["new:person:Ana"], as_of="2026-06-01")["id"]
    res = store.remember(text="Ana desk was on floor 2.", type="fact", about=["Ana"],
                         as_of="2026-01-01", supersedes=m1)  # fmt: skip
    m2 = res.structured["id"]
    assert "direction swapped" in res.text
    assert _intervals(store) == {m1: ("2026-06-01", None), m2: ("2026-01-01", "2026-06-01")}
    assert store.conn.execute("SELECT src, dst FROM edges WHERE rel='supersedes'").fetchone() == (
        int(m1[1:]), int(m2[1:]),
    )  # fmt: skip


def test_same_instant_correction_gets_empty_interval(store):
    m1 = _rem(store, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-11-01")["id"]
    m2 = _rem(store, "Ana desk is on floor 4.", ["Ana"], as_of="2026-11-01", supersedes=m1)["id"]
    assert _intervals(store) == {m1: ("2026-11-01", "2026-11-01"), m2: ("2026-11-01", None)}
    cur = store.recall(query="Ana desk floor", as_of="2026-11-02")
    assert [r["id"] for r in cur.structured["rows"]] == [m2]


def test_resave_superseded_text_is_saved_not_exists(store):
    m1 = _rem(store, "Ana desk is on floor 2.", ["new:person:Ana"], as_of="2026-01-01")["id"]
    m2 = _rem(store, "Ana desk is on floor 5.", ["Ana"], as_of="2026-06-01", supersedes=m1)["id"]
    again = _rem(store, "Ana desk is on floor 2.", ["Ana"], as_of="2026-09-01", supersedes=m2)
    assert again["status"] == "saved" and again["id"] not in (m1, m2)
    assert _intervals(store) == {m1: ("2026-01-01", "2026-06-01"),
                                 m2: ("2026-06-01", "2026-09-01"),
                                 again["id"]: ("2026-09-01", None)}  # fmt: skip


def test_same_instant_conflict_hint_uses_n1_vocabulary(cstore):
    old = _rem(cstore, "Alice weekly 1:1s with Ana are on Mondays.", ["new:person:Ana"],
               type="preference", as_of="2026-06-03")  # fmt: skip
    new = _rem(cstore, "Alice weekly 1:1s with Ana move to Thursdays.", ["Ana"],
               type="preference", as_of="2026-06-03")  # fmt: skip
    c = new["possible_conflicts"][0]
    assert (c["id"], c["hint"]) == (old["id"], "same_subjects, same instant")


# ---- ingest.new_checks_existing off: the pre-ADR-0027 `new:` rule (MEM2-272) -----------------


@pytest.fixture
def old_new(make_store):
    return make_store(ingest={"resolve_dates": True, "new_checks_existing": False})


def test_switch_off_new_name_next_to_another_type_creates_a_second_node(old_new):
    juniper = _entity(old_new, "Juniper", "place")
    r = _rem(old_new, "The offsite is in Juniper.", ["new:topic:Juniper"])
    assert r["status"] == "saved" and r["subjects"][0].endswith("Juniper (new)")
    assert int(r["subjects"][0].split()[0][1:]) != juniper and _entities(old_new) == 2


def test_switch_off_same_type_name_is_reused_but_aliases_are_not_consulted(old_new):
    juniper = _entity(old_new, "Juniper", "place", "Juni")
    r = _rem(old_new, "The offsite is in Juniper.", ["new:place:JUNIPER"])
    # case-insensitive; the old rule echoes the name as written
    assert r["subjects"] == [f"e{juniper} JUNIPER (matched)"]
    r = _rem(old_new, "Juni is cold.", ["new:place:Juni"])
    assert r["status"] == "saved" and r["subjects"][0].endswith("Juni (new)")
    assert _entities(old_new) == 2


def test_switch_off_owned_entity_is_not_reused(old_new):
    juniper = _entity(old_new, "Juniper", "place")
    old_new.conn.execute(
        "UPDATE nodes SET source_ref = 'vault:Juniper.md' WHERE id = ?", (juniper,)
    )
    r = _rem(old_new, "The offsite is in Juniper.", ["new:place:Juniper"])
    assert r["subjects"][0].endswith("Juniper (new)") and _entities(old_new) == 2


def test_switch_off_distinct_new_is_accepted_and_ignored(old_new):
    juniper = _entity(old_new, "Juniper", "place")
    r = _rem(old_new, "Juniper the town.", ["new:place:Juniper"], distinct_new=True)
    assert r["subjects"] == [f"e{juniper} Juniper (matched)"]
    r = _rem(old_new, "Juniper the band.", ["new:topic:Juniper"], distinct_new=True)
    assert r["status"] == "saved" and _entities(old_new) == 2


def test_switch_on_explicitly_is_the_adr_0027_rule(make_store):
    s = make_store(ingest={"resolve_dates": True, "new_checks_existing": True})
    juniper = _entity(s, "Juniper", "place")
    r = _rem(s, "The offsite is in Juniper.", ["new:topic:Juniper"])
    assert r["status"] == "needs_choice" and r["candidates"] == [f"e{juniper} · place · Juniper"]
