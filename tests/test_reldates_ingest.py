"""ADR 0021: `ingest.resolve_dates` stores the derived annotation next to the chunk at import and
ingest, indexes it for FTS, shows it in recall; off, the store and recall are as before."""

import sqlite3

import pytest

from mem import config, read, schema, write
from mem.importer import run_import
from mem.testing import HashEmbedder

ON = {"resolve_dates": True}
SESSION = [("I went hiking yesterday and sat by the lake.", "Sounds lovely.")]


def _dates(store):
    return [r[0] for r in store.conn.execute("SELECT dates FROM chunks ORDER BY id")]


def _texts(store):
    return [r[0] for r in store.conn.execute("SELECT text FROM chunks ORDER BY id")]


def _fts(store, query):
    store.conn.execute("CREATE TEMP TABLE IF NOT EXISTS allowed(chunk_id INTEGER PRIMARY KEY)")
    store.conn.execute("DELETE FROM allowed")
    store.conn.execute("INSERT INTO allowed SELECT id FROM chunks")
    return read.fts_leg(store.conn, query, 10)


# ---- config ----------------------------------------------------------------------------------


@pytest.mark.release_defaults
def test_resolve_dates_defaults_on():
    """On by default since MEM2-261 (F1 adopted, Round 9); off, the ingest section is outside `fp_all`."""
    cfg = config.default_config()
    assert cfg.ingest == {"resolve_dates": True} and cfg.resolve_dates is True


def test_resolve_dates_on_is_a_new_fingerprint(tmp_path):
    base = config.default_config().replace("ingest", resolve_dates=False)
    on = base.replace("ingest", resolve_dates=True)
    assert on.resolve_dates is True
    assert on.fp_all() != base.fp_all() and on.fp_embed() == base.fp_embed()
    p = tmp_path / "c.yaml"
    p.write_text("ingest:\n  resolve_dates: true\n")
    assert config.load(p).fp_all() == on.fp_all()
    p.write_text("ingest:\n  resolve_dates: yes please\n")
    with pytest.raises(config.ConfigError, match="true or false"):
        config.load(p)
    p.write_text("ingest:\n  other: 1\n")
    with pytest.raises(config.ConfigError, match="unknown field"):
        config.load(p)


# ---- ingest_session ----------------------------------------------------------------------------


def test_on_annotates_the_chunk_and_leaves_the_text_verbatim(make_store):
    store = make_store(ingest=ON)
    write.ingest_session(store, source_ref="s1", date="2023-03-10", rounds=SESSION)
    assert _dates(store) == ["yesterday [= 2023-03-09]"]
    assert _texts(store) == ["I went hiking yesterday and sat by the lake.\nSounds lovely."]


def test_on_annotation_is_found_by_fts_and_off_it_is_not(make_store):
    on, off = make_store("personal", ingest=ON), make_store("work")
    for s in (on, off):
        write.ingest_session(s, source_ref="s1", date="2023-03-10", rounds=SESSION)
    assert _fts(on, "09") == [1] and _fts(off, "09") == []


def test_recall_row_shows_the_annotation_on_and_nothing_off(make_store):
    on, off = make_store("personal", ingest=ON), make_store("work")
    for s in (on, off):
        write.ingest_session(s, source_ref="s1", date="2023-03-10", rounds=SESSION)
    res_on, res_off = on.recall(query="hiking lake"), off.recall(query="hiking lake")
    # stored as `yesterday [= 2023-03-09]`, shown absolute first (MEM2-254)
    assert res_on.structured["rows"][0]["dates"] == "2023-03-09 (yesterday)"
    assert "dates: 2023-03-09 (yesterday)" in res_on.text
    assert "dates" not in res_off.structured["rows"][0] and "dates:" not in res_off.text


def test_off_recall_is_the_output_it_was_before(make_store):
    store = make_store()
    write.ingest_session(store, source_ref="s1", date="2023-03-10", rounds=SESSION)
    res = store.recall(query="hiking lake")
    assert list(res.structured["rows"][0]) == [
        "id", "kind", "type", "date", "title", "about", "p", "flags"]  # fmt: skip
    assert res.text.splitlines()[1].startswith("c1 · chat_session · 2023-03-10 · chat session 2023-03-10")  # fmt: skip
    assert _dates(store) == [None]


def test_the_session_date_is_read_in_the_stores_timezone(make_store):
    # 2023-03-11 04:30 UTC is the evening of the 10th in New York: "yesterday" is the 9th.
    store = make_store(ingest=ON, tz="America/New_York")
    write.ingest_session(store, source_ref="s1", date="2023-03-11T04:30:00Z", rounds=SESSION)
    assert _dates(store) == ["yesterday [= 2023-03-09]"]
    utc = make_store("work", ingest=ON)
    write.ingest_session(utc, source_ref="s1", date="2023-03-11T04:30:00Z", rounds=SESSION)
    assert _dates(utc) == ["yesterday [= 2023-03-10]"]


def test_a_round_without_an_expression_gets_no_annotation(make_store):
    store = make_store(ingest=ON)
    write.ingest_session(
        store, source_ref="s1", date="2023-03-10", rounds=[("Hello there.", "Hi.")]
    )
    assert _dates(store) == [None]


# ---- remember ----------------------------------------------------------------------------------


def test_remember_annotates_against_as_of_only(make_store):
    store = make_store(ingest=ON)
    store.remember(text="Ana moved flats yesterday.", type="fact", about=["new:person:Ana"],
                   as_of="2023-03-10")  # fmt: skip
    store.remember(text="Ana adopted a cat yesterday.", type="fact", about=["new:person:Ana"])
    bodies = store.conn.execute("SELECT dates FROM chunks WHERE key_kind = 'body' ORDER BY id")
    assert [r[0] for r in bodies] == ["yesterday [= 2023-03-09]", None]


# ---- import ------------------------------------------------------------------------------------


def _vault(tmp_path):
    vault = tmp_path / "vault"
    (vault / "personal").mkdir(parents=True)
    (vault / "personal" / "Trip.md").write_text(
        "---\ncreated: 2023-03-10\n---\n# Trip\n\nWe hiked yesterday.\n\n## Plans\n\nSwim next Monday.\n",
        encoding="utf-8")  # fmt: skip
    return vault


def _import(vault, store):
    return run_import(vault, {"personal": store}, routes={})


def _chunk_state(store):
    return store.conn.execute(
        "SELECT id, chunk_key, text, dates FROM chunks ORDER BY id").fetchall()  # fmt: skip


def test_import_annotates_by_the_note_date_and_is_idempotent(make_store, tmp_path):
    store = make_store(ingest=ON)
    vault = _vault(tmp_path)
    _import(vault, store)
    first = _chunk_state(store)
    assert [r[3] for r in first] == ["yesterday [= 2023-03-09]; next Monday [= 2023-03-13]"]
    assert "We hiked yesterday." in " ".join(r[2] for r in first)
    report = _import(vault, store)
    assert report.unchanged == 1 and _chunk_state(store) == first


def test_a_stored_annotation_of_another_value_is_rewritten_on_re_import(make_store, tmp_path):
    """Unchanged means the same annotations, values included, not just the same annotated keys."""
    store = make_store(ingest=ON)
    vault = _vault(tmp_path)
    _import(vault, store)
    first = _chunk_state(store)
    store.conn.execute(
        "UPDATE chunks SET dates = 'yesterday [= 1999-01-01]' WHERE dates IS NOT NULL"
    )
    assert _import(vault, store).changed == 1 and _chunk_state(store) == first


def test_import_with_the_switch_off_stores_nothing(make_store, tmp_path):
    store = make_store()
    _import(_vault(tmp_path), store)
    assert {r[3] for r in _chunk_state(store)} == {None}


def test_switching_off_and_re_importing_drops_the_annotations_and_back_on_restores_them(
    tmp_path, make_store,
):  # fmt: skip
    vault = _vault(tmp_path)
    store = make_store(ingest=ON)
    _import(vault, store)
    annotated = _chunk_state(store)
    store.close()
    off = _reopen(tmp_path, None)
    assert _import(vault, off).changed == 1
    assert {r[3] for r in _chunk_state(off)} == {None}
    assert [r[:3] for r in _chunk_state(off)] == [r[:3] for r in annotated]  # ids and text kept
    assert _fts(off, "09") == []
    off.close()
    on = _reopen(tmp_path, ON)
    assert _import(vault, on).changed == 1 and _chunk_state(on) == annotated
    assert _fts(on, "09") != []
    on.close()


def _reopen(tmp_path, ingest):
    from mem.store import Store
    from tests.conftest import Clock

    return Store.open(tmp_path / "personal.db", HashEmbedder(), scope="personal", now=Clock(),
                      ingest=ingest)  # fmt: skip


# ---- schema ------------------------------------------------------------------------------------


def test_migration_adds_dates_to_an_existing_store_and_keeps_its_index(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    for number, sql in schema._migrations():
        if number <= 2:
            conn.executescript(f"BEGIN;\n{sql}\nUPDATE meta SET value = '{number}' "
                               "WHERE key = 'schema_version';\nCOMMIT;")  # fmt: skip
    conn.execute("INSERT INTO nodes(kind, name, valid_from, recorded_at, origin, content_hash)"
                 " VALUES ('note', 'n', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 'import', 'h')")  # fmt: skip
    conn.execute(
        "INSERT INTO chunks(node_id, chunk_key, ord, key_kind, ctx, text, tokens,"
        " content_hash) VALUES (1, 'a', 0, 'section', 'ctx', 'marmalade jar', 2, 'x')"
    )
    conn.commit()
    assert schema.migrate(conn) == 3
    assert conn.execute("SELECT dates FROM chunks").fetchall() == [(None,)]
    assert conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'marmalade'"
    ).fetchall() == [(1,)]
    conn.execute("UPDATE chunks SET dates = 'yesterday [= 2023-03-09]'")
    assert conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH '\"09\"'"
    ).fetchall() == [(1,)]
