import sqlite3

import pytest

from mem import schema
from mem.common import MemError

TABLES = {
    "meta", "nodes", "rels", "edges", "chunks", "link_occurrences", "aliases", "aliases_tri",
    "chunks_fts", "props", "source_files", "events",
}  # fmt: skip
TRIGGERS = {"chunks_ai", "chunks_ad", "chunks_au", "aliases_ai", "aliases_ad", "aliases_au"}


def _names(conn, kind):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,)).fetchall()
    return {r[0] for r in rows}


def _node(conn, name, **kw):
    cols = {
        "kind": "memory", "name": name, "valid_from": "2026-01-01T00:00:00Z",
        "recorded_at": "2026-01-01T00:00:00Z", "origin": "user", "content_hash": name,
    } | kw  # fmt: skip
    cur = conn.execute(
        f"INSERT INTO nodes({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        list(cols.values()),
    )
    return cur.lastrowid


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    yield c
    c.close()


def test_schema_v2_creates_tables_and_triggers(conn):
    assert TABLES <= _names(conn, "table")
    assert TRIGGERS <= _names(conn, "trigger")
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("3",)
    assert schema.schema_version(conn) == 3
    assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)


def test_rels_seeded(conn):
    rels = {r[0] for r in conn.execute("SELECT rel FROM rels")}
    assert {"about", "supersedes", "duplicate_of", "same_as", "relates_to", "links_to"} <= rels
    assert {"embeds", "tagged", "part_of"} <= rels


def test_migrate_is_idempotent(tmp_path):
    p = tmp_path / "t.db"
    schema.connect(p).close()
    c = schema.connect(p)
    assert schema.schema_version(c) == 3
    assert c.execute("SELECT count(*) FROM rels WHERE rel='about'").fetchone() == (1,)


def test_exact_duplicate_active_open_memory_rejected(conn):
    _node(conn, "a", content_hash="h")
    with pytest.raises(sqlite3.IntegrityError):
        _node(conn, "b", content_hash="h")


def test_superseded_text_may_be_saved_again(conn):
    _node(conn, "a", content_hash="h", valid_to="2026-02-01T00:00:00Z")
    _node(conn, "b", content_hash="h")


def test_self_link_rejected(conn):
    n = _node(conn, "a")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES (?,?,?,?,?)",
            (n, n, "relates_to", "user", "2026-01-01T00:00:00Z"),
        )


def test_valid_to_before_valid_from_rejected_equal_allowed(conn):
    _node(conn, "a", valid_to="2026-01-01T00:00:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        _node(conn, "b", valid_to="2025-12-31T00:00:00Z")


def test_chunk_fts_trigger_and_cascade_to_occurrences(conn):
    n = _node(conn, "a")
    conn.execute(
        "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,text,tokens,content_hash)"
        " VALUES (?,?,?,?,?,?,?)",
        (n, "body", 0, "body", "weekly one to one", 4, "x"),
    )
    cid = conn.execute("SELECT id FROM chunks").fetchone()[0]
    hit = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
    assert conn.execute(hit, ("weekly",)).fetchall() == [(cid,)]
    m = _node(conn, "b", content_hash="b")
    conn.execute(
        "INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES (?,?,?,?,?)",
        (n, m, "links_to", "import", "2026-01-01T00:00:00Z"),
    )
    conn.execute("INSERT INTO link_occurrences(edge_id,chunk_id,pos) VALUES (1,?,3)", (cid,))
    conn.execute("INSERT INTO link_occurrences(edge_id,chunk_id,pos) VALUES (1,?,9)", (cid,))
    conn.execute("DELETE FROM chunks WHERE id=?", (cid,))
    assert conn.execute("SELECT count(*) FROM link_occurrences").fetchone() == (0,)
    assert conn.execute(hit, ("weekly",)).fetchall() == []


def test_alias_update_reindexes_trigram(conn):
    n = _node(conn, "a")
    conn.execute(
        "INSERT INTO aliases(alias,alias_norm,node_id,origin) VALUES ('alpha','alpha',?,'user')",
        (n,),
    )
    conn.execute("UPDATE aliases SET alias_norm='omega'")
    q = "SELECT count(*) FROM aliases_tri WHERE aliases_tri MATCH ?"
    assert conn.execute(q, ("omega",)).fetchone() == (1,)
    assert conn.execute(q, ("alpha",)).fetchone() == (0,)


def test_vec_table_per_fingerprint(conn):
    name = schema.create_vec_table(conn, "abcdef0123456789", 4)
    assert name == "vec_abcdef01"
    conn.execute(f"INSERT INTO {name}(chunk_id, embedding) VALUES (1, ?)", ("[1,0,0,0]",))
    row = conn.execute(
        f"SELECT chunk_id FROM {name} WHERE embedding MATCH ? AND k = 1", ("[1,0,0,0]",)
    ).fetchone()
    assert row == (1,)


def test_a_store_newer_than_the_code_is_refused_not_opened(tmp_path):
    # MEM2-222 F21: older code must not open (and write) a store a newer migration has touched
    p = tmp_path / "t.db"
    c = schema.connect(p)
    newer = schema.latest_version() + 1
    c.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(newer),))
    c.commit()
    c.close()
    with pytest.raises(MemError, match=f"schema version {newer} is newer"):
        schema.connect(p)
    raw = sqlite3.connect(p)
    assert raw.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == (
        str(newer),
    )
    raw.close()
