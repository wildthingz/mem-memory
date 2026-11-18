"""P4 importer contract on the mini-vault fixture (architecture-v2 §2.4, roadmap P4 gate; routing
decision docs/decisions/0010). These tests are written before `mem.importer`: they fail until it
exists, and the expected dump (fixtures/minivault/expected_s1.json) is worked out by hand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from mem import config
from tests.minivault import FIXTURE, dump_store, expected, make_vault

TRIP = "personal/2026-03-02_Trip planning.md"
TRIP_RENAMED = "personal/2026-03-02_Trip plan.md"
MEETING = "personal/2026-04-10_Meeting with Ana.md"
ATLAS = "projects/Atlas.md"
ESSAY = "personal/Long essay.md"
ANA = "personal/people/Ana Silva.md"

INDEXED_FILES = 10  # 7 notes/entities in personal.db, 3 in work.db (see expected_s1.json)


@pytest.fixture
def stores(make_store):
    return {"personal": make_store("personal"), "work": make_store("work")}


@pytest.fixture
def vault(tmp_path) -> Path:
    return make_vault(tmp_path)


@pytest.fixture
def routes() -> dict[str, str]:
    return yaml.safe_load((FIXTURE / "config.yaml").read_text())["vault"]["routes"]


def run(vault, stores, routes=None):
    from mem.importer import run_import

    return run_import(vault, stores, routes=routes or {})


def counts(report) -> dict[str, int]:
    return {k: getattr(report, k) for k in
            ("added", "changed", "renamed", "deleted", "restored", "unchanged")}  # fmt: skip


def node_id(store, ref: str) -> int:
    return store.conn.execute("SELECT id FROM nodes WHERE source_ref = ?", (ref,)).fetchone()[0]


def chunk_ids(store, ref: str) -> dict[str, int]:
    return dict(store.conn.execute(
        "SELECT chunk_key, id FROM chunks WHERE node_id = ?", (node_id(store, ref),)))  # fmt: skip


def assert_dump(actual: dict[str, Any], want: dict[str, Any]) -> None:
    assert actual["nodes"] == want["nodes"]
    assert actual["props"] == want["props"]
    assert actual["chunks"] == want["chunks"]
    assert actual["edges"] == want["edges"]
    # Aliases also hold the filename stem, whose exact form the spec leaves open: the expected
    # list is what must be there, and nothing else may appear.
    assert set(actual["aliases"]) == set(want["aliases"])
    for ref, names in want["aliases"].items():
        assert set(names) <= set(actual["aliases"][ref]), ref


def test_config_yaml_accepts_vault_routes():
    assert config.load(FIXTURE / "config.yaml").vault_routes == {"projects/client-x": "work"}


def test_import_matches_expected_dump(vault, stores, routes):
    report = run(vault, stores, routes)
    assert counts(report) == {"added": INDEXED_FILES, "changed": 0, "renamed": 0, "deleted": 0,
                              "restored": 0, "unchanged": 0}  # fmt: skip
    want = expected("expected_s1.json")
    for name, store in stores.items():
        assert_dump(dump_store(store.conn), want[name])


def test_second_run_on_same_snapshot_reports_only_unchanged(vault, stores, routes):
    run(vault, stores, routes)
    before = {n: dump_store(s.conn) for n, s in stores.items()}
    writes = {n: s.conn.total_changes for n, s in stores.items()}
    report = run(vault, stores, routes)
    assert counts(report) == {"added": 0, "changed": 0, "renamed": 0, "deleted": 0, "restored": 0,
                              "unchanged": INDEXED_FILES}  # fmt: skip
    assert {n: s.conn.total_changes for n, s in stores.items()} == writes  # zero writes
    assert {n: dump_store(s.conn) for n, s in stores.items()} == before


def test_folder_override_wins_over_default_route(vault, stores, routes):
    run(vault, stores, routes)
    kick = "projects/client-x/Kickoff.md"
    assert _refs(stores["work"]) >= {kick}
    assert kick not in _refs(stores["personal"])
    assert ATLAS in _refs(stores["personal"])  # the rest of projects/ keeps its default


def test_without_override_projects_subfolder_goes_to_personal(vault, stores):
    run(vault, stores)
    assert "projects/client-x/Kickoff.md" in _refs(stores["personal"])
    assert "projects/client-x/Kickoff.md" not in _refs(stores["work"])


def _refs(store) -> set[str]:
    return {r[0] for r in store.conn.execute(
        "SELECT source_ref FROM nodes WHERE origin='import' AND source_ref IS NOT NULL")}  # fmt: skip


def test_import_skips_render_dir_and_skip_list(vault, stores, routes):
    run(vault, stores, routes)
    refs = _refs(stores["personal"]) | _refs(stores["work"])
    skipped = ("export/", ".obsidian/", ".smart-env/", ".stfolder/", "Templates/",
               "node_modules/", "CLAUDE.md", "obsidian-instructions.md")  # fmt: skip
    assert not [r for r in refs if r.startswith(skipped)]
    # not-indexed file types are link targets only: no node for a file nothing links to
    assert "personal/tasks.base" not in refs and "personal/sketch.excalidraw" not in refs
    assert {"personal/diagram.png", "personal/board.canvas"} <= refs


def test_comments_stripped_and_callout_type_recorded(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    text, props = s.conn.execute(
        "SELECT text, props FROM chunks WHERE node_id = ? AND chunk_key LIKE '%Agenda#1/p0'",
        (node_id(s, MEETING),),
    ).fetchone()
    assert "private reminder" not in text and "budget number" not in text
    assert "[!warning]" in text  # callouts stay inline
    assert "warning" in props


def test_future_dated_created_is_kept_as_stated(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    date, valid_from = s.conn.execute(
        "SELECT date, valid_from FROM nodes WHERE source_ref = 'personal/2099-01-01_Time capsule.md'"
    ).fetchone()
    assert date == valid_from == "2099-01-01T00:00:00Z"


def test_long_section_splits_by_paragraph_and_tiny_sections_merge(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    essay = [r[0] for r in s.conn.execute(
        "SELECT text FROM chunks WHERE node_id = ? ORDER BY ord", (node_id(s, ESSAY),))]  # fmt: skip
    assert len(essay) == 3 and [len(t.split()) for t in essay] == [
        255,
        250,
        250,
    ]  # + "# Long essay ## Body"
    trip = {r[0]: r[1] for r in s.conn.execute(
        "SELECT chunk_key, text FROM chunks WHERE node_id = ?", (node_id(s, TRIP),))}  # fmt: skip
    assert "Spring trip" in trip["Packing#1/p0"]  # 6-word preamble merged into the next section
    assert "Remember the umbrella." in trip["Budget#1/p0"]  # tiny "Notes" merged into the next
    assert "# not a heading" in trip["Budget#1/p0"]  # a fenced '#' line neither splits nor tags


def test_chunk_ids_stable_for_unchanged_sections(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    before = chunk_ids(s, ANA)
    path = vault / ANA
    path.write_text(
        path.read_text().replace(
            "answers messages within a few hours", "answers messages within one day"
        )
    )
    assert counts(run(vault, stores, routes))["changed"] == 1
    after = chunk_ids(s, ANA)
    assert after["Ana Silva > Role#1/p0"] == before["Ana Silva > Role#1/p0"]
    assert after["Ana Silva > Contact#1/p0"] != before["Ana Silva > Contact#1/p0"]


def test_rename_delete_edit_then_restore(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    trip_id, atlas_id = node_id(s, TRIP), node_id(s, ATLAS)
    essay_chunks = chunk_ids(s, ESSAY)
    atlas_text = (vault / ATLAS).read_text()

    (vault / TRIP).rename(vault / TRIP_RENAMED)  # same content: a rename, not delete + add
    (vault / ATLAS).unlink()
    meeting = vault / MEETING
    text = meeting.read_text()
    text = text.replace("aliases: [Ana sync, Weekly Ana]", "aliases: [Ana sync]")
    text = text.replace("tags: [meeting]\n", "tags: []\n")
    text = text.replace('attendees: ["[[Ana Silva]]", "[[Bruno]]"]', 'attendees: ["[[Ana Silva]]"]')
    text = text.replace("status: draft\n", "")
    meeting.write_text(text)

    report = run(vault, stores, routes)
    assert counts(report) == {"added": 0, "changed": 1, "renamed": 1, "deleted": 1, "restored": 0,
                              "unchanged": INDEXED_FILES - 3}  # fmt: skip

    assert node_id(s, TRIP_RENAMED) == trip_id  # the node id is the rename's continuity
    assert (
        s.conn.execute("SELECT name FROM nodes WHERE id = ?", (trip_id,)).fetchone()[0]
        == "Trip plan"
    )
    assert chunk_ids(s, ESSAY) == essay_chunks  # untouched notes keep their chunk ids
    assert (
        s.conn.execute("SELECT status FROM nodes WHERE id = ?", (atlas_id,)).fetchone()[0]
        == "deleted"
    )

    d = dump_store(s.conn)
    m = d["aliases"][MEETING]
    assert "Weekly Ana" not in m and "Ana sync" in m  # alias removal
    assert d["props"][MEETING] == {"room": "B12"}  # prop removal
    live = {(e["src"], e["rel"], e["dst"]): e["live"] for e in d["edges"]}
    assert live[(MEETING, "tagged", "tag::meeting")] is False  # tag removal
    assert live[(MEETING, "tagged", "tag::meeting/weekly")] is True
    assert live[(MEETING, "attendee", "entity:topic:Bruno")] is False  # frontmatter link removal
    assert live[(MEETING, "links_to", "entity:topic:Bruno")] is True  # the body link stays
    assert live[(MEETING, "project", ATLAS)] is True  # edges to a deleted node are kept
    assert TRIP not in {n["label"] for n in d["nodes"]}

    (vault / ATLAS).write_text(atlas_text)  # delete-then-restore: the same node comes back
    report = run(vault, stores, routes)
    assert counts(report) == {"added": 0, "changed": 0, "renamed": 0, "deleted": 0, "restored": 1,
                              "unchanged": INDEXED_FILES - 1}  # fmt: skip
    assert node_id(s, ATLAS) == atlas_id
    assert (
        s.conn.execute("SELECT status FROM nodes WHERE id = ?", (atlas_id,)).fetchone()[0]
        == "active"
    )


# ---- lifecycle cases of B-I1/B-I2/B-I3, review items A3 (4.5) and A10 ---------------------------


def _write(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


def _vec_ids(store) -> set[int]:
    return {r[0] for r in store.conn.execute(f"SELECT chunk_id FROM {store.vec_table}")}


def _chunk_rows(store) -> set[int]:
    return {r[0] for r in store.conn.execute("SELECT id FROM chunks")}


def test_import_is_deterministic_across_fresh_stores(vault, stores, routes, tmp_path):
    """B-I1: the same vault into two fresh stores gives the same dump and the same vectors."""
    from mem.store import Store
    from mem.testing import HashEmbedder

    run(vault, stores, routes)
    twin = {n: Store.open(tmp_path / f"twin_{n}.db", HashEmbedder(), scope=n) for n in stores}
    try:
        run(vault, twin, routes)
        for name in stores:
            assert dump_store(stores[name].conn) == dump_store(twin[name].conn)

            def vectors(s):
                return dict(s.conn.execute(
                    f"SELECT c.node_id || '/' || c.chunk_key, v.embedding FROM chunks c"
                    f" JOIN {s.vec_table} v ON v.chunk_id = c.id"))  # fmt: skip

            assert vectors(stores[name]) == vectors(twin[name])
    finally:
        for s in twin.values():
            s.close()


def test_touching_mtime_only_writes_nothing(vault, stores, routes):
    run(vault, stores, routes)
    writes = {n: s.conn.total_changes for n, s in stores.items()}
    import os

    for path in vault.rglob("*.md"):
        os.utime(path, (1.9e9, 1.9e9))
    report = run(vault, stores, routes)
    assert counts(report)["unchanged"] == INDEXED_FILES
    assert {n: s.conn.total_changes for n, s in stores.items()} == writes


def test_edit_one_section_replaces_exactly_that_chunk(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    nid = node_id(s, ANA)
    old_hash = s.conn.execute("SELECT content_hash FROM nodes WHERE id = ?", (nid,)).fetchone()[0]
    before = chunk_ids(s, ANA)
    _write(vault / ANA, "from the home office", "from the garden office")
    run(vault, stores, routes)
    after = chunk_ids(s, ANA)
    assert after["Ana Silva > Role#1/p0"] == before["Ana Silva > Role#1/p0"]
    assert after["Ana Silva > Contact#1/p0"] != before["Ana Silva > Contact#1/p0"]
    assert before["Ana Silva > Contact#1/p0"] not in _chunk_rows(s)  # the old row is gone
    assert s.conn.execute("SELECT content_hash FROM nodes WHERE id = ?", (nid,)).fetchone()[0] != old_hash  # fmt: skip
    assert _vec_ids(s) == _chunk_rows(s)  # the old vector went with its chunk, the new one exists
    hits = s.conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'garden'").fetchall()
    assert [h[0] for h in hits] == [after["Ana Silva > Contact#1/p0"]]


def test_stale_chunk_id_reported_gone(vault, stores, routes):
    """The replaced chunk is the one with the highest id: SQLite would hand that id to the new row
    if the old one were deleted first. A stale id must be gone, never another section (§2.3)."""
    run(vault, stores, routes)
    s = stores["work"]
    (top, ref) = s.conn.execute(
        "SELECT c.id, n.source_ref FROM chunks c JOIN nodes n ON n.id = c.node_id"
        " ORDER BY c.id DESC LIMIT 1").fetchone()  # fmt: skip
    path = vault / ref
    path.write_text(path.read_text().replace("The follow-up", "The revised follow-up", 1))
    run(vault, stores, routes)
    assert top not in _chunk_rows(s)
    assert max(_chunk_rows(s)) > top
    result = s.get([f"c{top}"])
    assert result.structured["items"][0]["found"] is False
    assert "no longer exists" in result.text


def test_inserting_a_duplicate_heading_never_reuses_a_stale_chunk_id(vault, stores, routes):
    """A10: a new section above 'Packing#1' shifts the duplicate numbers; the chunk whose key now
    holds other text gets a new id, and its old id is gone rather than pointing at the new text."""
    run(vault, stores, routes)
    s = stores["personal"]
    before = chunk_ids(s, TRIP)
    _write(vault / TRIP, "## Packing\n\nPack light",
           "## Packing\n\nA brand new first packing section about shoes, socks and the "
           "weather forecast for the whole week, with enough words to stay a chunk of its own "
           "and not merge into the next one.\n\n## Packing\n\nPack light")  # fmt: skip
    run(vault, stores, routes)
    after = chunk_ids(s, TRIP)
    assert set(after) == set(before) | {"Packing#3/p0"}
    assert after["Food#1/p0"] == before["Food#1/p0"]  # other headings keep their ids
    assert after["Budget#1/p0"] == before["Budget#1/p0"]
    old_ids = {before["Packing#1/p0"], before["Packing#2/p0"]}
    assert not old_ids & set(after.values())
    for old in old_ids:
        assert old not in _chunk_rows(s)


def test_insert_section_above_keeps_existing_chunk_ids(vault, stores, routes):
    """B-I2(i), no duplicate heading involved. The section goes below Role: the tiny `# Ana Silva`
    heading merges into the next section's chunk (§2.4), so a section inserted above Role would
    legitimately change Role's text."""
    run(vault, stores, routes)
    s = stores["personal"]
    before = chunk_ids(s, ANA)
    _write(vault / ANA, "## Contact\n", "## Hobbies\n\nAna climbs on weekends and keeps a long list of "
           "routes she wants to try this year, both indoors and outdoors, with notes on grades "
           "and partners for each one of them.\n\n## Contact\n")  # fmt: skip
    run(vault, stores, routes)
    after = chunk_ids(s, ANA)
    assert {k: after[k] for k in before} == before
    assert set(after) - set(before) == {"Ana Silva > Hobbies#1/p0"}


def test_adding_a_note_adds_exactly_its_rows(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    tables = ("nodes", "chunks", "edges", "aliases", "props", "link_occurrences", "source_files")

    def sizes():
        return {t: s.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}

    before = sizes()
    (vault / "personal" / "Plain.md").write_text(
        "# Plain\n\n## Only\n\n" + "A plain note about nothing in particular. " * 8 + "\n"
    )
    report = run(vault, stores, routes)
    assert counts(report)["added"] == 1 and counts(report)["unchanged"] == INDEXED_FILES
    after = sizes()
    assert {t: after[t] - before[t] for t in tables} == {
        "nodes": 1, "chunks": 1, "edges": 0, "aliases": 1, "props": 0, "link_occurrences": 0,
        "source_files": 1}  # fmt: skip


def test_delete_then_restore_keeps_agent_rows(vault, stores, routes):
    """B-I2(g): the importer owns import rows only; agent aliases and edges survive both ways."""
    run(vault, stores, routes)
    s = stores["personal"]
    atlas_id, ana_id = node_id(s, ATLAS), node_id(s, ANA)
    s.conn.execute("INSERT INTO aliases(alias, alias_norm, node_id, origin)"
                   " VALUES ('Project A', 'project a', ?, 'agent')", (atlas_id,))  # fmt: skip
    s.conn.execute("INSERT INTO edges(src, dst, rel, origin, recorded_at) VALUES"
                   " (?, ?, 'relates_to', 'agent', '2026-10-22T09:00:00Z')", (atlas_id, ana_id))  # fmt: skip
    text = (vault / ATLAS).read_text()
    (vault / ATLAS).unlink()
    run(vault, stores, routes)
    (vault / ATLAS).write_text(text)
    assert counts(run(vault, stores, routes))["restored"] == 1
    assert s.conn.execute("SELECT alias FROM aliases WHERE node_id = ? AND origin = 'agent'",
                          (atlas_id,)).fetchall() == [("Project A",)]  # fmt: skip
    assert s.conn.execute("SELECT removed_at FROM edges WHERE origin = 'agent'").fetchall() == [(None,)]  # fmt: skip


def test_removed_body_link_ends_the_edge_and_leaves_the_text(vault, stores, routes):
    """B-I2(h) for a body link: one of two Bruno links goes, the edge stays live; both go: ended."""
    run(vault, stores, routes)
    s = stores["personal"]
    bruno = "entity:topic:Bruno"
    path = vault / MEETING
    _write(path, "Follow-up with [[Bruno]] on staffing", "Follow-up on staffing")
    _write(path, 'attendees: ["[[Ana Silva]]", "[[Bruno]]"]', 'attendees: ["[[Ana Silva]]"]')
    run(vault, stores, routes)
    d = dump_store(s.conn)
    assert [e["live"] for e in d["edges"] if e["dst"] == bruno] == [False, False]


def test_kept_chunk_keeps_its_occurrences_when_another_section_changes(vault, stores, routes):
    """A3 step 4.5: occurrences of a chunk that is kept must not be inserted a second time
    (occ_body is UNIQUE on (chunk_id, pos)) and must not be lost."""
    run(vault, stores, routes)
    s = stores["personal"]
    before = [e for e in dump_store(s.conn)["edges"] if e["src"] == MEETING]
    _write(vault / MEETING, "Decided to keep the weekly cadence", "Decided to drop the weekly cadence")  # fmt: skip
    assert counts(run(vault, stores, routes))["changed"] == 1
    after = [e for e in dump_store(s.conn)["edges"] if e["src"] == MEETING]
    assert after == before


def test_renamed_title_updates_kept_chunks_context_and_vector(vault, stores, routes):
    run(vault, stores, routes)
    s = stores["personal"]
    before = chunk_ids(s, TRIP)
    (vault / TRIP).rename(vault / TRIP_RENAMED)
    run(vault, stores, routes)
    assert chunk_ids(s, TRIP_RENAMED) == before  # same text, same keys: same ids
    ctx = s.conn.execute("SELECT ctx FROM chunks WHERE id = ?", (before["Food#1/p0"],)).fetchone()[
        0
    ]
    assert ctx.startswith("Trip plan")
    assert _vec_ids(s) == _chunk_rows(s)


def test_link_occurrences_match_an_independent_count(vault, stores, routes):
    """B-I3 on the fixture: occurrences (body + frontmatter) against a regex count of the source
    with comments and code removed, and the unique-edge count per rel against distinct targets."""
    import re

    run(vault, stores, routes)
    link = re.compile(r"(!?)\[\[([^\]|#]+)")
    want_occ = 0
    for s in stores.values():
        for (ref,) in s.conn.execute("SELECT source_ref FROM nodes WHERE origin='import' AND"
                                     " source_ref LIKE '%.md'"):  # fmt: skip
            text = re.sub(r"%%.*?%%", "", (vault / ref).read_text(), flags=re.DOTALL)
            text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            text = re.sub(r"`[^`\n]*`", "", text)
            want_occ += len(link.findall(text))
    total = sum(
        s.conn.execute("SELECT count(*) FROM link_occurrences").fetchone()[0]
        for s in stores.values()
    )
    assert total == want_occ


def test_folder_outside_every_route_is_reported_not_imported(vault, stores, routes):
    (vault / "inbox").mkdir()
    (vault / "inbox" / "Loose.md").write_text("# Loose\n\nNowhere to go.\n")
    report = run(vault, stores, routes)
    assert report.unrouted == ["inbox/Loose.md"]
    assert counts(report)["added"] == INDEXED_FILES
    assert all("inbox/Loose.md" not in _refs(s) for s in stores.values())


def test_unreadable_frontmatter_is_reported_and_the_old_node_stays(vault, stores, routes):
    run(vault, stores, routes)
    (vault / ATLAS).write_text("---\ntags: [project\n---\n# Atlas\n")
    report = run(vault, stores, routes)
    assert [path for path, _ in report.errors] == [ATLAS]
    assert counts(report)["deleted"] == 0
    s = stores["personal"]
    assert s.conn.execute("SELECT status FROM nodes WHERE source_ref = ?", (ATLAS,)).fetchone()[0] == "active"  # fmt: skip


def test_page_created_for_an_unresolved_link_retargets_the_unchanged_linker(vault, stores, routes):
    """The linker's own text did not change, but [[Ghost Topic]] now resolves to a file."""
    run(vault, stores, routes)
    ghost = "personal/Ghost Topic.md"
    (vault / ghost).write_text("# Ghost Topic\n\n" + "Now it has a page of its own. " * 6 + "\n")
    report = run(vault, stores, routes)
    assert counts(report)["added"] == 1 and counts(report)["changed"] == 1
    live = {(e["src"], e["rel"], e["dst"]): e["live"] for e in dump_store(stores["personal"].conn)["edges"]}  # fmt: skip
    assert live[(MEETING, "links_to", "entity:topic:Ghost Topic")] is False
    assert live[(MEETING, "links_to", ghost)] is True
    assert counts(run(vault, stores, routes))["unchanged"] == INDEXED_FILES + 1


def test_chunks_of_a_real_import_reconstruct_every_note_body(vault, stores, routes):
    """B-I4(1) source fidelity (F4): heading lines are part of the chunk text."""
    from bench.suites.b_p4 import source_fidelity

    run(vault, stores, routes)
    assert source_fidelity({n: s.conn for n, s in stores.items()}, vault) == []


def test_agent_props_survive_a_reimport_of_a_changed_note(vault, stores, routes):
    """F5: the props reconcile replaces origin 'import' only."""
    run(vault, stores, routes)
    s = stores["personal"]
    ana = node_id(s, ANA)
    s.conn.execute("INSERT INTO props(node_id, key, value, origin) VALUES (?, 'mood', 'calm', 'agent')",
                   (ana,))  # fmt: skip
    _write(vault / ANA, "role: engineer", "role: lead")
    assert counts(run(vault, stores, routes))["changed"] == 1
    props = s.conn.execute("SELECT key, value, origin FROM props WHERE node_id = ? ORDER BY 1, 2",
                           (ana,)).fetchall()  # fmt: skip
    assert ("mood", "calm", "agent") in props
    assert ("role", "lead", "import") in props and ("role", "engineer", "import") not in props


def test_kept_chunk_whose_ctx_changes_gets_the_vector_of_the_new_ctx(vault, stores, routes):
    """F13 (mutant M5): the stored vector is the one of `ctx + text` as they are now."""
    import sqlite_vec

    run(vault, stores, routes)
    s = stores["personal"]
    before = chunk_ids(s, TRIP)
    (vault / TRIP).rename(vault / TRIP_RENAMED)
    run(vault, stores, routes)
    cid = before["Food#1/p0"]
    ctx, text = s.conn.execute("SELECT ctx, text FROM chunks WHERE id = ?", (cid,)).fetchone()
    assert ctx.startswith("Trip plan")
    (want,) = s.embed_chunks([(cid, ctx, text)])
    got = s.conn.execute(
        f"SELECT embedding FROM {s.vec_table} WHERE chunk_id = ?", (cid,)
    ).fetchone()[0]
    assert got == sqlite_vec.serialize_float32(want)


def test_unparseable_created_is_warned_not_silently_replaced_by_mtime(vault, stores, routes):
    """F15: the note is still imported (dated by mtime) and the report says why."""
    (vault / "personal" / "Odd date.md").write_text(
        "---\ncreated: next tuesday\n---\n# Odd\n\nBody.\n"
    )
    report = run(vault, stores, routes)
    assert report.errors == []
    assert [w for w in report.warnings if "personal/Odd date.md" in w and "created" in w]
    assert len(report.warnings) == 1


def test_nfc_link_resolves_to_an_nfd_named_file_not_a_topic_stub(vault, stores, routes):
    """F15: macOS-style decomposed filenames and typed (composed) links are the same name."""
    nfd = "Café notes"
    nfc = "Café notes"
    (vault / "personal" / f"{nfd}.md").write_text("# Cafe\n\nA place with a long enough body.\n")
    (vault / "personal" / "Visitor.md").write_text(f"# Visitor\n\nSee [[{nfc}]].\n")
    run(vault, stores, routes)
    s = stores["personal"]
    (dst,) = s.conn.execute(
        "SELECT d.source_ref FROM edges e JOIN nodes d ON d.id = e.dst JOIN nodes x ON x.id = e.src"
        " WHERE x.source_ref = 'personal/Visitor.md' AND e.removed_at IS NULL"
    ).fetchone()
    assert dst == f"personal/{nfd}.md"
