"""The P4 suites on the stand-in importer and the real store with stub models (CPU, no network).

Each suite is run green on the working stand-in, then red on a stand-in broken in the one way the
check is about (a mutant), so a check that cannot fail is caught here, not on the person's vault.
"""

import json
from pathlib import Path

import pytest
import yaml

from bench import i5, records, ref_parse
from bench.adapters import p4_ours
from bench.suites import b_p4
from bench.suites.base import BackendUnavailable, Context
from tests.bench_t0 import fake_vault, pg_world, vault_world


def failed(res):
    return [i["check"] for i in res.items if "check" in i and not i["ok"]]


@pytest.fixture
def setup(tmp_path):
    mini = vault_world.make_mini(tmp_path / "mini")
    lifecycle = tmp_path / "life.yaml"
    lifecycle.write_text(vault_world.LIFECYCLE, encoding="utf-8")
    made = []

    def new_backend(name):
        be = pg_world.backend(tmp_path / "stores" / name)
        made.append(be)
        return be

    s = b_p4.P4Setup(new_backend=new_backend, fixture_vault=mini, snapshot_vault=mini,
                     lifecycle=lifecycle, routes=vault_world.ROUTES, snapshot_routes=vault_world.ROUTES,
                     render=None, bootstrap_b=200)  # fmt: skip
    yield s
    s.close()
    for be in made:
        be.close()


def ctx(tmp_path, s):
    return Context(tmp_path / "ctx", lambda *a, **k: None, None, {b_p4.OPT: s})


# ---- the reference parser, hand-worked values --------------------------------------------------


def test_ref_parse_counts_the_meeting_note_by_hand():
    note = ref_parse.parse_note("personal/2026-04-10_Meeting with Ana.md",
                                vault_world.MINI["personal/2026-04-10_Meeting with Ana.md"])  # fmt: skip
    body = [o for o in note.occurrences if o[2] is None]
    fm = [o for o in note.occurrences if o[2] is not None]
    # body: Ana Silva x4 (plain, |alias, #heading, #^block), Ghost Topic, ![[diagram.png]], board.canvas, Bruno
    assert len(body) == 8
    assert sorted(t for t, rel, _ in body if rel == "embeds") == ["diagram.png"]
    # frontmatter: attendees x2 (rel attendee), project x1
    assert sorted((rel, t) for t, rel, _ in fm) == [("attendee", "Ana Silva"), ("attendee", "Bruno"),
                                                     ("project", "Atlas")]  # fmt: skip
    assert note.tags == {"meeting", "meeting/weekly"}  # frontmatter tag + the inline nested tag
    assert note.has_frontmatter and note.has_frontmatter_rows


def test_ref_parse_ignores_code_hex_colours_comments_and_self_links():
    trip = ref_parse.parse_note("personal/2026-03-02_Trip planning.md",
                                vault_world.MINI["personal/2026-03-02_Trip planning.md"])  # fmt: skip
    assert trip.tags == {"travel", "trip/2026"}  # not #ff8800, #notatag, #nottag
    n = ref_parse.parse_note(
        "a.md", "text %%[[Hidden]]%% and [[#Heading]] and [[a]] and `[[code]]`\n"
    )
    v = ref_parse.RefVault({"a.md": n}, [], set(), set())
    assert [t for t, _, _ in n.occurrences] == ["", "a"]  # the comment and the code are gone
    assert ref_parse.resolve("a", ["a.md", "x/a.md"]) == "a.md" and v.counts()["notes"] == 1


def test_ref_parse_resolution_is_exact_path_then_unique_basename_then_shortest():
    paths = ["work/Platform.md", "personal/people/Ana Silva.md", "a/Ana Silva.md"]
    assert ref_parse.resolve("work/Platform", paths) == "work/Platform.md"
    assert ref_parse.resolve("Ana Silva", paths) == "a/Ana Silva.md"  # shortest path wins
    assert ref_parse.resolve("Nobody", paths) is None
    assert (
        ref_parse.singular("attendees") == "attendee"
        and ref_parse.singular("reports_to") == "reports_to"
    )


def test_ref_parse_vault_counts(tmp_path):
    root = vault_world.make_mini(tmp_path / "v")
    ref = ref_parse.parse_vault(root)
    c = ref.counts()
    assert c["notes"] == 8  # CLAUDE.md, Templates/, .obsidian/ and export/ are skipped
    assert (
        "personal/2026-04-10_Meeting with Ana.md",
        "personal/people/Ana Silva.md",
        "links_to",
    ) in ref.edges
    assert ("personal/2026-04-10_Meeting with Ana.md", "stub:ghost topic", "links_to") in ref.edges
    assert c["link_occurrences"] > c["logical_edges"]  # four links to Ana Silva make one edge


# ---- B-I1 --------------------------------------------------------------------------------------


def test_b_i1_green_on_a_deterministic_importer(tmp_path, setup):
    res = b_p4.run_b_i1(ctx(tmp_path, setup))
    assert res.metrics["diff_lines_fixture"] == 0 and res.metrics["diff_lines_snapshot"] == 0
    assert res.metrics["dump_rows_fixture"] > 50 and res.violations == []


def test_b_i1_red_when_the_importer_is_not_deterministic(tmp_path, setup, monkeypatch):
    calls = {"n": 0}
    real = fake_vault._write_note

    def noisy(store, path, text, chash, nid, paths, now):
        calls["n"] += 1
        return real(store, path, text + f"\nextra words {calls['n']}\n", chash, nid, paths, now)

    monkeypatch.setattr(fake_vault, "_write_note", noisy)
    res = b_p4.run_b_i1(ctx(tmp_path, setup))
    assert res.metrics["diff_lines_fixture"] > 0 and res.violations


def test_b_i1_without_any_vault_is_unavailable(tmp_path, setup):
    setup.fixture_vault = setup.snapshot_vault = None
    with pytest.raises(BackendUnavailable):
        b_p4.run_b_i1(ctx(tmp_path, setup))


# ---- B-I2 --------------------------------------------------------------------------------------


def test_b_i2_all_nine_cases_pass_on_the_standin(tmp_path, setup):
    res = b_p4.run_b_i2(ctx(tmp_path, setup))
    assert res.violations == [] and res.metrics["case_failures"] == 0
    assert {c["check"][0] for c in res.items} == set("abcdefghi")  # every case ran


def _mutant(monkeypatch, name, fn):
    monkeypatch.setattr(fake_vault, name, fn)


def test_b_i2_red_a_reimport_that_rewrites_unchanged_files(tmp_path, setup, monkeypatch):
    real = fake_vault.import_vault

    def rewrites(store, root, **kw):
        rep = real(store, root, **kw)
        store.conn.execute("UPDATE nodes SET pinned = pinned + 1 WHERE source_ref IS NOT NULL")
        return rep

    _mutant(monkeypatch, "import_vault", rewrites)
    # run_import resolves import_vault at call time through the module global
    res = b_p4.run_b_i2(ctx(tmp_path, setup))
    assert "a: re-import unchanged writes nothing" in failed(res)
    assert res.metrics["case_failures"] >= 1


def test_b_i2_red_a_rename_that_is_delete_plus_add(tmp_path, setup, monkeypatch):
    real = fake_vault.import_vault

    def no_rename(store, root, **kw):
        # forget the node of any vanished path first, so the new path cannot match its hash
        files = fake_vault.scan(Path(root))
        for (ref,) in store.conn.execute(
            "SELECT source_ref FROM nodes WHERE origin='import' AND source_ref IS NOT NULL"
        ).fetchall():
            if ref not in files:
                store.conn.execute(
                    "UPDATE nodes SET content_hash='x', status='deleted' WHERE source_ref=?", (ref,)
                )
        return real(store, root, **kw)

    _mutant(monkeypatch, "import_vault", no_rename)
    res = b_p4.run_b_i2(ctx(tmp_path, setup))
    bad = failed(res)
    assert any(b.startswith("e:") for b in bad)


def test_b_i2_red_a_changed_section_that_replaces_every_chunk(tmp_path, setup, monkeypatch):
    real = fake_vault._chunks

    def replace_all(store, nid, title, body):
        store.conn.execute(
            f"DELETE FROM {store.vec_table} WHERE chunk_id IN"
            " (SELECT id FROM chunks WHERE node_id=?)",
            (nid,),
        )
        store.conn.execute("DELETE FROM chunks WHERE node_id=?", (nid,))
        return real(store, nid, title, body)

    _mutant(monkeypatch, "_chunks", replace_all)
    res = b_p4.run_b_i2(ctx(tmp_path, setup))
    bad = failed(res)
    assert "c: other chunk ids of the note are unchanged" in bad
    assert any(b.startswith("i:") for b in bad)  # insert-above moves the existing ids too


def test_b_i2_red_a_removed_alias_that_stays_and_agent_rows_that_are_wiped(
    tmp_path, setup, monkeypatch
):
    real = fake_vault._aliases_props

    def keeps_and_wipes(conn, nid, fm, title, stem):
        conn.execute("DELETE FROM aliases WHERE node_id=? AND origin='agent'", (nid,))
        real(conn, nid, {**fm, "aliases": [*(fm.get("aliases") or []), "Weekly Ana"]}, title, stem)

    _mutant(monkeypatch, "_aliases_props", keeps_and_wipes)
    res = b_p4.run_b_i2(ctx(tmp_path, setup))
    bad = failed(res)
    assert "h: the removed alias is gone from aliases and its FTS" in bad
    assert "h: agent rows untouched" in bad
    assert "g: agent rows intact" in bad  # the restore wipes them too


def test_b_i2_red_delete_that_removes_rows_and_a_restore_that_makes_a_new_node(
    tmp_path, setup, monkeypatch
):
    real = fake_vault.import_vault

    def hard_delete(store, root, **kw):
        files = fake_vault.scan(Path(root))
        for nid, ref in store.conn.execute(
            "SELECT id, source_ref FROM nodes WHERE origin='import'"
            " AND kind IN ('note', 'entity') AND source_ref IS NOT NULL"
        ).fetchall():
            if ref not in files:
                store.conn.execute(
                    f"DELETE FROM {store.vec_table} WHERE chunk_id IN"
                    " (SELECT id FROM chunks WHERE node_id=?)",
                    (nid,),
                )
                store.conn.execute(
                    "DELETE FROM link_occurrences WHERE chunk_id IN"
                    " (SELECT id FROM chunks WHERE node_id=?)",
                    (nid,),
                )
                store.conn.execute("DELETE FROM chunks WHERE node_id=?", (nid,))
                store.conn.execute(
                    "DELETE FROM link_occurrences WHERE edge_id IN"
                    " (SELECT id FROM edges WHERE src=? OR dst=?)",
                    (nid, nid),
                )
                store.conn.execute("DELETE FROM edges WHERE src=? OR dst=?", (nid, nid))
                store.conn.execute("DELETE FROM aliases WHERE node_id=?", (nid,))
                store.conn.execute("DELETE FROM props WHERE node_id=?", (nid,))
                store.conn.execute("DELETE FROM nodes WHERE id=?", (nid,))
        return real(store, root, **kw)

    _mutant(monkeypatch, "import_vault", hard_delete)
    bad = failed(b_p4.run_b_i2(ctx(tmp_path, setup)))
    assert "f: nothing removed" in bad or any(b.startswith("f:") for b in bad)
    assert any(b.startswith("g:") for b in bad)


# ---- B-I3 --------------------------------------------------------------------------------------


def test_b_i3_green_importer_and_reference_parser_agree_on_the_mini_vault(tmp_path, setup):
    res = b_p4.run_b_i3(ctx(tmp_path, setup))
    assert res.violations == [], res.items[0]["mismatches"][:5]
    assert res.metrics["mismatches_fixture"] == 0
    counts = res.items[0]["counts"]
    assert (
        counts["notes"] == [8, 8] and counts["link occurrences"][0] == counts["link occurrences"][1]
    )


def test_b_i3_red_when_the_importer_drops_one_occurrence_a_tag_and_a_frontmatter_link(
    tmp_path, setup, monkeypatch
):
    real = fake_vault._edges

    def lossy(conn, nid, path, fm, body, kept, paths, now):
        fm = {k: v for k, v in fm.items() if k not in ("attendees", "tags")}
        return real(conn, nid, path, fm, body.replace("[[Ghost Topic]]", ""), kept, paths, now)

    monkeypatch.setattr(fake_vault, "_edges", lossy)
    res = b_p4.run_b_i3(ctx(tmp_path, setup))
    lines = res.items[0]["mismatches"]
    assert res.metrics["mismatches_fixture"] > 0
    assert any("ghost topic" in x for x in lines) and any("tag edges" in x for x in lines)
    assert any("link occurrences" in x for x in lines)


# ---- B-I4 --------------------------------------------------------------------------------------


def make_renderer(*, nondeterministic=False, unconfined=False):
    """A stand-in for `mem render`: one file per node, memory status only on its own page."""
    state = {"n": 0}

    def render(dbs, out):
        import sqlite3

        state["n"] += 1
        for db in dbs:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            scope = Path(db).stem
            for nid, kind, name, status in conn.execute("SELECT id, kind, name, status FROM nodes"):
                if kind == "memory":
                    p = out / scope / "memories" / f"m{nid}.md"
                    body = f"{name}\nstatus: {status}\n"
                    if nondeterministic:
                        body += f"render {state['n']}\n"
                    if unconfined:
                        (out / scope / "index.md").parent.mkdir(parents=True, exist_ok=True)
                        (out / scope / "index.md").write_text(
                            f"{status}:{name}\n", encoding="utf-8"
                        )
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(body, encoding="utf-8")
            conn.close()

    return render


def test_b_i4_green_fidelity_determinism_and_confinement(tmp_path, setup):
    setup.render = make_renderer()
    res = b_p4.run_b_i4(ctx(tmp_path, setup))
    assert res.violations == [], res.violations
    assert res.metrics["source_fidelity_failures"] == 0
    assert res.metrics["render_nondeterminism"] == 0 and res.metrics["render_unconfined_files"] == 0


def test_b_i4_red_nondeterministic_render_and_unconfined_diff(tmp_path, setup):
    setup.render = make_renderer(nondeterministic=True)
    res = b_p4.run_b_i4(ctx(tmp_path, setup))
    assert res.metrics["render_nondeterminism"] > 0
    setup.render = make_renderer(unconfined=True)
    res = b_p4.run_b_i4(ctx(tmp_path, setup))
    assert res.metrics["render_unconfined_files"] >= 1


def test_b_i4_red_chunks_that_lose_text(tmp_path, setup, monkeypatch):
    real = fake_vault._chunks

    def lossy(store, nid, title, body):
        return real(store, nid, title, body.replace("Remember the umbrella.", ""))

    monkeypatch.setattr(fake_vault, "_chunks", lossy)
    setup.render = make_renderer()
    res = b_p4.run_b_i4(ctx(tmp_path, setup))
    assert res.metrics["source_fidelity_failures"] > 0
    assert any("Trip planning" in v for v in res.violations)


def test_b_i4_without_a_renderer_is_unavailable(tmp_path, setup):
    with pytest.raises(BackendUnavailable):
        b_p4.run_b_i4(ctx(tmp_path, setup))


def test_source_fidelity_normalises_whitespace_only():
    assert b_p4._ws("a  b\n\nc") == "a b c"


# ---- B-I5 --------------------------------------------------------------------------------------


def test_i5_allocation_is_proportional_with_a_floor_and_totals_100():
    a = i5.allocate({"links_to": 800, "tagged": 150, "embeds": 40, "part_of": 4, "owner": 6})
    assert sum(a.values()) == 100 and a["part_of"] == 4 and a["owner"] == 5 and a["embeds"] >= 5
    assert a["links_to"] > a["tagged"] > a["embeds"]
    assert i5.allocate({"links_to": 30, "tagged": 20}) == {
        "links_to": 30,
        "tagged": 20,
    }  # fewer than 100


def test_i5_sample_is_deterministic_stratified_and_the_sheet_scores(tmp_path, setup):
    be = setup.imported("snapshot")
    rows = i5.sample(be.conns())
    assert rows == i5.sample(be.conns()) and len({r["rel"] for r in rows}) >= 4
    assert all(r["edge_id"].split(":")[0] in ("personal", "work") for r in rows)
    sheet = tmp_path / "sheet.tsv"
    i5.write_sheet(rows, sheet)
    sc = i5.score(i5.read_sheet(sheet))
    assert sc.unlabelled == len(rows) and sc.n == 0
    for r in rows:
        r["label"] = "y"
    rows[0]["label"] = "n"
    i5.write_sheet(rows, sheet)
    sc = i5.score(i5.read_sheet(sheet))
    assert sc.n == len(rows) and sc.precision == pytest.approx(1 - 1 / len(rows))
    assert sc.wrong and sc.min_rel_precision < 1.0


def test_i5_label_sitting_resumes_and_stops_on_time(tmp_path, setup):
    rows = i5.sample(setup.imported("snapshot").conns())
    sheet = tmp_path / "s.tsv"
    i5.write_sheet(rows, sheet)
    it = iter("yyn")
    done, left = i5.label_sitting(
        sheet, minutes=None, ask=lambda p: next(it, "q"), out=lambda s: None
    )
    assert done == 3 and left == len(rows) - 3
    again, left2 = i5.label_sitting(sheet, minutes=None, ask=lambda p: "y", out=lambda s: None)
    assert again == len(rows) - 3 and left2 == 0
    assert [r["label"] for r in i5.read_sheet(sheet)][:3] == ["y", "y", "n"]


def test_b_i5_gate_values_and_a_stale_sheet_is_caught(tmp_path, setup):
    rows = i5.sample(setup.imported("snapshot").conns())
    for r in rows:
        r["label"] = "y"
    sheet = tmp_path / "ok.tsv"
    i5.write_sheet(rows, sheet)
    setup.i5_sheet = sheet
    res = b_p4.run_b_i5(ctx(tmp_path, setup))
    assert res.metrics["precision"] == 1.0 and res.metrics["unlabelled"] == 0
    assert res.metrics["sheet_matches_store"] == 1.0
    rows[0]["edge_id"] = "personal:99999"
    i5.write_sheet(rows, tmp_path / "stale.tsv")
    setup.i5_sheet = tmp_path / "stale.tsv"
    assert b_p4.run_b_i5(ctx(tmp_path, setup)).metrics["sheet_matches_store"] == 0.0
    setup.i5_sheet = None
    with pytest.raises(BackendUnavailable):
        b_p4.run_b_i5(ctx(tmp_path, setup))


# ---- B-C4 --------------------------------------------------------------------------------------


def test_c4_measure_times_cold_and_warm_and_checks_the_warm_run_is_a_no_op(tmp_path, setup):
    be = setup.new_backend("c4")
    ticks = iter([0.0, 100.0, 103.0])
    rec = b_p4.c4_measure(be, setup.fixture_vault, lambda: next(ticks), vault_world.ROUTES)
    assert rec["cold_s"] == 100.0 and rec["warm_s"] == 3.0 and rec["warm_all_unchanged"] is True
    assert rec["cold_report"]["added"] == 8 and rec["warm_report"]["unchanged"] == 8
    setup.dataset_meta = {"snapshot_sha256": "ab" * 32}
    rec |= {"snapshot_sha256": "ab" * 32, "git_sha": records.git_sha()}
    f = tmp_path / "c4.json"
    f.write_text(json.dumps(rec))
    setup.c4_run = f
    res = b_p4.run_b_c4(ctx(tmp_path, setup))
    assert res.metrics["cold_s"] == 100.0 and res.metrics["warm_all_unchanged"] == 1.0
    for key, wrong in (("snapshot_sha256", "cd" * 32), ("git_sha", "0" * 40), ("chunks", 1)):
        f.write_text(json.dumps(rec | {key: wrong}))  # another snapshot / commit / vault
        with pytest.raises(ValueError, match=f"does not match this run.*{key}"):
            b_p4.run_b_c4(ctx(tmp_path, setup))
    f.write_text(json.dumps({k: v for k, v in rec.items() if k != "git_sha"}))  # unanchored
    with pytest.raises(ValueError, match="git_sha"):
        b_p4.run_b_c4(ctx(tmp_path, setup))
    setup.dataset_meta = {}  # no snapshot to anchor to: unavailable, never a pass
    with pytest.raises(BackendUnavailable, match="snapshot"):
        b_p4.run_b_c4(ctx(tmp_path, setup))


def test_c4_warm_is_flagged_when_the_second_import_writes(tmp_path, setup, monkeypatch):
    real = fake_vault.import_vault
    seen = {"n": 0}

    def touching(store, root, **kw):
        seen["n"] += 1
        rep = real(store, root, **kw)
        if seen["n"] > 2:  # the second import (two stores per run)
            rep["changed"] = 1
        return rep

    monkeypatch.setattr(fake_vault, "import_vault", touching)
    rec = b_p4.c4_measure(
        setup.new_backend("c4b"), setup.fixture_vault, iter([0, 1, 2]).__next__, vault_world.ROUTES
    )
    assert rec["warm_all_unchanged"] is False


# ---- B-C1 / B-C3 at V5k ----------------------------------------------------------------------


def _latency(path, p95):
    stage = {"median": p95 / 2, "p95": p95}
    rec = {
        "cold_ms": 900.0,
        "n": 300,
        "one_client": {"total": stage, "rerank": stage},
        "four_clients": {"total": stage},
        "corpus": "V5k",
        "corpus_fingerprint": "f",
    }
    path.write_text(json.dumps(rec))
    return path


def test_b_c1_and_b_c3_at_v5k_take_the_median_of_three_runs(tmp_path, setup):
    runs = [_latency(tmp_path / f"r{i}.json", p) for i, p in enumerate((1000, 1400, 1200))]
    setup.latency_runs = {"V5k": runs}
    setup.c3_runs = {"V5k": runs}
    assert b_p4.run_b_c1(ctx(tmp_path, setup)).metrics["warm_p95_ms_median"] == 1200
    assert b_p4.run_b_c3(ctx(tmp_path, setup)).metrics["warm_p95_ms_median"] == 1200
    setup.latency_runs = {"V5k": runs[:2]}
    with pytest.raises(BackendUnavailable):
        b_p4.run_b_c1(ctx(tmp_path, setup))


# ---- adapter -----------------------------------------------------------------------------------


def test_adapter_normalises_importer_reports_and_resolves_the_contract():
    class Rep:
        added, unchanged = 3, 4

    assert p4_ours.normalise_report(Rep()) == {"added": 3, "changed": 0, "renamed": 0, "deleted": 0,
                                               "restored": 0, "unchanged": 4}  # fmt: skip
    assert p4_ours.normalise_report({"added": 1})["added"] == 1
    from mem import importer

    assert p4_ours.resolve_importer() is importer.run_import  # the real contract, on this branch


def test_resolve_importer_is_unavailable_when_the_module_or_entry_point_is_missing(monkeypatch):
    import sys
    import types

    import mem

    monkeypatch.delattr(mem, "importer", raising=False)
    monkeypatch.setitem(sys.modules, "mem.importer", None)  # import raises ImportError
    with pytest.raises(BackendUnavailable, match="mem.importer"):
        p4_ours.resolve_importer()
    monkeypatch.setitem(sys.modules, "mem.importer", types.ModuleType("mem.importer"))
    monkeypatch.setattr(mem, "importer", sys.modules["mem.importer"], raising=False)
    with pytest.raises(BackendUnavailable, match="run_import"):
        p4_ours.resolve_importer()


def test_ref_parse_frontmatter_rows_ignore_created_and_implicit_aliases():
    row = lambda fm, body="# T\n": ref_parse.parse_note(
        "a/2026-01-01_T.md", f"---\n{fm}\n---\n{body}").has_frontmatter_rows  # fmt: skip
    assert not row("created: 2026-01-01")
    assert not row("created: 2026-01-01\ntags: [x]\naliases: [T, 2026-01-01_T]")
    assert row("created: 2026-01-01\naliases: [Other]")
    assert row("status: open")
    assert not row("aliases: [T]", "# Some other heading\n")  # the date-less stem is implicit


def test_single_store_routes_send_every_folder_and_override_to_personal(tmp_path):
    root = vault_world.make_mini(tmp_path / "v")
    r = p4_ours.single_store_routes(root, {"projects/client-x": "work"})
    assert r["work"] == r["projects"] == r["projects/client-x"] == "personal"
    be = pg_world.backend(tmp_path / "s")
    try:
        rep = be.import_vault(root, routes=r)
        assert rep["added"] == 8
        n_work = (
            be.conns()["work"]
            .execute("SELECT count(*) FROM nodes WHERE source_ref IS NOT NULL")
            .fetchone()[0]
        )
        assert n_work == 0  # one store holds the whole vault
    finally:
        be.close()


def test_default_lifecycle_descriptor_matches_the_real_fixture_when_it_is_present():
    spec = yaml.safe_load(Path("bench/fixtures/vault-mini.lifecycle.yaml").read_text())
    root = Path(spec["vault"])
    if not root.exists():
        pytest.skip(
            "tests/fixtures/minivault is on bosmang/p4-fixture (MEM2-31), not merged here yet"
        )
    for key in ("edit", "delete", "insert_section"):
        assert (root / spec[key]["path"]).exists()
    text = (root / spec["edit"]["path"]).read_text()
    assert spec["edit"]["find"] in text
    assert (root / spec["rename"]["from"]).exists() and not (root / spec["rename"]["to"]).exists()
    assert not (root / spec["add"]["path"]).exists()
    meet = (root / spec["remove"]["path"]).read_text()
    for k in ("alias_edit", "tag_edit", "prop_edit", "link_edit"):
        assert spec["remove"][k]["find"] in meet
    assert spec["insert_section"]["before"] in (root / spec["insert_section"]["path"]).read_text()
