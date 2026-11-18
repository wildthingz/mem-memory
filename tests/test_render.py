"""P4 render (architecture-v2 §2, DR-5, B-I4): deterministic one-way Markdown/HTML mirror."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mem import render
from mem.cli import app

NOTE_REF = "personal/Trip planning.md"
CHUNKS = ["## Packing\n\nSocks & <shoes>.\n\n", "## Food\n\nRice.\n"]


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _add_note(conn, ref=NOTE_REF, chunks=CHUNKS, status="active"):
    nid = conn.execute(
        "INSERT INTO nodes(kind,type,name,date,valid_from,recorded_at,origin,source_ref,status)"
        " VALUES ('note','note','Trip planning','2026-03-02','2026-03-02T00:00:00Z',"
        "'2026-10-22T09:00:00Z','import',?,?)",
        (ref, status),
    ).lastrowid
    for i, text in enumerate(chunks):
        conn.execute(
            "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,text,tokens,content_hash)"
            " VALUES (?,?,?,'section',?,5,'h')",
            (nid, f"k{i}#1/p0", i, text),
        )
    return nid


@pytest.fixture
def filled(store, clock):
    """Memories m1, m3 (supersedes m1) about Ana e2, m4 about Bo, one note."""
    r1 = store.remember(text="Ana lives in Porto.", type="fact", about=["new:person:Ana"])
    clock.set("2026-10-23T09:00:00Z")
    store.remember(
        text="Ana lives in Lisbon <now>.", type="fact", about=["new:person:Ana"],
        supersedes=r1.structured["id"],
    )  # fmt: skip
    store.remember(text="Bo likes tea.", type="preference", about=["new:person:Bo"])
    _add_note(store.conn)
    store.conn.commit()
    return store


def test_render_byte_identical(filled, tmp_path):
    render.render_store(filled.conn, tmp_path / "a", html=True)
    render.render_store(filled.conn, tmp_path / "b", html=True)
    a = _tree(tmp_path / "a")
    assert a and a == _tree(tmp_path / "b")
    assert {"personal/index.md", "personal/memories/m1.md", "personal/memories/m1.html"} <= set(a)


def test_rerender_into_same_dir_rewrites_nothing(filled, tmp_path):
    out = tmp_path / "out"
    render.render_store(filled.conn, out)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    render.render_store(filled.conn, out)
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == before


def test_memory_page_content(filled, tmp_path):
    render.render_store(filled.conn, tmp_path)
    page = (tmp_path / "personal/memories/m1.md").read_text()
    assert page.startswith("---\nid: m1\n")
    assert "Ana lives in Porto." in page
    assert "superseded_by: m3" in page
    assert (
        'valid_to: "2026-10-23T09:00:00Z"' in page
    )  # timestamps are quoted so YAML keeps them strings
    assert "[Ana](../entities/e2.md)" in page
    assert "m3" in (tmp_path / "personal/entities/e2.md").read_text()


def test_one_change_diff_is_confined_to_that_memory(filled, tmp_path):
    out = tmp_path / "out"
    render.render_store(filled.conn, out, html=True)
    before = _tree(out)
    filled.forget(id="m4", reason="wrong")
    render.render_store(filled.conn, out, html=True)
    after = _tree(out)
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == {"personal/memories/m4.md", "personal/memories/m4.html"}


def test_note_chunks_reconstruct_the_rendered_body(filled, tmp_path):
    render.render_store(filled.conn, tmp_path)
    page = (tmp_path / "personal/notes/personal/Trip planning.md").read_text()
    front, sep, body = page.removeprefix("---\n").partition("\n---\n")
    assert sep and "source_ref: personal/Trip planning.md" in front
    assert body == "".join(CHUNKS)


def test_writes_are_confined_to_the_render_dir(filled, tmp_path):
    _add_note(filled.conn, ref="../../escape.md")
    filled.conn.commit()
    work = tmp_path / "work"
    work.mkdir()
    (work / "vault.md").write_text("# mine\n")
    snapshot = _tree(work)
    out = work / "export"
    render.render_store(filled.conn, out)
    assert {k: v for k, v in _tree(work).items() if not k.startswith("export/")} == snapshot
    assert not (tmp_path / "escape.md").exists() and not (tmp_path.parent / "escape.md").exists()
    assert all(out in p.resolve().parents for p in out.rglob("*") if p.is_file())


def test_stale_files_are_removed_but_other_content_is_kept(filled, tmp_path):
    out = tmp_path / "out"
    render.render_store(filled.conn, out)
    (out / ".git").mkdir()
    (out / ".git/HEAD").write_text("ref")
    (out / "README.txt").write_text("mine")
    filled.conn.execute("UPDATE nodes SET status='deleted' WHERE kind='note'")
    filled.conn.commit()
    render.render_store(filled.conn, out)
    assert not (out / "personal/notes/personal/Trip planning.md").exists()
    assert (out / ".git/HEAD").exists() and (out / "README.txt").exists()


def test_html_escapes_content(filled, tmp_path):
    render.render_store(filled.conn, tmp_path, html=True)
    html = (tmp_path / "personal/memories/m3.html").read_text()
    assert "&lt;now&gt;" in html and "<now>" not in html
    note = (tmp_path / "personal/notes/personal/Trip planning.html").read_text()
    assert "Socks &amp; &lt;shoes&gt;." in note


def test_render_does_not_modify_the_store(filled, tmp_path):
    path = tmp_path / "personal.db"
    filled.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    render.render_path(path, tmp_path / "out")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_cli_render_commits_a_git_mirror(filled, tmp_path):
    db, out = tmp_path / "personal.db", tmp_path / "export"
    filled.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    args = ["render", "--store", str(db), "--out", str(out), "--commit"]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert _git(out, "rev-list", "--count", "HEAD") == "1"
    assert CliRunner().invoke(app, args).exit_code == 0
    assert _git(out, "rev-list", "--count", "HEAD") == "1"  # nothing changed, no commit
    filled.forget(id="m4", reason="wrong")
    filled.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert CliRunner().invoke(app, args).exit_code == 0
    assert _git(out, "rev-list", "--count", "HEAD") == "2"
    assert _git(out, "show", "--name-only", "--format=", "HEAD") == "personal/memories/m4.md"


def test_cli_render_without_commit_leaves_no_repo(filled, tmp_path):
    filled.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out = tmp_path / "export"
    r = CliRunner().invoke(
        app, ["render", "--store", str(tmp_path / "personal.db"), "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert (out / "personal/index.md").exists() and not (out / ".git").exists()


def test_cli_registers_render_when_run_as_a_module():
    res = subprocess.run(
        [sys.executable, "-m", "mem.cli", "render", "--help"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    assert "--store" in res.stdout and "No such command" not in res.stderr


def test_note_and_file_stub_with_the_same_page_do_not_collide(filled, tmp_path):
    a = _add_note(filled.conn, ref="z.md", chunks=["note z.md\n"])
    b = _add_note(filled.conn, ref="z", chunks=["stub z\n"])
    c = _add_note(filled.conn, ref="Z.md", chunks=["note Z.md\n"])
    filled.conn.commit()
    render.render_store(filled.conn, tmp_path / "a")
    render.render_store(filled.conn, tmp_path / "b")
    tree = _tree(tmp_path / "a")
    assert tree == _tree(tmp_path / "b")
    bodies = sorted(v.split(b"\n---\n", 1)[1] for k, v in tree.items() if "/notes/z" in k.lower())
    assert bodies == [b"note Z.md\n", b"note z.md\n", b"stub z\n"]
    # the lowest id keeps the plain path; the others get an id suffix
    assert b"note z.md" in tree["personal/notes/z.md"]
    assert f"personal/notes/z~n{b}.md" in tree and f"personal/notes/Z~n{c}.md" in tree
    assert a < b < c


def test_note_chunks_follow_ord_not_insertion_order(filled, tmp_path):
    nid = _add_note(filled.conn, ref="order.md", chunks=[])
    for ord_, text in ((2, "third\n"), (0, "first\n"), (1, "second\n")):
        filled.conn.execute(
            "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,text,tokens,content_hash)"
            " VALUES (?,?,?,'section',?,1,'h')",
            (nid, f"k{ord_}", ord_, text),
        )
    filled.conn.commit()
    render.render_store(filled.conn, tmp_path)
    page = (tmp_path / "personal/notes/order.md").read_text()
    assert page.endswith("\n---\nfirst\nsecond\nthird\n")


@pytest.mark.parametrize(
    "value",
    ["a: b", "- x", "true", "null", "2026", "2026-03-02", "2026-10-23T09:00:00Z", "# c", "x # y",
     "", "No", "1e3", "0x1f", "~", "[a]", "{a: b}", "a: ", "é ü", "it's", 'say "hi"', "-", "?", "*a"],
)  # fmt: skip
def test_front_matter_scalars_round_trip_through_yaml(value):
    import yaml

    line = f"k: {render._scalar(value)}\n"
    assert yaml.safe_load(line) == {"k": value}


def test_rendered_front_matter_parses_as_strings(filled, tmp_path):
    import yaml

    filled.conn.execute("UPDATE nodes SET name='a: b' WHERE kind='note'")
    filled.conn.commit()
    render.render_store(filled.conn, tmp_path)
    for p in (tmp_path / "personal").rglob("*.md"):
        if p.name == "index.md":
            continue
        front = yaml.safe_load(p.read_text().split("---\n")[1])
        assert all(isinstance(v, str) for v in front.values()), p


def test_commit_mirror_leaves_foreign_files_out(filled, tmp_path):
    db, out = tmp_path / "personal.db", tmp_path / "export"
    filled.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    out.mkdir()
    (out / "secret.txt").write_text("not mine")
    _git(out, "init", "-q")
    _git(out, "add", "secret.txt")  # even pre-staged foreign files stay out
    (out / "other").mkdir()
    (out / "other/x.md").write_text("x")
    args = ["render", "--store", str(db), "--out", str(out), "--commit"]
    assert CliRunner().invoke(app, args).exit_code == 0
    files = _git(out, "show", "--name-only", "--format=", "HEAD").splitlines()
    assert files and all(f.startswith("personal/") for f in files)
    assert (out / "secret.txt").exists() and (out / "other/x.md").exists()
