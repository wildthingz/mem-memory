"""`mem import`: the command line over the importer (D5: it loads no model)."""

import sqlite3

from typer.testing import CliRunner

from mem.cli import app
from tests.minivault import FIXTURE, make_vault


def _args(tmp_path, vault, *extra):
    return ["import", str(vault), "--store", f"personal={tmp_path}/p.db",
            "--store", f"work={tmp_path}/w.db", "--config", str(FIXTURE / "config.yaml"),
            "--embedder", "hash", "--tz", "UTC", *extra]  # fmt: skip


def _refs(path):
    with sqlite3.connect(path) as conn:
        return {r[0] for r in conn.execute("SELECT source_ref FROM nodes WHERE origin = 'import'")}


def test_import_routes_by_config_and_a_second_run_changes_nothing(tmp_path):
    vault = make_vault(tmp_path)
    first = CliRunner().invoke(app, _args(tmp_path, vault))
    assert first.exit_code == 0, first.output
    assert "added 10  changed 0  renamed 0  deleted 0  restored 0  unchanged 0" in first.output
    assert "projects/client-x/Kickoff.md" in _refs(tmp_path / "w.db")  # routes from config.yaml
    second = CliRunner().invoke(app, _args(tmp_path, vault))
    assert "added 0  changed 0  renamed 0  deleted 0  restored 0  unchanged 10" in second.output


def test_unreadable_file_is_reported_with_exit_1_and_the_rest_is_imported(tmp_path):
    vault = make_vault(tmp_path)
    (vault / "projects" / "Atlas.md").write_text("---\ntags: [project\n---\n# Atlas\n")
    result = CliRunner().invoke(app, _args(tmp_path, vault))
    assert result.exit_code == 1
    assert "projects/Atlas.md" in result.output
    assert "added 9" in result.output


def test_import_with_the_pinned_models_needs_a_running_serve(tmp_path, free_port):
    vault = make_vault(tmp_path)
    args = ["import", str(vault), "--store", f"personal={tmp_path}/p.db",
            "--port", str(free_port)]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert "Nothing was imported" in result.output
    assert not (tmp_path / "p.db").exists()
