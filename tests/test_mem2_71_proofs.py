"""Reembed must respect a serve holding the same database through a path alias."""

from typer.testing import CliRunner

from mem import cli, serve
from mem.config import stub_config
from mem.store import Store
from mem.testing import HashEmbedder


def test_reembed_refuses_served_store_through_symlink(tmp_path, monkeypatch, free_port):
    db = tmp_path / "personal.db"
    alias = tmp_path / "alias.db"
    Store.open(db, HashEmbedder(256), fp=stub_config().fp_embed()).close()
    alias.symlink_to(db)
    loaded = []

    def spy(dim):
        loaded.append(dim)
        return HashEmbedder(dim)

    monkeypatch.setattr(cli, "HashEmbedder", spy)
    with serve.store_locks([db]):
        result = CliRunner().invoke(
            cli.app,
            [
                "reembed",
                "--embedder",
                "hash",
                "--port",
                str(free_port),
                "--store",
                f"personal={alias}",
            ],
        )
    assert loaded == [], f"Reembed loaded a second embedder for a served store: {loaded}"
    assert result.exit_code == 1 and "Stop mem serve" in result.output
