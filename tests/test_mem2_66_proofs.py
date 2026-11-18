"""D5 proofs: model constructors are spies; no network or real models."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from bench.datasets import lme
from mem import doctor, embedder, rerank
from mem import serve as serve_mod
from mem.cli import app
from mem.config import default_config
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker


@pytest.mark.parametrize("command", ["calibrate", "reembed"])
def test_model_cli_does_not_load_models_in_its_process(tmp_path, monkeypatch, command, free_port):
    """MEM2-67: reembed is D5's one exception (docs/decisions/0006): it may load the new embedder
    only when no serve is up, so here a serve holds its store and it must refuse. Both commands
    get a free --port so a real serve on the default port is never called."""
    loaded = []

    def emb(*args, **kwargs):
        loaded.append("embedder")
        return HashEmbedder(768)

    def rr(*args, **kwargs):
        loaded.append("reranker")
        return OverlapReranker()

    monkeypatch.setattr(doctor, "check_model", lambda _: SimpleNamespace(status="ok"))
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder", emb)
    monkeypatch.setattr(rerank, "CrossEncoderReranker", rr)
    if command == "reembed":
        db = tmp_path / "p.db"
        store = Store.open(db, HashEmbedder(768), fp=default_config().fp_embed())
        store.close()
        args = [command, "--store", f"personal={db}", "--port", str(free_port)]
    else:
        data = Path(__file__).parents[1] / "bench/fixtures/lme/lme_s_mini.json"
        q = lme.load_questions(data)[0]
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            json.dumps(
                {
                    "qid": q["question_id"],
                    "kind": "answerable",
                    "sessions": q["haystack_session_ids"],
                    "rel": {},
                }
            )
            + "\n"
        )
        args = [command, "--labels", str(labels), "--lme", str(data), "--port", str(free_port)]
    if command == "reembed":
        with serve_mod.store_locks([db]):  # a running mem serve holds the store
            result = CliRunner().invoke(app, args)
        assert result.exit_code == 1 and "Stop mem serve" in result.output
    else:
        CliRunner().invoke(app, args)
    assert loaded == [], f"D5 requires server model access; CLI loaded {loaded}"
