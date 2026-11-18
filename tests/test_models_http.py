"""D5 (architecture-v2 §2.1, docs/decisions/0006): only `mem serve` loads models; CLI tools use
them through its local-only `/models` routes. Stub models only, no network beyond loopback."""

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from mem import cli, embedder, models_http, rerank
from mem import serve as serve_mod
from mem.cli import app
from mem.config import default_config, stub_config
from mem.serve import build_app
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker

STUB = stub_config()


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")  # fmt: skip
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as r:
        return json.loads(r.read())


# ---- the endpoint ------------------------------------------------------------------------------


def test_models_info_names_the_served_sections(models_serve):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{models_serve}/models", timeout=10) as r:
        info = json.loads(r.read())
    assert info == {"embedder": STUB.embedder, "reranker": STUB.reranker,
                    "fp_embed": STUB.fp_embed(), "dim": 256}  # fmt: skip


def test_embed_and_rerank_return_what_the_models_return_in_process(models_serve):
    emb, rr = HashEmbedder(256), OverlapReranker()
    q = _post(models_serve, "/models/embed", {"kind": "query", "texts": ["dog Bosco"]})
    assert q == {"vectors": [emb.embed_query("dog Bosco")]}
    d = _post(models_serve, "/models/embed", {"kind": "docs", "texts": ["a cat", "a dog"]})
    assert d == {"vectors": emb.embed_docs(["a cat", "a dog"])}
    r = _post(models_serve, "/models/rerank", {"query": "dog", "docs": ["a cat", "a dog", ""]})
    assert r == {"logits": rr.score("dog", ["a cat", "a dog", ""])}


@pytest.mark.parametrize(
    ("path", "body", "msg"),
    [
        ("/models/embed", {"kind": "doc", "texts": ["a"]}, "kind"),
        ("/models/embed", {"kind": "docs", "texts": "a"}, "texts"),
        ("/models/embed", {"kind": "docs", "texts": [1]}, "texts"),
        ("/models/embed", {"kind": "docs", "texts": ["a"] * 513}, "512"),
        ("/models/rerank", {"query": 3, "docs": ["a"]}, "query"),
        ("/models/rerank", {"query": "q", "docs": ["a"], "x": 1}, "unknown"),
        ("/models/rerank", [], "object"),
    ],
)
def test_bad_bodies_are_400_with_a_reason(path, body, msg):
    client = TestClient(build_app({}, models=models_http.ModelHost(HashEmbedder(256),
                        OverlapReranker(), STUB)), client=("127.0.0.1", 5000))  # fmt: skip
    r = client.post(path, json=body)
    assert r.status_code == 400 and msg in r.json()["error"]
    assert client.post(path, content=b"{not json").status_code == 400


def test_non_loopback_clients_are_refused():
    host = models_http.ModelHost(HashEmbedder(256), OverlapReranker(), STUB)
    tailnet = TestClient(build_app({}, models=host), client=("100.64.0.7", 5000))
    assert tailnet.get("/models").status_code == 403
    assert tailnet.post("/models/rerank", json={"query": "q", "docs": []}).status_code == 403
    local = TestClient(build_app({}, models=host), client=("::1", 5000))
    assert local.get("/models").status_code == 200


# ---- the client --------------------------------------------------------------------------------


def test_remote_models_score_like_the_local_ones(models_serve):
    emb, rr = models_http.connect(models_serve, STUB)
    assert emb.dim == 256
    texts = [f"text {i}" for i in range(models_http.MAX_ITEMS + 3)]  # more than one request
    assert emb.embed_docs(texts) == HashEmbedder(256).embed_docs(texts)
    assert emb.embed_query("q") == HashEmbedder(256).embed_query("q")
    assert rr.score("dog", ["a dog", "a cat"]) == OverlapReranker().score("dog", ["a dog", "a cat"])
    assert rr.score("dog", []) == []


def test_connect_refuses_a_serve_with_other_models(models_serve):
    with pytest.raises(models_http.ServeUnavailable, match="--models-only"):
        models_http.connect(models_serve, default_config())


def test_connect_without_a_serve_says_to_start_one(free_port):
    with pytest.raises(models_http.ServeUnavailable, match="start it"):
        models_http.connect(free_port, STUB)
    assert not models_http.serve_listening(free_port)


def test_serve_listening_sees_a_running_serve(models_serve):
    assert models_http.serve_listening(models_serve)


# ---- mem serve --models-only ---------------------------------------------------------------------


def test_models_only_serve_needs_no_store_and_mounts_none(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(serve_mod, "run_http", lambda stores, emb, rr, cfg, **kw: seen.update(
        stores=stores, emb=emb, port=kw["port"]))  # fmt: skip
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--models-only", "--embedder", "hash",
                                 "--port", "9911", "--store", f"personal={tmp_path}/none.db"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert seen["stores"] == {} and seen["port"] == 9911 and isinstance(seen["emb"], HashEmbedder)
    assert not (tmp_path / "none.db").exists()
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--models-only", "--embedder", "hash"])
    assert r.exit_code != 0 and "--http" in r.output


# ---- store locks -------------------------------------------------------------------------------


def test_serve_lock_is_shared_and_excludes_reembed(tmp_path):
    db = tmp_path / "p.db"
    with (
        serve_mod.store_locks([db]),
        serve_mod.store_locks([db]),  # two serves may share
        pytest.raises(serve_mod.StoreBusy, match="mem serve"),
        serve_mod.store_locks([db], exclusive=True),
    ):
        pass
    with (
        serve_mod.store_locks([db], exclusive=True),  # free again
        pytest.raises(serve_mod.StoreBusy, match="reembed"),
        serve_mod.store_locks([db]),
    ):
        pass


def test_serve_cli_holds_the_store_lock_while_serving(monkeypatch, tmp_path):
    db = tmp_path / "p.db"
    held = []

    def fake_run_http(stores, *a, **kw):
        with (
            pytest.raises(serve_mod.StoreBusy),
            serve_mod.store_locks(stores.values(), exclusive=True),
        ):
            pass
        held.append(True)

    monkeypatch.setattr(serve_mod, "preflight", lambda stores, cfg: [])
    monkeypatch.setattr(serve_mod, "run_http", fake_run_http)
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--embedder", "hash", "--store",
                                 f"personal={db}"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert held == [True]


def test_serve_cli_refuses_a_store_reembed_holds(monkeypatch, tmp_path):
    db = tmp_path / "p.db"
    monkeypatch.setattr(serve_mod, "preflight", lambda stores, cfg: [])
    monkeypatch.setattr(serve_mod, "run_http", lambda *a, **kw: pytest.fail("served"))
    with serve_mod.store_locks([db], exclusive=True):
        r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--http", "--embedder", "hash", "--store",
                                     f"personal={db}"])  # fmt: skip
    assert r.exit_code == 1 and "reembed" in r.output


# ---- mem calibrate and mem reembed under D5 ------------------------------------------------------


def _spy_models(monkeypatch):
    loaded = []
    monkeypatch.setattr(cli.doctor_mod, "check_model", lambda _: SimpleNamespace(status="ok"))
    monkeypatch.setattr(embedder, "SentenceTransformerEmbedder",
                        lambda *a, **k: loaded.append("embedder") or HashEmbedder(768))  # fmt: skip
    monkeypatch.setattr(rerank, "CrossEncoderReranker",
                        lambda *a, **k: loaded.append("reranker") or OverlapReranker())  # fmt: skip
    return loaded


def _old_store(path):
    Store.open(path, HashEmbedder(768), fp=default_config().fp_embed()).close()


def test_reembed_refuses_while_a_serve_listens_and_loads_nothing(monkeypatch, tmp_path,
                                                                  models_serve):  # fmt: skip
    loaded = _spy_models(monkeypatch)
    _old_store(tmp_path / "p.db")
    r = CliRunner().invoke(app, ["reembed", "--port", str(models_serve), "--store",
                                 f"personal={tmp_path / 'p.db'}"])  # fmt: skip
    assert r.exit_code == 1 and "Stop mem serve" in r.output
    assert loaded == []


def test_reembed_refuses_a_store_a_serve_holds_and_loads_nothing(monkeypatch, tmp_path, free_port):
    loaded = _spy_models(monkeypatch)
    _old_store(tmp_path / "p.db")
    with serve_mod.store_locks([tmp_path / "p.db"]):  # a serve on another port
        r = CliRunner().invoke(app, ["reembed", "--port", str(free_port), "--store",
                                     f"personal={tmp_path / 'p.db'}"])  # fmt: skip
    assert r.exit_code == 1 and "Stop mem serve" in r.output
    assert loaded == []


def test_reembed_with_no_serve_loads_only_the_new_embedder(monkeypatch, tmp_path, free_port):
    loaded = _spy_models(monkeypatch)
    _old_store(tmp_path / "p.db")
    r = CliRunner().invoke(app, ["reembed", "--port", str(free_port), "--store",
                                 f"personal={tmp_path / 'p.db'}"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert loaded == ["embedder"]


def test_calibrate_labels_without_a_serve_fails_in_one_line(monkeypatch, tmp_path, free_port):
    from bench.datasets import lme
    from tests.bench_t0.test_lme_pipeline import S_FILE

    loaded = _spy_models(monkeypatch)
    q = lme.load_questions(S_FILE)[0]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"qid": q["question_id"], "kind": "answerable",
                                  "sessions": q["haystack_session_ids"], "rel": {}}) + "\n")  # fmt: skip
    r = CliRunner().invoke(app, ["calibrate", "--labels", str(labels), "--lme", str(S_FILE),
                                 "--port", str(free_port)])  # fmt: skip
    assert r.exit_code == 1 and loaded == []
    fails = [ln for ln in r.output.splitlines() if ln.startswith("[FAIL]")]
    assert len(fails) == 1 and "mem serve" in fails[0] and "start it" in fails[0]


def test_store_lock_is_keyed_on_the_real_file(tmp_path, monkeypatch):
    db = tmp_path / "d" / "p.db"
    db.parent.mkdir()
    (tmp_path / "alias.db").symlink_to(db)
    monkeypatch.chdir(tmp_path)
    for spelling in [tmp_path / "alias.db", Path("d/p.db"), tmp_path / "d" / ".." / "d" / "p.db"]:
        with (
            serve_mod.store_locks([db]),
            pytest.raises(serve_mod.StoreBusy, match="mem serve"),
            serve_mod.store_locks([spelling], exclusive=True),
        ):
            pass
    assert not (tmp_path / "alias.db.serve.lock").exists()


def test_reembed_lock_takes_two_spellings_of_one_store_once(tmp_path):
    db = tmp_path / "p.db"
    (tmp_path / "alias.db").symlink_to(db)
    with serve_mod.store_locks([db, tmp_path / "alias.db"], exclusive=True):
        pass
