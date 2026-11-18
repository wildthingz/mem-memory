"""D5 for the harness: the bench runs its own pinned `mem serve` and never constructs a model.

Stand-in models (`--embedder hash`) in a real `mem serve --http --models-only` child process; no
network, no GPU.
"""

import socket
from pathlib import Path

import pytest

from bench.adapters.lme_ours import OursLmeBackend, OursLoadServer
from bench.datasets import corpora, lme
from bench.served import PinnedServe
from bench.suites.base import BackendUnavailable
from mem import embedder, models_http, rerank, testing
from mem.config import stub_config
from mem.gate import Gate

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
S_FILE, M_FILE = FIX / "lme_s_mini.json", FIX / "lme_m_mini.json"
STUB = stub_config()
OPEN_GATE = Gate(1.0, 0.0, 0.0, 0.0, 0.0)


def test_pinned_serve_answers_for_its_config_and_stops(free_port):
    with PinnedServe(STUB, models="hash", port=free_port) as serve:
        assert serve.running and serve.port == free_port
        emb, rr = serve.models
        assert emb.dim == 256 and len(emb.embed_query("soccer")) == 256
        assert rr.score("soccer", ["soccer match", "piano"])[0] > rr.score("soccer", ["piano"])[0]
    assert not serve.running and not models_http.serve_listening(free_port)


def test_pinned_serve_refuses_a_port_in_use(free_port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", free_port))
        sock.listen()
        serve = PinnedServe(STUB, models="hash", port=free_port)
        with pytest.raises(BackendUnavailable, match="in use"):
            serve.start()
    assert not serve.running


def test_pinned_serve_reports_a_failed_start_and_cleans_up(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("nonsense: 1\n")
    serve = PinnedServe(STUB, models="hash", config_path=bad)
    with pytest.raises(BackendUnavailable, match="exited with code"):
        serve.start()
    assert not serve.running and serve.models is None


def test_pinned_serve_refuses_other_models(free_port):
    from mem.config import default_config

    serve = PinnedServe(default_config(), models="hash", port=free_port)  # serves the stubs
    with pytest.raises(BackendUnavailable, match="other models"):
        serve.start()
    assert not serve.running and not models_http.serve_listening(free_port)


def test_bench_process_constructs_no_model(tmp_path, monkeypatch):
    """Spies on every model constructor in the bench process: score, retrieve and the load
    server all run, and none fires (they run in the bench's own mem serve, a child process)."""
    fired: list[str] = []

    def spy(name, cls):
        def make(*a, **kw):
            fired.append(name)
            return cls(*a, **kw)

        return make

    for mod, name in ((embedder, "SentenceTransformerEmbedder"), (rerank, "CrossEncoderReranker"),
                      (testing, "HashEmbedder"), (testing, "OverlapReranker")):  # fmt: skip
        monkeypatch.setattr(mod, name, spy(name, getattr(mod, name)))

    q = lme.load_questions(S_FILE)[0]
    hay = lme.haystack_of(q)
    backend = OursLmeBackend(tmp_path / "work", models="hash", gate=OPEN_GATE)
    assert backend.score(q, hay)
    r = backend.retrieve(q, hay, want_legs=True)
    assert r.text and r.F and set(r.legs) == {"bm25", "dense", "rrf"}
    serve = backend.serve
    assert serve.running
    backend.close()
    assert not serve.running

    plan = corpora.plan_corpus(lme.load_questions(M_FILE), 30)
    server = OursLoadServer(plan, cfg=STUB, models="hash", gate=OPEN_GATE)
    assert "total" in server.timed_recall("laptop")
    load_serve = server.serve
    server.close()
    assert not load_serve.running
    assert fired == [], f"D5: the bench process constructed {fired}"
