"""MEM2-274: the shipped default serves Qwen3-Embedding-0.6B + bge-reranker-v2-m3 (the host's final config,
Round 10 r10-all). `mem doctor` and `mem setup` check and fetch exactly the default config's two models,
and neither is gated, so a fresh install needs no Hugging Face account (the Gemma terms were for
EmbeddingGemma, the old default; it stays selectable with `models.embedder: embeddinggemma-300m`)."""

import pytest

from mem import catalog, config, doctor, pins, setup_cmd

pytestmark = pytest.mark.release_defaults

QWEN = catalog.EMBEDDERS["qwen3-embedding-0.6b"].pin
BGE = catalog.RERANKERS["bge-reranker-v2-m3"].pin


def test_doctor_checks_the_default_configs_models():
    assert doctor.configured_pins(None) == (QWEN, BGE)
    assert doctor.configured_pins(config.default_config()) == (QWEN, BGE)


def test_setup_fetches_the_default_configs_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    calls = []

    def dl(repo, rev, progress):
        calls.append((repo, rev))
        return str(tmp_path)

    setup_cmd.fetch_models(lambda _line: None, dl=dl)
    assert calls == [(QWEN.id, QWEN.revision), (BGE.id, BGE.revision)]


def test_the_default_models_are_apache_2_and_not_gemma():
    cfg = config.default_config()
    assert catalog.embedder_entry(cfg.embedder).licence == "apache-2.0"
    assert catalog.reranker_entry(cfg.reranker).licence == "apache-2.0"
    assert pins.EMBEDDER not in doctor.configured_pins(None)  # the gated Gemma repo


def test_a_failed_default_fetch_does_not_send_the_user_to_the_gemma_terms():
    def fail(repo_id, revision):
        raise OSError("connection reset")

    for check in doctor.fetch_models(downloader=fail):
        assert check.status == "fail" and "Gemma" not in check.detail


def test_a_gated_error_names_the_repo_that_failed():
    msg = setup_cmd.explain("models", f"fetch {QWEN.id}: 403 Forbidden")
    assert QWEN.id in msg and pins.EMBEDDER.id not in msg
