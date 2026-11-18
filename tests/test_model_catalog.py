"""Option F (MEM2-219, ADR 0025): the pinned model catalog, `models:` config selection, per-model
fingerprints, the VRAM check and the score shape the Platt fit takes. Stub models only: no test
loads a real model or touches the network."""

import re

import pytest

from mem import catalog, config, doctor, reembed, schema, serve
from mem.calibrate import fit_platt
from mem.gate import Gate, write_gate
from mem.store import Store, doc_text
from mem.testing import HashEmbedder

QWEN_RR = "qwen3-reranker-4b"
MXBAI = "mxbai-rerank-large-v2"
QWEN_EMB = "qwen3-embedding-0.6b"


def _load(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return config.load(p)


# ---- pins and licences --------------------------------------------------------------------------


def test_new_entries_are_apache_and_pinned_to_the_checked_commits():
    assert catalog.RERANKERS[QWEN_RR].pin.id == "Qwen/Qwen3-Reranker-4B"
    assert catalog.RERANKERS[QWEN_RR].pin.revision == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert catalog.RERANKERS[MXBAI].pin.id == "mixedbread-ai/mxbai-rerank-large-v2"
    assert catalog.RERANKERS[MXBAI].pin.revision == "ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6"
    assert catalog.EMBEDDERS[QWEN_EMB].pin.id == "Qwen/Qwen3-Embedding-0.6B"
    assert catalog.EMBEDDERS[QWEN_EMB].pin.revision == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    for e in (*catalog.RERANKERS.values(), *catalog.EMBEDDERS.values()):
        assert re.fullmatch(r"[0-9a-f]{40}", e.pin.revision)
    assert catalog.RERANKERS[QWEN_RR].licence == catalog.RERANKERS[MXBAI].licence == "apache-2.0"
    assert catalog.EMBEDDERS[QWEN_EMB].licence == "apache-2.0"


def test_no_gemma_licensed_reranker_and_no_small_qwen_reranker():
    assert set(catalog.RERANKERS) == {"bge-reranker-v2-m3", QWEN_RR, MXBAI}
    for e in catalog.RERANKERS.values():
        assert "gemma" not in e.pin.id.lower() and "0.6b" not in e.pin.id.lower()


# ---- config selection ---------------------------------------------------------------------------


def test_default_selection_is_the_current_models_and_fingerprint_is_unchanged(tmp_path):
    base = config.default_config()
    explicit = _load(
        tmp_path, "models: {embedder: embeddinggemma-300m, reranker: bge-reranker-v2-m3}\n"
    )
    assert explicit.fp_all() == base.fp_all() and explicit.fp_embed() == base.fp_embed()
    assert base.embedder["id"] == "google/embeddinggemma-300m"
    assert base.reranker["id"] == "BAAI/bge-reranker-v2-m3"


def test_models_reranker_selects_a_pinned_entry(tmp_path):
    cfg = _load(tmp_path, f"models: {{reranker: {QWEN_RR}}}\n")
    assert cfg.reranker["id"] == "Qwen/Qwen3-Reranker-4B"
    assert cfg.reranker["revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert cfg.reranker["dtype"] == "bf16"
    assert cfg.embedder == config.default_config().embedder


def test_models_embedder_selects_with_its_documented_prompts(tmp_path):
    cfg = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}}}\n")
    assert cfg.embedder["id"] == "Qwen/Qwen3-Embedding-0.6B" and cfg.embedder["dim"] == 1024
    assert cfg.embedder["query_prompt"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query"
        "\nQuery:"
    )
    assert cfg.embedder["doc_prompt"] == ""  # the card: no instruction on the document side


@pytest.mark.parametrize(
    "name", ["Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b", "gemma-reranker", "nope"]
)
def test_unknown_or_excluded_model_is_refused_naming_the_choices(tmp_path, name):
    with pytest.raises(config.ConfigError, match=r"bge-reranker-v2-m3.*qwen3-reranker-4b"):
        _load(tmp_path, f"models: {{reranker: {name}}}\n")


def test_unknown_models_key_is_refused(tmp_path):
    with pytest.raises(config.ConfigError, match="unknown field"):
        _load(tmp_path, "models: {vllm: true}\n")


def test_rerank_batch_size_default_per_model_and_overridable(tmp_path):
    assert config.default_config().rerank_batch_size == 0  # 0 = the whole pool, as before
    assert _load(tmp_path, f"models: {{reranker: {QWEN_RR}}}\n").rerank_batch_size == 4
    cfg = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: 2}}\n")
    assert cfg.rerank_batch_size == 2
    for bad in ("0", "-1", "'x'", "true"):
        with pytest.raises(config.ConfigError, match="rerank_batch_size"):
            _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: {bad}}}\n")


# ---- fingerprints -------------------------------------------------------------------------------


def test_each_model_changes_the_fingerprint(tmp_path):
    base = config.default_config()
    rr = {n: _load(tmp_path, f"models: {{reranker: {n}}}\n") for n in (QWEN_RR, MXBAI)}
    emb = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}}}\n")
    assert len({base.fp_all(), rr[QWEN_RR].fp_all(), rr[MXBAI].fp_all(), emb.fp_all()}) == 4
    for cfg in rr.values():  # a reranker swap leaves the vectors alone, the gate void
        assert cfg.fp_embed() == base.fp_embed() and cfg.vec_table() == base.vec_table()
    assert emb.fp_embed() != base.fp_embed() and emb.vec_table() != base.vec_table()


def test_batch_size_is_in_the_fingerprint_when_set(tmp_path):
    """MEM2-222 F23: a batch size moves half-precision scores, so a gate fitted at one is void at
    another; at 0 (the default, the whole pool) the fingerprint is the one stores already carry."""
    a = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: 2}}\n")
    b = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: 16}}\n")
    assert a.fp_all() != b.fp_all() and a.fp_embed() == b.fp_embed()
    base = config.default_config()
    assert base.rerank_batch_size == 0
    assert config.Config(**{**base.__dict__, "rerank_batch_size": 0}).fp_all() == base.fp_all()
    assert config.Config(**{**base.__dict__, "rerank_batch_size": 8}).fp_all() != base.fp_all()


# ---- a gate or vec table from another model is refused -------------------------------------------


def _gate(cfg):
    return Gate(1.0, 0.0, 0.5, 0.5, 0.0, fit_id="t", fingerprint_hash=cfg.fp_all())


def test_switch_refuses_a_gate_fitted_for_another_reranker(tmp_path):
    old = config.stub_config()
    new = old.replace("reranker", **{**config.default_config().reranker, "id": "x/other"})
    db = tmp_path / "s.db"
    s = Store.open(db, HashEmbedder(256), fp=old.fp_embed())
    s.close()
    with pytest.raises(ValueError, match="gate was fitted for fingerprint"):
        reembed.switch(db, new, _gate(old))


def test_switch_refuses_a_missing_vec_table_of_the_new_embedder(tmp_path):
    cfg = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}}}\n")
    db = tmp_path / "s.db"
    config_old = config.default_config()
    Store.open(db, HashEmbedder(768), fp=config_old.fp_embed()).close()
    with pytest.raises(ValueError, match="no vec_.*run mem reembed"):
        reembed.switch(db, cfg, _gate(cfg))


def test_serve_preflight_refuses_other_embedder_and_other_reranker(tmp_path):
    base = config.default_config()
    db = tmp_path / "s.db"
    s = Store.open(db, HashEmbedder(768), fp=base.fp_embed())
    write_gate(s.conn, _gate(base))
    s.close()
    assert serve.preflight({"personal": db}, base) == []
    emb = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}}}\n")
    assert "embedder fingerprint mismatch" in serve.preflight({"personal": db}, emb)[0]
    rr = _load(tmp_path, f"models: {{reranker: {MXBAI}}}\n")
    assert "gate fingerprint mismatch" in serve.preflight({"personal": db}, rr)[0]


def test_reembed_builds_one_vec_table_per_fingerprint(tmp_path):
    base = config.default_config()
    qwen = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}}}\n")
    db = tmp_path / "s.db"
    s = Store.open(db, HashEmbedder(768), fp=base.fp_embed(), reranker=None)
    s.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    s.close()
    conn = schema.connect(db)
    n_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    assert n_chunks >= 1
    t1 = reembed.reembed(conn, HashEmbedder(768), base.fp_embed())
    t2 = reembed.reembed(conn, HashEmbedder(1024), qwen.fp_embed())
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'vec_%'")}
    assert t1 != t2 and {t1, t2} <= tables
    for t in (t1, t2):
        assert conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == n_chunks
    conn.close()


# ---- the entry behind a section -----------------------------------------------------------------


def test_a_section_that_is_not_a_catalog_entry_is_refused():
    cfg = config.default_config()
    assert catalog.embedder_entry(cfg.embedder).name == "embeddinggemma-300m"
    assert catalog.reranker_entry(cfg.reranker).name == "bge-reranker-v2-m3"
    with pytest.raises(ValueError, match="not a pinned catalog entry"):
        catalog.reranker_entry({**cfg.reranker, "revision": "0" * 40})
    with pytest.raises(ValueError, match="not a pinned catalog entry"):
        catalog.embedder_entry({**cfg.embedder, "dim": 3})


# ---- prompts ------------------------------------------------------------------------------------


class _Qwen(HashEmbedder):
    prompts = catalog.prompts_of(catalog.EMBEDDERS[QWEN_EMB].section())


def test_default_prompts_are_the_gemma_ones_for_any_embedder_without_prompts():
    assert catalog.query_prompt(HashEmbedder()) == "task: search result | query: "
    assert doc_text("Ctx", "body", HashEmbedder()) == "title: Ctx | text: body"
    assert doc_text("Ctx", "body") == "title: Ctx | text: body"


def test_qwen_prompts_apply_to_queries_and_documents():
    q = catalog.query_prompt(_Qwen())
    assert q.startswith("Instruct: ") and q.endswith("\nQuery:")
    assert doc_text("Ctx", "body", _Qwen()) == "Ctx\nbody"
    assert doc_text("", "body", _Qwen()) == "body"


def test_store_query_vector_uses_the_embedders_query_prompt(tmp_path):
    seen = []

    class Spy(_Qwen):
        def embed_query(self, text):
            seen.append(text)
            return super().embed_query(text)

    s = Store.open(tmp_path / "s.db", Spy(16))
    s.query_vector("what dog")
    assert seen == [catalog.query_prompt(_Qwen()) + "what dog"]
    s.close()


# ---- the VRAM check -----------------------------------------------------------------------------


# Round 9 F host run on 2d50067: nvidia-smi peak - idle, MiB
MEASURED_DEFAULT_GIB = (9461 - 450) / 1024


def test_default_serving_set_fits(tmp_path):
    need = catalog.vram_gib(config.default_config())
    assert need < doctor.VRAM_BUDGET_GIB
    c = doctor.check_vram(config.default_config())
    assert c.status == "ok" and f"{need:.1f} GiB" in c.detail and "12" in c.detail


def test_default_serving_set_estimate_is_within_10_percent_of_the_measured_peak():
    """MEM2-264: the host measured the default set at 9,461 MiB peak over 450 idle (8.8 GiB); the
    first estimate (0.5 GiB context per model) said 6.8, 2 GiB low."""
    need = catalog.vram_gib(config.default_config())
    assert abs(need - MEASURED_DEFAULT_GIB) / MEASURED_DEFAULT_GIB < 0.10


def test_vram_is_weights_plus_batch_activations_plus_one_process_overhead():
    cfg = config.default_config()
    e, r = catalog.embedder_entry(cfg.embedder), catalog.reranker_entry(cfg.reranker)
    want = (e.weights_gib + e.act_gib + r.weights_gib + r.act_gib_per_item * 30
            + catalog.PROCESS_GIB)  # fmt: skip
    assert catalog.vram_gib(cfg) == pytest.approx(want)  # pool 30, batch 0 = the whole pool
    assert [name for name, _ in catalog.vram_parts(cfg)][-1] == "CUDA context + allocator"


def test_a_batch_that_overflows_12_gib_fails_doctor_and_names_the_batch(tmp_path):
    small = _load(tmp_path, f"models: {{reranker: {MXBAI}, rerank_batch_size: 2}}\n")
    big = _load(tmp_path, f"models: {{reranker: {MXBAI}, rerank_batch_size: 8}}\n")
    assert catalog.vram_gib(small) < doctor.VRAM_BUDGET_GIB < catalog.vram_gib(big)
    assert doctor.check_vram(small).status == "ok"
    c = doctor.check_vram(big)
    assert c.status == "fail" and "rerank_batch_size" in c.detail and "12" in c.detail


def test_mxbai_default_batch_fits_the_budget(tmp_path):
    cfg = _load(tmp_path, f"models: {{reranker: {MXBAI}}}\n")
    assert cfg.rerank_batch_size == 2
    assert doctor.check_vram(cfg).status == "ok"


def test_the_4b_reranker_does_not_fit_12_gib_at_any_batch_on_the_recalibrated_estimate(tmp_path):
    """Its bf16 weights (7.5 GiB) + the embedder's (1.1) + the measured overhead are 11.6 GiB
    before any activation: batch 1 is over, and the pool does not enter once a batch is set."""
    for batch in (1, 4):
        cfg = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: {batch}}}\n")
        assert doctor.check_vram(cfg).status == "fail"
    one = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: 1}}\n")
    small_pool = _load(tmp_path, f"models: {{reranker: {QWEN_RR}, rerank_batch_size: 1}}\n"
                       "retrieval: {pool: 10}\n")  # fmt: skip
    assert catalog.vram_gib(one) == pytest.approx(catalog.vram_gib(small_pool))


def test_the_4b_reranker_with_the_fp32_qwen_embedder_does_not_fit_even_at_the_default_batch(
    tmp_path,
):
    cfg = _load(tmp_path, f"models: {{embedder: {QWEN_EMB}, reranker: {QWEN_RR}}}\n")
    c = doctor.check_vram(cfg)
    assert c.status == "fail" and "12" in c.detail  # 3.2 (embedder) + 8.5 (4B, batch 4) + 3.0
    assert catalog.vram_gib(cfg) == pytest.approx(14.7, abs=0.1)


def test_doctor_run_includes_the_vram_check():
    names = [c.name for c in doctor.run(cfg=config.default_config(), device="cpu")]
    assert "vram budget" in names


# ---- the score shape fed to Platt ---------------------------------------------------------------


def test_scores_are_batched_in_order_and_unsquashed():
    calls = []

    def predict(pairs):
        calls.append(len(pairs))
        return [float(len(d)) - 20.0 for _, d in pairs]  # raw, negative and > 1: no activation

    docs = ["a" * n for n in range(1, 11)]
    z = catalog.score_in_batches(predict, "q", docs, batch_size=4)
    assert calls == [4, 4, 2]
    assert z == [float(n) - 20.0 for n in range(1, 11)]
    assert all(isinstance(x, float) for x in z)


def test_batch_size_zero_is_one_batch_and_empty_is_empty():
    calls = []
    z = catalog.score_in_batches(
        lambda p: calls.append(len(p)) or [0.0] * len(p), "q", ["a"] * 7, 0
    )
    assert calls == [7] and len(z) == 7
    assert catalog.score_in_batches(lambda p: pytest.fail("no call"), "q", [], 4) == []


def test_a_wrong_number_of_scores_is_an_error():
    with pytest.raises(ValueError, match="2 scores for 3"):
        catalog.score_in_batches(lambda p: [0.0, 1.0], "q", ["a", "b", "c"], 0)


def test_platt_fits_on_the_raw_logit_difference_of_a_yes_no_reranker():
    # Qwen3-Reranker's card: p(yes) = softmax over [no, yes] = σ(logit_yes − logit_no); the CrossEncoder
    # path returns that difference. Platt on it must recover a ≈ 1 when the labels follow σ(z).
    z = [-6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0] * 5
    y = [False, False, False, True, False, True, True, True] * 5
    a, b = fit_platt(z, y)
    assert a > 0 and abs(b) < 3


# ---- the runbook's Qwen3-Reranker score sanity is the model card's own example ------------------

CARD_QUERY = "What is the capital of China?"
CARD_DOCS = [
    "The capital of China is Beijing.",
    (
        "Gravity is a force that attracts two bodies towards each other. It gives weight to "
        "physical objects and is responsible for the movement of planets around the sun."
    ),
]  # Qwen/Qwen3-Reranker-4B README at 22e68366, which prints [6.4375, -14.375] for these


def _round_9_f():
    from pathlib import Path

    runbook = Path("deliverables/runbooks/p3-gate.md")
    if not runbook.exists():
        pytest.skip("deliverables/ is not part of this tree")
    text = runbook.read_text(encoding="utf-8")
    start = text.index("### Round 9 · F")
    return text[start : text.index("\n### ", start + 1)]


def test_round_9_f_score_sanity_feeds_the_cards_documents_and_checks_its_numbers():
    """MEM2-264: the host got [6.3125, -11.8125] against the card's [6.4375, -14.375] because the
    runbook scored only the first sentence of the card's second document. The same inputs, or
    the comparison means nothing."""
    import ast

    sec = _round_9_f()
    i = sec.index("QWEN_CARD = ")
    card = ast.literal_eval(sec[i + len("QWEN_CARD = ") : sec.index("\n", i)])
    assert card["query"] == CARD_QUERY and card["docs"] == CARD_DOCS
    assert card["expect"] == [6.4375, -14.375] and card["tol"] == 1.0
    assert sec.index("QWEN_CARD = ") < sec.index("r9_gate f-rr4b")  # sanity first, then the rerun
