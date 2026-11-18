import hashlib

import pytest

from mem import config

pytestmark = pytest.mark.release_defaults  # the default config itself (MEM2-261)

GOLDEN_GEMMA_JSON = (
    '{"backend":"sentence-transformers","dim":768,"doc_prompt":"title: none | text: ",'
    '"dtype":"fp32","id":"google/embeddinggemma-300m","max_len":512,"normalize":true,'
    '"query_prompt":"task: search result | query: ",'
    '"revision":"57c266a740f537b4dc058e1b0cda161fd15afa75"}'
)  # the embedder of every default up to MEM2-273, and of the Round 8 / all-off config
GOLDEN_QWEN_JSON = (
    '{"backend":"sentence-transformers","dim":1024,"doc_format":"{ctx}\\n{text}",'
    '"doc_format_no_ctx":"{text}","doc_prompt":"","dtype":"fp32","id":"Qwen/Qwen3-Embedding-0.6B",'
    '"max_len":512,"normalize":true,"padding_side":"left","pooling":"last-token",'
    '"query_prompt":"Instruct: Given a web search query, retrieve relevant passages that answer the query'
    '\\nQuery:","revision":"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}'
)  # MEM2-274: the default embedder (ADR 0025, Round 10 r10-all)
GOLDEN_FP_EMBED_GEMMA = "a8d84766ba2193a070eb2e27c183eab0fe000ccdc4d3f53fc7ffe8654816a47b"
GOLDEN_FP_EMBED = "82244f329c7199abeb185be8f0f05a726f40e256ff60fa6f29d3a776e1818086"  # MEM2-274: Qwen3-Embedding-0.6B
GOLDEN_FP_ALL = "3f8c9c11ed6e26fed85e2c0f262ca27db519fff07ef89f15591120a222bdea6f"  # MEM2-274: the default = the host's final config data/r9/final/config.yaml, Round 10 r10-all (9c + read.list_fill 16 + read.window_boost {delta 3.0, slack 0.1} + models.embedder qwen3-embedding-0.6b)
GOLDEN_FP_ALL_9C = "4b73eb54d800b81fec22ea069a04c2ccec4d2833ab751ef71a0ec7de08fd31b3"  # MEM2-261 to MEM2-273: Round 9's final config, F1 + F2+F3 on, Gemma embedder; its gate is the host's 9c one
R10_ALL_YAML = (
    "models: {embedder: qwen3-embedding-0.6b}\n"
    "read: {window_boost: {delta: 3.0, slack: 0.1}, list_fill: 16}\n"
)  # the Round 10 r10-all arm over the 9c defaults (p3-gate.md Round 10, steps 4-5)
GOLDEN_FP_ALL_OFF = "55f53d2e2baba49616bfa0631b3396df15fd59660dd89060fe6e696ea97ac613"  # every switch off: the Round 8 default, MEM2-183 to MEM2-260 (was eaabca64… at MEM2-164: pool 30, guard off)
# the single-switch goldens below are each one switch on over GOLDEN_FP_ALL_OFF
GOLDEN_FP_ALL_RESOLVE_DATES = "1001d404f53f0dd529fe1ae7abfa8b52fa94ffb374870ba43915627e42409fa0"  # MEM2-192, ADR 0021: ingest.resolve_dates on; off is GOLDEN_FP_ALL_OFF, unchanged
GOLDEN_FP_ALL_SESSION_LEG = "d4a85f93ec15b6d373f9f3975bcd9a4c88541309fba96525c9f201425a49bbd5"  # MEM2-195, ADR 0022: retrieval.session_leg on; off is GOLDEN_FP_ALL_OFF, unchanged
GOLDEN_FP_ALL_SPLIT_ENDPOINTS = "72fe0e9513c841c7a6f819dfd56f1f6576565c2261e3f55f7cd78f81dec2b1a3"  # MEM2-213, ADR 0026: read.split_endpoints on; off is GOLDEN_FP_ALL_OFF, unchanged
GOLDEN_FP_ALL_WINDOW_BOOST = "8563682dd840bb620b03f0c39972b6b2de388d393e05a0752f6d0d0e90482633"  # MEM2-266, ADR 0031: read.window_boost on at the tuned {delta: 1.0, slack: 0.1}; off is GOLDEN_FP_ALL_OFF, unchanged
GOLDEN_FP_ALL_LIST_MODE = "74bf0417138248c1c98406c651807271982267dd50b6efa6a3021fc39afffe50"  # MEM2-193, ADR 0023: read.list_mode on; off is GOLDEN_FP_ALL_OFF, unchanged
GOLDEN_FP_ALL_LIST_PAGING = "8d21307d0cdb467a89e4068ef1d91108990faffe31c663366aba75cbd0bba3b2"  # MEM2-212, ADR 0024: read.list_mode and read.list_paging on; paging off is GOLDEN_FP_ALL_LIST_MODE, unchanged
GOLDEN_FP_ALL_NEW_CHECKS_OFF = "3961c21cb008916527fea7cf62169d4ddc968028d81fec636c62f9cdde1b8ee9"  # MEM2-272, ADR 0027 "Switch": ingest.new_checks_existing false over the default (absent / true is GOLDEN_FP_ALL); MEM2-274: over the r10-all default (was dac44d3c… over 9c)
GOLDEN_FP_ALL_LIST_FILL = "3510b26af7d359526bb3b54a5a756c3a191eff28961de410a08258dedc8c7516"  # MEM2-268, ADR 0030: read.list_mode on and read.list_fill 16; fill off (0 / absent) is GOLDEN_FP_ALL_LIST_MODE, unchanged


def all_off() -> config.Config:
    """Every switch off (the Round 8 default): the base the single-switch goldens are taken over."""
    return config.all_off_config()


def test_embedder_section_canonical_json_is_frozen():
    assert config.canonical_json(config.default_config().embedder) == GOLDEN_QWEN_JSON
    assert config.canonical_json(config.all_off_config().embedder) == GOLDEN_GEMMA_JSON


def test_golden_fingerprints():
    cfg = config.default_config()
    assert cfg.fp_embed() == hashlib.sha256(GOLDEN_QWEN_JSON.encode()).hexdigest()
    assert cfg.fp_embed() == GOLDEN_FP_EMBED
    assert cfg.fp_all() == GOLDEN_FP_ALL
    cfg = all_off()
    assert cfg.fp_embed() == GOLDEN_FP_EMBED_GEMMA
    assert cfg.fp_all() == GOLDEN_FP_ALL_OFF
    assert cfg.replace("ingest", resolve_dates=True).fp_all() == GOLDEN_FP_ALL_RESOLVE_DATES
    assert cfg.replace("retrieval", session_leg=True).fp_all() == GOLDEN_FP_ALL_SESSION_LEG
    assert cfg.replace("read", list_mode=True).fp_all() == GOLDEN_FP_ALL_LIST_MODE
    assert (
        cfg.replace("read", list_mode=True, list_paging=True).fp_all() == GOLDEN_FP_ALL_LIST_PAGING
    )
    assert cfg.replace("read", split_endpoints=True).fp_all() == GOLDEN_FP_ALL_SPLIT_ENDPOINTS
    assert cfg.replace("read", list_mode=True, list_fill=16).fp_all() == GOLDEN_FP_ALL_LIST_FILL
    on = cfg.replace("read", window_boost={"delta": 1.0, "slack": 0.1})
    assert on.fp_all() == GOLDEN_FP_ALL_WINDOW_BOOST


def test_default_is_the_hosts_final_config_r10_all():
    """MEM2-274 (the owner, MEM2-1, 30 Oct): the shipped default is the host's final config
    `data/r9/final/config.yaml` = Round 10 `r10-all`: Round 9's F1 + F2+F3, plus `read.list_fill` 16,
    `read.window_boost` {delta 3.0, slack 0.1} and the Qwen3-Embedding-0.6B embedder. Every other switch
    stays as on 09e1bb7 (`ingest.new_checks_existing` on)."""
    cfg = config.default_config()
    assert cfg.resolve_dates is True and cfg.list_mode is True and cfg.new_checks_existing is True
    assert cfg.ingest == {"resolve_dates": True}
    assert cfg.read == {
        "conflict_guard": False, "p3_model": False, "list_mode": True, "list_fill": 16,
        "window_boost": {"delta": 3.0, "slack": 0.1},
    }  # fmt: skip
    assert cfg.list_fill == 16 and cfg.window_boost == {"delta": 3.0, "slack": 0.1}
    assert not (cfg.list_paging or cfg.split_endpoints or cfg.session_leg or cfg.conflict_guard
                or cfg.p3_model or cfg.at_rest_required)  # fmt: skip
    assert cfg.rerank_batch_size == 0 and "session_leg" not in cfg.retrieval
    assert cfg.embedder == config.catalog.EMBEDDERS["qwen3-embedding-0.6b"].section()
    assert cfg.reranker == config.catalog.RERANKERS["bge-reranker-v2-m3"].section()
    assert cfg.fp_all() == GOLDEN_FP_ALL and cfg.fp_all().startswith("3f8c9c11")


def test_the_r10_all_file_and_an_empty_file_are_the_default_fingerprint(tmp_path):
    p = tmp_path / "c.yaml"
    explicit = (
        "models: {embedder: qwen3-embedding-0.6b, reranker: bge-reranker-v2-m3}\n"
        "ingest: {resolve_dates: true, new_checks_existing: true}\n"
        "read: {list_mode: true, list_fill: 16, window_boost: {delta: 3.0, slack: 0.1}}\n"
    )
    for text in ("", R10_ALL_YAML, explicit):
        p.write_text(text)
        assert config.load(p).fp_all() == GOLDEN_FP_ALL, text


def test_the_9c_config_is_still_loadable_with_its_fingerprint(tmp_path):
    """A store and gate of Round 9's final config (MEM2-261 to MEM2-273) keep their fingerprint when
    the file writes the Round 10 additions off and names the Gemma embedder."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "models: {embedder: embeddinggemma-300m}\nread: {list_fill: 0, window_boost: false}\n"
    )
    assert config.load(p).fp_all() == GOLDEN_FP_ALL_9C


def test_both_switches_off_in_the_file_is_the_round_8_fingerprint(tmp_path):
    """A store and gate fitted on the old default keep loading with the two switches written off."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "models: {embedder: embeddinggemma-300m}\ningest: {resolve_dates: false}\n"
        "read: {list_mode: false, list_fill: 0, window_boost: false}\n"
    )
    cfg = config.load(p)
    assert cfg.fp_all() == GOLDEN_FP_ALL_OFF and not (cfg.resolve_dates or cfg.list_mode)
    assert "list_mode" not in cfg.read


def test_retrieval_defaults_match_spec():
    r = config.default_config().retrieval
    assert r == {
        "fts_tokenizer": "porter unicode61 remove_diacritics 2", "leg_k": 50, "rrf_k": 60,
        "pool": 30, "per_node_cap": 4, "limit": 8, "budget": 600,
    }  # fmt: skip


def test_reranker_has_raw_logit_extraction_and_512():
    rr = config.default_config().reranker
    assert rr["id"] == "BAAI/bge-reranker-v2-m3" and rr["max_len"] == 512
    assert rr["activation"] == "none" and rr["dtype"] == "fp32"


@pytest.mark.parametrize(
    ("section", "field", "value", "changes_embed"),
    [
        ("embedder", "dtype", "bf16", True),
        ("embedder", "revision", "0" * 40, True),
        ("reranker", "max_len", 256, False),
        ("retrieval", "pool", 20, False),
    ],
)
def test_pinned_field_change_changes_fingerprint(section, field, value, changes_embed):
    base = config.default_config()
    changed = base.replace(section, **{field: value})
    assert changed.fp_all() != base.fp_all()
    assert (changed.fp_embed() != base.fp_embed()) is changes_embed


def test_device_and_gate_are_outside_fingerprints():
    base = config.default_config()
    other = base.replace("gate", platt_a=2.0, tau_keep=0.5)
    assert other.fp_all() == base.fp_all()
    assert config.default_config().device == "cuda"


def test_vec_table_name_from_fp_embed():
    cfg = config.default_config()
    assert cfg.vec_table() == "vec_" + cfg.fp_embed()[:8]


def test_load_yaml_overrides_non_pinned_and_roundtrips(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("device: cpu\nretrieval:\n  limit: 5\n")
    cfg = config.load(p)
    assert cfg.device == "cpu" and cfg.retrieval["limit"] == 5
    assert cfg.fp_embed() == config.default_config().fp_embed()


def test_load_rejects_unknown_section_and_bad_device(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("bogus: 1\n")
    with pytest.raises(config.ConfigError):
        config.load(p)
    p.write_text("device: tpu\n")
    with pytest.raises(config.ConfigError):
        config.load(p)


def test_default_device_is_cuda_and_outside_the_fingerprint():
    import dataclasses

    from mem import config

    cfg = config.default_config()
    assert cfg.device == "cuda"
    assert config.DEVICES == ("auto", "cpu", "cuda")
    assert dataclasses.replace(cfg, device="cpu").fp_all() == cfg.fp_all()
    assert dataclasses.replace(cfg, device="cpu").fp_embed() == cfg.fp_embed()


def test_read_conflict_guard_is_in_the_fingerprint_and_defaults_off(tmp_path):
    """ADR 0018: `read.conflict_guard` is a setting of the read path, inside fp_all (a gate fitted
    with it on is void with it off); default off (the owner, 2026-10-25); only true/false is accepted."""
    base = config.default_config()
    assert base.conflict_guard is False and base.read["conflict_guard"] is False
    on = base.replace("read", conflict_guard=True)
    assert on.fp_all() != base.fp_all() and on.fp_embed() == base.fp_embed()
    p = tmp_path / "c.yaml"
    p.write_text("read:\n  conflict_guard: true\n")
    assert config.load(p).conflict_guard is True and config.load(p).fp_all() == on.fp_all()
    p.write_text("read:\n  conflict_guard: maybe\n")
    with pytest.raises(config.ConfigError, match="true or false"):
        config.load(p)
    p.write_text("read:\n  other: 1\n")
    with pytest.raises(config.ConfigError, match="unknown field"):
        config.load(p)


def test_security_at_rest_required_defaults_off_and_is_outside_the_fingerprint(tmp_path):
    """the owner, 2026-10-25: encryption at rest is deferred (owner's backlog); the doctor warns until
    `security.at_rest_required: true`. A policy switch, not a retrieval setting: not in fp_all."""
    base = config.default_config()
    assert base.security == {"at_rest_required": False} and base.at_rest_required is False
    on = base.replace("security", at_rest_required=True)
    assert on.at_rest_required is True and on.fp_all() == base.fp_all()
    p = tmp_path / "c.yaml"
    p.write_text("security:\n  at_rest_required: true\n")
    assert config.load(p).at_rest_required is True
    p.write_text("security:\n  at_rest_required: yes please\n")
    with pytest.raises(config.ConfigError, match="true or false"):
        config.load(p)
    p.write_text("security:\n  other: 1\n")
    with pytest.raises(config.ConfigError, match="unknown field"):
        config.load(p)


def test_all_off_config_is_the_round_8_fingerprint():
    off = config.all_off_config()
    assert off.fp_all() == GOLDEN_FP_ALL_OFF and off.fp_embed() == GOLDEN_FP_EMBED_GEMMA
    assert not (off.list_mode or off.resolve_dates or off.list_paging or off.split_endpoints
                or off.session_leg or off.conflict_guard or off.p3_model or off.list_fill
                or off.window_boost)  # fmt: skip


def test_new_checks_existing_is_on_by_default_and_out_of_the_fingerprint(tmp_path):
    """MEM2-272 (ADR 0027 "Switch"): `ingest.new_checks_existing` defaults to true. Absent or written
    true, the key is not in the section, so the default `fp_all` (and every store and gate on it) is
    unchanged; written false, it joins the `ingest` section and so `fp_all`."""
    cfg = config.default_config()
    assert cfg.new_checks_existing is True and "new_checks_existing" not in cfg.ingest
    on = cfg.replace("ingest", new_checks_existing=True)
    assert on.ingest == cfg.ingest and on.fp_all() == GOLDEN_FP_ALL
    p = tmp_path / "c.yaml"
    p.write_text("ingest: {new_checks_existing: true}\n")
    assert config.load(p).fp_all() == GOLDEN_FP_ALL
    p.write_text("ingest: {new_checks_existing: false}\n")
    off = config.load(p)
    assert off.new_checks_existing is False
    assert off.ingest == {"resolve_dates": True, "new_checks_existing": False}
    assert off.fp_all() == GOLDEN_FP_ALL_NEW_CHECKS_OFF != GOLDEN_FP_ALL
    assert (
        off.fp_all()
        == config.default_config().replace("ingest", new_checks_existing=False).fp_all()
    )
    # with resolve_dates off the section is otherwise left out; the switch off still enters it
    base = all_off()
    assert base.replace("ingest", new_checks_existing=True).fp_all() == GOLDEN_FP_ALL_OFF
    assert base.replace("ingest", new_checks_existing=False).fp_all() not in (
        GOLDEN_FP_ALL_OFF, GOLDEN_FP_ALL_NEW_CHECKS_OFF
    )  # fmt: skip
    p.write_text("ingest: {new_checks_existing: 'no'}\n")
    with pytest.raises(config.ConfigError, match="new_checks_existing"):
        config.load(p)
