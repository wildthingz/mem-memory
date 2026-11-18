"""Option C, pseudo-relevance expansion terms (MEM2-216): selection on hand-made chunks, a stub tagger."""

import pytest

from mem import prf


def stub(text: str) -> prf.Tagged:
    """Capitalised words are names, lower-case words of 4+ letters are nouns; no model."""
    words = [w.strip(".,") for w in text.split()]
    return prf.Tagged(
        nouns=[w.lower() for w in words if w.isalpha() and len(w) >= 4],
        names=[w for w in words if w[:1].isupper() and len(w) > 1],
    )


def test_noun_in_two_chunks_is_chosen_and_in_one_is_not():
    chunks = ["garden tomatoes", "garden basil", "kitchen lamps"]
    assert prf.expansion_terms("what", chunks, stub, min_chunks=2) == ["garden"]


def test_a_noun_counts_once_per_chunk_not_per_mention():
    assert prf.expansion_terms("x", ["tomato tomato tomato", "lamps"], stub, min_chunks=2) == []


def test_terms_already_in_the_query_are_dropped():
    chunks = ["garden tomatoes", "garden basil"]
    assert prf.expansion_terms("my Garden?", chunks, stub) == []


def test_name_in_one_chunk_is_chosen_unless_the_query_has_it():
    chunks = ["met Priya today", "nothing here"]
    assert prf.expansion_terms("who", chunks, stub, min_chunks=2) == ["Priya"]
    assert prf.expansion_terms("did priya call", chunks, stub, min_chunks=2) == []


def test_only_the_first_top_chunks_are_read():
    chunks = ["alpha"] * 2 + ["beta"] * 2
    assert prf.expansion_terms("q", chunks, stub, top=2, min_chunks=2) == ["alpha"]


def test_order_is_by_chunk_count_then_alphabetical_nouns_before_names():
    chunks = ["zebra apple Oslo", "zebra apple", "zebra mango", "mango Bern"]
    assert prf.expansion_terms("q", chunks, stub, cap=10) == [
        "zebra",
        "apple",
        "mango",
        "Bern",
        "Oslo",
    ]


def test_cap_keeps_the_first_terms_of_that_order():
    chunks = ["zebra apple Oslo", "zebra apple", "zebra mango", "mango Bern"]
    assert prf.expansion_terms("q", chunks, stub, cap=2) == ["zebra", "apple"]


def test_empty_chunks_give_no_terms():
    assert prf.expansion_terms("q", [], stub) == []


def test_expanded_query_appends_the_terms():
    assert prf.expand_query("what plant", ["garden", "Priya"]) == "what plant garden Priya"
    assert prf.expand_query("what plant", []) == "what plant"


def test_real_spacy_model_tags_nouns_and_names():
    pytest.importorskip("spacy")
    spacy_tagger = prf.spacy_tagger
    try:
        tag = spacy_tagger()
    except prf.NlpUnavailable:
        pytest.skip("en_core_web_sm not installed (extra `nlp`)")
    t = tag("Barack Obama planted tomatoes in his garden in Paris.")
    assert "tomato" in t.nouns and "garden" in t.nouns
    assert "Barack Obama" in t.names and "Paris" in t.names


# ---- the second pass: bench prf_check.expanded_pool (Option C is not in serving, MEM2-249) -----


@pytest.fixture
def store(tmp_path):
    from mem import write
    from mem.config import stub_config
    from mem.store import Store
    from mem.testing import HashEmbedder, OverlapReranker

    s = Store.open(tmp_path / "p.db", HashEmbedder(), tz="UTC", retrieval=stub_config().retrieval,
                   reranker=OverlapReranker())  # fmt: skip
    texts = ["alpha one", "bravo two", "charlie three", "delta four", "echo five", "foxtrot six"]
    write.ingest_session(s, source_ref="lme:q:s1", date="2023-05-20T02:21:00Z",
                         rounds=[(t, "") for t in texts])  # fmt: skip
    yield s
    s.close()


def _ids(pool):
    return [cid for cid, _ in pool]


def test_the_serving_pool_has_no_expanded_leg(store):
    import inspect

    from mem import read

    assert "expand" not in inspect.signature(read.candidate_pool).parameters
    t = "2023-06-01T00:00:00Z"
    a, _ = read.candidate_pool(store, "alpha one", t=t, size=2)
    read.build_allowed(store.conn, t)
    k = store.retrieval["leg_k"]
    legs = [read.fts_leg(store.conn, "alpha one", k),
            read.vec_leg(store.conn, store.vec_table, store.query_vector("alpha one"), k)]  # fmt: skip
    assert read.pool_from_legs(store, legs, 2) == a  # the split-out fusion is the same pool


def test_expand_adds_the_expanded_queries_fts_and_vec_legs(store):
    from bench.prf_check import expanded_pool
    from mem import read

    t = "2023-06-01T00:00:00Z"
    before, _ = read.candidate_pool(store, "alpha one", t=t, size=2)
    after = expanded_pool(store, "alpha one", "foxtrot six", t=t, size=2)
    (fox,) = store.conn.execute("SELECT id FROM chunks WHERE text = 'foxtrot six'").fetchone()
    assert fox not in _ids(before)
    assert fox in _ids(after)
