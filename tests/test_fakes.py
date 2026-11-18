import math

from mem.testing import HashEmbedder, OverlapReranker


def test_embedder_is_deterministic_unit_norm_and_dim():
    e = HashEmbedder(256)
    a = e.embed_query("weekly one on one")
    assert a == e.embed_query("weekly one on one") and len(a) == 256
    assert math.isclose(sum(x * x for x in a), 1.0)


def test_shared_words_are_closer_than_disjoint():
    e = HashEmbedder()
    q = e.embed_query("ana meeting")
    near, far = e.embed_docs(["meeting with ana", "grocery list"])
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))
    assert dot(q, near) > dot(q, far)
    assert dot(q, far) == 0.0


def test_empty_text_embeds_to_zero_vector():
    assert HashEmbedder(8).embed_query("") == [0.0] * 8


def test_reranker_logits_are_exact_and_ordered():
    r = OverlapReranker()
    assert r.score("ana thursday", ["ana on thursday", "ana only", "nothing"]) == [3.0, 0.0, -3.0]
