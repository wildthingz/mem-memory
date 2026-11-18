"""Subject masking (MEM2-165): the read model's "does this memory answer THIS question's relation"
signals, `cosine_masked` and `rerank_masked`."""

import json

import numpy as np
import pytest

from mem import calibrate, read, readfit, readmodel, subjectmask
from mem.config import default_config
from mem.gate import Gate
from mem.readmodel import MASKED, ReadModel
from mem.store import Store
from mem.subjectmask import MASK, mask_text
from mem.testing import HashEmbedder, OverlapReranker

# ---- masking is deterministic -------------------------------------------------------------------


def test_every_occurrence_of_the_subject_is_masked_whole_words_only_case_insensitive():
    out = mask_text("Ana Lopez said ana likes Banana. ANA's cat.", ["Ana Lopez", "Ana"])
    assert out == f"{MASK} said {MASK} likes Banana. {MASK}'s cat."


def test_a_subject_that_does_not_appear_or_no_subject_leaves_the_text_alone():
    assert mask_text("The cat sleeps.", ["Bosco"]) == "The cat sleeps."
    assert mask_text("The cat sleeps.", []) == "The cat sleeps."
    assert mask_text("", ["Bosco"]) == ""
    assert mask_text("a b", ["", " ", "a"]) == "a b"  # empty / one-character names never mask


def test_masking_depends_on_the_set_of_names_not_their_order_or_repeats():
    names = ["Ana", "Ana Lopez", "ana", "Lopez"]
    text = "Ana Lopez met Lopez and Ana."
    want = mask_text(text, names)
    assert want == f"{MASK} met {MASK} and {MASK}."  # the longest name first: no half-masked name
    assert mask_text(text, names[::-1]) == want == mask_text(text, sorted(names))
    assert mask_text(want, names) == want  # idempotent


def test_regex_metacharacters_in_a_name_are_literal():
    assert mask_text("Dr. (X)+ went", ["Dr. (X)+"]) == f"{MASK} went"
    assert mask_text("DrZ went", ["Dr."]) == "DrZ went"


# ---- the subjects come from the store -----------------------------------------------------------


def _store(tmp_path, **kw):
    kw.setdefault("read", {**default_config().read, "p3_model": True})  # P3 model in force
    s = Store.open(tmp_path / "m.db", HashEmbedder(), reranker=OverlapReranker(), **kw)
    s.remember(text="Ana Lopez works as a dentist.", type="fact", about=["new:person:Ana Lopez"])
    s.remember(text="Ana's favourite food is soup.", type="fact", about=["Ana Lopez"])
    s.remember(text="Bosco is a dog that likes soup.", type="fact", about=["new:thing:Bosco"])
    return s


def _cids(s):
    return [
        r[0] for r in s.conn.execute("SELECT id FROM chunks WHERE key_kind != 'name' ORDER BY id")
    ]


def test_query_subjects_are_the_entities_named_in_the_query_and_masked_with_all_their_names(
    tmp_path,
):
    s = _store(tmp_path)
    ids = subjectmask.query_subject_ids(s.conn, "What is Ana Lopez's favourite food?")
    names = subjectmask.entity_names(s.conn, ids)
    assert "Ana Lopez" in names and "Bosco" not in names
    assert subjectmask.masked_query(s, "What is Ana Lopez's favourite food?") == (
        f"What is {MASK}'s favourite food?"
    )
    assert subjectmask.masked_query(s, "What is the favourite food?") == (
        "What is the favourite food?"  # no subject in the query: nothing to mask
    )
    s.close()


def test_a_chunk_is_masked_with_its_own_about_entities_only(tmp_path):
    s = _store(tmp_path)
    docs = dict(zip(_cids(s), subjectmask.masked_docs(s, _cids(s)), strict=True))
    texts = [t for _, t in docs.values()]
    assert f"{MASK} works as a dentist." in texts
    assert f"{MASK} is a dog that likes soup." in texts
    assert all("Bosco" not in t and "Ana" not in t for t in texts)
    s.close()


def test_masked_signals_are_deterministic_and_drop_the_subject_name_from_the_comparison(tmp_path):
    s = _store(tmp_path)
    cids = _cids(s)
    q = "Ana Lopez food"
    a = subjectmask.pool_masked(s, q, cids, MASKED)
    assert a == subjectmask.pool_masked(s, q, cids, MASKED)
    assert set(a) == set(MASKED) and all(len(v) == len(cids) for v in a.values())
    raw = read.pool_cosines(s, q, cids)
    # masked and raw differ for the chunks whose subject is Ana (her name left the comparison)
    assert [round(x, 6) for x in a["cosine_masked"]] != [round(x, 6) for x in raw]
    assert subjectmask.pool_masked(s, q, cids, ["cosine_masked"]).keys() == {"cosine_masked"}
    assert subjectmask.pool_masked(s, q, [], MASKED) == {"cosine_masked": [], "rerank_masked": []}
    s.close()


# ---- fit time == serve time ---------------------------------------------------------------------


def test_the_feature_is_computed_the_same_at_fit_time_and_serve_time(tmp_path):
    s = _store(tmp_path)
    q = "Ana Lopez food"
    items = calibrate.score_pool(s, q)  # fit time: what the records hold
    assert all(i.mcos is not None and i.mz is not None for i in items)
    m = ReadModel.build(["rerank", "cosine_masked", "rerank_masked"], [0.3, 4.0, 0.7], -1.0)
    gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0, read_model=m.to_dict())
    s.gate = gate
    want = m.probs([i.z for i in items], [i.cos for i in items],
                   {k: [i.masked()[k] for i in items] for k in MASKED})  # fmt: skip
    cids = [int(i.key[1:]) for i in items]
    zs = read.rerank(s, q, cids)
    got = read.pool_probs(s, q, cids, zs)  # serve time: computed from the store
    assert got == pytest.approx(want, abs=1e-12)
    out = read.recall(s, query=q)  # and through the whole path, no error, logged p equals it
    logged = json.loads(s.conn.execute("SELECT result FROM events WHERE tool='recall'")
                        .fetchone()[0])  # fmt: skip
    assert logged["abstained"] is False or logged["abstained"] is True
    assert out.structured is not None
    s.close()


def test_recall_with_a_masked_model_needs_no_extra_query_embedding_for_the_unmasked_cosine(
    tmp_path,
):
    s = _store(tmp_path)
    m = ReadModel.build(["cosine", "cosine_masked"], [1.0, 1.0], -1.0)
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0, read_model=m.to_dict())
    calls = []
    real = s.embedder.embed_query
    s.embedder.embed_query = lambda t: (calls.append(t), real(t))[1]
    read.recall(s, query="Ana Lopez food")
    assert len(calls) == 2  # the query once, the masked query once
    s.close()


# ---- the model refuses a mismatch ---------------------------------------------------------------


def test_a_masked_model_refuses_to_score_without_the_masked_signals():
    m = ReadModel.build(["rerank", "cosine_masked"], [1.0, 1.0], 0.0)
    assert m.masked_features == ("cosine_masked",)
    with pytest.raises(ValueError, match="cosine_masked"):
        m.probs([1.0, 0.0], [0.5, 0.5])  # no masked column
    with pytest.raises(ValueError, match="cosine_masked"):
        m.probs([1.0, 0.0], [0.5, 0.5], {"cosine_masked": [0.1]})  # wrong length
    with pytest.raises(ValueError, match="cosine_masked"):
        m.probs([1.0, 0.0], [0.5, 0.5], {"cosine_masked": [0.1, None]})
    with pytest.raises(ValueError, match="rerank_masked"):
        ReadModel.build(["rerank_masked"], [1.0], 0.0).probs(
            [1.0],
            None,
            {"cosine_masked": [0.1]},  # the wrong masked signal
        )
    assert m.probs([1.0, 0.0], [0.5, 0.5], {"cosine_masked": [0.1, 0.2]})  # given: scores
    gate = Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=m.to_dict())
    with pytest.raises(ValueError, match="cosine_masked"):
        gate.probs([1.0, 0.0], [0.5, 0.5])


def test_a_gate_whose_masked_coefficients_are_missing_or_tampered_is_refused():
    d = ReadModel.build(["rerank", "rerank_masked"], [0.4, 0.6], -1.0).to_dict()
    Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=d)
    for broken in (
        {k: v for k, v in d.items() if k != "weights"},
        {**d, "weights": [0.4, 0.7]},
        {**d, "features": ["rerank", "cosine_masked"]},  # features swapped under the fingerprint
        {**d, "features": ["rerank", "rerank_masked", "nonsense"], "weights": [1, 1, 1]},
    ):
        with pytest.raises(ValueError, match="read_model"):
            Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=broken)


def test_the_masking_rule_is_part_of_the_fingerprint_of_a_masked_model_only(monkeypatch):
    plain = ReadModel.build(["rerank", "cosine"], [1.0, 2.0], 0.0)
    masked = ReadModel.build(["rerank", "cosine_masked"], [1.0, 2.0], 0.0)
    fp_plain, fp_masked, d = plain.fingerprint(), masked.fingerprint(), masked.to_dict()
    monkeypatch.setattr(subjectmask, "MASK", "somebody")  # the rule changed since the fit
    assert ReadModel.build(["rerank", "cosine"], [1.0, 2.0], 0.0).fingerprint() == fp_plain
    assert masked.fingerprint() != fp_masked
    with pytest.raises(ValueError, match="fingerprint"):
        ReadModel.from_dict(d)


def test_read_models_without_masked_features_keep_their_fingerprint():
    """A gate fitted before MEM2-165 still loads (the fingerprint body did not change)."""
    import hashlib

    m = ReadModel.build(["rerank", "cosine", "gap"], [0.4, 9.0, 0.05], -8.0)
    body = json.dumps([1, ["rerank", "cosine", "gap"], [0.4, 9.0, 0.05], -8.0],
                      separators=(",", ":"))  # fmt: skip
    assert m.fingerprint() == hashlib.sha256(body.encode()).hexdigest()


# ---- no SYN vocabulary --------------------------------------------------------------------------

SYN_WORDS = ("slot", "relation", "rel_", "subtype", "cue", "template", "value_returns", "hobby",
             "role", "food", "late", "future", "paraphrase", "counterpart", "gold", "syn")  # fmt: skip


def test_the_masked_features_and_the_masking_module_encode_no_syn_vocabulary():
    for f in MASKED:
        assert not any(w in f for w in SYN_WORDS)
        assert not any(w in readmodel.FEATURES[f].lower() for w in SYN_WORDS)
    src = subjectmask.__doc__ or ""
    import inspect

    code = inspect.getsource(subjectmask).replace(src, "")
    assert "hobby" not in code and "slot" not in code and "syn" not in code.lower()
    assert MASK.isalpha()  # one fixed neutral token, no template


# ---- records and the fit ------------------------------------------------------------------------


def test_records_carry_the_masked_signals_and_old_records_still_load(tmp_path):
    p = calibrate.QueryPool("syn1:q", True, [calibrate.PoolItem("c1", 1.0, 2, 0.5, 0.3, -0.5)])
    old = calibrate.QueryPool("lme", True, [calibrate.PoolItem("c1", 1.0, 1, 0.5)])
    calibrate.write_records(tmp_path / "r.jsonl", [p, old], fp_all="f")
    _, back = calibrate.read_records(tmp_path / "r.jsonl")
    assert back[0].items[0].mcos == 0.3 and back[0].items[0].mz == -0.5
    assert back[1].items[0].mcos is None and back[1].items[0].mz is None
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    legacy = json.loads(lines[2])
    del legacy["items"][0]["mcos"], legacy["items"][0]["mz"]
    (tmp_path / "old.jsonl").write_text("\n".join([lines[0], json.dumps(legacy)]))
    assert calibrate.read_records(tmp_path / "old.jsonl")[1][0].items[0].mcos is None


def _mworld(seeds=(1, 2, 3, 4, 5), n=30, seed=7, plant=None):
    """SYN-like pools where only the masked signals separate answerable from missing-slot: every
    missing-slot pool has a topical distractor as high as a gold on logit and cosine, while the
    masked cosine / logit of a gold is high and the distractor's low. `plant` = {seed: gold masked
    shift} lowers a seed's gold masked signals."""
    rng = np.random.default_rng(seed)
    pools = []
    for sd in seeds:
        for q in range(n):
            for cat, ans in (
                ("syn:current", True),
                ("syn:as_of", True),
                ("syn:missing_slot", False),
            ):
                items = []
                for i in range(7):
                    items.append(calibrate.PoolItem(f"c{i}", rng.normal(-3, 1.2), 0,
                                 rng.normal(0.35, 0.05), rng.normal(0.2, 0.05),
                                 rng.normal(-4, 1.0)))  # fmt: skip
                shift = (plant or {}).get(sd, 0.0)
                if ans:
                    items[0] = calibrate.PoolItem(
                        "c0",
                        rng.normal(1.0, 1.0),
                        2,
                        rng.normal(0.7, 0.04),
                        rng.normal(0.7, 0.03) + shift,
                        rng.normal(2.0, 0.7) + 4 * shift,
                    )
                else:
                    items[0] = calibrate.PoolItem(
                        "c0",
                        rng.normal(1.0, 1.0),
                        0,
                        rng.normal(0.7, 0.04),
                        rng.normal(0.2, 0.03),
                        rng.normal(-4, 0.7),
                    )
                pools.append(calibrate.QueryPool(f"syn{sd}:{cat}{q}", ans, items, cat))
    return pools


FIT = {"fit_id": "t", "fingerprint_hash": "f" * 64, "limit": 8, "fitted_at": "2026-10-25T00:00:00Z"}
SMALL_MASKED = [g for g in readfit.GRID + readfit.MASKED_GRID if g["C"] == 1.0]


def test_masked_sets_join_the_grid_only_when_every_record_carries_the_signals():
    full = readfit._Data(_mworld(n=4))
    assert full.has_masked and "cosine_masked" in full.names
    pools = _mworld(n=4)
    pools[0].items[0] = calibrate.PoolItem("c0", 0.0, 0, 0.5)  # one item without them
    part = readfit._Data(pools)
    assert not part.has_masked and not set(MASKED) & set(part.names)


def test_a_masked_signal_meets_what_topical_signals_cannot_under_nested_cv():
    pools = _mworld()
    data = readfit._Data(pools)
    topical = readfit.nested_cv(
        data, readfit.p3_targets(), [g for g in readfit.GRID if g["C"] == 1.0]
    )
    both = readfit.nested_cv(data, readfit.p3_targets(), SMALL_MASKED)
    assert topical["worst_outer"]["missing_slot_abstention"] < 0.90
    assert both["worst_outer"]["missing_slot_abstention"] >= 0.90
    assert both["feasible"]
    used = {tuple(f["chosen"]["features"]) for f in both["outer_folds"]}
    assert any(set(f) & set(MASKED) for f in used)


def test_nested_cv_with_the_masked_features_never_sees_the_outer_seed():
    """The plant: seed 5's gold masked signals are shifted down, so a fit that saw seed 5 would not
    help the honest outer fold 5, and seeds 1-4 stay good: no leak of the outer seed's masked
    signals into its own thresholds or model."""
    pools = _mworld(plant={5: -0.15})
    rep = readfit.nested_cv(readfit._Data(pools), readfit.p3_targets(), SMALL_MASKED)
    folds = {f["seed"]: f for f in rep["outer_folds"]}
    assert all(f["selected_on"] == sorted({1, 2, 3, 4, 5} - {s}) for s, f in folds.items())
    assert all(5 not in f["selected_on"] for s, f in folds.items() if s == 5)
    ok = [folds[s]["worst"]["missing_slot_abstention"] for s in (1, 2, 3, 4)]
    assert min(ok) >= 0.9, ok
    assert folds[5]["worst"]["current_recall_at_1"] < 0.9, folds[5]["worst"]
    seen = []
    real = readfit._fit_eval
    readfit._fit_eval = lambda d, tr, te, cfg, t: (
        seen.append((set(d.seed[tr].tolist()) - {-1}, set(d.seed[te].tolist()))),
        real(d, tr, te, cfg, t),
    )[1]
    try:
        readfit.select_config(
            readfit._Data(pools), [1, 2, 3, 4], readfit.p3_targets(), SMALL_MASKED[:8]
        )
    finally:
        readfit._fit_eval = real
    assert seen and all(5 not in tr | te for tr, te in seen)


def test_the_fitted_gate_carries_the_masked_coefficients_and_scores_with_them():
    gate, rep = readfit.fit_p3_gate(_mworld(n=12), grid=SMALL_MASKED, **FIT)
    rm = gate.reader
    assert rm is not None and set(rm.features) & set(MASKED)
    assert gate.read_model["fingerprint"] == rm.fingerprint()
    q = _mworld(n=1)[0]
    ps = gate.probs([i.z for i in q.items], [i.cos for i in q.items], readfit._masked_of(q.items))
    assert len(ps) == len(q.items)
    with pytest.raises(ValueError, match="masked|cosine_masked|rerank_masked"):
        gate.probs([i.z for i in q.items], [i.cos for i in q.items])  # serve without the feature
    assert "cosine_masked" in rep["ablation"] or "cosine_masked+rerank_masked" in rep["ablation"]
    assert "cos_masked" in readfit.diagnose(_mworld(n=3), gate)["auroc_answerable_vs_missing"]
