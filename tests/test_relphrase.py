"""MEM2-171 (B-R2 (ii), one bounded pass): the relation-phrase match (`phrase_cosine`,
`phrase_overlap`, mem.relphrase) and the masked margins (`cosine_masked_margin`,
`rerank_masked_margin`) as read-model features."""

import hashlib
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mem import calibrate, read, readfit, readmodel, relphrase, subjectmask
from mem.cli import app
from mem.config import default_config
from mem.gate import Gate
from mem.readmodel import MARGINS, MASKED, PHRASE, SIGNALS, ReadModel, feature_matrix
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker

# ---- the phrase: generic text processing, deterministic -----------------------------------------


def test_the_phrase_drops_the_subject_question_words_determiners_and_stop_words():
    q = "What is the favourite food of Ana Lopez?"
    assert relphrase.content_words(q, ["Ana Lopez", "Ana", "Lopez"]) == ["favourite", "food"]
    assert relphrase.phrase(q, ["Ana Lopez"]) == "favourite food"
    # the subject several times, possessive, mixed case; and not at all
    t = "ANA LOPEZ said Ana Lopez's favourite food is soup; ana agrees."
    assert relphrase.content_words(t, ["Ana Lopez", "Ana"]) == [
        "said", "favourite", "food", "soup", "agrees"]  # fmt: skip
    assert relphrase.content_words("Where does she live now?", ["Bosco"]) == ["live", "now"]
    assert relphrase.content_words("", ["Ana"]) == []
    assert relphrase.content_words("Who is Ana?", ["Ana"]) == []  # nothing but function words


def test_the_phrase_is_deterministic_and_independent_of_the_order_of_the_names():
    t = "Ana Lopez works at Acme with Lopez; Ana commutes by bike."
    names = ["Ana", "Lopez", "Ana Lopez"]
    want = relphrase.content_words(t, names)
    assert want == ["works", "acme", "commutes", "bike"]
    assert relphrase.content_words(t, names[::-1]) == want == relphrase.content_words(t, set(names))
    assert relphrase.content_words(t, []) == ["ana", "lopez", "works", "acme", "lopez", "ana",
                                              "commutes", "bike"]  # fmt: skip
    assert relphrase.content_words("born in 1984, lives in Oslo", []) == ["born", "lives", "oslo"]


def test_the_light_stemmer_joins_the_inflections_of_a_predicate():
    same = [("lives", "lived", "living", "live"), ("hobby", "hobbies"), ("reading", "reads", "read"),
            ("launches", "launch"), ("speaks", "speak"), ("loves", "loved", "love"),
            ("drives", "drive", "driving"), ("names", "named", "name"), ("cities", "city"),
            ("running", "runs", "run"), ("commutes", "commute")]  # fmt: skip
    for group in same:
        stems = {relphrase.stem(w) for w in group}
        assert len(stems) == 1, (group, stems)
    assert relphrase.stem("bus") == "bus" and relphrase.stem("famous") == "famous"
    assert relphrase.stem("food") != relphrase.stem("work")
    assert relphrase.stem("Lives") == relphrase.stem("lives")  # casefolded


def test_overlap_is_the_share_of_the_question_stems_found_in_the_memory():
    q = ["favourite", "food"]
    assert relphrase.overlap(q, ["favourite", "food", "soup"]) == 1.0
    assert relphrase.overlap(q, ["favorite", "foods"]) == 0.5  # "foods" ~ "food"; "favorite" no
    assert relphrase.overlap(q, ["works", "dentist"]) == 0.0
    assert relphrase.overlap([], ["anything"]) == 0.0  # no question phrase: no evidence
    assert relphrase.overlap(["live"], []) == 0.0


# ---- the store: fit time == serve time -------------------------------------------------------


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


def _text(s, cid):
    return s.conn.execute("SELECT text FROM chunks WHERE id = ?", (cid,)).fetchone()[0]


def test_the_phrase_signals_prefer_the_memory_that_answers_the_question_not_the_subject(tmp_path):
    s = _store(tmp_path)
    cids = _cids(s)
    q = "What is Ana Lopez's favourite food?"
    assert relphrase.query_phrase(s, q) == "favourite food"
    sig = subjectmask.pool_masked(s, q, cids, PHRASE)
    assert set(sig) == set(PHRASE) and all(len(v) == len(cids) for v in sig.values())
    assert sig == subjectmask.pool_masked(s, q, cids, PHRASE)  # deterministic
    by = {_text(s, c): (pc, po) for c, pc, po in
          zip(cids, sig["phrase_cosine"], sig["phrase_overlap"], strict=True)}  # fmt: skip
    food, job = by["Ana's favourite food is soup."], by["Ana Lopez works as a dentist."]
    assert food[1] == 1.0 and job[1] == 0.0  # both are about Ana; only one answers the question
    assert food[0] > job[0]
    assert subjectmask.pool_masked(s, q, [], PHRASE) == {"phrase_cosine": [], "phrase_overlap": []}
    # the masked signals and the phrase ones in one call, each only when asked for
    both = subjectmask.pool_masked(s, q, cids, SIGNALS)
    assert set(both) == set(SIGNALS)
    assert subjectmask.pool_masked(s, q, cids, ["phrase_overlap"]).keys() == {"phrase_overlap"}
    s.close()


def test_a_question_with_no_content_words_falls_back_to_the_masked_text(tmp_path):
    s = _store(tmp_path)
    cids = _cids(s)
    q = "Who is Ana Lopez?"
    assert relphrase.query_phrase(s, q) == ""
    sig = subjectmask.pool_masked(s, q, cids, PHRASE)
    assert sig["phrase_overlap"] == [0.0] * len(cids)
    masked = subjectmask.masked_cosines(s, q, cids)
    assert all(np.isfinite(sig["phrase_cosine"]))
    assert len(sig["phrase_cosine"]) == len(masked)
    s.close()


def test_the_features_are_computed_the_same_at_fit_time_and_serve_time(tmp_path):
    s = _store(tmp_path)
    q = "Ana Lopez favourite food"
    items = calibrate.score_pool(s, q)  # fit time: what the records hold
    assert all(i.pcos is not None and i.pov is not None for i in items)
    m = ReadModel.build(
        ["rerank", "cosine_masked", "cosine_masked_margin", "phrase_cosine", "phrase_overlap"],
        [0.3, 4.0, 2.0, 3.0, 1.0],
        -1.0,
    )
    assert m.masked_features == ("cosine_masked", "phrase_cosine", "phrase_overlap")
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0, read_model=m.to_dict())
    want = m.probs([i.z for i in items], [i.cos for i in items], calibrate._masked_columns(items))
    cids = [int(i.key[1:]) for i in items]
    zs = read.rerank(s, q, cids)
    got = read.pool_probs(s, q, cids, zs)  # serve time: computed from the store
    assert got == pytest.approx(want, abs=1e-12)
    out = read.recall(s, query=q)
    assert out.structured is not None
    s.close()


# ---- the margins --------------------------------------------------------------------------------


def test_a_masked_margin_is_best_minus_second_best_of_the_masked_signal_in_the_pool():
    masked = {"cosine_masked": [0.2, 0.7, 0.4], "rerank_masked": [-3.0, 2.0, 1.5]}
    x = feature_matrix([0.0, 1.0, 2.0], None, list(MARGINS), masked)
    assert x.shape == (3, 2)
    assert x[:, 0].tolist() == pytest.approx([0.3, 0.3, 0.3])
    assert x[:, 1].tolist() == pytest.approx([0.5, 0.5, 0.5])
    one = feature_matrix([0.0], None, ["cosine_masked_margin"], {"cosine_masked": [0.9]})
    assert one.tolist() == [[0.0]]  # one candidate: no margin
    assert feature_matrix([], None, ["rerank_masked_margin"], {"rerank_masked": []}).shape == (0, 1)


def test_a_margin_model_needs_the_masked_signal_it_is_derived_from():
    m = ReadModel.build(["rerank", "rerank_masked_margin"], [1.0, 1.0], 0.0)
    assert m.masked_features == ("rerank_masked",)  # what the store must compute for it
    with pytest.raises(ValueError, match="rerank_masked"):
        m.probs([1.0, 0.0], None)
    with pytest.raises(ValueError, match="rerank_masked"):
        m.probs([1.0, 0.0], None, {"cosine_masked": [0.1, 0.2]})
    with pytest.raises(ValueError, match="rerank_masked"):
        m.probs([1.0, 0.0], None, {"rerank_masked": [0.1, None]})
    assert m.probs([1.0, 0.0], None, {"rerank_masked": [0.1, 0.2]})
    p = ReadModel.build(["phrase_cosine"], [1.0], 0.0)
    with pytest.raises(ValueError, match="phrase_cosine"):
        p.probs([1.0, 0.0], None, {"phrase_overlap": [0.1, 0.2]})
    with pytest.raises(ValueError, match="unknown"):
        ReadModel.build(["rerank", "phrase_margin"], [1.0, 1.0], 0.0)


# ---- the fingerprint ----------------------------------------------------------------------------


def test_the_phrase_rule_is_part_of_the_fingerprint_of_a_phrase_model_only(monkeypatch):
    plain = ReadModel.build(["rerank", "cosine", "gap"], [0.4, 9.0, 0.05], -8.0)
    masked = ReadModel.build(["rerank", "cosine_masked"], [1.0, 2.0], 0.0)
    phrase = ReadModel.build(["rerank", "phrase_overlap"], [1.0, 2.0], 0.0)
    fp = {m: m.fingerprint() for m in (plain, masked, phrase)}
    d = phrase.to_dict()
    body = json.dumps([1, ["rerank", "cosine", "gap"], [0.4, 9.0, 0.05], -8.0],
                      separators=(",", ":"))  # fmt: skip
    assert fp[plain] == hashlib.sha256(body.encode()).hexdigest()  # pre-MEM2-165 gates load
    monkeypatch.setattr(relphrase, "STOP", frozenset(relphrase.STOP - {"the"}))
    assert plain.fingerprint() == fp[plain] and masked.fingerprint() == fp[masked]
    assert phrase.fingerprint() != fp[phrase]
    with pytest.raises(ValueError, match="fingerprint"):
        ReadModel.from_dict(d)


def test_the_masking_rule_is_part_of_the_fingerprint_of_a_margin_or_phrase_model(monkeypatch):
    margin = ReadModel.build(["rerank", "cosine_masked_margin"], [1.0, 2.0], 0.0)
    phrase = ReadModel.build(["phrase_cosine"], [1.0], 0.0)
    fm, fph = margin.fingerprint(), phrase.fingerprint()
    monkeypatch.setattr(subjectmask, "MAX_NAME_WORDS", 2)  # the subject rule changed since the fit
    assert margin.fingerprint() != fm and phrase.fingerprint() != fph
    assert ReadModel.build(["rerank", "gap"], [1.0, 2.0], 0.0).fingerprint() == (
        ReadModel.build(["rerank", "gap"], [1.0, 2.0], 0.0).fingerprint()
    )


def test_a_gate_whose_phrase_or_margin_coefficients_are_tampered_is_refused():
    d = ReadModel.build(["rerank", "phrase_cosine", "rerank_masked_margin"], [0.4, 0.6, 0.2],
                        -1.0).to_dict()  # fmt: skip
    Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=d)
    for broken in (
        {**d, "weights": [0.4, 0.6, 0.3]},
        {**d, "features": ["rerank", "phrase_overlap", "rerank_masked_margin"]},
        {k: v for k, v in d.items() if k != "fingerprint"},
    ):
        with pytest.raises(ValueError, match="read_model"):
            Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=broken)


# ---- no SYN vocabulary --------------------------------------------------------------------------

SYN_WORDS = ("slot", "relation", "rel_", "subtype", "cue", "template", "value_returns", "hobby",
             "role", "food", "late", "future", "paraphrase", "counterpart", "gold", "syn")  # fmt: skip
# function words a SYN cue may share with any English stop list ("lives in", "works at", ...)
FUNCTION_WORDS = {"a", "the", "in", "to", "is", "at", "as", "by", "on", "with", "for", "has", "of",
                  "will"}  # fmt: skip


def test_no_feature_and_no_phrase_code_encodes_syn_vocabulary():
    import inspect

    from bench.synth import data as syn

    for f, desc in readmodel.FEATURES.items():
        assert not any(w in f for w in SYN_WORDS), f
        assert not any(w in desc.lower() for w in SYN_WORDS), f
    code = inspect.getsource(relphrase).replace(relphrase.__doc__ or "", "")
    assert not any(w in code.lower() for w in ("hobby", "slot", "syn", "favourite", "works at"))
    # the stop list holds function words only: every SYN cue word in it is a function word, and no
    # relation id, cue phrase or value is in it
    cue_words = {w for cs in syn.all_cues().values() for c in cs for w in c.split()}
    assert relphrase.STOP & cue_words <= FUNCTION_WORDS, relphrase.STOP & cue_words
    assert not relphrase.STOP & set(syn.RELATIONS)
    assert not relphrase.STOP & {" ".join(cs) for cs in syn.all_cues().values()}
    values = {w.casefold() for vs in syn.VALUES.values() for v in vs for w in v.split()}
    assert not relphrase.STOP & values
    assert all(w.isalpha() and w == w.casefold() for w in relphrase.STOP)


# ---- records ------------------------------------------------------------------------------------


def test_records_carry_the_phrase_signals_and_old_records_still_load(tmp_path):
    p = calibrate.QueryPool("syn1:q", True,
                            [calibrate.PoolItem("c1", 1.0, 2, 0.5, 0.3, -0.5, 0.6, 1.0)])  # fmt: skip
    masked_only = calibrate.QueryPool("lme", True,
                                      [calibrate.PoolItem("c1", 1.0, 1, 0.5, 0.3, -0.5)])  # fmt: skip
    calibrate.write_records(tmp_path / "r.jsonl", [p, masked_only], fp_all="f")
    _, back = calibrate.read_records(tmp_path / "r.jsonl")
    assert back[0].items[0].pcos == 0.6 and back[0].items[0].pov == 1.0
    assert back[1].items[0].pcos is None and back[1].items[0].pov is None
    assert back[1].items[0].mcos == 0.3
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    legacy = json.loads(lines[1])
    for k in ("pcos", "pov", "mcos", "mz"):
        del legacy["items"][0][k]
    (tmp_path / "old.jsonl").write_text("\n".join([lines[0], json.dumps(legacy)]))
    old = calibrate.read_records(tmp_path / "old.jsonl")[1][0].items[0]
    assert old.pcos is None and old.mcos is None and old.cos == 0.5
    assert set(calibrate._masked_columns([p.items[0]])) == set(SIGNALS)


# ---- synthetic worlds ---------------------------------------------------------------------------


def _pool(rng, items, qid, ans, cat):
    """A pool with its items in a random order (so a tie by index cannot rank the gold first)."""
    order = rng.permutation(len(items))
    return calibrate.QueryPool(qid, ans, [items[i] for i in order], cat)


def _pworld(seeds=(1, 2, 3, 4, 5), n=30, seed=11, plant=None):
    """SYN-like pools where only the phrase signals separate answerable from missing-slot: every
    missing-slot pool has a same-subject distractor as high as a gold on logit, cosine AND the
    masked signals (it is about the subject, with another predicate); the phrase cosine / overlap
    of a gold is high and the distractor's low. `plant` = {seed: shift} lowers a seed's gold
    phrase signals."""
    rng = np.random.default_rng(seed)
    pools = []
    for sd in seeds:
        for q in range(n):
            for cat, ans in (("syn:current", True), ("syn:as_of", True),
                             ("syn:missing_slot", False)):  # fmt: skip
                items = [calibrate.PoolItem(f"c{i}", rng.normal(-3, 1.2), 0,
                                            rng.normal(0.35, 0.05), rng.normal(0.2, 0.05),
                                            rng.normal(-4, 1.0), rng.normal(0.15, 0.05), 0.0)
                         for i in range(7)]  # fmt: skip
                shift = (plant or {}).get(sd, 0.0)
                top = (rng.normal(1.0, 1.0), rng.normal(0.7, 0.04), rng.normal(0.7, 0.03),
                       rng.normal(2.0, 0.7))  # fmt: skip
                if ans:
                    items[0] = calibrate.PoolItem("c0", *top[:1], 2, *top[1:],
                                                  rng.normal(0.7, 0.04) + shift,
                                                  min(1.0, max(0.0, rng.normal(0.9, 0.1) + shift)))  # fmt: skip
                else:
                    items[0] = calibrate.PoolItem("c0", *top[:1], 0, *top[1:],
                                                  rng.normal(0.2, 0.04), 0.0)  # fmt: skip
                pools.append(_pool(rng, items, f"syn{sd}:{cat}{q}", ans, cat))
    return pools


def _gworld(seeds=(1, 2, 3, 4, 5), n=30, seed=13):
    """Pools where the masked signals cannot do it alone: a same-subject distractor with a low
    logit has a HIGHER masked cosine / logit than the gold (so a masked signal ranks the gold
    second), and the best masked value is as high in a missing-slot pool as in an answerable one
    (so no threshold on it separates). What differs: an answerable pool has one clear masked best,
    a missing-slot pool has two. Beside the logit that ranks, the masked margin separates; the
    gate's own p-gap rule cannot, since the logit dominates p1 - p2 in both classes."""
    rng = np.random.default_rng(seed)
    pools = []
    for sd in seeds:
        for q in range(n):
            for cat, ans in (("syn:current", True), ("syn:as_of", True),
                             ("syn:missing_slot", False)):  # fmt: skip
                items = [calibrate.PoolItem(f"c{i}", rng.normal(-3, 1.2), 0,
                                            rng.normal(0.35, 0.05), rng.normal(0.2, 0.05),
                                            rng.normal(-4, 1.0))
                         for i in range(7)]  # fmt: skip
                top_z, top_cos = rng.normal(1.0, 1.0), rng.normal(0.7, 0.04)
                if ans:
                    items[0] = calibrate.PoolItem("c0", top_z, 2, top_cos, rng.normal(0.5, 0.03),
                                                  rng.normal(0.0, 0.5))  # fmt: skip
                else:
                    items[0] = calibrate.PoolItem("c0", top_z, 0, top_cos, rng.normal(0.7, 0.03),
                                                  rng.normal(2.0, 0.5))  # fmt: skip
                items[1] = calibrate.PoolItem("c1", rng.normal(-3, 1.2), 0, rng.normal(0.35, 0.05),
                                              rng.normal(0.7, 0.03), rng.normal(2.0, 0.5))  # fmt: skip
                pools.append(_pool(rng, items, f"syn{sd}:{cat}{q}", ans, cat))
    return pools


FIT = {"fit_id": "t", "fingerprint_hash": "f" * 64, "limit": 8, "fitted_at": "2026-10-25T00:00:00Z"}
C1 = lambda grid: [g for g in grid if g["C"] == 1.0]  # the tests keep the grid small


def test_the_new_sets_join_the_grid_only_when_every_record_carries_their_signals():
    full = readfit._Data(_pworld(n=3))
    assert full.has_masked and full.has_phrase
    assert set(PHRASE) | set(MARGINS) <= set(full.names)
    assert len(readfit.grid_for(full)) == len(readfit.GRID) + len(readfit.MASKED_GRID) + len(
        readfit.MARGIN_GRID) + len(readfit.PHRASE_GRID) + len(readfit.BOTH_GRID)  # fmt: skip
    masked_only = readfit._Data(_gworld(n=3))
    assert masked_only.has_masked and not masked_only.has_phrase
    assert set(MARGINS) <= set(masked_only.names) and not set(PHRASE) & set(masked_only.names)
    assert len(readfit.grid_for(masked_only)) == len(readfit.GRID) + len(
        readfit.MASKED_GRID) + len(readfit.MARGIN_GRID)  # fmt: skip
    pools = _pworld(n=3)
    pools[0].items[0] = calibrate.PoolItem("c0", 0.0, 0, 0.5, 0.1, 0.1)  # one item without them
    part = readfit._Data(pools)
    assert part.has_masked and not part.has_phrase
    for s in (*readfit.PHRASE_SETS, *readfit.BOTH_SETS):
        assert set(s) & set(PHRASE)
    for s in readfit.MARGIN_SETS:
        assert set(s) & set(MARGINS)


def test_the_phrase_match_meets_what_topical_and_masked_signals_cannot_under_nested_cv():
    data = readfit._Data(_pworld())
    t = readfit.p3_targets()
    without = readfit.nested_cv(data, t, C1(readfit.GRID + readfit.MASKED_GRID))
    phrase = readfit.nested_cv(data, t, C1(readfit.PHRASE_GRID))
    assert without["worst_outer"]["missing_slot_abstention"] < 0.90, without["worst_outer"]
    assert phrase["worst_outer"]["missing_slot_abstention"] >= 0.90, phrase["worst_outer"]
    assert phrase["feasible"]
    assert all(set(f["chosen"]["features"]) & set(PHRASE) for f in phrase["outer_folds"])


def test_the_masked_margin_meets_what_the_masked_signals_alone_cannot_under_nested_cv():
    data = readfit._Data(_gworld())
    t = readfit.p3_targets()
    masked = readfit.nested_cv(data, t, C1(readfit.GRID + readfit.MASKED_GRID))
    margin = readfit.nested_cv(data, t, C1(readfit.MARGIN_GRID))
    assert masked["worst_outer"]["missing_slot_abstention"] < 0.90, masked["worst_outer"]
    assert margin["worst_outer"]["missing_slot_abstention"] >= 0.90, margin["worst_outer"]
    assert margin["feasible"]
    assert all(set(f["chosen"]["features"]) & set(MARGINS) for f in margin["outer_folds"])


def test_nested_cv_with_the_phrase_features_never_sees_the_outer_seed():
    """The plant: seed 5's gold phrase signals are shifted down, so its honest outer fold (model
    and thresholds from seeds 1-4) misses while seeds 1-4 stay good; the inner selection of fold 5
    never touches seed 5."""
    pools = _pworld(plant={5: -0.3})
    grid = C1(readfit.PHRASE_GRID + readfit.BOTH_GRID)
    rep = readfit.nested_cv(readfit._Data(pools), readfit.p3_targets(), grid)
    folds = {f["seed"]: f for f in rep["outer_folds"]}
    assert all(f["selected_on"] == sorted({1, 2, 3, 4, 5} - {s}) for s, f in folds.items())
    ok = [folds[s]["worst"]["missing_slot_abstention"] for s in (1, 2, 3, 4)]
    assert min(ok) >= 0.9, ok
    assert folds[5]["worst"]["current_recall_at_1"] < 0.9, folds[5]["worst"]
    assert rep["feasible"] is False
    seen = []
    real = readfit._fit_eval
    readfit._fit_eval = lambda d, tr, te, cfg, t: (
        seen.append((set(d.seed[tr].tolist()) - {-1}, set(d.seed[te].tolist()))),
        real(d, tr, te, cfg, t),
    )[1]
    try:
        readfit.select_config(readfit._Data(pools), [1, 2, 3, 4], readfit.p3_targets(), grid[:6])
    finally:
        readfit._fit_eval = real
    assert seen and all(5 not in tr | te for tr, te in seen)
    assert all(len(te) == 1 and not (te & tr) for tr, te in seen)


def test_the_fitted_gate_carries_the_new_coefficients_and_the_report_the_new_rows():
    pools = _pworld(n=10)
    gate, rep = readfit.fit_p3_gate(pools, grid=C1(readfit.GRID + readfit.PHRASE_GRID), **FIT)
    rm = gate.reader
    assert rm is not None and set(rm.features) & set(PHRASE)
    assert gate.read_model["fingerprint"] == rm.fingerprint()
    assert rep["phrase_grid"] is True and rep["phrase_missing_items"] == 0
    assert any(set(k.split("+")) & set(PHRASE) for k in rep["ablation"])
    q = pools[0]
    ps = gate.probs([i.z for i in q.items], [i.cos for i in q.items], readfit._masked_of(q.items))
    assert len(ps) == len(q.items)
    with pytest.raises(ValueError, match="phrase"):
        gate.probs([i.z for i in q.items], [i.cos for i in q.items])  # serve without the feature
    d = readfit.diagnose(pools[:9], gate)["auroc_answerable_vs_missing"]
    assert {"phrase_cosine", "phrase_overlap", "cos_masked_margin", "rerank_masked_margin"} <= set(
        d
    )
    assert d["phrase_cosine"] > 0.9 and d["phrase_cosine"] > d["cos_masked"]  # what separates


def test_the_fit_says_when_the_phrase_signals_are_missing_and_can_refuse():
    pools = _pworld(n=4)
    q = pools[2]
    pools[2] = calibrate.QueryPool(q.qid, q.answerable, [calibrate.PoolItem(
        i.key, i.z, i.rel, i.cos, i.mcos, i.mz) for i in q.items], q.category)  # fmt: skip
    _, rep = readfit.fit_p3_gate(pools, grid=C1(readfit.GRID), **FIT)
    assert rep["phrase_missing_items"] == 7 and rep["phrase_grid"] is False
    assert "lack pcos/pov" in readfit.format_report(rep)
    with pytest.raises(ValueError, match="lack pcos/pov"):
        readfit.fit_p3_gate(pools, grid=C1(readfit.GRID), require_masked=True, **FIT)


def test_calibrate_cli_p3_with_a_phrase_winner(tmp_path):
    from tests.helpers import STUB

    path = tmp_path / "records.jsonl"
    calibrate.write_records(path, _pworld(n=6), fp_all=STUB.fp_all())
    out, rep = tmp_path / "gate.json", tmp_path / "report.json"
    r = CliRunner().invoke(app, ["calibrate", "--records", str(path), "--embedder", "hash",
                                 "--objective", "p3", "--out", str(out), "--report", str(rep),
                                 "--fit-id", "m", "--require-masked"])  # fmt: skip
    assert r.exit_code == 0, r.output
    gate = json.loads(out.read_text())
    assert set(gate["read_model"]["features"]) & set(PHRASE)
    assert "phrase_cosine" in r.output and "margin" in r.output  # the ablation rows
    with pytest.raises(ValueError, match="phrase"):
        Gate(**gate).probs([1.0, 0.0], [0.5, 0.5], {k: [0.1, 0.2] for k in MASKED})
