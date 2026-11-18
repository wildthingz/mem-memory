"""The P3 read gate (MEM2-155): the cosine read model, its gate JSON, and the fit that targets the
consuming gates (B-R2 (ii), B-R5, B-R2 (i) FAR on SYN dev) with nested leave-one-seed-out CV."""

import itertools
import json
import math
import sqlite3

import numpy as np
import pytest
from typer.testing import CliRunner

from bench.gates import GATES_YAML, load_gates
from mem import calibrate, readfit, readmodel
from mem.cli import app
from mem.config import default_config
from mem.gate import Gate, gate_json, read_gate, write_gate
from mem.readmodel import ReadModel
from mem.store import Store, init_store

SMALL = [g for g in readfit.GRID if g["C"] == 1.0]  # the tests keep the grid small: speed
FIT = {
    "fit_id": "t",
    "fingerprint_hash": "f" * 64,
    "limit": 8,
    "grid": SMALL,
    "fitted_at": "2026-10-25T00:00:00Z",
}


def _item(key, z, cos, rel):
    return calibrate.PoolItem(key, float(z), rel, float(cos))


def _world(seeds=(1, 2, 3, 4, 5), n=30, gold_z=None, seed=3):
    """SYN-like pools. Answerable: the gold item has a high cosine and a wide-ranging logit; some
    answerables have a stale distractor with a high logit too. Missing-slot: all items have low
    cosine, but a third carry a distractor logit as high as a gold's. z alone cannot separate them
    (so one threshold cannot meet abstention and FAR); the cosine does."""
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
                    items.append(_item(f"c{i}", rng.normal(-3, 1.2), rng.normal(0.35, 0.05), 0))
                if ans:
                    gz = (gold_z or {}).get(sd, rng.normal(1.0, 2.0))
                    items[0] = _item("c0", gz, rng.normal(0.8, 0.03), 2)
                elif q % 3 == 0:
                    items[0] = _item("c0", rng.normal(1.5, 1.0), rng.normal(0.4, 0.04), 0)
                pools.append(calibrate.QueryPool(f"syn{sd}:{cat}{q}", ans, items, cat))
    for k in range(40):  # LME-like pools: training data and the recall guard only
        its = [_item(f"c{i}", rng.normal(-3, 1), rng.normal(0.35, 0.05), 0) for i in range(10)]
        if k % 2 == 0:
            its[1] = _item("c1", rng.normal(1.0, 1.5), rng.normal(0.7, 0.05), 1)
        pools.append(calibrate.QueryPool(f"lme{k}", k % 2 == 0, its, "lme"))
    return pools


# ---- the model and the gate JSON ---------------------------------------------------------------


def _model():
    return ReadModel.build(["rerank", "cosine", "gap"], [0.4, 9.0, 0.05], -8.0)


def test_read_model_is_deterministic_and_round_trips_through_the_gate_json(tmp_path):
    m = _model()
    zs, cs = [2.0, -1.0, 0.5], [0.7, 0.4, 0.5]
    assert m.probs(zs, cs) == m.probs(zs, cs)
    z = np.array([2.0, -1.0, 0.5])
    want = 1 / (1 + np.exp(-(0.4 * z + 9.0 * np.array(cs) + 0.05 * 1.5 - 8.0)))  # gap = 2.0 - 0.5
    assert m.probs(zs, cs) == pytest.approx(want.tolist(), abs=1e-12)
    gate = Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=m.to_dict())
    init_store(tmp_path / "t.db", "personal", "ab" * 32, 8).close()
    conn = sqlite3.connect(tmp_path / "t.db")
    write_gate(conn, gate)
    back = read_gate(conn)
    assert back is not None and back.reader == m
    assert back.probs(zs, cs) == m.probs(zs, cs)
    assert json.loads(gate_json(back))["read_model"]["fingerprint"] == m.fingerprint()


def test_gap_feature_is_best_minus_second_best_logit():
    m = ReadModel.build(["gap"], [1.0], 0.0)
    p = m.probs([3.0, 1.0, -2.0], None)
    assert p == pytest.approx([1 / (1 + math.exp(-2.0))] * 3)
    assert ReadModel.build(["gap"], [1.0], 0.0).probs([5.0], None) == [0.5]  # one candidate: 0


def test_a_gate_without_coefficients_is_refused_no_fallback():
    good = _model().to_dict()
    for broken in (
        {"features": ["rerank", "cosine"]},  # no weights, no bias
        {k: v for k, v in good.items() if k != "weights"},
        {k: v for k, v in good.items() if k != "bias"},
        {**good, "weights": [0.4, 9.0, 0.06]},  # coefficients changed under the fingerprint
        {**good, "fingerprint": None},
    ):
        with pytest.raises(ValueError, match="read_model"):
            Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=broken)


def test_read_gate_refuses_an_incomplete_read_model_in_meta(tmp_path):
    init_store(tmp_path / "t.db", "personal", "ab" * 32, 8).close()
    conn = sqlite3.connect(tmp_path / "t.db")
    bad = json.loads(gate_json(Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=_model().to_dict())))
    del bad["read_model"]["weights"]
    conn.execute("INSERT INTO meta(key, value) VALUES ('gate', ?)", (json.dumps(bad),))
    with pytest.raises(ValueError, match="read_model"):
        read_gate(conn)


def test_a_read_model_gate_does_not_answer_p_of_z_alone():
    g = Gate(1.0, 0.0, 0.3, 0.5, 0.1, read_model=_model().to_dict())
    with pytest.raises(ValueError, match="read_model"):
        g.p(1.0)
    with pytest.raises(ValueError, match="cosine"):
        g.probs([1.0, 0.0], None)  # the cosines are required, not defaulted


# ---- the fit -----------------------------------------------------------------------------------


def test_the_fit_is_reproducible():
    pools = _world(n=10)
    a, ra = readfit.fit_p3_gate(pools, **FIT)
    b, rb = readfit.fit_p3_gate(pools, **FIT)
    assert gate_json(a) == gate_json(b)
    assert json.dumps(ra, sort_keys=True, default=str) == json.dumps(
        rb, sort_keys=True, default=str
    )


@pytest.fixture(scope="module")
def fitted():
    pools = _world()
    return pools, *readfit.fit_p3_gate(pools, **FIT)


def test_cosine_meets_the_targets_that_the_logit_alone_cannot(fitted):
    _, gate, rep = fitted
    ab = rep["ablation"]
    assert ab["rerank"]["feasible"] is False  # one logit threshold: abstention vs FAR
    assert ab["rerank+cosine"]["feasible"] is True, ab["rerank+cosine"]
    assert rep["nested"]["feasible"] is True and gate.tune["feasible"] is True
    assert gate.reader is not None and "cosine" in gate.reader.features
    assert rep["flat_cv_optimistic"]["chosen"]["features"] != ["rerank"]
    text = readfit.format_report(rep)
    assert "nested" in text and "optimistic" in text and "before (P2 floors)" in text


def test_fitted_gate_applies_on_the_pools_it_was_fitted_to(fitted):
    pools, gate, _ = fitted
    miss = [q for q in pools if q.category == "syn:missing_slot"]
    ab = 0
    for q in miss:
        ps = gate.probs([i.z for i in q.items], [i.cos for i in q.items])
        ab += gate.select(ps)[1]
    assert ab / len(miss) >= 0.90


def test_nested_cv_never_sees_the_outer_seed():
    """Seeds 1-4: the gold logits are high, so thresholds fitted on them sit above seed 5's gold.
    Seed 5 (the plant) has gold logits far lower but still well separated from the missing-slot
    queries: a fit that saw seed 5 lowers τ_keep and meets every target, the honest outer fold 5
    cannot (its thresholds come from seeds 1-4) and must report the miss."""
    pools = _world(gold_z={1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: -1.0})
    only_rerank_cos = [g for g in readfit.GRID if g["features"] == ["rerank", "cosine"]]
    rep = readfit.nested_cv(readfit._Data(pools), readfit.p3_targets(), only_rerank_cos)
    folds = {f["seed"]: f for f in rep["outer_folds"]}
    assert all(f["selected_on"] == sorted({1, 2, 3, 4, 5} - {s}) for s, f in folds.items())
    ok = [folds[s]["worst"]["current_recall_at_1"] for s in (1, 2, 3, 4)]
    assert min(ok) >= 0.9, ok
    assert folds[5]["worst"]["current_recall_at_1"] < 0.9, folds[5]["worst"]
    assert rep["feasible"] is False
    # the same seed 5 inside the training data (what a leak would use) is easy to satisfy:
    _, full = readfit.fit_p3_gate(pools, **FIT)
    assert full["final"]["dev_in_sample"]["worst"]["current_recall_at_1"] >= 0.9


def test_selection_inside_a_fold_uses_only_the_seeds_it_is_given():
    data = readfit._Data(_world())
    seen = []
    real = readfit._fit_eval

    def spy(d, train, test, cfg, targets):
        seen.append((set(d.seed[train].tolist()) - {-1}, set(d.seed[test].tolist())))
        return real(d, train, test, cfg, targets)

    readfit._fit_eval, orig = spy, readfit._fit_eval
    try:
        readfit.select_config(data, [1, 2, 3, 4], readfit.p3_targets(), readfit.GRID[:2])
    finally:
        readfit._fit_eval = orig
    assert seen and all(5 not in tr | te for tr, te in seen)  # the held-out seed 5 never appears
    assert all(len(te) == 1 and not (te & tr) for tr, te in seen)


def test_nested_cv_needs_three_seeds():
    with pytest.raises(ValueError, match="three seeds"):
        readfit.nested_cv(readfit._Data(_world(seeds=(1, 2))), readfit.p3_targets())


# ---- the objective: exactly the consuming gates' rows ------------------------------------------


def test_objective_is_the_consuming_gates_rows_read_from_gates_yaml():
    specs = {(s.id, s.metric): s for s in load_gates(GATES_YAML)["P3"]}
    ts = readfit.p3_targets()
    assert [(t.gate_metric) for t in ts] == [
        "missing_slot_abstention", "current_recall_at_1", "as_of_accuracy", "far"]  # fmt: skip
    for t in ts:
        gid = readfit.P3_ROWS[t.gate_metric][0]
        assert (gid, t.gate_metric) in specs
        assert (t.op, t.threshold) == (
            specs[(gid, t.gate_metric)].op,
            specs[(gid, t.gate_metric)].threshold,
        )
    assert {t.threshold for t in ts} == {0.90, 0.95, 0.05}
    # LME precision-side rows are descriptive and are not targets
    assert not {"precision", "ira", "coverage_any", "abstention_recall"} & {t.metric for t in ts}


def test_the_search_sees_syn_queries_only():
    pools = _world()
    data = readfit._Data(pools)
    model = readfit.fit_model(data, np.ones(len(pools), bool), ["rerank", "cosine"], 1.0)
    v = readfit._view(data, readfit._probs(data, model), np.ones(len(pools), bool))
    assert len(v.p1) == sum(readfit.is_syn(q) for q in pools)  # the LME pools are not in the view
    assert v.seeds == [1, 2, 3, 4, 5]


def test_search_is_the_strictest_feasible_setting_by_brute_force():
    pools = [q for q in _world(seeds=(1, 2, 3), n=5) if readfit.is_syn(q)]
    data = readfit._Data(pools)
    full = np.ones(len(pools), bool)
    model = readfit.fit_model(data, full, ["rerank", "cosine"], 1.0)
    v = readfit._view(data, readfit._probs(data, model), full)
    ts = readfit.p3_targets()
    st = readfit.search(v, ts, margin=0.0)
    ev = readfit.evaluate(v, st.tau_keep, st.tau_high, st.gap, ts)
    assert st.feasible and st.rule == "strictest"
    assert all(t.met(ev["worst"][t.metric]) for t in ts)
    assert readfit.search(v, ts).tau_keep <= st.tau_keep  # the margin never makes it stricter
    taus = sorted({0.0, *v.p1.tolist()}, reverse=True)
    cuts = calibrate._between(v.p1.tolist())
    gaps = [0.0, *calibrate._between(v.d.tolist())]
    best = None
    for tau in taus:
        for th, g in itertools.product([tau, *[c for c in cuts if c > tau]], gaps):
            e = readfit.evaluate(v, tau, th, g, ts)["worst"]
            if all(t.met(e[t.metric]) for t in ts):
                key = (tau, th, g)
                best = max(best, key) if best else key
        if best and best[0] == tau:
            break
    assert st.tau_keep == pytest.approx(best[0])  # the strictest tau_keep; the cells at it tie


def test_infeasible_fit_reports_the_miss_and_does_not_relax_a_target():
    rng = np.random.default_rng(0)
    pools = [  # nothing separates answerable from missing: a coin flip per query
        calibrate.QueryPool(f"syn{s}:{cat}{q}", ans, [_item("c0", rng.normal(), rng.normal(), rel),
                                                      _item("c1", rng.normal(), rng.normal(), 0)], cat)
        for s in (1, 2, 3) for q in range(20)
        for cat, ans, rel in (("syn:current", True, 2), ("syn:missing_slot", False, 0))
    ]  # fmt: skip
    gate, rep = readfit.fit_p3_gate(pools, **FIT)
    assert gate.tune["feasible"] is False and gate.tune["rule"] == "fallback_maximin"
    assert rep["nested"]["feasible"] is False
    assert {t["threshold"] for t in gate.tune["targets"].values()} == {0.90, 0.95, 0.05}
    assert any(not t["met"] for t in gate.tune["targets"].values())


def test_lme_recall_guard_rejects_a_model_that_hurts_the_p2_row():
    data = readfit._Data(_world())
    inverted = ReadModel.build(["rerank"], [-1.0], 0.0)  # ranks the best logit last
    ok, rec = readfit._guard_ok(data, inverted)
    assert ok is False and rec["by_model"] < rec["by_z"]
    same_order = ReadModel.build(["rerank"], [1.0], 0.0)  # the logit's own order
    assert readfit._guard_ok(data, same_order)[0] is True


# ---- records and the CLI -----------------------------------------------------------------------


def test_records_carry_the_cosine_and_old_records_still_load(tmp_path):
    pools = [calibrate.QueryPool("q", True, [_item("c1", 1.0, 0.5, 1)], "x")]
    path = tmp_path / "r.jsonl"
    calibrate.write_records(path, pools, fp_all="f")
    assert calibrate.read_records(path)[1][0].items[0].cos == 0.5
    old = tmp_path / "old.jsonl"
    old.write_text('{"fp_all": "f"}\n{"qid": "q", "answerable": true, "items": '
                   '[{"key": "c1", "z": 1.0, "rel": 1}]}\n')  # fmt: skip
    assert calibrate.read_records(old)[1][0].items[0].cos is None


def test_records_without_cosines_are_refused_by_the_p3_fit():
    pools = [calibrate.QueryPool("syn1:x", True, [calibrate.PoolItem("c", 1.0, 1)], "syn:current")]
    with pytest.raises(ValueError, match="cosine"):
        readfit.fit_p3_gate(pools, **FIT)


def test_diagnose_separates_pool_ranking_and_gate_causes():
    def pool(qid, cat, zs, rels, cos=None):
        return calibrate.QueryPool(
            qid,
            True,
            [_item(f"c{i}", z, 0.5, r) for i, (z, r) in enumerate(zip(zs, rels, strict=True))],
            cat,
        )

    pools = [
        pool("syn1:a", "syn:current", [2, 0], [0, 0]),  # gold not in the pool
        pool("syn1:b", "syn:current", [0, 3], [2, 0]),  # ranked second
        pool("syn1:c", "syn:current", [3, 0], [2, 0]),  # first, delivered
        pool("syn1:d", "syn:as_of", [-5, -6], [2, 0]),  # first, the gate drops it
        calibrate.QueryPool("syn1:m", False, [_item("c0", 1.0, 0.4, 0)], "syn:missing_slot"),
    ]
    gate = Gate(1.0, 0.0, 0.5, 0.5, 0.0)
    d = readfit.diagnose(pools, gate)
    cur, asof = d["by_class"]["syn:current"], d["by_class"]["syn:as_of"]
    assert (cur["gold_not_in_pool"], cur["gold_not_first"], cur["delivered_gold"]) == (1, 1, 1)
    assert asof["gate_drops_gold"] == 1
    assert d["by_class"]["syn:current"]["examples"]["gold_not_first"][0]["qid"] == "syn1:b"
    assert "ranked below first" in readfit.format_diagnosis(d)


def test_calibrate_cli_p3_objective_writes_a_gate_with_the_read_model(tmp_path):
    from tests.helpers import STUB

    path = tmp_path / "records.jsonl"
    calibrate.write_records(path, _world(n=8), fp_all=STUB.fp_all())
    out = tmp_path / "gate.json"
    r = CliRunner().invoke(app, ["calibrate", "--records", str(path), "--embedder", "hash",
                                 "--objective", "p3", "--out", str(out),
                                 "--fit-id", "p3-test"])  # fmt: skip
    assert r.exit_code == 0, r.output
    gate = json.loads(out.read_text())
    assert gate["read_model"]["fingerprint"] and gate["fit_id"] == "p3-test"
    assert Gate(**gate).reader is not None
    assert "nested" in r.output
    bad = CliRunner().invoke(app, ["calibrate", "--records", str(path), "--embedder", "hash",
                                   "--objective", "nope"])  # fmt: skip
    assert bad.exit_code != 0


# ---- the read path -----------------------------------------------------------------------------


def _store(tmp_path, gate):
    from mem.testing import HashEmbedder, OverlapReranker

    read_cfg = {**default_config().read, "p3_model": gate.read_model is not None}
    s = Store.open(tmp_path / "p.db", HashEmbedder(), reranker=OverlapReranker(), read=read_cfg)
    s.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    s.remember(text="The cat sleeps.", type="fact", about=["new:thing:Cat"])
    write_gate(s.conn, gate)
    s.gate = gate
    return s


def test_recall_scores_with_the_read_model_logs_the_features_and_embeds_the_query_once(tmp_path):
    from mem import read

    cosine_only = ReadModel.build(
        ["rerank", "cosine"],
        [
            0.0,
            20.0,
        ],
        -4.0,
    )
    gate = Gate(1.0, 0.0, 0.5, 0.5, 0.0, read_model=cosine_only.to_dict())
    s = _store(tmp_path, gate)
    calls = []
    real = s.embedder.embed_query
    s.embedder.embed_query = lambda q: (calls.append(q), real(q))[1]
    out = read.recall(s, query="dog Bosco")
    assert len(calls) == 1  # the cosine reuses the vector candidate_pool embedded: no extra call
    row = json.loads(
        s.conn.execute("SELECT result FROM events WHERE tool = 'recall'").fetchone()[0]
    )
    assert row["feat"] and row["feat"][0]["cos"] is not None and "z_gap" in row
    # p comes from the cosine, not from the rerank logit: the dog memory's p matches the model
    pool, _ = read.candidate_pool(s, "dog Bosco", t=s.now_ts())
    cids = [c for c, _ in pool]
    zs = read.rerank(s, "dog Bosco", cids)
    cos = read.pool_cosines(s, "dog Bosco", cids)
    assert all(-1.0 <= c <= 1.0 for c in cos)
    want = max(cosine_only.probs(zs, cos))
    assert row["feat"][0]["p"] == pytest.approx(want, abs=1e-3)
    assert out.structured["abstained"] is (want < 0.5)
    s.close()


def test_a_platt_gate_reads_exactly_as_before_and_computes_no_cosine(tmp_path, monkeypatch):
    from mem import read

    s = _store(tmp_path, Gate(1.0, 0.0, 0.5, 0.5, 0.0))
    monkeypatch.setattr(read, "pool_cosines", lambda *a, **k: pytest.fail("no cosine for Platt"))
    out = read.recall(s, query="dog Bosco")
    assert [r["id"] for r in out.structured["rows"]] == ["m1"]
    s.close()


def test_bench_read_gate_diagnose_reads_records_and_a_gate(tmp_path):
    from bench.cli import app as bench_app

    path = tmp_path / "r.jsonl"
    calibrate.write_records(path, _world(n=4), fp_all="f")
    gate = tmp_path / "g.json"
    gate.write_text(gate_json(Gate(1.0, 0.0, 0.5, 0.5, 0.0)))
    out = tmp_path / "d.json"
    r = CliRunner().invoke(bench_app, ["read-gate-diagnose", str(path), "--gate", str(gate),
                                       "--out", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    d = json.loads(out.read_text())
    assert (
        set(d["by_class"]) == {"syn:current", "syn:as_of"}
        and d["by_class"]["syn:current"]["n"] == 20
    )
    assert {"z", "cos"} <= set(d["auroc_answerable_vs_missing"])
    assert "AUROC" in r.output


def test_bench_gate_p3_refuses_a_gate_file_without_a_read_model_when_p3_model_is_on(tmp_path):
    from dataclasses import asdict

    from bench.adapters.lme_ours import load_config
    from bench.cli import app as bench_app
    from tests.bench_t0.test_p3_real import stub_gate

    gate = asdict(stub_gate(load_config(None, "hash")))
    gate["read_model"] = None  # the P2-style gate: Platt only
    g = tmp_path / "gate.json"
    cfg = tmp_path / "config.yaml"
    cfg.write_text("read: {p3_model: true}\n", encoding="utf-8")
    gate["fingerprint_hash"] = load_config(cfg, "hash").fp_all()  # the gate belongs to that config
    g.write_text(json.dumps(gate), encoding="utf-8")
    r = CliRunner().invoke(bench_app, ["gate", "P3", "--embedder", "hash", "--gate", str(g),
                                       "--config", str(cfg), "--syn-size", "S",
                                       "--syn-seeds", "1"])  # fmt: skip
    assert r.exit_code == 2 and "read_model" in r.output and "no fallback" in r.output


# ---- MEM2-167: the masked signals through the CLI -----------------------------------------------


def _masked_world(n=8, drop_masked_on=None):
    """_world with an uninformative cosine and the informative signal moved into mcos / mz, so a
    masked feature set must win. LME pools carry the masked columns too (they take the
    descriptive-rows path that dropped them). `drop_masked_on`: qids to strip mcos/mz from."""
    out = []
    rng = np.random.default_rng(7)
    for q in _world(n=n):
        items = []
        for i in q.items:
            mcos = i.cos
            items.append(calibrate.PoolItem(i.key, i.z, i.rel, float(rng.normal(0.5, 0.2)),
                                            mcos, i.z))  # fmt: skip
        if drop_masked_on and q.qid in drop_masked_on:
            items = [calibrate.PoolItem(i.key, i.z, i.rel, i.cos) for i in items]
        out.append(calibrate.QueryPool(q.qid, q.answerable, items, q.category))
    return out


def _calibrate_p3(tmp_path, pools, *extra):
    from tests.helpers import STUB

    path = tmp_path / "records.jsonl"
    calibrate.write_records(path, pools, fp_all=STUB.fp_all())
    out, rep = tmp_path / "gate.json", tmp_path / "report.json"
    r = CliRunner().invoke(app, ["calibrate", "--records", str(path), "--embedder", "hash",
                                 "--objective", "p3", "--out", str(out), "--report", str(rep),
                                 "--fit-id", "m", *extra])  # fmt: skip
    return r, out, rep


def test_calibrate_cli_p3_with_a_masked_winner_on_mixed_lme_and_syn_pools(tmp_path):
    r, out, _ = _calibrate_p3(tmp_path, _masked_world())
    assert r.exit_code == 0, r.output
    gate = json.loads(out.read_text())
    # a subject-masked signal, or (MEM2-171) its margin, which the full grid now also offers
    derived = set(readmodel.MASKED) | set(readmodel.MARGINS)
    assert set(gate["read_model"]["features"]) & derived


def test_p3_fit_reports_and_refuses_records_without_the_masked_signals(tmp_path):
    lacking = {q.qid for q in _world(n=8)[:5]}
    r, _, rep = _calibrate_p3(tmp_path, _masked_world(drop_masked_on=lacking))
    assert r.exit_code == 0, r.output  # the base grid still fits, but loudly
    assert "lack mcos/mz" in r.output and "NOT in the grid" in r.output
    want = sum(len(q.items) for q in _world(n=8) if q.qid in lacking)
    assert json.loads(rep.read_text())["masked_missing_items"] == want
    (tmp_path / "strict").mkdir()
    r2, out2, _ = _calibrate_p3(tmp_path / "strict", _masked_world(drop_masked_on=lacking),
                                "--require-masked")  # fmt: skip
    assert r2.exit_code != 0 and "lack mcos/mz" in r2.output and not out2.exists()


# ---- MEM2-176: the LME no-regression guard ------------------------------------------------------

RC = [g for g in readfit.GRID if g["features"] == ["rerank", "cosine"] and g["C"] == 1.0]


def _lme_world(gold, topical=None, n=10, seed=11):
    """_world's SYN pools with new LME pools: `gold(rng)` gives an answerable pool's rel-1 item
    (z, cos); `topical(rng)`, if given, a rel-0 item every LME pool carries."""
    rng = np.random.default_rng(seed)
    out = [q for q in _world(n=n) if readfit.is_syn(q)]
    for k in range(40):
        its = [_item(f"c{i}", rng.normal(-3, 1), rng.normal(0.35, 0.05), 0) for i in range(10)]
        if topical:
            its[2] = _item("c2", *topical(rng), 0)
        if k % 2 == 0:
            its[1] = _item("c1", *gold(rng), 1)
        out.append(calibrate.QueryPool(f"lme{k}", k % 2 == 0, its, "lme"))
    return out


def _blocking_world():
    """The answerable LME golds look like SYN missing-slot distractors (high logit, low cosine):
    the abstention the SYN target asks for is bought exactly by dropping them."""
    return _lme_world(lambda r: (r.normal(1.5, 1.0), r.normal(0.4, 0.04)))


def test_guard_search_is_the_strictest_setting_meeting_both_by_brute_force():
    pools = _world(seeds=(1, 2, 3), n=5)
    data = readfit._Data(pools, 8)
    full = np.ones(len(pools), bool)
    model = readfit.fit_model(data, full, ["rerank", "cosine"], 1.0)
    v = readfit._view(data, readfit._probs(data, model), full)
    ts = readfit.p3_targets()
    free = readfit.search(v, ts, margin=0.0)
    lme = data.pools_of(data.seed < 0)
    table = calibrate._Table(lme, 0.0, 0.0, 8, model)
    # a reference the unguarded strictest setting misses: the LME numbers at a lower τ_keep
    m = table.metrics(free.tau_keep / 2, 0.0, 0.0)
    guard = readfit._LmeGuard(table, float(m["coverage_any"]), float(m["far"]))
    assert not guard.at(free.tau_keep, free.tau_high, free.gap)["ok"]
    st = readfit.search(v, ts, margin=0.0, guard=guard)
    assert st.feasible and st.guard_ok and st.tau_keep < free.tau_keep
    taus = sorted({0.0, *v.p1.tolist()}, reverse=True)
    cuts = calibrate._between(v.p1.tolist())
    gaps = [0.0, *calibrate._between(v.d.tolist())]
    best = None
    for tau in taus:
        for th, g in itertools.product([tau, *[c for c in cuts if c > tau]], gaps):
            e = readfit.evaluate(v, tau, th, g, ts)["worst"]
            if all(t.met(e[t.metric]) for t in ts) and guard.at(tau, th, g)["ok"]:
                best = max(best, (tau, th, g)) if best else (tau, th, g)
        if best and best[0] == tau:
            break
    assert st.tau_keep == pytest.approx(best[0])


def test_guard_blocks_the_best_abstaining_setting_and_the_fit_says_infeasible():
    pools = _blocking_world()
    gate, rep = readfit.fit_p3_gate(pools, **{**FIT, "grid": RC})
    # without the guard the SYN targets are met: the guard is what blocks them
    data = readfit._Data(pools)
    full = np.ones(len(pools), bool)
    model = gate.reader
    assert readfit.search(readfit._view(data, readfit._probs(data, model), full),
                          readfit.p3_targets()).feasible  # fmt: skip
    fin, g = rep["final"], rep["guard_lme_no_regression"]
    assert fin["feasible"] is False and fin["blocked_by_guard"] is True
    assert fin["rule"] == "fallback_maximin" and gate.tune["feasible"] is False
    # the fallback keeps the guard; the reference is the P2 floors gate on the same records
    assert g["ok"] and g["coverage_any"] >= g["p2_coverage_any"]
    assert g["empty_answerable"] <= g["p2_empty_answerable"]
    before = rep["lme_descriptive"]["before_p2_floors"]
    assert (g["p2_coverage_any"], g["p2_empty_answerable"]) == (
        before["coverage_any"],
        before["far"],
    )
    assert rep["lme_descriptive"]["after_p3"]["coverage_any"] == g["coverage_any"]
    assert gate.tune["guard_lme_no_regression"] == g and gate.tune["blocked_by_guard"] is True
    assert fin["dev_in_sample"]["worst"]["missing_slot_abstention"] < 0.90  # no target relaxed
    assert rep["nested"]["feasible"] is False
    assert all(f["blocked_by_guard"] for f in rep["nested"]["outer_folds"])
    text = readfit.format_report(rep)
    assert "lme_no_regression guard blocks every setting" in text
    assert f"coverage_any P2 {g['p2_coverage_any']} -> chosen {g['coverage_any']}" in text
    assert f"P2 {g['p2_empty_answerable']} -> chosen {g['empty_answerable']}" in text


def test_guard_met_with_the_targets_picks_the_setting_and_reports_both_values(fitted):
    _, gate, rep = fitted
    g = rep["guard_lme_no_regression"]
    assert rep["nested"]["met"]["lme_no_regression"] is True and rep["nested"]["feasible"]
    assert gate.tune["feasible"] is True and rep["final"]["blocked_by_guard"] is False
    assert g["ok"] and g["coverage_any"] >= g["p2_coverage_any"]
    assert g["empty_answerable"] <= g["p2_empty_answerable"]
    assert json.loads(gate_json(gate))["tune"]["guard_lme_no_regression"] == g
    assert all(f["guard_lme_no_regression"]["ok"] for f in rep["nested"]["outer_folds"])
    text = readfit.format_report(rep)
    assert "guard lme_no_regression (ok)" in text and str(g["p2_coverage_any"]) in text


def test_guard_reference_uses_the_training_folds_only():
    """Every P2 reference is fitted on exactly the training pools of the fold it guards: never the
    held-out seed (outer or inner)."""
    data = readfit._Data(_world(n=6), 8)
    refs, fits = [], []
    real_ref, real_fit = readfit.p2_reference, calibrate.fit_gate

    def spy_ref(d, train):
        refs.append(frozenset(d.seed[train].tolist()) - {-1})
        return real_ref(d, train)

    def spy_fit(pools, **kw):
        fits.append(frozenset(readfit.seed_of(q.qid) for q in pools) - {None})
        return real_fit(pools, **kw)

    real_eval = readfit._fit_eval
    evals = []

    def spy_eval(d, train, test, cfg, targets):
        evals.append((frozenset(d.seed[train].tolist()) - {-1}, set(d.seed[test].tolist())))
        return real_eval(d, train, test, cfg, targets)

    readfit.p2_reference, calibrate.fit_gate, readfit._fit_eval = spy_ref, spy_fit, spy_eval
    try:
        rep = readfit.nested_cv(data, readfit.p3_targets(), RC)
    finally:
        readfit.p2_reference, calibrate.fit_gate, readfit._fit_eval = real_ref, real_fit, real_eval
    assert evals and {tr for tr, _ in evals} == set(refs)  # one reference per training set
    assert all(not (te & tr) for tr, te in evals)
    assert set(fits) <= set(refs) and all(len(s) <= 4 for s in fits)  # never all five seeds
    assert {len(s) for s in fits} == {3, 4}  # inner (two held out) and outer (one) folds
    for f in rep["outer_folds"]:
        assert f["guard_lme_no_regression"] is not None


def _unmet_world():
    """limit 1: the LME golds have the top logit (the P2 gate delivers them first) but a topical
    rel-0 chunk outranks them under the cosine model, so no setting keeps the coverage."""
    return _lme_world(lambda r: (r.normal(2.5, 0.2), r.normal(0.35, 0.02)),
                      topical=lambda r: (r.normal(-0.5, 0.2), r.normal(0.8, 0.02)))  # fmt: skip


@pytest.fixture(scope="module")
def unmet():
    with pytest.raises(readfit.GuardUnmet) as err:
        readfit.fit_p3_gate(_unmet_world(), **{**FIT, "grid": RC, "limit": 1})
    return err.value


def test_no_gate_when_no_setting_keeps_the_guard(unmet):
    g = unmet.report["guard_lme_no_regression"]
    assert g["ok"] is False and g["coverage_any"] < g["p2_coverage_any"]
    assert unmet.report["final"]["rule"] == "fallback_maximin_guard_unmet"
    assert "lme_no_regression guard unmet" in str(unmet) and str(g["p2_coverage_any"]) in str(unmet)
    assert "UNMET" in readfit.format_report(unmet.report)


def test_calibrate_cli_writes_no_gate_but_the_report_when_the_guard_is_unmet(
    tmp_path, monkeypatch, unmet
):
    def raise_unmet(*a, **k):
        raise unmet

    monkeypatch.setattr(readfit, "fit_p3_gate", raise_unmet)
    r, out, rep = _calibrate_p3(tmp_path, _world(n=2))
    assert r.exit_code == 1 and "No gate was written" in r.output
    assert "lme_no_regression guard unmet" in r.output and "UNMET" in r.output
    assert not out.exists()
    assert json.loads(rep.read_text())["guard_lme_no_regression"]["ok"] is False
