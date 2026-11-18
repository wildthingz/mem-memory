"""The gate (architecture-v2 §3.2 steps 5-6) and `mem calibrate`'s fitting (§3.3 steps 3-5)."""

import math
import sqlite3

import numpy as np
import pytest

from mem import calibrate, schema
from mem.gate import Gate, read_gate, write_gate


def _gate(**kw):
    base = {"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5, "gap": 0.0}
    return Gate(**{**base, **kw})


def test_platt_maps_logit_through_sigmoid():
    g = _gate(platt_a=2.0, platt_b=-1.0)
    assert g.p(0.5) == pytest.approx(0.5)
    assert g.p(1.0) == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_select_keeps_p_at_or_above_tau_in_p_order():
    kept, abstained = _gate(tau_keep=0.5).select([0.4, 0.9, 0.5, 0.2])
    assert (kept, abstained) == ([1, 2], False)


@pytest.mark.parametrize(
    ("ps", "abstain"),
    [
        ([0.62, 0.55], True),  # p1 < tau_high and gap .07 < g .2
        ([0.62, 0.30], False),  # gap .32 >= g
        ([0.85, 0.80], False),  # p1 >= tau_high: the gap rule does not apply
        ([0.25, 0.10], True),  # p1 < tau_keep
        ([0.62], False),  # one candidate: p2 = 0, gap .62
        ([], True),  # nothing to rank
    ],
)
def test_select_abstention_rule(ps, abstain):
    kept, abstained = _gate(tau_keep=0.3, tau_high=0.8, gap=0.2).select(ps)
    assert abstained is abstain
    assert kept == ([] if abstain else [i for i, p in sorted(enumerate(ps), key=lambda x: -x[1])
                                        if p >= 0.3])  # fmt: skip


def test_gate_round_trips_through_meta(tmp_path):
    conn = schema.connect(tmp_path / "s.db")
    assert read_gate(conn) is None
    g = _gate(fit_id="lme-tune-v1", fitted_at="2026-10-23T10:00:00Z", fingerprint_hash="ab" * 32,
              tune={"f05": 0.71})  # fmt: skip
    write_gate(conn, g)
    assert read_gate(conn) == g
    conn.close()


def test_read_gate_rejects_an_incomplete_gate(tmp_path):
    conn = schema.connect(tmp_path / "s.db")
    conn.execute("INSERT INTO meta(key, value) VALUES ('gate', '{\"platt_a\": 1.0}')")
    with pytest.raises(ValueError, match="gate"):
        read_gate(conn)
    conn.close()


# ---- calibrate: Platt -----------------------------------------------------------------------


def test_platt_fit_recovers_synthetic_threshold():
    # Labels drawn from p = sigmoid(1.5 z - 3): the p = 0.5 point is z = 2.0 by construction.
    rng = np.random.default_rng(7)
    z = rng.uniform(-6, 6, 20_000)
    y = rng.random(z.size) < 1 / (1 + np.exp(-(1.5 * z - 3.0)))
    a, b = calibrate.fit_platt(z, y)
    assert -b / a == pytest.approx(2.0, abs=0.1)
    assert a == pytest.approx(1.5, rel=0.1)


def test_platt_fit_needs_both_classes():
    with pytest.raises(ValueError, match="both"):
        calibrate.fit_platt([1.0, 2.0], [True, True])


# ---- calibrate: thresholds (architecture-v2 §3.3 steps 4-5, the "floors" rule of 2026-10-24) ----


def _pool(answerable, items, category=""):
    """items: (key, p, rel). The fitting below works on p, so a=1, b=0 and z = logit(p)."""
    return calibrate.QueryPool(
        qid=f"q{id(items)}", answerable=answerable,
        items=[calibrate.PoolItem(k, math.log(p / (1 - p)), rel) for k, p, rel in items],
        category=category,
    )  # fmt: skip


def _deliver(pools, a, b, tau_keep, tau_high, gap, limit):
    """The five P2 tune numbers of benchmarks-v2 §2.0 at one gate setting, computed with
    `Gate.select` on every pool (independent of `calibrate._Table`): S_q = the kept rows, top
    `limit`."""
    g = Gate(a, b, tau_keep, tau_high, gap)
    rows = hits = covered = nonempty = empty_ans = empty_neg = 0
    n_ans = sum(q.answerable for q in pools)
    n_neg = len(pools) - n_ans
    for q in pools:
        kept, abstained = g.select([g.p(i.z) for i in q.items])
        kept = [] if abstained else kept[:limit]
        h = sum(q.items[i].rel >= 1 for i in kept)
        rows, hits, nonempty = rows + len(kept), hits + h, nonempty + bool(kept)
        covered += q.answerable and h > 0
        empty_ans += q.answerable and not kept
        empty_neg += (not q.answerable) and not kept
    return {"precision": hits / rows if rows else 0.0, "ira": (rows - hits) / nonempty if nonempty
            else 0.0, "coverage_any": covered / n_ans if n_ans else 0.0,
            "abstention_recall": empty_neg / n_neg if n_neg else math.nan,
            "far": empty_ans / n_ans if n_ans else 0.0}  # fmt: skip


def _meets(m, targets):
    return all(t.met(m[t.metric]) for t in targets)


def _grid(pools, a, b):
    """Every setting the search considers (calibrate.choose_thresholds docstring): tau_keep in
    {0} + the observed p; tau_high in {tau_keep} + the cuts between p1 values above it; g in {0}
    + the cuts between the p1 - p2 gaps."""
    ps = [[1 / (1 + math.exp(-(a * i.z + b))) for i in q.items] for q in pools]
    p1 = [max(x) for x in ps if x]
    d = [max(x) - (sorted(x, reverse=True)[1] if len(x) > 1 else 0.0) for x in ps if x]
    gaps = [0.0, *calibrate._between(d)]
    for tau_keep in sorted({0.0, *(p for x in ps for p in x)}, reverse=True):
        highs = [tau_keep, *(c for c in calibrate._between(p1) if c > tau_keep)]
        for tau_high in sorted(highs, reverse=True):
            for gap in sorted(gaps, reverse=True):
                yield tau_keep, tau_high, gap


def _strictest_feasible(pools, a, b, limit, targets):
    """Brute force: the lexicographically largest (tau_keep, tau_high, gap) meeting all targets."""
    for tk, th, gp in _grid(pools, a, b):  # the grid is yielded strictest first
        if (th == tk) != (gp == 0.0):  # the gap rule is off either way: one canonical form
            continue
        if _meets(_deliver(pools, a, b, tk, th, gp, limit), targets):
            return tk, th, gp
    return None


def test_p2_targets_are_read_from_the_gates_file():
    """The five targets of steps 4-5 are the P2 B-R1 / B-R2 rows of bench/configs/gates.yaml
    (frozen or not), not a copy of their numbers."""
    from bench.gates import load_gates

    rows = {(s.id, s.metric): s for s in load_gates()["P2"]}
    targets = {t.gate_metric: t for t in calibrate.p2_targets()}
    assert set(targets) == {
        "delivered_precision",
        "ira",
        "coverage_any",
        "abstention_recall",
        "far",
    }
    for name, t in targets.items():
        spec = rows[("B-R2" if name == "abstention_recall" else "B-R1", name)]
        assert (t.op, t.threshold) == (spec.op, spec.threshold)
    assert targets["delivered_precision"].metric == "precision"
    assert rows[("B-R1", "far")].threshold == rows[("B-R2", "far")].threshold


def test_strictest_feasible_setting_is_chosen_when_a_feasible_region_exists():
    """(a) The chosen (tau_keep, tau_high, g) meets all five targets and is the strictest such
    setting: highest tau_keep, ties by highest tau_high, then highest g (brute force over the
    whole candidate grid with an independent oracle)."""
    pools = [_pool(True, [("a", 0.9, 1), ("b", 0.4, 0), ("c", 0.1, 0)]) for _ in range(4)]
    pools += [_pool(True, [("a", 0.8, 1), ("b", 0.3, 0)]) for _ in range(2)]
    pools += [_pool(True, [("a", 0.7, 1), ("b", 0.3, 0), ("c", 0.1, 0)]) for _ in range(2)]
    pools += [_pool(True, [("w", 0.6, 1), ("b", 0.2, 0)]) for _ in range(2)]  # weak golds
    pools += [_pool(False, [("x", 0.3, 0), ("y", 0.28, 0)]) for _ in range(7)]
    pools += [_pool(False, [("x", 0.65, 0), ("y", 0.63, 0)]) for _ in range(3)]  # close pairs
    targets = calibrate.p2_targets()
    c = calibrate.choose_thresholds(pools, a=1.0, b=0.0, limit=8, targets=targets)
    assert c.feasible is True and c.rule == "strictest"
    assert _meets(_deliver(pools, 1.0, 0.0, c.tau_keep, c.tau_high, c.gap, 8), targets)
    assert (c.tau_keep, c.tau_high, c.gap) == _strictest_feasible(pools, 1.0, 0.0, 8, targets)
    # what that means here: tau_keep stops at the weak golds (losing them costs coverage 0.80 <
    # 0.85 and FAR 0.20); the three close-pair negatives must abstain by the gap rule (tau_high
    # above their p1 0.65, g above their gap 0.02) without catching an answerable (every
    # answerable gap is >= 0.4, so g stays below it); the widest such band is the strictest.
    assert c.tau_keep == pytest.approx(0.6)
    assert c.tau_high > 0.65 and 0.02 < c.gap < 0.4
    assert c.tau_high == pytest.approx(max(th for _, th, _ in _grid(pools, 1.0, 0.0)))
    for t in targets:
        assert c.targets[t.gate_metric]["met"] is True
    assert c.front == []


def test_strictest_means_highest_tau_keep_then_tau_high_then_gap():
    """With limit 1 every tau_keep up to the gold's p delivers the same row, so many settings
    tie on all five numbers: the strictest wins, in the documented order."""
    pools = [_pool(True, [("a", 0.9, 1), ("b", 0.5, 0), ("c", 0.2, 0)]) for _ in range(10)]
    pools += [_pool(False, [("x", 0.05, 0)])]
    targets = calibrate.p2_targets()
    c = calibrate.choose_thresholds(pools, a=1.0, b=0.0, limit=1, targets=targets)
    assert c.feasible and c.tau_keep == pytest.approx(0.9)
    assert (c.tau_keep, c.tau_high, c.gap) == _strictest_feasible(pools, 1.0, 0.0, 1, targets)
    assert c.tau_high > 0.9 and 0.05 < c.gap < 0.4  # the gap rule on, as wide as FAR allows


def test_host_shaped_tune_set_no_longer_pins_tau_keep_to_the_third_weakest_gold():
    """(b) The MEM2-83 host-shaped set with 3 all-low pools (gold tops a pool of z = -10 noise at
    z = -7.5). The old step-4 floor (lose <= 0.02 coverage_any = 2 of 115 answerables) put tau_keep
    at the 3rd-weakest gold, z = -7.6, and the gate stayed permissive. Now tau_keep is the highest
    candidate at which some (tau_high, g) meets all five targets: the three weak pools are given
    up (coverage_any 112/115 = 0.974 >= 0.85) and the cut sits in the reranker's useful range."""
    pools = _bge_like_pools(seed=1, weak=3, gold=(3.0, 1.5))
    g = calibrate.fit_gate(pools, fit_id="t", fingerprint_hash="f" * 64, limit=8,
                           fitted_at="2026-10-23T10:00:00Z")  # fmt: skip
    targets = calibrate.p2_targets()
    assert g.tune["feasible"] is True and g.tune["rule"] == "strictest"
    m = _deliver(pools, g.platt_a, g.platt_b, g.tau_keep, g.tau_high, g.gap, 8)
    assert _meets(m, targets)
    for t in targets:
        assert g.tune[t.metric] == pytest.approx(m[t.metric], abs=1e-6)
    z_at_tau = (math.log(g.tau_keep / (1 - g.tau_keep)) - g.platt_b) / g.platt_a
    assert z_at_tau > -5  # not the 3rd-weakest gold at z = -7.6
    # strictness certificate: at every higher candidate tau_keep, coverage_any (largest with the
    # gap rule off) is already below 0.85 or FAR (smallest with it off) above 0.05, so no
    # (tau_high, g) can rescue it; at the chosen one the gap rule's setting is the strictest
    # feasible by brute force.
    cands = _candidates(pools, g.platt_a, g.platt_b)
    for tau in (t for t in cands if t > g.tau_keep):
        base = _deliver(pools, g.platt_a, g.platt_b, tau, tau, 0.0, 8)
        assert base["coverage_any"] < 0.85 - 1e-12 or base["far"] > 0.05 + 1e-12
    report = calibrate.tune_report(pools, g, limit=8)
    assert report["choice"]["feasible"] is True
    assert "INFEASIBLE" not in calibrate.format_report(report)


def _dominates(x, y):
    """x dominates y on the five numbers (better or equal on all, better on one)."""
    better = {"precision": 1, "coverage_any": 1, "abstention_recall": 1, "ira": -1, "far": -1}
    ge = [s * x[k] >= s * y[k] - 1e-12 for k, s in better.items()]
    gt = [s * x[k] > s * y[k] + 1e-12 for k, s in better.items()]
    return all(ge) and any(gt)


def test_infeasible_tune_set_reports_the_pareto_front_and_does_not_relax_a_target():
    """(c) 8 of 46 answerables have no gold in the pool, so coverage_any <= 38/46 = 0.826 < 0.85
    at every setting. calibrate says `feasible: false`, lists the Pareto front on tune (the
    non-dominated settings, each with its five numbers and the targets it misses) and the gate
    it writes is the documented fallback, flagged infeasible: no target is relaxed to make a
    setting look feasible."""
    pools = _bge_like_pools(seed=3, n=46, weak=2, gold=(3.0, 1.5), missing=8)
    g = calibrate.fit_gate(pools, fit_id="t", fingerprint_hash="f" * 64, limit=8,
                           fitted_at="2026-10-23T10:00:00Z")  # fmt: skip
    targets = calibrate.p2_targets()
    tune = g.tune
    assert tune["feasible"] is False
    assert tune["targets"]["coverage_any"]["met"] is False
    assert tune["targets"]["coverage_any"]["best"] == pytest.approx(38 / 46, abs=1e-6)
    front = tune["front"]
    assert len(front) >= 2
    for pt in front:
        assert set(pt) >= {"tau_keep", "tau_high", "gap", "precision", "ira", "coverage_any",
                           "abstention_recall", "far", "unmet"}  # fmt: skip
        m = _deliver(pools, g.platt_a, g.platt_b, pt["tau_keep"], pt["tau_high"], pt["gap"], 8)
        for k in ("precision", "ira", "coverage_any", "abstention_recall", "far"):
            assert pt[k] == pytest.approx(m[k], abs=1e-6)
        assert not _meets(m, targets)
        assert pt["unmet"] == [t.gate_metric for t in targets if not t.met(m[t.metric])]
        assert "coverage_any" in pt["unmet"]
    assert not any(_dominates(x, y) for x in front for y in front if x is not y)
    # the written gate: no setting meets coverage_any, so the fallback keeps the best reachable
    # coverage_any (the pool-recall ceiling) with FAR <= 0.05, strictest, and says so
    assert tune["rule"] == "fallback_coverage_ceiling"
    m = _deliver(pools, g.platt_a, g.platt_b, g.tau_keep, g.tau_high, g.gap, 8)
    assert m["coverage_any"] == pytest.approx(38 / 46, abs=1e-6) and m["far"] <= 0.05
    assert not _meets(m, targets)
    report = calibrate.tune_report(pools, g, limit=8)
    assert report["choice"]["feasible"] is False and report["choice"]["front"] == front
    text = calibrate.format_report(report).splitlines()
    infeasible = [ln for ln in text if "INFEASIBLE" in ln]
    assert len(infeasible) == 1 and text[-1] == infeasible[0]  # the line just before the gate
    assert "coverage_any >= 0.85" in infeasible[0] and "best 0.826" in infeasible[0]
    assert "flagged infeasible" in infeasible[0]
    assert any(ln.startswith("Pareto front") for ln in text)


def test_infeasible_fallback_prefers_a_setting_that_meets_coverage_and_far():
    """When coverage_any and FAR are reachable but precision is not (every pool carries an
    irrelevant chunk scored above its gold), the written gate is the strictest setting meeting
    those two, flagged infeasible; nothing is written as if it were feasible."""
    pools = [_pool(True, [("b", 0.95, 0), ("a", 0.9, 1), ("c", 0.8, 0)]) for _ in range(10)]
    pools += [_pool(False, [("x", 0.1, 0)]) for _ in range(10)]
    c = calibrate.choose_thresholds(pools, a=1.0, b=0.0, limit=8, targets=calibrate.p2_targets())
    assert c.feasible is False and c.rule == "fallback_coverage_far"
    assert c.targets["delivered_precision"]["met"] is False
    assert c.targets["delivered_precision"]["best"] == pytest.approx(0.5)
    assert c.targets["coverage_any"]["met"] and c.targets["far"]["met"]
    assert c.tau_keep == pytest.approx(0.9)  # strictest cut that still covers every answerable
    assert c.tau_high == c.tau_keep and c.gap == 0.0  # any g would abstain the answerables
    m = _deliver(pools, 1.0, 0.0, c.tau_keep, c.tau_high, c.gap, 8)
    assert m["coverage_any"] == 1.0 and m["far"] == 0.0 and m["precision"] == 0.5
    assert c.front and all("delivered_precision" in pt["unmet"] for pt in c.front)
    assert any(pt["tau_keep"] == pytest.approx(0.9) for pt in c.front)


def test_grid_agrees_with_the_per_setting_metrics():
    """`_grid` (all (tau_high, g) at one tau_keep by cumulative sums) equals `_Table.metrics`
    at every cell, on random pools with ties and empty pools."""
    rng = np.random.default_rng(5)
    pools = []
    for q in range(40):
        n = int(rng.integers(0, 6))
        items = [calibrate.PoolItem(f"c{i}", float(rng.choice([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])),
                                    int(rng.random() < 0.3)) for i in range(n)]  # fmt: skip
        pools.append(calibrate.QueryPool(f"q{q}", bool(q % 3), items))
    table = calibrate._Table(pools, 1.0, 0.0, 3)
    gaps = np.array([0.0, *calibrate._between(table.d[table.has].tolist())])
    for tau in calibrate._tau_keep_candidates(table)[::4]:
        highs, m = calibrate._grid(table, tau, gaps)
        for h, th in enumerate(highs.tolist()):
            want = table.metrics(tau, th, gaps)
            for k in calibrate.FRONT_KEYS:
                assert np.allclose(np.nan_to_num(m[k][h]), np.nan_to_num(want[k]), atol=1e-12), (
                    tau,
                    th,
                    k,
                )


def test_fit_gate_writes_fingerprint_and_tune_metrics():
    pools = [
        _pool(True, [("a", 0.9, 1), ("b", 0.2, 0)]),
        _pool(True, [("c", 0.8, 1), ("d", 0.3, 0)]),
        _pool(False, [("x", 0.25, 0), ("y", 0.1, 0)]),
        _pool(False, [("u", 0.15, 0), ("v", 0.05, 0)]),
    ]
    g = calibrate.fit_gate(pools, fit_id="t", fingerprint_hash="f" * 64, limit=8,
                           fitted_at="2026-10-23T10:00:00Z")  # fmt: skip
    assert g.fingerprint_hash == "f" * 64 and g.fit_id == "t"
    assert g.tau_keep <= g.tau_high
    assert g.tune["coverage_any"] == 1.0
    assert g.tune["abstention_recall"] == 1.0
    assert g.tune["precision"] == 1.0
    assert g.tune["feasible"] is True and g.tune["rule"] == "strictest"
    assert set(g.tune) >= {"f05", "precision", "coverage_any", "ira", "far", "abstention_recall",
                           "queries", "feasible", "rule", "targets", "strictest"}  # fmt: skip
    assert "front" not in g.tune  # only an infeasible fit carries the front


def test_records_round_trip(tmp_path):
    pools = [_pool(True, [("a", 0.9, 1)]), _pool(False, [("b", 0.2, 0)])]
    path = tmp_path / "r.jsonl"
    calibrate.write_records(path, pools, fp_all="e" * 64)
    fp, back = calibrate.read_records(path)
    assert fp == "e" * 64
    assert back == pools


def test_write_gate_keeps_other_meta(tmp_path):
    conn = schema.connect(tmp_path / "s.db")
    before = dict(conn.execute("SELECT key, value FROM meta"))
    write_gate(conn, _gate())
    after = dict(conn.execute("SELECT key, value FROM meta"))
    assert {k: v for k, v in after.items() if k != "gate"} == before
    conn.close()
    assert sqlite3.connect(tmp_path / "s.db").execute(
        "SELECT count(*) FROM meta WHERE key = 'gate'").fetchone() == (1,)  # fmt: skip


# ---- MEM2-83 (F-P2b): inputs, records and the tune report -----------------------------------


def test_records_carry_the_query_category_and_old_records_still_load(tmp_path):
    pools = [calibrate.QueryPool("q1:answerable", True, [calibrate.PoolItem("a", 2.0, 1)],
                                 category="multi-session")]  # fmt: skip
    path = tmp_path / "r.jsonl"
    calibrate.write_records(path, pools, fp_all="e" * 64)
    assert calibrate.read_records(path)[1] == pools
    path.write_text('{"fp_all": "x"}\n{"qid": "q", "answerable": true, "items": '
                    '[{"key": "a", "z": 1.0, "rel": 1}]}\n')  # fmt: skip
    assert calibrate.read_records(path)[1][0].category == ""


def test_platt_fit_on_reranker_scale_logits_recovers_sign_and_scale():
    # bge-reranker-v2-m3-scale raw logits (about -11 .. +9); labels drawn from p = σ(1.0 z + 2).
    rng = np.random.default_rng(11)
    z = rng.uniform(-11, 9, 20_000)
    y = rng.random(z.size) < 1 / (1 + np.exp(-(1.0 * z + 2.0)))
    a, b = calibrate.fit_platt(z, y)
    assert a == pytest.approx(1.0, rel=0.1)
    assert b == pytest.approx(2.0, abs=0.3)
    # The two inputs the spec rules out look different: sigmoid outputs (z in [0, 1]) give a slope
    # far above 1, negated logits a slope below 0. `tune_report` flags both.
    a_sig, _ = calibrate.fit_platt(1 / (1 + np.exp(-z)), y)
    assert a_sig > 5
    a_neg, _ = calibrate.fit_platt(-z, y)
    assert a_neg < 0


def test_fit_gate_reports_tune_metrics_per_category():
    pools = [
        _pool(True, [("a", 0.9, 1), ("b", 0.2, 0)], "multi-session"),
        _pool(False, [("x", 0.25, 0), ("y", 0.1, 0)], "multi-session"),
        _pool(True, [("c", 0.8, 1), ("d", 0.3, 0)], "temporal-reasoning"),
        _pool(True, [("e", 0.3, 0), ("f", 0.2, 1)], "temporal-reasoning"),  # gold below the top
        _pool(False, [("u", 0.15, 0), ("v", 0.05, 0)], "temporal-reasoning"),
    ]
    # limit 1: S_q is the top row, so the e/f query is never covered: coverage_any <= 2/3 at
    # every setting, the fit is infeasible and the fallback keeps that ceiling with FAR 0, at the
    # strictest cut that does (tau_keep = e's 0.3, the gap rule off: any g would abstain e/f).
    g = calibrate.fit_gate(pools, fit_id="t", fingerprint_hash="f" * 64, limit=1,
                           fitted_at="2026-10-23T10:00:00Z")  # fmt: skip
    assert g.tune["feasible"] is False and g.tune["rule"] == "fallback_coverage_ceiling"
    assert g.tau_keep == pytest.approx(g.p(math.log(0.3 / 0.7)))  # e's p under the fitted Platt
    assert g.tau_high == g.tau_keep and g.gap == 0.0
    pc = g.tune["per_category"]
    assert set(pc) == {"multi-session", "temporal-reasoning"}
    ms, tr = pc["multi-session"], pc["temporal-reasoning"]
    assert (ms["n_answerable"], ms["n_negative"]) == (1, 1)
    assert (tr["n_answerable"], tr["n_negative"]) == (2, 1)
    assert ms["pool_recall"] == tr["pool_recall"] == 1.0  # every gold item is in its pool
    assert ms["coverage_any"] == 1.0 and ms["precision"] == 1.0 and ms["ira"] == 0.0
    assert ms["abstention_recall"] == 1.0
    assert tr["coverage_any"] == 0.5  # the second query delivers e, not its gold f
    assert tr["far"] == 0.0 and tr["abstention_recall"] == 1.0 and tr["ira"] == 0.5
    for row in (ms, tr):
        assert set(row) >= {"n_answerable", "n_negative", "pool_recall", "coverage_any",
                            "precision", "ira", "far", "abstention_recall", "f05",
                            "rows_per_answer"}  # fmt: skip


def _bge_like_pools(seed=2, n=115, weak=3, pool=30, gold=(0.5, 3.0), missing=0):
    """Pools shaped like the P2 host run: 115 answerables + 115 evidence-removed negatives, raw
    logits on the reranker's scale, gold drawn from N(*gold) against irrelevant N(-8, 3). `weak`
    answerables have an all-low pool whose gold still tops it (z ≈ -7.5): covered at tau 0, lost
    at any useful tau. The last `missing` answerables have no gold in the pool (a pool-recall
    miss): never covered, so coverage_any ≤ (n - missing) / n at every tau."""
    rng = np.random.default_rng(seed)
    pools = []
    for q in range(n):
        ng = 1 + q % 2
        if q >= n - missing:
            items = [(float(rng.normal(-8, 3)), 0) for _ in range(pool)]
        elif q < weak:
            items = [(-7.5 - 0.1 * g, 1) for g in range(ng)]
            items += [(float(rng.normal(-10, 1)), 0) for _ in range(pool - ng)]
        else:
            items = [(float(rng.normal(*gold)), 1) for _ in range(ng)]
            items += [(float(rng.normal(-8, 3)), 0) for _ in range(pool - ng)]
        pools.append(calibrate.QueryPool(f"a{q}", True, [calibrate.PoolItem(f"c{i}", z, r)
                                                          for i, (z, r) in enumerate(items)],
                                         category="multi-session" if q % 2 else "knowledge-update"))  # fmt: skip
    for q in range(n):
        items = [calibrate.PoolItem(f"c{i}", float(rng.normal(-7.5, 3)), 0) for i in range(pool)]
        pools.append(calibrate.QueryPool(f"n{q}", False, items, category="multi-session"))
    return pools


def _candidates(pools, a, b):
    return sorted({0.0, *(1 / (1 + math.exp(-(a * i.z + b))) for q in pools for i in q.items)})


def test_tune_report_frontier_inputs_and_warnings():
    pools = [
        _pool(True, [("a", 0.9, 1), ("b", 0.6, 0), ("c", 0.3, 0)], "multi-session"),
        _pool(True, [("d", 0.7, 1), ("e", 0.4, 0)], "multi-session"),
        _pool(True, [("f", 0.35, 1), ("g", 0.2, 0)], "temporal-reasoning"),
        _pool(False, [("h", 0.5, 0), ("i", 0.1, 0)], "temporal-reasoning"),
    ]
    g = calibrate.fit_gate(pools, fit_id="t", fingerprint_hash="f" * 64, limit=8,
                           fitted_at="2026-10-23T10:00:00Z")  # fmt: skip
    r = calibrate.tune_report(pools, g, limit=8)
    assert r["queries"] == 4 and r["coverage_at_zero"] == 1.0
    taus = [row["tau_keep"] for row in r["frontier"]]
    assert taus[0] == 0.0 and taus == sorted(taus) and g.tau_keep in taus
    row0 = r["frontier"][0]
    assert row0["coverage_any"] == 1.0 and row0["precision"] == pytest.approx(3 / 9)
    assert set(row0) >= {"tau_keep", "z", "precision", "coverage_any", "ira", "far", "f05",
                         "abstention_recall", "rows_per_answer", "feasible"}  # fmt: skip
    # precision 3/6 at the best coverage-keeping cut and the one negative (p1 .5, gap .4) cannot
    # abstain without an answerable abstaining too: infeasible; the fallback keeps coverage and FAR
    assert r["choice"]["feasible"] is False and r["choice"]["rule"] == "fallback_coverage_far"
    assert g.tau_keep == pytest.approx(g.p(math.log(0.35 / 0.65)))  # f's p under the fitted Platt
    assert g.tau_high == g.tau_keep and g.gap == 0.0
    assert not any(row["feasible"] for row in r["frontier"])
    assert r["choice"]["front"] == g.tune["front"] and r["choice"]["targets"] == g.tune["targets"]
    inp = r["inputs"]
    assert inp["n_items"] == 9 and inp["n_rel"] == 3
    assert inp["z_min"] < 0 < inp["z_max"] and inp["auroc_z"] > 0.5
    assert inp["z_rel_median"] > inp["z_irrel_median"]
    assert r["warnings"] == []
    assert r["per_category"] == g.tune["per_category"]
    # a sigmoid-fed or sign-flipped fit is named
    flipped = calibrate.QueryPool
    bad = [flipped(p.qid, p.answerable, [calibrate.PoolItem(i.key, -i.z, i.rel) for i in p.items],
                   p.category) for p in pools]  # fmt: skip
    gb = calibrate.fit_gate(bad, fit_id="t", fingerprint_hash="f" * 64, limit=8, fitted_at="x")
    assert any("sign" in w for w in calibrate.tune_report(bad, gb, limit=8)["warnings"])
    unit = [flipped(p.qid, p.answerable, [calibrate.PoolItem(i.key, g.p(i.z), i.rel)
                                          for i in p.items], p.category) for p in pools]  # fmt: skip
    gu = calibrate.fit_gate(unit, fit_id="t", fingerprint_hash="f" * 64, limit=8, fitted_at="x")
    assert any("[0, 1]" in w for w in calibrate.tune_report(unit, gu, limit=8)["warnings"])


def test_b_r3_targets_are_the_pg_rows_not_the_lme_ones():
    """A fit on PG-v1 tune applies the floors rule with B-R3's thresholds (FAR 0.10, abstention
    0.70), read from the P4 rows (MEM2-128)."""
    t = {x.gate_metric: x for x in calibrate.b_r3_targets()}
    assert set(t) == {"delivered_precision", "ira", "coverage_any", "abstention_recall", "far"}
    assert (t["far"].op, t["far"].threshold) == ("<=", 0.10)
    assert (t["abstention_recall"].op, t["abstention_recall"].threshold) == (">=", 0.70)
    assert t["delivered_precision"].threshold == 0.70 and t["coverage_any"].threshold == 0.85
    p2 = {x.gate_metric: x for x in calibrate.p2_targets()}
    assert p2["far"].threshold == 0.05 and p2["abstention_recall"].threshold == 0.85


def test_fit_gate_uses_the_targets_it_is_given():
    """The recorded targets are the ones passed in (B-R3's FAR 0.10 vs P2's 0.05)."""
    pools = [_pool(True, [("a", 0.9, 1), ("b", 0.2, 0)]) for _ in range(8)]
    pools += [_pool(False, [("x", 0.6 - 0.01 * i, 0)]) for i in range(10)]
    g3 = calibrate.fit_gate(pools, fit_id="pg", fingerprint_hash="f", limit=8, fitted_at="t",
                            targets=calibrate.b_r3_targets())  # fmt: skip
    g2 = calibrate.fit_gate(pools, fit_id="pg", fingerprint_hash="f", limit=8, fitted_at="t")
    assert g3.tune["targets"]["far"]["threshold"] == 0.10
    assert g2.tune["targets"]["far"]["threshold"] == 0.05


def test_pool_recall_counts_answerables_with_a_gold_chunk_in_the_first_size_items():
    """MEM2-128: pool recall = share of answerables whose pool holds a rel >= 1 chunk; a smaller
    `size` takes the first items (pool order), the same list a smaller pool would have kept."""
    pools = [
        _pool(True, [("a", 0.9, 0), ("b", 0.8, 1)]),
        _pool(True, [("a", 0.9, 0), ("b", 0.8, 0), ("c", 0.7, 2)]),
        _pool(True, [("a", 0.9, 1), ("b", 0.8, 0)]),
        _pool(False, [("a", 0.9, 1)]),  # negatives never count
    ]
    full = calibrate.pool_recall(pools)
    assert (full["n"], full["hits"], full["recall"]) == (3, 3, 1.0)
    cut = calibrate.pool_recall(pools, size=1)
    assert (cut["hits"], cut["recall"]) == (1, pytest.approx(1 / 3))
    assert 0.0 <= cut["ci95"][0] < cut["recall"] < cut["ci95"][1] <= 1.0
    assert calibrate.pool_recall([], size=3)["recall"] is None
