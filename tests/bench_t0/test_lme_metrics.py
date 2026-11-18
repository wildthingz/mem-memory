import math
import random

import pytest

from bench.metrics import delivered as dv
from bench.metrics import ir, stats, upstream_lme


def _upstream(ranked, gold, k=10):
    """Run the vendored upstream code on a ranking of string ids."""
    corpus = sorted({*ranked, *gold})
    idx = {d: i for i, d in enumerate(corpus)}
    return upstream_lme.ndcg([idx[d] for d in ranked], set(gold), corpus, k)


def test_rank2_counterexample_upstream_vs_std():
    """A6: one relevant item at rank 2: upstream 1.0, std 0.63093. They never share a threshold."""
    ranked, gold = ["x", "a", "y"], {"a"}
    assert _upstream(ranked, gold) == pytest.approx(1.0)
    assert ir.ndcg_any_upstream(ranked, gold) == pytest.approx(1.0)
    assert ir.ndcg_std(ranked, dict.fromkeys(gold, 1)) == pytest.approx(0.63093, abs=1e-5)


def test_dependency_free_ndcg_any_agrees_with_the_vendored_upstream_code():
    rng = random.Random(3)
    for _ in range(300):
        docs = [f"d{i}" for i in range(rng.randint(1, 40))]
        gold = set(rng.sample(docs, rng.randint(1, min(4, len(docs)))))
        ranked = rng.sample(docs, len(docs))
        assert ir.ndcg_any_upstream(ranked, gold, 10) == pytest.approx(_upstream(ranked, gold, 10))


def test_upstream_recall_any_all_semantics():
    corpus = ["a", "b", "c", "d"]
    any_, all_, _ = upstream_lme.evaluate_retrieval([0, 2], {"a", "b"}, corpus, k=2)
    assert (any_, all_) == (1.0, 0.0)


def test_coverage_all_uses_rel2_set_and_binary_default():
    qs = [
        dv.Delivered(["a", "b"], frozenset({"a", "b"}), True),
        dv.Delivered(["a"], frozenset({"a", "b"}), True),
        dv.Delivered(["a"], frozenset({"a", "b"}), True, required=frozenset({"a"})),
        dv.Delivered(["x"], frozenset(), False),
    ]
    assert dv.coverage_all(qs) == pytest.approx(2 / 3)
    assert math.isnan(dv.coverage_all([dv.Delivered([], frozenset(), False)]))


def test_f05_and_nan_guard():
    assert dv.f05(0.8, 0.9) == pytest.approx(1.25 * 0.8 * 0.9 / (0.25 * 0.8 + 0.9))
    assert math.isnan(dv.f05(math.nan, 0.5)) and dv.f05(0.0, 0.0) == 0.0


def test_auroc_ties_and_empty():
    assert stats.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert stats.auroc([0.5], [0.5]) == 0.5
    assert stats.auroc([0.2, 0.9], [0.5, 0.6]) == pytest.approx(0.5)
    assert math.isnan(stats.auroc([], [0.1]))


def test_bootstrap_is_seeded_stratified_and_brackets_the_point():
    vals = [1.0] * 40 + [0.0] * 10 + [1.0] * 5 + [0.0] * 5
    strata = ["a"] * 50 + ["b"] * 10
    p1 = stats.bootstrap_mean(vals, strata, b=2000, seed=1)
    assert p1 == stats.bootstrap_mean(vals, strata, b=2000, seed=1)
    point, lo, hi = p1
    assert point == pytest.approx(45 / 60) and lo < point < hi
    assert hi - lo == pytest.approx(2 * 1.96 * math.sqrt(0.75 * 0.25 / 60), rel=0.35)
    assert (
        stats.bootstrap_ratio([0, 0], [0, 0], ["a", "a"])[0]
        != stats.bootstrap_ratio([0, 0], [0, 0], ["a", "a"])[0]
    )  # nan


def test_ratio_bootstrap_for_precision():
    num, den = [1, 1, 0, 3], [2, 1, 1, 4]
    point, lo, hi = stats.bootstrap_ratio(num, den, ["a"] * 4, b=1000)
    assert point == pytest.approx(5 / 8) and lo <= point <= hi


def test_paired_bootstrap_detects_a_consistent_gain_and_not_noise():
    rng = random.Random(0)
    base = [rng.random() for _ in range(200)]
    better = [x + 0.05 + rng.gauss(0, 0.01) for x in base]
    strata = ["a"] * 100 + ["b"] * 100
    p = stats.paired_bootstrap(better, base, strata, b=2000)
    assert p.better and p.delta == pytest.approx(0.05, abs=0.01)
    noise = [x + rng.gauss(0, 0.2) for x in base]
    assert not stats.paired_bootstrap(noise, base, strata, b=2000).better


def test_mdd_and_power_formulas_match_section_5_3():
    assert stats.mdd_binary(0.05, 269) == pytest.approx(0.038, abs=0.001)
    assert stats.mdd_binary(0.10, 269) == pytest.approx(0.054, abs=0.001)
    assert stats.noninferiority_power(0.05, 0.10, 269) == pytest.approx(0.74, abs=0.01)
    assert stats.noninferiority_power(0.03, 0.10, 269) == pytest.approx(0.34, abs=0.01)
    assert stats.mdd_continuous(0.2, 100) == pytest.approx(2.80 * 0.2 / 10, rel=1e-3)
    mdd = stats.mdd_continuous(0.2, 100)  # an effect equal to the MDD has ~80% power
    assert stats.superiority_power(mdd, 0.2, 100) == pytest.approx(0.80, abs=0.01)


def test_holm_and_nearest_rank():
    assert stats.holm([0.001, 0.04, 0.03]) == [True, False, False]
    assert stats.holm([0.001, 0.01, 0.02]) == [True, True, True]
    assert stats.percentile_nearest_rank(list(range(1, 101)), 0.95) == 95
