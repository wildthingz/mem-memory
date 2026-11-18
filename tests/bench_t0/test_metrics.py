import math

import pytest

from bench.metrics import delivered as dv
from bench.metrics import ir, tokens


def test_rank2_counterexample_std_vs_upstream():
    """A6: one relevant item at rank 2 -> upstream 1.0, std 0.63093; they never share a bar."""
    ranked = ["x", "a", "y"]
    assert ir.ndcg_any_upstream(ranked, {"a"}) == pytest.approx(1.0)
    assert ir.ndcg_std(ranked, {"a": 1}) == pytest.approx(0.63093, abs=1e-5)


def test_ndcg_std_graded_and_perfect():
    gold = {"a": 2, "b": 1}
    assert ir.ndcg_std(["a", "b"], gold) == pytest.approx(1.0)
    assert ir.ndcg_std(["b", "a"], gold) == pytest.approx(
        (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    )
    assert ir.ndcg_std(["z"], gold) == 0.0
    assert ir.ndcg_std(["a"], {}) == 0.0


def test_upstream_perfect_with_two_gold_and_cutoff():
    assert ir.ndcg_any_upstream(["a", "b", "c"], {"a", "b"}) == pytest.approx(1.0)
    assert ir.ndcg_any_upstream(["x"] * 10 + ["a"], {"a"}) == 0.0


def test_recall_and_mrr():
    assert ir.recall_any(["x", "a"], {"a", "b"}, 1) == 0.0
    assert ir.recall_any(["x", "a"], {"a", "b"}, 2) == 1.0
    assert ir.recall_all(["a", "b"], {"a", "b"}, 2) == 1.0
    assert ir.recall_all(["a"], {"a", "b"}, 5) == 0.0
    assert ir.mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert ir.mrr(["x"], {"a"}) == 0.0


def test_a16_one_to_one_case_fails_precision_and_ira():
    """One irrelevant row per relevant row: precision 0.5, IRA 1.0 (benchmarks-v2 B-R1 A16)."""
    qs = [dv.Delivered(["r", "i"], frozenset({"r"}), True) for _ in range(4)]
    assert dv.delivered_precision(qs) == 0.5
    assert dv.ira(qs) == 1.0
    three_plus_one = [dv.Delivered(["r1", "r2", "r3", "i"], frozenset({"r1", "r2", "r3"}), True)]
    assert dv.delivered_precision(three_plus_one) == 0.75
    assert dv.ira(three_plus_one) == 1.0


def test_delivered_counts_noanswer_rows_as_irrelevant_and_far_and_abstention():
    qs = [
        dv.Delivered(["r"], frozenset({"r"}), True),
        dv.Delivered([], frozenset({"r"}), True),
        dv.Delivered(["s"], frozenset(), False),
        dv.Delivered([], frozenset(), False),
    ]
    assert dv.delivered_precision(qs) == 0.5
    assert dv.coverage_any(qs) == 0.5
    assert dv.far(qs) == 0.5
    assert dv.abstention_recall(qs) == 0.5
    assert dv.ira(qs) == 0.5


def test_empty_denominator_is_nan_not_zero():
    assert math.isnan(dv.far([dv.Delivered([], frozenset(), False)]))


def test_est_tokens_and_summary():
    assert tokens.est_tokens("abc") == 1 and tokens.est_tokens("abcd") == 2
    assert tokens.est_tokens("é" * 3) == 2  # 6 bytes
    s = tokens.summarize(list(range(1, 21)))
    assert s == {"n": 20, "mean": 10.5, "p95": 19}
    assert tokens.summarize([]) == {"n": 0, "mean": 0.0, "p95": 0}
