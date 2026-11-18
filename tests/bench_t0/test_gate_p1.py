"""The P1 gates (benchmarks-v2 §4) against the real code (`mem.serve.open_inprocess`)."""

import pytest

from bench.adapters import ours
from bench.suites import REGISTRY
from bench.suites.base import Context


@pytest.mark.parametrize(
    ("bench_id", "metric"),
    [
        pytest.param("B-S1", "failures", id="B-S1"),
        pytest.param("B-R6", "violations", id="B-R6-p1-form"),
        pytest.param("B-W6", "violations", id="B-W6-items-1-2-3-5-6"),
        pytest.param("B-R5", "case_failures", id="B-R5-cases-a-e"),
    ],
)
def test_p1_gate(tmp_path, bench_id, metric):
    res = REGISTRY[bench_id](Context(tmp_path, ours.open_client))
    assert res.violations == []
    assert res.metrics[metric] == 0


@pytest.mark.parametrize("bench_id", ["B-C1", "B-C2"])
def test_p1_recorders_run(tmp_path, bench_id):
    res = REGISTRY[bench_id](Context(tmp_path, ours.open_client))
    assert res.violations == [] and res.metrics["n"] > 0
