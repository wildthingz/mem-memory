"""Descriptive gate rows (MEM2-128, the owner 2026-10-24): reported with value and threshold, never failing."""

import pytest

from bench import gates
from bench.suites.base import SuiteResult

YAML = """bench_version: 2
phases:
  PX:
    - {id: B-X, metric: hard, op: ">=", threshold: 0.9, split: test, frozen_at: null}
    - {id: B-X, metric: soft, op: ">=", threshold: 0.9, split: test, frozen_at: null, descriptive: true}
    - {id: B-X, metric: absent, op: "<=", threshold: 1, split: test, frozen_at: null, descriptive: true}
"""


def _run(tmp_path, monkeypatch, metrics):
    y = tmp_path / "g.yaml"
    y.write_text(YAML)
    monkeypatch.setattr(
        gates,
        "suite_for",
        lambda phase, bid: lambda ctx: SuiteResult(bench_id=bid, metrics=metrics),
    )
    return gates.run_phase("PX", None, results_dir=None, gates_path=y)[0]


def test_a_failing_descriptive_row_is_reported_and_does_not_fail(tmp_path, monkeypatch):
    o = _run(tmp_path, monkeypatch, {"hard": 0.95, "soft": 0.1})
    assert o.status == "PASS"
    assert "DESCRIPTIVE soft=0.1" in o.detail and "FAILED" not in o.detail
    assert "absent" in o.detail  # a missing descriptive metric is named, not fatal


def test_a_failing_hard_row_still_fails_beside_descriptive_ones(tmp_path, monkeypatch):
    o = _run(tmp_path, monkeypatch, {"hard": 0.5, "soft": 0.99})
    assert o.status == "FAIL" and "FAILED hard=0.5" in o.detail


def test_gates_yaml_marks_exactly_the_lme_precision_rows_descriptive():
    hard_kept = {"recall_any5_R", "ndcg_std_superior", "warm_p95_ms_median"}
    desc = {"delivered_precision", "ira", "coverage_any", "abstention_recall", "far", "far_drift"}
    for phase, specs in gates.load_gates().items():
        for s in specs:
            if s.id == "B-R3":
                assert not s.descriptive, (phase, s.metric)
            if s.id in ("B-R1", "B-R2", "B-C1") and s.metric in hard_kept:
                assert not s.descriptive, (phase, s.id, s.metric)
            if phase in ("P2", "P3", "P4") and s.id in ("B-R1", "B-R2") and s.metric in desc:
                assert s.descriptive, (phase, s.id, s.metric)
    # P3's SYN-only B-R2 row is not an LME number (a known miss, not descriptive)
    p3 = {(s.id, s.metric): s for s in gates.load_gates()["P3"]}
    assert not p3[("B-R2", "missing_slot_abstention")].descriptive


def test_thresholds_are_copied_not_relaxed():
    for phase in ("P2", "P3", "P4"):
        rows = {(s.id, s.metric): s.threshold for s in gates.load_gates()[phase]}
        assert rows[("B-R1", "recall_any5_R")] == pytest.approx(0.93)
        assert rows[("B-R1", "delivered_precision")] == pytest.approx(0.70)
        assert rows[("B-R2", "abstention_recall")] == pytest.approx(0.85)
        assert rows[("B-R1", "far")] == pytest.approx(0.05)


def test_bench_pool_recall_command_reads_scored_records(tmp_path):
    from typer.testing import CliRunner

    from bench.cli import app
    from mem import calibrate

    pools = [
        calibrate.QueryPool("q1:answerable", True,
                            [calibrate.PoolItem("a", 0.0, 0), calibrate.PoolItem("b", 0.0, 1)], "x"),
        calibrate.QueryPool("q2:answerable", True, [calibrate.PoolItem("a", 0.0, 1)], "y"),
    ]  # fmt: skip
    f = tmp_path / "rec.jsonl"
    calibrate.write_records(f, pools, fp_all="f" * 64)
    res = CliRunner().invoke(
        app, ["pool-recall", "--records", str(f), "--size", "1", "--size", "2"]
    )
    assert res.exit_code == 0, res.output
    assert "pool 1: recall 0.500 (1/2" in res.output and "pool 2: recall 1.000 (2/2" in res.output


KM_YAML = """bench_version: 2
phases:
  PX:
    - {id: B-X, metric: hard, op: ">=", threshold: 0.9, split: test, frozen_at: null}
    - {id: B-X, metric: miss, op: ">=", threshold: 0.9, split: test, frozen_at: null, known_miss: true}
"""


def _run_km(tmp_path, monkeypatch, metrics):
    y = tmp_path / "g.yaml"
    y.write_text(KM_YAML)
    monkeypatch.setattr(
        gates, "suite_for",
        lambda phase, bid: lambda ctx: SuiteResult(bench_id=bid, metrics=metrics),
    )  # fmt: skip
    return gates.run_phase("PX", None, results_dir=tmp_path, gates_path=y)[0]


def test_a_known_miss_below_threshold_is_named_and_does_not_fail(tmp_path, monkeypatch):
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.95, "miss": 0.856})
    assert o.status == "PASS" and "FAILED" not in o.detail
    assert "KNOWN MISS miss=0.856 (need >= 0.9)" in o.detail
    assert "accepted by the owner, 2026-10-25" in o.detail and "PG-v1" in o.detail


def test_a_known_miss_that_meets_its_threshold_is_plain_and_a_hard_row_still_fails(
    tmp_path, monkeypatch
):
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.95, "miss": 0.95})
    assert o.status == "PASS" and "KNOWN MISS" not in o.detail and "miss=0.95" in o.detail
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.5, "miss": 0.1})
    assert (
        o.status == "FAIL" and "FAILED hard=0.5" in o.detail and "KNOWN MISS miss=0.1" in o.detail
    )


def test_a_missing_known_miss_metric_still_fails(tmp_path, monkeypatch):
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.95})
    assert o.status == "FAIL" and "not produced" in o.detail


def test_gates_yaml_marks_exactly_the_b_r5_syn_rows_and_b_r2_ii_as_known_misses_with_thresholds_kept():
    p3 = {(s.id, s.metric): s for s in gates.load_gates()["P3"]}
    assert {k for k, s in p3.items() if s.known_miss} == {
        ("B-R5", "leakage"),
        ("B-R5", "as_of_accuracy"),
        ("B-R5", "current_recall_at_1"),
        (
            "B-R2",
            "missing_slot_abstention",
        ),  # accepted by the owner 2026-10-25 (MEM2-183, ADR 0020)
    }
    assert p3[("B-R2", "missing_slot_abstention")].threshold == pytest.approx(0.90)
    assert p3[("B-R5", "leakage")].threshold == 0
    assert p3[("B-R5", "as_of_accuracy")].threshold == pytest.approx(0.95)
    assert p3[("B-R5", "current_recall_at_1")].threshold == pytest.approx(0.90)
    assert not p3[("B-R5", "case_failures")].known_miss
    for phase, specs in gates.load_gates().items():
        assert phase == "P3" or not any(s.known_miss for s in specs)


def test_freeze_keeps_the_known_miss_flag(tmp_path):
    from bench import freeze

    text = (gates.GATES_YAML).read_text()
    assert "known_miss: true" in text
    assert freeze._ROW.match(
        next(ln for ln in text.splitlines() if ln.lstrip().startswith("- {") and "known_miss" in ln)
    )


def test_a_nan_known_miss_metric_fails_and_a_produced_one_is_a_known_miss(tmp_path, monkeypatch):
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.95, "miss": float("nan")})
    assert o.status == "FAIL" and "KNOWN MISS" not in o.detail
    o = _run_km(tmp_path, monkeypatch, {"hard": 0.95, "miss": 0.5})
    assert o.status == "PASS" and "KNOWN MISS" in o.detail
