import numpy as np
import pytest

from mem.spike import cosine_parity, logit_parity, percentile, summarize_ms, verdict


def test_percentile_interpolates_linearly():
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 95) == pytest.approx(38.5)


def test_percentile_single_value_and_empty():
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_ms_reports_p50_p95_and_count():
    s = summarize_ms([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s == {"n": 5, "p50": 3.0, "p95": pytest.approx(4.8), "min": 1.0, "max": 5.0}


def test_cosine_parity_min_over_rows():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 1.0]])
    r = cosine_parity(a, b)
    assert r["min_cosine"] == pytest.approx(0.70710678, abs=1e-7)
    assert r["ok"] is False


def test_cosine_parity_passes_at_threshold():
    a = np.array([[1.0, 0.0]])
    assert cosine_parity(a, a.copy())["ok"] is True


def test_logit_parity_uses_max_abs_delta_with_1e3_limit():
    ok = logit_parity([1.0, 2.0], [1.0005, 2.0])
    assert ok["max_abs_delta"] == pytest.approx(0.0005)
    assert ok["ok"] is True
    assert logit_parity([1.0], [1.002])["ok"] is False


def _lat(avg, worst):
    return {"avg": {"p95": avg}, "worst": {"p95": worst}}


def test_verdict_pass():
    v = verdict(gpu=_lat(300.0, 700.0), cosine_ok=True, logit_ok=True)
    assert v["outcome"] == "pass"
    assert v["device_in_fingerprint"] is False


def test_verdict_parity_failure_puts_device_in_fingerprint():
    v = verdict(gpu=_lat(300.0, 700.0), cosine_ok=True, logit_ok=False)
    assert v["outcome"] == "parity-fail"
    assert v["device_in_fingerprint"] is True


def test_verdict_slow_gpu_needs_dr2_fallback():
    v = verdict(gpu=_lat(1200.0, 2000.0), cosine_ok=True, logit_ok=True)
    assert v["outcome"] == "dr2-fallback"


def test_verdict_worst_case_over_limit_does_not_fail_average_pass():
    # the 1,000 ms limit is judged on the average-pair p95; worst case is reported only
    v = verdict(gpu=_lat(900.0, 1500.0), cosine_ok=True, logit_ok=True)
    assert v["outcome"] == "pass"


def test_verdict_no_gpu():
    assert verdict(gpu=None, cosine_ok=None, logit_ok=None)["outcome"] == "no-gpu"
