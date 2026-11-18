"""`bench auth-overhead` (B-H1): the B-C1 p95 increase from auth/TLS, client-side, base vs secure."""

import json

from typer.testing import CliRunner

from bench import auth_overhead
from bench.cli import app
from mem import auth
from tests.helpers import serve_thread


def _servers(tmp_path):
    tf = auth.TokenFile(tmp_path / "tokens")
    token = tf.add("bench")
    log = auth.AuditLog(tmp_path / "audit.jsonl")
    base, stop_base = serve_thread(tmp_path / "base")
    secure, stop_secure = serve_thread(tmp_path / "secure", auth.Access(
        tokens=tf.load(), limiter=auth.RateLimiter(10_000), audit=log))  # fmt: skip
    return base, secure, token, log, lambda: (stop_base(), stop_secure())


def test_measure_reports_both_legs_and_the_p95_delta(tmp_path):
    base, secure, token, log, stop = _servers(tmp_path)
    try:
        res = auth_overhead.measure(base, secure, token=token, n=30)
    finally:
        stop()
    assert res["n"] == 30 and res["budget_ms"] == 50
    for leg in ("base", "secure"):
        assert 0 < res[leg]["median_ms"] <= res[leg]["p95_ms"]
    assert res["delta_p95_ms"] == round(res["secure"]["p95_ms"] - res["base"]["p95_ms"], 2)
    assert res["pass"] is (res["delta_p95_ms"] <= 50)
    lines = [json.loads(x) for x in log.path.read_text().splitlines()]
    calls = [x for x in lines if x["tool"] == "recall"]
    assert len(calls) == 30 + auth_overhead.WARMUP  # the secure leg went through the guard
    assert {x["client"] for x in calls} == {"bench"}


def test_cli_writes_the_result_and_fails_over_budget(tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    fake = {"n": 3, "base": {"p95_ms": 1.0, "median_ms": 1.0},
            "secure": {"p95_ms": 80.0, "median_ms": 70.0}, "delta_p95_ms": 79.0,
            "budget_ms": 50, "pass": False}  # fmt: skip
    monkeypatch.setattr(auth_overhead, "measure", lambda *a, **kw: fake)
    r = CliRunner().invoke(app, ["auth-overhead", "--base", "http://127.0.0.1:1",
                                 "--secure", "https://x.ts.net", "--token", "t",
                                 "--out", str(out)])  # fmt: skip
    assert r.exit_code == 1 and "FAIL" in r.output
    assert json.loads(out.read_text())["delta_p95_ms"] == 79.0
