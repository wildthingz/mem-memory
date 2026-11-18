"""MEM2-156: `syn-score-dev --fit`: conflict records from a replay gated by a fitted scorer.

The server stays unfiltered (theta_conf = -inf, every pick with its features); the oracle replay
applies the fit's list cut itself (`calibration.fit_cut` = `mem.confmodel.list_cut` on the recorded
features) and links only what that cut surfaces, so the store state evolves as under the gate: a
miss leaves its stale memory active for every later op. Stub-model numbers are smoke only."""

import json

import pytest
from typer.testing import CliRunner

from bench.adapters.p3_ours import P3Backend
from bench.cli import app
from bench.suites.base import Context
from bench.synth import calibration
from bench.synth.run import SynSetup, replay_seed
from mem import confmodel
from mem.testing import HashEmbedder, OverlapReranker, stub_conf_model
from tests.bench_t0.fake_serve import list_tools
from tests.bench_t0.test_p3_real import stub_gate

FIT = {"theta_conf": 0.5, "conf_model": stub_conf_model()}


def _run(tmp_path, *, unfiltered: bool, cut=None):
    b = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()),
                  unfiltered_conflicts=unfiltered)  # fmt: skip
    b.gate = stub_gate(b.cfg)
    try:
        setup = SynSetup((1,), "S", "fixture", cut=cut)
        ctx = Context(tmp_path, b.factory, list_tools, {"syn": setup})
        return replay_seed(ctx, 1, "S")
    finally:
        b.close()


def test_fit_cut_is_the_list_cut_on_the_recorded_features():
    cut = calibration.fit_cut(FIT)
    rows = [{"id": "m3", "features": {"rerank": 2.0}}, {"id": "m1", "features": {"rerank": 2.0}},
            {"id": "m2", "features": {"rerank": 0.2}}]  # fmt: skip
    assert cut(rows) == ["m1", "m3"]  # theta 0.5 drops m2; ties by node id, as the server
    assert cut.gated_by == {"fingerprint": FIT["conf_model"]["fingerprint"], "theta_conf": 0.5}
    with pytest.raises(ValueError, match="features"):
        cut([{"id": "m1", "score": 1.0}])  # a cut list from a gated server: refused, not re-cut


def test_client_cut_surfaces_exactly_what_the_server_gate_surfaces(tmp_path):
    """Same seed, same scorer: the unfiltered server + client cut and the gated server agree per op
    on the surfaced flag and the cut list, hence on the final store (chains, leaks)."""
    gated = _run(tmp_path / "gated", unfiltered=False)
    client = _run(tmp_path / "client", unfiltered=True, cut=calibration.fit_cut(FIT))
    g = [(lg.op["i"], lg.surfaced, lg.conflicts) for lg in gated.logs]
    c = [(lg.op["i"], lg.surfaced, lg.conflicts) for lg in client.logs]
    assert g == c
    assert any(lg.surfaced is False for lg in client.logs), "stub models miss some: a real test"
    assert max(len(lg.conflict_rows) for lg in client.logs) > 5  # the full pool is still recorded
    assert all("features" in r for lg in client.logs for r in lg.conflict_rows)


def test_gated_records_differ_from_clean_records_only_by_state(tmp_path):
    clean = calibration.conflict_records(_run(tmp_path / "clean", unfiltered=True))
    gated = calibration.conflict_records(
        _run(tmp_path / "gated", unfiltered=True, cut=calibration.fit_cut(FIT))
    )
    assert all(r["gated_by"] is None and r["surfaced"] == r["counterpart_returned"] for r in clean)
    assert all(r["gated_by"]["theta_conf"] == 0.5 for r in gated)
    assert any(r["surfaced"] is False and r["counterpart_returned"] for r in gated)
    # the gated state loses a few records: a missed supersession leaves the old text open, so a
    # later op that repeats that value comes back `exists`, not `saved` (never recorded)
    assert {r["op"] for r in gated} < {r["op"] for r in clean}


def test_cli_syn_score_dev_fit_gates_the_replay(tmp_path):
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps(FIT), encoding="utf-8")
    out, conf = tmp_path / "pools.jsonl", tmp_path / "conf.jsonl"
    r = CliRunner().invoke(app, ["syn-score-dev", "--out", str(out), "--conflicts-out", str(conf),
                                 "--seeds", "1", "--size", "S", "--embedder", "hash",
                                 "--fit", str(fit)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "gated by" in r.output
    recs = [json.loads(ln) for ln in conf.read_text(encoding="utf-8").splitlines()]
    assert recs and all(r["gated_by"]["fingerprint"] == FIT["conf_model"]["fingerprint"]
                        for r in recs)  # fmt: skip
    assert max(len(x["candidates"]) for x in recs) > 5  # still the full pool with features
    assert all(set(c["features"]) == set(confmodel.FEATURES) for x in recs for c in x["candidates"])
    # each candidate's validity at pick time and the op's as_of ride along (MEM2-156 round 4), so
    # store-time signals can be computed offline; the fit reads only candidates and flags
    assert all(c["valid_from"] and "valid_to" in c for x in recs for c in x["candidates"])
    assert all(x["as_of"] for x in recs if x["kind"] == "update")
    # the records refit as before (the fit only reads candidates and flags)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theta_conf": 0.5}), encoding="utf-8")
    r = CliRunner().invoke(app, ["syn-score-dev", "--out", str(out), "--conflicts-out", str(conf),
                                 "--seeds", "1", "--size", "S", "--embedder", "hash",
                                 "--fit", str(bad)])  # fmt: skip
    assert r.exit_code == 2 and "conf_model" in r.output
