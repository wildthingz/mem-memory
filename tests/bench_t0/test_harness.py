import json

import pytest
from typer.testing import CliRunner

from bench import gates, records
from bench.cli import app
from bench.dump import canonical_dump
from bench.suites import REGISTRY
from tests.bench_t0.fake_serve import Flags, factory_with, list_tools

P1_IDS = ["B-S1", "B-R6", "B-W6", "B-R5", "B-C1", "B-C2"]


def test_gates_yaml_p1_lists_exactly_the_spec_ids():
    specs = gates.load_gates()["P1"]
    assert [s.id for s in specs] == P1_IDS
    assert set(P1_IDS) == set(REGISTRY)
    by = {s.id: s for s in specs}
    assert all(by[i].op == "==" and by[i].threshold == 0 for i in ("B-S1", "B-R6", "B-W6", "B-R5"))
    assert all(by[i].op == "record" and by[i].threshold is None for i in ("B-C1", "B-C2"))


def test_run_phase_writes_record_with_fingerprint_and_items(tmp_path):
    out = gates.run_phase("P1", factory_with(), results_dir=tmp_path, list_tools=list_tools)
    assert [o.status for o in out] == ["PASS"] * 6
    rec_file = next(o.run_file for o in out if o.id == "B-R6")
    rec = json.loads(rec_file.read_text())
    assert rec["bench_version"] == 2 and rec["bench_id"] == "B-R6" and rec["tier"] == "t0"
    assert len(rec["config"]["fp_all"]) == 64 and rec["git_sha"] and rec["hardware"]["cpu_count"]
    assert rec["stubs"]["embedder"].startswith("HashEmbedder")
    assert (
        rec["gates"]
        == [{"phase": "P1", "metric": "violations", "op": "==", "threshold": 0.0, "passed": True}]
        or rec["gates"][0]["passed"] is True
    )
    items = [json.loads(ln) for ln in rec_file.with_suffix(".items.jsonl").read_text().splitlines()]
    assert len(items) == rec["n_items"] > 10 and all("check" in i for i in items)


def test_a_broken_server_fails_its_gate_and_names_why(tmp_path):
    out = gates.run_phase("P1", factory_with(Flags(cross_store=True)), results_dir=None,
                          list_tools=list_tools, only={"B-W6", "B-S1"})  # fmt: skip
    status = {o.id: o for o in out}
    assert status["B-S1"].passed
    assert (
        status["B-W6"].status == "FAIL"
        and "6_id_from_other_store_is_not_found" in status["B-W6"].detail
    )


def _unavailable(*a, **k):
    from bench.suites.base import BackendUnavailable

    raise BackendUnavailable("no backend")


def test_unavailable_backend_is_not_a_pass(tmp_path):
    out = gates.run_phase("P1", _unavailable, results_dir=None)
    assert {o.status for o in out} == {"UNAVAILABLE"}


def test_crashing_suite_is_a_fail_with_the_cause(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("kaput")

    out = gates.run_phase("P1", boom, results_dir=None, list_tools=list_tools, only={"B-R6"})
    assert out[0].status == "FAIL" and "RuntimeError: kaput" in out[0].detail


def test_evaluate_rejects_nan_and_missing_metric():
    from bench.suites.base import SuiteResult

    spec = gates.GateSpec("X", "violations", "==", 0, "fixture", None)
    assert gates.evaluate(spec, SuiteResult("X"))[0] is False
    assert gates.evaluate(spec, SuiteResult("X", metrics={"violations": float("nan")}))[0] is False
    assert gates.evaluate(spec, SuiteResult("X", metrics={"violations": 1.0}))[0] is False
    assert gates.evaluate(spec, SuiteResult("X", metrics={"violations": 0.0}))[0] is True


def test_ledger_appends_one_json_line_and_requires_who_why(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    records.append_ledger({"who": "h", "why": "p1 look", "phase": "P1", "split": "fixture",
                           "config_hash": "abc"}, ledger)  # fmt: skip
    records.append_ledger({"who": "h", "why": "again", "phase": "P1", "split": "fixture",
                           "config_hash": "abc"}, ledger)  # fmt: skip
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert [r["why"] for r in rows] == ["p1 look", "again"] and rows[0]["at"].endswith("Z")
    with pytest.raises(ValueError, match="why"):
        records.append_ledger({"who": "h", "phase": "P1", "split": "t", "config_hash": "x"}, ledger)


def test_cli_gate_prints_status_per_id_and_exits_1_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.cli.ours.open_client", _unavailable)
    r = CliRunner().invoke(app, ["gate", "P1", "--results-dir", str(tmp_path)])
    assert r.exit_code == 1
    lines = r.output.splitlines()
    assert [ln.split()[1] for ln in lines[:6]] == P1_IDS
    assert all(ln.startswith("UNAVAILABLE") for ln in lines[:6])
    assert lines[-1] == "0/6 gates pass for P1"


def test_cli_gate_p1_passes_on_the_real_code(tmp_path):
    r = CliRunner().invoke(app, ["gate", "P1", "--results-dir", str(tmp_path)])
    lines = r.output.splitlines()
    assert r.exit_code == 0, r.output
    assert [ln.split()[1] for ln in lines[:6]] == P1_IDS
    assert all(ln.startswith("PASS") for ln in lines[:6])
    assert lines[-1] == "6/6 gates pass for P1"


def test_cli_gate_unknown_phase_is_an_error(tmp_path):
    r = CliRunner().invoke(app, ["gate", "P9", "--results-dir", str(tmp_path)])
    assert r.exit_code != 0


def test_canonical_dump_is_sorted_jsonl_and_drops_session(tmp_path):
    ctx_factory = factory_with()
    c = ctx_factory(tmp_path / "d.db", scope="personal")
    c.call("remember", {"text": "x one", "type": "fact", "about": ["new:topic:T"]},
           now="2026-10-01T00:00:00Z")  # fmt: skip
    c.close()
    dump = canonical_dump(c.db_path).decode()
    rows = [json.loads(ln) for ln in dump.splitlines()]
    assert {r["t"] for r in rows} >= {"nodes", "edges", "chunks"}
    assert all("session" not in r for r in rows if r["t"] == "nodes")
