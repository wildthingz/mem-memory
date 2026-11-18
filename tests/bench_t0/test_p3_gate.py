"""`bench gate P3`: the gate rows, the registry, freeze/looks, the CLI (incl. one run against the real
server on the stub models), B-C3 and the SYN calibration data. The P3 T0 suites on the real store,
case by case: test_p3_real.py."""

import json
import math
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bench import freeze, gates, latency
from bench.cli import app
from bench.suites import PHASE_REGISTRY, REGISTRY, b_c3
from bench.suites.base import BackendUnavailable, Context
from bench.synth import calibration
from bench.synth import data as D
from bench.synth.run import SynSetup, replay_seed
from mem.confmodel import FEATURES
from mem.testing import stub_conf_model
from tests.bench_t0.fake_p3 import P3Flags, factory_p3
from tests.bench_t0.fake_serve import list_tools

SPEC = Path(__file__).resolve().parents[2] / "deliverables" / "benchmarks-v2.md"

# benchmarks-v2 §4 P3 row and the B-W/B-R/B-C sections, written out by hand: (id, metric) -> (op, thr).
SPEC_THRESHOLDS = {
    ("B-W1", "wrong_merge_rate"): ("<=", 0.005), ("B-W1", "ask_back_recall"): (">=", 0.90),
    ("B-W1", "ask_rate"): ("<=", 0.10), ("B-W1", "split_rate"): ("<=", 0.02),
    ("B-W1", "final_link_accuracy"): (">=", 0.98),
    ("B-W2", "exists_rate"): (">=", 1), ("B-W2", "paraphrase_surfacing"): (">=", 0.90),
    ("B-W2", "final_duplicate_rate"): ("<=", 0.03), ("B-W2", "wrong_invalidation_compat"): ("==", 0),
    ("B-W3", "surfacing_recall"): (">=", 0.95), ("B-W3", "list_precision"): (">=", 0.50),
    ("B-W3", "chain_correctness"): (">=", 0.95), ("B-W3", "wrong_invalidation"): ("==", 0),
    ("B-W3", "slot_accuracy"): (">=", 0.95), ("B-W3", "history"): (">=", 1),
    ("B-W5", "capture"): (">=", 0.80), ("B-W5", "noise_rate"): ("<=", 0.20),
    ("B-W5", "about_fill"): (">=", 0.98), ("B-W5", "conflict_action"): (">=", 0.85),
    ("B-W5", "wrong_invalidation"): ("<=", 0.02), ("B-W5", "alignment_audit_wrong"): ("<=", 1),
    ("B-W6", "violations"): ("==", 0), ("B-R5", "case_failures"): ("==", 0),
    ("B-R5", "leakage"): ("==", 0), ("B-R5", "as_of_accuracy"): (">=", 0.95),
    ("B-R5", "current_recall_at_1"): (">=", 0.90), ("B-R2", "missing_slot_abstention"): (">=", 0.90),
    ("B-R2", "abstention_recall"): (">=", 0.85), ("B-R2", "far_drift"): ("<=", 0.03),
    ("B-R1", "recall_any5_R"): (">=", 0.93), ("B-R1", "coverage_any"): (">=", 0.85),
    ("B-R1", "far"): ("<=", 0.05), ("B-R1", "delivered_precision"): (">=", 0.70),
    ("B-R1", "ira"): ("<=", 0.5), ("B-C3", "warm_p95_ms_median"): ("<=", 1500),
    ("B-E1", "accuracy"): (">=", 0.75),
}  # fmt: skip
HARNESS_ROWS = {("B-W1", "table_case_failures"), ("B-W2", "case_failures"),
                ("B-W3", "case_failures"), ("B-W5", "run_complete"), ("B-E1", "run_complete"), ("B-R2", "far"),
                ("B-R1", "ndcg_std_superior")}  # fmt: skip


def rows():
    return gates.load_gates()["P3"]


def test_every_spec_threshold_is_a_gate_row_and_nothing_else_is_invented():
    got = {(r.id, r.metric): (r.op, r.threshold) for r in rows()}
    for key, want in SPEC_THRESHOLDS.items():
        assert got[key] == want, key
    extra = set(got) - set(SPEC_THRESHOLDS) - HARNESS_ROWS
    assert extra == set(), (
        f"gate rows that are in neither the spec table nor the harness list: {extra}"
    )


def test_p3_gate_ids_are_exactly_the_spec_p3_row():
    if not SPEC.exists():
        pytest.skip("deliverables/ is not part of this tree")
    text = SPEC.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| **P3 Write quality**"))
    ids = set(re.findall(r"\*\*(B-[A-Z]\d)\*\*", row)) - {"B-R1"} | {"B-R1", "B-R2"}
    assert {r.id for r in rows()} == ids | {"B-C3"}


def test_every_gate_id_has_a_p3_runner():
    for bench_id in {r.id for r in rows()}:
        assert bench_id in PHASE_REGISTRY["P3"], bench_id
    assert REGISTRY["B-W6"] is not PHASE_REGISTRY["P3"]["B-W6"]  # the P1 form is kept for P1


def test_syn_gates_use_held_out_seeds_and_dev_seeds_tune():
    assert D.HELD_OUT_SEEDS == (101, 102, 103, 104, 105) and D.DEV_SEEDS == (1, 2, 3, 4, 5)


def test_run_phase_p3_on_the_stand_in_records_per_seed_and_the_config(tmp_path):
    out = gates.run_phase(
        "P3", factory_p3(P3Flags(oracle_recall=True)), results_dir=tmp_path,
        only={"B-W1", "B-W2", "B-W3", "B-W6", "B-R5"}, split="tune",
        options={"syn": SynSetup((1, 2), "S", "tune")})  # fmt: skip
    by = {o.id: o for o in out}
    assert {"B-W1", "B-W2", "B-W3", "B-W6", "B-R5"} == set(by)
    assert by["B-W6"].status == "PASS" and by["B-W2"].status == "PASS"
    rec = json.loads(by["B-W3"].run_file.read_text(encoding="utf-8"))
    assert rec["bench_id"] == "B-W3" and rec["config"]["fp_all"] and rec["mode"] == "oracle_client"
    assert rec["dataset"]["name"] == "SYN" and rec["dataset"]["seeds"] == [1, 2]
    assert set(rec["extra"]["per_seed"]) == {"1", "2"}  # per-seed values are recorded


def test_unavailable_server_is_a_failed_gate_never_a_pass(tmp_path):
    from tests.bench_t0.fake_serve import factory_with

    out = gates.run_phase("P3", factory_with(), results_dir=None, only={"B-W3", "B-W6"},
                          options={"syn": SynSetup((1,), "S", "fixture")})  # fmt: skip
    assert [o.status for o in out] == ["UNAVAILABLE", "UNAVAILABLE"]
    assert not any(o.passed for o in out)


def test_w5_gate_is_unavailable_without_a_run_and_fails_without_the_audit(tmp_path):
    out = gates.run_phase("P3", factory_p3(), results_dir=None, only={"B-W5"})
    assert out[0].status == "UNAVAILABLE"


def test_thresholds_freeze_once_and_then_test_looks_are_allowed_up_to_three(tmp_path):
    g, ledger = tmp_path / "gates.yaml", tmp_path / "ledger.jsonl"
    g.write_text(gates.GATES_YAML.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(freeze.Refused, match="not frozen"):
        freeze.check_test_look_allowed("P3", g, ledger)
    freeze.freeze("P3", who="t", reason="tune data only", config_hash="x", gates_path=g,
                  ledger=ledger)  # fmt: skip
    freeze.check_test_look_allowed("P3", g, ledger)
    frozen = [r for r in gates.load_gates(g)["P3"] if r.split == "test"]
    assert frozen and all(r.frozen_at for r in frozen)
    assert all(r.frozen_at is None for r in gates.load_gates(g)["P3"] if r.split == "fixture")
    with pytest.raises(freeze.Refused, match="already frozen"):
        freeze.freeze("P3", who="t", reason="again", config_hash="x", gates_path=g, ledger=ledger)
    yaml.safe_load(g.read_text(encoding="utf-8"))  # still valid


# ---- CLI -----------------------------------------------------------------------------------------


def test_cli_syn_generate_writes_the_three_files(tmp_path):
    r = CliRunner().invoke(app, ["syn-generate", "--seed", "1", "--size", "S", "--out",
                                 str(tmp_path)])  # fmt: skip
    assert r.exit_code == 0, r.output
    d = tmp_path / "S1"
    assert {p.name for p in d.iterdir()} == {"ops.jsonl", "expected_state.json", "queries.jsonl"}
    want = dict(line.split()[::-1] for line in
                (Path(__file__).resolve().parents[2] / "bench/synth/golden/seed1_S.sha256")
                .read_text().splitlines())  # fmt: skip
    assert json.loads(r.output[r.output.index("{") :])["sha256"] == want


def test_cli_w5_run_refuses_without_yes_and_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = CliRunner().invoke(app, ["w5-run", "--out", str(tmp_path / "r.json")])
    assert r.exit_code == 2 and "--yes" in r.output
    r = CliRunner().invoke(app, ["w5-run", "--out", str(tmp_path / "r.json"), "--yes"])
    assert r.exit_code == 2 and "ANTHROPIC_API_KEY" in r.output


def test_cli_gate_p3_test_look_needs_a_ledger_reason_and_frozen_thresholds():
    r = CliRunner().invoke(app, ["gate", "P3", "--split", "test"])
    assert r.exit_code == 2 and "--ledger" in r.output
    r = CliRunner().invoke(app, ["gate", "P3", "--split", "test", "--ledger", "why"])
    assert r.exit_code == 2 and "not frozen" in r.output


def test_cli_syn_score_dev_refuses_held_out_seeds(tmp_path):
    r = CliRunner().invoke(app, ["syn-score-dev", "--out", str(tmp_path / "a"), "--conflicts-out",
                                 str(tmp_path / "b"), "--seeds", "101", "--embedder", "hash"])  # fmt: skip
    assert r.exit_code == 2 and "not dev seeds" in r.output


def test_cli_gate_p3_runs_the_link_and_forget_cases_against_the_real_server(tmp_path):
    """`bench gate P3 --embedder hash` on SYN-S seed 1: the bench's own serve with the stub models,
    real stores. B-W6 (link idempotence, same_as, two writers) passes; B-R5 runs its link/forget
    cases (a)-(f) and the only one failing is (d) (F1, xfailed in test_p3_real.py). B-R5's three
    SYN rows miss with the stub reranker (it needs the real one, T1, runbooks/p3-gate.md); they are
    KNOWN MISSES (the owner, 2026-10-25): named in the output, the run does not fail."""
    from dataclasses import asdict

    from bench.adapters.lme_ours import load_config
    from tests.bench_t0.test_p3_real import stub_gate

    g = tmp_path / "gate.json"
    g.write_text(json.dumps(asdict(stub_gate(load_config(None, "hash")))), encoding="utf-8")
    res = tmp_path / "res"
    r = CliRunner().invoke(app, ["gate", "P3", "--embedder", "hash", "--gate", str(g),
                                 "--syn-size", "S", "--syn-seeds", "1", "--only", "B-W6",
                                 "--only", "B-R5", "--results-dir", str(res)])  # fmt: skip
    assert "UNAVAILABLE" not in r.output, r.output
    assert re.search(r"^PASS +B-W6 ", r.output, re.MULTILINE), r.output
    assert re.search(
        r"^PASS +B-R5 .*KNOWN MISS leakage=.*KNOWN MISS as_of_accuracy=.*"
        r"KNOWN MISS current_recall_at_1=",
        r.output,
        re.MULTILINE,
    ), r.output
    assert r.exit_code == 0 and r.output.splitlines()[-1].startswith("2/2 gates pass for P3")
    (items,) = res.glob("*-B-R5.items.jsonl")
    checks = [json.loads(x) for x in items.read_text(encoding="utf-8").splitlines()]
    checks = [c for c in checks if "check" in c]
    for case in "abcdef":
        assert any(c["check"].startswith(f"{case}_") for c in checks), f"case ({case}) not run"
    bad = {c["check"] for c in checks if not c["ok"]}
    assert bad <= {"d_chain_rederives_without_the_retracted", "d_retracted_member_excluded"}, bad


# ---- B-C3 ----------------------------------------------------------------------------------------


def _run(p95, cold=900.0):
    s = {"median": p95 / 2, "p95": p95}
    return {"cold_ms": cold, "n": 300, "one_client": {"total": s}, "four_clients": {"total": s},
            "corpus_fingerprint": "f"}  # fmt: skip


def test_write_process_uses_distinct_calls_for_every_phase():
    seen = []

    def server(q):
        seen.append(q)
        return {"total": 10.0}

    n = 1 + latency.WARMUP + 2 * latency.N_QUERIES
    res = latency.run_write_process(server, [str(i) for i in range(n)])
    assert len(seen) == n and len(set(seen)) == n  # nothing repeated: no `exists` hits
    assert res["one_client"]["total"]["p95"] == 10.0
    with pytest.raises(ValueError, match="need"):
        latency.run_write_process(server, ["0"] * 10)


def test_c3_gate_value_is_the_median_of_three_runs(tmp_path):
    paths = []
    for i, p95 in enumerate((900.0, 1700.0, 1100.0)):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_run(p95)), encoding="utf-8")
        paths.append(p)
    res = b_c3.run(Context(tmp_path, factory_p3(), list_tools, {b_c3.OPT: {"L5k": paths}}))
    assert res.metrics["warm_p95_ms_median"] == 1100.0 and res.metrics["warm_p95_ms_max"] == 1700.0
    assert "warm_p95_ms_median_l50k" not in res.metrics
    out = gates.run_phase("P3", factory_p3(), results_dir=None, only={"B-C3"},
                          options={b_c3.OPT: {"L5k": paths}})  # fmt: skip
    assert out[0].status == "PASS"


def test_c3_needs_three_runs_and_records_l50k_when_given(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_run(2000.0)), encoding="utf-8")
    with pytest.raises(BackendUnavailable, match="needs 3"):
        b_c3.run(Context(tmp_path, factory_p3(), list_tools, {b_c3.OPT: {"L5k": [p, p]}}))
    res = b_c3.run(Context(tmp_path, factory_p3(), list_tools,
                           {b_c3.OPT: {"L5k": [p] * 3, "L50k": [p] * 3}}))  # fmt: skip
    assert res.metrics["warm_p95_ms_median"] == 2000.0 == res.metrics["warm_p95_ms_median_l50k"]
    out = gates.run_phase("P3", factory_p3(), results_dir=None, only={"B-C3"},
                          options={b_c3.OPT: {"L5k": [p] * 3}})  # fmt: skip
    assert out[0].status == "FAIL"  # 2,000 ms > 1,500 ms


# ---- SYN dev data for mem calibrate --------------------------------------------------------------


def _ctx(tmp_path, seed=1):
    return Context(tmp_path, factory_p3(P3Flags(oracle_recall=True)), list_tools,
                   {"syn": SynSetup((seed,), "S", "tune")})  # fmt: skip


def test_conflict_records_carry_scores_and_gold_flags(tmp_path):
    recs = calibration.conflict_records(replay_seed(_ctx(tmp_path), 1, "S"))
    assert recs and {r["kind"] for r in recs} == {"update", "dup_paraphrase"}
    assert all(0 <= len(r["candidates"]) <= 5 for r in recs)
    hit = [r for r in recs if r["counterpart_returned"]]
    assert len(hit) / len(recs) > 0.9
    assert any(c["same_slot"] and c["counterpart"] for r in hit for c in r["candidates"])


def test_fit_theta_conf_maximises_recall_subject_to_precision():
    def rec(counterpart_score, noise_scores):
        cands = [{"score": counterpart_score, "same_slot": True, "counterpart": True}]
        cands += [{"score": s, "same_slot": False, "counterpart": False} for s in noise_scores]
        return {"candidates": cands}

    # 4 ops; the counterpart scores 5.0 (x3) and 2.0 (x1); noise scores 1.0 or 3.0
    recs = [rec(5.0, [1.0, 1.0, 1.0]), rec(5.0, [1.0, 3.0]), rec(5.0, [1.0]), rec(2.0, [1.0, 1.0])]
    fit = calibration.fit_theta_conf(recs)
    assert fit["feasible"] and fit["list_precision"] >= 0.5
    assert fit["theta_conf"] == 2.0 and fit["surfacing_recall"] == 1.0  # all four ops surfaced
    assert fit["at_theta_0"]["list_precision"] < 0.5  # theta 0 would drown the list
    # a cut that keeps precision >= .5 but loses an op is worse than the one that keeps all four


def test_fit_theta_conf_reports_infeasible_instead_of_pretending():
    recs = [
        {
            "candidates": [
                {"score": 1.0, "same_slot": False, "counterpart": False},
                {"score": 2.0, "same_slot": False, "counterpart": False},
                {"score": 3.0, "same_slot": True, "counterpart": True},
            ]
        }
    ]
    fit = calibration.fit_theta_conf(recs, floor=1.1)
    assert fit["feasible"] is False


def test_fit_theta_conf_cap_is_top_five():
    cands = [{"score": float(i), "same_slot": i == 0, "counterpart": i == 0} for i in range(8)]
    fit = calibration.fit_theta_conf([{"candidates": cands}], floor=0.0)
    assert fit["surfacing_recall"] in (0.0, 1.0) and fit["returned"] <= 5


def test_merge_records_needs_one_fingerprint(tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    a.write_text('{"fp_all": "x"}\n{"qid": "1", "answerable": true, "items": []}\n')
    b.write_text('{"fp_all": "x"}\n{"qid": "2", "answerable": false, "items": []}\n')
    c.write_text('{"fp_all": "y"}\n{"qid": "3", "answerable": false, "items": []}\n')
    out = tmp_path / "m.jsonl"
    assert calibration.merge_records([a, b], out) == 2
    assert out.read_text().splitlines()[0] == '{"fp_all": "x"}'
    with pytest.raises(ValueError, match="differs"):
        calibration.merge_records([a, c], out)


def test_dev_pools_label_the_valid_member_and_no_answer_probes(tmp_path):
    """The labels, on a store with the stub reranker: gold current = rel 2, stale members 0,
    missing-slot probes are no-answer queries."""
    from mem.calibrate import PoolItem  # noqa: F401
    from mem.testing import HashEmbedder, OverlapReranker

    run = replay_seed(_ctx(tmp_path), 1, "S")
    from mem.store import Store

    store = Store.open(run.db, HashEmbedder(256), scope="personal", reranker=OverlapReranker(),
                       tz=D.TZ)  # fmt: skip
    try:

        class _S:
            pass

        srv = _S()
        srv.store = store
        pools = calibration.dev_pools([(run, srv)])
    finally:
        store.close()
    q = {f"syn1:{x['id']}": x for x in run.syn.queries}
    assert len(pools) == len(q)
    assert all(p.answerable == (q[p.qid]["type"] != "missing_slot") for p in pools)
    miss = [p for p in pools if not p.answerable]
    assert miss and all(i.rel == 0 for p in miss for i in p.items)
    cur = [p for p in pools if q[p.qid]["type"] == "current"]
    assert any(i.rel == 2 for p in cur for i in p.items)
    assert all(i.rel in (0, 2) for p in pools for i in p.items)
    assert not math.isnan(sum(i.z for p in pools for i in p.items))
    # MEM2-155: the read model's cosine is recorded for every candidate (real store states)
    assert all(i.cos is not None and -1.0 <= i.cos <= 1.0 for p in pools for i in p.items)


def test_cli_gate_add_theta_sets_only_theta_conf(tmp_path):
    gate = {"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.6, "gap": 0.1,
            "theta_conf": None, "fit_id": "x", "fingerprint_hash": "abc"}  # fmt: skip
    g, t, o = tmp_path / "g.json", tmp_path / "t.json", tmp_path / "o.json"
    g.write_text(json.dumps(gate))
    model = stub_conf_model()
    t.write_text(json.dumps({"theta_conf": 2.5, "feasible": True, "surfacing_recall": 0.97,
                             "list_precision": 0.61, "conf_model": model}))  # fmt: skip
    r = CliRunner().invoke(app, ["gate-add-theta", str(g), str(t), "--out", str(o)])
    assert r.exit_code == 0, r.output
    assert json.loads(o.read_text()) == {**gate, "theta_conf": 2.5, "conf_model": model}
    t.write_text(json.dumps({"theta_conf": 2.5, "feasible": True}))  # no coefficients
    r = CliRunner().invoke(
        app, ["gate-add-theta", str(g), str(t), "--out", str(tmp_path / "m.json")]
    )
    assert r.exit_code == 2 and "conf_model" in r.output and not (tmp_path / "m.json").exists()
    t.write_text(json.dumps({"theta_conf": 0.0, "feasible": False}))
    r = CliRunner().invoke(
        app, ["gate-add-theta", str(g), str(t), "--out", str(tmp_path / "n.json")]
    )
    assert r.exit_code == 2 and not (tmp_path / "n.json").exists()


def test_cli_syn_score_dev_writes_unfiltered_conflict_candidates(tmp_path):
    """MEM2-130: without a gate store.gate was None and `conflicts.find` returned [] for every op, so
    syn-fit-theta was infeasible. The dev run uses theta_conf = -inf: lists are non-empty and
    carry the counterpart of at least one update/paraphrase op."""
    out, conf = tmp_path / "pools.jsonl", tmp_path / "conf.jsonl"
    r = CliRunner().invoke(app, ["syn-score-dev", "--out", str(out), "--conflicts-out", str(conf),
                                 "--seeds", "1", "--size", "S", "--embedder", "hash"])  # fmt: skip
    assert r.exit_code == 0, r.output
    recs = [json.loads(ln) for ln in conf.read_text(encoding="utf-8").splitlines()]
    assert recs and any(x["candidates"] for x in recs)
    assert any(x["counterpart_returned"] for x in recs)
    # every pick before the MAX_OUT cut is kept, with its features
    assert max(len(x["candidates"]) for x in recs) > 5
    assert all(set(c["features"]) == set(FEATURES) for x in recs for c in x["candidates"])
    assert all(x["subtype"] for x in recs if x["kind"] == "update")


def test_p3_backend_without_theta_conf_is_refused_not_silently_unfiltered(tmp_path):
    """A gate-less (or theta-less) P3 store skips conflict detection; B-W2/W3/W5/C3 must not run on it."""
    from dataclasses import replace

    from bench.adapters.p3_ours import P3Backend
    from mem.testing import HashEmbedder, OverlapReranker
    from tests.bench_t0.test_p3_real import stub_gate

    b = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()))
    try:
        with pytest.raises(BackendUnavailable, match="theta_conf"):
            b.open_server(tmp_path / "a.db")
        b.gate = replace(stub_gate(b.cfg), theta_conf=None)
        with pytest.raises(BackendUnavailable, match="theta_conf"):
            b.open_server(tmp_path / "b.db")
        b.gate = stub_gate(b.cfg)
        b.open_server(tmp_path / "c.db")[0].close()
    finally:
        b.close()
