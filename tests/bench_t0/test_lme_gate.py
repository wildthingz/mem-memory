"""B-R1, B-R2 (i), B-C1, B-C2, B-R6 (P2 form), `bench gate P2`, freeze and looks, on fixtures."""

import json
import math
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import freeze, gates, latency, lme_eval, records
from bench.cli import app
from bench.datasets import lme, negatives
from bench.suites import b_p2_cost
from bench.suites.lme_setup import LmeSetup
from tests.bench_t0.fake_lme import FakeLme
from tests.bench_t0.fake_serve import factory_with, list_tools

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
YAML = Path(__file__).resolve().parents[2] / "bench" / "configs" / "gates.yaml"


def setup(split="test", backend=None, **kw) -> LmeSetup:
    qs = lme.load_questions(FIX / "lme_s_mini.json")
    manifest = lme.build_manifest(qs)
    return LmeSetup(backend=backend or FakeLme(), questions={q["question_id"]: q for q in qs},
                    manifest=manifest, negatives=negatives.build_negatives(qs, manifest),
                    split=split, bootstrap_b=300, **kw)  # fmt: skip


def run_p2(s, only, tmp_path, **kw):
    return gates.run_phase("P2", factory_with(), results_dir=tmp_path, list_tools=list_tools,
                           options={"lme": s}, split=s.split, only=set(only), **kw)  # fmt: skip


def test_run_lme_pairs_answerable_negative_and_abs():
    items = lme_eval.run_lme(FakeLme(), setup().questions, setup().manifest, "test")
    kinds = [i.kind for i in items]
    assert kinds.count("answerable") == kinds.count("negative") == 8 and kinds.count("abs") == 2
    neg = next(i for i in items if i.kind == "negative")
    assert neg.gold_rounds == frozenset() and neg.abstained
    ans = next(i for i in items if i.kind == "answerable")
    assert ans.r.R and lme.session_of(ans.r.R[0]) in ans.gold_sessions
    assert ans.top8_text  # for the B-C2 ratio


def test_scores_on_a_perfect_backend_and_a_never_abstaining_one():
    s = setup()
    items = lme_eval.run_lme(FakeLme(), s.questions, s.manifest, "test")
    sc = lme_eval.score(items, b=300)
    m = sc.metrics
    assert m["recall_any5_R"] == m["recall_any5_F"] == 1.0 and m["recall_all10_R"] == 1.0
    assert m["ndcg_std10_R"] == pytest.approx(1.0) and m["ndcg_any10_upstream_R"] == pytest.approx(
        1.0
    )
    assert (m["coverage_any"], m["far"], m["abstention_recall"]) == (1.0, 0.0, 1.0)
    assert m["delivered_precision"] == pytest.approx(1.0)  # only the has_answer round qualifies
    assert m["abs_abstention_recall"] == 1.0 and m["abs_n"] == 2
    assert m["auroc_p1"] == 1.0
    assert set(sc.per_category["recall_any5_R"]) == {
        "knowledge-update",
        "temporal-reasoning",
        "multi-session",
        "single-session-user",
    }
    lo, hi = sc.ci95["coverage_any"]
    assert lo <= m["coverage_any"] <= hi
    # MEM2-83: the delivered metrics per question type, and `_abs` on its own (abstention only)
    for name in ("delivered_precision", "ira", "far", "abstention_recall", "coverage_any", "n"):
        assert set(sc.per_category[name]) == set(sc.per_category["recall_any5_R"]) | {"_abs"}, name
    assert sc.per_category["far"]["multi-session"] == 0.0
    assert sc.per_category["abstention_recall"]["_abs"] == 1.0
    assert sc.per_category["n"]["_abs"] == 2 and sc.per_category["n"]["multi-session"] == 4
    assert all(math.isnan(sc.per_category[k]["_abs"]) for k in ("coverage_any", "far"))
    # fixed top 8, no gate (the A16 failure shape): never abstains, mostly irrelevant rows
    bad = lme_eval.score(lme_eval.run_lme(FakeLme(never_abstain=True), s.questions, s.manifest,
                                          "test"), b=300)  # fmt: skip
    assert bad.metrics["abstention_recall"] == 0.0 and bad.metrics["ira"] > 5
    assert bad.metrics["delivered_precision"] < 0.2 and bad.metrics["coverage_any"] == 1.0


def test_session_level_ranking_dedupes_before_ranking():
    it = lme_eval.Item("q", "multi-session", "answerable", frozenset({"g"}), frozenset(),
                       lme_eval.Retrieved(F=[], R=["x#0", "x#1", "x#2", "g#0"], S=[]))  # fmt: skip
    v = lme_eval.ranking_vectors([it])
    assert v["ndcg_std10_R"] == [pytest.approx(1 / math.log2(3))]  # g is session rank 2, not 4
    assert v["recall_any5_R"] == [1.0]


def test_b_r1_gate_rows_pass_on_the_perfect_backend_and_name_failures(tmp_path):
    s = setup()
    s.tune_record = {"metrics": {"ndcg_power": 0.5, "far": 0.0}}
    (out,) = run_p2(s, {"B-R1"}, tmp_path)
    assert out.status == "PASS", out.detail
    rec = json.loads(out.run_file.read_text())
    assert rec["dataset"]["split"] == "test" and rec["bench_id"] == "B-R1" and rec["tier"] == "t1"
    assert len(rec["gates"]) == 6 and all(g["passed"] for g in rec["gates"])
    assert rec["extra"]["ci95"]["coverage_any"] and rec["extra"]["n"]["answerable"] == 8
    items = [
        json.loads(x) for x in out.run_file.with_suffix(".items.jsonl").read_text().splitlines()
    ]
    assert {"qid", "category", "F", "R", "S", "abstained", "gold", "latency_ms"} <= items[0].keys()
    assert len(items) == 18
    bad = setup(backend=FakeLme(never_abstain=True))
    bad.tune_record = s.tune_record
    (out,) = run_p2(bad, {"B-R1"}, tmp_path)
    assert (
        out.status
        == "PASS"  # LME precision-side rows are descriptive (MEM2-128): shown, not gating
        and "DESCRIPTIVE ira=" in out.detail
        and "DESCRIPTIVE delivered_precision=" in out.detail
    )


def test_tune_run_prints_the_per_category_table_from_the_record(tmp_path):
    """MEM2-83: `bench gate P2 --split tune` shows the delivered metrics per question type and
    `_abs` after the gate lines, read back from the B-R1 record (`extra.per_category`)."""
    from bench.cli import per_category_table

    s = setup(split="tune")
    s.tune_record = {"metrics": {"ndcg_power": 0.5, "far": 0.0}}
    (out,) = run_p2(s, {"B-R1"}, tmp_path)
    table = per_category_table(out.id, out.run_file)
    lines = table.splitlines()
    assert lines[0].startswith("  B-R1 per category (tune):") and "precision" in lines[0]
    cats = [ln.split()[0] for ln in lines[1:]]
    assert cats[-1] == "_abs" and "multi-session" in cats and len(cats) == 5
    ms = next(ln for ln in lines if ln.strip().startswith("multi-session")).split()
    assert ms[1:] == ["2", "1.000", "1.000", "0.000", "0.000", "1.000"]  # n cov prec ira far abst
    absr = lines[-1].split()
    assert absr[1] == "0" and absr[2] == "-"  # no _abs in the mini tune split: n 0, nan -> "-"
    # a P1-style record without per-category values prints nothing
    rec = json.loads(out.run_file.read_text())
    rec["extra"].pop("per_category")
    (tmp_path / "p1.json").write_text(json.dumps(rec))
    assert per_category_table("B-R1", tmp_path / "p1.json") == ""


def test_nan_metric_cannot_pass_and_superiority_needs_tune_power(tmp_path):
    s = setup()  # test split, no tune record: power unknown
    (out,) = run_p2(s, {"B-R1"}, tmp_path)
    assert out.status == "FAIL" and "FAILED ndcg_std_superior=nan" in out.detail
    on_tune = setup("tune")
    (out,) = run_p2(on_tune, {"B-R1"}, tmp_path)
    rec = json.loads(out.run_file.read_text())
    assert not math.isnan(rec["metrics"]["ndcg_power"]) and rec["dataset"]["split"] == "tune"


def test_b_r2_drift_comes_from_the_tune_record(tmp_path):
    s = setup()
    s.tune_record = {"metrics": {"far": 0.0}}
    (out,) = run_p2(s, {"B-R2"}, tmp_path)
    assert out.status == "PASS", out.detail
    s2 = setup()
    s2.tune_record = {"metrics": {"far": 0.10}}
    (out,) = run_p2(s2, {"B-R2"}, tmp_path)
    assert out.status == "PASS" and "DESCRIPTIVE far_drift=0.1" in out.detail
    (out,) = run_p2(setup("tune"), {"B-R2"}, tmp_path)
    assert "far_drift=nan" in out.detail  # reported, though descriptive


def test_b_c2_tokens_and_the_exactness_condition(tmp_path):
    s = setup()
    (out,) = run_p2(s, {"B-C2"}, tmp_path)
    rec = json.loads(out.run_file.read_text())
    m = rec["metrics"]
    assert m["tokens_n"] == 16 and m["counter_exact"] == 0.0 and m["reduction_vs_top8_full"] > 0.5
    assert m["empty_tokens_max"] <= 30 and m["n_empty"] == 8
    assert (
        out.status == "FAIL" and "FAILED counter_exact=0" in out.detail
    )  # an estimate never gates
    s2 = setup(token_counter=lme_eval.est_counter, token_counter_exact=True)
    assert run_p2(s2, {"B-C2"}, tmp_path)[0].status == "PASS"


def test_token_sample_is_fixed_and_capped_at_300():
    items = [lme_eval.Item(f"q{i}", "x", "answerable", frozenset(), frozenset(),
                           lme_eval.Retrieved([], [], [], "t")) for i in range(500)]  # fmt: skip
    a, b = lme_eval.token_sample(items), lme_eval.token_sample(list(reversed(items)))
    assert len(a) == 300 and [i.qid for i in a] == [i.qid for i in b]


def test_b_r6_p2_form_checks_every_response(tmp_path):
    (out,) = run_p2(setup(), {"B-R6"}, tmp_path)
    assert out.status == "PASS", out.detail
    for text, abstained, needle in [
        ("0 results · personal\nextra", True, "not one abstention line"),
        (
            "2 results (of 30 checked) · personal · uncalibrated (P1 build)\nc1 · note · 2023-01-01 · t · .5",
            False,
            "uncalibrated",
        ),
        ("1 results (of 30 checked) · personal · calibrated\nnot a row", False, "row format"),
        (
            "1 results (of 30 checked) · personal · calibrated\nc1 · note · 2023-01-01 · "
            + "w " * 13
            + "· .5",
            False,
            "12 words",
        ),
        ("no relevant memories · personal " + "x" * 120, True, "est. tokens"),
        ("no results · personal", True, "not one abstention line"),  # the P1 line, not P2's
        (
            "1 results (of 30 checked) · personal · calibrated\nc1 · note · 2023-01-01 · t · 1.5",
            False,
            "outside",
        ),
    ]:
        assert any(needle in v for v in b_p2_cost.response_violations(text, abstained)), text
    ok = "2 results (of 30 checked) · personal · calibrated\nc1 · note · 2023-01-01 · t · about: A, B · .5\nc2 · fact · - · t · .25"
    assert b_p2_cost.response_violations(ok, False) == []
    for line in ("no relevant memories (best .22) · personal", "no relevant memories · work"):
        assert b_p2_cost.response_violations(line, True) == [], line
    bad = setup(backend=FakeLme(tag="uncalibrated (P1 build)"))
    (out,) = run_p2(bad, {"B-R6"}, tmp_path)
    assert out.status == "FAIL" and "uncalibrated" in out.detail


def fake_server(base):
    state = {"n": 0}

    def call(_q):
        state["n"] += 1
        t = base * (3 if state["n"] == 1 else 1)
        return {"fts": 1.0, "vec": 2.0, "fusion": 0.1, "rerank": t, "total": t + 3.1}

    return call


def test_latency_protocol_cold_warm_four_clients_and_aggregate(tmp_path):
    qs = [f"q{i}" for i in range(1 + latency.WARMUP + latency.N_QUERIES)]
    run = latency.run_process(fake_server(100.0), qs)
    assert run["cold_ms"] == pytest.approx(303.1) and run["n"] == 300
    assert run["one_client"]["total"]["p95"] == pytest.approx(103.1)
    assert run["four_clients"]["total"]["p95"] > 0 and "rerank" in run["one_client"]
    with pytest.raises(ValueError):
        latency.run_process(fake_server(1.0), qs[:50])
    runs = [run] * 3
    agg = latency.aggregate(runs)
    assert agg["warm_p95_ms_median"] == pytest.approx(103.1) and agg["n_runs"] == 3
    with pytest.raises(ValueError, match="3 independent"):
        latency.aggregate(runs[:2])
    paths = []
    for i, r in enumerate(runs):
        p = tmp_path / f"lat{i}.json"
        p.write_text(json.dumps(r))
        paths.append(p)
    (out,) = run_p2(setup(latency_runs=paths), {"B-C1"}, tmp_path / "r")
    assert out.status == "PASS", out.detail
    (out,) = run_p2(setup(latency_runs=paths[:2]), {"B-C1"}, tmp_path / "r")
    assert out.status == "UNAVAILABLE"
    slow = dict(run, one_client={**run["one_client"], "total": {"median": 1, "p95": 1600.0}})
    p = tmp_path / "slow.json"
    p.write_text(json.dumps(slow))
    (out,) = run_p2(setup(latency_runs=[p] * 3), {"B-C1"}, tmp_path / "r")
    assert out.status == "FAIL" and "warm_p95_ms_median=1600" in out.detail


def test_latency_regression_rule_needs_both_conditions():
    assert latency.regression([130, 135, 140], [100, 105, 110])
    assert not latency.regression([130, 100, 140], [100, 105, 110])  # lowest new <= highest base
    assert not latency.regression([115, 118, 120], [100, 105, 110])  # median not > 1.2x


def test_p2_gate_is_unavailable_without_setup_never_a_pass(tmp_path):
    out = gates.run_phase(
        "P2", factory_with(), results_dir=None, list_tools=list_tools, split="tune"
    )
    by = {o.id: o for o in out}
    assert [o.id for o in out] == ["B-R1", "B-R2", "B-R5", "B-R6", "B-C1", "B-C2"]
    assert all(by[i].status == "UNAVAILABLE" for i in ("B-R1", "B-R2", "B-R6", "B-C1", "B-C2"))
    assert by["B-R5"].status == "PASS"  # T0 fixture cases run with the P1 code


def test_gates_yaml_p2_rows_are_the_spec_thresholds_and_unfrozen():
    rows = {(s.id, s.metric): s for s in gates.load_gates()["P2"]}
    want = {("B-R1", "recall_any5_R"): (">=", 0.93), ("B-R1", "coverage_any"): (">=", 0.85),
            ("B-R1", "far"): ("<=", 0.05), ("B-R1", "delivered_precision"): (">=", 0.70),
            ("B-R1", "ira"): ("<=", 0.5), ("B-R2", "abstention_recall"): (">=", 0.85),
            ("B-R2", "far_drift"): ("<=", 0.03), ("B-C1", "warm_p95_ms_median"): ("<=", 1500),
            ("B-C2", "tokens_mean"): ("<=", 250), ("B-C2", "tokens_p95"): ("<=", 600),
            ("B-C2", "empty_tokens_max"): ("<=", 30), ("B-C2", "reduction_vs_top8_full"): (">=", 0.80)}  # fmt: skip
    for k, (op, thr) in want.items():
        assert (rows[k].op, rows[k].threshold) == (op, thr), k
    assert all(r.frozen_at is None for r in rows.values())


@pytest.fixture
def sandbox(tmp_path):
    y = tmp_path / "gates.yaml"
    shutil.copy(YAML, y)
    return y, tmp_path / "ledger.jsonl"


def test_freeze_stamps_test_rows_once_and_logs(sandbox):
    y, led = sandbox
    with pytest.raises(freeze.Refused, match="not frozen"):
        freeze.check_test_look_allowed("P2", y, led)
    changes = freeze.freeze("P2", who="t", reason="tune looks fine", config_hash="h", gates_path=y,
                            ledger=led)  # fmt: skip
    assert changes == []
    rows = gates.load_gates(y)["P2"]
    assert all((r.frozen_at is not None) == (r.split == "test") for r in rows)
    assert {r.threshold for r in rows if r.id == "B-R1" and r.metric == "far"} == {0.05}
    freeze.check_test_look_allowed("P2", y, led)
    assert json.loads(led.read_text())["kind"] == "freeze"
    with pytest.raises(freeze.Refused, match="already frozen"):
        freeze.freeze("P2", who="t", reason="again", config_hash="h", gates_path=y, ledger=led)


def test_freeze_override_once_with_reason_and_hard_floors(sandbox):
    y, led = sandbox
    with pytest.raises(freeze.Refused, match="written reason"):
        freeze.freeze("P2", who="t", reason=" ", config_hash="h", gates_path=y, ledger=led)
    with pytest.raises(freeze.Refused, match="hard floor"):
        freeze.freeze("P2", who="t", reason="r", config_hash="h", gates_path=y, ledger=led,
                      overrides={("B-R1", "delivered_precision"): 0.55})  # fmt: skip
    with pytest.raises(freeze.Refused, match="hard floor"):
        freeze.freeze("P2", who="t", reason="r", config_hash="h", gates_path=y, ledger=led,
                      overrides={("B-R1", "ira"): 1.0})  # fmt: skip
    with pytest.raises(freeze.Refused, match="no such"):
        freeze.freeze("P2", who="t", reason="r", config_hash="h", gates_path=y, ledger=led,
                      overrides={("B-R1", "nope"): 1.0})  # fmt: skip
    assert gates.load_gates(y)["P2"][0].frozen_at is None  # a refused freeze wrote nothing
    ch = freeze.freeze("P2", who="t", reason="tune FAR 0.04", config_hash="h", gates_path=y,
                       ledger=led, overrides={("B-R1", "delivered_precision"): 0.65})  # fmt: skip
    assert ch == ["B-R1.delivered_precision: 0.70 -> 0.65"]
    by = {(r.id, r.metric): r for r in gates.load_gates(y)["P2"]}
    assert by[("B-R1", "delivered_precision")].threshold == 0.65


def test_three_test_looks_then_refused_and_no_threshold_change_after_a_look(sandbox):
    y, led = sandbox
    freeze.freeze("P2", who="t", reason="r", config_hash="h", gates_path=y, ledger=led)
    for k in range(3):
        freeze.check_test_look_allowed("P2", y, led)
        records.append_ledger({"who": "t", "why": f"look {k}", "phase": "P2", "split": "test",
                               "config_hash": "h"}, led)  # fmt: skip
    with pytest.raises(freeze.Refused, match="3 test looks"):
        freeze.check_test_look_allowed("P2", y, led)
    y2 = y.with_name("g2.yaml")
    shutil.copy(YAML, y2)
    with pytest.raises(freeze.Refused, match="never change"):
        freeze.freeze("P2", who="t", reason="late", config_hash="h", gates_path=y2, ledger=led)
    assert freeze.test_looks("P1", led) == 0


def test_cli_test_look_requires_freeze_ledger_reason_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "GATES_YAML", YAML)  # unfrozen
    r = CliRunner().invoke(app, ["gate", "P2", "--split", "test", "--results-dir", str(tmp_path)])
    assert r.exit_code == 2 and "needs --ledger" in r.output
    r = CliRunner().invoke(app, ["gate", "P2", "--split", "test", "--ledger", "why",
                                 "--results-dir", str(tmp_path)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and "not frozen" in r.output


def test_cli_tune_run_is_labelled_informational_and_unavailable_without_data(tmp_path):
    r = CliRunner().invoke(app, ["gate", "P2", "--results-dir", str(tmp_path)])
    assert r.exit_code == 1
    assert "UNAVAILABLE B-R1" in r.output and "informational, not a look" in r.output
    r = CliRunner().invoke(app, ["gate", "P2", "--split", "dev"])
    assert r.exit_code != 0
