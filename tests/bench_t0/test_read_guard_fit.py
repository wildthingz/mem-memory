"""The θ_read fit (`bench syn-fit-read-guard`, ADR 0018): the rule, the nested CV, the CLI on the
stub models, `gate-add-read-guard`, and `mem calibrate --objective p3` on SYN v3 records."""

import json
from dataclasses import asdict

import pytest
from typer.testing import CliRunner

from bench.cli import app
from bench.synth import readguard_fit as RF
from mem.confmodel import ConfModel
from mem.gate import Gate
from mem.testing import stub_conf_model

CM_FP = ConfModel.from_dict(stub_conf_model()).fingerprint()


def rec(seed, qid, rows, pairs, type="current", t="2026-10-22T09:00:00Z"):
    """rows: (id, slot, valid, gold[, dup]); pairs: (later, earlier, score, label, sub)."""
    return {"seed": seed, "qid": f"syn{seed}:{qid}", "type": type, "t": t,
            "rows": [{"id": r[0], "op": 0, "slot": r[1], "valid": r[2], "gold": r[3],
                      "dup": r[4] if len(r) > 4 else False} for r in rows],
            "pairs": [{"later": p[0], "earlier": p[1], "score": p[2], "label": p[3], "sub": p[4]}
                      for p in pairs]}  # fmt: skip


def world(seeds=(1, 2, 3), stale_score=4.0, co_score=1.0):
    """Per seed: one missed update (new m2 over stale m1), one co-valid cross-slot pair, one clean
    current read, one as-of read where the stale row sits first."""
    out = []
    for s in seeds:
        out.append(
            rec(
                s,
                "cur:a",
                [("m1", True, False, False), ("m2", True, True, True)],
                [("m2", "m1", stale_score + 0.1 * s, "stale", "same_slot")],
            )
        )
        out.append(
            rec(
                s,
                "cur:b",
                [("m3", True, True, True), ("m4", False, True, False)],
                [("m4", "m3", co_score - 0.1 * s, "co_valid", "cross_slot")],
            )
        )
        out.append(rec(s, "cur:c", [("m5", True, True, True)], []))
        out.append(
            rec(
                s,
                "asof:a",
                [("m1", True, False, False), ("m2", True, True, True)],
                [("m2", "m1", stale_score, "stale", "same_slot")],
                type="as_of",
            )
        )
    return out


def test_choose_theta_sits_just_above_the_best_co_valid_pair_never_below_theta_conf():
    recs = world()
    theta, how = RF.choose_theta(recs, theta_conf=0.5)
    assert how["max_co_valid"] == pytest.approx(0.9) and how["min_stale_above"] == pytest.approx(
        4.0
    )
    assert theta == pytest.approx((0.9 + 4.0) / 2) and how["note"] == ""
    assert RF.choose_theta(recs, 0.5, margin=1.0)[0] == pytest.approx(theta + 1.0)
    assert RF.choose_theta(recs, theta_conf=10.0)[0] == 10.0  # strict: never below theta_conf
    # no stale pair above the best co-valid: the cut hides nothing stale, and says so
    theta, how = RF.choose_theta(world(stale_score=0.0), 0.0)
    assert theta > how["max_co_valid"] and "hides nothing stale" in how["note"]


def test_evaluate_counts_wrongly_hidden_leakage_recall_and_as_of_before_and_after():
    recs = world((1,))
    ev = RF.evaluate(recs, theta=2.0)
    assert ev["wrongly_hidden"] == 0 and (ev["stale_hidden"], ev["stale_total"]) == (2, 2)
    assert ev["leakage_before"] == pytest.approx(1 / 3) and ev["leakage_after"] == 0.0
    assert ev["current_recall_at_1_before"] == pytest.approx(2 / 3)
    assert ev["current_recall_at_1_after"] == 1.0
    assert ev["as_of_accuracy_before"] == 0.0 and ev["as_of_accuracy_after"] == 1.0
    low = RF.evaluate(recs, theta=0.0)  # everything hides: the co-valid m3 goes too
    assert low["wrongly_hidden"] == 1 and low["current_recall_at_1_after"] == pytest.approx(2 / 3)
    high = RF.evaluate(recs, theta=9.0)
    assert high["leakage_after"] == high["leakage_before"] and high["stale_hidden"] == 0
    assert high["residual_leaks"][0]["cause"] == "below_theta"


def test_evaluate_names_the_cause_of_a_residual_leak():
    dup = rec(
        1,
        "cur:d",
        [("m7", True, True, True), ("m8", True, False, False, True)],
        [("m8", "m7", 5.0, "co_valid", "paraphrase")],
    )  # the paraphrase is the stale row
    alone = rec(1, "cur:e", [("m9", True, False, False)], [])
    ev = RF.evaluate([dup, alone], theta=9.0)
    assert [r["cause"] for r in ev["residual_leaks"]] == ["paraphrase", "no_partner"]


def test_nested_fit_is_feasible_when_every_outer_fold_hides_zero_co_valid_and_leaks_zero():
    fit = RF.fit_read_guard(world(), 0.5, CM_FP)
    assert fit["feasible"] and fit["nested"]["worst"]["wrongly_hidden"] == 0
    assert fit["nested"]["worst"]["leakage_after"] == 0.0
    assert [f["seed"] for f in fit["nested"]["outer_folds"]] == [1, 2, 3]
    assert all(f["selected_on"] == [s for s in (1, 2, 3) if s != f["seed"]]
               for f in fit["nested"]["outer_folds"])  # fmt: skip
    assert fit["chosen"]["margin"] == max(RF.MARGINS)  # room for the largest margin: the strictest
    assert fit["read_guard"]["theta_read"] == fit["theta_read"] > 0.9
    assert fit["read_guard"]["conf_model_fingerprint"] == CM_FP
    Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5, conf_model=stub_conf_model(),
         read_guard=fit["read_guard"])  # fmt: skip
    text = RF.format_report(fit)
    assert "PASS" in text and "theta_read" in text


def test_nested_fit_reports_infeasible_when_a_held_out_co_valid_pair_outscores_the_stale_ones():
    recs = world()
    # seed 3's co-valid pair scores like a stale one: a cut fitted on seeds 1-2 hides it
    for r in recs:
        if r["seed"] == 3 and r["qid"].endswith("cur:b"):
            r["pairs"][0]["score"] = 4.5
    fit = RF.fit_read_guard(recs, 0.5, CM_FP)
    assert not fit["feasible"]
    fold3 = next(f for f in fit["nested"]["outer_folds"] if f["seed"] == 3)
    assert fold3["wrongly_hidden"] == 1
    assert "MISS" in RF.format_report(fit)
    # a cut that cannot reach leakage 0 is infeasible too, with the numbers
    fit2 = RF.fit_read_guard(world(stale_score=0.5, co_score=3.0), 0.5, CM_FP)
    assert not fit2["feasible"] and fit2["nested"]["worst"]["leakage_after"] > 0
    assert fit2["nested"]["worst"]["wrongly_hidden"] == 0


def test_fit_refuses_held_out_seeds_and_too_few_seeds():
    with pytest.raises(ValueError, match="not dev seeds"):
        RF.fit_read_guard(world((1, 2, 101)), 0.5, CM_FP)
    with pytest.raises(ValueError, match="three seeds"):
        RF.fit_read_guard(world((1, 2)), 0.5, CM_FP)


# ---- CLI --------------------------------------------------------------------------------------


def _gate_file(tmp_path, theta_read=None):
    from bench.adapters.lme_ours import load_config
    from tests.bench_t0.test_p3_real import stub_gate

    g = tmp_path / "gate.json"
    g.write_text(json.dumps(asdict(stub_gate(load_config(None, "hash"), theta_read))))
    return g


def test_cli_syn_fit_read_guard_runs_on_the_stub_models_and_writes_records_fit_and_report(tmp_path):
    """SYN-S seeds 1-3 under the stub gate: the fit runs end to end and is honest about the stub
    scorer (the raw overlap logit cannot separate stale from co-valid pairs there: infeasible,
    exit 1). The records carry gold validity and the pairs' labels."""
    g = _gate_file(tmp_path)  # no read_guard yet: the fit completes the gate
    out, recs = tmp_path / "fit.json", tmp_path / "recs.jsonl"
    r = CliRunner().invoke(app, ["syn-fit-read-guard", "--gate", str(g), "--out", str(out),
                                 "--records-out", str(recs), "--seeds", "1,2,3", "--size", "S",
                                 "--embedder", "hash"])  # fmt: skip
    assert r.exit_code in (0, 1), r.output
    assert "read guard nested CV" in r.output and out.exists() and out.with_suffix(".txt").exists()
    fit = json.loads(out.read_text())
    assert fit["seeds"] == [1, 2, 3] and fit["pairs"] > 0 and fit["theta_conf"] == 0.5
    assert fit["read_guard"]["conf_model_fingerprint"] == CM_FP
    assert fit["gate"]["conf_model"] == CM_FP
    rows = [json.loads(ln) for ln in recs.read_text().splitlines()]
    assert {x["type"] for x in rows} == {"current", "as_of", "missing_slot"}
    assert all(p["label"] in ("stale", "co_valid", "other") for x in rows for p in x["pairs"])
    assert any(p["label"] == "stale" and p["sub"] == "same_slot" for x in rows for p in x["pairs"])
    assert any(p["label"] == "co_valid" for x in rows for p in x["pairs"])
    assert (r.exit_code == 0) == fit["feasible"]
    # refit from the records: the same fit without a replay
    r2 = CliRunner().invoke(app, ["syn-fit-read-guard", "--gate", str(g), "--out",
                                  str(tmp_path / "fit2.json"), "--from-records", str(recs),
                                  "--embedder", "hash"])  # fmt: skip
    assert r2.exit_code == r.exit_code
    assert json.loads((tmp_path / "fit2.json").read_text())["theta_read"] == fit["theta_read"]


def test_cli_syn_fit_read_guard_refuses_held_out_seeds_and_a_gate_without_theta_conf(tmp_path):
    g = _gate_file(tmp_path)
    r = CliRunner().invoke(app, ["syn-fit-read-guard", "--gate", str(g), "--out",
                                 str(tmp_path / "f.json"), "--seeds", "101", "--embedder", "hash"])  # fmt: skip
    assert r.exit_code == 2 and "not dev seeds" in r.output
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({**json.loads(g.read_text()), "theta_conf": None,
                                "conf_model": None, "read_guard": None}))  # fmt: skip
    r = CliRunner().invoke(app, ["syn-fit-read-guard", "--gate", str(bare), "--out",
                                 str(tmp_path / "f.json"), "--embedder", "hash"])  # fmt: skip
    assert r.exit_code == 2 and "theta_conf" in r.output


def test_cli_gate_add_read_guard_sets_only_the_guard_and_refuses_bad_fits(tmp_path):
    g = _gate_file(tmp_path)
    gate = json.loads(g.read_text())
    fit = RF.fit_read_guard(world(), 0.5, CM_FP)
    f = tmp_path / "fit.json"
    f.write_text(json.dumps(fit))
    o = tmp_path / "o.json"
    r = CliRunner().invoke(app, ["gate-add-read-guard", str(g), str(f), "--out", str(o)])
    assert r.exit_code == 0, r.output
    assert json.loads(o.read_text()) == {**gate, "read_guard": fit["read_guard"]}
    assert Gate(**json.loads(o.read_text())).guard.theta_read == fit["theta_read"]
    # infeasible: refused
    f.write_text(json.dumps({**fit, "feasible": False}))
    r = CliRunner().invoke(
        app, ["gate-add-read-guard", str(g), str(f), "--out", str(tmp_path / "x")]
    )
    assert r.exit_code == 2 and "infeasible" in r.output and not (tmp_path / "x").exists()
    # fitted against another conf_model: refused
    other = ConfModel.build(["rerank", "cosine"], [1.0, 1.0], 0.0, 5, 100.0).to_dict()
    f.write_text(json.dumps(RF.fit_read_guard(world(), 0.5, other["fingerprint"])))
    r = CliRunner().invoke(
        app, ["gate-add-read-guard", str(g), str(f), "--out", str(tmp_path / "y")]
    )
    assert r.exit_code == 2 and "another conf_model" in r.output
    # a gate without a conf_model: refused
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({**gate, "theta_conf": None, "conf_model": None}))
    f.write_text(json.dumps(fit))
    r = CliRunner().invoke(
        app, ["gate-add-read-guard", str(bare), str(f), "--out", str(tmp_path / "z")]
    )
    assert r.exit_code == 2 and "no conf_model" in r.output


def test_p3_backend_refuses_a_gate_without_read_guard_unless_the_switch_is_off(tmp_path):
    from bench.adapters.p3_ours import P3Backend
    from bench.suites.base import BackendUnavailable
    from mem.testing import HashEmbedder, OverlapReranker
    from tests.bench_t0.test_p3_real import stub_gate

    b = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()),
                  conflict_guard=True)  # fmt: skip
    try:
        b.gate = stub_gate(b.cfg, None)
        with pytest.raises(BackendUnavailable, match="read_guard"):
            b.open_server(tmp_path / "a.db")
        b.gate = stub_gate(b.cfg)
        b.open_server(tmp_path / "b.db")[0].close()
    finally:
        b.close()
    off = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()))  # the default
    try:
        off.gate = stub_gate(off.cfg, None)
        server, _ = off.open_server(tmp_path / "c.db")
        assert server.store.conflict_guard is False
        server.close()
    finally:
        off.close()


def test_calibrate_p3_objective_accepts_syn_v3_records_with_the_validity_fields(tmp_path):
    """MEM2-159 (3): the SYN v3 export (candidate validity on the conflict rows, MEM2-156) merged
    with the read pools still fits the P3 read gate; the conflict records carry the fields."""
    pools, conf = tmp_path / "pools.jsonl", tmp_path / "conf.jsonl"
    r = CliRunner().invoke(app, ["syn-score-dev", "--out", str(pools), "--conflicts-out",
                                 str(conf), "--seeds", "1,2,3", "--size", "S", "--embedder",
                                 "hash"])  # fmt: skip
    assert r.exit_code == 0, r.output
    recs = [json.loads(ln) for ln in conf.read_text().splitlines()]
    assert all("valid_from" in c and "valid_to" in c for x in recs for c in x["candidates"])
    merged = tmp_path / "merged.jsonl"
    r = CliRunner().invoke(app, ["merge-records", str(pools), "--out", str(merged)])
    assert r.exit_code == 0, r.output
    from mem.cli import app as mem_app

    out = tmp_path / "gate.json"
    r = CliRunner().invoke(mem_app, ["calibrate", "--records", str(merged), "--embedder", "hash",
                                     "--objective", "p3", "--out", str(out), "--fit-id", "v3"])  # fmt: skip
    assert r.exit_code == 0, r.output
    gate = json.loads(out.read_text())
    assert gate["read_model"]["fingerprint"] and gate["fit_id"] == "v3"
    assert gate.get("read_guard") is None  # added later by gate-add-read-guard, never by default
    assert "nested" in r.output
