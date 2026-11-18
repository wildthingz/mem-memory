"""MEM2-156: the B-W3 miss / chain / B-R5 leak diagnosis (`bench.synth.w3_diagnose`) on stub data.
Stub-model numbers are smoke tests only; the real counts come from the host run
(deliverables/runbooks/p3-gate.md, "B-W3 margin and B-R5 leakage")."""

import json

from typer.testing import CliRunner

from bench.cli import app
from bench.synth import w3_diagnose as W
from mem import confmodel

MODEL = confmodel.ConfModel.build(["rerank", "cosine"], [1.0, 0.0], 0.0, 2, 1.0)
FIT = {"theta_conf": 0.0, "conf_model": MODEL.to_dict()}


def cand(id_, z, counterpart=False):
    return {"id": id_, "score": z, "same_slot": counterpart, "counterpart": counterpart,
            "features": {"rerank": z, "cosine": 0.0}}  # fmt: skip


def rec(cands, op=1, sub="late"):
    return {"seed": 1, "op": op, "kind": "update", "subtype": sub, "candidates": cands}


def test_each_cause_is_named():
    cases = {
        "surfaced": [cand("m1", 3.0, True), cand("m2", 2.5)],
        "not_picked": [cand("m1", 3.0), cand("m2", 2.5)],
        "below_theta": [cand("m1", 3.0), cand("m9", -0.5, True)],
        "cut_delta": [cand("m1", 3.0), cand("m9", 1.5, True)],  # >= theta, > delta under the best
        "cut_top_k": [cand("m1", 3.0), cand("m2", 3.0), cand("m9", 3.0, True)],  # ties, id order
    }
    got = {k: W.classify_op(rec(v), MODEL, 0.0) for k, v in cases.items()}
    assert got == {k: k for k in cases}


def test_classify_counts_per_subtype():
    recs = [rec([cand("m1", 3.0, True)], 1, "late"), rec([cand("m1", 3.0)], 2, "late"),
            rec([cand("m1", 3.0, True)], 3, "future")]  # fmt: skip
    c = W.classify_records(recs, FIT)
    assert c["by_subtype"]["late"] == {"surfaced": 1, "not_picked": 1}
    assert c["by_subtype"]["future"] == {"surfaced": 1}
    assert c["total"] == {"surfaced": 2, "not_picked": 1}


def test_join_leaks_confirms_and_rejects():
    slot_of = {10: "s1", 11: "s1", 20: "s2"}
    missed = {"s1": [11]}
    ok = W.join_leaks([{"id": "q1", "slot_ops": [10]}], missed, slot_of)
    assert ok["hypothesis_holds"] and ok["explained_by_miss"] == 1
    no = W.join_leaks([{"id": "q1", "slot_ops": [10]}, {"id": "q2", "slot_ops": [20]}], missed,
                      slot_of)  # fmt: skip
    assert not no["hypothesis_holds"] and no["unexplained"] == [{"id": "q2", "missed_ops": []}]
    assert not W.join_leaks([], missed, slot_of)["hypothesis_holds"]  # nothing leaked: not shown


def _w3_leak_run(tmp_path, force_all):
    """Seed 1 S on the real store with stub models; `force_all` surfaces every counterpart."""
    from bench.adapters.p3_ours import P3Backend
    from bench.suites import b_r5_p3, b_w123
    from bench.suites.base import Context
    from bench.synth import replay as R
    from bench.synth.run import SynSetup
    from mem.testing import HashEmbedder, OverlapReranker
    from tests.bench_t0.fake_serve import list_tools
    from tests.bench_t0.test_p3_real import stub_gate

    orig = R.Replay._act

    def forced(self, op, log):
        if force_all and log.status == "saved" and op["kind"] in ("update", "dup_paraphrase"):
            cp = self.mid.get(op["gold"]["counterpart"])
            if cp and cp not in log.conflicts:
                log.conflicts.append(cp)
        return orig(self, op, log)

    R.Replay._act = forced
    b = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()))
    b.gate = stub_gate(b.cfg)
    try:
        ctx = Context(tmp_path, b.factory, list_tools, {"syn": SynSetup((1,), "S", "fixture")})
        return b_w123.run_b_w3(ctx), b_r5_p3.run_b_r5(ctx)
    finally:
        R.Replay._act = orig
        b.close()


def test_perfect_surfacing_leaves_no_chain_error_and_no_unexplained_leak(tmp_path):
    """With every counterpart surfaced the chain is gold: a chain error then is its own bug."""
    w3, r5 = _w3_leak_run(tmp_path, force_all=True)
    d = w3.extra["diagnosis"]["1"]
    assert w3.metrics["chain_correctness"] == 1.0 and d["chain"] == []
    assert d["missed_slots"] == {}
    assert "leaked_queries" in r5.extra


def test_missed_updates_are_attributed_to_the_chain_not_to_a_chain_bug(tmp_path):
    w3, r5 = _w3_leak_run(tmp_path, force_all=False)
    d = w3.extra["diagnosis"]["1"]
    assert d["chain"], "the stub reranker misses updates: some chain must be wrong"
    assert {r["cause"] for r in d["chain"]} <= {"missed_link", "unsaved_update"}
    assert not any(r["cause"] == "surfaced_all" for r in d["chain"])
    from bench.synth.generator import generate

    slots = {int(k): v["slot"] for k, v in generate(1, "S").expected["memories"].items()}
    j = W.join_leaks(r5.extra["leaked_queries"]["1"], d["missed_slots"], slots)
    assert j["unexplained"] == []  # every leak sits in a slot with a missed update


def test_cli_runs_on_result_files(tmp_path):
    w3, r5 = _w3_leak_run(tmp_path / "run", force_all=False)
    p3, p5 = tmp_path / "w3.json", tmp_path / "r5.json"
    p3.write_text(json.dumps({"extra": w3.extra}), encoding="utf-8")
    p5.write_text(json.dumps({"extra": r5.extra}), encoding="utf-8")
    r = CliRunner().invoke(app, ["syn-w3-diagnose", "--w3", str(p3), "--r5", str(p5),
                                 "--size", "S"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "chain failures seed 1" in r.output


# ---- per-miss rows and the per-seed gap (host round 2) ------------------------------------------


def test_misses_carry_scores_and_contributions():
    """Every miss names its cause, the counterpart's score against theta and the list's best, its
    rank, and the per-feature contribution gap (best minus counterpart), so the host paste shows
    WHY a normal update scored under theta, not only that it did."""
    recs = [rec([cand("m1", 3.0), cand("m9", -0.5, True)], 1, "normal"),  # below theta
            rec([cand("m1", 3.0, True)], 2, "late")]  # fmt: skip
    c = W.classify_records(recs, FIT)
    assert c["fit"] == {"theta_conf": 0.0, "delta": 1.0, "top_k": 2}
    (m,) = c["misses"]
    assert (m["seed"], m["op"], m["subtype"], m["cause"]) == (1, 1, "normal", "below_theta")
    assert (m["score"], m["best"], m["rank"], m["n"]) == (-0.5, 3.0, 1, 2)
    assert m["counterpart"] == {"rerank": -0.5, "cosine": 0.0}
    assert m["best_candidate"] == {"rerank": 3.0, "cosine": 0.0}
    assert m["gap_by_feature"] == {"rerank": 3.5, "cosine": 0.0}  # w * (x_best - x_cp)


def test_not_picked_miss_has_no_score_but_a_row():
    recs = [rec([cand("m1", 3.0), cand("m2", 2.5)], 7, "future")]
    (m,) = W.classify_records(recs, FIT)["misses"]
    assert m["cause"] == "not_picked" and m["score"] is None and m["counterpart"] is None
    assert m["best"] == 3.0 and m["n"] == 2


def test_per_seed_in_sample_surfacing_counts_updates_only():
    recs = [rec([cand("m1", 3.0, True)], 1), rec([cand("m1", 3.0)], 2),
            {**rec([cand("m1", 3.0)], 3), "kind": "dup_paraphrase", "subtype": None}]  # fmt: skip
    recs[2]["seed"] = 2
    c = W.classify_records(recs, FIT)
    assert c["per_seed"] == {"1": {"updates": 2, "surfaced": 1, "surfacing": 0.5},
                             "2": {"updates": 0, "surfaced": 0, "surfacing": None}}  # fmt: skip


def test_gate_gap_joins_in_sample_to_the_gate_per_seed():
    """The honest comparison: in-sample surfacing on the same seed's updates against the gate's
    live B-W3 surfacing. A positive gap is what a miss costs later ops (compounding)."""
    recs = [rec([cand("m1", 3.0, True)], i) for i in range(1, 5)]
    w3x = {"per_seed": {"1": {"surfacing_recall": 0.75}},
           "diagnosis": {"1": {"chain": [{"cause": "missed_link", "missed_ops": [3, 5]},
                                         {"cause": "missed_link", "missed_ops": [8]}],
                               "missed_slots": {}, "subtype_misses": {}}}}  # fmt: skip
    rep = W.report(recs, FIT, w3x, None)
    assert rep["gate_gap"] == {"1": {"in_sample": 1.0, "gate": 0.75, "gap": 0.25}}
    assert rep["multi_miss_slots"] == {"1": 1}
    text = W.format_report(rep)
    assert "seed 1: in-sample 1.000 (4/4 updates), gate 0.750, gap +0.250" in text
    assert "slots with 2+ misses" in text


def test_format_report_lists_each_miss():
    recs = [rec([cand("m1", 3.0), cand("m9", -0.5, True)], 1, "normal")]
    text = W.format_report(W.report(recs, FIT, None, None))
    assert "misses (counterpart vs the list's best; theta 0, delta 1, top_k 2)" in text
    assert "rerank +3.50" in text  # the feature that opens the gap, best minus counterpart
