"""List-everything-then-count mode (MEM2-193, ADR 0023): the detector, the layout, tau_list, the
cap, the fit (nested CV, no outer-fold leak) and the switch. T0, on the hash models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from mem import aggcue, config, listfit, listmode, read, write
from mem.calibrate import PoolItem, QueryPool
from mem.common import MemError
from mem.gate import Gate, write_gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker
from tests.test_config import GOLDEN_FP_ALL_LIST_MODE, GOLDEN_FP_ALL_OFF, all_off

# ---- the detector -------------------------------------------------------------------------

POSITIVE = [
    "How many books did I read this year?",
    "how many times did I go to the gym",
    "How often do I go hiking?",
    "How many days did I spend camping in total?",
    "What is the total amount I spent on gifts?",
    "In total, which cities have I visited?",
    "Which restaurants did I try last month?",
    "Which of the trips did I enjoy most?",
    "List the museums I visited.",
    "What are all the plants I bought?",
    "Can you count the movies I watched?",
    "What is the number of pets I own?",
    "How many weeks did I work on the project?",
    "What did I spend altogether on shoes?",
]
NEAR_MISS = [
    "Which restaurant did I go to last week?",
    "Which is bigger, my old or my new car?",
    "How many days ago did I buy the shoes?",
    "How many weeks have passed since I started running?",
    "How many months between the wedding and the move?",
    "Did I like the film at all?",
    "Did I listen to jazz yesterday?",
    "What is my account number?",
    "What country did I visit in May?",
    "Where did I park the car?",
    "How long was my flight?",
    "How much did I pay for the shoes?",
    "Is that all right with the plan?",
    "",
]


@pytest.mark.parametrize("q", POSITIVE)
def test_detector_positives(q):
    assert aggcue.is_aggregation(q), q


@pytest.mark.parametrize("q", NEAR_MISS)
def test_detector_near_misses(q):
    assert not aggcue.is_aggregation(q), q


def test_detector_names_the_cues_in_order():
    assert aggcue.cues("List all the trips: how many in total?") == [
        "list", "all", "how many", "in total",
    ]  # fmt: skip


# ---- the mode -------------------------------------------------------------------------------

DAYS = ["2024-03-01", "2024-03-09", "2024-04-02", "2024-04-20", "2024-05-11"]
Q = "How many hiking trips did I take"


def _store(
    tmp_path,
    *,
    list_mode: bool | None,
    tau: float = 0.5,
    platt=(1.0, 0.0),
    nrounds=1,
    retrieval=None,
    gate=True,
):
    cfg = dict(config.default_config().read)
    if list_mode is not None:
        cfg["list_mode"] = list_mode
    s = Store.open(tmp_path / "p.db", HashEmbedder(), reranker=OverlapReranker(), read=cfg,
                   now=lambda: datetime(2024, 6, 1, tzinfo=UTC), retrieval=retrieval)  # fmt: skip
    # five hiking sessions (one has two matching rounds), two off-topic ones
    for i, d in enumerate(DAYS):
        rounds = [("I did take many hiking trips with friends", "Nice, how many hiking trips")]
        if i == 1:
            rounds *= 2
        write.ingest_session(s, source_ref=f"s{i}", date=d, rounds=rounds * nrounds)
    write.ingest_session(s, source_ref="x1", date="2024-04-10",
                         rounds=[("I took my car to the garage", "Done")])  # fmt: skip
    write.ingest_session(s, source_ref="x2", date="2024-04-11",
                         rounds=[("Dinner was pasta", "Yum")])  # fmt: skip
    g = Gate(*platt, 0.5, 0.5, 0.0)
    if gate:
        g = Gate(*platt, 0.5, 0.5, 0.0, list_model=_lm(tau, platt))
    write_gate(s.conn, g)
    s.gate = g
    return s


def _sub(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def _folds(n=5):
    return {"outer_folds": [{"fold": i, "tau_list": 0.5} for i in range(n)]}


def _lm(tau, platt=(1.0, 0.0), cap=listmode.CAP_TOKENS):
    return listmode.ListModel.build(tau, cap, listmode.FLOOR, *platt, _folds()).to_dict()


def test_list_mode_groups_by_session_with_date_and_the_instruction_line(tmp_path):
    s = _store(tmp_path, list_mode=True)
    out = read.recall(s, query=Q)
    lines = out.text.splitlines()
    assert lines[0].startswith("6 results (of ") and "list mode" in lines[0]
    assert "5 sessions" in lines[0]
    assert lines[1] == listmode.INSTRUCTION
    heads = [ln for ln in lines[2:] if not ln.startswith("  ")]
    assert [h.split(" · ")[0] for h in heads] == DAYS  # one line per session, in date order
    assert sum(ln.startswith("  ") for ln in lines) == 6  # the row lines sit under their session
    assert lines[2].startswith("2024-03-01") and lines[3].startswith("  ")
    assert lines[4].startswith("2024-03-09") and lines[5].startswith("  ")
    assert lines[6].startswith("  ")  # the session with two matching rounds
    assert out.structured["list_mode"] is True and len(out.structured["rows"]) == 6
    s.close()


def test_list_mode_is_every_row_above_tau_not_the_top_k(tmp_path):
    s = _store(tmp_path, list_mode=True, nrounds=3, retrieval={**config.default_config().retrieval,
                                                                "per_node_cap": 9})  # fmt: skip
    n = len(read.recall(s, query=Q).structured["rows"])
    assert n > s.retrieval["limit"]  # more than the normal top-k
    s.close()


def test_tau_list_is_applied(tmp_path):
    hi = _store(_sub(tmp_path, "a"), list_mode=True, tau=0.999)
    out = read.recall(hi, query=Q)
    assert out.structured["abstained"] is True and out.text.startswith("no relevant memories")
    hi.close()
    lo = _store(_sub(tmp_path, "b"), list_mode=True, tau=0.001)
    rows = read.recall(lo, query=Q).structured["rows"]
    assert len(rows) > 6  # a low cut lets the off-topic rounds in
    lo.close()


def test_the_cap_is_respected_and_the_cut_is_said(tmp_path):
    s = _store(tmp_path, list_mode=True)
    out = read.recall(s, query=Q, budget=120)
    assert listmode.response_tokens(out.text) <= 120
    assert "more matches above tau_list not shown" in out.text
    assert 0 < len(out.structured["rows"]) < 6
    s.close()


def test_default_cap_is_the_models_and_the_normal_budget_is_unchanged(tmp_path):
    s = _store(tmp_path, list_mode=True)
    assert s.gate.lister.cap_tokens == listmode.CAP_TOKENS == 1000
    assert s.retrieval["budget"] == 600
    out = read.recall(s, query=Q)
    assert listmode.response_tokens(out.text) <= 1000
    s.close()


def test_non_aggregation_query_takes_the_normal_path(tmp_path):
    on = _store(_sub(tmp_path, "a"), list_mode=True)
    off = _store(_sub(tmp_path, "b"), list_mode=False)
    q = "Which hiking trip was in March"  # a single answer
    assert read.recall(on, query=q).text == read.recall(off, query=q).text
    on.close()
    off.close()


def test_a_gate_without_list_model_is_refused_when_the_mode_is_on(tmp_path):
    s = _store(tmp_path, list_mode=True, gate=False)
    with pytest.raises(MemError, match="list_model"):
        read.recall(s, query=Q)
    with pytest.raises(MemError, match="list_model"):
        read.recall(s, query="Where did I park")  # refused for every query, not only aggregations
    s.close()


# ---- the switch off ---------------------------------------------------------------------------


@pytest.mark.release_defaults
def test_switch_off_leaves_the_fingerprint_alone():
    """On by default since MEM2-261 (F2+F3 adopted, Round 9); off, the key is absent and `fp_all` is the
    one stores fitted without list mode carry."""
    assert config.default_config().list_mode is True
    cfg = all_off()
    assert "list_mode" not in cfg.read and cfg.list_mode is False
    assert cfg.fp_all() == GOLDEN_FP_ALL_OFF
    assert cfg.replace("read", list_mode=True).fp_all() == GOLDEN_FP_ALL_LIST_MODE


def test_unmarked_tests_run_on_the_round_8_switches():
    """conftest's `_round8_switches`: an unmarked T0 test sees every switch off in the default config."""
    cfg = config.default_config()
    assert cfg == all_off() and cfg.fp_all() == GOLDEN_FP_ALL_OFF
    assert not (cfg.list_mode or cfg.resolve_dates)


def test_switch_must_be_a_bool(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("read: {list_mode: 'yes'}\n")
    with pytest.raises(config.ConfigError, match="list_mode"):
        config.load(p)
    p.write_text("read: {list_mode: true}\n")
    assert config.load(p).list_mode is True


def test_switch_off_is_todays_output_even_with_a_list_model_in_the_gate(tmp_path):
    a = _store(_sub(tmp_path, "a"), list_mode=None, gate=False)
    b = _store(_sub(tmp_path, "b"), list_mode=False, gate=True)
    for q in (Q, "Where did I park the car", "List all the dinners I had"):
        x, y = read.recall(a, query=q), read.recall(b, query=q)
        assert x.text == y.text and x.structured == y.structured
        assert "list mode" not in x.text and listmode.INSTRUCTION not in x.text
    a.close()
    b.close()


def test_switch_off_is_the_pre_list_mode_output(tmp_path):
    """Against output recorded by the code before list mode existed (08018ea^, same store and
    gate, `tests/fixtures/golden/list_mode_off.json`), not against the new code with the mode off."""
    golden = json.loads((Path(__file__).parent / "fixtures/golden/list_mode_off.json").read_text())
    s = _store(tmp_path, list_mode=False, gate=True)  # off, with a list model in the gate
    try:
        for q, (text, structured) in golden.items():
            r = read.recall(s, query=q)
            assert r.text == text and r.structured == structured, q
    finally:
        s.close()


# ---- the gate's list_model ----------------------------------------------------------------------


def test_list_model_roundtrip_fingerprint_and_tamper():
    d = _lm(0.4)
    assert listmode.ListModel.from_dict(d).tau_list == 0.4
    bad = {**d, "tau_list": 0.41}
    with pytest.raises(ValueError, match="fingerprint"):
        listmode.ListModel.from_dict(bad)
    with pytest.raises(ValueError, match="tau_list is required"):
        listmode.ListModel.from_dict({**d, "tau_list": None})
    with pytest.raises(ValueError, match="outer folds"):
        listmode.ListModel.build(0.4, 1000, 0.9, 1.0, 0.0, {})


def test_gate_refuses_a_list_model_fitted_against_another_platt():
    with pytest.raises(ValueError, match="Platt"):
        Gate(2.0, 0.0, 0.5, 0.5, 0.0, list_model=_lm(0.4, (1.0, 0.0)))
    assert Gate(1.0, 0.0, 0.5, 0.5, 0.0, list_model=_lm(0.4)).lister.tau_list == 0.4


def test_cost_model_never_undercounts_a_rendered_response(tmp_path):
    s = _store(tmp_path, list_mode=True)
    out = read.recall(s, query=Q)
    n_rows, n_groups = len(out.structured["rows"]), 5
    assert listmode.tokens_for(n_rows, n_groups) >= listmode.response_tokens(out.text)
    s.close()


# ---- the fit ------------------------------------------------------------------------------------


def _pool(qid, i, hi=0.95, noise=3):
    """An aggregation question with 3 gold sessions (z high) and `noise` distractors (z low)."""
    items = [PoolItem(f"s{i}{j}#0", 3.0 + 0.1 * j, 1) for j in range(3)]
    items += [PoolItem(f"n{i}{j}#0", -1.0 - 0.3 * j, 0) for j in range(noise)]
    return QueryPool(qid, True, items, "multi-session")


def _data(n=30):
    pools = [_pool(f"q{i}", i) for i in range(n)]
    queries = {p.qid: "How many trips did I take in total?" for p in pools}
    return pools, queries


def test_only_aggregation_answerable_pools_with_gold_are_fitted():
    pools, queries = _data(12)
    queries["q0"] = "Where did I park?"
    pools[1] = QueryPool("q1", False, pools[1].items)
    pools[2] = QueryPool("q2", True, [PoolItem("a#0", 1.0, 0)])
    assert {q.qid for q in listfit.eligible(pools, queries, (1.0, 0.0))} == {
        f"q{i}" for i in range(3, 12)
    }


def test_fit_cuts_between_gold_and_distractors_and_reports_the_outer_folds():
    pools, queries = _data(30)
    lm, _ = listfit.fit_list_model(pools, queries, (1.0, 0.0))
    nested = lm["nested"]
    assert len(nested["outer_folds"]) == 5 and nested["feasible"] is True
    assert nested["mean_heldout_coverage"] == 1.0
    # gold p = σ(3.0..3.2) ≈ .95-.96, distractors σ(-1..) ≈ .27 and below: tau lands among the gold
    assert 0.9 < lm["tau_list"] < 1.0
    assert all(f["heldout"]["tokens_p95"] <= listmode.CAP_TOKENS for f in nested["outer_folds"])
    assert listmode.ListModel.from_dict(lm).fingerprint() == lm["fingerprint"]
    assert nested["final"]["flat_cv_optimistic"] is True


def test_the_token_cap_binds_the_fit():
    pools, queries = _data(30)
    small = listmode.tokens_for(1, 1) + 2  # room for one row only
    _, rep = listfit.fit_list_model(pools, queries, (1.0, 0.0), cap=small)
    assert rep["nested"]["mean_heldout_coverage"] < 0.5  # one of three gold sessions fits
    assert rep["nested"]["feasible"] is False


def test_nested_cv_has_no_outer_fold_leak():
    """Planted: the held-out fold's pools are replaced by a trap (gold at the lowest p, a flood of
    high-p distractors) that would move the cut if it were seen. The fold's floor and tau do not
    move, and neither does any other fold's held-out score of its own data."""
    pools, queries = _data(40)
    qs = listfit.eligible(pools, queries, (1.0, 0.0))
    base = listfit.nested_cv(qs)
    for f in range(listfit.FOLDS):
        trapped = []
        for q in qs:
            if listfit.fold_of(q.qid, listfit.FOLDS) == f:
                trap = listfit.Query(q.qid, q.p[::-1].copy(), q.sess, q.gold)
                trapped.append(trap)
            else:
                trapped.append(q)
        got = listfit.nested_cv(trapped)
        mine, theirs = base["outer_folds"][f], got["outer_folds"][f]
        assert (mine["tau_list"], mine["floor"], mine["train_coverage"]) == (
            theirs["tau_list"], theirs["floor"], theirs["train_coverage"])  # fmt: skip
        assert mine["heldout"] != theirs["heldout"]  # the trap did change what it is scored on


def test_inner_cv_fits_on_its_training_part_only(monkeypatch):
    """The floor's inner CV: every inner tau is fitted on questions disjoint from those it is scored
    on, and the inner folds cover `qs` (a leak such as `tr = qs` fails here)."""
    pools, queries = _data(40)
    qs = listfit.eligible(pools, queries, (1.0, 0.0))
    fitted, scored, inside = [], [], []
    fit_tau, score = listfit.fit_tau, listfit.score

    def spy_fit(tr, floor, cap):
        fitted.append({q.qid for q in tr})
        inside.append(1)
        try:
            return fit_tau(tr, floor, cap)
        finally:
            inside.pop()

    def spy_score(te, tau, cap):
        if not inside:  # the held-out score, not fit_tau's own scoring of its training part
            scored.append({q.qid for q in te})
        return score(te, tau, cap)

    monkeypatch.setattr(listfit, "fit_tau", spy_fit)
    monkeypatch.setattr(listfit, "score", spy_score)
    listfit.choose_floor(qs, listmode.CAP_TOKENS)
    assert len(fitted) == len(scored) == len(listfit.FLOORS) * listfit.INNER
    every = {q.qid for q in qs}
    for tr, te in zip(fitted, scored, strict=True):
        assert tr and te and not tr & te and tr | te == every


def test_fold_assignment_is_by_question_id_alone():
    assert listfit.fold_of("q7", 5) == listfit.fold_of("q7", 5)
    a = [listfit.fold_of(f"q{i}", 5) for i in range(200)]
    assert set(a) == set(range(5)) and min(a.count(i) for i in range(5)) > 20


def test_too_few_questions_refuse():
    pools, queries = _data(6)
    with pytest.raises(ValueError, match="at least"):
        listfit.fit_list_model(pools, queries, (1.0, 0.0))


# ---- the report rows and the CLI ---------------------------------------------------------------


def test_b_c2_list_token_rows(tmp_path):
    from types import SimpleNamespace as NS

    from bench import lme_eval

    s = _store(tmp_path, list_mode=True)
    listed = read.recall(s, query=Q).text
    plain = read.recall(s, query="Where did I park the car").text
    s.close()
    items = [NS(kind="answerable", r=NS(text=t)) for t in (listed, plain)]
    items.append(NS(kind="abs", r=NS(text=listed)))  # abstention items are not B-C2's
    m = lme_eval.list_token_metrics(items, lme_eval.est_counter)
    assert m["list_tokens_n"] == 1.0 and m["list_cap"] == 1000.0
    assert m["list_tokens_p95"] == m["list_tokens_max"] == float(listmode.response_tokens(listed))
    off = lme_eval.list_token_metrics(items[1:2], lme_eval.est_counter)
    assert off["list_tokens_n"] == 0.0 and off["list_tokens_p95"] == 0.0


def test_gate_add_tau_list_writes_a_fingerprinted_list_model_and_refuses_infeasible(tmp_path):
    import json

    from typer.testing import CliRunner

    from bench.cli import app

    pools, queries = _data(30)
    lm, rep = listfit.fit_list_model(pools, queries, (1.0, 0.0))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5,
                                "gap": 0.0}))  # fmt: skip
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"feasible": True, "list_model": lm, "report": rep}))
    out = tmp_path / "out.json"
    r = CliRunner().invoke(app, ["gate-add-tau-list", str(gate), str(fit), "--out", str(out)])
    assert r.exit_code == 0, r.output
    written = json.loads(out.read_text())
    assert written["list_model"]["fingerprint"] == lm["fingerprint"] and written["gap"] == 0.0
    Gate(**written)  # loads
    fit.write_text(json.dumps({"feasible": False, "list_model": lm}))
    r = CliRunner().invoke(app, ["gate-add-tau-list", str(gate), str(fit), "--out", str(out)])
    assert r.exit_code == 2 and "infeasible" in r.output
    gate.write_text(json.dumps({"platt_a": 2.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5,
                                "gap": 0.0}))  # fmt: skip
    fit.write_text(json.dumps({"feasible": True, "list_model": lm}))
    r = CliRunner().invoke(app, ["gate-add-tau-list", str(gate), str(fit), "--out", str(out)])
    assert r.exit_code == 2 and "Platt" in r.output


# ---- MEM2-255: `mem calibrate` records carry `<qid>:<kind>`, the manifest the bare question id ---


def _calibrate_format(n=30):
    """Pools as `bench.datasets.calibration.score_tune` writes them: an answerable and a negative
    run of every question, qid `<question id>:<kind>`."""
    pools, queries = _data(n)
    out = []
    for p in pools:
        out.append(QueryPool(f"{p.qid}:answerable", True, p.items, p.category))
        out.append(QueryPool(f"{p.qid}:negative", False, p.items[3:], p.category))
    return out, queries


def _fit_tau_list_cli(tmp_path, monkeypatch, pools, queries, split=lambda _q: "tune"):
    from typer.testing import CliRunner

    from bench import cli as bench_cli
    from bench.datasets import lme
    from mem import calibrate

    records = tmp_path / "records.jsonl"
    calibrate.write_records(records, pools, fp_all="a" * 64)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5,
                                "gap": 0.0, "fingerprint_hash": "a" * 64}))  # fmt: skip
    monkeypatch.setattr(lme, "load_questions",
                        lambda _p: [{"question_id": q, "question": t} for q, t in queries.items()])  # fmt: skip
    monkeypatch.setattr(lme, "read_manifest",
                        lambda _p: [lme.ManifestRow(q, "multi-session", False, split(q), 0)
                                    for q in queries])  # fmt: skip
    out = tmp_path / "fit.json"
    r = CliRunner().invoke(bench_cli.app, ["fit-tau-list", str(records), "--lme-file",
                                           str(tmp_path / "x.json"), "--gate", str(gate),
                                           "--out", str(out)])  # fmt: skip
    return r, out


def test_question_of_strips_the_pool_kind_only():
    assert listfit.question_of("9bbe84a2:answerable") == "9bbe84a2"
    assert listfit.question_of("9bbe84a2:negative") == "9bbe84a2"
    assert listfit.question_of("9bbe84a2") == "9bbe84a2"
    assert listfit.question_of("syn3:q:7") == "syn3:q:7"  # not a pool kind: left alone


def test_fit_tau_list_accepts_calibrate_records_and_fits_their_aggregation_questions(
    tmp_path, monkeypatch
):
    pools, queries = _calibrate_format(30)
    queries["q0"] = "Where did I park?"  # not an aggregation: out of the fit
    r, out = _fit_tau_list_cli(tmp_path, monkeypatch, pools, queries)
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())["report"]
    # 29 aggregation questions, their answerable pools only (ADR 0023 §4: negatives stay out)
    assert rep["n"] == 29 and rep["nested"]["feasible"] is True
    qs = listfit.eligible(pools, queries, (1.0, 0.0))
    assert {listfit.question_of(q.qid) for q in qs} == {f"q{i}" for i in range(1, 30)}


def test_a_questions_pools_share_one_fold_keyed_by_its_bare_id():
    pools, _ = _calibrate_format(40)
    for p in pools:
        q = listfit.Query(p.qid, np.array([0.9]), ["s"], frozenset({"s"}))
        bare = listfit.question_of(p.qid)
        assert listfit.query_fold(q, listfit.FOLDS) == listfit.fold_of(bare, listfit.FOLDS)
        assert listfit.query_fold(q, listfit.INNER, inner=True) == listfit.fold_of(
            "i" + bare, listfit.INNER)  # fmt: skip


def test_fit_tau_list_still_refuses_a_pool_of_a_test_question(tmp_path, monkeypatch):
    pools, queries = _calibrate_format(30)
    r, out = _fit_tau_list_cli(tmp_path, monkeypatch, pools, queries,
                               split=lambda q: "test" if q == "q5" else "tune")  # fmt: skip
    assert r.exit_code == 2 and "not LME tune" in r.output and "q5:" in r.output, r.output
    assert not out.exists()
