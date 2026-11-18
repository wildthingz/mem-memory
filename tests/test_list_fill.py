"""Round-level list fill (MEM2-268): the round unit of the tau_list fit, the dry run, and the
`read.list_fill` switch. T0, on the hash models."""

from __future__ import annotations

import numpy as np
import pytest

from mem import listfit
from mem.calibrate import PoolItem, QueryPool

PLATT = (1.0, 0.0)  # p = sigmoid(z)
HOW_MANY = "How many hikes did I do?"


def _pool(qid, rows, *, answerable=True, category="multi-session"):
    """rows: (key, z, rel), any order."""
    return QueryPool(qid, answerable, [PoolItem(k, z, rel) for k, z, rel in rows], category)


# ---- step 1: the fit's unit --------------------------------------------------------------------


def test_round_unit_covers_gold_rounds_not_sessions():
    # the gold session's best row is small talk (rel 0); its gold round sits below the cut
    pool = _pool("q1:answerable", [("s1#0", 2.0, 0), ("s1#1", 0.0, 1), ("s2#0", -1.0, 0)])
    by_session = listfit.eligible([pool], {"q1": HOW_MANY}, PLATT)
    by_round = listfit.eligible([pool], {"q1": HOW_MANY}, PLATT, unit="round")
    assert by_session[0].gold == frozenset({"s1"})
    assert by_round[0].gold == frozenset({"s1#1"})
    # cut at p 0.6: only s1#0 (p 0.88) is delivered
    assert listfit.score(by_session, 0.6, 1000)["coverage"] == 1.0
    assert listfit.score(by_round, 0.6, 1000)["coverage"] == 0.0
    # cut at p 0.4: s1#0 and s1#1 (p 0.5) are delivered: both units covered
    assert listfit.score(by_round, 0.4, 1000)["coverage"] == 1.0


def test_round_unit_still_prices_tokens_by_session_groups():
    pool = _pool("q1:answerable", [("s1#0", 2.0, 1), ("s1#1", 1.0, 1), ("s2#0", 0.5, 0)])
    (q,) = listfit.eligible([pool], {"q1": HOW_MANY}, PLATT, unit="round")
    idx, tokens, _ = listfit.deliver(q, 0.5, 1000)
    assert idx == [0, 1, 2]
    assert tokens == listfit.listmode.tokens_for(3, 2)  # three rows in two session groups


def test_unit_must_be_session_or_round():
    pool = _pool("q1:answerable", [("s1#0", 2.0, 1)])
    with pytest.raises(ValueError, match="unit"):
        listfit.eligible([pool], {"q1": HOW_MANY}, PLATT, unit="chunk")


def test_query_without_unit_falls_back_to_the_session():
    q = listfit.Query("q", np.array([0.9, 0.8]), ["s1", "s2"], frozenset({"s2"}))
    assert listfit.score([q], 0.85, 1000)["coverage"] == 0.0
    assert listfit.score([q], 0.7, 1000)["coverage"] == 1.0


# ---- step 2: the dry run (no models: recorded pools and deliveries) ---------------------------

TAU_LIST = 0.5  # with PLATT (1, 0): keep if z >= 0


def _dryrun_data():
    """Four answerable multi-session-or-not questions and one negative pool, as `mem calibrate`
    and the delivered dump record them."""
    from bench.list_fill_dryrun import Delivered

    pools = [
        # q1 (cued): the second gold round sits below tau_list at rank 3: a fill to 3 recovers it
        _pool(
            "q1:answerable", [("a#0", 2.0, 1), ("b#0", 1.0, 0), ("c#0", -1.0, 1), ("d#0", -2.0, 0)]
        ),
        # q2 (cued): every gold round is above the cut already
        _pool("q2:answerable", [("a#0", 2.0, 1), ("b#0", 1.5, 1)]),
        # q3 (not cued): the normal path; the sim's rule would fill it, the switch leaves it alone
        _pool("q3:answerable", [("x#0", 2.0, 1), ("y#0", -1.0, 1)]),
        # q4 (cued, knowledge-update): complete today; a fill only adds a distractor row
        _pool("q4:answerable", [("k#0", 2.0, 1), ("m#0", -1.0, 0)], category="knowledge-update"),
        # q5 (cued): nothing above tau_list: abstains today and under every fill
        _pool("q5:answerable", [("e#0", -0.5, 1), ("f#0", -3.0, 0)]),
        _pool("q1:negative", [("n#0", -1.0, 0), ("n#1", -2.0, 0)], answerable=False),
    ]
    queries = {"q1": HOW_MANY, "q2": HOW_MANY, "q3": "Where did I hike in May?", "q4": HOW_MANY,
               "q5": HOW_MANY}  # fmt: skip
    delivered = {
        "q1": Delivered("q1", "multi-session", ["a#0", "b#0"], {"a#0", "c#0"}, {"a", "c"}),
        "q2": Delivered("q2", "multi-session", ["a#0", "b#0"], {"a#0", "b#0"}, {"a", "b"}),
        "q3": Delivered("q3", "multi-session", ["x#0"], {"x#0", "y#0"}, {"x", "y"}),
        "q4": Delivered("q4", "knowledge-update", ["k#0"], {"k#0"}, {"k"}),
        "q5": Delivered("q5", "multi-session", [], {"e#0"}, {"e"}),
    }
    return pools, queries, delivered


def test_dryrun_fill_recovers_the_gold_round_below_the_cut_on_cued_questions_only():
    from bench import list_fill_dryrun as dr

    pools, queries, delivered = _dryrun_data()
    rep = dr.run(pools, delivered, queries, PLATT, TAU_LIST, ns=(2, 3), cap=1000, refit=False)
    ms = rep["multi_session"]
    assert ms["n"] == 4  # q1, q2, q3, q5: every answerable multi-session question, cued or not
    assert ms["arms"]["today"]["complete"] == 1  # q2
    assert ms["arms"]["fill_2"]["complete"] == 1  # max(N, rows above tau_list) = 2: nothing added
    assert ms["arms"]["fill_3"]["complete"] == 2  # q1 recovers c#0; q3 is not cued; q5 abstains
    assert ms["arms"]["fill_3"]["recovered"] == ["q1"]
    # the sim's rule (`sim.py`: delivered | top N on every multi-session question, no cue test)
    # it fills q3 (not cued) and q5 (an abstention) too, which the switch never does
    assert ms["arms"]["sim_top_3"]["complete"] == 4
    assert rep["cued"] == ["q1", "q2", "q4", "q5"]


def test_dryrun_reports_tokens_rows_and_the_non_multi_session_effects():
    from bench import list_fill_dryrun as dr
    from mem import listmode

    pools, queries, delivered = _dryrun_data()
    rep = dr.run(pools, delivered, queries, PLATT, TAU_LIST, ns=(3,), cap=1000, refit=False)
    fill = rep["multi_session"]["arms"]["fill_3"]
    # q1: a#0 b#0 c#0 in three sessions; q2: two rows in two sessions; q3 untouched (one row); q5 none
    assert fill["rows"] == {"q1": 3, "q2": 2, "q3": 1, "q5": 0}
    assert fill["tokens_p95"] == listmode.tokens_for(3, 3)
    other = rep["other_list_mode"]  # cued questions that are not multi-session: q4
    assert other["n"] == 1
    assert (
        other["arms"]["today"]["rows_mean"] == 1.0 and other["arms"]["fill_3"]["rows_mean"] == 2.0
    )
    assert other["arms"]["today"]["complete"] == other["arms"]["fill_3"]["complete"] == 1


def test_dryrun_fill_never_changes_an_abstention():
    from bench import list_fill_dryrun as dr

    pools, queries, delivered = _dryrun_data()
    rep = dr.run(pools, delivered, queries, PLATT, TAU_LIST, ns=(3,), cap=1000, refit=False)
    abst = rep["abstentions"]  # over the cued pools, answerable and negative
    assert abst["today"] == {"answerable": 1, "negative": 1}  # q5, q1:negative
    assert abst["fill_3"] == abst["today"] and rep["abstention_changes"]["fill_3"] == 0


def test_dryrun_fill_respects_the_cap_from_the_bottom():
    from bench import list_fill_dryrun as dr
    from mem import listmode

    pools, queries, delivered = _dryrun_data()
    cap = listmode.tokens_for(2, 2)  # room for two rows in two sessions, not three
    rep = dr.run(pools, delivered, queries, PLATT, TAU_LIST, ns=(3,), cap=cap, refit=False)
    fill = rep["multi_session"]["arms"]["fill_3"]
    assert fill["rows"]["q1"] == 2 and fill["complete"] == 1 and fill["tokens_p95"] <= cap


def test_dryrun_refit_arm_needs_enough_questions_and_says_so():
    from bench import list_fill_dryrun as dr

    pools, queries, delivered = _dryrun_data()
    rep = dr.run(pools, delivered, queries, PLATT, TAU_LIST, ns=(3,), cap=1000, refit=True)
    assert "refit_round" not in rep["multi_session"]["arms"]
    assert "at least" in rep["refit_round"]["skipped"]


def _dryrun_cli(tmp_path, monkeypatch, pools, queries, delivered, *, fp="a" * 64,
                split=lambda _q: "tune", extra=()):  # fmt: skip
    import json

    from typer.testing import CliRunner

    from bench import cli as bench_cli
    from bench.datasets import lme
    from mem import calibrate

    records = tmp_path / "records.jsonl"
    calibrate.write_records(records, pools, fp_all=fp)
    dl = tmp_path / "delivered.jsonl"
    dl.write_text("".join(json.dumps({"qid": d.qid, "category": d.category, "delivered": d.rows,
                                      "gold_rounds": sorted(d.gold_rounds),
                                      "gold_sessions": sorted(d.gold_sessions), "paged": []}) + "\n"
                          for d in delivered.values()))  # fmt: skip
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.4, "tau_high": 0.5,
                                "gap": 0.0, "fingerprint_hash": "a" * 64,
                                "list_model": {"tau_list": TAU_LIST}}))  # fmt: skip
    monkeypatch.setattr(lme, "iter_questions",
                        lambda _p: iter([{"question_id": q, "question": t}
                                         for q, t in queries.items()]))  # fmt: skip
    monkeypatch.setattr(lme, "read_manifest",
                        lambda _p: [lme.ManifestRow(q, "multi-session", False, split(q), 0)
                                    for q in queries])  # fmt: skip
    out = tmp_path / "dryrun.json"
    r = CliRunner().invoke(bench_cli.app, ["list-fill-dryrun", str(records), "--delivered", str(dl),
                                           "--gate", str(gate), "--lme-file", str(tmp_path / "x.json"),
                                           "--out", str(out), "--ns", "2,3", "--no-refit", *extra])  # fmt: skip
    return r, out


def test_dryrun_cli_writes_the_report_and_prints_the_table(tmp_path, monkeypatch):
    import json

    pools, queries, delivered = _dryrun_data()
    r, out = _dryrun_cli(tmp_path, monkeypatch, pools, queries, delivered)
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    assert rep["multi_session"]["arms"]["fill_3"]["complete"] == 2
    assert "fill_3" in r.output and "2/4" in r.output
    assert out.with_suffix(".txt").exists()


def test_dryrun_cli_refuses_a_test_question_and_a_foreign_gate(tmp_path, monkeypatch):
    pools, queries, delivered = _dryrun_data()
    r, out = _dryrun_cli(tmp_path, monkeypatch, pools, queries, delivered,
                         split=lambda q: "test" if q == "q2" else "tune")  # fmt: skip
    assert r.exit_code == 2 and "not LME tune" in r.output and not out.exists()
    r, out = _dryrun_cli(tmp_path, monkeypatch, pools, queries, delivered, fp="b" * 64)
    assert r.exit_code == 2 and "fingerprint_hash" in r.output and not out.exists()


# ---- step 3: the switch `read.list_fill` in the config and the fingerprint ----------------------


def test_list_fill_is_off_by_default_and_fingerprinted_only_when_set():
    from mem import config
    from tests.test_config import GOLDEN_FP_ALL_LIST_FILL, GOLDEN_FP_ALL_LIST_MODE, all_off

    on = all_off().replace("read", list_mode=True)
    assert on.list_fill == 0 and "list_fill" not in on.read
    assert on.fp_all() == GOLDEN_FP_ALL_LIST_MODE
    filled = on.replace("read", list_fill=16)
    assert filled.list_fill == 16 and filled.fp_all() == GOLDEN_FP_ALL_LIST_FILL
    assert (
        filled.replace("read", list_fill=0).fp_all() == GOLDEN_FP_ALL_LIST_MODE
    )  # 0 drops the key
    assert "list_fill" not in filled.replace("read", list_fill=0).read
    assert config.default_config().list_fill == 0


@pytest.mark.release_defaults
def test_list_fill_16_is_in_the_release_default(tmp_path):
    """MEM2-274: the release default is Round 10 r10-all, list fill 16 on (ADR 0030 adopted)."""
    from mem import config
    from tests.test_config import GOLDEN_FP_ALL

    assert config.default_config().fp_all() == GOLDEN_FP_ALL
    assert config.default_config().list_fill == 16
    p = tmp_path / "c.yaml"
    p.write_text("read: {list_fill: 16}\n")  # the default written out: the same fingerprint
    assert config.load(p).fp_all() == GOLDEN_FP_ALL
    p.write_text("read: {list_fill: 0}\n")  # off drops the key: another fingerprint
    cfg = config.load(p)
    assert cfg.list_fill == 0 and "list_fill" not in cfg.read and cfg.fp_all() != GOLDEN_FP_ALL


def test_list_fill_in_the_config_file_is_a_non_negative_int_and_needs_list_mode(tmp_path):
    from mem import config

    p = tmp_path / "c.yaml"
    p.write_text("read: {list_mode: true, list_fill: 16}\n")
    assert config.load(p).list_fill == 16
    p.write_text("read: {list_mode: true, list_fill: 0}\n")
    assert config.load(p).list_fill == 0 and "list_fill" not in config.load(p).read
    for bad in ("true", "-1", "'16'", "1.5"):
        p.write_text(f"read: {{list_mode: true, list_fill: {bad}}}\n")
        with pytest.raises(config.ConfigError, match="list_fill"):
            config.load(p)
    p.write_text("read: {list_mode: false, list_fill: 16}\n")
    with pytest.raises(config.ConfigError, match="needs read.list_mode"):
        config.load(p)


def test_store_reads_list_fill_only_with_list_mode(tmp_path):
    from datetime import UTC, datetime

    from mem.store import Store
    from mem.testing import HashEmbedder, OverlapReranker

    now = lambda: datetime(2024, 6, 1, tzinfo=UTC)
    s = Store.open(tmp_path / "a.db", HashEmbedder(), reranker=OverlapReranker(),
                   read={"list_mode": True, "list_fill": 16}, now=now)  # fmt: skip
    assert s.list_fill == 16
    s.close()
    s = Store.open(tmp_path / "b.db", HashEmbedder(), reranker=OverlapReranker(),
                   read={"list_mode": False, "list_fill": 16}, now=now)  # fmt: skip
    assert s.list_fill == 0
    s.close()


# ---- step 4: the switch in `_recall_list` (T0, hash models) -------------------------------------
# `tests.test_list_mode._store`: five hiking sessions (one with two matching rounds: six rows at
# p 0.95 under Platt (1, 0)) and two off-topic sessions (the garage, p about 0.10; dinner, p about
# 0.05); tau_list 0.5 keeps the six. A fill to 8 adds the two off-topic rows, by p.


def _filled(tmp_path, name, fill, **kw):
    from tests.test_list_mode import _store, _sub

    s = _store(_sub(tmp_path, name), list_mode=True, **kw)
    if fill is not None:
        s.read_cfg = {**s.read_cfg, "list_fill": fill}
    return s


def _recall(s, q=None):
    from mem import read
    from tests.test_list_mode import Q

    return read.recall(s, query=q or Q)


def test_abstention_is_byte_identical_with_the_fill_on(tmp_path):
    off, on = _filled(tmp_path, "a", None, tau=0.999), _filled(tmp_path, "b", 16, tau=0.999)
    x, y = _recall(off), _recall(on)
    assert x.text.startswith("no relevant memories")
    assert x.text == y.text and x.structured == y.structured
    off.close()
    on.close()


def test_fill_delivers_the_top_n_by_p_after_the_cut(tmp_path):
    s = _filled(tmp_path, "a", 8)
    out = _recall(s)
    rows = out.structured["rows"]
    assert len(rows) == 8 and out.text.startswith("8 results (of ")
    ps = [r["p"] for r in rows]
    assert ps == sorted(ps, reverse=True)  # the cut rows first, then the fill, by p
    assert all(p >= 0.5 for p in ps[:6]) and all(p < 0.5 for p in ps[6:])
    assert out.structured["matches"] == 6 and out.structured["filled"] == 2
    assert "7 sessions" in out.text.splitlines()[0]
    s.close()
    s = _filled(tmp_path, "b", 7)
    rows = _recall(s).structured["rows"]
    assert (
        len(rows) == 7 and rows[-1]["date"] == "2024-04-10"
    )  # the garage (p 0.10) before dinner (0.05)
    s.close()


def test_fill_adds_nothing_when_more_rows_pass_the_cut_than_n(tmp_path):
    off, on = _filled(tmp_path, "a", None), _filled(tmp_path, "b", 3)
    x, y = _recall(off), _recall(on)
    assert x.text == y.text and y.structured["rows"] == x.structured["rows"]
    assert y.structured["matches"] == 6 and y.structured["filled"] == 0
    off.close()
    on.close()


def test_the_cap_holds_with_a_long_fill_and_the_over_cap_line_counts_the_matches_only(tmp_path):
    from bench.suites.b_p2_cost import list_violations
    from mem import listmode

    s = _filled(tmp_path, "a", 8)
    out = _recall(s)  # the budget argument is the cap, as in list mode today
    from mem import read
    from tests.test_list_mode import Q

    out = read.recall(s, query=Q, budget=120)
    assert listmode.response_tokens(out.text) <= 120
    shown = len(out.structured["rows"])
    assert 0 < shown < 6 and all(r["p"] >= 0.5 for r in out.structured["rows"])
    last = out.text.splitlines()[-1]
    assert last == listmode.MORE.format(n=6 - shown, cap=120)  # the fill rows are not "matches"
    assert list_violations(out.text) == []
    s.close()


def test_fill_rows_cut_by_the_cap_do_not_claim_matches_above_tau_list(tmp_path):
    from mem import listmode, read
    from tests.test_list_mode import Q

    off = _filled(tmp_path, "a", None)
    six = _recall(off).text
    cap = listmode.response_tokens(six)  # room for the six matches, not for a fill row
    on = _filled(tmp_path, "b", 8)
    out = read.recall(on, query=Q, budget=cap)
    assert len(out.structured["rows"]) == 6 and "more matches" not in out.text
    assert out.text == six
    assert out.structured["filled"] == 0
    off.close()
    on.close()


def test_non_list_queries_are_untouched_by_the_fill(tmp_path):
    off, on = _filled(tmp_path, "a", None), _filled(tmp_path, "b", 8)
    for q in ("Where did I park the car", "Which hiking trip was in March"):
        x, y = _recall(off, q), _recall(on, q)
        assert x.text == y.text and x.structured == y.structured, q
        assert "list mode" not in x.text
    off.close()
    on.close()


def test_switch_off_is_todays_output(tmp_path):
    import json
    from pathlib import Path

    from mem import read
    from tests.test_list_mode import Q, _store, _sub

    absent, zero = _filled(tmp_path, "a", None), _filled(tmp_path, "b", 0)
    for q in (Q, "Where did I park the car", "List all the dinners I had"):
        x, y = _recall(absent, q), _recall(zero, q)
        assert x.text == y.text and x.structured == y.structured and "filled" not in x.structured
    absent.close()
    zero.close()
    # list mode off with a fill set: the store ignores the fill, and the pre-list-mode golden holds
    golden = json.loads((Path(__file__).parent / "fixtures/golden/list_mode_off.json").read_text())
    s = _store(_sub(tmp_path, "c"), list_mode=False, gate=True)
    s.read_cfg = {**s.read_cfg, "list_fill": 16}
    assert s.list_fill == 0
    for q, (text, structured) in golden.items():
        r = read.recall(s, query=q)
        assert r.text == text and r.structured == structured, q
    s.close()


def test_the_list_grammar_holds_with_the_fill(tmp_path):
    from bench.suites.b_p2_cost import list_violations

    s = _filled(tmp_path, "a", 8)
    assert list_violations(_recall(s).text) == []
    s.close()


def test_paging_pages_the_filled_rows(tmp_path):
    from mem import read
    from tests.test_list_mode import Q

    s = _filled(tmp_path, "a", 8)
    s.read_cfg = {**s.read_cfg, "list_paging": True}
    out = read.recall(s, query=Q, budget=220)
    seen = [r["id"] for r in out.structured["rows"]]
    assert out.structured["pages"] > 1 and out.structured["matches"] == 6
    while "next_page" in out.structured:
        out = read.recall(s, query=Q, budget=220, page=out.structured["next_page"])
        seen += [r["id"] for r in out.structured["rows"]]
    assert len(seen) == 8 and len(set(seen)) == 8
    s.close()
