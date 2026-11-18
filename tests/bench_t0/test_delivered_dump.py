"""`bench delivered-dump` (MEM2-210, report §7): which (session, round) rows recall delivers for each
tune question against the gold evidence, recall-any/all@k and the gold-session fraction. Hash
models, no API."""

import json

import pytest
from typer.testing import CliRunner

from bench import delivered_dump as dd
from bench.cli import app
from bench.lme_eval import Item, Retrieved
from tests.bench_t0.test_e1_expanded import cli_env  # noqa: F401  (fixture)


def item(qid, cat, gold_rounds, delivered, gold_sessions=None):
    gs = gold_sessions or {g.split("#")[0] for g in gold_rounds}
    return Item(qid, cat, "answerable", frozenset(gs), frozenset(gold_rounds),
                Retrieved(F=[], R=[], S=list(delivered)))  # fmt: skip


def test_recall_all_is_one_only_when_every_gold_round_is_delivered():
    gold = ["a#0", "a#1", "b#2"]
    two_of_three = dd.question_record(item("q1", "multi-session", gold, ["a#0", "b#2", "z#0"]))
    assert two_of_three["at"]["5"]["recall_all"] == 0
    assert two_of_three["at"]["5"]["recall_any"] == 1
    assert two_of_three["at"]["5"]["gold_session_frac"] == 1.0  # a and b both have a row in
    all_three = dd.question_record(item("q1", "multi-session", gold, ["a#0", "b#2", "a#1"]))
    assert all_three["at"]["5"]["recall_all"] == 1


def test_k_cuts_the_delivered_list_in_order():
    rec = dd.question_record(item("q", "t", ["a#0"], ["x#0"] * 6 + ["a#0"]))
    assert [rec["at"][str(k)]["recall_any"] for k in (5, 8, 10)] == [0, 1, 1]


def test_gold_session_fraction_counts_sessions_with_any_delivered_row():
    rec = dd.question_record(item("q", "t", ["a#0", "b#0", "c#0", "d#0"], ["a#0", "a#1", "c#3"]))
    assert rec["at"]["5"]["gold_session_frac"] == 0.5
    assert rec["at"]["5"]["recall_all"] == 0


def test_an_empty_delivery_is_all_zero_not_missing():
    rec = dd.question_record(item("q", "t", ["a#0"], []))
    assert rec["delivered"] == [] and rec["at"]["5"] == {
        "recall_any": 0,
        "recall_all": 0,
        "gold_round_frac": 0.0,
        "gold_session_frac": 0.0,
        "session_any": 0,
    }


def test_summary_rolls_up_overall_and_by_type():
    recs = [dd.question_record(x) for x in (
        item("q1", "A", ["a#0", "a#1"], ["a#0", "a#1"]),   # all
        item("q2", "A", ["b#0", "b#1"], ["b#0"]),          # any, not all
        item("q3", "B", ["c#0"], ["z#9"]),                 # none
        item("q4", "B", ["d#0"], ["d#0"]))]  # fmt: skip
    s = dd.summarize(recs)
    assert s["n"] == 4
    assert s["overall"]["5"] == {
        "n": 4,
        "recall_any": 0.75,
        "recall_all": 0.5,
        "gold_round_frac": 0.625,
        "gold_session_frac": 0.75,
        "session_any": 0.75,
    }
    assert s["by_type"]["A"]["5"]["recall_all"] == 0.5 and s["by_type"]["A"]["5"]["n"] == 2
    assert s["by_type"]["B"]["5"]["recall_any"] == 0.5
    assert s["ks"] == [5, 8, 10]


def test_a_question_without_gold_rounds_is_counted_aside_not_scored():
    recs = [dd.question_record(item("q1", "A", ["a#0"], ["a#0"])),
            dd.question_record(item("q2", "A", [], ["a#0"], gold_sessions={"a"}))]  # fmt: skip
    s = dd.summarize(recs)
    assert s["n"] == 1 and s["skipped_no_gold"] == ["q2"]


def test_cli_writes_a_jsonl_per_question_and_a_summary(cli_env, tmp_path, monkeypatch):  # noqa: F811
    from bench.datasets import lme

    qs, _llms, gate, _out = cli_env
    rows = [lme.ManifestRow(q["question_id"], q["question_type"], False, "tune", i)
            for i, q in enumerate(qs)]  # fmt: skip
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    out = tmp_path / "dump.jsonl"
    r = CliRunner().invoke(app, ["delivered-dump", "--lme-file", "x.json", "--out", str(out),
                                 "--embedder", "hash", "--gate", str(gate)])  # fmt: skip
    assert r.exit_code == 0, r.output
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x["qid"] for x in lines] == [q["question_id"] for q in qs]
    assert all(x["delivered"] and set(x["at"]) == {"5", "8", "10", "all"} for x in lines)
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == len(qs) and summary["split"] == "tune"
    assert "recall_all" in summary["overall"]["10"]
    assert "recall-all@10" in r.output


@pytest.mark.parametrize("split", ["dev", "test"])
def test_cli_refuses_the_test_split(cli_env, tmp_path, split):  # noqa: F811
    _qs, _l, gate, _o = cli_env
    r = CliRunner().invoke(
        app,
        [
            "delivered-dump",
            "--lme-file",
            "x.json",
            "--out",
            str(tmp_path / "d.jsonl"),
            "--split",
            split,
            "--gate",
            str(gate),
        ],
    )
    assert r.exit_code != 0


# ---- bench delivered-pair (MEM2-221): two dumps, paired by question -------------------------------


def _rec(qid, cat, k8_all):
    return {"qid": qid, "category": cat, "gold_rounds": ["x#0"],
            "at": {"8": {"recall_any": 1, "recall_all": k8_all, "gold_session_frac": 1.0, "session_any": 1}}}  # fmt: skip


def test_pair_reports_the_delta_of_a_feature_dump_over_the_baseline_on_shared_questions():
    base = [_rec(f"q{i}", "temporal-reasoning" if i % 2 else "multi-session", 0) for i in range(10)]
    feat = [
        _rec(f"q{i}", "temporal-reasoning" if i % 2 else "multi-session", int(i < 4))
        for i in range(10)
    ]
    feat.append(_rec("only-in-feature", "multi-session", 1))  # not paired: ignored
    out = dd.pair(base, feat, ks=(8,))
    row = out["overall"]["8"]["recall_all"]
    assert row["n"] == 10 and row["delta"] == pytest.approx(0.4)
    assert row["lo"] <= 0.4 <= row["hi"]
    assert out["by_type"]["multi-session"]["8"]["recall_all"]["delta"] == pytest.approx(0.4)
    assert out["by_type"]["temporal-reasoning"]["8"]["recall_all"]["delta"] == pytest.approx(0.4)


def test_pair_skips_questions_without_gold_rounds():
    a = [{**_rec("q", "multi-session", 0), "gold_rounds": []}]
    assert dd.pair(a, a, ks=(8,))["overall"] == {}


def test_delivered_pair_command_prints_overall_and_per_type_rows(tmp_path):
    def recs(hit):
        return [{**_rec(f"q{i}", "multi-session", int(i < hit)),
                 "at": {str(k): _rec("q", "m", int(i < hit))["at"]["8"] for k in dd.KS}}
                for i in range(6)]  # fmt: skip

    for name, hit in (("a", 1), ("b", 4)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs(hit)))
    res = CliRunner().invoke(
        app, ["delivered-pair", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    )
    assert res.exit_code == 0, res.output
    assert "n=6" in res.output and "recall-all@8" in res.output and "+0.500" in res.output
    assert "multi-session" in res.output


# ---- whole delivery (MEM2-259): every delivered row, every page followed -----------------------


def paged_item(qid, cat, gold_rounds, page1, later):
    it = item(qid, cat, gold_rounds, page1)
    it.r.paged = list(later)
    return it


def test_whole_delivery_counts_rows_past_k_and_on_later_pages():
    # a list response: 12 rows on page 1 (gold a#0 at position 11, past k=10), gold b#0 on page 2
    page1 = [f"x#{i}" for i in range(10)] + ["a#0", "y#0"]
    rec = dd.question_record(
        paged_item("q", "multi-session", ["a#0", "b#0"], page1, ["z#0", "b#0"])
    )
    assert rec["at"]["10"]["recall_any"] == 0 and rec["at"]["10"]["recall_all"] == 0
    assert rec["at"]["all"] == {"recall_any": 1, "recall_all": 1, "gold_round_frac": 1.0,
                                "gold_session_frac": 1.0, "session_any": 1}  # fmt: skip
    assert rec["paged"] == ["z#0", "b#0"]
    one = dd.question_record(item("q", "multi-session", ["a#0", "b#0"], page1))
    assert one["at"]["all"]["recall_all"] == 0 and one["at"]["all"]["gold_round_frac"] == 0.5
    assert one["paged"] == []


def test_gold_round_fraction_is_the_share_of_gold_rounds_delivered():
    rec = dd.question_record(item("q", "t", ["a#0", "a#1", "b#0", "c#0"], ["a#0", "z#0", "c#0"]))
    assert rec["at"]["5"]["gold_round_frac"] == 0.5


def test_summary_and_pair_carry_the_whole_delivery_column():
    base = [dd.question_record(item(f"q{i}", "multi-session", ["a#0", "b#0"], ["a#0"]))
            for i in range(6)]  # fmt: skip
    page1 = [f"x#{i}" for i in range(10)] + ["a#0"]
    feat = [dd.question_record(paged_item(f"q{i}", "multi-session", ["a#0", "b#0"], page1,
                                          ["b#0"] if i < 3 else []))
            for i in range(6)]  # fmt: skip
    s = dd.summarize(feat)
    assert (
        s["overall"]["all"]["recall_all"] == 0.5 and s["by_type"]["multi-session"]["all"]["n"] == 6
    )
    out = dd.pair(base, feat)
    whole = out["overall"]["all"]
    assert whole["recall_all"]["delta"] == pytest.approx(0.5)
    assert whole["recall_all"]["lo"] <= 0.5 <= whole["recall_all"]["hi"]
    assert whole["recall_any"]["delta"] == 0 and whole["gold_round_frac"]["delta"] == pytest.approx(
        0.25
    )
    assert out["by_type"]["multi-session"]["all"]["recall_all"]["delta"] == pytest.approx(0.5)
    assert out["overall"]["10"]["recall_all"]["delta"] == 0  # the extra rows fall past k


def test_pair_derives_the_whole_column_of_an_older_dump_from_its_delivered_rows():
    old = dd.question_record(item("q", "t", ["a#0"], ["z#0"] * 11 + ["a#0"]))
    del old["at"]["all"], old["paged"]
    for k in dd.KS:
        del old["at"][str(k)]["gold_round_frac"]
    new = dd.question_record(item("q", "t", ["a#0"], ["z#0"] * 11 + ["a#0"]))
    out = dd.pair([old], [new])
    assert out["overall"]["all"]["recall_all"]["delta"] == 0
    assert out["overall"]["5"]["gold_round_frac"]["delta"] == 0


def test_delivered_pair_command_prints_the_whole_delivery_column(tmp_path):
    base = [dd.question_record(item(f"q{i}", "multi-session", ["a#0", "b#0"], ["a#0"]))
            for i in range(4)]  # fmt: skip
    feat = [dd.question_record(paged_item(f"q{i}", "multi-session", ["a#0", "b#0"], ["a#0"], ["b#0"]))
            for i in range(4)]  # fmt: skip
    for name, recs in (("a", base), ("b", feat)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = CliRunner().invoke(
        app, ["delivered-pair", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    )
    assert res.exit_code == 0, res.output
    assert "whole: recall-any +0.000" in res.output and "recall-all +1.000" in res.output
    assert "gold-rounds +0.500" in res.output


def test_the_bench_backend_follows_every_page_of_a_paged_answer(tmp_path):
    from bench.adapters.lme_ours import later_pages
    from mem import read
    from tests.test_list_mode import Q
    from tests.test_list_paging import CAP, _paged, _walk

    s = _paged(tmp_path)
    pages = _walk(s)
    assert len(pages) > 2
    first = read.recall(s, query=Q, budget=CAP)
    later = later_pages(s, first, query=Q, budget=CAP)
    assert [r["id"] for r in later] == [r["id"] for p in pages[1:] for r in p.structured["rows"]]
    one = read.recall(s, query="where did I park")  # not paged: nothing to follow
    assert later_pages(s, one, query="where did I park") == []
    s.close()


def test_cli_dump_follows_pages_and_records_them(cli_env, tmp_path, monkeypatch):  # noqa: F811
    from bench.adapters import lme_ours
    from bench.datasets import lme

    qs, _llms, gate, _out = cli_env
    rows = [lme.ManifestRow(q["question_id"], q["question_type"], False, "tune", i)
            for i, q in enumerate(qs)]  # fmt: skip
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    seen = {}
    init = lme_ours.OursLmeBackend.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        init(self, *a, **kw)

    monkeypatch.setattr(lme_ours.OursLmeBackend, "__init__", spy)
    out = tmp_path / "dump.jsonl"
    r = CliRunner().invoke(app, ["delivered-dump", "--lme-file", "x.json", "--out", str(out),
                                 "--embedder", "hash", "--gate", str(gate)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert seen.get("follow_pages") is True
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert all(x["paged"] == [] for x in lines)  # the default config does not page
    assert all("all" in x["at"] for x in lines)
    assert "whole" in r.output
