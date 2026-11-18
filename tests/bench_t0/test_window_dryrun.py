"""`bench window-dryrun` (MEM2-266): the anchored-window boost replayed on recorded pools, no models.
Hand-made pools under the 9c gate's Platt and tau_keep; one tiny LongMemEval-shaped file for the
loader and the CLI."""

from __future__ import annotations

import json
from datetime import date

import pytest
from typer.testing import CliRunner

from bench import window_dryrun as wd
from bench.cli import app
from bench.datasets import lme
from mem.gate import Gate
from mem.listmode import ListModel

A, B = 0.5125562764300347, -0.08145967892844123
LIST = ListModel.build(
    0.022889767566145768, 1000, 0.95, A, B, {"outer_folds": [{"fold": 0}]}
).to_dict()
GATE = Gate(A, B, 0.019434903617718057, 1.0, 3.358e-05, list_model=LIST)  # z >= -7.49 is kept
REF = date(2023, 3, 9)
IN, OUT = date(2023, 3, 4), date(2023, 1, 2)  # "last Saturday" = 2023-03-04


def _q(items, *, question="Who gave me the jewelry I got last Saturday?", gold=("g#0",),
       kind="answerable", recorded=None):  # fmt: skip
    """items: (key, z, day)."""
    return wd.Query(
        qid="q1", kind=kind, category="temporal-reasoning", question=question, ref=REF,
        keys=[k for k, _, _ in items], zs=[z for _, z, _ in items], days=[d for _, _, d in items],
        lines=[f"c{i} · chat_session · {d} · chat session {d}" for i, (_, _, d) in enumerate(items)],
        gold=frozenset(gold if kind == "answerable" else ()), recorded=recorded,
    )  # fmt: skip


def test_no_boost_is_the_gate_in_p_order_to_the_limit():
    q = _q([(f"n{i}#0", -1.0 - i / 2, OUT) for i in range(10)])
    assert wd.deliver(q, GATE, delta=0.0, flags=[False] * 10) == [f"n{i}#0" for i in range(8)]


def test_an_in_window_gate_cut_is_rescued_and_ordered_first():
    q = _q([("n1#0", -2.0, OUT), ("g#0", -7.51, IN), ("n2#0", -9.0, OUT)])
    assert wd.deliver(q, GATE, delta=0.0, flags=[False] * 3) == ["n1#0"]
    flags = wd.flags(q, slack=0.2)
    assert flags == [False, True, False]
    assert wd.deliver(q, GATE, delta=0.5, flags=flags) == ["g#0", "n1#0"]


def test_boosting_never_filters_an_out_of_window_row():
    q = _q([("n1#0", -2.0, OUT), ("n2#0", -3.0, OUT), ("g#0", -7.0, IN)])
    out = wd.deliver(q, GATE, delta=3.0, flags=wd.flags(q, slack=0.2))
    assert out == ["g#0", "n1#0", "n2#0"]


def test_in_window_rows_first_can_push_gold_past_the_row_limit():
    items = [("g#0", -0.5, OUT)] + [(f"w{i}#0", -2.0 - i / 10, IN) for i in range(8)]
    q = _q(items)
    base = wd.deliver(q, GATE, delta=0.0, flags=[False] * 9)
    assert "g#0" in base
    out = wd.deliver(q, GATE, delta=1.0, flags=wd.flags(q, slack=0.2))
    assert "g#0" not in out and len(out) == 8


def test_a_boost_that_ties_the_top_two_abstains_as_the_gate_does():
    q = _q([("g#0", -1.0, OUT), ("w#0", -2.0, IN)])
    assert wd.deliver(q, GATE, delta=1.0, flags=[False, True]) == []  # p1 - p2 < gap


def test_list_mode_question_uses_tau_list_and_the_token_cap():
    q = _q([(f"n{i}#0", -1.0, OUT) for i in range(80)], question="How many plants did I buy?")
    out = wd.deliver(q, GATE, delta=0.0, flags=[False] * 80)
    assert 8 < len(out) < 80  # list mode: past the row limit, stopped by the 1000-token cap


def test_compare_counts_rescue_push_admit_and_false_accepts():
    rescue = _q([("n1#0", -2.0, OUT), ("g#0", -7.51, IN), ("x#0", -7.6, IN)])
    pushed = _q([("g#0", -0.5, OUT)] + [(f"w{i}#0", -2.0 - i / 10, IN) for i in range(8)])
    neg = _q([("x#0", -7.6, IN), ("y#0", -9.0, OUT)], kind="negative")
    plain = _q([("g#0", -1.0, OUT)], question="Where did I park?")
    rows = [wd.compare(q, GATE, delta=1.0, slack=0.2) for q in (rescue, pushed, neg, plain)]
    s = wd.summarise(rows)
    assert s["fires"] == {"answerable": 2, "negative": 1}
    assert s["rescued"] == 1 and s["pushed_out"] == 1 and s["net"] == 0
    # x#0 on the rescue, w7#0 (the eighth in-window row) on the push
    assert s["admitted"]["questions"] == 2 and s["admitted"]["rows"] == 2
    assert s["admitted"]["tokens"] > 0
    assert s["negatives"]["newly_answered"] == 1 and s["negatives"]["rows"] == 1
    assert rows[3]["window"] == [] and rows[3]["base"] == rows[3]["new"]


def test_gate_cuts_are_gold_in_pool_none_delivered_with_their_z():
    cut = _q([("n1#0", -2.0, OUT), ("g#0", -7.51, IN)])
    row = wd.compare(cut, GATE, delta=1.0, slack=0.2)
    assert row["gate_cut"] == {"z": -7.51, "in_window": True, "rescued": True}
    hit = wd.compare(_q([("g#0", -1.0, OUT)]), GATE, delta=1.0, slack=0.2)
    assert hit["gate_cut"] is None


def test_replay_mismatch_with_the_recorded_delivery_is_reported():
    good = _q([("g#0", -1.0, OUT)], recorded=["g#0"])
    bad = _q([("g#0", -1.0, OUT)], recorded=["other#0"])
    rows = [wd.compare(q, GATE, delta=1.0, slack=0.2) for q in (good, bad)]
    assert wd.summarise(rows)["replay"] == {"checked": 2, "match": 1, "mismatch": ["q1"]}


# ---- loader and CLI over a tiny LongMemEval-shaped file ---------------------------------------


def _dataset(tmp_path):
    def sess(text, has=False):
        return [{"role": "user", "content": text, "has_answer": has},
                {"role": "assistant", "content": "ok"}]  # fmt: skip

    qs = []
    for i in range(4):  # one stratum of 4: round(0.3 x 4) = 1 tune question
        qs.append({
            "question_id": f"t{i}", "question_type": "temporal-reasoning",
            "question": "Who gave me the necklace last Saturday?", "answer": "aunt",
            "question_date": "2023/03/09 (Thu) 15:47",
            "haystack_session_ids": ["s_gold", "s_other"],
            "haystack_dates": ["2023/03/04 (Sat) 16:45", "2023/01/02 (Mon) 10:00"],
            "haystack_sessions": [sess("My aunt gave me a necklace yesterday", True),
                                  sess("I went running")],
            "answer_session_ids": ["s_gold"],
        })  # fmt: skip
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(qs))
    tune = [r.question_id for r in lme.build_manifest(qs) if r.split == "tune"]
    return path, tune[0], next(q["question_id"] for q in qs if q["question_id"] not in tune)


def _files(tmp_path, qid, delivered=()):
    rec = tmp_path / "rec.jsonl"
    items = [{"key": "s_other#0", "z": -2.0, "rel": 0}, {"key": "s_gold#0", "z": -7.51, "rel": 1}]
    neg = [{"key": "s_other#0", "z": -9.0, "rel": 0}]
    rec.write_text("\n".join(json.dumps(x) for x in (
        {"fp_all": "f"},
        {"qid": f"{qid}:answerable", "answerable": True, "items": items, "category": "temporal-reasoning"},
        {"qid": f"{qid}:negative", "answerable": False, "items": neg, "category": "temporal-reasoning"},
    )) + "\n")  # fmt: skip
    dl = tmp_path / "dl.jsonl"
    dl.write_text(json.dumps({"qid": qid, "delivered": list(delivered), "gold_rounds": ["s_gold#0"],
                              "category": "temporal-reasoning"}) + "\n")  # fmt: skip
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({**{k: getattr(GATE, k) for k in wd.GATE_KEYS}, "list_model": LIST}))
    return rec, dl, gate


def test_load_gives_session_dates_rows_and_the_recorded_delivery(tmp_path):
    data, tune, _ = _dataset(tmp_path)
    rec, dl, _ = _files(tmp_path, tune, delivered=["s_other#0"])
    qs = wd.load(rec, dl, data)
    a, n = qs
    assert (a.kind, n.kind) == ("answerable", "negative") and a.ref == date(2023, 3, 9)
    assert a.days == [date(2023, 1, 2), date(2023, 3, 4)] and a.gold == {"s_gold#0"}
    assert a.recorded == ["s_other#0"] and n.recorded is None and n.gold == frozenset()
    # the row line as recall prints it: the chunk id in import order, the resolved dates field
    assert (
        a.lines[1]
        == "c1 · chat_session · 2023-03-04 · chat session 2023-03-04 · dates: 2023-03-03 (yesterday)"
    )
    assert n.lines[0].startswith(
        "c1 · chat_session · 2023-01-02"
    )  # gold session removed: ids shift


def test_load_refuses_a_test_split_question(tmp_path):
    data, _, test_qid = _dataset(tmp_path)
    rec, dl, _ = _files(tmp_path, test_qid)
    with pytest.raises(ValueError, match="tune"):
        wd.load(rec, dl, data)


def test_cli_writes_the_report_and_the_table(tmp_path):
    data, tune, _ = _dataset(tmp_path)
    rec, dl, gate = _files(tmp_path, tune, delivered=["s_other#0"])
    out = tmp_path / "out"
    res = CliRunner().invoke(app, ["window-dryrun", "--records", str(rec), "--delivered", str(dl),
                                   "--gate", str(gate), "--dataset", str(data), "--out", str(out),
                                   "--deltas", "0.5,1", "--slacks", "0.2"])  # fmt: skip
    assert res.exit_code == 0, res.output
    s = json.loads((out / "summary.json").read_text())
    assert s["replay"]["match"] == 1
    best = s["best"]
    # one rescue at both deltas: the smaller wins the tie, and 1 net rescue is under the stop point
    assert best["delta"] == 0.5 and best["net"] == 1 and s["verdict"] == "drop"
    assert (out / "per_query.jsonl").exists()
    table = (out / "table.md").read_text()
    assert "| 0.5 |" in table and "| 1.0 |" in table and table in res.output


def test_verdict_drops_below_two_net_rescues():
    assert wd.verdict(1) == "drop" and wd.verdict(2) == "pass"
