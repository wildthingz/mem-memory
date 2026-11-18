"""`bench temporal-dryrun` (MEM2-209): A's date-window parser and D's two-endpoint patterns, no models."""

import json

from typer.testing import CliRunner

from bench import temporal_dryrun as td
from bench.cli import app

# qid, type, question, question_date: five hand-worked cases.
FIXTURE = [
    ("q1", "temporal-reasoning", "What gardening activity did I do two weeks ago?", "2023/05/05 (Fri) 16:42"),
    ("q2", "temporal-reasoning", "Which device did I set up first, the smart thermostat or the new router?", "2023/03/28 (Tue) 01:25"),
    ("q3", "temporal-reasoning", "What is the order of the sports events I watched in January?", "2023/01/30 (Mon) 01:53"),
    ("q4", "temporal-reasoning", "How many days ago did I buy a smoker?", "2023/03/25 (Sat) 02:46"),
    ("q5", "knowledge-update", "Do I like tea or coffee better?", "2023/03/25 (Sat) 02:46"),
]  # fmt: skip


def test_relative_weeks_ago_gives_a_window_around_that_day():
    w = td.parse_window("What did I do two weeks ago?", "2023/05/05 (Fri) 16:42")
    assert w.rule == "n_units_ago"
    assert (w.start, w.end) == ("2023-04-21", "2023-04-21")


def test_in_month_gives_the_whole_month_of_the_latest_past_occurrence():
    w = td.parse_window("sports events I watched in January?", "2023/01/30 (Mon) 01:53")
    assert (w.rule, w.start, w.end) == ("in_month", "2023-01-01", "2023-01-31")


def test_how_many_days_ago_is_not_a_window():
    assert (
        td.parse_window("How many days ago did I buy a smoker?", "2023/03/25 (Sat) 02:46") is None
    )


def test_last_weekday_is_the_most_recent_such_day_before_the_question():
    w = td.parse_window("jewelry last Saturday from whom?", "2023/03/09 (Thu) 15:47")
    assert (w.rule, w.start, w.end) == ("last_weekday", "2023-03-04", "2023-03-04")


def test_split_which_came_first_or():
    s = td.split_endpoints(
        "Which device did I set up first, the smart thermostat or the new router?"
    )
    assert s.pattern == "a_or_b"
    assert s.parts == ["the smart thermostat", "the new router"]


def test_split_between_and_and_how_long_when():
    assert td.split_endpoints("How many days passed between the day I replaced my spark plugs and the day I raced?").parts == [
        "the day I replaced my spark plugs", "the day I raced"]  # fmt: skip
    s = td.split_endpoints(
        "How long had I been taking guitar lessons when I bought the new guitar amp?"
    )
    assert (s.pattern, s.parts) == (
        "how_long_when",
        ["taking guitar lessons", "I bought the new guitar amp"],
    )


def test_single_event_question_does_not_split():
    assert td.split_endpoints("How many days ago did I buy a smoker?") is None


def test_cli_writes_per_question_jsonl_and_summary(tmp_path):
    tune = tmp_path / "tune.jsonl"
    tune.write_text("".join(json.dumps({"qid": q, "question_type": t, "question": s, "question_date": d, "is_abs": False}) + "\n" for q, t, s, d in FIXTURE))  # fmt: skip
    e1 = tmp_path / "e1.jsonl"
    rows = [{"qid": "q1", "type": "temporal-reasoning", "cond": "mem", "correct": False, "shown": 0},
            {"qid": "q1", "type": "temporal-reasoning", "cond": "oracle", "correct": True, "shown": 8},
            {"qid": "q4", "type": "temporal-reasoning", "cond": "mem", "correct": False, "shown": 3},
            {"qid": "q4", "type": "temporal-reasoning", "cond": "oracle", "correct": True, "shown": 8},
            {"qid": "zz", "type": "temporal-reasoning", "cond": "mem", "correct": False, "shown": 0}]  # fmt: skip
    e1.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    res = CliRunner().invoke(
        app, ["temporal-dryrun", str(tune), "--e1", str(e1), "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    summ = json.loads((out / "summary.json").read_text())
    assert summ["a"]["tune_losses"] == 2  # zz is not a tune qid: never read
    assert summ["a"]["losses_with_window"] == 1  # q1 yes, q4 no
    assert summ["d"]["temporal_matched"] == 1 and summ["d"]["temporal_total"] == 4
    assert summ["d"]["false_splits"] == [
        {"qid": "q5", "pattern": "a_or_b", "parts": ["I like tea", "coffee better"]}
    ]
    per_q = [json.loads(line) for line in (out / "per_question.jsonl").read_text().splitlines()]
    assert [r["qid"] for r in per_q] == ["q1", "q2", "q3", "q4", "q5"]
