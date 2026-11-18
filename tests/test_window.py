"""MEM2-266: the anchored window of a query (`mem.window`). The query's relative date is resolved
with the `mem.reldates` rules against the question day (no new parser, no model); the window's
slack grows with the anchor's distance, `max(1 day, ceil(slack x distance))`."""

from __future__ import annotations

from datetime import date

import pytest

from mem import window

REF = date(2023, 3, 9)  # a Thursday


def _span(ws):
    return [(w.start.isoformat(), w.end.isoformat()) for w in ws]


def test_last_weekday_is_a_day_with_one_day_slack():
    (w,) = window.windows("I received a piece of jewelry last Saturday from whom?", REF, 0.2)
    assert (w.phrase, w.value) == ("last Saturday", "2023-03-04")
    assert _span([w]) == [("2023-03-03", "2023-03-05")]  # 5 days back: 0.2 x 5 = 1


def test_slack_grows_with_distance():
    (w,) = window.windows("I visited a museum two months ago", date(2023, 3, 11), 0.2)
    assert w.value == "2023-01-11"
    # 59 days back: ceil(0.2 x 59) = 12
    assert _span([w]) == [("2022-12-30", "2023-01-23")]


def test_slack_zero_is_the_one_day_floor():
    (w,) = window.windows("what did I do two weeks ago", REF, 0.0)
    assert _span([w]) == [("2023-02-22", "2023-02-24")]


def test_a_month_value_is_the_whole_month_plus_slack():
    (w,) = window.windows("what did I buy last month", REF, 0.2)
    assert w.value == "2023-02"
    # the month's middle is 2023-02-14, 23 days back: slack ceil(4.6) = 5
    assert _span([w]) == [("2023-01-27", "2023-03-05")]


def test_the_window_never_reaches_past_the_question_day():
    (w,) = window.windows("what did I eat yesterday", REF, 0.2)
    assert _span([w]) == [("2023-03-07", "2023-03-09")]
    (w,) = window.windows("what did I cook this week", REF, 0.2)  # Monday 03-06 to Sunday 03-12
    assert _span([w]) == [("2023-03-05", "2023-03-09")]


@pytest.mark.parametrize(
    "q",
    [
        "What is the order of airlines I flew with before today?",  # today: the store's edge, no anchor
        "What will I do tomorrow?",  # the future: nothing to recall
        "What should I plan for next Friday?",
        "Where did I park the car?",  # no time phrase
        "How many days ago did I buy the shoes?",  # the date is the answer, not the query
        "",
    ],
)
def test_no_window(q):
    assert window.windows(q, REF, 0.2) == []


def test_two_phrases_give_two_windows():
    ws = window.windows("what did I buy a week ago and what did I return yesterday", REF, 0.2)
    assert [w.value for w in ws] == ["2023-03-02", "2023-03-08"]


def test_inside_is_inclusive_on_both_ends():
    ws = window.windows("last Saturday", REF, 0.2)
    assert window.inside(date(2023, 3, 3), ws) and window.inside(date(2023, 3, 5), ws)
    assert not window.inside(date(2023, 3, 2), ws) and not window.inside(date(2023, 3, 6), ws)
    assert not window.inside(None, ws) and not window.inside(date(2023, 3, 4), [])


def test_in_window_first_is_a_stable_partition():
    assert window.in_window_first([4, 1, 3, 0, 2], [False, True, False, True, False]) == [
        1,
        3,
        4,
        0,
        2,
    ]
    assert window.in_window_first([2, 0, 1], [False] * 3) == [2, 0, 1]


def test_settings_true_is_the_tuned_values_and_a_mapping_is_checked():
    assert window.settings(True) == {"delta": window.DELTA, "slack": window.SLACK}
    assert window.settings({"delta": 2, "slack": 0.1}) == {"delta": 2.0, "slack": 0.1}
    for bad in ({"delta": 0, "slack": 0.2}, {"delta": 1}, {"delta": 1, "slack": -1},
                {"delta": 1, "slack": 0.2, "x": 1}, {"delta": True, "slack": 0.2}, "yes", 3):  # fmt: skip
        with pytest.raises(ValueError, match="window_boost"):
            window.settings(bad)
