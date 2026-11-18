"""ADR 0021: `mem.reldates` resolves relative time expressions against a reference date, locally and
deterministically; what it cannot resolve for certain it leaves alone."""

from datetime import date

import pytest

from mem import reldates

# 2023-03-15 is a Wednesday.
WED = date(2023, 3, 15)


def _pairs(text, ref):
    return reldates.resolve(text, ref)


@pytest.mark.parametrize(
    ("text", "ref", "phrase", "value"),
    [
        ("I went hiking yesterday.", WED, "yesterday", "2023-03-14"),
        ("Today is busy.", WED, "Today", "2023-03-15"),
        ("See you tomorrow", WED, "tomorrow", "2023-03-16"),
        ("the day before yesterday", WED, "the day before yesterday", "2023-03-13"),
        ("the day after tomorrow", WED, "the day after tomorrow", "2023-03-17"),
        ("two days ago", WED, "two days ago", "2023-03-13"),
        ("3 days ago", WED, "3 days ago", "2023-03-12"),
        ("a day ago", WED, "a day ago", "2023-03-14"),
        ("two weeks ago", WED, "two weeks ago", "2023-03-01"),
        ("a month ago", WED, "a month ago", "2023-02-15"),
        ("a year ago", WED, "a year ago", "2022-03-15"),
        ("this morning", WED, "this morning", "2023-03-15"),
        ("tonight", WED, "tonight", "2023-03-15"),
        ("last night", WED, "last night", "2023-03-14"),
        ("last week", WED, "last week", "2023-03-06..2023-03-12"),
        ("this week", WED, "this week", "2023-03-13..2023-03-19"),
        ("next week", WED, "next week", "2023-03-20..2023-03-26"),
        ("last month", WED, "last month", "2023-02"),
        ("next month", WED, "next month", "2023-04"),
        ("last year", WED, "last year", "2022"),
        ("next year", WED, "next year", "2024"),
        ("last Tuesday", WED, "last Tuesday", "2023-03-14"),
        ("LAST FRIDAY", WED, "LAST FRIDAY", "2023-03-10"),
        ("next Monday", WED, "next Monday", "2023-03-20"),
        ("next Wednesday", WED, "next Wednesday", "2023-03-22"),
        # month, year and leap edges
        ("yesterday", date(2023, 3, 1), "yesterday", "2023-02-28"),
        ("yesterday", date(2024, 1, 1), "yesterday", "2023-12-31"),
        ("tomorrow", date(2023, 12, 31), "tomorrow", "2024-01-01"),
        ("tomorrow", date(2024, 2, 28), "tomorrow", "2024-02-29"),
        ("next month", date(2023, 12, 20), "next month", "2024-01"),
        ("last month", date(2023, 1, 10), "last month", "2022-12"),
        ("a month ago", date(2023, 3, 31), "a month ago", "2023-02-28"),
        ("a year ago", date(2024, 2, 29), "a year ago", "2023-02-28"),
        ("last week", date(2023, 1, 2), "last week", "2022-12-26..2023-01-01"),
        ("last year", date(2024, 2, 29), "last year", "2023"),
        # a Sunday session: "last Sunday" is a week back, never today
        ("last Sunday", date(2023, 3, 12), "last Sunday", "2023-03-05"),
        ("next Sunday", date(2023, 3, 12), "next Sunday", "2023-03-19"),
        ("last week", date(2023, 3, 12), "last week", "2023-02-27..2023-03-05"),
        ("this week", date(2023, 3, 12), "this week", "2023-03-06..2023-03-12"),
        # bare weekday: only when the sentence's tense is plain
        ("I met her on Tuesday.", WED, "Tuesday", "2023-03-14"),
        ("We were at the lake on Wednesday.", WED, "Wednesday", "2023-03-08"),
        ("I will call her on Friday.", WED, "Friday", "2023-03-17"),
        ("The dentist is scheduled for Wednesday.", WED, "Wednesday", "2023-03-22"),
    ],
)
def test_resolves(text, ref, phrase, value):
    assert _pairs(text, ref) == [(phrase, value)]


@pytest.mark.parametrize(
    "text",
    [
        "recently",
        "a while ago",
        "a few days ago",
        "a couple of weeks ago",
        "the other day",
        "soon",
        "in two weeks",
        "last weekend",
        "this Friday",
        "every Tuesday",
        "on Tuesdays",
        "Tuesday",
        "I met her on Tuesday and will see her on Tuesday.",  # tense unclear
        "on the 5th",
        "in March",
        "yesterdays",
        "someday",
        "many days ago",
        "2 fortnights ago",
        "two decades ago",
        "todays",
    ],
)
def test_ambiguous_or_unresolvable_gets_nothing(text):
    assert _pairs(text, WED) == []


def test_several_expressions_in_order_without_repeats():
    text = "Yesterday I ran; two days ago I swam; yesterday again; last Tuesday too."
    assert _pairs(text, WED) == [
        ("Yesterday", "2023-03-14"),
        ("two days ago", "2023-03-13"),
        ("last Tuesday", "2023-03-14"),
    ]


def test_last_weekday_does_not_also_resolve_the_bare_name():
    assert _pairs("It was last Tuesday.", WED) == [("last Tuesday", "2023-03-14")]


def test_annotation_format_and_empty():
    assert reldates.annotation("yesterday and last month", WED) == (
        "yesterday [= 2023-03-14]; last month [= 2023-02]"
    )
    assert reldates.annotation("nothing relative here", WED) is None


def test_annotation_is_capped():
    text = " ".join(f"{n} days ago" for n in range(1, 40))
    assert len(_pairs(text, WED)) == reldates.MAX_ANNOTATIONS
