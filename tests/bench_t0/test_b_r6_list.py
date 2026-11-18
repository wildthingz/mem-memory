"""The P2 form of B-R6 on list-mode responses (MEM2-259, ADR 0023 "Response grammar", ADR 0024 header
tail): a list response is checked against the list grammar, a normal one exactly as before, and the
record counts violations per grammar. T0: hand-built texts and the server's own list/paged output on
the hash models."""

from __future__ import annotations

import pytest

from bench.lme_eval import Item, Retrieved
from bench.suites import b_p2_cost, b_r6
from bench.suites.base import SuiteResult
from mem import listmode, read
from tests.test_list_mode import Q, _store, _sub
from tests.test_list_paging import _paged, _walk

INS = listmode.INSTRUCTION
HEAD = "3 results (of 30 checked) · personal · list mode · 2 sessions"
G1 = "2023-08-11 · chat session 2023-08-11"
G2 = "2023-08-20 · chat session 2023-08-20"
R1 = "  c175 · chat_session · 2023-08-11 · chat session 2023-08-11 · .77"
R2 = "  c176 · chat_session · 2023-08-11 · chat session 2023-08-11 · dates: 2023-08-05 (last Saturday) · .61"
R3 = "  c201 · chat_session · 2023-08-20 · chat session 2023-08-20 · .58"
MORE = "+4 more matches above tau_list not shown (token cap 1000); narrow the query"
PAGE1 = " · 3 of 9 shown · more: recall(…, page=2-944f92b292)"
PAGE2 = " · 6 of 9 shown · more: recall(…, page=3-944f92b292)"
LAST = " · 9 of 9 shown · page 3 of 3 · last page"


def lines(*ls: str) -> str:
    return "\n".join(ls)


VALID = {
    "plain": lines(HEAD, INS, G1, R1, R2, G2, R3),
    "over-cap line": lines(HEAD, INS, G1, R1, R2, G2, R3, MORE),
    "page 1 token": lines(HEAD + PAGE1, INS, G1, R1, R2, G2, R3),
    "middle page token": lines(HEAD + PAGE2, INS, G1, R1, R2, G2, R3),
    "last page": lines(HEAD + LAST, INS, G1, R1, R2, G2, R3),
    "one session": lines("1 results (of 30 checked) · work · list mode · 1 session", INS, G1, R1),
    "undated session first": lines(
        HEAD, INS, "- · notes", "  n4 · note · - · notes · .9", G1, R1, R2
    ),
}


@pytest.mark.parametrize("name", list(VALID))
def test_a_valid_list_response_has_no_violations(name):
    text = VALID[name]
    assert b_p2_cost.response_violations(text, False) == [], (name, text)


BAD = {
    "unindented row": (lines(HEAD, INS, G1, R1.strip(), R2, G2, R3), "group header"),
    "row before any group header": (lines(HEAD, INS, R1, G1, R2, G2, R3), "before any group"),
    "groups out of date order": (lines(HEAD, INS, G2, R3, G1, R1, R2), "date order"),
    "missing instruction line": (lines(HEAD, G1, R1, R2, G2, R3), "instruction"),
    "token without its tag": (
        lines(HEAD + " · 3 of 9 shown · more: recall(…, page=2)", INS, G1, R1, R2, G2, R3),
        "list header",
    ),
    "shown above the total": (
        lines(
            HEAD + " · 10 of 9 shown · more: recall(…, page=2-944f92b292)", INS, G1, R1, R2, G2, R3
        ),
        "shown",
    ),
    "continuation as its own line": (
        lines(HEAD, INS, G1, R1, R2, G2, R3, "3 of 9 shown · more: recall(…, page=2-944f92b292)"),
        "group header",
    ),
    "over-cap line with paging": (lines(HEAD + PAGE1, INS, G1, R1, R2, G2, R3, MORE), "paging"),
    "over-cap line not last": (lines(HEAD, INS, G1, R1, MORE, G2, R3), "last line"),
    "bad row under a group": (lines(HEAD, INS, G1, "  c1 · not a row", G2, R3), "row format"),
    "row count disagrees with the header": (lines(HEAD, INS, G1, R1, G2, R3), "header says 3"),
    "session count disagrees with the header": (lines(HEAD, INS, G1, R1, R2, R3), "sessions"),
    "empty group": (lines(HEAD, INS, G1, G2, R1, R2, R3), "no rows"),
    "row over 12 title words": (
        lines(
            "1 results (of 30 checked) · personal · list mode · 1 session",
            INS,
            G1,
            "  c1 · note · 2023-08-11 · " + "w " * 13 + "· .5",
        ),
        "12 words",
    ),
    "over the list cap": (
        lines("60 results (of 90 checked) · personal · list mode · 1 session", INS, G1, *[R1] * 60),
        f"> list cap {listmode.CAP_TOKENS}",
    ),
}


@pytest.mark.parametrize("name", list(BAD))
def test_a_malformed_list_response_fails(name):
    text, needle = BAD[name]
    v = b_p2_cost.response_violations(text, False)
    assert any(needle in x for x in v), (name, v)


def test_the_list_grammar_is_chosen_by_the_instruction_line_or_the_list_header():
    assert b_p2_cost.is_list_text(VALID["plain"])
    assert b_p2_cost.is_list_text(BAD["missing instruction line"][0])  # header says list mode
    assert not b_p2_cost.is_list_text(
        "2 results (of 30 checked) · personal · calibrated\nc1 · n · - · t · .5"
    )


NORMAL = [
    ("2 results (of 30 checked) · personal · calibrated\nc1 · note · 2023-01-01 · t · about: A, B · .5\nc2 · fact · - · t · .25", False),
    ("1 results (of 30 checked) · personal · calibrated\nnot a row", False),
    ("1 results (of 30 checked) · personal · calibrated\n  c1 · note · 2023-01-01 · t · .5", False),
    ("2 results (of 30 checked) · personal · uncalibrated (P1 build)\nc1 · note · 2023-01-01 · t · .5", False),
    ("no relevant memories (best .22) · personal", True),
    ("no results · personal", True),
]  # fmt: skip


@pytest.mark.parametrize("text,abstained", NORMAL)
def test_a_normal_response_is_checked_exactly_as_before(text, abstained):
    assert b_p2_cost.response_violations(text, abstained) == b_p2_cost.normal_violations(
        text, abstained
    )
    # an indented row is still a row-format violation in a normal response
    if text.endswith("  c1 · note · 2023-01-01 · t · .5"):
        assert any("row format" in v for v in b_p2_cost.response_violations(text, abstained))


# ---- the server's own output (hash models): every list and paged response passes -------------


def test_the_servers_list_responses_pass(tmp_path):
    s = _store(_sub(tmp_path, "a"), list_mode=True)
    for budget in (None, 120, 200):
        out = read.recall(s, query=Q, budget=budget)
        assert b_p2_cost.is_list_text(out.text)
        assert b_p2_cost.response_violations(out.text, False) == [], out.text
    assert "more matches above tau_list" in read.recall(s, query=Q, budget=120).text
    s.close()
    p = _paged(_sub(tmp_path, "b"))
    pages = _walk(p)
    assert len(pages) > 2  # first, middle and last page forms
    for out in pages:
        assert b_p2_cost.response_violations(out.text, False) == [], out.text
    p.close()


# ---- the record: violations per grammar ------------------------------------------------------


def _item(qid, text, kind="answerable"):
    return Item(qid, "multi-session", kind, frozenset(), frozenset(),
                Retrieved(F=[], R=[], S=["a#0"], text=text))  # fmt: skip


def test_b_r6_reports_violations_per_grammar(monkeypatch):
    items = [
        _item("ok-list", VALID["over-cap line"]),
        _item("bad-list", BAD["unindented row"][0]),
        _item("ok-normal", NORMAL[0][0]),
        _item("bad-normal", NORMAL[1][0]),
        _item("abs", "whatever", kind="abs"),
    ]
    monkeypatch.setattr(b_p2_cost, "shared_run", lambda ctx: (None, items, None))
    for f in ("_check_empty", "_check_get", "_check_edges"):
        monkeypatch.setattr(b_r6, f, lambda res, ctx: None)
    res: SuiteResult = b_p2_cost.run_b_r6(None)
    m = res.metrics
    assert m["responses_checked"] == 4 and m["responses_list"] == 2 and m["responses_normal"] == 2
    assert m["violations_list"] >= 1 and m["violations_normal"] == 1
    assert (
        m["violations"] == m["violations_list"] + m["violations_normal"] + m["violations_fixture"]
    )
    assert m["violations_fixture"] == 0
    assert all(
        v.startswith(("answerable:bad-list:list", "answerable:bad-normal:normal"))
        for v in res.violations
    )
