"""H3 (MEM2-77): LongMemEval_M re-dates the same session per question (measured on the real M
file: 237,655 occurrences, 51,661 unique ids, 47,077 ids with more than one date, 0 with
different content). Rule, for `plan_corpus` and the streaming `plan_corpus_file` alike: dedupe
by session id, content must match (a conflict raises, naming the id), the EARLIEST date wins."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from bench.datasets import corpora, lme

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
M_MINI = FIX / "lme_m_mini.json"

DATES = ["2023/05/20 (Sat) 02:21", "2023/03/02 (Thu) 09:00", "2023/05/19 (Fri) 23:59"]
EARLIEST = "2023-03-02T09:00:00Z"
TURNS = [
    {"role": "user", "content": "I adopted a cat called Miso."},
    {"role": "assistant", "content": "Congratulations on Miso!"},
    {"role": "user", "content": "She is three."},
    {"role": "assistant", "content": "A young cat."},
]


def _question(qid: str, sessions: list[tuple[str, str, list[dict]]]) -> dict:
    return {
        "question_id": qid, "question_type": "single-session-user", "question": "q",
        "answer": "a", "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": [s for s, _, _ in sessions],
        "haystack_dates": [d for _, d, _ in sessions],
        "haystack_sessions": [t for _, _, t in sessions],
        "answer_session_ids": [sessions[0][0]],
    }  # fmt: skip


def _redated(dates: list[str]) -> list[dict]:
    """`shared` once per question under a different date, plus one unique filler per question."""
    return [
        _question(f"q{i}", [("shared", d, TURNS), (f"z{i}", "2023/04/01 (Sat) 08:00", TURNS[:2])])
        for i, d in enumerate(dates)
    ]


def _write(tmp_path: Path, qs: list[dict]) -> Path:
    f = tmp_path / "m.json"
    f.write_text(json.dumps(qs))
    return f


@pytest.mark.parametrize("dates", [DATES[:2], DATES[:2][::-1], *itertools.permutations(DATES)])
def test_redated_session_is_one_session_with_the_earliest_date(tmp_path, dates):
    qs = _redated(list(dates))
    n = 2 + len(dates)  # all of `shared` plus every filler: no truncation
    plan = corpora.plan_corpus(qs, n)
    streamed = corpora.plan_corpus_file(_write(tmp_path, qs), n)
    assert streamed == plan
    assert corpora.corpus_fingerprint(streamed) == corpora.corpus_fingerprint(plan)
    shared = [p for p in plan if p.session_id == "shared"]
    assert len(shared) == 1
    assert shared[0].date == EARLIEST
    assert len(shared[0].rounds) == 2 and not shared[0].truncated
    assert [p.session_id for p in plan] == sorted(p.session_id for p in plan)


def test_fingerprint_is_independent_of_the_order_the_dates_are_seen_in(tmp_path):
    prints = set()
    for dates in itertools.permutations(DATES):
        qs = _redated(list(dates))
        prints.add(corpora.corpus_fingerprint(corpora.plan_corpus(qs, 5)))
        prints.add(corpora.corpus_fingerprint(corpora.plan_corpus_file(_write(tmp_path, qs), 5)))
    assert len(prints) == 1


def test_collect_sessions_keeps_the_earliest_date_and_the_content():
    sessions = corpora.collect_sessions(_redated(DATES))
    assert sessions["shared"] == (EARLIEST, TURNS)
    assert len(sessions) == 1 + len(DATES)


@pytest.mark.parametrize("same_date", [True, False])
def test_content_conflict_for_the_same_id_still_raises_naming_the_id(tmp_path, same_date):
    qs = _redated(DATES[:2])
    if same_date:
        qs[1]["haystack_dates"][0] = qs[0]["haystack_dates"][0]
    qs[1]["haystack_sessions"][0] = [*TURNS[:3], {"role": "assistant", "content": "An old cat."}]
    with pytest.raises(ValueError, match="session shared .*content"):
        corpora.plan_corpus(qs, 5)
    with pytest.raises(ValueError, match="session shared .*content"):
        corpora.plan_corpus_file(_write(tmp_path, qs), 5)


def test_mini_fixture_redated_in_place_matches_the_untouched_plan_except_dates(tmp_path):
    """Re-date every repeated occurrence in the real-layout fixture (as LongMemEval_M does) to a
    later time: the plan keeps the original (earliest) dates and the fingerprint is unchanged."""
    m = json.loads(M_MINI.read_text())
    base = corpora.plan_corpus(lme.load_questions(M_MINI), 20)
    seen: set[str] = set()
    redated = 0
    for q in m:
        for si, sid in enumerate(q["haystack_session_ids"]):
            if sid in seen:
                q["haystack_dates"][si] = "2024/12/31 (Tue) 23:59"
                redated += 1
            seen.add(sid)
    assert redated > 0, "fixture has no session shared by two haystacks"
    f = _write(tmp_path, m)
    assert corpora.plan_corpus(m, 20) == base
    assert corpora.plan_corpus_file(f, 20) == base
    assert corpora.corpus_fingerprint(
        corpora.plan_corpus_file(f, 20)
    ) == corpora.corpus_fingerprint(base)
    # and re-dating the FIRST occurrence to an earlier time moves the plan's date to it
    m2 = json.loads(M_MINI.read_text())
    first = {}
    for q in m2:
        for si, sid in enumerate(q["haystack_session_ids"]):
            first.setdefault(sid, (q, si))
    dup = next(sid for sid in first if sum(sid in q["haystack_session_ids"] for q in m2) > 1)
    q, si = first[dup]
    q["haystack_dates"][si] = "2000/01/01 (Sat) 00:00"
    got = {p.session_id: p.date for p in corpora.plan_corpus_file(_write(tmp_path, m2), 20)}
    if dup in got:
        assert got[dup] == "2000-01-01T00:00:00Z"
