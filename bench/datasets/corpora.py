"""L5k / L50k load corpora from LongMemEval_M (benchmarks-v2 B-C).

L<N> = the first N chunks of LME_M sessions ingested with `ingest_session`, sessions taken in
sorted session-id order (deduplicated by id across haystacks). A chunk is one round, so N chunks
= N rounds; the session that crosses N is truncated to the rounds that fit (stated, not hidden:
`PlannedSession.truncated`). The plan is pure data; `corpus_fingerprint` pins it.

Dedupe rule (MEM2-77 H3): LongMemEval_M re-dates the same session per question (real M file:
237,655 occurrences, 51,661 ids, 47,077 ids under more than one date, 0 with different content).
A session id seen more than once must have identical content (a conflict raises, naming the id)
and gets the EARLIEST of the dates seen. Deterministic; the corpus only measures latency/scale.

`plan_corpus` takes questions in memory (fixtures, the S file). `plan_corpus_file` gives the same
plan from the LongMemEval_M file (2.7 GB; a whole-file load needs > 10 GB) in two streamed passes:
pass 1 keeps per session id only (date, content digest, round count), pass 2 the chosen rounds.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.datasets.lme import Round, iter_questions, parse_lme_date, rounds_of

SIZES = {"L5k": 5_000, "L50k": 50_000}


@dataclass(frozen=True)
class PlannedSession:
    session_id: str
    date: str  # ISO UTC haystack date
    rounds: tuple[Round, ...]
    truncated: bool = False


def _content_conflict(sid: str) -> ValueError:
    return ValueError(f"session {sid} appears with different content under the same id")


def collect_sessions(questions: Iterable[dict[str, Any]]) -> dict[str, tuple[str, list[dict]]]:
    """session_id -> (earliest iso date, turns). A duplicate id must repeat the same turns (else
    an error, not a pick); its date is the earliest seen, whatever the order of the questions."""
    out: dict[str, tuple[str, list[dict]]] = {}
    for q in questions:
        for sid, iso, turns in _occurrences(q):
            if sid in out:
                seen_iso, seen_turns = out[sid]
                if seen_turns != turns:
                    raise _content_conflict(sid)
                iso = min(iso, seen_iso)  # ISO UTC strings: lexical order is time order
            out[sid] = (iso, turns)
    return out


def plan_corpus(questions: Iterable[dict[str, Any]], n_chunks: int) -> list[PlannedSession]:
    sessions = collect_sessions(questions)
    plan, total = [], 0
    for sid in sorted(sessions):
        iso, turns = sessions[sid]
        rounds = rounds_of(turns)
        room = n_chunks - total
        if room <= 0:
            break
        if len(rounds) > room:
            plan.append(PlannedSession(sid, iso, tuple(rounds[:room]), truncated=True))
            total += room
            break
        plan.append(PlannedSession(sid, iso, tuple(rounds)))
        total += len(rounds)
    if total < n_chunks:
        raise ValueError(f"only {total} chunks available, wanted {n_chunks}")
    return plan


def _occurrences(q: dict[str, Any]) -> Iterable[tuple[str, str, list[dict]]]:
    for sid, date, turns in zip(
        q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True
    ):
        yield sid, parse_lme_date(date), turns


def _digest(turns: list[dict]) -> bytes:
    """Stands in for `turns` equality in `collect_sessions`' content check."""
    text = json.dumps(turns, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).digest()


def plan_corpus_file(path: Path, n_chunks: int) -> list[PlannedSession]:
    """`plan_corpus(load_questions(path), n_chunks)` without holding the file: same plan, same
    dedupe rule (same id: identical content, earliest date), same errors (duplicate question id,
    content conflict under one session id, too few chunks)."""
    # pass 1: session id -> (earliest iso date, content digest, round count)
    meta: dict[str, tuple[str, bytes, int]] = {}
    qids: Counter[str] = Counter()
    for q in iter_questions(path):
        qids[q["question_id"]] += 1
        for sid, iso, turns in _occurrences(q):
            d = _digest(turns)
            if sid in meta:
                seen_iso, seen_d, n = meta[sid]
                if seen_d != d:
                    raise _content_conflict(sid)
                meta[sid] = (min(iso, seen_iso), d, n)
                continue
            meta[sid] = (iso, d, len(rounds_of(turns)))
    if dup := [k for k, n in qids.items() if n > 1]:
        raise ValueError(f"{path}: duplicate question ids {dup[:5]}")
    # choose: the rule of `plan_corpus`, on round counts
    take: dict[str, int] = {}  # session id -> rounds kept
    total = 0
    for sid in sorted(meta):
        room = n_chunks - total
        if room <= 0:
            break
        k = min(meta[sid][2], room)
        take[sid] = k
        total += k
        if k == room:
            break
    if total < n_chunks:
        raise ValueError(f"only {total} chunks available, wanted {n_chunks}")
    # pass 2: the chosen sessions' rounds (first occurrence; content is identical by pass 1)
    rounds: dict[str, tuple[Round, ...]] = {}
    for q in iter_questions(path):
        for sid, _, turns in _occurrences(q):
            if sid in take and sid not in rounds:
                rounds[sid] = tuple(rounds_of(turns)[: take[sid]])
        if len(rounds) == len(take):
            break
    return [
        PlannedSession(sid, meta[sid][0], rounds[sid], truncated=k < meta[sid][2])
        for sid, k in take.items()
    ]


def n_chunks(plan: Iterable[PlannedSession]) -> int:
    return sum(len(p.rounds) for p in plan)


def corpus_fingerprint(plan: Iterable[PlannedSession]) -> str:
    """sha256 over (session id, date, rounds) in plan order. The date of a session that
    LongMemEval_M lists under several dates is the EARLIEST one (content must match; a content
    conflict raises instead). Fingerprints from before this rule (MEM2-77 H3) are not comparable;
    no successful L5k/L50k build existed before it, so none is lost."""
    h = hashlib.sha256()
    for p in plan:
        h.update(
            json.dumps(
                [p.session_id, p.date, [(r.user, r.assistant) for r in p.rounds]],
                ensure_ascii=False,
            ).encode()
        )
    return h.hexdigest()


def build_corpus(
    plan: Iterable[PlannedSession], ingest_session: Callable[[str, list[tuple[str, str]], str], Any]
) -> int:
    """Feed the plan to `ingest_session(date, rounds, session_id)`; returns chunks ingested."""
    total = 0
    for p in plan:
        ingest_session(p.date, [(r.user, r.assistant) for r in p.rounds], p.session_id)
        total += len(p.rounds)
    return total
