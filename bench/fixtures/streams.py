"""Hand-written fixtures and op streams for the P1 T0 gates (benchmarks-v2 §4, gate dependencies).

An op is {"now": ISO UTC op date, "tool": ..., "args": {...}}. Nothing here is random.
"""

from __future__ import annotations

from typing import Any

Op = dict[str, Any]


def _ts(hour: int, day: int = 1) -> str:
    return f"2026-10-{day:02d}T{hour:02d}:00:00Z"


def _rem(now: str, text: str, type_: str, about: list[str], **extra: Any) -> Op:
    return {"now": now, "tool": "remember",
            "args": {"text": text, "type": type_, "about": about, **extra}}  # fmt: skip


def _recall(now: str, query: str, **extra: Any) -> Op:
    return {"now": now, "tool": "recall", "args": {"query": query, **extra}}


# Exercises P1 features only: new: subjects, exact-name lookup, exact duplicate, future-dated
# and date-only `as_of`, recall. (Supersession and needs_choice arrive in P3.)
OP_STREAM: list[Op] = [
    _rem(_ts(8), "Ana lives in Lisbon", "fact", ["new:person:Ana"]),
    _rem(_ts(9), "Atlas kickoff is planned for March", "event", ["new:project:Atlas"]),
    _rem(_ts(10), "Ana prefers green tea in the afternoon", "preference", ["Ana"]),
    _recall(_ts(11), "where does Ana live"),
    _rem(_ts(12), "Ana lives in Lisbon", "fact", ["Ana"]),  # exact duplicate -> exists
    _rem(
        _ts(13),
        "Atlas budget review moves to January",
        "decision",
        ["Atlas", "new:org:Acme"],
        as_of="2026-12-01T00:00:00Z",
    ),  # future-dated
    _rem(
        _ts(14), "Ana started the Lisbon office on this day", "event", ["Ana"], as_of="2026-09-15"
    ),  # date-only, zone-normalised
    _recall(_ts(15), "Atlas budget", include_history=True),
    _rem(_ts(16), "Acme sells ceramic filters", "fact", ["Acme"]),
]

# Run unchanged against the personal and the work store (B-W6 item 6).
TWO_STORE_STREAM: list[Op] = [
    _rem(_ts(8), "Ana runs the Lisbon standup", "fact", ["new:person:Ana"]),
    _rem(_ts(9), "Atlas release train leaves on Thursday", "fact", ["new:project:Atlas"]),
]

# B-R5 (a): the November example (benchmarks-v2 B-R5). Intervals are written by the fixture in
# P1 (valid_to set directly); from P3 `link` derives them.
NOV_M302 = {"text": "Ana's standing one-on-one is on Monday", "type": "fact",
            "from": "2026-06-03T00:00:00Z", "to": "2026-11-01T00:00:00Z"}  # fmt: skip
NOV_M451 = {"text": "Ana's standing one-on-one is on Thursday", "type": "fact",
            "from": "2026-11-01T00:00:00Z", "to": None}  # fmt: skip
NOV_QUERY = "Ana standing one-on-one"
# (as_of, ids expected). 2026-10-22 and 2026-12-01 in the issue text; the boundary is half-open.
NOV_CASES: list[tuple[str, set[str]]] = [
    ("2026-10-22T09:00:00Z", {"M302"}),
    ("2026-10-31T23:59:59Z", {"M302"}),
    ("2026-11-01T00:00:00Z", {"M451"}),
    ("2026-12-01T00:00:00Z", {"M451"}),
]
# B-R5 (e): date-only input is local midnight in the configured zone, converted to UTC.
# Europe/Amsterdam is CET (UTC+1) on 2026-11-01 and CEST (UTC+2) on 2026-06-03.
DATE_ONLY_CASES: list[tuple[str, str, str]] = [
    # (date-only input, expected UTC instant)
    ("2026-11-01", "2026-10-31T23:00:00Z"),
    ("2026-06-03", "2026-06-02T22:00:00Z"),
]

# B-R6 / B-C1 / B-C2 corpus: topics are disjoint enough that each query has a clear top hit.
CORPUS: list[tuple[str, str, list[str]]] = [
    ("Ana lives in Lisbon and works remotely", "fact", ["new:person:Ana"]),
    ("Ana prefers green tea over coffee", "preference", ["Ana"]),
    ("Atlas uses a monorepo with weekly releases", "fact", ["new:project:Atlas"]),
    (
        "Decided to run one memory server and never sync the database",
        "decision",
        ["new:topic:Memory"],
    ),
    ("The garden needs pruning before the first frost", "fact", ["new:topic:Garden"]),
    ("Invoice 4471 from Acme is due in November", "event", ["new:org:Acme"]),
    ("Kubernetes cluster upgrade is scheduled for the spring", "event", ["new:topic:Infra"]),
    ("Bosco the dog is allergic to chicken", "fact", ["new:thing:Bosco"]),
    ("Sourdough starter needs feeding twice a day", "procedure", ["new:topic:Baking"]),
    ("Quarterly budget review happens on the first Monday", "fact", ["Atlas"]),
]
CORPUS_QUERIES = [
    "where does Ana live", "green tea", "monorepo releases", "memory server database sync",
    "garden frost", "Acme invoice", "kubernetes upgrade", "dog allergy",
]  # fmt: skip
NO_MATCH_QUERY = "zzzxqv unmatched gibberish"


# B-R6: one 3,000-token code block, verbatim (est. tokens = ceil(bytes / 3) -> ~9,000 bytes),
# with tabs, trailing spaces, non-ASCII and no trailing newline, so cuts cannot normalise it.
def code_block() -> str:
    lines = ["def f(x):  # ünïcode ✓"]
    i = 0
    while sum(len(s.encode()) + 1 for s in lines) < 9000:
        lines.append(f"\tvalue_{i} = x * {i} + {i % 7}  ")
        i += 1
    return "\n".join(lines)


EDGE_ENTITY = "new:topic:EdgeHub"
EDGE_COUNT = 25
