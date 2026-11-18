"""LongMemEval official file layout, round mapping and the frozen manifest (benchmarks-v2 §1, A15).

Official entry (`longmemeval_s_cleaned.json`, a JSON list):
  question_id, question_type, question, answer, question_date, haystack_session_ids,
  haystack_dates, haystack_sessions (list of sessions; a session is a list of
  {role, content, has_answer?}), answer_session_ids.
`_abs` questions are those whose id ends in `_abs`: the answer is absent by design.

Definitions fixed here (decision record: bench/datasets/README.md):
- Round: starts at each `user` turn and takes the assistant turn(s) that follow it. Assistant turns
  before the first user turn form a round of their own (index 0). A round is relevant iff any of
  its turns has `has_answer: true` (A6 mapping).
- Chunk ref: `<session_id>#<round_index>`; `session_of(ref)` strips the round.
- Manifest: stratify by (question_type, is_abs); in each stratum sort by sha1(question_id) hex,
  the first round-half-up(0.3 n) go to tune, the rest to test.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TYPES = (
    "knowledge-update", "temporal-reasoning", "multi-session", "single-session-user",
    "single-session-assistant", "single-session-preference",
)  # fmt: skip
CORE = ("knowledge-update", "temporal-reasoning", "multi-session")  # C-ans types
CONTROL = "single-session-user"
TUNE_FRACTION_PERCENT = 30
MANIFEST_COLUMNS = ("question_id", "question_type", "is_abs", "split", "rank")


def is_abs(question_id: str) -> bool:
    return question_id.endswith("_abs")


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path}: expected a JSON list of questions")
    ids = [q["question_id"] for q in data]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"{path}: duplicate question ids {dup[:5]}")
    return data


def iter_questions(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the file's questions one at a time (ijson): memory is one question, not the file.
    For LongMemEval_M (2.7 GB, about 5 MB per question); duplicate ids are the caller's check."""
    import ijson

    with open(path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_lme_date(s: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00Z' (the haystack date, UTC)."""
    day, _, rest = s.partition(" ")
    hhmm = rest.split(") ", 1)[1] if ") " in rest else "00:00"
    return datetime.strptime(f"{day} {hhmm}", "%Y/%m/%d %H:%M").strftime("%Y-%m-%dT%H:%M:00Z")


def question_as_of(question_date: str) -> str:
    """The `as_of` every LME read runs at: the question's DAY at 23:59:59Z, not its time of day
    (ADR 0028, MEM2-265). '2023/05/20 (Sat) 02:21' -> '2023-05-20T23:59:59Z'. Some gold sessions
    are stamped later on the question's day than the question itself (44/500 LME_S questions,
    none on a later day); upstream LongMemEval does not time-filter at all."""
    day = datetime.strptime(question_date.partition(" ")[0], "%Y/%m/%d")
    return day.strftime("%Y-%m-%dT23:59:59Z")


@dataclass(frozen=True)
class Round:
    index: int
    user: str
    assistant: str
    has_answer: bool


def rounds_of(session: Sequence[dict[str, Any]]) -> list[Round]:
    out: list[list[Any]] = []  # [user, assistant, has_answer]
    for turn in session:
        flag = bool(turn.get("has_answer"))
        if turn["role"] == "user" or not out:
            out.append(["", "", False])
            if turn["role"] == "user":
                out[-1][0] = turn["content"]
            else:
                out[-1][1] = turn["content"]
        else:
            out[-1][1] += ("\n" if out[-1][1] else "") + turn["content"]
        out[-1][2] = out[-1][2] or flag
    return [Round(i, u, a, h) for i, (u, a, h) in enumerate(out)]


def ref(session_id: str, round_index: int) -> str:
    return f"{session_id}#{round_index}"


def session_of(chunk_ref: str) -> str:
    return chunk_ref.rsplit("#", 1)[0]


def dedupe_sessions(refs: Iterable[str]) -> list[str]:
    """Session order of a chunk ranking: each session once, at the rank of its best chunk."""
    seen: dict[str, None] = {}
    for r in refs:
        seen.setdefault(session_of(r), None)
    return list(seen)


@dataclass(frozen=True)
class Haystack:
    """One question's sessions, deduplicated by session id (first occurrence wins)."""

    sessions: tuple[tuple[str, str, tuple[Round, ...]], ...]  # (session_id, iso date, rounds)
    dropped_duplicates: int

    @property
    def session_ids(self) -> list[str]:
        return [s[0] for s in self.sessions]


def haystack_of(q: dict[str, Any], *, exclude: Iterable[str] = ()) -> Haystack:
    skip, seen, kept, dups = set(exclude), set(), [], 0
    for sid, date, turns in zip(
        q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True
    ):
        if sid in skip:
            continue
        if sid in seen:
            dups += 1
            continue
        seen.add(sid)
        kept.append((sid, parse_lme_date(date), tuple(rounds_of(turns))))
    return Haystack(tuple(kept), dups)


def gold_sessions(q: dict[str, Any]) -> set[str]:
    return set(q["answer_session_ids"])


def gold_rounds(q: dict[str, Any]) -> set[str]:
    """Chunk refs of rounds that contain a `has_answer` turn (benchmarks-v2 B-R1 step 4)."""
    out = set()
    for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"], strict=True):
        out |= {ref(sid, r.index) for r in rounds_of(turns) if r.has_answer}
    return out


# ---- manifest --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    question_id: str
    question_type: str
    is_abs: bool
    split: str
    rank: int  # position in the stratum's sha1 order


def n_tune(n: int) -> int:
    """round(0.3 n), half up, in integers (no float ties)."""
    return (TUNE_FRACTION_PERCENT * n + 50) // 100


def build_manifest(questions: Iterable[dict[str, Any]]) -> list[ManifestRow]:
    strata: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for q in questions:
        strata[(q["question_type"], is_abs(q["question_id"]))].append(q["question_id"])
    rows = []
    for (qtype, abs_), ids in sorted(strata.items()):
        ordered = sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())
        cut = n_tune(len(ordered))
        rows += [
            ManifestRow(i, qtype, abs_, "tune" if k < cut else "test", k)
            for k, i in enumerate(ordered)
        ]
    return sorted(rows, key=lambda r: (r.question_type, r.is_abs, r.rank))


def manifest_text(rows: Sequence[ManifestRow]) -> str:
    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines += [
        f"{r.question_id}\t{r.question_type}\t{int(r.is_abs)}\t{r.split}\t{r.rank}" for r in rows
    ]
    return "\n".join(lines) + "\n"


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> str:
    text = manifest_text(rows)
    Path(path).write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_manifest(path: Path) -> list[ManifestRow]:
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t")
        if tuple(rd.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: columns {rd.fieldnames} != {MANIFEST_COLUMNS}")
        return [ManifestRow(r["question_id"], r["question_type"], r["is_abs"] == "1",
                            r["split"], int(r["rank"])) for r in rd]  # fmt: skip


def verify_manifest(questions: Iterable[dict[str, Any]], path: Path) -> None:
    """The frozen file must equal what the rule gives on this data file (no silent drift)."""
    frozen = read_manifest(path)
    fresh = build_manifest(questions)
    if frozen != fresh:
        a, b = {r.question_id: r for r in frozen}, {r.question_id: r for r in fresh}
        diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        raise ValueError(
            f"{path} differs from the rule on this data in {len(diff)} ids: {diff[:5]}"
        )


def select(rows: Sequence[ManifestRow], split: str, which: str) -> list[ManifestRow]:
    """which: 'br1' (C-ans + control answerable), 'c_ans', 'abs', 'all'."""
    pick = {
        "br1": lambda r: not r.is_abs and (r.question_type in CORE or r.question_type == CONTROL),
        "c_ans": lambda r: not r.is_abs and r.question_type in CORE,
        "abs": lambda r: r.is_abs,
        "all": lambda r: True,
    }[which]
    return [r for r in rows if r.split == split and pick(r)]


def manifest_counts(rows: Sequence[ManifestRow]) -> dict[str, int]:
    out = {"total": len(rows)}
    for split in ("tune", "test"):
        out[split] = sum(r.split == split for r in rows)
        for which in ("c_ans", "br1", "abs"):
            out[f"{which}_{split}"] = len(select(rows, split, which))
    return out
