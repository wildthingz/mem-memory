"""Evidence-removed negatives for B-R2 (i) and the 30-pair audit sheet (benchmarks-v2 §2, A15).

Each B-R1 question is paired with its own haystack minus every `answer_session_id`: the answer is
absent by construction because LME filler sessions come from unrelated sources. A hand audit of 30
sampled pairs checks that; an audited pair that still contains the answer is dropped, and the audit
is reported. The automatic columns only help the auditor; they never decide.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.datasets.lme import ManifestRow, haystack_of, select

AUDIT_N = 30
AUDIT_SEED = 7
NEG_COLUMNS = ("question_id", "split", "n_sessions", "n_removed", "n_remaining", "removed_ids",
               "auto_hit_sessions", "audit_dropped")  # fmt: skip
AUDIT_COLUMNS = ("question_id", "split", "question", "answer", "auto_hit_sessions",
                 "still_contains_answer", "notes")  # fmt: skip
MIN_ANSWER_LEN = 4  # shorter answers ("5", "no") match everything; the human audit covers them


def _norm(s: str) -> str:
    return " ".join(re.sub(r"\W+", " ", unicodedata.normalize("NFKC", s).casefold()).split())


def auto_hits(q: dict[str, Any]) -> list[str]:
    """Remaining sessions whose text contains the normalised answer string (a hint, not a verdict)."""
    ans = _norm(str(q["answer"]))
    if len(ans) < MIN_ANSWER_LEN:
        return []
    gold = set(q["answer_session_ids"])
    hits = []
    for sid, turns in zip(q["haystack_session_ids"], q["haystack_sessions"], strict=True):
        if sid not in gold and ans in _norm(" ".join(t["content"] for t in turns)):
            hits.append(sid)
    return sorted(set(hits))


@dataclass(frozen=True)
class NegRow:
    question_id: str
    split: str
    n_sessions: int
    n_removed: int
    n_remaining: int
    removed_ids: tuple[str, ...]
    auto_hit_sessions: tuple[str, ...]
    audit_dropped: bool = False


def build_negatives(
    questions: Sequence[dict[str, Any]], manifest: Sequence[ManifestRow]
) -> list[NegRow]:
    by_id = {q["question_id"]: q for q in questions}
    rows = []
    for split in ("tune", "test"):
        for m in select(manifest, split, "br1"):
            q = by_id[m.question_id]
            full = haystack_of(q)
            gold = set(q["answer_session_ids"])
            rest = haystack_of(q, exclude=gold)
            removed = tuple(sorted(gold & set(full.session_ids)))
            if not removed:
                raise ValueError(f"{m.question_id}: no answer session present in its haystack")
            rows.append(NegRow(m.question_id, split, len(full.sessions), len(removed),
                               len(rest.sessions), removed, tuple(auto_hits(q))))  # fmt: skip
    return rows


def audit_sample(rows: Sequence[NegRow], n: int = AUDIT_N, seed: int = AUDIT_SEED) -> list[NegRow]:
    """Deterministic: order by sha1('audit<seed>:<qid>'), take the first n."""
    key = lambda r: hashlib.sha1(f"audit{seed}:{r.question_id}".encode()).hexdigest()
    return sorted(sorted(rows, key=key)[:n], key=lambda r: r.question_id)


def _write(path: Path, cols: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)


def write_negatives(rows: Sequence[NegRow], path: Path) -> None:
    _write(path, NEG_COLUMNS, ([r.question_id, r.split, r.n_sessions, r.n_removed, r.n_remaining,
                                ",".join(r.removed_ids), ",".join(r.auto_hit_sessions),
                                int(r.audit_dropped)] for r in rows))  # fmt: skip


def read_negatives(path: Path) -> list[NegRow]:
    with open(path, encoding="utf-8", newline="") as f:
        return [NegRow(r["question_id"], r["split"], int(r["n_sessions"]), int(r["n_removed"]),
                       int(r["n_remaining"]), tuple(filter(None, r["removed_ids"].split(","))),
                       tuple(filter(None, r["auto_hit_sessions"].split(","))),
                       r["audit_dropped"] == "1")
                for r in csv.DictReader(f, delimiter="\t")]  # fmt: skip


def write_audit_sheet(
    sample: Sequence[NegRow], questions: Sequence[dict[str, Any]], path: Path
) -> None:
    by_id = {q["question_id"]: q for q in questions}
    _write(path, AUDIT_COLUMNS, ([r.question_id, r.split, by_id[r.question_id]["question"],
                                  by_id[r.question_id]["answer"], ",".join(r.auto_hit_sessions),
                                  "", ""] for r in sample))  # fmt: skip


def apply_audit(rows: Sequence[NegRow], sheet: Path) -> tuple[list[NegRow], dict[str, int]]:
    """Read the filled sheet (`still_contains_answer` = y/n for every row) and mark dropped pairs."""
    with open(sheet, encoding="utf-8", newline="") as f:
        audited = {r["question_id"]: r["still_contains_answer"].strip().lower()
                   for r in csv.DictReader(f, delimiter="\t")}  # fmt: skip
    bad = {k: v for k, v in audited.items() if v not in ("y", "n")}
    if bad:
        raise ValueError(f"audit sheet incomplete or invalid (need y/n): {sorted(bad)[:5]}")
    unknown = set(audited) - {r.question_id for r in rows}
    if unknown:
        raise ValueError(f"audit sheet has ids not in the negatives: {sorted(unknown)[:5]}")
    out = [
        NegRow(**{**r.__dict__, "audit_dropped": audited.get(r.question_id) == "y"}) for r in rows
    ]
    return out, {"audited": len(audited), "dropped": sum(v == "y" for v in audited.values()),
                 "pairs_after": sum(not r.audit_dropped for r in out)}  # fmt: skip
