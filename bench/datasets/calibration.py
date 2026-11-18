"""Tune-split labels for `mem calibrate` (architecture-v2 §3.3 step 1; P2 data = LME tune + negatives).

One JSON line per query: the question, its date, the sessions of the haystack it runs on, and the
chunk labels. Answerable queries run on the full haystack with rel=1 on `has_answer` rounds (rel 2
on them as well: LME evidence is binary); negatives run on the evidence-removed haystack with no
gold. Only the tune split is ever exported, so calibration cannot see test questions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.datasets import lme
from bench.datasets.negatives import NegRow


def tune_label_rows(
    questions: Mapping[str, dict[str, Any]], manifest: Sequence[lme.ManifestRow],
    negs: Sequence[NegRow],
) -> list[dict[str, Any]]:  # fmt: skip
    dropped = {n.question_id for n in negs if n.audit_dropped}
    rows = []
    for kind in ("answerable", "negative"):
        for m in lme.select(manifest, "tune", "br1"):
            if kind == "negative" and m.question_id in dropped:
                continue
            q = questions[m.question_id]
            hay = lme.haystack_of(q, exclude=lme.gold_sessions(q) if kind == "negative" else ())
            rows.append({
                "qid": m.question_id, "kind": kind, "category": m.question_type,
                "question": q["question"], "as_of": lme.question_as_of(q["question_date"]),
                "sessions": hay.session_ids,
                "rel": {r: 1 for r in sorted(lme.gold_rounds(q))} if kind == "answerable" else {},
            })  # fmt: skip
    return rows


def write_tune_labels(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def read_tune_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]  # fmt: skip


def label_haystack(q: dict[str, Any], row: dict[str, Any]) -> lme.Haystack:
    """The haystack a label row was exported with: its question's sessions, restricted to
    `row["sessions"]` (the evidence-removed one for a negative). A mismatch is an error."""
    keep = set(row["sessions"])
    hay = lme.haystack_of(q, exclude=[s for s in q["haystack_session_ids"] if s not in keep])
    if hay.session_ids != list(row["sessions"]):
        raise ValueError(f"{row['qid']} ({row['kind']}): the LME file's haystack differs from the "
                         "exported sessions; export the labels from this file")  # fmt: skip
    return hay


def score_tune(
    rows: Sequence[dict[str, Any]], questions: Mapping[str, dict[str, Any]], backend: Any,
) -> list[Any]:  # fmt: skip
    """Label rows -> `mem.calibrate.QueryPool`s: each query's rerank pool scored by the production
    pipeline (`backend.score`, architecture-v2 §3.3 step 2), rel from the row's chunk labels.

    `qid` in the records is `<qid>:<kind>`, so a question's answerable and negative runs stay
    two queries; `category` is the question type (the per-category tune report)."""
    from mem.calibrate import PoolItem, QueryPool

    pools = []
    for row in rows:
        q = questions[row["qid"]]
        scored = backend.score(q, label_haystack(q, row))
        pools.append(QueryPool(f"{row['qid']}:{row['kind']}", row["kind"] == "answerable",
                               [PoolItem(r, z, int(row["rel"].get(r, 0)), *cos) for r, z, *cos in scored],
                               str(row.get("category", ""))))  # fmt: skip
    return pools
