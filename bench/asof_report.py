"""MEM2-265: which LME tune questions gain eligible gold when `as_of` is the question's day end.

Reads only the dataset's dates and gold session ids and the frozen manifest's tune rows: no run
output, no response, nothing of the test split beyond the manifest's split column. Before =
`as_of` at the question's time of day (`parse_lme_date`, the old harness); after = the day at
23:59:59Z (`lme.question_as_of`, ADR 0028). A session is eligible iff its date <= `as_of`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from bench.datasets import lme


def question_row(q: Mapping[str, Any]) -> dict[str, Any]:
    """Eligible gold and haystack sessions of one question, before and after."""
    before, after = lme.parse_lme_date(q["question_date"]), lme.question_as_of(q["question_date"])
    hay = lme.haystack_of(q)
    gold = lme.gold_sessions(q)
    dates = [d for sid, d, _ in hay.sessions if sid in gold]
    return {
        "qid": q["question_id"], "type": q["question_type"], "is_abs": lme.is_abs(q["question_id"]),
        "as_of_before": before, "as_of_after": after, "gold": len(dates),
        "gold_before": sum(d <= before for d in dates), "gold_after": sum(d <= after for d in dates),
        "gold_later_day": sum(d > after for d in dates),
        "hay_before": sum(d <= before for _, d, _ in hay.sessions),
        "hay_after": sum(d <= after for _, d, _ in hay.sessions),
    }  # fmt: skip


def report(questions: Mapping[str, Mapping[str, Any]],
           manifest: Sequence[lme.ManifestRow]) -> dict[str, Any]:  # fmt: skip
    """Tune rows only (every kind: answerable and `_abs`)."""
    rows = [question_row(questions[m.question_id]) for m in lme.select(manifest, "tune", "all")]
    changed = [r for r in rows if r["gold_after"] != r["gold_before"]]

    def by_type(rs: Iterable[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(r["type"] for r in rs).items()))

    return {
        "split": "tune", "n": len(rows), "gold_changed": len(changed),
        "gold_changed_by_type": by_type(changed),
        "all_gold_was_ineligible": sum(r["gold_before"] == 0 < r["gold_after"] for r in changed),
        "gold_later_day": sum(r["gold_later_day"] for r in rows),
        "haystack_changed": sum(r["hay_after"] != r["hay_before"] for r in rows),
        "rows": sorted(changed, key=lambda r: (r["type"], r["qid"])),
    }  # fmt: skip


def table(s: Mapping[str, Any]) -> str:
    """Markdown: one line per tune qid whose eligible gold changes."""
    head = ["| qid | type | gold sessions | eligible before | eligible after |",
            "|---|---|---|---|---|"]  # fmt: skip
    body = [f"| {r['qid']} | {r['type']}{' (abs)' if r['is_abs'] else ''} | {r['gold']} | "
            f"{r['gold_before']} | {r['gold_after']} |" for r in s["rows"]]  # fmt: skip
    return "\n".join(head + body)


def summary_lines(s: Mapping[str, Any]) -> list[str]:
    types = ", ".join(f"{k} {v}" for k, v in s["gold_changed_by_type"].items()) or "none"
    return [
        f"tune questions: {s['n']}; eligible gold changes on {s['gold_changed']} ({types})",
        f"  all gold ineligible before, some after: {s['all_gold_was_ineligible']}",
        f"  gold sessions on a later day than the question: {s['gold_later_day']} (expected 0)",
        f"  questions whose eligible haystack changes (gold or not): {s['haystack_changed']}",
    ]
