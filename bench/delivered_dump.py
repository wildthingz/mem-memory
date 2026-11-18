"""The delivered-rows dump (MEM2-210, report §7): for each LME tune question, which (session, round)
rows recall delivers under the current config, against the gold evidence. No model call, no API.

Per question and k (first k delivered rows, in the order recall shows them):
- `recall_any`: a gold round (a round with a `has_answer` turn) is among them;
- `recall_all`: every gold round is (1/0 per question);
- `gold_round_frac`: the share of gold rounds among them;
- `gold_session_frac`: the share of gold sessions (`answer_session_ids`) with a delivered row;
- `session_any`: some gold session has a delivered row.
The same over the **whole delivery** (key `all`, MEM2-259): every row recall delivered, past k, and with
paging (ADR 0024) every later page followed (`paged`), so list mode's extra rows are counted.
Questions without gold rounds cannot be scored and are listed, never counted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from bench.lme_eval import Item

KS = (5, 8, 10)
WHOLE = "all"  # the whole delivery: every delivered row, every page
METRICS = ("recall_any", "recall_all", "gold_round_frac", "gold_session_frac", "session_any")


def _metrics(got: Sequence[str], gold_r: set[str] | frozenset[str],
             gold_s: set[str] | frozenset[str]) -> dict[str, float]:  # fmt: skip
    hit_s = {x.partition("#")[0] for x in got} & gold_s
    return {
        "recall_any": int(bool(gold_r & set(got))),
        "recall_all": int(bool(gold_r) and gold_r <= set(got)),
        "gold_round_frac": len(gold_r & set(got)) / len(gold_r) if gold_r else 0.0,
        "gold_session_frac": len(hit_s) / len(gold_s) if gold_s else 0.0,
        "session_any": int(bool(hit_s)),
    }


def question_record(it: Item, ks: Sequence[int] = KS) -> dict[str, Any]:
    gold_r, gold_s = it.gold_rounds, it.gold_sessions
    at = {str(k): _metrics(it.r.S[:k], gold_r, gold_s) for k in ks}
    at[WHOLE] = _metrics([*it.r.S, *it.r.paged], gold_r, gold_s)
    return {"qid": it.qid, "category": it.category, "gold_rounds": sorted(gold_r),
            "gold_sessions": sorted(gold_s), "delivered": list(it.r.S), "paged": list(it.r.paged),
            "at": at}  # fmt: skip


def _at(rec: dict[str, Any], key: str) -> dict[str, float]:
    """`rec["at"][key]`; a metric a dump of an older version lacks (the whole column, the gold-round
    fraction) is derived from its `delivered` rows (+ `paged`) and gold, when it has them."""
    have = rec.get("at", {}).get(key, {})
    if all(m in have for m in METRICS) or "delivered" not in rec:
        return have
    rows = (
        [*rec["delivered"], *rec.get("paged", [])] if key == WHOLE else rec["delivered"][: int(key)]
    )
    return _metrics(rows, set(rec["gold_rounds"]), set(rec.get("gold_sessions", ()))) | have


def _keys(ks: Sequence[int]) -> list[str]:
    return [*map(str, ks), WHOLE]


def _mean(rows: Sequence[dict[str, Any]], k: str) -> dict[str, float]:
    n = len(rows)
    return {"n": n, **{m: sum(_at(r, k)[m] for r in rows) / n for m in METRICS}}


def summarize(recs: Sequence[dict[str, Any]], ks: Sequence[int] = KS) -> dict[str, Any]:
    scored = [r for r in recs if r["gold_rounds"]]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        by_type[r["category"]].append(r)
    return {
        "n": len(scored), "ks": list(ks),
        "skipped_no_gold": sorted(r["qid"] for r in recs if not r["gold_rounds"]),
        "paged_questions": sum(bool(r.get("paged")) for r in scored),
        "overall": {k: _mean(scored, k) for k in _keys(ks)} if scored else {},
        "by_type": {t: {k: _mean(rows, k) for k in _keys(ks)}
                    for t, rows in sorted(by_type.items())},
    }  # fmt: skip


def pair(base: Sequence[dict[str, Any]], feat: Sequence[dict[str, Any]],
         ks: Sequence[int] = KS) -> dict[str, Any]:  # fmt: skip
    """Feature minus baseline on the questions both dumps scored: the mean delta of each metric at
    each k and over the whole delivery (`all`) with its paired-bootstrap 95% CI (resampled within
    question type), overall and per type. A metric some record lacks (a hand-made or older dump
    without `delivered`) is left out."""
    from bench.metrics import stats

    a = {r["qid"]: r for r in base if r["gold_rounds"]}
    b = {r["qid"]: r for r in feat if r["gold_rounds"]}
    ids = sorted(a.keys() & b.keys())

    def deltas(sel: Sequence[str]) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for k in _keys(ks):
            fa, fb = [_at(a[i], k) for i in sel], [_at(b[i], k) for i in sel]
            if not all(fa) or not all(fb):
                continue
            out[k] = {}
            for m in METRICS:
                if not all(m in x for x in fa + fb):
                    continue
                p = stats.paired_bootstrap([x[m] for x in fb], [x[m] for x in fa],
                                           [a[i]["category"] for i in sel])  # fmt: skip
                out[k][m] = {"n": p.n, "delta": p.delta, "lo": p.lo, "hi": p.hi}
        return out

    types = sorted({a[i]["category"] for i in ids})
    return {"n": len(ids), "overall": deltas(ids) if ids else {},
            "by_type": {t: deltas([i for i in ids if a[i]["category"] == t]) for t in types}}  # fmt: skip
