"""`bench temporal-dryrun` (MEM2-209): falsification dry run for options A and D. No models.

A: a deterministic date-window parser over the question text (`dateparser` plus rules), counted on the tune
temporal losses. D: deterministic two-endpoint patterns, counted on the tune temporal questions, with the false
splits on the rest. Neither reads the retrieval path; both only count how often the pattern fires.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from mem.splitq import Split, split_endpoints  # noqa: F401  (the frozen D patterns live in mem.splitq)

A_MIN_FRACTION = (
    0.5  # A is falsified if the parser fires on fewer than this share of the tune temporal losses
)
D_MIN_MATCHED = 10  # D is dropped if fewer than this many of the tune temporal questions match

_UNITS = {"day": 1, "week": 7, "month": 30, "year": 365}
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple": 2}  # fmt: skip
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]  # fmt: skip
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_N = r"(\d+|" + "|".join(_NUM) + r")"
_UNIT = r"(day|week|month|year)s?"
_DATE_RE = (r"(?:(?:" + "|".join(_MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?"
            r"|\d{4}-\d{2}-\d{2})")  # fmt: skip


@dataclass(frozen=True)
class Window:
    rule: str
    start: str | None  # ISO date; None = open
    end: str | None
    text: str  # what in the question produced it


def _base(question_date: str) -> date:
    return datetime.strptime(question_date[:10], "%Y/%m/%d").date()


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM[tok]


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _explicit_date(text: str, base: date) -> date | None:
    import dateparser

    return _as_date(dateparser.parse(text, settings={"RELATIVE_BASE": datetime.combine(base, datetime.min.time()),
                                                     "PREFER_DATES_FROM": "past"}))  # fmt: skip


def _as_date(dt: datetime | None) -> date | None:
    return dt.date() if dt else None


def parse_window(question: str, question_date: str) -> Window | None:
    """The time window the question names, or None. Rules first (they are exact), `dateparser` last."""
    import dateparser.search

    q, base = question.lower(), _base(question_date)
    if m := re.search(rf"\b{_N} {_UNIT} ago\b", q):
        n, unit = _num(m[1]), m[2]
        t = base - timedelta(days=n * _UNITS[unit])
        slack = {"day": 0, "week": 0, "month": 7, "year": 30}[unit]
        return Window(
            "n_units_ago", _iso(t - timedelta(days=slack)), _iso(t + timedelta(days=slack)), m[0]
        )
    if m := re.search(rf"\b(?:past|last|previous) {_N} {_UNIT}\b", q):
        return Window(
            "past_n", _iso(base - timedelta(days=_num(m[1]) * _UNITS[m[2]])), _iso(base), m[0]
        )
    if m := re.search(rf"\blast ({'|'.join(_WEEKDAYS)})\b", q):
        back = (base.weekday() - _WEEKDAYS.index(m[1])) % 7 or 7
        d = base - timedelta(days=back)
        return Window("last_weekday", _iso(d), _iso(d), m[0])
    if m := re.search(r"\blast (week|month|year)\b", q):
        if m[1] == "week":
            end = base - timedelta(days=base.weekday() + 1)
            start = end - timedelta(days=6)
        elif m[1] == "month":
            end = base.replace(day=1) - timedelta(days=1)
            start = end.replace(day=1)
        else:
            start, end = date(base.year - 1, 1, 1), date(base.year - 1, 12, 31)
        return Window("last_unit", _iso(start), _iso(end), m[0])
    if m := re.search(rf"\bin ({'|'.join(_MONTHS)})(?:,? (\d{{4}}))?\b", q):
        mon = _MONTHS.index(m[1]) + 1
        year = int(m[2]) if m[2] else base.year - (mon > base.month)
        start = date(year, mon, 1)
        end = (date(year + (mon == 12), mon % 12 + 1, 1)) - timedelta(days=1)
        return Window("in_month", _iso(start), _iso(end), m[0])
    m = re.search(rf"\b(before|after|since|until) ({_DATE_RE})", q)
    if m and (d := _explicit_date(m[2], base)):
        if m[1] in ("before", "until"):
            return Window("before_date", None, _iso(d - timedelta(days=1)), m[0])
        return Window("after_date", _iso(d + timedelta(days=1)), _iso(base), m[0])
    settings = {
        "RELATIVE_BASE": datetime.combine(base, datetime.min.time()),
        "PREFER_DATES_FROM": "past",
    }
    for text, dt in (
        dateparser.search.search_dates(question, languages=["en"], settings=settings) or []
    ):
        # dateparser reads words like "may" or "second" as dates; keep only anchored matches
        if (
            re.search(r"\d|yesterday|ago\b|" + "|".join(_MONTHS), text.lower())
            and dt.date() != base
        ):
            return Window("dateparser", _iso(dt.date()), _iso(dt.date()), text)
    return None


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(tune: Path, e1: Path | None, out: Path) -> dict:
    qs = _read(tune)
    tune_ids = {q["qid"] for q in qs}
    losses: dict[str, dict] = {}
    if e1:  # stream line by line, keep tune qids only: the test qids are never kept
        mem: dict[str, dict] = {}
        oracle: dict[str, bool] = {}
        for line in e1.read_text().splitlines():
            r = json.loads(line)
            if r["qid"] not in tune_ids or "temporal" not in r["type"]:
                continue
            if r["cond"] == "mem":
                mem[r["qid"]] = r
            elif r["cond"] == "oracle":
                oracle[r["qid"]] = r["correct"]
        losses = {
            k: {**v, "oracle_correct": oracle.get(k)} for k, v in mem.items() if not v["correct"]
        }
    rows = []
    for q in qs:
        w = parse_window(q["question"], q["question_date"])
        s = split_endpoints(q["question"])
        loss = losses.get(q["qid"])
        rows.append({
            "qid": q["qid"], "type": q["question_type"], "is_abs": q.get("is_abs", False),
            "question": q["question"], "question_date": q["question_date"],
            "window": None if w is None else w.__dict__, "no_window_reason": None if w else _why_none(q["question"]),
            "split": None if s is None else {"pattern": s.pattern, "parts": s.parts},
            "mem_loss": loss is not None, "oracle_right": loss and loss["oracle_correct"],
            "shown": loss["shown"] if loss else None,
        })  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_question.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _why_none(question: str) -> str:
    q = question.lower()
    if re.search(r"how many (days|weeks|months|years) ago|how long ago", q):
        return "asks for the elapsed time; the date is the answer, not the query"
    if re.search(r"\b(ago|last|past|recent|before|after|first|most recently|yesterday|today)\b", q):
        return "time words present but no anchored date"
    return "no time reference in the question"


def summarise(rows: list[dict]) -> dict:
    temporal = [r for r in rows if r["type"] == "temporal-reasoning" and not r["is_abs"]]
    losses = [r for r in temporal if r["mem_loss"]]
    zero = [r for r in losses if r["shown"] == 0]
    with_w = [r for r in losses if r["window"]]
    a_n = len(losses)
    d_hit = [r for r in temporal if r["split"]]
    others = [r for r in rows if r not in temporal and r["split"]]
    return {
        "a": {
            "tune_losses": a_n, "losses_with_window": len(with_w),
            "fraction": len(with_w) / a_n if a_n else None,
            "threshold": A_MIN_FRACTION, "falsified": a_n > 0 and len(with_w) / a_n < A_MIN_FRACTION,
            "shown0_losses": len(zero), "shown0_with_window": sum(1 for r in zero if r["window"]),
            "oracle_right_losses": sum(1 for r in losses if r["oracle_right"]),
            "oracle_right_with_window": sum(1 for r in losses if r["oracle_right"] and r["window"]),
            "all_temporal_with_window": sum(1 for r in temporal if r["window"]),
            "all_multi_session_with_window": sum(1 for r in rows if r["type"] == "multi-session" and r["window"]),
            "multi_session_total": sum(1 for r in rows if r["type"] == "multi-session"),
        },
        "d": {
            "temporal_total": len(temporal), "temporal_matched": len(d_hit), "threshold": D_MIN_MATCHED,
            "dropped": len(d_hit) < D_MIN_MATCHED,
            "false_splits": [{"qid": r["qid"], **r["split"]} for r in others],
        },
    }  # fmt: skip


def register(app) -> None:
    @app.command("temporal-dryrun")
    def temporal_dryrun(
        tune: Annotated[Path, typer.Argument(help="tune_questions.jsonl")],
        e1: Annotated[
            Path | None, typer.Option(help="e1-full.jsonl (tune qids only are read).")
        ] = None,
        out: Annotated[Path, typer.Option(help="Output dir.")] = Path("data/temporal-dryrun"),
    ) -> None:
        """Falsification dry run for options A (date window) and D (two-endpoint split): no models."""
        s = run(tune, e1, out)
        a, d = s["a"], s["d"]
        typer.echo(f"A: {a['losses_with_window']}/{a['tune_losses']} tune temporal losses get a window "
                   f"({'FALSIFIED' if a['falsified'] else 'holds'}, threshold {a['threshold']:.0%})")  # fmt: skip
        typer.echo(f"D: {d['temporal_matched']}/{d['temporal_total']} tune temporal match "
                   f"({'DROPPED' if d['dropped'] else 'holds'}, threshold {d['threshold']}); "
                   f"{len(d['false_splits'])} false splits")  # fmt: skip
