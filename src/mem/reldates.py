"""Relative time expressions resolved against a reference date (ADR 0021, MEM2-192).

"yesterday", "two days ago", "last Tuesday" mean nothing once a note is read months later. At
import (`ingest.resolve_dates` on) each chunk gets a derived annotation, `yesterday [= 2023-03-09]`,
stored next to the verbatim text, indexed for FTS and shown in recall. Local text processing only:
no model, no network, no clock (the reference date is an argument). An expression that is not
certain gets no annotation; see the ADR for the rules and for what is never resolved.
"""

from __future__ import annotations

import calendar
import re
from bisect import bisect_left
from datetime import date, timedelta

MAX_ANNOTATIONS = 16
# The value of an annotation: a day, a month, a year, or an ISO week's `a..b` range.
DATE_VALUE = r"\d{4}(?:-\d{2}(?:-\d{2})?)?(?:\.\.\d{4}-\d{2}-\d{2})?"
# The recall row's `dates:` field (ADR 0021, amended MEM2-254): `2023-05-27 (last Saturday), …`, the
# absolute date first; under `ROW_CHARS` the words go, then whole trailing dates, never part of one.
ROW_ENTRY = rf"{DATE_VALUE}(?: \([A-Za-z0-9 ]+\))?"
ROW_FIELD = rf"{ROW_ENTRY}(?:, {ROW_ENTRY})*"
ROW_CHARS = 40
_STORED = re.compile(rf"([A-Za-z0-9\s]+?) \[= ({DATE_VALUE})\]")

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_NUMBERS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}  # fmt: skip
_NUMBERS |= {"a": 1, "an": 1}
_NUM = r"\d{1,3}|" + "|".join(w for w in _NUMBERS if w != "zero")
_DAY = "|".join(_WEEKDAYS)

_TOKEN = re.compile(
    rf"""\b(?:
      (?P<before>(?:the\s+)?day\s+before\s+yesterday)
    | (?P<after>(?:the\s+)?day\s+after\s+tomorrow)
    | (?P<day>yesterday|today|tomorrow|tonight)
    | (?P<part>(?P<pw>this|last)\s+(?:morning|afternoon|evening|night))
    | (?P<ago>(?P<n>{_NUM})\s+(?P<unit>day|week|month|year)s?\s+ago)
    | (?P<span>(?P<sw>last|this|next)\s+(?P<su>week|month|year))
    | (?P<named>(?P<nw>last|next)\s+(?P<nd>{_DAY}))
    | (?P<skip>(?:this|every|each|all)\s+(?:{_DAY})s?)
    | (?P<bare>{_DAY})
    )\b(?![-'’]?\w)""",
    re.IGNORECASE | re.VERBOSE,
)
_SENTENCE_END = re.compile(r"[.!?\n]")
_PAST = re.compile(
    r"\b(?:was|were|had|did|went|got|came|saw|met|bought|took|made|told|said|started|finished"
    r"|ended|ran|ate|visited|called|arrived|left)\b", re.IGNORECASE)  # fmt: skip
_FUTURE = re.compile(
    r"\b(?:will|won't|shall|gonna|going\s+to|upcoming|planning|plan\s+to|scheduled|tomorrow)\b"
    r"|'ll\b", re.IGNORECASE)  # fmt: skip


def _add_months(d: date, n: int) -> date:
    """The same day-of-month n months on; a day the month lacks is clamped to its last day."""
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def _span(word: str, unit: str, ref: date) -> str:
    step = {"last": -1, "this": 0, "next": 1}[word]
    if unit == "week":
        start = ref - timedelta(days=ref.weekday()) + timedelta(weeks=step)
        return f"{start.isoformat()}..{(start + timedelta(days=6)).isoformat()}"
    if unit == "month":
        return _add_months(ref.replace(day=1), step).isoformat()[:7]
    return str(ref.year + step)


def _weekday(name: str, ref: date, direction: int) -> date:
    """The nearest `name` strictly before (-1) or after (+1) `ref`: a Sunday's "last Sunday" is a
    week back, never itself."""
    delta = (_WEEKDAYS.index(name) - ref.weekday()) % 7
    return ref + timedelta(days=(delta or 7) if direction > 0 else -((-delta) % 7 or 7))


def _bare(text: str, ends: list[int], tense: dict, start: int, name: str, ref: date) -> date | None:
    """A weekday name alone: past or future from its sentence's words, else unresolvable. `ends`
    are the sentence-end offsets (sorted); `tense` caches each sentence's cues, so a text of many
    weekdays is scanned once, not once per match. The weekday itself is no cue, so the sentence is
    read whole."""
    i = bisect_left(ends, start)
    lo, hi = (ends[i - 1] + 1 if i else 0), (ends[i] if i < len(ends) else len(text))
    if (lo, hi) not in tense:
        sentence = text[lo:hi]
        tense[lo, hi] = bool(_PAST.search(sentence)), bool(_FUTURE.search(sentence))
    past, future = tense[lo, hi]
    if past == future:
        return None
    return _weekday(name, ref, 1 if future else -1)


def resolve(text: str, ref: date) -> list[tuple[str, str]]:
    """(phrase as written, resolved value) in order of appearance, a repeated phrase once.
    A value is a date, a range `a..b`, a month `YYYY-MM` or a year."""
    out: list[tuple[str, str]] = []
    ends: list[int] | None = None
    tense: dict[tuple[int, int], tuple[bool, bool]] = {}
    for m in _TOKEN.finditer(text):
        phrase, value = m.group(0), None
        if m.group("before"):
            value = ref - timedelta(days=2)
        elif m.group("after"):
            value = ref + timedelta(days=2)
        elif word := m.group("day"):
            value = ref + timedelta(days={"yesterday": -1, "tomorrow": 1}.get(word.lower(), 0))
        elif m.group("part"):
            last = m.group("pw").lower() == "last"
            if not last or phrase.lower().endswith(("night", "evening")):
                value = ref - timedelta(days=int(last))
        elif m.group("ago"):
            n, unit = m.group("n").lower(), m.group("unit").lower()
            n = int(n) if n.isdigit() else _NUMBERS[n]
            value = (ref - timedelta(days=n * {"day": 1, "week": 7}[unit]) if unit in ("day", "week")
                     else _add_months(ref, -n * (12 if unit == "year" else 1)))  # fmt: skip
        elif m.group("span"):
            out.append((phrase, _span(m.group("sw").lower(), m.group("su").lower(), ref)))
            continue
        elif m.group("named"):
            value = _weekday(
                m.group("nd").lower(), ref, -1 if m.group("nw").lower() == "last" else 1
            )
        elif m.group("bare"):
            if ends is None:
                ends = [e.start() for e in _SENTENCE_END.finditer(text)]
            value = _bare(text, ends, tense, m.start(), phrase.lower(), ref)
        if value is not None and (phrase, value.isoformat()) not in out:
            out.append((phrase, value.isoformat()))
    return _dedupe(out)[:MAX_ANNOTATIONS]


def _dedupe(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for phrase, value in pairs:
        key = (phrase.lower(), value)
        if key not in seen:
            seen.add(key)
            out.append((phrase, value))
    return out


def annotation(text: str, ref: date) -> str | None:
    """`yesterday [= 2023-03-09]; last month [= 2023-02]`, or None when nothing resolved."""
    pairs = resolve(text, ref)
    return "; ".join(f"{p} [= {v}]" for p, v in pairs) if pairs else None


def entries(annotation: str | None) -> list[tuple[str, str]]:
    """The resolved (phrase, value) pairs of a stored annotation; an entry without a date is
    dropped, and a phrase's whitespace is collapsed so the field stays on one line."""
    out = []
    for part in (annotation or "").split(";"):
        if m := _STORED.fullmatch(part.strip()):
            out.append((" ".join(m.group(1).split()), m.group(2)))
    return out


def render(annotation: str | None, limit: int | None = None) -> str | None:
    """`2023-03-09 (yesterday), 2023-02 (last month)`, or None when nothing resolved. Over `limit`
    characters: the dates alone, distinct, as many whole ones as fit (the first always)."""
    pairs = list(dict.fromkeys(entries(annotation)))  # a memory's chunks can repeat a pair
    if not pairs:
        return None
    full = ", ".join(f"{v} ({p})" for p, v in pairs)
    if limit is None or len(full) <= limit:
        return full
    values = list(dict.fromkeys(v for _, v in pairs))
    out = values[0]
    for v in values[1:]:
        if len(out) + 2 + len(v) > limit:
            break
        out += ", " + v
    return out
