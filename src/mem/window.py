"""The anchored window of a query (MEM2-266, ADR 0031): `read.window_boost`.

A question that names a relative date ("last Saturday", "two weeks ago", "last month") is resolved
with the `mem.reldates` rules against the question day (`as_of`, else today): no new parser, no
model. The window is the resolved day, week, month or year widened by a slack that grows with the
anchor's distance from the question day, `max(1 day, ceil(slack x distance))` (Weber-style: people
date distant events less precisely). The read path adds `delta` to the reranker logit of pool
members whose `valid_from` falls in a window and delivers the gated in-window rows first: a boost,
never a filter. Anchors on or after the question day ("today", "tomorrow", "next Friday") give no
window: the question day is the store's edge, and the future holds nothing to recall.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from mem import reldates

DELTA = 1.0  # logits added in-window; picked on tune (MEM2-266 dry run, ADR 0031)
SLACK = 0.1  # slack per day of distance; picked on tune (MEM2-266 dry run, ADR 0031)


@dataclass(frozen=True)
class Window:
    phrase: str  # as written in the query
    value: str  # the reldates value: a day, `a..b`, a month `YYYY-MM` or a year
    start: date
    end: date


def _anchor(value: str) -> tuple[date, date]:
    if ".." in value:
        a, b = value.split("..")
        return date.fromisoformat(a), date.fromisoformat(b)
    if len(value) == 10:
        d = date.fromisoformat(value)
        return d, d
    if len(value) == 7:
        y, m = int(value[:4]), int(value[5:])
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    return date(int(value), 1, 1), date(int(value), 12, 31)


def windows(query: str, ref: date, slack: float) -> list[Window]:
    """The query's windows in order of appearance; [] when no phrase resolves to the past."""
    out = []
    for phrase, value in reldates.resolve(query, ref):
        start, end = _anchor(value)
        if start >= ref:
            continue
        distance = (ref - (start + (end - start) // 2)).days
        pad = timedelta(days=max(1, math.ceil(round(slack * distance, 9))))
        out.append(Window(phrase, value, start - pad, min(end + pad, ref)))
    return out


def inside(day: date | None, ws: Sequence[Window]) -> bool:
    return day is not None and any(w.start <= day <= w.end for w in ws)


def in_window_first(order: Sequence[int], flags: Sequence[bool]) -> list[int]:
    """`order` with the in-window members first, each part in its own order."""
    return [i for i in order if flags[i]] + [i for i in order if not flags[i]]


def settings(value: Any) -> dict[str, float]:
    """`read.window_boost` as `{delta, slack}`: `true` is the tuned values; a mapping is checked."""
    if value is True:
        return {"delta": DELTA, "slack": SLACK}
    if not isinstance(value, dict) or set(value) != {"delta", "slack"}:
        raise ValueError(
            "read.window_boost must be true, false or {delta: <logits>, slack: <per day>}"
        )
    nums = {
        k: v for k, v in value.items() if isinstance(v, int | float) and not isinstance(v, bool)
    }
    if len(nums) != 2 or not nums["delta"] > 0 or not nums["slack"] >= 0:
        raise ValueError("read.window_boost: delta must be > 0 and slack >= 0")
    return {"delta": float(nums["delta"]), "slack": float(nums["slack"])}
