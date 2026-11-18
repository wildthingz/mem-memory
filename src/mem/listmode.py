"""List-everything-then-count mode (MEM2-193, ADR 0023): the gate's `list_model` and the layout.

For a query the detector (`mem.aggcue`) flags as an aggregation, `read.list_mode` makes recall
return *every* pool member whose p is at least `tau_list` (not the top-k, no τ_high / gap
abstention), grouped by session with its date, plus one line telling the reader to list all
matches before counting. `tau_list` is its own calibrated threshold, fitted on LME tune by nested
CV (`mem.listfit`) and kept in `meta.gate.list_model` with the fingerprint of the fit and of the
Platt it is a cut of; the mode has its own token cap (`CAP_TOKENS`). With the switch off none of
this runs and recall is byte for byte what it was.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mem.common import estimate_tokens

VERSION = 1
CAP_TOKENS = 1000  # the mode's own response cap (ADR 0023); the normal budget (600) is unchanged
FLOOR = 0.90  # fit rule: the strictest tau_list whose training mean session coverage is >= FLOOR
# Here, not in mem.gate: mem.read raises it, and mem.gate -> readguard -> conflicts -> read (MEM2-240).
NO_LIST_MODEL = (
    "read.list_mode is on but the gate has no list_model (tau_list, ADR 0023): fit it (bench"
    " fit-tau-list, bench gate-add-tau-list), use the gate fitted on this config, or set"
    " read.list_mode: false in config.yaml (a new fingerprint)"
)
INSTRUCTION = "List every match below before you count; one line per match."
MORE = "+{n} more matches above tau_list not shown (token cap {cap}); narrow the query"
# Cost model of a list response, for the fit (pools carry no text). Measured on the layout below with
# a typical row; `tests/test_list_mode.py` checks it never undercounts a rendered response.
HEAD_TOKENS = 40
GROUP_TOKENS = 16
ROW_TOKENS = 30


def tokens_for(n_rows: int, n_groups: int) -> int:
    return HEAD_TOKENS + GROUP_TOKENS * n_groups + ROW_TOKENS * n_rows


def fingerprint(
    tau_list: float, cap: int, floor: float, platt: tuple[float, float], nested: Mapping[str, Any]
) -> str:
    body = json.dumps([VERSION, tau_list, cap, floor, list(platt), nested], sort_keys=True,
                      separators=(",", ":"))  # fmt: skip
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ListModel:
    tau_list: float
    cap_tokens: int
    floor: float
    platt_a: (
        float  # tau_list is a cut of p = σ(a·z + b) with these: a gate with other Platt is refused
    )
    platt_b: float
    nested: dict[str, Any]  # the outer folds of the nested CV, kept with the cut they justify

    @classmethod
    def build(
        cls,
        tau_list: float,
        cap_tokens: int,
        floor: float,
        platt_a: float,
        platt_b: float,
        nested: Mapping[str, Any],
    ) -> ListModel:
        if not isinstance(tau_list, int | float) or not 0.0 < tau_list < 1.0:
            raise ValueError(f"list_model: tau_list must be in (0, 1), got {tau_list!r}")
        if not isinstance(cap_tokens, int) or cap_tokens <= 0:
            raise ValueError(
                f"list_model: cap_tokens must be a positive integer, got {cap_tokens!r}"
            )
        if not nested or not nested.get("outer_folds"):
            raise ValueError("list_model: the nested CV outer folds are required")
        return cls(float(tau_list), cap_tokens, float(floor), float(platt_a), float(platt_b),
                   dict(nested))  # fmt: skip

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ListModel:
        """A record without `tau_list`, or whose fingerprint disagrees with its fields, is refused."""
        if d.get("tau_list") is None:
            raise ValueError("list_model: tau_list is required (no default cut; fit it: bench"
                             " fit-tau-list, then bench gate-add-tau-list)")  # fmt: skip
        m = cls.build(
            d["tau_list"],
            d.get("cap_tokens", 0),
            d.get("floor", math.nan),
            d.get("platt_a", math.nan),
            d.get("platt_b", math.nan),
            d.get("nested") or {},
        )
        if d.get("fingerprint") != m.fingerprint():
            raise ValueError("list_model: fingerprint does not match its fields")
        return m

    def fingerprint(self) -> str:
        return fingerprint(self.tau_list, self.cap_tokens, self.floor,
                           (self.platt_a, self.platt_b), self.nested)  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {"tau_list": self.tau_list, "cap_tokens": self.cap_tokens, "floor": self.floor,
                "platt_a": self.platt_a, "platt_b": self.platt_b, "nested": self.nested,
                "version": VERSION, "fingerprint": self.fingerprint()}  # fmt: skip


def layout(header: str, groups: list[tuple[str, list[str]]], *, more: str = "") -> str:
    """The response: header, the instruction line, then per session its date line and its rows."""
    lines = [header, INSTRUCTION]
    for title, rows in groups:
        lines.append(title)
        lines += [f"  {r}" for r in rows]
    if more:
        lines.append(more)
    return "\n".join(lines)


def response_tokens(text: str) -> int:
    return estimate_tokens(text)
