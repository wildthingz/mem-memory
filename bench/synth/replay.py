"""Oracle-client replay of a SYN world (benchmarks-v2 §2 B-W suites, "oracle-client mode").

The harness plays Claude with gold knowledge and uses only supported operations: `remember` with the
`about` forms of architecture-v2 §4.1, `link`, `forget`. It records, per op, what the server did
FIRST (the decision that B-W1 scores) and what the final state was after the oracle answered a
`needs_choice`. Scoring is in `bench.synth.score`; this module only drives and records.

The oracle:
- explicit first mentions use `new:<type>:<name>`; the first use of an alias is an introduction op,
  written `alias → #e<id>` (not scored, B-W1);
- on `needs_choice` it answers with `#e<id>` of the gold entity if the candidates offer it, else
  `new:<type>:<name>` with `distinct_new` (benchmarks-v2 B-W1, ADR 0027: a cross-kind pair's
  second `new:` asks); the `#e<id>` form teaches no alias;
- `link(new, 'supersedes', old)` only if the gold counterpart is in the returned
  `possible_conflicts`; `link(new, 'duplicate_of', old)` for a surfaced paraphrase duplicate;
  nothing for `compatible_add`; `forget` for `retract`.

`cut` (the dev export `bench syn-score-dev --fit`, MEM2-156): a callable that turns the server's
unfiltered candidate rows (theta_conf = -inf, every pick with its features) into the ids a fitted
scorer would have returned (`bench.synth.calibration.fit_cut` = `mem.confmodel.list_cut` on the
recorded features). The oracle then sees and acts on that cut list only, so the store evolves as
under the gate (a miss leaves its stale memory active for every later op) while `conflict_rows`
still hold the whole pool for the fit. Without `cut` the server's list is used as returned.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bench.suites.base import Client
from bench.synth.generator import Synth

_ENT = re.compile(r"^e(\d+)\b")
_SUBJ = re.compile(r"^e(\d+)\s.*\((new|alias learned|matched)[^)]*\)\s*$")


class ReplayError(RuntimeError):
    """The server answered in a way the oracle cannot continue from (named in the message)."""


@dataclass
class Decision:
    """One scored `about` entry: what the server did first, and where it ended."""

    entity: str
    style: str
    first: str  # ask | new | link | error
    first_id: int | None  # entity number the server linked or created on the first call
    final_id: int | None
    offered_gold: bool | None = None  # on ask: was the gold entity among the candidates


@dataclass
class OpLog:
    op: dict[str, Any]
    status: str | None = None  # of the final remember reply
    mid: str | None = None  # m<id> saved, or the existing one for `exists`
    conflicts: list[str] = field(default_factory=list)
    conflict_rows: list[dict[str, Any]] = field(default_factory=list)
    surfaced: bool | None = None  # gold counterpart in `possible_conflicts`
    acted: str | None = None  # link rel or `forget`
    action_ok: bool | None = None
    error: str | None = None
    decisions: list[Decision] = field(default_factory=list)
    asked: bool = False
    ms: float = 0.0


def conflict_ids(structured: dict[str, Any]) -> list[str]:
    out = []
    for c in structured.get("possible_conflicts") or []:
        if isinstance(c, dict):
            out.append(str(c["id"]))
        else:
            out.append(str(c).split()[0])
    return out


def candidate_ids(structured: dict[str, Any]) -> list[int]:
    out = []
    for c in structured.get("candidates") or []:
        m = _ENT.match(str(c["id"] if isinstance(c, dict) else c).strip())
        if m:
            out.append(int(m.group(1)))
    return out


def subject_info(structured: dict[str, Any]) -> tuple[int, str] | None:
    """(entity number, state) of the first subject line: `e45 Ana Silva (new)`."""
    for line in structured.get("subjects") or []:
        m = _SUBJ.match(str(line).strip())
        if m:
            return int(m.group(1)), m.group(2)
    return None


class Replay:
    def __init__(
        self, client: Client, syn: Synth, *, act: bool = True,
        on_op: Callable[[OpLog], None] | None = None,
        cut: Callable[[list[dict[str, Any]]], list[str]] | None = None,
    ) -> None:  # fmt: skip
        self.client, self.syn, self.act, self.on_op, self.cut = client, syn, act, on_op, cut
        self.eid: dict[str, int] = {}  # entity key -> server entity number
        self.mid: dict[int, str] = {}  # op index -> memory id shown (m12)
        self.logs: list[OpLog] = []
        self._ent = {e["key"]: e for e in syn.expected["entities"]}

    # -- helpers --------------------------------------------------------------------------
    def _about(self, m: dict[str, Any]) -> str:
        ent = self._ent[m["entity"]]
        style, surface = m["style"], m["surface"]
        if style == "new":
            return f"new:{ent['type']}:{surface}"
        if style == "alias_intro":
            return f"{surface} → #e{self._eid(m['entity'])}" if m["entity"] in self.eid \
                else surface  # fmt: skip
        if style == "id":
            return f"#e{self._eid(m['entity'])}" if m["entity"] in self.eid else surface
        return surface

    def _eid(self, key: str) -> int:
        return self.eid[key]

    def _call(self, tool: str, args: dict[str, Any], now: str) -> Any:
        return self.client.call(tool, args, now=now)

    # -- one op -----------------------------------------------------------------------------
    def step(self, op: dict[str, Any]) -> OpLog:
        log = OpLog(op)
        try:
            if op["tool"] == "forget":
                self._forget(op, log)
            else:
                self._remember(op, log)
        except Exception as e:  # noqa: BLE001 - a tool error is data for the scorer
            log.error = f"{type(e).__name__}: {e}"
        self.logs.append(log)
        if self.on_op:
            self.on_op(log)
        return log

    def run(self, upto: int | None = None) -> list[OpLog]:
        for op in self.syn.ops[:upto]:
            self.step(op)
        return self.logs

    def _forget(self, op: dict[str, Any], log: OpLog) -> None:
        target = self.mid.get(op["args"]["ref"])
        if target is None:
            log.error = "retract target was never saved"
            return
        r = self._call("forget", {"id": target, "reason": op["args"]["reason"]}, op["now"])
        log.acted, log.action_ok, log.status = "forget", True, "forgotten"
        log.mid = target
        del r

    def _remember(self, op: dict[str, Any], log: OpLog) -> None:
        a = op["args"]
        mentions = op["mentions"]
        about = [self._about(m) for m in mentions]
        args = {**a, "about": about}
        r = self._call("remember", args, op["now"])
        s = r.structured
        first = s
        if s["status"] == "needs_choice":
            log.asked = True
            offered = candidate_ids(s)
            fixed, distinct = [], False
            for m, entry in zip(mentions, about, strict=True):
                gold = self.eid.get(m["entity"])
                ent = self._ent[m["entity"]]
                if gold is not None and gold in offered:
                    fixed.append(f"#e{gold}")
                else:
                    fixed.append(f"new:{ent['type']}:{ent['name']}")
                    distinct = True
                del entry
            # a `new:` name held by an entity of another type asks too (ADR 0027): the gold
            # entity is not offered, so it is a different thing
            extra = {"distinct_new": True} if distinct else {}
            r = self._call("remember", {**a, "about": fixed, **extra}, op["now"])
            s = r.structured
            if s["status"] == "needs_choice":
                raise ReplayError("needs_choice again after the oracle's answer")
        log.status = s["status"]
        log.mid = s.get("id")
        log.conflicts = conflict_ids(s)
        log.conflict_rows = [c for c in (s.get("possible_conflicts") or []) if isinstance(c, dict)]
        if self.cut is not None and log.conflict_rows:
            log.conflicts = self.cut(log.conflict_rows)
        self._decisions(op, mentions, first, s, log)
        if s["status"] == "saved":
            self.mid[op["i"]] = s["id"]
        self._act(op, log)

    def _decisions(self, op, mentions, first, final, log) -> None:
        f_info, s_info = subject_info(first), subject_info(final)
        for m in mentions:
            if first["status"] == "needs_choice":
                how, first_id = "ask", None
            elif f_info is None:
                how, first_id = "error", None
            else:
                first_id = f_info[0]
                how = "new" if f_info[1] == "new" else "link"
            offered = None
            if how == "ask":
                gold = self.eid.get(m["entity"])
                offered = gold in candidate_ids(first) if gold is not None else None
            final_id = s_info[0] if s_info else None
            log.decisions.append(Decision(m["entity"], m["style"], how, first_id, final_id,
                                          offered))  # fmt: skip
            if (
                m["entity"] not in self.eid
                and final_id is not None
                or m["style"] == "new"
                and final_id is not None
            ):
                self.eid[m["entity"]] = final_id

    def _act(self, op: dict[str, Any], log: OpLog) -> None:
        kind, gold = op["kind"], op["gold"]
        if log.status != "saved":
            return
        if kind in ("update", "dup_paraphrase"):
            cp = self.mid.get(gold["counterpart"])
            log.surfaced = cp is not None and cp in log.conflicts
            if log.surfaced and self.act:
                rel = "supersedes" if kind == "update" else "duplicate_of"
                try:
                    self._call("link", {"src": log.mid, "rel": rel, "dst": cp}, op["now"])
                    log.acted, log.action_ok = rel, True
                except Exception as e:  # noqa: BLE001
                    log.acted, log.action_ok, log.error = rel, False, f"{type(e).__name__}: {e}"
        elif kind == "compatible_add":
            cp = self.mid.get(gold["counterpart"])
            log.surfaced = cp is not None and cp in log.conflicts
