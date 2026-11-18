"""Freeze-on-tune and the ≤ 3 looks rule (benchmarks-v2 §5.4).

- Freeze: before the first test look every `split: test` row of the phase gets `frozen_at`, either
  with the spec's threshold unchanged or changed ONCE with a written reason based on tune data only.
  After any test look the thresholds never change for that bench_version.
- Looks: ≤ 3 test looks per phase, each logged in bench/ledger.jsonl (who, why, config hash).
- Hard floors (B-R1 A16): delivered precision is never below 0.60 and IRA never ≥ 1.0.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from bench import records
from bench.gates import GATES_YAML, load_gates

MAX_LOOKS = 3
HARD_FLOORS = {  # (id, metric): (op, bound, inclusive)
    ("B-R1", "delivered_precision"): (">=", 0.60, True),
    ("B-R1", "ira"): ("<", 1.0, False),
}
_ROW = re.compile(
    r'^(?P<head>\s*- \{id: (?P<id>[^,]+), metric: (?P<metric>[^,]+), op: (?P<op>"[^"]+"), threshold: )'
    r"(?P<thr>[^,]+)(?P<mid>, split: (?P<split>[^,]+), frozen_at: )(?P<frozen>[^,}]+)(?P<tail>(?:, (?:descriptive|known_miss): true)*)\}\s*$"
)


class Refused(RuntimeError):
    """A rule of §5.4 forbids the requested action."""


def read_ledger(ledger: Path = records.LEDGER) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_looks(phase: str, ledger: Path = records.LEDGER) -> int:
    return sum(e.get("phase") == phase and e.get("split") == "test" for e in read_ledger(ledger))


def unfrozen_test_rows(phase: str, gates_path: Path = GATES_YAML) -> list[str]:
    return [f"{s.id}.{s.metric}" for s in load_gates(gates_path)[phase]
            if s.split == "test" and not s.frozen_at]  # fmt: skip


def check_test_look_allowed(
    phase: str, gates_path: Path = GATES_YAML, ledger: Path = records.LEDGER
) -> None:
    rows = unfrozen_test_rows(phase, gates_path)
    if rows:
        raise Refused(f"{phase} thresholds are not frozen (run `bench freeze {phase}` on tune "
                      f"first): {', '.join(rows[:4])}{' ...' if len(rows) > 4 else ''}")  # fmt: skip
    looks = test_looks(phase, ledger)
    if looks >= MAX_LOOKS:
        raise Refused(f"{phase} already has {looks} test looks in the ledger (max {MAX_LOOKS})")


def _floor_ok(key: tuple[str, str], value: float) -> bool:
    if key not in HARD_FLOORS:
        return True
    op, bound, inclusive = HARD_FLOORS[key]
    return (value >= bound if inclusive else value < bound) if op in (">=", "<") else True


def freeze(
    phase: str, *, who: str, reason: str, config_hash: str,
    overrides: dict[tuple[str, str], float] | None = None,
    gates_path: Path = GATES_YAML, ledger: Path = records.LEDGER, now: datetime | None = None,
) -> list[str]:  # fmt: skip
    """Stamp `frozen_at` on the phase's test rows; apply `overrides` once. Returns change lines."""
    overrides = overrides or {}
    if test_looks(phase, ledger):
        raise Refused(f"{phase} has a test look in the ledger: thresholds never change after it")
    if not reason.strip():
        raise Refused("a freeze needs a written reason")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines, changes, seen, in_phase = [], [], set(), False
    for line in gates_path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^  P\d+:\s*$", line):
            in_phase = line.strip() == f"{phase}:"
        m = _ROW.match(line) if in_phase else None
        if m and m["split"] == "test":
            key = (m["id"], m["metric"])
            if m["frozen"].strip() != "null":
                raise Refused(f"{key[0]}.{key[1]} is already frozen ({m['frozen'].strip()})")
            thr = m["thr"]
            if key in overrides:
                if not _floor_ok(key, overrides[key]):
                    raise Refused(f"{key[0]}.{key[1]}={overrides[key]} breaks the hard floor "
                                  f"{HARD_FLOORS[key]}")  # fmt: skip
                changes.append(f"{key[0]}.{key[1]}: {thr} -> {overrides[key]:g}")
                thr = f"{overrides[key]:g}"
                seen.add(key)
            line = f'{m["head"]}{thr}{m["mid"]}"{stamp}"{m["tail"] or ""}}}'
        lines.append(line)
    missing = set(overrides) - seen
    if missing:
        raise Refused(f"no such test rows in {phase}: {sorted(missing)}")
    gates_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records.append_ledger({"who": who, "why": f"freeze {phase}: {reason}", "phase": phase,
                           "split": "tune", "kind": "freeze", "config_hash": config_hash,
                           "changes": changes}, ledger)  # fmt: skip
    return changes
