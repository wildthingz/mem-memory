"""Gate runner: gates.yaml + suites -> one PASS / FAIL / UNAVAILABLE / DEFERRED line per gate id.

DEFERRED: every row the suite ran passed, and the suite explicitly deferred the others
(`SuiteResult.deferred`, e.g. the P5 cross-machine rows with `--home-only`). It is not a pass.
"""

from __future__ import annotations

import math
import operator
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from bench import records
from bench.suites import suite_for
from bench.suites.base import BackendUnavailable, ClientFactory, Context, SuiteResult

GATES_YAML = Path(__file__).resolve().parent / "configs" / "gates.yaml"
_OPS: dict[str, Callable[[float, float], bool]] = {
    "==": operator.eq, "<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt,
}  # fmt: skip


KNOWN_MISS_NOTE = "accepted by the owner, 2026-10-25; thresholds unchanged; revisit on PG-v1"


@dataclass(frozen=True)
class GateSpec:
    id: str
    metric: str
    op: str
    threshold: float | None
    split: str
    frozen_at: str | None
    # reported with its value and threshold, never failing (MEM2-128: LongMemEval's round-level
    # labels make its precision-side numbers descriptive; the threshold is kept, not relaxed)
    descriptive: bool = False
    # a miss the owner accepted (P3 B-R5 on SYN, 2026-10-25): the threshold is kept; below it the row is
    # reported as a KNOWN MISS and does not fail the run; a metric that is not produced still fails
    known_miss: bool = False


@dataclass
class Outcome:
    id: str
    status: str  # PASS | FAIL | UNAVAILABLE | DEFERRED
    detail: str
    run_file: Path | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def deferred(self) -> bool:
        return self.status == "DEFERRED"


def load_gates(path: Path = GATES_YAML) -> dict[str, list[GateSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("bench_version") != records.BENCH_VERSION:
        raise ValueError(f"gates.yaml is bench_version {raw.get('bench_version')}")
    return {ph: [GateSpec(**g) for g in specs] for ph, specs in raw["phases"].items()}


def evaluate(spec: GateSpec, res: SuiteResult) -> tuple[bool, str]:
    if spec.descriptive and spec.metric not in res.metrics:
        return True, f"DESCRIPTIVE {spec.metric} not produced (not gating)"
    if spec.metric in res.deferred and spec.metric not in res.metrics:
        return False, f"DEFERRED {spec.metric} (not run: {res.deferred[spec.metric]})"
    if spec.metric not in res.metrics:
        return False, f"metric {spec.metric!r} not produced"
    value = res.metrics[spec.metric]
    if spec.op == "record":
        return (not math.isnan(value)), f"recorded {spec.metric}={value:g} (no threshold)"
    ok = not math.isnan(value) and _OPS[spec.op](value, spec.threshold)  # type: ignore[arg-type]
    detail = f"{spec.metric}={value:g} (need {spec.op} {spec.threshold:g})"
    if spec.known_miss and not ok and not math.isnan(value):
        return True, f"KNOWN MISS {detail} [{KNOWN_MISS_NOTE}]"
    if spec.descriptive:
        return True, f"DESCRIPTIVE {detail}" + ("" if ok else " [below threshold, not gating]")
    return ok, detail


def run_phase(
    phase: str,
    factory: ClientFactory,
    *,
    results_dir: Path | None = records.RESULTS_DIR,
    gates_path: Path = GATES_YAML,
    only: set[str] | None = None,
    list_tools: Callable[[], list[dict[str, Any]]] | None = None,
    options: dict[str, Any] | None = None,
    split: str | None = None,
    cfg: Any = None,
) -> list[Outcome]:
    """One outcome per gate id: the suite runs once and every spec row of that id is evaluated.

    `split` (tune | test) is the data split the suites run on and is written to the record. Every
    row is evaluated on either split; a tune run is informational (the CLI says so), only a test
    run is a look (§5.4: logged in the ledger, ≤ 3, thresholds frozen first). `cfg` is the
    `mem` config the suites ran with (recorded; default: the pinned production config).
    """
    specs = load_gates(gates_path).get(phase)
    if specs is None:
        raise KeyError(f"no gates for phase {phase!r} in {gates_path.name}")
    by_id: dict[str, list[GateSpec]] = {}
    for spec in specs:
        if only and spec.id not in only:
            continue
        by_id.setdefault(spec.id, []).append(spec)
    outcomes = []
    for bench_id, rows in by_id.items():
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"bench-{bench_id}-") as tmp:
            try:
                res = suite_for(phase, bench_id)(
                    Context(Path(tmp), factory, list_tools, dict(options or {}))
                )
            except BackendUnavailable as e:
                outcomes.append(Outcome(bench_id, "UNAVAILABLE", str(e)))
                continue
            except Exception as e:  # noqa: BLE001 - a crashing suite is a failed gate, with the cause
                outcomes.append(
                    Outcome(bench_id, "FAIL", f"suite crashed: {type(e).__name__}: {e}")
                )
                continue
        verdicts = [(spec, *evaluate(spec, res)) for spec in rows]
        is_def = {spec.metric: spec.metric in res.deferred and spec.metric not in res.metrics
                  for spec in rows}  # fmt: skip
        ok = all(v[1] for v in verdicts if not is_def[v[0].metric])
        if any(is_def.values()):  # a partial run is reported only if nothing it ran went wrong
            ok = ok and not res.violations
        detail = "; ".join(("" if v[1] or is_def[v[0].metric] else "FAILED ") + v[2]
                           for v in verdicts)  # fmt: skip
        if res.violations:
            detail += (
                "; " + "; ".join(res.violations[:5]) + (" ..." if len(res.violations) > 5 else "")
            )
        run_file = None
        if results_dir is not None:
            ds = res.extra.get("dataset") or {"name": "p1-fixtures", "version": 1}
            rec = records.RunRecord(
                run_id=records.new_run_id(bench_id, started), bench_id=bench_id, tier=res.tier,
                mode=res.mode,
                dataset={**ds, "split": split or rows[0].split, "n": len(res.items)},
                config=records.config_fingerprint(cfg), stubs=records.T0_STUBS if res.tier == "t0" else {},
                git_sha=records.git_sha(), hardware=records.hardware(), metrics=res.metrics,
                gates=[{"phase": phase, "metric": spec.metric, "op": spec.op,
                        "threshold": spec.threshold, "passed": None if is_def[spec.metric] or res.invalid else v_ok,
                        **({"descriptive": True} if spec.descriptive else {}),
                        **({"known_miss": True} if spec.known_miss and "KNOWN MISS" in d else {}),
                        **({"deferred": res.deferred[spec.metric]} if is_def[spec.metric] else {})}
                       for spec, v_ok, d in verdicts],
                violations=res.violations, n_items=len(res.items),
                wall_s=round(time.perf_counter() - t0, 3),
                started_at=started.strftime("%Y-%m-%dT%H:%M:%SZ"), notes=res.notes,
                extra={**{k: v for k, v in res.extra.items() if k != "dataset"},
                       **({"invalid": res.invalid} if res.invalid else {})},
            )  # fmt: skip
            run_file = records.write_run(rec, res.items, results_dir)
        status = ("DEFERRED" if any(is_def.values()) else "PASS") if ok else "FAIL"
        if res.invalid:
            status, detail = "INVALID", f"INVALID {res.invalid}; {detail}"
        outcomes.append(Outcome(bench_id, status, detail, run_file))
    return outcomes
