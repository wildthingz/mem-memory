"""B-E1 gate suite: scores a recorded `bench e1-run` file (tier T2; the paid part never repeats)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench import e1
from bench.suites.base import BackendUnavailable, Context, SuiteResult

OPT = "e1"


@dataclass
class E1Setup:
    run_path: Path
    split: str  # tune | test: the run must have been made on the split the gate runs on
    # `--oracle-pool`: the other runs of the harness (e1.HARNESS_FIELDS) whose oracle answers pool
    # with this run's. None: the pool the run file recorded (`bench e1-run --oracle-pool`), its
    # files re-read and their sha256 checked; [] is this run alone.
    pool: list[Path] | None = None


def load(s: E1Setup) -> tuple[dict, list[dict]]:
    """The run record and its pool (`e1.load_pool`); raises `e1.PoolRefused` naming the cause."""
    rec = json.loads(Path(s.run_path).read_text(encoding="utf-8"))
    stored = rec.get("oracle_pool") or []
    paths = s.pool if s.pool is not None else [Path(m["file"]) for m in stored]
    try:
        pool = e1.load_pool(paths, rec)
    except OSError as e:
        raise e1.PoolRefused(f"a pooled run file cannot be read: {e}") from e
    if s.pool is None:
        for m, got in zip(stored, pool, strict=True):
            if m["sha256"] != got["sha256"]:
                raise e1.PoolRefused(f"{m['file']}: sha256 {got['sha256'][:12]} is not the "
                                     f"{m['sha256'][:12]} the run recorded (the file changed)")  # fmt: skip
    e1.oracle_pool(rec, pool)  # the current run listed in its own pool is refused here too
    return rec, pool


def run(ctx: Context) -> SuiteResult:
    s = ctx.options.get(OPT)
    if s is None:
        raise BackendUnavailable(
            "B-E1 needs a recorded run: `bench e1-run ...` (T2, Anthropic key), then "
            "`bench gate P3 --e1-run <file>` (deliverables/runbooks/p3-gate.md, Round 8)"
        )
    res = SuiteResult("B-E1", tier="t2", mode="ingest_verbatim")
    rec, pool = load(s)
    metrics, detail = e1.score_run(rec, pool=pool)
    res.metrics |= metrics
    if rec.get("split") != s.split:
        res.violations.append(
            f"the run was made on split {rec.get('split')!r}, the gate is on {s.split!r}"
        )
        res.metrics.pop(
            "accuracy", None
        )  # a run of another split proves nothing here: not produced
        res.metrics["run_complete"] = 0.0
    res.invalid = e1.pool_invalid_reason(detail["oracle"])
    if "mem" not in rec["conditions"]:
        res.violations.append("the run has no mem condition")
    if not rec["complete"]:
        res.notes.append(f"incomplete run: {rec.get('stopped') or 'fewer items than planned'}")
    if metrics["judge_flagged"]:
        res.notes.append(f"{int(metrics['judge_flagged'])} {e1.UNPARSED}")
    if rec.get("judge_version") != e1.JUDGE_VERSION:
        res.notes.append(f"the run's judge_version {rec.get('judge_version')!r} is not the harness's "
                         f"{e1.JUDGE_VERSION} (ADR 0029: the judge's cap and reply parsing changed; "
                         "re-scored here with the current parser)")  # fmt: skip
    res.items = rec["items"]
    res.extra |= {
        "dataset": {**rec["dataset"], "n": len({i["qid"] for i in rec["items"]})},
        "per_condition": detail["per_condition"], "cost": rec["cost"],
        "answerer": rec["answerer"], "judge": rec["judge"], "conditions": rec["conditions"],
        "prompts_sha256": rec["prompts_sha256"],
        "prompt_version": rec.get("prompt_version"), "judge_version": rec.get("judge_version"),
        "run_config": rec.get("config"),
        "run_file": str(s.run_path), "oracle": detail["oracle"],
    }  # fmt: skip
    return res
