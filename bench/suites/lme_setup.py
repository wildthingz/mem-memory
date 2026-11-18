"""What the P2 LongMemEval suites need from the outside, and the shared run (one pass per gate)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import lme_eval
from bench.datasets import lme
from bench.datasets.negatives import NegRow
from bench.suites.base import BackendUnavailable, Context

OPT = "lme"


@dataclass
class LmeSetup:
    backend: lme_eval.LmeBackend
    questions: dict[str, dict[str, Any]]
    manifest: list[lme.ManifestRow]
    negatives: list[NegRow]
    split: str  # tune | test
    dataset_meta: dict[str, Any] = field(default_factory=dict)  # file sha256, manifest sha256, ...
    token_counter: lme_eval.TokenCounter = lme_eval.est_counter
    token_counter_exact: bool = False  # True only for Anthropic count_tokens
    tune_record: dict[str, Any] | None = None  # the tune run's B-R1 metrics, for power and drift
    n_target: int | None = None  # test-set size for the power estimate
    latency_runs: list[Path] = field(default_factory=list)  # `bench latency-run` outputs (L5k)
    include_abs: bool = True
    bootstrap_b: int = 10_000
    cache: dict[str, Any] = field(default_factory=dict)


def setup_of(ctx: Context) -> LmeSetup:
    s = ctx.options.get(OPT)
    if s is None:
        raise BackendUnavailable(
            "no LongMemEval setup: needs the data file, manifest and a backend (mem P2 pipeline); "
            "see deliverables/runbooks/p2-gate.md"
        )
    return s


def shared_run(ctx: Context) -> tuple[LmeSetup, list[lme_eval.Item], lme_eval.Scores]:
    """B-R1, B-R2 (i), B-C2 and the P2 form of B-R6 read the same responses: run them once."""
    s = setup_of(ctx)
    if "items" not in s.cache:
        dropped = frozenset(r.question_id for r in s.negatives if r.audit_dropped)
        s.cache["items"] = lme_eval.run_lme(
            s.backend, s.questions, s.manifest, s.split,
            drop_negatives=dropped, include_abs=s.include_abs,
        )  # fmt: skip
        s.cache["scores"] = lme_eval.score(s.cache["items"], b=s.bootstrap_b, n_target=s.n_target)
    return s, s.cache["items"], s.cache["scores"]


def fill_result(res, s: LmeSetup, items: list[lme_eval.Item], sc: lme_eval.Scores) -> None:
    """Common fields of a P2 SuiteResult: items file, CIs, per-category values, dataset."""
    res.tier, res.mode = "t1", "ingest_verbatim"
    res.items = [i.record() for i in items]
    res.extra |= {"ci95": {k: list(v) for k, v in sc.ci95.items()}, "per_category": sc.per_category,
                  "n": sc.n, "split": s.split, "dataset": s.dataset_meta}  # fmt: skip
    res.notes += sc.notes


def tune_metric(s: LmeSetup, name: str) -> float:
    rec = s.tune_record or {}
    return float(rec.get("metrics", rec).get(name, float("nan")))


def load_tune_record(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
