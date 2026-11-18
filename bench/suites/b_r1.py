"""B-R1: LongMemEval_S retrieval, ingest-verbatim (benchmarks-v2 B-R1). T1; needs the P2 pipeline.

Metrics (definitions in bench/lme_eval.py): R_q Recall_any@5, nDCG@10 (std) beside the upstream
`ndcg_any@10` (kept apart, A6), S_q coverage/precision/IRA/FAR with the B-R2 (i) negatives in the
pool. `ndcg_std_superior` is 1 iff the paired-bootstrap CI lower bound of Δ vs the better single
leg is > 0; when the tune power for Δ is < 80% the comparison is reported only (value 1, noted).
On the test split the power comes from the tune record; without it the metric is NaN (gate fails).
"""

from __future__ import annotations

import math

from bench.suites.base import Context, SuiteResult
from bench.suites.lme_setup import fill_result, shared_run, tune_metric

POWER_MIN = 0.80


def run(ctx: Context) -> SuiteResult:
    s, items, sc = shared_run(ctx)
    res = SuiteResult("B-R1")
    fill_result(res, s, items, sc)
    res.metrics |= {k: v for k, v in sc.metrics.items() if not k.startswith("auroc")}
    lo = sc.metrics.get("ndcg_delta_ci_lo", math.nan)
    power = sc.metrics["ndcg_power"] if s.split == "tune" else tune_metric(s, "ndcg_power")
    res.metrics["ndcg_power_tune"] = power
    if math.isnan(lo) or math.isnan(power):
        res.metrics["ndcg_std_superior"] = math.nan
        res.notes.append("superiority not decidable: legs or tune power missing")
    elif power < POWER_MIN:
        res.metrics["ndcg_std_superior"] = 1.0
        res.notes.append(f"tune power {power:.2f} < 0.80: nDCG@10 (std) vs best leg is report-only "
                         f"(Δ={sc.metrics['ndcg_delta_vs_best_leg']:.4f}, CI lo {lo:.4f})")  # fmt: skip
    else:
        res.metrics["ndcg_std_superior"] = float(lo > 0)
    if not sc.n.get("answerable"):
        res.violations.append("no answerable items")
    return res
