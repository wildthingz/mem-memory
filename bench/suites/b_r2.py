"""B-R2 (i): evidence-removed abstention (benchmarks-v2 B-R2). T1; needs the P2 pipeline.

`far_drift` = |FAR_test - FAR_tune|, taken from the tune record (NaN on tune, so the gate can only
pass on test with a tune record). τ_keep, τ_high and g are frozen in the store's gate by
`mem calibrate` on tune before this is run on test.
"""

from __future__ import annotations

import math

from bench.suites.base import Context, SuiteResult
from bench.suites.lme_setup import fill_result, shared_run, tune_metric


def run(ctx: Context) -> SuiteResult:
    s, items, sc = shared_run(ctx)
    res = SuiteResult("B-R2")
    fill_result(res, s, items, sc)
    keys = (
        "abstention_recall",
        "far",
        "auroc_p1",
        "auroc_p1_gap",
        "abs_abstention_recall",
        "abs_n",
    )
    res.metrics |= {k: sc.metrics[k] for k in keys if k in sc.metrics}
    far_tune = tune_metric(s, "far")
    res.metrics["far_tune"] = far_tune
    res.metrics["far_drift"] = (abs(sc.metrics["far"] - far_tune)
                                if s.split == "test" and not math.isnan(far_tune) else math.nan)  # fmt: skip
    if not sc.n.get("negative"):
        res.violations.append("no negative items")
    return res
