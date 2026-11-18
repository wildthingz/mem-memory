"""B-C3: write latency (benchmarks-v2 B-C: `remember` server time).

Aggregates `bench c3-run` outputs: 3 fresh-process runs per corpus (SYN-M replay at L5k and L50k;
20 warm-up + 300 measured `remember` calls with 1 client, then 300 more with 4 concurrent clients;
the first call after process start is `cold`, reported separately). Gate (provisional, P3): warm p95
<= 1,500 ms at L5k, the median of the 3 runs' p95. L50k is recorded (`warm_p95_ms_median_l50k`),
and so is the 4-client figure.
"""

from __future__ import annotations

import statistics

from bench import latency
from bench.suites.base import BackendUnavailable, Context, SuiteResult

OPT = "c3_runs"  # {"L5k": [paths], "L50k": [paths]}


def run(ctx: Context) -> SuiteResult:
    runs = ctx.options.get(OPT) or {}
    res = SuiteResult("B-C3", tier="t1", mode="oracle_client")
    for corpus in ("L5k", "L50k"):
        paths = runs.get(corpus, [])
        if len(paths) < latency.RUNS_REQUIRED:
            if corpus == "L5k":
                raise BackendUnavailable(
                    f"B-C3 needs {latency.RUNS_REQUIRED} `bench c3-run --corpus L5k` outputs, "
                    f"got {len(paths)}"
                )
            res.notes.append(f"L50k: {len(paths)} runs, not recorded")
            continue
        loaded = latency.load_runs(paths)
        agg = latency.aggregate(loaded)
        suffix = "" if corpus == "L5k" else "_l50k"
        res.metrics[f"warm_p95_ms_median{suffix}"] = agg["warm_p95_ms_median"]
        res.metrics[f"warm_p95_ms_max{suffix}"] = agg["warm_p95_ms_max"]
        res.metrics[f"warm_p95_4clients_ms_median{suffix}"] = agg["warm_p95_4clients_ms_median"]
        res.metrics[f"cold_ms_median{suffix}"] = agg["cold_ms_median"]
        res.items += [{"corpus": corpus, "run": str(p), "cold_ms": r["cold_ms"],
                       "p95": r["one_client"]["total"]["p95"]} for p, r in zip(paths, loaded,
                                                                              strict=True)]  # fmt: skip
        res.extra[f"fingerprints_{corpus}"] = sorted(
            {r.get("corpus_fingerprint", "") for r in loaded}
        )
    res.metrics["n_runs_l5k"] = float(len(runs.get("L5k", [])))
    res.notes.append(
        "B-C3 at L5k on the serving host; median of 3 fresh-process runs' warm p95; "
        f"median cold {statistics.median(i['cold_ms'] for i in res.items if i['corpus'] == 'L5k'):.0f} ms"
    )
    return res
