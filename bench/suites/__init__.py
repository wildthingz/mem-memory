"""Suite registry: bench id -> run(ctx) -> SuiteResult."""

from __future__ import annotations

from collections.abc import Callable

from bench.suites import (
    b_c,
    b_c3,
    b_e1,
    b_p2_cost,
    b_p4,
    b_p5,
    b_p6,
    b_r1,
    b_r2,
    b_r5,
    b_r5_p3,
    b_r6,
    b_s1,
    b_w5,
    b_w6,
    b_w6_p3,
    b_w123,
)
from bench.suites.base import Context, SuiteResult

REGISTRY: dict[str, Callable[[Context], SuiteResult]] = {
    "B-S1": b_s1.run,
    "B-R6": b_r6.run,
    "B-W6": b_w6.run,
    "B-R5": b_r5.run,
    "B-C1": b_c.run_b_c1,
    "B-C2": b_c.run_b_c2,
}

# Ids first gated or re-defined in a later phase: P2 swaps in the P2 forms (LongMemEval, calibrated
# responses). Ids not listed here use REGISTRY. `bench gate <phase>` resolves through `suite_for`.
PHASE_REGISTRY: dict[str, dict[str, Callable[[Context], SuiteResult]]] = {
    "P2": {
        "B-R1": b_r1.run,
        "B-R2": b_r2.run,
        "B-R6": b_p2_cost.run_b_r6,
        "B-C1": b_p2_cost.run_b_c1,
        "B-C2": b_p2_cost.run_b_c2,
    },
}


PHASE_REGISTRY["P3"] = {
    "B-W1": b_w123.run_b_w1,
    "B-W2": b_w123.run_b_w2,
    "B-W3": b_w123.run_b_w3,
    "B-W5": b_w5.run,
    "B-W6": b_w6_p3.run,
    "B-R5": b_r5_p3.run_b_r5,
    "B-R2": b_r5_p3.run_b_r2_p3,
    "B-R1": b_r1.run,
    "B-C3": b_c3.run,
    "B-E1": b_e1.run,
}


PHASE_REGISTRY["P4"] = {
    "B-I1": b_p4.run_b_i1,
    "B-I2": b_p4.run_b_i2,
    "B-I3": b_p4.run_b_i3,
    "B-I4": b_p4.run_b_i4,
    "B-I5": b_p4.run_b_i5,
    "B-C4": b_p4.run_b_c4,
    "B-R3": b_p4.run_b_r3,
    "B-C1": b_p4.run_b_c1,
    "B-C3": b_p4.run_b_c3,
    "B-R1": b_r1.run,
    "B-R2": b_r2.run,
}


PHASE_REGISTRY["P5"] = {"B-E5": b_p5.run_b_e5, "B-C6": b_p5.run_b_c6}


PHASE_REGISTRY["P6"] = {"B-E4": b_p6.run_b_e4, "B-R3-PGv2": b_p6.run_pgv2}


def suite_for(phase: str, bench_id: str) -> Callable[[Context], SuiteResult]:
    return PHASE_REGISTRY.get(phase, {}).get(bench_id) or REGISTRY[bench_id]
