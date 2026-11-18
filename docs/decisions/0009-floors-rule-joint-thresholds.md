# 0009: τ_keep, τ_high and g by the "floors" rule (strictest setting meeting all five P2 tune targets)

Status: accepted (decided by the owner, on MEM2-1; implemented in MEM2-87) · Spec: architecture-v2 §3.3 steps 4–5 (spec change dated 2026-10-24) · Fixes: MEM2-85 finding N1

## Decision
1. **One joint choice.** `mem calibrate` picks (τ_keep, τ_high, g) together as the strictest setting that meets
   all five P2 tune targets at once: delivered precision ≥ 0.70, IRA ≤ 0.5, coverage_any ≥ 0.85, abstention
   recall ≥ 0.85, FAR ≤ 0.05. Targets are read from `bench/configs/gates.yaml` (B-R1, B-R2), never copied.
2. **Strictest** = highest τ_keep, then highest τ_high, then highest g (`mem.calibrate.STRICTEST`). The
   "gap rule off" cells of the grid are one setting, written (τ_keep, τ_keep, 0). Rejected alternative:
   "most tune abstentions" as the tie-break; it is defensible too, but the person named the order above,
   and switching is a one-line change in `_lexmax`'s key.
3. **Infeasible = reported, not relaxed.** `feasible: false`, the Pareto front on tune (non-dominated
   settings over the five numbers, built from each τ_keep's FAR staircase, i.e. step 5's candidate at
   every FAR level) with the targets each point misses, and a flagged fallback gate: strictest meeting
   coverage_any and FAR; else strictest keeping the pool-recall coverage ceiling with FAR met; else (0, 0, 0).
   `mem serve` serves it with a WARNING. Rejected: refusing to write a gate (the person could not see
   the numbers in use, and `mem serve` would not start at all).
4. **Where it is recorded.** `meta.gate.tune`: `feasible`, `rule`, `strictest`, `targets` (op, threshold,
   best reachable, met, value at the chosen setting), the five numbers, `front` (infeasible only).
   `bench gate` run records: `dataset.gate` carries the same (`bench.cli.gate_record`).

## Metric definitions used
benchmarks-v2 §2.0, on S_q = the kept rows (top `limit`) of every tune query, answerable and negative:
delivered precision, IRA, coverage_any, FAR, abstention recall. Platt is unchanged (step 3).

## Candidate grid (exact for the data)
τ_keep: 0 and the p of every item ranked ≤ `limit` in its query. τ_high: τ_keep and the cuts between the
p1 values above it (top cut 1.0). g: 0 and the cuts between the p1 − p2 gaps. Values within 1e-9 are one.
Per τ_keep every (τ_high, g) is evaluated at once by a 2-D cumulative sum (`_grid`), checked against
`_Table.metrics` cell by cell in `tests/test_gate.py::test_grid_agrees_with_the_per_setting_metrics`.

## Evidence (T0, synthetic host-shaped pools, `tests/test_gate.py::_bge_like_pools`)
- Feasible shape (3 all-low pools, gold ~ N(3, 1.5)): τ_keep at z ≈ 0.85 (was the 3rd-weakest gold at
  z ≈ −7.6); tune precision 0.889, IRA 0.164, coverage_any 0.948, abstention 0.896, FAR 0.043.
- Infeasible shape (19 of 115 answerables with no gold in the pool): coverage ceiling 0.835 < 0.85;
  front of 44 settings; fallback = open gate flagged infeasible. The fit takes ~10 s on 230 queries.
- These are synthetic; the host numbers come from runbook p2-gate.md §4 when the person runs it.
