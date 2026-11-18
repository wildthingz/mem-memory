"""The exhaustive steps 4-5 search `mem calibrate` ran before MEM2-129, kept verbatim as the
reference the fast `calibrate._search` must agree with: every τ_keep candidate scanned, a dense
(τ_high, g) grid built for each, the front from every τ_keep's staircase."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mem.calibrate import (
    Choice,
    Target,
    _between,
    _pareto,
    _points,
    _shortfall,
    _staircase,
    _Table,
    _tau_keep_candidates,
)


def _grid_dense(
    table: _Table, tau_keep: float, gaps: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """The five metrics at (`tau_keep`, τ_high, g) for every τ_high in step 5's grid at this τ_keep
    (τ_keep itself, then the cuts between the p1 values above it) and every g in `gaps`, as (H, G)
    arrays. Query q is abstained by the gap rule exactly when τ_high > p1_q and g > d_q: a
    quadrant of the grid, so the totals removed are a 2-D cumulative sum of per-query counts."""
    q_idx = np.arange(len(table.p1))
    base_abstain = (~table.has) | (table.p1 < tau_keep)
    n = np.minimum((table.p >= tau_keep).sum(axis=1), table.limit)
    rows0 = np.where(base_abstain, 0, n)
    hits0 = table.cumrel[q_idx, rows0]
    active = ~base_abstain
    ans = table.answerable
    highs = np.array([tau_keep, *(c for c in _between(table.p1[active].tolist()) if c > tau_keep)])
    hi = np.searchsorted(highs, table.p1, side="right")
    gi = np.searchsorted(gaps, table.d, side="right")
    comps = np.stack([rows0, hits0, ans & (hits0 > 0), active & ans, active & ~ans, rows0 > 0],
                     axis=1).astype(float)  # fmt: skip
    acc = np.zeros((len(highs) + 1, len(gaps) + 1, comps.shape[1]))
    sel = active & (hi < len(highs)) & (gi < len(gaps))
    np.add.at(acc, (hi[sel], gi[sel]), comps[sel])
    removed = acc.cumsum(axis=0).cumsum(axis=1)[: len(highs), : len(gaps)]
    left = (
        comps.sum(axis=0) - removed
    )  # rows, hits, covered, answering ans, answering neg, nonempty
    rows, hits, covered, ans_on, neg_on, nonempty = (left[..., k] for k in range(6))
    n_ans, n_neg = int(ans.sum()), int((~ans).sum())
    shape = rows.shape
    prec = np.divide(hits, rows, out=np.zeros(shape), where=rows > 0)
    ira = np.divide(rows - hits, nonempty, out=np.zeros(shape), where=nonempty > 0)
    cov = covered / n_ans if n_ans else np.zeros(shape)
    far = (n_ans - ans_on) / n_ans if n_ans else np.zeros(shape)
    abst = (n_neg - neg_on) / n_neg if n_neg else np.full(shape, np.nan)
    return highs, {"precision": prec, "ira": ira, "coverage_any": cov, "abstention_recall": abst,
                   "far": far}  # fmt: skip


def _lexmax(ok: np.ndarray) -> tuple[int, int] | None:
    """The (h, j) of the True cell with the highest h, then the highest j: the strictest."""
    hs, js = np.nonzero(ok)
    if hs.size == 0:
        return None
    h = int(hs.max())
    return h, int(js[hs == h].max())


def search_exhaustive(table: _Table, targets: Sequence[Target]) -> Choice:
    """Steps 4-5 on a table (module docstring). τ_keep candidates are scanned strictest first; at
    each, every (τ_high, g) of step 5's grid is evaluated at once (`_grid`), and the first τ_keep
    with a feasible cell ends the search with that cell's strictest (τ_high, g). The grid's "gap
    rule off" cells are one setting, written as (τ_keep, τ_keep, 0).

    Infeasible: the scan completes, the Pareto front is built from each τ_keep's FAR staircase
    (`_staircase`), and the fallback is, in order: the strictest setting meeting coverage_any and
    FAR; else the strictest keeping the best reachable coverage_any (the pool-recall ceiling) with
    FAR met; else the open gate (0, 0, 0)."""
    gaps = np.array([0.0, *_between(table.d[table.has].tolist())])
    by_name = {t.gate_metric: t for t in targets}
    cov_t, far_t = by_name["coverage_any"], by_name["far"]
    ceiling = float(table.metrics(0.0, 0.0, 0.0)["coverage_any"])
    best: dict[str, float] = {}
    fallback: tuple[str, float, float, float] | None = None
    ceiling_fb: tuple[str, float, float, float] | None = None
    points: list[dict[str, Any]] = []

    def record_best(m: dict[str, np.ndarray], canonical: np.ndarray) -> None:
        for t in targets:
            v = float(t.better()(m[t.metric][canonical]))
            prev = best.get(t.gate_metric)
            best[t.gate_metric] = v if prev is None else float(t.better()([prev, v]))

    def finish(tau_keep: float, highs: np.ndarray, h: int, j: int, m: dict[str, np.ndarray],
               feasible: bool, rule: str) -> Choice:  # fmt: skip
        tau_high, gap = (tau_keep, 0.0) if h == 0 or j == 0 else (float(highs[h]), float(gaps[j]))
        tgt = {t.gate_metric: {"op": t.op, "threshold": t.threshold,
                               "best": round(best[t.gate_metric], 6),
                               "met": t.met(best[t.gate_metric]),
                               "value": round(float(np.nan_to_num(m[t.metric][h, j])), 6)}
               for t in targets}  # fmt: skip
        front = [] if feasible else sorted(_pareto(points), key=lambda pt: (len(pt["unmet"]),
                                                                            _shortfall(pt, targets),
                                                                            -pt["tau_keep"]))  # fmt: skip
        return Choice(tau_keep, tau_high, gap, feasible, rule, tgt, front)

    grids: dict[float, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for tau in _tau_keep_candidates(table):
        highs, m = _grid_dense(table, tau, gaps)
        hh, jj = np.meshgrid(np.arange(len(highs)), np.arange(len(gaps)), indexing="ij")
        canonical = (hh > 0) == (jj > 0)  # off only as (0, 0); on needs both knobs
        record_best(m, canonical)
        ok = canonical.copy()
        for t in targets:
            ok &= (m[t.metric] >= t.threshold - 1e-12 if t.op == ">=" else
                   m[t.metric] <= t.threshold + 1e-12)  # fmt: skip
        if (cell := _lexmax(ok)) is not None:
            return finish(tau, highs, *cell, m, True, "strictest")
        cov_ok = canonical & (m["coverage_any"] >= cov_t.threshold - 1e-12)
        far_ok = canonical & (m["far"] <= far_t.threshold + 1e-12)
        if fallback is None and (cell := _lexmax(cov_ok & far_ok)) is not None:
            fallback = ("fallback_coverage_far", tau, *cell)
            grids[tau] = (highs, m)
        at_ceiling = canonical & (m["coverage_any"] >= ceiling - 1e-12)
        if ceiling_fb is None and (cell := _lexmax(at_ceiling & far_ok)) is not None:
            ceiling_fb = ("fallback_coverage_ceiling", tau, *cell)
            grids[tau] = (highs, m)
        points += _points(tau, highs, gaps, m, *_staircase(m, canonical), targets)
    chosen = fallback or ceiling_fb
    if chosen is None:  # even (0, 0, 0) abstains on > FAR of the answerables: empty pools
        highs, m = _grid_dense(table, 0.0, gaps)
        return finish(0.0, highs, 0, 0, m, False, "fallback_open")
    rule, tau, h, j = chosen
    highs, m = grids[tau]
    return finish(tau, highs, h, j, m, False, rule)


def synth_pools(n, seed=0, *, shift=5.0, miss=0.02, neg_shift=0.0, size=30, neg=0.3):
    """`n` host-shaped pools of `size` items: gold (1-2 per answerable, a few missing) scored
    `shift` logits above the noise, negatives optionally shifted by `neg_shift`."""
    from mem.calibrate import PoolItem, QueryPool

    rng = np.random.default_rng(seed)
    out = []
    for q in range(n):
        ans = bool(rng.random() > neg)
        z = rng.normal(-3, 2, size)
        rel = np.zeros(size, int)
        if ans and rng.random() >= miss:
            k = int(rng.integers(1, 3))
            idx = rng.choice(size, k, replace=False)
            rel[idx] = 1
            z[idx] += rng.normal(shift, 1.5, k)
        elif not ans:
            z += neg_shift
        out.append(
            QueryPool(
                f"q{q}", ans, [PoolItem(f"c{i}", float(z[i]), int(rel[i])) for i in range(size)]
            )
        )
    return out
