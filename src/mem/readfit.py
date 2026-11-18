"""The P3 read-gate fit (architecture-v2 §3.3 step 4b, MEM2-155; objective: the owner, 2026-10-25).

Two changes against the P2 fit (`mem.calibrate.fit_gate`):

1. The probability is a small logistic over generic signals (`mem.readmodel`: rerank logit, query-
   chunk cosine, pool gap) instead of Platt on the logit alone. Which signals and how strongly it
   is regularised (C) are chosen by nested CV; the coefficients go into `meta.gate.read_model`.
2. The thresholds (τ_keep, τ_high, g) target the gates that consume them, read from gates.yaml (P3
   rows): B-R2 (ii) missing_slot_abstention, B-R5 current_recall_at_1 and as_of_accuracy, and B-R2
   (i) FAR, all on SYN dev, each as the worst dev seed (the gate's own aggregation, benchmarks-v2
   §5.1). The LME numbers (delivered precision, IRA, coverage_any, abstention recall) stay
   descriptive; LME enters the model fit as training data, and as two guards: recall_any5 on LME
   tune by the model's order must not fall below the order by the raw logit (P2 hard row), and the
   LME no-regression guard (MEM2-176): a setting is eligible only if on the LME tune pools its
   coverage_any is >= and its share of answerable queries with an empty result is <= those of the
   P2 floors gate fitted on the same training records. It is a constraint of the threshold search
   (each fold refits the P2 reference on that fold's training pools only), not a report line, so
   missing-slot abstention cannot be bought by holding back answerable LME results.

Strictest setting = highest τ_keep, then τ_high, then g, as in the P2 floors rule. Infeasible: no
target is relaxed; the fit says so, and writes the maximin setting (the largest worst relative
slack) among the settings that keep the LME no-regression guard as a flagged fallback. When no
setting keeps the guard at all, no gate is written (`GuardUnmet`).

Honest estimate: NESTED leave-one-seed-out CV over the dev seeds. For each outer seed the
configuration (feature set, C) is chosen by inner leave-one-seed-out CV on the other seeds only,
the model and thresholds are refitted on them, and the outer seed is scored once. The final model
is chosen on all dev seeds (flat CV: optimistic, labelled so).

Metrics from pools. A pool has no row text, so `current_recall_at_1` and `as_of_accuracy` count a
query as hit when the gate delivers and the top-ranked pool item is the gold memory (rel >= 1).
B-R5 proper looks at the top-ranked *slot* memory, so the as-of figure here is a lower bound.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mem import calibrate
from mem.calibrate import PoolItem, QueryPool, Target
from mem.gate import Gate
from mem.readmodel import (
    FEATURES,
    MARGINS,
    MASKED,
    PHRASE,
    SIGNALS,
    ReadModel,
    feature_matrix,
    sigmoid,
)

P3_ROWS = {
    "missing_slot_abstention": ("B-R2", "missing_slot_abstention"),
    "current_recall_at_1": ("B-R5", "current_recall_at_1"),
    "as_of_accuracy": ("B-R5", "as_of_accuracy"),
    "far": ("B-R2", "far"),
}
CAT_MISSING, CAT_CURRENT, CAT_ASOF = "syn:missing_slot", "syn:current", "syn:as_of"
FEATURE_SETS = (("rerank",), ("rerank", "cosine"), ("rerank", "cosine", "gap"))
# MEM2-165: the subject-masked signals, alone (replacing the topical ones) and with the base set.
# Only in the grid when every record carries them (`MASKED_GRID`).
_BASE = ("rerank", "cosine", "gap")
MASKED_SETS = (
    ("cosine_masked",),
    ("rerank_masked",),
    ("cosine_masked", "rerank_masked"),
    ("rerank", "cosine_masked"),
    ("rerank", "cosine", "gap", "cosine_masked"),
    ("rerank", "cosine", "gap", "rerank_masked"),
    ("rerank", "cosine", "gap", "cosine_masked", "rerank_masked"),
)
# MEM2-171 (one bounded pass): the masked margins (best minus second best of a masked signal in
# the pool; `MARGIN_GRID`, with the masked records), the relation-phrase match (`PHRASE_GRID`, when
# every record carries pcos/pov) and both together (`BOTH_GRID`).
MARGIN_SETS = (
    ("cosine_masked_margin",),
    ("rerank_masked_margin",),
    ("cosine_masked", "cosine_masked_margin"),
    ("rerank_masked", "rerank_masked_margin"),
    ("rerank", "cosine_masked", "cosine_masked_margin"),
    (*_BASE, "cosine_masked", "rerank_masked", "cosine_masked_margin", "rerank_masked_margin"),
)
PHRASE_SETS = (
    ("phrase_cosine",),
    ("phrase_overlap",),
    ("phrase_cosine", "phrase_overlap"),
    ("rerank", "phrase_cosine"),
    (*_BASE, "phrase_cosine"),
    (*_BASE, "phrase_cosine", "phrase_overlap"),
)
BOTH_SETS = (
    ("rerank", "cosine_masked", "cosine_masked_margin", "phrase_cosine"),
    (*_BASE, "cosine_masked", "rerank_masked", "phrase_cosine"),
    (
        *_BASE,
        "cosine_masked",
        "rerank_masked",
        "cosine_masked_margin",
        "rerank_masked_margin",
        "phrase_cosine",
        "phrase_overlap",
    ),
)
ALL_SETS = (*FEATURE_SETS, *MASKED_SETS, *MARGIN_SETS, *PHRASE_SETS, *BOTH_SETS)
CS = (0.1, 1.0, 10.0)
GRID = [{"features": list(f), "C": c} for f in FEATURE_SETS for c in CS]
MASKED_GRID = [{"features": list(f), "C": c} for f in MASKED_SETS for c in CS]
MARGIN_GRID = [{"features": list(f), "C": c} for f in MARGIN_SETS for c in CS]
PHRASE_GRID = [{"features": list(f), "C": c} for f in PHRASE_SETS for c in CS]
BOTH_GRID = [{"features": list(f), "C": c} for f in BOTH_SETS for c in CS]
MAX_AXIS = 120  # τ_high / g cut points per axis before the grid is thinned (`exact` False)
TAU_CAP = 150  # τ_keep candidates scanned before the scan is thinned (`exact` False)
FALLBACK_TAUS = 150
TRAIN_MARGIN = 0.03  # absolute margin on the training seeds (the held-out seed is judged without)
_SEED = re.compile(r"^syn(\d+):")
GUARD = "lme_no_regression"  # the MEM2-176 guard's name in `met` and the infeasible messages


def p3_targets(gates_path: Path | None = None) -> tuple[Target, ...]:
    """The four P3 consuming-gate targets as gates.yaml holds them (never hard-coded here)."""
    return calibrate._targets("P3", P3_ROWS, gates_path)


def seed_of(qid: str) -> int | None:
    m = _SEED.match(qid)
    return int(m.group(1)) if m else None


def is_syn(pool: QueryPool) -> bool:
    return seed_of(pool.qid) is not None


# ---- data ---------------------------------------------------------------------------------------


class _Data:
    """Pools as arrays: per pool the full feature matrix, labels, seed, category."""

    def __init__(self, pools: Sequence[QueryPool], limit: int | None = None) -> None:
        """`limit`: rows per answer, for the LME no-regression guard; None leaves the guard off
        (unit tests of the SYN side only; `fit_p3_gate` always passes it)."""
        if not pools:
            raise ValueError("no pools to fit")
        if any(i.cos is None for q in pools for i in q.items):
            raise ValueError(
                "records without query-chunk cosines: re-score them (bench syn-score-dev, mem"
                " calibrate --labels) or fit the Platt gate (mem calibrate without --objective p3)"
            )
        # the subject-masked columns (and their margins) exist only when every record carries them
        # (MEM2-165); the phrase columns likewise (MEM2-171)
        self.has_masked = all(v is not None for q in pools for i in q.items for v in (i.mcos, i.mz))
        self.has_phrase = all(
            v is not None for q in pools for i in q.items for v in (i.pcos, i.pov)
        )
        names = tuple(
            f for f in FEATURES
            if (self.has_masked or (f not in MASKED and f not in MARGINS))
            and (self.has_phrase or f not in PHRASE)
        )  # fmt: skip
        self.pools = list(pools)
        self.x = [
            feature_matrix(
                [i.z for i in q.items],
                [i.cos for i in q.items],
                names,
                {k: [i.masked()[k] for i in q.items] for k in SIGNALS},
            )
            for q in pools
        ]
        self.y = [np.array([i.rel >= 1 for i in q.items], dtype=bool) for q in pools]
        self.seed = np.array([-1 if (s := seed_of(q.qid)) is None else s for q in pools])
        self.cat = np.array([q.category for q in pools], dtype=object)
        self.ans = np.array([q.answerable for q in pools], dtype=bool)
        self.names = names
        self.limit = limit
        self.memo: dict[Any, Any] = {}  # (train, test, config) -> fit; deterministic, so shareable

    def cols(self, features: Sequence[str]) -> list[int]:
        return [self.names.index(f) for f in features]

    def subset(self, mask: np.ndarray) -> _Data:
        return _Data(self.pools_of(mask), self.limit)

    def pools_of(self, mask: np.ndarray) -> list[QueryPool]:
        return [q for q, m in zip(self.pools, mask.tolist(), strict=True) if m]

    @property
    def syn_seeds(self) -> list[int]:
        return sorted({int(s) for s in self.seed if s >= 0})


def fit_model(data: _Data, mask: np.ndarray, features: Sequence[str], c: float) -> ReadModel:
    """Logistic of rel >= 1 on `features` over the items of the pools in `mask`. Standardised for
    the optimiser, folded back to raw coefficients, rounded to 8 decimals (what the gate stores)."""
    from sklearn.linear_model import LogisticRegression

    idx = data.cols(features)
    rows = np.flatnonzero(mask)
    x = np.vstack([data.x[r][:, idx] for r in rows])
    y = np.concatenate([data.y[r] for r in rows])
    if y.all() or not y.any():
        raise ValueError("the read model needs both relevant and irrelevant pool items")
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    lr = LogisticRegression(C=c, solver="lbfgs", max_iter=5000)
    lr.fit((x - mu) / sd, y)
    w = lr.coef_[0] / sd
    b = float(lr.intercept_[0] - (w * mu).sum())
    return ReadModel.build(list(features), [round(float(v), 8) for v in w], round(b, 8))


def _probs(data: _Data, model: ReadModel) -> list[np.ndarray]:
    idx = data.cols(model.features)
    w = np.asarray(model.weights)
    return [sigmoid(x[:, idx] @ w + model.bias) for x in data.x]


# ---- the objective ------------------------------------------------------------------------------


@dataclass
class _View:
    """Per query what the four metrics need: p1, the p1 - p2 gap, whether the top item is gold, the
    class (0 other, 1 current, 2 as-of, 3 missing) and the seed index."""

    p1: np.ndarray
    d: np.ndarray
    good: np.ndarray
    kind: np.ndarray
    seed: np.ndarray
    seeds: list[int]
    has: np.ndarray


def _view(data: _Data, probs: list[np.ndarray], mask: np.ndarray) -> _View:
    rows = [r for r in np.flatnonzero(mask) if data.seed[r] >= 0]
    seeds = sorted({int(data.seed[r]) for r in rows})
    p1, d, good, kind, sd, has = [], [], [], [], [], []
    for r in rows:
        p = probs[r]
        order = sorted(range(len(p)), key=lambda i: (-p[i], i))
        has.append(bool(order))
        top = p[order[0]] if order else -1.0
        p2 = p[order[1]] if len(order) > 1 else 0.0
        p1.append(top)
        d.append(top - p2 if order else 0.0)
        good.append(bool(order) and bool(data.y[r][order[0]]))
        cat = data.cat[r]
        kind.append({CAT_CURRENT: 1, CAT_ASOF: 2, CAT_MISSING: 3}.get(cat, 0))
        sd.append(seeds.index(int(data.seed[r])))
    return _View(
        np.array(p1),
        np.array(d),
        np.array(good, dtype=bool),
        np.array(kind, dtype=int),
        np.array(sd, dtype=int),
        seeds,
        np.array(has, dtype=bool),
    )


def _comps(v: _View) -> np.ndarray:
    """(queries, 4·seeds) indicators: missing, good current, good as-of, answerable (current or
    as-of), each in its seed's block."""
    n, s = len(v.p1), len(v.seeds)
    ind = np.stack(
        [
            v.kind == 3,
            (v.kind == 1) & v.good,
            (v.kind == 2) & v.good,
            (v.kind == 1) | (v.kind == 2),
        ],
        axis=1,
    ).astype(float)
    out = np.zeros((n, 4 * s))
    for k in range(4):
        out[np.arange(n), v.seed * 4 + k] = ind[:, k]
    return out


def _denoms(v: _View) -> np.ndarray:
    """(seeds, 4) query counts: missing, current, as-of, answerable."""
    s = len(v.seeds)
    den = np.zeros((s, 4))
    for i in range(s):
        k = v.kind[v.seed == i]
        den[i] = [(k == 3).sum(), (k == 1).sum(), (k == 2).sum(), ((k == 1) | (k == 2)).sum()]
    return den


def _metrics_from_kept(kept: np.ndarray, den: np.ndarray) -> dict[str, np.ndarray]:
    """Per-seed metric arrays from kept counts (4·seeds, ...) and the denominators."""
    s = den.shape[0]
    k = kept.reshape(s, 4, *kept.shape[1:])
    shape = (s,) + (1,) * (kept.ndim - 1)

    def ratio(i: int) -> np.ndarray:
        dn = den[:, i].reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(dn > 0, k[:, i] / np.where(dn > 0, dn, 1), np.nan)

    return {
        "missing_slot_abstention": 1.0 - ratio(0),
        "current_recall_at_1": ratio(1),
        "as_of_accuracy": ratio(2),
        "far": 1.0 - ratio(3),
    }


def _worst(m: dict[str, np.ndarray], targets: Sequence[Target]) -> dict[str, np.ndarray]:
    """The gate's aggregation: the worst seed (min for >=, max for <=), seeds without the class
    ignored (NaN when none has it)."""
    out = {}
    for t in targets:
        a = m[t.metric]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a class absent from every seed: NaN
            out[t.metric] = np.nanmin(a, axis=0) if t.op == ">=" else np.nanmax(a, axis=0)
    return out


def _slack(t: Target, v: np.ndarray) -> np.ndarray:
    """Relative margin to the threshold (positive = met); NaN (class absent) counts as met."""
    s = (v - t.threshold) / t.threshold if t.op == ">=" else (t.threshold - v) / t.threshold
    return np.where(np.isnan(s), np.inf, s)


def evaluate(
    v: _View, tau_keep: float, tau_high: float, gap: float, targets: Sequence[Target]
) -> dict[str, Any]:
    """The four numbers per seed and worst, for one setting."""
    abstain = (~v.has) | (v.p1 < tau_keep) | ((v.p1 < tau_high) & (v.d < gap))
    kept = (_comps(v) * (~abstain)[:, None]).sum(axis=0)
    per = _metrics_from_kept(kept, _denoms(v))
    worst = _worst(per, targets)
    return {
        "per_seed": {str(sd): {k: _num(per[k][i]) for k in per} for i, sd in enumerate(v.seeds)},
        "worst": {k: _num(x) for k, x in worst.items()},
    }


def _num(x: Any) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 6)


def _met(targets: Sequence[Target], worst: dict[str, Any]) -> dict[str, bool]:
    return {t.gate_metric: worst[t.metric] is None or t.met(worst[t.metric]) for t in targets}


# ---- the threshold search -----------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    tau_keep: float
    tau_high: float
    gap: float
    feasible: bool
    rule: str  # strictest | fallback_maximin | fallback_maximin_guard_unmet
    worst: dict[str, float | None]
    exact: bool = True
    # the LME no-regression guard at this setting (None: no guard, see `_Data`): coverage_any,
    # empty_answerable, the P2 reference values, ok
    guard: dict[str, Any] | None = None
    blocked_by_guard: bool = False  # some setting met the SYN targets, none of those kept the guard

    @property
    def guard_ok(self) -> bool:
        return self.guard is None or bool(self.guard["ok"])


@dataclass
class _LmeGuard:
    """The LME no-regression guard for one model on one training set: the LME tune pools under the
    model (`calibrate._Table`, the delivered rows) and the P2 floors gate's values on the same
    records (`coverage_any` and `empty_answerable` = the share of answerable queries with 0 rows,
    `_Table`'s `far`)."""

    table: calibrate._Table
    coverage_any: float
    empty_answerable: float

    def cells(self, tau: float, highs: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        """(H, G) bools: the guard holds at (`tau`, τ_high, g)."""
        t = self.table
        active = t.has & (t.p1 >= tau)
        n = np.minimum((t.p >= tau).sum(axis=1), t.limit)  # rows a delivered query gets
        hit = t.cumrel[np.arange(len(t.p1)), n] > 0
        comps = np.stack([hit & t.answerable, t.answerable], axis=1).astype(float)
        kept = _kept(t.p1, t.d, active, comps, np.asarray(highs), np.asarray(gaps))
        n_ans = int(t.answerable.sum())
        cov, empty = kept[0] / n_ans, 1.0 - kept[1] / n_ans
        return (cov >= self.coverage_any - 1e-12) & (empty <= self.empty_answerable + 1e-12)

    def at(self, tau_keep: float, tau_high: float, gap: float) -> dict[str, Any]:
        m = self.table.metrics(tau_keep, tau_high, gap)
        cov, empty = _num(m["coverage_any"]), _num(m["far"])
        ok = cov is not None and empty is not None and cov >= self.coverage_any - 1e-12
        return {
            "coverage_any": cov,
            "empty_answerable": empty,
            "p2_coverage_any": self.coverage_any,
            "p2_empty_answerable": self.empty_answerable,
            "ok": bool(ok and empty <= self.empty_answerable + 1e-12),
        }


def _cuts(values: np.ndarray, floor: float | None = None) -> list[float]:
    cut = calibrate._between(values.tolist()) if len(values) else []
    return [c for c in cut if floor is None or c > floor]


def _thin(a: np.ndarray, cap: int) -> np.ndarray:
    return calibrate._thin(a, cap)


def _grid(
    v: _View, comps: np.ndarray, den: np.ndarray, tau: float, targets: Sequence[Target]
) -> tuple:
    """For one τ_keep: the worst-seed metrics over (τ_high, g) cells as (H, G) arrays, via the
    quadrant counts of `calibrate._grid_blocks` (a query is gap-abstained iff p1 < τ_high and
    d < g). Returns (highs, gaps, metrics, exact)."""
    active = v.has & (v.p1 >= tau)
    highs = np.array([tau, *_cuts(v.p1[active], tau)])
    gaps = np.array([0.0, *_cuts(v.d[active])])
    exact = len(highs) <= MAX_AXIS and len(gaps) <= MAX_AXIS
    highs, gaps = _thin(highs, MAX_AXIS), _thin(gaps, MAX_AXIS)
    kept = _kept(v.p1, v.d, active, comps, highs, gaps)
    return highs, gaps, _worst(_metrics_from_kept(kept, den), targets), exact


def _kept(
    p1: np.ndarray, d: np.ndarray, active: np.ndarray, comps: np.ndarray,
    highs: np.ndarray, gaps: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """(columns of `comps`, H, G) sums over the `active` queries the gap rule keeps at each
    (τ_high, g) cell: a query is gap-abstained iff p1 < τ_high and d < g."""
    n_h, n_g = len(highs), len(gaps)
    hi = np.searchsorted(highs, p1, side="right")
    gi = np.searchsorted(gaps, d, side="right")
    sel = active & (hi < n_h) & (gi < n_g)
    flat = hi[sel] * n_g + gi[sel]
    removed = np.stack(
        [
            np.bincount(flat, weights=comps[sel, k], minlength=n_h * n_g)
            for k in range(comps.shape[1])
        ]
    ).reshape(comps.shape[1], n_h, n_g)
    removed = removed.cumsum(axis=1).cumsum(axis=2)
    return comps[active].sum(axis=0)[:, None, None] - removed


def search(
    v: _View, targets: Sequence[Target], margin: float = TRAIN_MARGIN, guard: _LmeGuard | None = None
) -> Setting:  # fmt: skip
    """The strictest (τ_keep, τ_high, g) meeting every target on the training seeds with `margin`
    to spare (the strictest setting sits on the edge of the targets, so an exact-edge fit would miss
    them on a new seed by chance; the held-out numbers are always judged against the real
    thresholds). When no setting keeps the margin, the exact targets; when none meets those, the
    flagged maximin fallback. `guard` (the LME no-regression guard, no margin) constrains every
    pass and the fallback. See `_search`."""
    tight = tuple(
        Target(t.gate_metric, t.metric, t.op, t.threshold + (margin if t.op == ">=" else -margin))
        for t in targets
    )
    st = _search(v, tight, targets, margin_used=margin > 0, guard=guard)
    return st if st.feasible else _search(v, tuple(targets), targets, False, guard)


def _search(
    v: _View, targets: Sequence[Target], real: Sequence[Target], margin_used: bool = True,
    guard: _LmeGuard | None = None,
) -> Setting:  # fmt: skip
    """The strictest (τ_keep, τ_high, g) meeting `targets` and `guard`; the reported numbers and
    the infeasible fallback are judged against `real` (see the module docstring). The fallback
    only takes settings that keep `guard`; the guard is best at τ_keep 0 with the gap rule off
    (rows and deliveries only shrink as either rises), so when it fails there it fails everywhere
    and the fallback (rule `fallback_maximin_guard_unmet`, never written as a gate) ignores it."""
    comps, den = _comps(v), _denoms(v)
    taus = sorted({0.0, *v.p1[v.has].tolist()}, reverse=True)
    exact = True
    by = {t.metric: t for t in targets}
    off = {}  # gap rule off at each τ_keep: the best case for recall and FAR
    for tau in taus:
        kept = (comps * (v.has & (v.p1 >= tau))[:, None]).sum(axis=0)
        off[tau] = _worst(_metrics_from_kept(kept, den), targets)

    def passes(tau: float) -> bool:
        return all(
            bool(np.all(_slack(by[m], off[tau][m]) >= -1e-12))
            for m in ("current_recall_at_1", "as_of_accuracy", "far")
            if m in by
        )

    def setting(tau, highs, gaps, h, j, m, rule, feasible, ex, blocked=False) -> Setting:
        t_high, g = (tau, 0.0) if h == 0 or j == 0 else (float(highs[h]), float(gaps[j]))
        worst = {k: _num(x[h, j]) for k, x in m.items()}
        gd = None if guard is None else guard.at(tau, t_high, g)
        return Setting(tau, t_high, g, feasible, rule, worst, ex, gd, blocked)

    def guard_cells(tau: float, highs: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        if guard is None:
            return np.ones((len(highs), len(gaps)), dtype=bool)
        return guard.cells(tau, highs, gaps)

    def guard_off(tau: float) -> bool:  # the guard at τ_keep with the gap rule off: its best case
        return bool(guard_cells(tau, np.array([tau]), np.array([0.0]))[0, 0])

    scan = [tau for tau in taus if passes(tau)]
    if len(scan) > TAU_CAP:
        scan, exact = _thin(np.array(scan), TAU_CAP).tolist()[::-1], False
        scan = sorted(scan, reverse=True)
    syn_met = False  # some setting met `targets` (the guard aside): names the guard if infeasible
    for tau in scan:
        g_off = guard_off(tau)
        if not g_off and syn_met:
            continue  # the guard fails at this τ_keep whatever τ_high and g are
        highs, gaps, m, ex = _grid(v, comps, den, tau, targets)
        exact &= ex
        canon = calibrate._canonical(0, len(highs), len(gaps))
        ok = np.ones(next(iter(m.values())).shape, dtype=bool)
        for t in targets:
            ok &= _slack(t, m[t.metric]) >= -1e-12
        syn_met |= bool((ok & canon).any())
        if not g_off:
            continue
        if cell := calibrate._last_true(ok & canon & guard_cells(tau, highs, gaps)):
            rule = "strictest+margin" if margin_used else "strictest"
            return setting(tau, highs, gaps, *cell, m, rule, True, exact)
    # infeasible: the maximin setting over a thinned τ_keep axis (it keeps τ_keep 0), among the
    # settings that keep the guard when any does
    keep_guard = guard_off(0.0)
    best: tuple[float, tuple] | None = None
    for tau in _thin(np.array(taus), FALLBACK_TAUS).tolist():
        if keep_guard and not guard_off(tau):
            continue
        highs, gaps, m, ex = _grid(v, comps, den, tau, targets)
        exact &= ex
        s = np.minimum.reduce([_slack(t, m[t.metric]) for t in real])
        s = np.where(np.isinf(s), 1e9, s)
        s = np.where(calibrate._canonical(0, len(highs), len(gaps)), s, -np.inf)
        if keep_guard:
            s = np.where(guard_cells(tau, highs, gaps), s, -np.inf)
        h, j = np.unravel_index(int(np.argmax(s)), s.shape)
        key = (float(s[h, j]), tau)  # larger slack, then stricter (higher) τ_keep
        if best is None or key > best[0]:
            best = (key, (tau, highs, gaps, int(h), int(j), m))
    assert best is not None
    tau, highs, gaps, h, j, m = best[1]
    rule = "fallback_maximin" if keep_guard else "fallback_maximin_guard_unmet"
    return setting(tau, highs, gaps, h, j, m, rule, False, False, syn_met and guard is not None)


# ---- guards -------------------------------------------------------------------------------------


def lme_recall_any5(data: _Data, probs: list[np.ndarray] | None) -> float | None:
    """Share of answerable non-SYN queries with a rel >= 1 item in the top 5 (by p, ties in pool
    order; by the raw logit when `probs` is None): the P2 recall_any5 row on tune pools."""
    hits = n = 0
    for r in range(len(data.pools)):
        if data.seed[r] >= 0 or not data.ans[r]:
            continue
        score = data.x[r][:, 0] if probs is None else probs[r]
        order = sorted(range(len(score)), key=lambda i: (-score[i], i))[:5]
        n += 1
        hits += bool(data.y[r][order].any()) if len(order) else False
    return hits / n if n else None


def _guard_ok(data: _Data, model: ReadModel) -> tuple[bool, dict[str, float | None]]:
    base, new = lme_recall_any5(data, None), lme_recall_any5(data, _probs(data, model))
    return (base is None or (new is not None and new >= base - 1e-12)), {
        "by_z": base,
        "by_model": new,
    }


def p2_reference(data: _Data, train: np.ndarray) -> Gate:
    """The P2 floors gate (`calibrate.fit_gate`) fitted on the `train` pools only (LME + the
    training seeds; never a held-out seed), memoised per training set."""
    key = ("p2_reference", train.tobytes())
    if key not in data.memo:
        assert data.limit is not None
        data.memo[key] = calibrate.fit_gate(
            data.pools_of(train), fit_id="p2-reference", fingerprint_hash="0" * 64,
            limit=data.limit, fitted_at="1970-01-01T00:00:00Z",
        )  # fmt: skip
    return data.memo[key]


def _p2_model(p2: Gate) -> ReadModel:
    return ReadModel.build(["rerank"], [p2.platt_a], p2.platt_b)


def lme_guard(data: _Data, train: np.ndarray, model: ReadModel) -> _LmeGuard | None:
    """The LME no-regression guard for `model` on `train`: the LME tune pools in `train` under the
    model, against the P2 floors gate fitted on the same `train` pools. None without a `limit` or
    without answerable LME pools."""
    lme = train & (data.seed < 0)
    if data.limit is None or not (lme & data.ans).any():
        return None
    p2 = p2_reference(data, train)
    pools = data.pools_of(lme)
    ref = calibrate._Table(pools, 0.0, 0.0, data.limit, _p2_model(p2))  # as `before_p2_floors`
    m = ref.metrics(p2.tau_keep, p2.tau_high, p2.gap)
    return _LmeGuard(
        calibrate._Table(pools, 0.0, 0.0, data.limit, model),
        round(float(m["coverage_any"]), 6),
        round(float(m["far"]), 6),
    )


# ---- nested CV ----------------------------------------------------------------------------------


def _simpler(cfg: dict[str, Any]) -> tuple[Any, ...]:
    """Tie-break: fewer features, then the stronger regularisation (smaller C)."""
    return (-len(cfg["features"]), -cfg["C"])


def _fit_eval(
    data: _Data, train: np.ndarray, test: np.ndarray, cfg: dict[str, Any], targets: Sequence[Target]
) -> dict[str, Any]:
    """Model on `train` pools (LME + the train seeds), thresholds on the train SYN pools under the
    LME no-regression guard on `train` (`lme_guard`), scored once on the `test` pools."""
    key = (train.tobytes(), test.tobytes(), tuple(cfg["features"]), cfg["C"],
           tuple((t.metric, t.op, t.threshold) for t in targets))  # fmt: skip
    if key not in data.memo:
        model = fit_model(data, train, cfg["features"], cfg["C"])
        probs = _probs(data, model)
        st = search(_view(data, probs, train), targets, guard=lme_guard(data, train, model))
        ev = evaluate(_view(data, probs, test), st.tau_keep, st.tau_high, st.gap, targets)
        data.memo[key] = {"model": model, "setting": st, "eval": ev}
    return data.memo[key]


def _config_slack(worst: dict[str, Any], targets: Sequence[Target]) -> float:
    vals = [
        float(np.min(_slack(t, np.array([np.nan if worst[t.metric] is None else worst[t.metric]]))))
        for t in targets
    ]
    return min(vals)


def select_config(
    data: _Data, seeds: Sequence[int], targets: Sequence[Target], grid: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Leave-one-seed-out CV over `seeds` ONLY (LME pools are always training data): per fold and
    config, fit on LME + the other seeds, thresholds on the other seeds, score the held-out seed.
    Chosen: the best worst-fold relative slack among configs whose every fold keeps the LME
    no-regression guard and that pass the recall guard; none: among those keeping the
    no-regression guard, then those passing the recall guard, then all. Ties by `_simpler`."""
    if len(seeds) < 2:
        raise ValueError("selection needs at least two seeds (one inner fold)")
    in_scope = np.isin(data.seed, list(seeds)) | (data.seed < 0)
    results = []
    for cfg in grid:
        folds = []
        for sd in seeds:
            train = in_scope & (data.seed != sd)
            test = data.seed == sd
            r = _fit_eval(data, train, test, cfg, targets)
            folds.append({"seed": sd, "setting": r["setting"], "worst": r["eval"]["worst"]})
        slack = min(_config_slack(f["worst"], targets) for f in folds)
        cov_ok = all(f["setting"].guard_ok for f in folds)
        ok, rec = _guard_ok(data, fit_model(data, in_scope, cfg["features"], cfg["C"]))
        results.append(
            {
                "config": cfg,
                "worst_slack": slack,
                "guard_ok": ok,
                "lme_recall_any5": rec,
                "lme_no_regression_ok": cov_ok,
                "folds": [{"seed": f["seed"], "worst": f["worst"]} for f in folds],
            }
        )
    pool = (
        [r for r in results if r["lme_no_regression_ok"] and r["guard_ok"]]
        or [r for r in results if r["lme_no_regression_ok"]]
        or [r for r in results if r["guard_ok"]]
        or results
    )
    best = max(pool, key=lambda r: (r["worst_slack"], _simpler(r["config"])))
    return best, results


def nested_cv(
    data: _Data, targets: Sequence[Target], grid: Sequence[dict[str, Any]] = GRID
) -> dict[str, Any]:
    """The outer loop: hold out one dev seed; everything else (config choice, model, thresholds)
    sees the other seeds only (the LME no-regression guard's P2 reference included); the outer seed
    is scored once. `worst_outer` is the gate-style aggregation of the outer folds, `met` says which
    targets it meets, and (with the guard on) `lme_no_regression`: every outer fold's setting kept
    the guard on its training pools."""
    seeds = data.syn_seeds
    if len(seeds) < 3:
        raise ValueError("nested CV needs records of at least three seeds (two inner folds)")
    outer = []
    for sd in seeds:
        rest = [s for s in seeds if s != sd]
        chosen, _ = select_config(data, rest, targets, grid)
        train = np.isin(data.seed, rest) | (data.seed < 0)
        r = _fit_eval(data, train, data.seed == sd, chosen["config"], targets)
        st: Setting = r["setting"]
        outer.append(
            {
                "seed": sd,
                "selected_on": rest,
                "chosen": chosen["config"],
                "inner_worst_slack": chosen["worst_slack"],
                "train_feasible": st.feasible,
                "blocked_by_guard": st.blocked_by_guard,
                "guard_lme_no_regression": st.guard,
                "thresholds": [st.tau_keep, st.tau_high, st.gap],
                "model": r["model"].to_dict(),
                "worst": r["eval"]["worst"],
            }
        )
    worst = {}
    for t in targets:
        vals = [f["worst"][t.metric] for f in outer if f["worst"][t.metric] is not None]
        worst[t.metric] = (min(vals) if t.op == ">=" else max(vals)) if vals else None
    met = _met(targets, worst)
    if any(f["guard_lme_no_regression"] is not None for f in outer):
        met[GUARD] = all(
            f["guard_lme_no_regression"] is None or f["guard_lme_no_regression"]["ok"]
            for f in outer
        )
    return {
        "outer_folds": outer,
        "worst_outer": worst,
        "met": met,
        "feasible": all(met.values()),
        "targets": {t.gate_metric: {"op": t.op, "threshold": t.threshold} for t in targets},
    }


# ---- the fit ------------------------------------------------------------------------------------


def grid_for(data: _Data) -> list[dict[str, Any]]:
    """The full grid for these records: the base sets; the masked and margin sets when every record
    carries mcos/mz; the phrase sets when every record carries pcos/pov; both-sets when both."""
    grid = [*GRID]
    if data.has_masked:
        grid += [*MASKED_GRID, *MARGIN_GRID]
    if data.has_phrase:
        grid += PHRASE_GRID
    if data.has_masked and data.has_phrase:
        grid += BOTH_GRID
    return grid


def descriptive_rows(data: _Data, model: ReadModel, st: Setting, limit: int) -> dict[str, Any]:
    """The LME (non-SYN) delivered metrics under a setting: descriptive, never a target."""
    pools = data.pools_of(data.seed < 0)
    if not pools:
        return {}
    table = calibrate._Table(pools, 0.0, 0.0, limit, model)
    m = table.metrics(st.tau_keep, st.tau_high, st.gap)
    return {k: _num(x) for k, x in m.items()}


def fit_p3_gate(
    pools: Sequence[QueryPool],
    *,
    fit_id: str,
    fingerprint_hash: str,
    limit: int,
    fitted_at: str,
    targets: Sequence[Target] | None = None,
    grid: Sequence[dict[str, Any]] | None = None,
    require_masked: bool = False,
) -> tuple[Gate, dict[str, Any]]:
    """The P3 gate and its report: nested CV (per feature set, to show what the cosine adds, and
    the full grid), the final config chosen on all dev seeds (flat CV: optimistic), the model and
    thresholds fitted on all of them, the LME descriptive rows before (P2 floors rule on the same
    records) and after."""
    targets = p3_targets() if targets is None else tuple(targets)
    data = _Data(pools, limit)
    lacking = sum(1 for q in pools for i in q.items if i.mcos is None or i.mz is None)
    lacking_phrase = sum(1 for q in pools for i in q.items if i.pcos is None or i.pov is None)
    if (lacking or lacking_phrase) and require_masked:
        what = [f"{lacking} pool items lack mcos/mz (subject-masked signals)"] * bool(lacking) + [
            f"{lacking_phrase} pool items lack pcos/pov (relation-phrase signals)"
        ] * bool(lacking_phrase)
        raise ValueError(
            "; ".join(what) + ": re-score the records with a build that has MEM2-165 and"
            " MEM2-171, or drop --require-masked to fit the sets the records allow"
        )
    if grid is None:  # the masked / phrase sets join the grid when the records carry the signals
        grid = grid_for(data)
    seeds = data.syn_seeds
    allmask = np.ones(len(pools), dtype=bool)
    before = calibrate.fit_gate(
        pools, fit_id=fit_id, fingerprint_hash=fingerprint_hash, limit=limit, fitted_at=fitted_at
    )
    data.memo[("p2_reference", allmask.tobytes())] = before  # the guard's all-records reference
    nested = nested_cv(data, targets, grid)
    ablation = {
        "+".join(fs): nested_cv(data, targets, [g for g in grid if tuple(g["features"]) == fs])
        for fs in ALL_SETS
        if any(tuple(g["features"]) == fs for g in grid)
    }
    chosen, flat = select_config(data, seeds, targets, grid)
    model = fit_model(data, allmask, chosen["config"]["features"], chosen["config"]["C"])
    probs = _probs(data, model)
    st = search(_view(data, probs, allmask), targets, guard=lme_guard(data, allmask, model))
    ev = evaluate(_view(data, probs, allmask), st.tau_keep, st.tau_high, st.gap, targets)
    # Platt on the raw logit is kept in the gate for the record (REQUIRED fields, P2 tools); with
    # a read_model present `Gate.p(z)` refuses, so nothing reads it by accident.
    a, b = calibrate.fit_platt(
        [i.z for q in pools for i in q.items], [i.rel >= 1 for q in pools for i in q.items]
    )
    bm = _p2_model(before)
    bst = Setting(
        before.tau_keep,
        before.tau_high,
        before.gap,
        bool(before.tune.get("feasible")),
        "p2_floors",
        {},
        True,
    )
    report = {
        "objective": "p3",
        "masked_missing_items": lacking,
        "masked_grid": data.has_masked,
        "phrase_missing_items": lacking_phrase,
        "phrase_grid": data.has_phrase,
        "targets": nested["targets"],
        "seeds": seeds,
        "nested": nested,
        "ablation": {k: _ablation_row(v) for k, v in ablation.items()},
        "flat_cv_optimistic": {
            "chosen": chosen["config"],
            "worst_slack": chosen["worst_slack"],
            "all": [
                {
                    "config": r["config"],
                    "worst_slack": r["worst_slack"],
                    "guard_ok": r["guard_ok"],
                    "lme_no_regression_ok": r["lme_no_regression_ok"],
                }
                for r in flat
            ],
        },
        "guard_lme_recall_any5": _guard_ok(data, model)[1],
        "guard_lme_no_regression": st.guard,
        "final": {
            "config": chosen["config"],
            "feasible": st.feasible,
            "rule": st.rule,
            "blocked_by_guard": st.blocked_by_guard,
            "exact": st.exact,
            "dev_in_sample": ev,
        },
        "lme_descriptive": {
            "before_p2_floors": descriptive_rows(data, bm, bst, limit),
            "after_p3": descriptive_rows(data, model, st, limit),
        },
        "before_p2_floors_on_syn": evaluate(
            _view(data, _probs(data, bm), allmask),
            before.tau_keep,
            before.tau_high,
            before.gap,
            targets,
        ),
    }
    tune = {
        "objective": "p3",
        "feasible": st.feasible,
        "rule": st.rule,
        "queries": len(pools),
        "targets": {
            t.gate_metric: {
                "op": t.op,
                "threshold": t.threshold,
                "value": ev["worst"][t.metric],
                "met": _met(targets, ev["worst"])[t.gate_metric],
            }
            for t in targets
        },
        "nested_feasible": nested["feasible"],
        "nested_worst": nested["worst_outer"],
        "dev_in_sample": ev["worst"],
        "guard_lme_no_regression": st.guard,
        "blocked_by_guard": st.blocked_by_guard,
    }
    if not st.guard_ok:
        raise GuardUnmet(report)
    gate = Gate(
        a,
        b,
        st.tau_keep,
        st.tau_high,
        st.gap,
        read_model=model.to_dict(),
        fit_id=fit_id,
        fitted_at=fitted_at,
        fingerprint_hash=fingerprint_hash,
        tune=tune,
    )
    return gate, report


class GuardUnmet(ValueError):
    """No setting of the final model keeps the LME no-regression guard: no gate is written (a
    guard-breaking fallback never is). `report` is the fit report, for the record."""

    def __init__(self, report: dict[str, Any]) -> None:
        g = report["guard_lme_no_regression"]
        super().__init__(
            f"{GUARD} guard unmet: no setting of the final model keeps LME coverage_any >="
            f" {g['p2_coverage_any']} and empty-answerable share <= {g['p2_empty_answerable']} (the"
            f" P2 floors gate on the same records; the maximin setting has coverage_any"
            f" {g['coverage_any']}, empty-answerable {g['empty_answerable']})"
        )
        self.report = report


def _ablation_row(n: dict[str, Any]) -> dict[str, Any]:
    return {
        "worst_outer": n["worst_outer"],
        "met": n["met"],
        "feasible": n["feasible"],
        "chosen": [f["chosen"] for f in n["outer_folds"]],
    }


_SHORT = {"missing_slot_abstention": "missAbst", "current_recall_at_1": "cur@1",
          "as_of_accuracy": "asOf", "far": "FAR"}  # fmt: skip


def format_report(r: dict[str, Any]) -> str:
    """The text the runbook asks to be pasted back."""
    tg = r["targets"]
    names = list(tg)

    def row(label: str, worst: dict[str, Any]) -> str:
        cells = [("--" if worst[n] is None else f"{worst[n]:.3f}").rjust(8) for n in names]
        return f"  {label:<26}" + "".join(cells)

    out = [
        f"P3 read gate fit (B-R2 (ii), B-R5, B-R2 (i) FAR on SYN dev; seeds {r['seeds']})",
        *(
            [
                (
                    f"  WARNING: {r['masked_missing_items']} pool items lack mcos/mz: the masked"
                    " feature sets were NOT in the grid"
                )
            ]
            if r.get("masked_missing_items")
            else []
        ),
        *(
            [
                (
                    f"  WARNING: {r['phrase_missing_items']} pool items lack pcos/pov: the"
                    " relation-phrase feature sets were NOT in the grid"
                )
            ]
            if r.get("phrase_missing_items")
            else []
        ),
        "  targets: " + ", ".join(f"{n} {tg[n]['op']} {tg[n]['threshold']}" for n in names),
        f"  {'':<26}" + "".join(_SHORT.get(n, n[:8]).rjust(8) for n in names),
        "nested leave-one-seed-out CV, worst outer fold (outer seed unseen by every choice):",
    ]
    ab = r["ablation"]
    for k, v in ab.items():
        out.append(
            row(k, v["worst_outer"])
            + (
                "  MET"
                if v["feasible"]
                else "  miss: " + ",".join(n for n, ok in v["met"].items() if not ok)
            )
        )
    n = r["nested"]
    out.append(
        row("full grid", n["worst_outer"])
        + (
            "  MET"
            if n["feasible"]
            else "  miss: " + ",".join(k for k, ok in n["met"].items() if not ok)
        )
    )
    out.append("per outer fold (seed: config, thresholds, worst-seed numbers):")
    for f in n["outer_folds"]:
        out.append(
            row(
                f"seed {f['seed']} {'+'.join(f['chosen']['features'])} C={f['chosen']['C']:g}",
                f["worst"],
            )
            + ("" if f["train_feasible"] else "  (train infeasible)")
            + ("  (guard blocks)" if f.get("blocked_by_guard") else "")
        )
    fin = r["final"]
    why = fin["rule"] + (f"; the {GUARD} guard blocks every setting meeting the SYN targets"
                         if fin.get("blocked_by_guard") else "")  # fmt: skip
    out.append(
        f"final: {'+'.join(fin['config']['features'])} C={fin['config']['C']:g}, "
        f"{'feasible' if fin['feasible'] else 'INFEASIBLE (' + why + ')'}; "
        "chosen on all dev seeds (flat CV, optimistic by construction)"
    )
    out.append(row("dev in-sample", fin["dev_in_sample"]["worst"]))
    out.append(row("before (P2 floors)", r["before_p2_floors_on_syn"]["worst"]))
    d = r["lme_descriptive"]
    out.append("LME descriptive rows (not targets), before -> after:")
    for k in ("precision", "ira", "coverage_any", "abstention_recall", "far"):
        out.append(f"  {k:<18} {d['before_p2_floors'].get(k)} -> {d['after_p3'].get(k)}")
    g = r["guard_lme_recall_any5"]
    out.append(f"guard: LME recall_any5 by logit {g['by_z']} vs by model {g['by_model']}")
    if (c := r.get("guard_lme_no_regression")) is not None:
        out.append(
            f"guard {GUARD} ({'ok' if c['ok'] else 'UNMET'}): LME coverage_any P2"
            f" {c['p2_coverage_any']} -> chosen {c['coverage_any']} (>=), empty-answerable share"
            f" P2 {c['p2_empty_answerable']} -> chosen {c['empty_answerable']} (<=)"
        )
    return "\n".join(out)


# ---- diagnosis ----------------------------------------------------------------------------------


def diagnose(
    pools: Sequence[QueryPool], gate: Gate | None = None, examples: int = 5
) -> dict[str, Any]:
    """Why current_recall_at_1 / as_of_accuracy miss, from records: for each SYN current / as-of
    query, is the gold memory in the pool (eligibility / retrieval), ranked first (ranking), and
    delivered (the gate)? With examples. Also the AUROC of the top logit, cosine and (given a
    read-model gate) p for answerable vs missing-slot queries. Ranking is by the gate's p when it
    has a read model, else by the raw logit."""
    from sklearn.metrics import roc_auc_score

    rm = gate.reader if gate else None
    out: dict[str, Any] = {"by_class": {}}
    top: dict[str, list[tuple[float, bool]]] = {"z": [], "cos": [], "p": []}
    for cat in (CAT_CURRENT, CAT_ASOF):
        c = {
            "n": 0,
            "gold_not_in_pool": 0,
            "gold_not_first": 0,
            "gate_drops_gold": 0,
            "delivered_gold": 0,
            "examples": {"gold_not_in_pool": [], "gold_not_first": [], "gate_drops_gold": []},
        }
        for q in pools:
            if q.category != cat or not q.items:
                continue
            c["n"] += 1
            zs = [i.z for i in q.items]
            ps = rm.probs(zs, [i.cos for i in q.items], _masked_of(q.items)) if rm else zs
            order = sorted(range(len(ps)), key=lambda i: (-ps[i], i))
            gold = [r for r, i in enumerate(order) if q.items[i].rel >= 1]
            if not gold:
                c["gold_not_in_pool"] += 1
                _ex(c, "gold_not_in_pool", q.qid, examples, top=q.items[order[0]].key)
                continue
            if gold[0] != 0:
                c["gold_not_first"] += 1
                _ex(
                    c,
                    "gold_not_first",
                    q.qid,
                    examples,
                    gold_rank=gold[0] + 1,
                    top=q.items[order[0]].key,
                    z_top=round(zs[order[0]], 3),
                    z_gold=round(zs[order[gold[0]]], 3),
                )
                continue
            if gate is not None:
                pg = (
                    rm.probs(zs, [i.cos for i in q.items], _masked_of(q.items))
                    if rm
                    else [gate.p(z) for z in zs]
                )
                po = sorted(pg, reverse=True)
                if gate.abstains(po[0], po[1] if len(po) > 1 else 0.0):
                    c["gate_drops_gold"] += 1
                    _ex(
                        c,
                        "gate_drops_gold",
                        q.qid,
                        examples,
                        p1=round(po[0], 3),
                        p2=round(po[1] if len(po) > 1 else 0.0, 3),
                    )
                    continue
            c["delivered_gold"] += 1
        out["by_class"][cat] = c
    for q in pools:
        if q.category not in (CAT_CURRENT, CAT_ASOF, CAT_MISSING) or not q.items:
            continue
        i = max(range(len(q.items)), key=lambda i: (q.items[i].z, -i))
        y = q.category != CAT_MISSING
        top["z"].append((q.items[i].z, y))
        if q.items[i].cos is not None:
            top["cos"].append((q.items[i].cos, y))
        for key, name in (("cos_masked", "mcos"), ("rerank_masked", "mz"),
                          ("phrase_cosine", "pcos"), ("phrase_overlap", "pov")):  # fmt: skip
            vals = [getattr(x, name) for x in q.items]
            if all(v is not None for v in vals):  # best signal in the pool (MEM2-165, MEM2-171)
                top.setdefault(key, []).append((max(vals), y))
                if name in ("mcos", "mz"):  # and its margin: best minus second best
                    srt = sorted(vals, reverse=True)
                    margin = srt[0] - srt[1] if len(srt) > 1 else 0.0
                    top.setdefault(key + "_margin", []).append((margin, y))
        if rm:
            ps = rm.probs([x.z for x in q.items], [x.cos for x in q.items], _masked_of(q.items))
            top["p"].append((max(ps), y))
    out["auroc_answerable_vs_missing"] = {
        k: (
            float(roc_auc_score([y for _, y in v], [s for s, _ in v]))
            if len({y for _, y in v}) == 2
            else None
        )
        for k, v in top.items()
        if v
    }
    return out


def _masked_of(items: Sequence[PoolItem]) -> dict[str, list[float | None]]:
    return {k: [i.masked()[k] for i in items] for k in SIGNALS}


def _ex(c: dict[str, Any], key: str, qid: str, cap: int, **kw: Any) -> None:
    if len(c["examples"][key]) < cap:
        c["examples"][key].append({"qid": qid, **kw})


def format_diagnosis(d: dict[str, Any]) -> str:
    out = []
    for cat, c in d["by_class"].items():
        n = max(c["n"], 1)
        out.append(
            f"{cat}: n={c['n']}  gold not in pool {c['gold_not_in_pool']} "
            f"({c['gold_not_in_pool'] / n:.3f})  ranked below first {c['gold_not_first']} "
            f"({c['gold_not_first'] / n:.3f})  first but gate abstains {c['gate_drops_gold']} "
            f"({c['gate_drops_gold'] / n:.3f})  delivered first {c['delivered_gold']} "
            f"({c['delivered_gold'] / n:.3f})"
        )
        for k, exs in c["examples"].items():
            out += [f"    {k}: {e}" for e in exs]
    out.append(f"AUROC answerable vs missing-slot (top item): {d['auroc_answerable_vs_missing']}")
    return "\n".join(out)


__all__ = ["GRID", "P3_ROWS", "PoolItem", "diagnose", "fit_p3_gate", "format_report", "p3_targets"]
