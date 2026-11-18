"""Statistics (benchmarks-v2 §5): stratified percentile bootstrap, paired bootstrap, power, Holm.

Every function takes an explicit seed; B defaults to 10,000. Micro averages are the headline.
Ratio metrics (delivered precision, IRA) bootstrap the ratio of sums over queries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

B_DEFAULT = 10_000
SEED = 20261023
Z_ALPHA = 1.959964
Z_POWER80 = 0.841621


def _indices(strata: Sequence[str], b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n) resample indices, drawing within each stratum, keeping stratum sizes."""
    labels = np.asarray(strata)
    n = len(labels)
    idx = np.empty((b, n), dtype=np.int64)
    for s in sorted(set(strata)):
        pos = np.flatnonzero(labels == s)
        idx[:, pos] = pos[rng.integers(0, len(pos), size=(b, len(pos)))]
    return idx


def bootstrap_ratio(
    num: Sequence[float], den: Sequence[float], strata: Sequence[str],
    b: int = B_DEFAULT, seed: int = SEED,
) -> tuple[float, float, float]:  # fmt: skip
    """(point, lo, hi): 95% percentile CI of sum(num)/sum(den), resampling queries by stratum."""
    num_a, den_a = np.asarray(num, float), np.asarray(den, float)
    if den_a.sum() == 0:
        return (math.nan, math.nan, math.nan)
    idx = _indices(strata, b, np.random.default_rng(seed))
    d = den_a[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.where(d > 0, num_a[idx].sum(axis=1) / d, np.nan)
    lo, hi = np.nanpercentile(stat, [2.5, 97.5])
    return (float(num_a.sum() / den_a.sum()), float(lo), float(hi))


def bootstrap_mean(
    values: Sequence[float], strata: Sequence[str], b: int = B_DEFAULT, seed: int = SEED
) -> tuple[float, float, float]:
    return bootstrap_ratio(values, [1.0] * len(values), strata, b, seed)


@dataclass(frozen=True)
class Paired:
    delta: float
    lo: float
    hi: float
    sd: float
    n: int

    @property
    def better(self) -> bool:
        return self.lo > 0


def paired_bootstrap(
    a: Sequence[float], b_: Sequence[float], strata: Sequence[str],
    b: int = B_DEFAULT, seed: int = SEED,
) -> Paired:  # fmt: skip
    """Delta = mean(a - b_) over the same items, with its 95% percentile CI and SD(delta)."""
    d = np.asarray(a, float) - np.asarray(b_, float)
    point, lo, hi = bootstrap_mean(d, strata, b, seed)
    return Paired(point, lo, hi, float(d.std(ddof=1)) if len(d) > 1 else math.nan, len(d))


def mdd_continuous(sd: float, n: int) -> float:
    """80% power, alpha 0.05: 2.80 SD(delta) / sqrt(n)."""
    return (Z_ALPHA + Z_POWER80) * sd / math.sqrt(n)


def mdd_binary(p_d: float, n: int) -> float:
    return (Z_ALPHA + Z_POWER80) * math.sqrt(p_d / n)


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def superiority_power(delta: float, sd: float, n: int) -> float:
    """Power to find `delta` > 0 at n items: Phi(delta sqrt(n) / sd - 1.96)."""
    if math.isnan(sd) or math.isnan(delta):
        return math.nan
    if sd <= 0:  # identical on every item (or a constant gap): nothing to estimate noise from
        return 1.0 if delta > 0 else 0.0
    return _phi(delta * math.sqrt(n) / sd - Z_ALPHA)


def noninferiority_power(m: float, p_d: float, n: int) -> float:
    return _phi(m / math.sqrt(p_d / n) - Z_ALPHA)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: which hypotheses are rejected."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [False] * len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha / (len(pvalues) - rank):
            break
        out[i] = True
    return out


def percentile_nearest_rank(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)), 1) - 1]


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), by rank sums. nan if either class is empty."""
    if not pos or not neg:
        return math.nan
    allv = np.concatenate([np.asarray(pos, float), np.asarray(neg, float)])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
