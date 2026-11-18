"""Cohen's kappa for the blind check of Claude's PG-v1 judgments (benchmarks-v2 §3.1).

**Definition.** Two raters grade the same n (question, chunk) pairs 0 / 1 / 2. With the observed
joint proportions o_ij and the chance proportions e_ij = r_i * c_j (row and column marginals), and
linear disagreement weights w_ij = |i - j| / (k - 1) over the k = 3 grades:

    kappa_w = 1 - (sum w_ij o_ij) / (sum w_ij e_ij)

Perfect agreement gives 1, agreement no better than chance gives 0. If the chance disagreement is 0
(both raters used one single grade) kappa is undefined: `nan`, and `nan` never passes the gate.
The sample is a seeded random draw of Claude's judgments; the human sees the question, the short answer
and the chunk, never Claude's grade.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

GRADES = (0, 1, 2)
KAPPA_MIN = 0.7
SAMPLE_N = 50
SAMPLE_SEED = 20261024


def confusion(pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Rows = rater A (Claude), columns = rater B (the human)."""
    m = [[0] * len(GRADES) for _ in GRADES]
    for a, b in pairs:
        m[GRADES.index(a)][GRADES.index(b)] += 1
    return m


def weighted_kappa(pairs: Sequence[tuple[int, int]]) -> float:
    """Linear-weighted Cohen's kappa of (A, B) grade pairs; `nan` when undefined or n = 0."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    m = confusion(pairs)
    k = len(GRADES)
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]
    w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    observed = sum(w[i][j] * m[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w[i][j] * rows[i] * cols[j] for i in range(k) for j in range(k)) / n**2
    return float("nan") if expected == 0 else 1.0 - observed / expected


def draw_sample(keys: Sequence[tuple[str, str]], n: int = SAMPLE_N, seed: int = SAMPLE_SEED
                ) -> list[tuple[str, str]]:  # fmt: skip
    """`min(n, len(keys))` keys, a function of (sorted keys, seed) only."""
    ordered = sorted(set(keys))
    return random.Random(seed).sample(ordered, min(n, len(ordered)))


@dataclass(frozen=True)
class KappaResult:
    kappa: float
    n: int
    agree: float  # share of exactly equal grades
    table: list[list[int]]

    @property
    def stands(self) -> bool:
        return self.kappa == self.kappa and self.kappa >= KAPPA_MIN

    def text(self) -> str:
        head = (f"Cohen's kappa (linear weights, grades 0/1/2) = "
                f"{'undefined' if self.kappa != self.kappa else f'{self.kappa:.3f}'} "
                f"on n = {self.n} blind pairs; exact agreement {self.agree:.0%}")  # fmt: skip
        rows = "\n".join(f"    claude {i}: " + "  ".join(f"{v:>3}" for v in r)
                         for i, r in enumerate(self.table))  # fmt: skip
        return f"{head}\n    (columns: your 0 1 2)\n{rows}"


def result_of(pairs: Sequence[tuple[int, int]]) -> KappaResult:
    return KappaResult(weighted_kappa(pairs), len(pairs),
                       sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0,
                       confusion(pairs))  # fmt: skip
