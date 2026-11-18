"""`mem calibrate` (architecture-v2 §3.3): the one fitting routine serving and `bench` share.

Two halves. `score_pool` runs read-path steps 1-5 with the gate off and returns the raw logits of
the pool; the harness calls it in-process on the tune queries and saves the pools as records
(JSONL). `fit_gate` turns records into a `Gate`: Platt on `rel ≥ 1` (step 3), then `τ_keep`,
`τ_high` and `g` together (steps 4-5). Fitting loads no model (D5). `θ_conf` is fitted from P3 (SYN).

Steps 4-5 (the person's "floors" rule, 2026-10-24, MEM2-87): the three thresholds are chosen
**together** as the **strictest setting that meets all five P2 tune targets at once** (delivered
precision, IRA, coverage_any, abstention recall and FAR, read from bench/configs/gates.yaml so a
`bench freeze P2` is picked up). Strictest = highest τ_keep, ties by highest τ_high, then highest g
(`STRICTEST`). If no setting meets them all, the fit is `feasible: false`, reports the Pareto front
on tune (the non-dominated settings with all five numbers) and writes a documented fallback,
flagged infeasible, so the person sees the numbers: no target is relaxed (`choose_thresholds`).

`tune_report` (MEM2-83, F-P2b) says what the fit saw and what bound it: the Platt inputs (raw-logit
range, rel/irrel medians, AUROC of z), the §2.0 delivered metrics per query category, the τ_keep
frontier with the gap rule off, the choice with its five numbers and, when infeasible, the front.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mem import read, readmodel, subjectmask
from mem.common import MemError, normalize_as_of
from mem.gate import Gate
from mem.readmodel import ReadModel

if TYPE_CHECKING:
    from mem.store import Store

STRICTEST = "highest tau_keep, then highest tau_high, then highest gap"


@dataclass(frozen=True)
class Target:
    """One P2 tune target steps 4-5 must meet: `metric` (a `_Table.metrics` key) `op` threshold."""

    gate_metric: str  # the metric's name in gates.yaml
    metric: str
    op: str  # ">=" or "<="
    threshold: float

    def met(self, value: float) -> bool:
        if self.op == ">=":
            return value >= self.threshold - 1e-12
        return value <= self.threshold + 1e-12

    def better(self) -> Any:
        """The reducer that picks the best value of this metric from many (NaN ignored)."""
        return np.nanmax if self.op == ">=" else np.nanmin


# the five targets, in reporting order: gates.yaml name -> (gate id, `_Table` key)
TARGET_ROWS = {
    "delivered_precision": ("B-R1", "precision"),
    "ira": ("B-R1", "ira"),
    "coverage_any": ("B-R1", "coverage_any"),
    "abstention_recall": ("B-R2", "abstention_recall"),
    "far": ("B-R1", "far"),
}
# the same five on PG-v1 (B-R3, P4 rows): the floors rule for a fit on PG tune
B_R3_ROWS = {name: ("B-R3", key) for name, (_, key) in TARGET_ROWS.items()}


def _targets(
    phase: str, rows: dict[str, tuple[str, str]], gates_path: Path | None
) -> tuple[Target, ...]:
    from bench.gates import GATES_YAML, load_gates

    specs = {(s.id, s.metric): s for s in load_gates(gates_path or GATES_YAML)[phase]}
    missing = [f"{gid} {m}" for m, (gid, _) in rows.items() if (gid, m) not in specs]
    if missing:
        raise ValueError(f"gates.yaml has no {phase} row for {', '.join(missing)}")
    far2 = specs.get(("B-R2", "far"))
    if (
        phase == "P2"
        and far2 is not None
        and far2.threshold != specs[(rows["far"][0], "far")].threshold
    ):
        raise ValueError(f"gates.yaml: {phase} B-R1 far and B-R2 far disagree")
    out = []
    for name, (gid, key) in rows.items():
        spec = specs[(gid, name)]
        if spec.op not in (">=", "<=") or spec.threshold is None:
            raise ValueError(
                f"{phase} {gid} {name}: steps 4-5 need a >= or <= threshold, not {spec.op}"
            )
        out.append(Target(name, key, spec.op, float(spec.threshold)))
    return tuple(out)


def p2_targets(gates_path: Path | None = None) -> tuple[Target, ...]:
    """The five P2 tune targets as gates.yaml holds them now (provisional, or as frozen by
    `bench freeze P2` on tune): B-R1 delivered precision, IRA, coverage_any and FAR, B-R2 abstention
    recall. B-R2's FAR row must agree with B-R1's. Descriptive rows keep their thresholds: the LME
    fit still reports the floors front against them."""
    return _targets("P2", TARGET_ROWS, gates_path)


def b_r3_targets(gates_path: Path | None = None) -> tuple[Target, ...]:
    """The five floors targets of a fit on PG-v1 tune: the P4 B-R3 rows (FAR 0.10, abstention
    0.70), not the P2 ones."""
    return _targets("P4", B_R3_ROWS, gates_path)


@dataclass(frozen=True)
class PoolItem:
    key: str  # chunk id as shown (`c812`) or the harness's own label
    z: float  # raw reranker logit
    rel: int  # 0, 1 or 2
    cos: float | None = (
        None  # query-chunk embedding cosine (read model, MEM2-155); None in old records
    )
    mcos: float | None = None  # the same with the subject masked (MEM2-165); None in old records
    mz: float | None = None  # raw logit on the subject-masked pair (MEM2-165)
    pcos: float | None = None  # cosine of the predicate phrases (MEM2-171, mem.relphrase)
    pov: float | None = None  # share of the question's predicate stems in the chunk's (MEM2-171)

    def masked(self) -> dict[str, float | None]:
        """The store-computed signals by feature name (`readmodel.SIGNALS`)."""
        return {"cosine_masked": self.mcos, "rerank_masked": self.mz,
                "phrase_cosine": self.pcos, "phrase_overlap": self.pov}  # fmt: skip


@dataclass(frozen=True)
class QueryPool:
    qid: str
    answerable: bool
    items: list[PoolItem]  # the rerank pool (≤ 30), any order
    category: str = ""  # LME question type, SYN query type; "" in records written before MEM2-83


# ---- scoring (needs the store's models; run by the harness) -------------------------------------


def score_pool(store: Store, query: str, as_of: str | None = None) -> list[PoolItem]:
    """Steps 1-5 with the gate off: the pool in RRF order with raw logits, rel unknown (0).
    No event is logged and nothing is returned to a client: calibration input only (A2)."""
    if store.reranker is None:
        raise MemError("score_pool needs a store opened with a reranker")
    t = normalize_as_of(as_of, store.tz) if as_of else store.now_ts()
    pool, _ = read.candidate_pool(store, query, t=t)
    zs = read.rerank(store, query, [cid for cid, _ in pool])
    ids = [cid for cid, _ in pool]
    cos = read.pool_cosines(store, query, ids)
    m = subjectmask.pool_masked(store, query, ids, readmodel.SIGNALS)
    return [PoolItem(f"c{cid}", z, 0, c, *sig)
            for (cid, _), z, c, *sig in zip(pool, zs, cos, *(m[k] for k in readmodel.SIGNALS),
                                            strict=True)]  # fmt: skip


# ---- records ------------------------------------------------------------------------------------


def write_records(path: str | Path, pools: Iterable[QueryPool], *, fp_all: str) -> None:
    """JSONL: a header line `{"fp_all": ...}`, then one line per query."""
    with Path(path).open("w", encoding="utf-8") as f:
        f.write(json.dumps({"fp_all": fp_all}) + "\n")
        f.writelines(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in pools)


def read_records(path: str | Path) -> tuple[str, list[QueryPool]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or "fp_all" not in json.loads(lines[0]):
        raise MemError(f'{path}: the first line must be the header {{"fp_all": ...}}')
    pools = []
    for line in lines[1:]:
        if line.strip():
            d = json.loads(line)
            pools.append(QueryPool(d["qid"], bool(d["answerable"]),
                                   [PoolItem(i["key"], float(i["z"]), int(i["rel"]),
                                              *(None if i.get(k) is None else float(i[k])
                                                for k in ("cos", "mcos", "mz", "pcos", "pov")))
                                    for i in d["items"]], str(d.get("category", ""))))  # fmt: skip
    return json.loads(lines[0])["fp_all"], pools


def pool_recall(pools: Sequence[QueryPool], size: int | None = None) -> dict[str, Any]:
    """Share of answerable queries whose first `size` pool items (pool order; all by default) hold
    a rel >= 1 chunk: the ceiling of coverage_any. With a Wilson 95% interval."""
    ans = [q for q in pools if q.answerable]
    hits = sum(any(i.rel >= 1 for i in q.items[:size]) for q in ans)
    n = len(ans)
    if not n:
        return {"n": 0, "hits": 0, "recall": None, "ci95": None}
    p, z = hits / n, 1.959964
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "n": n,
        "hits": hits,
        "recall": p,
        "ci95": [max(0.0, centre - half), min(1.0, centre + half)],
    }


# ---- step 3: Platt ------------------------------------------------------------------------------


def fit_platt(
    z: Sequence[float] | np.ndarray, y: Sequence[bool] | np.ndarray
) -> tuple[float, float]:
    """Logistic regression of `y` (rel ≥ 1) on the raw logit `z`: p = σ(a·z + b).

    Unpenalised, with Platt's smoothed targets (N₊+1)/(N₊+2) and 1/(N₋+2), so a separable tune
    set still gives finite coefficients. Each point enters as a positive with weight t and a
    negative with weight 1 − t."""
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        raise ValueError("Platt needs both relevant and irrelevant pool items in the tune data")
    t = np.where(y, (pos + 1) / (pos + 2), 1 / (neg + 2))
    x = np.concatenate([z, z]).reshape(-1, 1)
    labels = np.concatenate([np.ones(z.size), np.zeros(z.size)])
    weights = np.concatenate([t, 1 - t])
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)  # unpenalised
    model.fit(x, labels, sample_weight=weights)
    return float(model.coef_[0][0]), float(model.intercept_[0])


# ---- steps 4-5: thresholds ----------------------------------------------------------------------


def _masked_columns(items: Sequence[PoolItem]) -> dict[str, list[float | None]]:
    return {k: [i.masked()[k] for i in items] for k in readmodel.SIGNALS}


class _Table:
    """The tune pools after Platt, as arrays: p sorted descending per query (padded with -1) and
    the running count of relevant items, so S_q for any gate is a few vector operations."""

    def __init__(
        self, pools: Sequence[QueryPool], a: float, b: float, limit: int,
        read_model: ReadModel | None = None,
    ) -> None:  # fmt: skip
        g = Gate(a, b, 0.0, 0.0, 0.0)
        width = max((len(q.items) for q in pools), default=0)
        self.p = np.full((len(pools), width), -1.0)
        self.cumrel = np.zeros((len(pools), width + 1), dtype=int)
        for r, q in enumerate(pools):
            ps = (read_model.probs([i.z for i in q.items], [i.cos for i in q.items],
                                   _masked_columns(q.items))
                  if read_model else [g.p(i.z) for i in q.items])  # fmt: skip
            pairs = sorted(((p, i.rel >= 1) for p, i in zip(ps, q.items, strict=True)),
                           key=lambda pr: -pr[0])  # fmt: skip
            self.p[r, : len(pairs)] = [p for p, _ in pairs]
            self.cumrel[r, 1 : len(pairs) + 1] = np.cumsum([rel for _, rel in pairs])
        self.answerable = np.array([q.answerable for q in pools], dtype=bool)
        self.category = [q.category for q in pools]
        self.gold_in_pool = self.cumrel.max(axis=1) > 0 if width else np.zeros(len(pools), bool)
        self.p1 = self.p[:, 0] if width else np.zeros(len(pools))
        p2 = np.where(self.p[:, 1] >= 0, self.p[:, 1], 0.0) if width > 1 else np.zeros(len(pools))
        self.d = self.p1 - p2
        self.has = self.p1 >= 0
        self.limit = limit

    def rows_hits(
        self, tau_keep: float, tau_high: float, gap: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per query: |S_q| and |S_q ∩ G_q| under the gate (0 rows when it abstains)."""
        gap = np.asarray(gap, dtype=float)
        abstain = (~self.has) | (self.p1 < tau_keep)
        abstain = abstain | ((self.p1 < tau_high) & (self.d < gap[..., None]))
        n = np.minimum((self.p >= tau_keep).sum(axis=1), self.limit)
        rows = np.where(abstain, 0, n)
        hits = self.cumrel[np.arange(len(self.p1)), rows]  # broadcasts over a gap array
        return rows, hits

    def metrics(self, tau_keep: float, tau_high: float, gap: float | np.ndarray) -> dict[str, Any]:
        """Delivered metrics of benchmarks-v2 §2.0 on S_q = kept rows (top `limit`). `gap` may be
        an array: every metric is then an array over it."""
        rows, hits = self.rows_hits(tau_keep, tau_high, gap)
        return self._aggregate(rows, hits, self.answerable)

    def _aggregate(self, rows: np.ndarray, hits: np.ndarray, ans: np.ndarray) -> dict[str, Any]:
        neg = ~ans
        total_rows = rows.sum(axis=-1)
        prec = np.divide(hits.sum(axis=-1), total_rows, out=np.zeros(total_rows.shape),
                         where=total_rows > 0)  # fmt: skip
        n_ans, n_neg = ans.sum(), neg.sum()
        cov = ((hits > 0) & ans).sum(axis=-1) / n_ans if n_ans else np.zeros(prec.shape)
        far = ((rows == 0) & ans).sum(axis=-1) / n_ans if n_ans else np.zeros(prec.shape)
        abst = ((rows == 0) & neg).sum(axis=-1) / n_neg if n_neg else np.full(prec.shape, np.nan)
        nonempty = (rows > 0).sum(axis=-1)
        ira = np.divide((rows - hits).sum(axis=-1), nonempty, out=np.zeros(prec.shape),
                        where=nonempty > 0)  # fmt: skip
        den = 0.25 * prec + cov
        f05 = np.divide(1.25 * prec * cov, den, out=np.zeros(prec.shape), where=den > 0)
        return {"f05": f05, "precision": prec, "coverage_any": cov, "ira": ira, "far": far,
                "abstention_recall": abst}  # fmt: skip

    def per_category(self, tau_keep: float, tau_high: float, gap: float) -> dict[str, dict]:
        """The §2.0 delivered metrics restricted to each category's queries (answerable and
        negative together, as the headline), plus counts, pool recall (answerable queries whose
        pool holds a rel ≥ 1 chunk: the ceiling of coverage_any) and rows per non-empty answer.
        Undefined ratios (no queries of that class) are None."""
        rows, hits = self.rows_hits(tau_keep, tau_high, gap)
        out: dict[str, dict[str, Any]] = {}
        for cat in sorted(set(self.category)):
            sel = np.array([c == cat for c in self.category], dtype=bool)
            ans = self.answerable[sel]
            m = self._aggregate(rows[sel], hits[sel], ans)
            nonempty = int((rows[sel] > 0).sum())
            row: dict[str, Any] = {"n_answerable": int(ans.sum()), "n_negative": int((~ans).sum()),
                                   "pool_recall": _ratio(int((self.gold_in_pool[sel] & ans).sum()),
                                                         int(ans.sum()))}  # fmt: skip
            for k, v in m.items():
                row[k] = _nan_none(float(v))
            if not ans.any():
                row["coverage_any"] = row["far"] = row["f05"] = None
            if not nonempty:
                row["ira"] = None
            row["rows_per_answer"] = _ratio(int(rows[sel].sum()), nonempty)
            out[cat] = row
        return out


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 6) if den else None


def _nan_none(x: float) -> float | None:
    return None if np.isnan(x) else round(x, 6)


def _between(values: Iterable[float]) -> list[float]:
    """Cut points for a strict `<` test: midpoints between sorted distinct values, plus one
    above the largest, so every subset 'below the cut' is reachable. Values within 1e-9 count as
    one (p is a sigmoid of a float; two equal gaps may differ in the last digit)."""
    v = sorted({round(x, 9) for x in values})
    top = [max(1.0, v[-1] + 1.0) if v[-1] >= 1.0 else 1.0] if v else []  # p and gaps live in [0, 1]
    return [(x + y) / 2 for x, y in itertools.pairwise(v)] + top


@dataclass(frozen=True)
class Choice:
    """Steps 4-5's result: the setting, whether it meets all five targets, the rule that chose it
    (`strictest`, or a fallback when infeasible), per target its threshold, the best value any
    evaluated setting reaches, the value at the chosen setting, and (infeasible only) the front."""

    tau_keep: float
    tau_high: float
    gap: float
    feasible: bool
    rule: str  # strictest | fallback_coverage_far | fallback_coverage_ceiling | fallback_open
    targets: dict[str, dict[str, Any]]
    front: list[dict[str, Any]]
    exact: bool = True  # False: the feasibility scan outgrew EXACT_CELLS (see `_search`)


FRONT_KEYS = ("precision", "ira", "coverage_any", "abstention_recall", "far")
_SIGN = {"precision": 1.0, "ira": -1.0, "coverage_any": 1.0, "abstention_recall": 1.0, "far": -1.0}


def _tau_keep_candidates(table: _Table) -> list[float]:
    """Step 4's grid, strictest first: 0 and the p of every item ranked ≤ `limit` in its query.
    Exact: an item ranked below `limit` enters no S_q when τ drops to its p (rows are
    min(#{p ≥ τ}, limit)), so its p repeats the next higher candidate's metrics."""
    top = table.p[:, : table.limit]
    return sorted({0.0, *top[top >= 0].tolist()}, reverse=True)


# Work caps of `_search`. `EXACT_CELLS`: (τ_keep, τ_high, g) cells the strictest-setting scan may
# evaluate before it falls back to the thinned grid; `FRONT_TAUS` / `FRONT_CUTS`: τ_keep candidates
# and cut points per axis (and `FRONT_LEVELS` FAR levels per τ_keep) that the Pareto front and the
# `best` figures are built from;
# `BLOCK_CELLS`: cells of one grid block (the peak memory of a scan is a few of these).
EXACT_CELLS = 100_000_000
FRONT_TAUS = 300
FRONT_CUTS = 300
FRONT_LEVELS = 100  # FAR levels kept per τ_keep in the front's candidates
BLOCK_CELLS = 1 << 20


def _thin(values: np.ndarray, keep: int) -> np.ndarray:
    """`values` itself, or `keep` of them evenly spread by index, both ends kept."""
    if len(values) <= keep:
        return values
    return values[np.unique(np.linspace(0, len(values) - 1, keep).round().astype(int))]


def _highs_for(table: _Table, tau_keep: float) -> np.ndarray:
    """Step 5's τ_high grid at `tau_keep`: itself, then the cuts between the p1 values above it."""
    active = table.has & (table.p1 >= tau_keep)
    return np.array([tau_keep, *(c for c in _between(table.p1[active].tolist()) if c > tau_keep)])


def _active_gaps(table: _Table, tau_keep: float, gaps: np.ndarray) -> np.ndarray:
    """`gaps` without the cells that repeat another's numbers at this τ_keep: g = 0 (the gap rule
    off) and, per class of g that abstains the same active queries, its highest g (the strictest of
    the class, so the one a search would pick)."""
    if len(gaps) < 2:
        return gaps
    active = table.has & (table.p1 >= tau_keep)
    gi = np.unique(np.searchsorted(gaps, table.d[active], side="right"))
    gi = gi[gi < len(gaps)]
    tops = {int(v) - 1 for v in gi[1:]} | {len(gaps) - 1}
    if gi.size and gi[0] > 1:
        tops.add(int(gi[0]) - 1)
    return gaps[[0, *sorted(tops)]]


def _grid_blocks(
    table: _Table, tau_keep: float, gaps: np.ndarray, highs: np.ndarray | None = None
) -> tuple[np.ndarray, Iterator[tuple[int, dict[str, np.ndarray]]]]:
    """`_grid`'s numbers in blocks of consecutive τ_high rows: (highs, an iterator of (first row,
    metrics of that block)), so a scan holds one block, not the whole (H, G) grid. `highs` may be a
    subset of the full grid (it must start at `tau_keep`)."""
    q_idx = np.arange(len(table.p1))
    base_abstain = (~table.has) | (table.p1 < tau_keep)
    n = np.minimum((table.p >= tau_keep).sum(axis=1), table.limit)
    rows0 = np.where(base_abstain, 0, n)
    hits0 = table.cumrel[q_idx, rows0]
    active = ~base_abstain
    ans = table.answerable
    if highs is None:
        highs = _highs_for(table, tau_keep)
    n_h, n_g = len(highs), len(gaps)
    hi = np.searchsorted(highs, table.p1, side="right")
    gi = np.searchsorted(gaps, table.d, side="right")
    comps = np.stack([rows0, hits0, ans & (hits0 > 0), active & ans, active & ~ans, rows0 > 0],
                     axis=1).astype(float)  # fmt: skip
    totals = comps.sum(axis=0)
    sel = np.nonzero(active & (hi < n_h) & (gi < n_g))[0]
    sel = sel[np.argsort(hi[sel], kind="stable")]
    hi_s, gi_s, comps_s = hi[sel], gi[sel], comps[sel]
    n_ans, n_neg = int(ans.sum()), int((~ans).sum())
    step = max(1, BLOCK_CELLS // max(1, n_g))

    def blocks() -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        carry = np.zeros((comps.shape[1], n_g))
        for h0 in range(0, n_h, step):
            h1 = min(n_h, h0 + step)
            a, b = np.searchsorted(hi_s, [h0, h1], side="left")
            flat = (hi_s[a:b] - h0) * n_g + gi_s[a:b]
            acc = np.stack(
                [
                    np.bincount(flat, weights=comps_s[a:b, k], minlength=(h1 - h0) * n_g)
                    for k in range(comps.shape[1])
                ]
            ).reshape(comps.shape[1], h1 - h0, n_g)
            removed = acc.cumsum(axis=2).cumsum(axis=1) + carry[:, None, :]
            carry = removed[:, -1, :]
            rows, hits, covered, ans_on, neg_on, nonempty = (
                totals[k] - removed[k] for k in range(6))  # fmt: skip
            shape = rows.shape
            prec = np.divide(hits, rows, out=np.zeros(shape), where=rows > 0)
            ira = np.divide(rows - hits, nonempty, out=np.zeros(shape), where=nonempty > 0)
            cov = covered / n_ans if n_ans else np.zeros(shape)
            far = (n_ans - ans_on) / n_ans if n_ans else np.zeros(shape)
            abst = (n_neg - neg_on) / n_neg if n_neg else np.full(shape, np.nan)
            yield h0, {"precision": prec, "ira": ira, "coverage_any": cov,
                       "abstention_recall": abst, "far": far}  # fmt: skip

    return highs, blocks()


def _canonical(h0: int, n_rows: int, n_g: int) -> np.ndarray:
    """The block's cells that are one setting each: the gap rule off only as (0, 0), on needs both
    knobs."""
    hh = np.arange(h0, h0 + n_rows)[:, None] > 0
    return hh == (np.arange(n_g)[None, :] > 0)


def _grid(table: _Table, tau_keep: float, gaps: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """The five metrics at (`tau_keep`, τ_high, g) for every τ_high in step 5's grid at this τ_keep
    (τ_keep itself, then the cuts between the p1 values above it) and every g in `gaps`, as (H, G)
    arrays. Query q is abstained by the gap rule exactly when τ_high > p1_q and g > d_q: a
    quadrant of the grid, so the totals removed are a 2-D cumulative sum of per-query counts."""
    highs, blocks = _grid_blocks(table, tau_keep, gaps)
    parts = [m for _, m in blocks]
    return highs, {k: np.concatenate([m[k] for m in parts]) for k in parts[0]}


def _last_true(ok: np.ndarray) -> tuple[int, int] | None:
    """The (h, j) of the True cell with the highest h, then the highest j: the strictest."""
    hs, js = np.nonzero(ok)
    if hs.size == 0:
        return None
    h = int(hs.max())
    return h, int(js[hs == h].max())


def _tolerant(t: Target, v: np.ndarray) -> np.ndarray:
    return v >= t.threshold - 1e-12 if t.op == ">=" else v <= t.threshold + 1e-12


class _Scan:
    """One pass over a τ_keep's grid in blocks: per predicate the strictest cell (highest τ_high
    row, then highest g) where it holds, with the five numbers there, and the best value of every
    target over the canonical cells (what `Choice.targets` reports as `best`)."""

    def __init__(
        self,
        table: _Table,
        tau_keep: float,
        gaps: np.ndarray,
        targets: Sequence[Target],
        preds: dict[str, Any],
        highs: np.ndarray | None = None,
    ) -> None:
        self.highs, blocks = _grid_blocks(table, tau_keep, gaps, highs)
        self.cells: dict[str, tuple[int, int, dict[str, float]] | None] = dict.fromkeys(preds)
        self.best: dict[str, float] = {}
        for h0, m in blocks:
            canon = _canonical(h0, next(iter(m.values())).shape[0], len(gaps))
            for t in targets:
                v = float(t.better()(m[t.metric][canon]))
                prev = self.best.get(t.gate_metric)
                self.best[t.gate_metric] = v if prev is None else float(t.better()([prev, v]))
            for name, pred in preds.items():
                if (cell := _last_true(pred(m) & canon)) is not None:
                    h, j = cell
                    self.cells[name] = (h0 + h, j, {k: float(np.nan_to_num(v[h, j]))
                                                    for k, v in m.items()})  # fmt: skip


def _merge_best(best: dict[str, float], new: dict[str, float], targets: Sequence[Target]) -> None:
    for t in targets:
        v = new[t.gate_metric]
        prev = best.get(t.gate_metric)
        best[t.gate_metric] = v if prev is None else float(t.better()([prev, v]))


def _staircase(m: dict[str, np.ndarray], canonical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per FAR level the (h, j) with the highest abstention recall, ties by precision, then the
    strictest: the settings step 5 would consider at this τ_keep, one per FAR budget."""
    hs, js = np.nonzero(canonical)
    far, abst, prec = (np.nan_to_num(m[k][hs, js]) for k in ("far", "abstention_recall",
                                                            "precision"))  # fmt: skip
    order = np.lexsort((js, hs, prec, abst))[::-1]  # abst desc, then prec, h, j
    _, first = np.unique(far[order], return_index=True)  # first = the best cell of each FAR level
    picks = order[first]  # by FAR ascending
    if len(picks) > FRONT_LEVELS:
        picks = picks[np.unique(np.linspace(0, len(picks) - 1, FRONT_LEVELS).round().astype(int))]
    return hs[picks], js[picks]


def _points(
    tau_keep: float,
    highs: np.ndarray,
    gaps: np.ndarray,
    m: dict[str, np.ndarray],
    hs: np.ndarray,
    js: np.ndarray,
    targets: Sequence[Target],
) -> list[dict[str, Any]]:
    """Front candidates at the cells (hs, js): the setting, its five numbers (6 decimals) and the
    targets it misses. The gap rule off is written (τ_keep, τ_keep, 0)."""  # fmt: skip
    vals = {k: np.nan_to_num(m[k][hs, js]) for k in FRONT_KEYS}
    unmet = np.stack([~np.vectorize(t.met)(vals[t.metric]) if hs.size else np.zeros(0, bool)
                      for t in targets], axis=-1)  # fmt: skip
    off = (hs == 0) | (js == 0)
    th = np.where(off, tau_keep, highs[hs]).tolist()
    gp = np.where(off, 0.0, gaps[js]).tolist()
    rounded = {k: np.round(v, 6).tolist() for k, v in vals.items()}
    names = [t.gate_metric for t in targets]
    return [{"tau_keep": tau_keep, "tau_high": th[i], "gap": gp[i],
             **{k: rounded[k][i] for k in FRONT_KEYS},
             "unmet": [n for n, u in zip(names, unmet[i].tolist(), strict=True) if u]}
            for i in range(len(th))]  # fmt: skip


def _pareto(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The non-dominated points (better or equal on all five numbers and better on one dominates).
    Exact duplicates keep the strictest setting."""
    if not points:
        return []
    strict = sorted(points, key=lambda pt: (-pt["tau_keep"], -pt["tau_high"], -pt["gap"]))
    vals = np.array([[_SIGN[k] * pt[k] for k in FRONT_KEYS] for pt in strict])
    _, keep = np.unique(vals.round(9), axis=0, return_index=True)
    strict = [strict[i] for i in sorted(keep)]
    vals = vals[sorted(keep)]
    order = np.lexsort(vals.T[::-1])[::-1]  # precision desc, then the others
    front: list[int] = []
    fv = np.empty((0, vals.shape[1]))
    for i in order.tolist():
        v = vals[i]
        if fv.size and bool((np.all(fv >= v - 1e-12, axis=1) & np.any(fv > v + 1e-12,
                                                                        axis=1)).any()):  # fmt: skip
            continue  # dominated by a front member
        if fv.size:
            dominated = np.all(v >= fv - 1e-12, axis=1) & np.any(v > fv + 1e-12, axis=1)
            front = [f for f, d in zip(front, dominated.tolist(), strict=True) if not d]
            fv = fv[~dominated]
        front.append(i)
        fv = np.vstack([fv, v])
    return [strict[i] for i in front]


def _shortfall(pt: dict[str, Any], targets: Sequence[Target]) -> float:
    return sum(max(0.0, (t.threshold - pt[t.metric]) * (1 if t.op == ">=" else -1))
               for t in targets)  # fmt: skip


def _search(table: _Table, targets: Sequence[Target]) -> Choice:
    """Steps 4-5 on a table (module docstring). The strictest setting is the first τ_keep, strictest
    first, with a feasible cell (τ_high, g) of step 5's grid, and in it the highest τ_high, then g.
    The gap rule off is one setting, written (τ_keep, τ_keep, 0).

    coverage_any and FAR are monotone: at a fixed τ_keep the gap rule off has the highest
    coverage_any and the lowest FAR of any cell, and both only improve as τ_keep falls. So the
    τ_keep that can meet both are a suffix of the candidates, found by bisection; only those are
    scanned, each grid in blocks (`_Scan`), so memory is capped. Both fallbacks are one scan at a
    bisected τ_keep.

    Infeasible: the fallback is, in order: the strictest setting meeting coverage_any and FAR; else
    the strictest keeping the best reachable coverage_any (the pool-recall ceiling) with FAR met;
    else the open gate (0, 0, 0). The Pareto front (each τ_keep's FAR staircase, `_staircase`) and
    the `best` figures are built on the thinned grid (FRONT_TAUS x FRONT_CUTS x FRONT_CUTS), which
    is the whole grid whenever the table is within those caps. A feasibility scan that outgrows
    EXACT_CELLS goes on with the thinned grid and says so (`Choice.exact` False)."""
    gaps = np.array([0.0, *_between(table.d[table.has].tolist())])
    by_name = {t.gate_metric: t for t in targets}
    cov_t, far_t = by_name["coverage_any"], by_name["far"]
    cands = _tau_keep_candidates(table)
    ceiling = float(table.metrics(0.0, 0.0, 0.0)["coverage_any"])
    best: dict[str, float] = {}

    def first_reaching(cov_floor: float) -> int:
        """Index of the strictest candidate whose gap-off cell reaches `cov_floor` and meets FAR
        (len(cands) if none); monotone in the index, so bisected."""
        lo, hi = 0, len(cands)
        while lo < hi:
            mid = (lo + hi) // 2
            m = table.metrics(cands[mid], cands[mid], 0.0)
            ok = float(m["coverage_any"]) >= cov_floor - 1e-12 and far_t.met(float(m["far"]))
            lo, hi = (lo, mid) if ok else (mid + 1, hi)
        return lo

    def all_met(m: dict[str, np.ndarray]) -> np.ndarray:
        ok = np.ones(m["far"].shape, dtype=bool)
        for t in targets:
            ok &= _tolerant(t, m[t.metric])
        return ok

    def finish(tau_keep: float, highs: np.ndarray, gaps_: np.ndarray, cell: tuple, rule: str,
               exact: bool, points: list[dict[str, Any]] | None = None) -> Choice:  # fmt: skip
        h, j, vals = cell
        tau_high, gap = (tau_keep, 0.0) if h == 0 or j == 0 else (float(highs[h]), float(gaps_[j]))
        tgt = {t.gate_metric: {"op": t.op, "threshold": t.threshold,
                               "best": round(best[t.gate_metric], 6),
                               "met": t.met(best[t.gate_metric]),
                               "value": round(vals[t.metric], 6)}
               for t in targets}  # fmt: skip
        front = sorted(_pareto(points or []), key=lambda pt: (len(pt["unmet"]),
                                                              _shortfall(pt, targets),
                                                              -pt["tau_keep"]))  # fmt: skip
        return Choice(tau_keep, tau_high, gap, rule == "strictest", rule, tgt, front, exact)

    def scan(tau: float, preds: dict[str, Any], gaps_: np.ndarray) -> _Scan:
        s = _Scan(table, tau, gaps_, targets, preds)
        _merge_best(best, s.best, targets)
        return s

    exact, spent, first = True, 0, first_reaching(cov_t.threshold)
    for i in range(first, len(cands)):
        g_i = _active_gaps(table, cands[i], gaps)
        cells = len(_highs_for(table, cands[i])) * len(g_i)
        if spent + cells > EXACT_CELLS:
            exact = False
            break
        spent += cells
        s = scan(cands[i], {"strict": all_met}, g_i)
        if (cell := s.cells["strict"]) is not None:
            return finish(cands[i], s.highs, g_i, cell, "strictest", True)

    # No exact feasible cell: the fallback (exact) and the front and `best` (thinned grid).
    points: list[dict[str, Any]] = []
    cut_gaps = _thin(gaps, FRONT_CUTS)
    for tau in _thin(np.array(cands), FRONT_TAUS).tolist():
        highs = _thin(_highs_for(table, tau), FRONT_CUTS)
        _, blocks = _grid_blocks(table, tau, cut_gaps, highs)
        m = next(blocks)[1]
        canon = _canonical(0, len(highs), len(cut_gaps))
        _merge_best(
            best, {t.gate_metric: float(t.better()(m[t.metric][canon])) for t in targets}, targets
        )
        if not exact and (cell := _last_true(all_met(m) & canon)) is not None:
            # the exact scan gave up: the strictest cell of the thinned grid, flagged
            h, j = cell
            vals = {k: float(np.nan_to_num(v[h, j])) for k, v in m.items()}
            return finish(tau, highs, cut_gaps, (h, j, vals), "strictest", False)
        points += _points(tau, highs, cut_gaps, m, *_staircase(m, canon), targets)

    def cov_far(floor: float) -> Any:
        return lambda m: (m["coverage_any"] >= floor - 1e-12) & _tolerant(far_t, m["far"])

    for rule, floor in (("fallback_coverage_far", cov_t.threshold),
                        ("fallback_coverage_ceiling", ceiling)):  # fmt: skip
        if (i := first_reaching(floor)) < len(cands):
            g_i = _active_gaps(table, cands[i], gaps)
            s = scan(cands[i], {"fb": cov_far(floor)}, g_i)
            return finish(cands[i], s.highs, g_i, s.cells["fb"], rule, exact, points)
    open_ = {k: float(np.nan_to_num(v)) for k, v in table.metrics(0.0, 0.0, 0.0).items()}
    return finish(0.0, np.array([0.0]), gaps, (0, 0, open_), "fallback_open", exact, points)


def choose_thresholds(
    pools: Sequence[QueryPool], *, a: float, b: float, limit: int,
    targets: Sequence[Target] | None = None,
) -> Choice:  # fmt: skip
    """Steps 4-5: the strictest (τ_keep, τ_high, g) meeting all five P2 tune targets
    (`p2_targets()` unless given); see `_search` for the grid and the infeasible case."""
    return _search(_Table(pools, a, b, limit), p2_targets() if targets is None else targets)


def fit_gate(
    pools: Sequence[QueryPool], *, fit_id: str, fingerprint_hash: str, limit: int, fitted_at: str,
    targets: Sequence[Target] | None = None,
) -> Gate:  # fmt: skip
    """Steps 3-5 and 7: the gate for `fingerprint_hash` (= fp_all), with its tune metrics, whether
    the fit is feasible, the rule that chose it, the five targets and (infeasible) the front."""
    z = [i.z for q in pools for i in q.items]
    y = [i.rel >= 1 for q in pools for i in q.items]
    a, b = fit_platt(z, y)
    choice = choose_thresholds(pools, a=a, b=b, limit=limit, targets=targets)
    gate = Gate(a, b, choice.tau_keep, choice.tau_high, choice.gap, fit_id=fit_id,
                fitted_at=fitted_at, fingerprint_hash=fingerprint_hash)  # fmt: skip
    table = _Table(pools, a, b, limit)
    m = table.metrics(choice.tau_keep, choice.tau_high, choice.gap)
    tune: dict[str, Any] = {k: round(float(v), 6) for k, v in m.items()} | {"queries": len(pools)}
    tune |= {"feasible": choice.feasible, "rule": choice.rule, "strictest": STRICTEST,
             "targets": choice.targets}  # fmt: skip
    if not choice.exact:
        tune["search_exact"] = False  # the strictest-setting scan was thinned: see `_search`
    if not choice.feasible:
        tune["front"] = choice.front
    tune["per_category"] = table.per_category(choice.tau_keep, choice.tau_high, choice.gap)
    return Gate(**{**asdict(gate), "tune": tune})


# ---- the tune report ----------------------------------------------------------------------------

FRONTIER_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FRONT_ROWS_SHOWN = 15


def _z_of(gate: Gate, p: float) -> float | None:
    """The raw logit at which p = σ(a·z + b) equals `p` (None at p ∈ {0, 1} or a = 0)."""
    if not 0 < p < 1 or gate.platt_a == 0:
        return None
    return round((math.log(p / (1 - p)) - gate.platt_b) / gate.platt_a, 4)


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(z_rel > z_irrel) + ½ P(tie): does the raw logit rank relevant chunks above the rest?"""
    if pos.size == 0 or neg.size == 0:
        return math.nan
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def tune_report(
    pools: Sequence[QueryPool], gate: Gate, *, limit: int, targets: Sequence[Target] | None = None
) -> dict[str, Any]:
    """What `fit_gate` saw and what bound it (module docstring). Pure: recomputed from the records
    and the gate, so `mem calibrate --records` reproduces it without a model. The choice (feasible,
    rule, targets, front) is read from `gate.tune` when the gate carries it (every gate `fit_gate`
    writes), else re-derived."""
    targets = p2_targets() if targets is None else targets
    table = _Table(pools, gate.platt_a, gate.platt_b, limit, gate.reader)
    if "feasible" in gate.tune:
        choice = {k: gate.tune.get(k) for k in ("feasible", "rule", "targets")}
        choice["front"] = gate.tune.get("front", [])
    else:
        c = _search(table, targets)
        choice = {"feasible": c.feasible, "rule": c.rule, "targets": c.targets, "front": c.front}
    n_ans = int(table.answerable.sum())
    cov0 = float(table.metrics(0.0, 0.0, 0.0)["coverage_any"])
    cands = sorted({0.0, *table.p[table.p >= 0].tolist()})
    f05 = {tau: float(table.metrics(tau, tau, 0.0)["f05"]) for tau in cands}
    best_free = min(t for t in cands if f05[t] >= max(f05.values()) - 1e-12)  # ties: lower tau

    def row(tau: float, tau_high: float | None = None, gap: float = 0.0) -> dict[str, Any]:
        m = table.metrics(tau, tau if tau_high is None else tau_high, gap)
        rows, _ = table.rows_hits(tau, tau if tau_high is None else tau_high, gap)
        nonempty = int((rows > 0).sum())
        vals = {k: float(np.nan_to_num(v)) for k, v in m.items()}
        return {"tau_keep": tau, "z": _z_of(gate, tau),
                **{k: _nan_none(float(v)) for k, v in m.items()},
                "rows_per_answer": _ratio(int(rows.sum()), nonempty),
                "feasible": all(t.met(vals[t.metric]) for t in targets)}  # fmt: skip

    taus = sorted({0.0, gate.tau_keep, best_free, *FRONTIER_GRID})
    z_all = np.array([i.z for q in pools for i in q.items], dtype=float)
    rel = np.array([i.rel >= 1 for q in pools for i in q.items], dtype=bool)
    z_rel, z_irr = z_all[rel], z_all[~rel]
    inputs = {"n_items": int(z_all.size), "n_rel": int(rel.sum()), "n_irrel": int((~rel).sum()),
              "z_min": _nan_none(float(z_all.min())) if z_all.size else None,
              "z_max": _nan_none(float(z_all.max())) if z_all.size else None,
              "z_rel_median": _nan_none(float(np.median(z_rel))) if z_rel.size else None,
              "z_rel_p10": _nan_none(float(np.percentile(z_rel, 10))) if z_rel.size else None,
              "z_irrel_median": _nan_none(float(np.median(z_irr))) if z_irr.size else None,
              "z_irrel_p90": _nan_none(float(np.percentile(z_irr, 90))) if z_irr.size else None,
              "auroc_z": _nan_none(_auroc(z_rel, z_irr)),
              "platt_a": gate.platt_a, "platt_b": gate.platt_b}  # fmt: skip
    warnings = []
    if gate.platt_a <= 0:
        warnings.append("platt_a <= 0: higher logits are less often relevant; the logit sign "
                        "is inverted or the labels are attached to the wrong chunks")  # fmt: skip
    if z_all.size and 0.0 <= float(z_all.min()) and float(z_all.max()) <= 1.0:
        warnings.append("every z lies in [0, 1]: these look like sigmoid outputs, not the raw "
                        "logits step 5 records")  # fmt: skip
    if inputs["auroc_z"] is not None and inputs["auroc_z"] < 0.5:
        warnings.append(f"AUROC(z, rel) = {inputs['auroc_z']:.3f} < 0.5 (sign)")
    gated = row(gate.tau_keep, gate.tau_high, gate.gap)
    return {
        "queries": len(pools),
        "answerable": n_ans,
        "negative": len(pools) - n_ans,
        "coverage_at_zero": round(cov0, 6),
        "choice": {
            "tau_keep": gate.tau_keep,
            "tau_high": gate.tau_high,
            "gap": gate.gap,
            "z": _z_of(gate, gate.tau_keep),
            "strictest": STRICTEST,
            **choice,
            "front_size": len(choice["front"]),
            **{k: gated[k] for k in (*FRONT_KEYS, "f05", "rows_per_answer")},
        },
        "unconstrained_f05": row(best_free),
        "inputs": inputs,
        "warnings": warnings,
        "per_category": table.per_category(gate.tau_keep, gate.tau_high, gate.gap),
        "frontier": [row(t) for t in taus],
    }


def _cell(x: Any, width: int = 6, digits: int = 3) -> str:
    return f"{x:>{width}.{digits}f}" if isinstance(x, int | float) else f"{'-':>{width}}"


RULES = {
    "strictest": "the strictest setting meeting all five targets on tune",
    "fallback_coverage_far": "FALLBACK: the strictest setting meeting coverage_any and far only",
    "fallback_coverage_ceiling": "FALLBACK: the strictest setting keeping the best reachable "
                                 "coverage_any (the pool-recall ceiling) with far met",
    "fallback_open": "FALLBACK: the open gate (0, 0, 0); even it abstains on too many answerables",
}  # fmt: skip


def format_report(report: dict[str, Any]) -> str:
    """The human form of `tune_report`, printed by `mem calibrate` before the gate line."""
    ch, inp, free = report["choice"], report["inputs"], report["unconstrained_f05"]
    c = _cell
    n, n_ans, n_neg = report["queries"], report["answerable"], report["negative"]
    out = [f"tune: {n} queries ({n_ans} answerable, {n_neg} negative)"]
    out.append(
        f"inputs: {inp['n_items']} pool items, {inp['n_rel']} rel>=1; z in "
        f"[{c(inp['z_min'], 1, 2)}, {c(inp['z_max'], 1, 2)}]; median z rel "
        f"{c(inp['z_rel_median'], 1, 2)} vs irrel {c(inp['z_irrel_median'], 1, 2)}; "
        f"AUROC(z) {c(inp['auroc_z'], 1, 3)}; platt a={inp['platt_a']:.4f} b={inp['platt_b']:.4f}"
    )
    targets = ", ".join(f"{k} {t['op']} {t['threshold']:g}" for k, t in ch["targets"].items())
    out.append(f"targets (P2 tune, bench/configs/gates.yaml): {targets}")
    out.append(
        f"chosen: tau_keep {ch['tau_keep']:.4f} (z {c(ch['z'], 1, 2)}) tau_high {ch['tau_high']:.4f}"
        f" gap {ch['gap']:.4f}; feasible: {'yes' if ch['feasible'] else 'NO'}; rule: {ch['rule']} "
        f"({RULES.get(ch['rule'], ch['rule'])}; strictest = {ch['strictest']})"
    )
    out.append(
        f"  on tune: precision {c(ch['precision'], 1)} ira {c(ch['ira'], 1)} coverage_any "
        f"{c(ch['coverage_any'], 1)} abstention_recall {c(ch['abstention_recall'], 1)} far "
        f"{c(ch['far'], 1)} (f05 {c(ch['f05'], 1)}, rows/answer {c(ch['rows_per_answer'], 1, 2)})"
    )
    out.append(
        f"unconstrained F0.5 optimum (gap rule off): tau_keep {free['tau_keep']:.4f} f05 "
        f"{c(free['f05'], 1)} precision {c(free['precision'], 1)} coverage_any "
        f"{c(free['coverage_any'], 1)} ira {c(free['ira'], 1)}"
    )
    out.append(f"{'per category:':<30} n_ans n_neg poolrec   cov  prec   ira   far  abst  rows/ans")
    for cat, r in report["per_category"].items():
        out.append(
            f"  {cat[:28]:<28} {r['n_answerable']:>5} {r['n_negative']:>5} "
            f"{c(r['pool_recall'], 7)} {c(r['coverage_any'], 5)} {c(r['precision'], 5)} "
            f"{c(r['ira'], 5)} {c(r['far'], 5)} {c(r['abstention_recall'], 5)} "
            f"{c(r['rows_per_answer'], 9, 2)}"
        )
    out.append(
        f"{'frontier (gap rule off: tau_high = tau_keep, g = 0):':<52} tau       z  prec   cov"
        "   ira   far   f05  abst rows/ans  feas"
    )
    for r in report["frontier"]:
        mark = " <- tau_keep" if r["tau_keep"] == ch["tau_keep"] else ""
        out.append(
            f"  {c(r['tau_keep'], 56, 4)} {c(r['z'], 7, 2)} {c(r['precision'], 5)} "
            f"{c(r['coverage_any'], 5)} {c(r['ira'], 5)} {c(r['far'], 5)} {c(r['f05'], 5)} "
            f"{c(r['abstention_recall'], 5)} {c(r['rows_per_answer'], 8, 2)} "
            f"{'yes' if r['feasible'] else 'no':>5}{mark}"
        )
    if not ch["feasible"]:
        front = ch["front"]
        shown = front[:FRONT_ROWS_SHOWN]
        out.append(
            f"Pareto front on tune: {len(front)} non-dominated settings, nearest the targets first"
            + (f" (first {len(shown)}; all in the report JSON)" if len(shown) < len(front) else "")
            + ":"
        )
        out.append(f"  {'tau_keep':>8} {'tau_high':>8} {'gap':>6}  prec   ira   cov  abst   far  "
                   "unmet")  # fmt: skip
        for pt in shown:
            out.append(
                f"  {c(pt['tau_keep'], 8, 4)} {c(pt['tau_high'], 8, 4)} {c(pt['gap'], 6, 4)} "
                f"{c(pt['precision'], 5)} {c(pt['ira'], 5)} {c(pt['coverage_any'], 5)} "
                f"{c(pt['abstention_recall'], 5)} {c(pt['far'], 5)}  {', '.join(pt['unmet'])}"
            )
    out += [f"WARNING: {w}" for w in report["warnings"]]
    if not ch["feasible"]:  # last, so it sits just above the gate line
        out.append(infeasible_line(ch))
    return "\n".join(out)


def infeasible_line(ch: dict[str, Any]) -> str:
    """One line: which P2 target(s) no setting meets on tune, with the best value each reaches
    (if every target is met by some setting alone, all are named: none meets them jointly), and
    which fallback the written gate is."""
    targets = ch["targets"]
    unmet = {k: t for k, t in targets.items() if not t["met"]} or targets
    what = "; ".join(f"{k} {t['op']} {t['threshold']:g} (best {t['best']:.3f})"
                     for k, t in unmet.items())  # fmt: skip
    joint = "" if any(not t["met"] for t in targets.values()) else " jointly"
    return (
        f"INFEASIBLE: no (tau_keep, tau_high, g) meets the P2 target"
        f"{'s' if len(unmet) > 1 else ''}{joint} on tune: {what}; the written gate "
        f"(tau_keep {ch['tau_keep']:.4f} tau_high {ch['tau_high']:.4f} gap {ch['gap']:.4f}) is "
        f"{RULES.get(ch['rule'], ch['rule']).removeprefix('FALLBACK: ')}, flagged infeasible; "
        "no target was relaxed"
    )
