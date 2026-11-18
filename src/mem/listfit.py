"""tau_list: the list mode's own cut (MEM2-193, ADR 0023), fitted on LME tune by nested CV.

Data: the answerable LME tune pools (`mem calibrate` records) whose question the generic detector
(`mem.aggcue`) flags as an aggregation and that hold at least one gold row. p = σ(a·z + b) of the
gate's own Platt. The unit is the *session* (the key's part before `#`; a key without one is its own
session): a list answer is complete when every gold session has a row in it. Pools are the reranked
top-N candidates, so coverage is of the gold sessions inside the pool (the pool ceiling is B-R1's).

Floors rule, as elsewhere: the strictest cut (highest tau, fewest rows) whose training mean
coverage of gold sessions, after the token cap has dropped the lowest-p rows, is at least the floor;
when no cut reaches it the cut with the best coverage is written and flagged `floor_met: false`.
The floor is chosen from FLOORS by an inner CV on the training folds only (the highest floor whose
inner held-out coverage stays within MARGIN of it). Honest estimate: the outer folds, assigned by a
hash of the question id (adding or changing one question moves no other), each refit everything on
the other folds and are scored once. The final cut is chosen on all pools (flat CV: optimistic,
labelled so).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mem import aggcue, listmode
from mem.calibrate import QueryPool

FOLDS = 5
INNER = 4
FLOORS = (0.80, 0.85, 0.90, 0.95)
MARGIN = 0.03
MAX_CUTS = 200


def session_of(key: str) -> str:
    return key.split("#", 1)[0]


# `mem calibrate` records name a pool `<question id>:<kind>` (bench.datasets.calibration), so a
# question's answerable and negative runs are two pools; the manifest and the LME file key the bare id.
POOL_KINDS = ("answerable", "negative")


def question_of(qid: str) -> str:
    """The question id of a pool: its qid without a `:<kind>` suffix (a bare id is returned as is)."""
    head, sep, kind = qid.rpartition(":")
    return head if sep and kind in POOL_KINDS else qid


def fold_of(qid: str, k: int) -> int:
    return int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:8], 16) % k


def query_fold(q: Query, k: int, inner: bool = False) -> int:
    """A query's fold, by the hash of its *question* id: a question's pools never split."""
    return fold_of(("i" if inner else "") + question_of(q.qid), k)


# what a list answer must cover: the gold sessions (ADR 0023 §4) or the gold rounds (MEM2-268)
UNITS = ("session", "round")


@dataclass(frozen=True)
class Query:
    qid: str
    p: np.ndarray  # p of each pool item, descending
    sess: list[str]  # session of each item, same order (the token cost groups by it)
    gold: frozenset[str]  # the units (sessions, or rounds: MEM2-268) with a rel >= 1 item
    unit: list[str] | None = None  # the coverage unit of each item; None: its session

    def units(self) -> list[str]:
        return self.sess if self.unit is None else self.unit


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def eligible(
    pools: Sequence[QueryPool], queries: Mapping[str, str], platt: tuple[float, float],
    unit: str = "session",
) -> list[Query]:  # fmt: skip
    """The queries the fit runs on. `unit`: what a list answer must cover, the gold *sessions* (ADR
    0023 §4, the default) or the gold *rounds* (the pool keys themselves; MEM2-268's dry run)."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    a, b = platt
    out = []
    for q in pools:
        # ADR 0023 §4: the answerable pools only (a negative run has no gold session to cover)
        if not q.answerable or not aggcue.is_aggregation(queries.get(question_of(q.qid), "")):
            continue
        items = sorted(q.items, key=lambda i: -(a * i.z + b))
        key_of = session_of if unit == "session" else (lambda k: k)
        gold = frozenset(key_of(i.key) for i in items if i.rel >= 1)
        if gold:
            out.append(Query(q.qid, np.array([sigmoid(a * i.z + b) for i in items]),
                             [session_of(i.key) for i in items], gold,
                             None if unit == "session" else [i.key for i in items]))  # fmt: skip
    return out


def deliver(q: Query, tau: float, cap: int) -> tuple[list[int], int, int]:
    """The rows list mode would deliver: p >= tau, lowest p dropped until the cost model fits the
    cap. Returns (item indexes, tokens, tokens before the cap)."""
    rows = [i for i in range(len(q.p)) if q.p[i] >= tau]
    demand = listmode.tokens_for(len(rows), len({q.sess[i] for i in rows}))
    while rows and listmode.tokens_for(len(rows), len({q.sess[i] for i in rows})) > cap:
        rows.pop()
    return (
        rows,
        listmode.tokens_for(len(rows), len({q.sess[i] for i in rows})) if rows else 0,
        demand,
    )


def score(qs: Sequence[Query], tau: float, cap: int) -> dict[str, Any]:
    cov, toks, demand, trunc, rows = [], [], [], 0, []
    for q in qs:
        idx, t, d = deliver(q, tau, cap)
        units = q.units()
        got = {units[i] for i in idx} & q.gold
        cov.append(len(got) / len(q.gold))
        toks.append(t)
        demand.append(d)
        rows.append(len(idx))
        trunc += d > cap
    n = len(qs)
    return {"n": n, "coverage": float(np.mean(cov)) if n else None,
            "tokens_p95": float(np.percentile(toks, 95)) if n else None,
            "demand_p95": float(np.percentile(demand, 95)) if n else None,
            "truncated": trunc / n if n else None,
            "rows_mean": float(np.mean(rows)) if n else None}  # fmt: skip


def cuts(qs: Sequence[Query]) -> list[float]:
    """Candidate taus: the distinct p values of the pools (a row is in iff p >= tau), thinned."""
    v = np.unique(np.concatenate([q.p for q in qs])) if qs else np.array([])
    v = v[(v > 0.0) & (v < 1.0)]
    if len(v) > MAX_CUTS:
        v = np.unique(np.quantile(v, np.linspace(0, 1, MAX_CUTS), method="nearest"))
    return [float(x) for x in v]


def fit_tau(qs: Sequence[Query], floor: float, cap: int) -> dict[str, Any]:
    """The strictest tau whose coverage on `qs` is >= floor, else the best-coverage tau."""
    best: tuple[float, float] | None = None
    for tau in sorted(cuts(qs), reverse=True):
        c = score(qs, tau, cap)["coverage"]
        if c >= floor:
            return {"tau_list": tau, "floor_met": True, "train_coverage": c}
        if best is None or c > best[1]:
            best = (tau, c)
    if best is None:
        raise ValueError("tau_list: no candidate cut (empty or degenerate pools)")
    return {"tau_list": best[0], "floor_met": False, "train_coverage": best[1]}


def choose_floor(qs: Sequence[Query], cap: int) -> tuple[float, dict[str, float]]:
    """Inner CV on `qs` only: the highest floor whose held-out coverage is within MARGIN of it."""
    inner = {}
    for fl in FLOORS:
        cov = []
        for f in range(INNER):
            tr = [q for q in qs if query_fold(q, INNER, inner=True) != f]
            te = [q for q in qs if query_fold(q, INNER, inner=True) == f]
            if tr and te:
                cov.append(score(te, fit_tau(tr, fl, cap)["tau_list"], cap)["coverage"])
        inner[fl] = float(np.mean(cov)) if cov else 0.0
    ok = [fl for fl in FLOORS if inner[fl] >= fl - MARGIN]
    return (max(ok) if ok else min(FLOORS)), inner


def fit(qs: Sequence[Query], cap: int) -> dict[str, Any]:
    floor, inner = choose_floor(qs, cap)
    return {**fit_tau(qs, floor, cap), "floor": floor, "inner_coverage": inner}


def nested_cv(
    qs: Sequence[Query], cap: int = listmode.CAP_TOKENS, k: int = FOLDS
) -> dict[str, Any]:
    if len(qs) < 2 * k:
        raise ValueError(f"nested CV needs at least {2 * k} aggregation questions, got {len(qs)}")
    outer = []
    for f in range(k):
        tr = [q for q in qs if query_fold(q, k) != f]
        te = [q for q in qs if query_fold(q, k) == f]
        if not te:
            continue
        fit_f = fit(tr, cap)
        outer.append({"fold": f, "n_train": len(tr), "floor": fit_f["floor"],
                      "floor_met": fit_f["floor_met"], "tau_list": fit_f["tau_list"],
                      "train_coverage": round(fit_f["train_coverage"], 6),
                      "heldout": _round(score(te, fit_f["tau_list"], cap))})  # fmt: skip
    cov = [o["heldout"]["coverage"] for o in outer]
    return {"outer_folds": outer, "k": k, "cap_tokens": cap,
            "mean_heldout_coverage": round(float(np.mean(cov)), 6),
            "worst_heldout_coverage": round(float(np.min(cov)), 6),
            "max_heldout_tokens_p95": max(o["heldout"]["tokens_p95"] for o in outer),
            "n": len(qs), "feasible": all(o["floor_met"] for o in outer)}  # fmt: skip


def _round(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def fit_list_model(
    pools: Sequence[QueryPool],
    queries: Mapping[str, str],
    platt: tuple[float, float],
    cap: int = listmode.CAP_TOKENS,
    k: int = FOLDS,
    unit: str = "session",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the gate's `list_model` dict, the full report). Refuses nothing: the caller reads
    `report['nested']['feasible']` (every outer fold met its floor). `unit` as in `eligible`; the
    gate's list model is fitted on sessions (ADR 0023 §4), the round unit is the dry run's."""
    qs = eligible(pools, queries, platt, unit)
    nested = nested_cv(qs, cap, k)
    final = fit(qs, cap)
    nested = {**nested, "final": {"flat_cv_optimistic": True, "floor": final["floor"],
                                  "floor_met": final["floor_met"],
                                  "train_coverage": round(final["train_coverage"], 6)}}  # fmt: skip
    lm = listmode.ListModel.build(
        final["tau_list"], cap, final["floor"], platt[0], platt[1], nested
    )
    return lm.to_dict(), {"nested": nested, "tau_list": final["tau_list"], "n": len(qs),
                          "inner_coverage": final["inner_coverage"]}  # fmt: skip
