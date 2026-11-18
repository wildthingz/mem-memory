"""B-R3 on PG-v1 (benchmarks-v2 B-R3, §2.0): run a backend over a split and score it.

Nothing here touches production code: the backend answers `retrieve(question)` with the three lists
of §0.3 as gold ids `path::chunk_key` (`bench.lme_eval.Retrieved`).

Definitions (decision record 0010):
- Ranking metrics (nDCG@10 std with graded rel 2/1, Recall_any@5, MRR) on R_q over the answerable
  questions. nDCG@10 (std) only: the upstream ndcg_any is an LME-only metric (A6) and is not
  computed here.
- Delivered-context metrics (coverage_any, coverage_all, delivered precision, IRA, FAR, abstention
  recall, F0.5) on S_q over answerable + no-answer questions, §2.0 verbatim; coverage_all needs every
  rel-2 chunk delivered.
- Baselines are descriptive. Legs (BM25, dense, RRF with no reranker) are scored at chunk level like
  ours. qmd and `rg -il` return files or their own chunks, so the DR-4 comparison is at **path
  level** for every system: a path's rel is the best rel of its gold chunks, a ranking is
  deduplicated by path in rank order, nDCG@10 (std). DR-4 rule: if qmd's path-level nDCG@10 point
  estimate is >= ours, DR-4 is reopened (recorded, never a pass/fail gate).
- Unjudged@10 per system: the share of its top 10 (chunks; paths for qmd/rg: a path with no judged
  chunk) that nobody judged. Pooling used ours, BM25 and dense only, so qmd and rg are the systems
  a pooling bias hurts; the share is reported next to their score.
- CIs: stratified percentile bootstrap over questions, strata = category, B = 10,000 (§5.2).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from bench.gold import pg
from bench.gold.chunks import split_gold_id
from bench.lme_eval import Retrieved
from bench.metrics import delivered as dv
from bench.metrics import ir, stats
from bench.metrics.tokens import est_tokens

TOP8 = 8
LEGS = ("bm25", "dense", "rrf")


class PgBackend(Protocol):
    def retrieve(self, q: pg.Question) -> Retrieved: ...


@dataclass
class PgItem:
    q: pg.Question
    r: Retrieved
    top8_text: str = ""

    @property
    def abstained(self) -> bool:
        return not self.r.S

    def record(self) -> dict[str, Any]:
        return {"qid": self.q.qid, "category": self.q.category, "kind":
                "answerable" if self.q.answerable else "no-answer", "filters": self.q.filters,
                "F": self.r.F[:100], "R": self.r.R[:50], "S": self.r.S, "abstained": self.abstained,
                "gold": dict(sorted(self.q.gold.items())), "latency_ms": self.r.latency_ms,
                "p_scores": self.r.p_scores[:5], "legs": {k: v[:50] for k, v in self.r.legs.items()},
                "tokens_est": est_tokens(self.r.text), "response_chars": len(self.r.text)}  # fmt: skip


def run_pg(
    backend: PgBackend, questions: Sequence[pg.Question], *,
    on_item: Callable[[PgItem], None] | None = None,
) -> list[PgItem]:  # fmt: skip
    items = []
    for q in sorted(questions, key=lambda q: q.qid):
        it = PgItem(q, backend.retrieve(q))
        items.append(it)
        if on_item:
            on_item(it)
    return items


# ---- path level (DR-4) --------------------------------------------------------------------------


def _path(gid: str) -> str:
    """A chunk's path; a memory id (`m12`, PG-v2) is its own path."""
    return split_gold_id(gid)[0] if "::" in gid else gid


def path_gold(gold: Mapping[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for g, rel in gold.items():
        p = _path(g)
        out[p] = max(out.get(p, 0), rel)
    return out


def dedupe_paths(ranked: Sequence[str]) -> list[str]:
    """A chunk ranking (gold ids) as a path ranking: first occurrence of each path."""
    seen: dict[str, None] = {}
    for g in ranked:
        seen.setdefault(_path(g), None)
    return list(seen)


def ndcg_path(ranked_paths: Sequence[str], gold: Mapping[str, int]) -> float:
    return ir.ndcg_std(list(ranked_paths), path_gold(gold), 10)


# ---- scoring ------------------------------------------------------------------------------------


@dataclass
class PgScores:
    metrics: dict[str, float] = field(default_factory=dict)
    ci95: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(default_factory=dict)


def _delivered(items: Sequence[PgItem]) -> list[dv.Delivered]:
    return [dv.Delivered(i.r.S, frozenset(i.q.gold),  i.q.answerable,
                         frozenset(g for g, r in i.q.gold.items() if r == 2)) for i in items]  # fmt: skip


def score_pg(
    items: Sequence[PgItem], *, baselines: Mapping[str, Mapping[str, list[str]]] | None = None,
    b: int = stats.B_DEFAULT, seed: int = stats.SEED,
) -> PgScores:  # fmt: skip
    """`baselines`: name -> {qid: ranked paths} for systems outside the pipeline (qmd, rg)."""
    out = PgScores()
    ans = [i for i in items if i.q.answerable]
    out.n = {"answerable": len(ans), "no_answer": len(items) - len(ans), "all": len(items)}
    if not ans:
        out.notes.append("no answerable questions")
        return out
    strata = [i.q.category for i in ans]
    vec: dict[str, list[float]] = defaultdict(list)
    for i in ans:
        g = dict(i.q.gold)
        gs = set(g)
        vec["ndcg_std10_R"].append(ir.ndcg_std(i.r.R, g, 10))
        vec["recall_any5_R"].append(ir.recall_any(i.r.R, gs, 5))
        vec["recall_any5_F"].append(ir.recall_any(i.r.F, gs, 5))
        vec["mrr_R"].append(ir.mrr(i.r.R, gs))
        for leg, refs in i.r.legs.items():
            vec[f"ndcg_std10_{leg}"].append(ir.ndcg_std(refs, g, 10))
            vec[f"recall_any5_{leg}"].append(ir.recall_any(refs, gs, 5))
        vec["ndcg_std10_path_R"].append(ndcg_path(dedupe_paths(i.r.R), g))
        for leg, refs in i.r.legs.items():
            vec[f"ndcg_std10_path_{leg}"].append(ndcg_path(dedupe_paths(refs), g))
    for name, runs in (baselines or {}).items():
        missing = sorted(i.q.qid for i in ans if i.q.qid not in runs)
        if missing:  # an absent question is not a miss: the baseline run is incomplete
            raise ValueError(
                f"baseline {name!r} has no run for {len(missing)} of {len(ans)} "
                f"answerable questions (first: {missing[:3]}): re-run it on this gold"
            )
        vec[f"ndcg_std10_path_{name}"] = [ndcg_path(runs[i.q.qid][:10], i.q.gold)
                                         for i in ans]  # fmt: skip
    out.vectors = dict(vec)
    for name, vals in vec.items():
        out.metrics[name], lo, hi = stats.bootstrap_mean(vals, strata, b, seed)
        out.ci95[name] = (lo, hi)
    out.metrics["n_answerable"] = float(len(ans))

    dl = _delivered(items)
    cats = [i.q.category for i in items]
    out.metrics |= {"coverage_any": dv.coverage_any(dl), "coverage_all": dv.coverage_all(dl),
                    "delivered_precision": dv.delivered_precision(dl), "ira": dv.ira(dl),
                    "far": dv.far(dl), "abstention_recall": dv.abstention_recall(dl)}  # fmt: skip
    out.metrics["delivered_f05"] = dv.f05(out.metrics["delivered_precision"],
                                          out.metrics["coverage_any"])  # fmt: skip
    a = [float(i.q.answerable) for i in items]
    na = [1.0 - x for x in a]
    S = lambda i: set(i.r.S)
    gold = lambda i: set(i.q.gold)
    out.ci95["coverage_any"] = stats.bootstrap_ratio(
        [float(bool(S(i) & gold(i))) for i in items], a, cats, b, seed
    )[1:]
    out.ci95["coverage_all"] = stats.bootstrap_ratio(
        [float(i.q.answerable and {g for g, r in i.q.gold.items() if r == 2} <= S(i)) for i in items],
        a, cats, b, seed)[1:]  # fmt: skip
    out.ci95["delivered_precision"] = stats.bootstrap_ratio(
        [len(S(i) & gold(i)) for i in items], [len(i.r.S) for i in items], cats, b, seed
    )[1:]
    out.ci95["ira"] = stats.bootstrap_ratio(
        [len(S(i) - gold(i)) for i in items], [float(bool(i.r.S)) for i in items], cats, b, seed
    )[1:]
    out.ci95["far"] = stats.bootstrap_ratio(
        [float(i.q.answerable and not i.r.S) for i in items], a, cats, b, seed
    )[1:]
    out.ci95["abstention_recall"] = stats.bootstrap_ratio(
        [float(not i.q.answerable and not i.r.S) for i in items], na, cats, b, seed
    )[1:]

    if all("rrf" in i.r.legs for i in ans):
        rrf8 = [dv.Delivered(i.r.legs["rrf"][:TOP8] if i.q.answerable else [], frozenset(i.q.gold),
                             i.q.answerable) for i in items]  # fmt: skip
        out.metrics["rrf8_coverage_any"] = dv.coverage_any(rrf8)

    # macro beside micro, per-category values
    for name in ("ndcg_std10_R", "recall_any5_R"):
        by: dict[str, list[float]] = defaultdict(list)
        for c, x in zip(strata, vec[name], strict=True):
            by[c].append(x)
        out.per_category[name] = {c: sum(v) / len(v) for c, v in sorted(by.items())}
        out.metrics[f"{name}_macro"] = sum(out.per_category[name].values()) / len(by)
    by_cat: dict[str, list[dv.Delivered]] = defaultdict(list)
    for i, d in zip(items, dl, strict=True):
        by_cat[i.q.category].append(d)
    fns = {"coverage_any": dv.coverage_any, "delivered_precision": dv.delivered_precision,
           "ira": dv.ira, "far": dv.far, "abstention_recall": dv.abstention_recall}  # fmt: skip
    for name, fn in fns.items():
        out.per_category[name] = {c: fn(v) if v else math.nan for c, v in sorted(by_cat.items())}
    out.per_category["n"] = {c: float(len(v)) for c, v in sorted(by_cat.items())}

    # DR-4 (descriptive): qmd's path-level nDCG@10 point estimate against ours.
    if "ndcg_std10_path_qmd" in out.metrics:
        qmd_n, ours_n = out.metrics["ndcg_std10_path_qmd"], out.metrics["ndcg_std10_path_R"]
        out.metrics["dr4_reopen"] = float(qmd_n >= ours_n)
        p = stats.paired_bootstrap(vec["ndcg_std10_path_R"], vec["ndcg_std10_path_qmd"], strata,
                                   b, seed)  # fmt: skip
        out.metrics |= {"dr4_delta_ours_minus_qmd": p.delta, "dr4_delta_ci_lo": p.lo,
                        "dr4_delta_ci_hi": p.hi}  # fmt: skip
        out.notes.append(f"DR-4: qmd {qmd_n:.3f} vs ours {ours_n:.3f} (path-level nDCG@10 std); "
                         f"{'REOPENED' if qmd_n >= ours_n else 'not reopened'}")  # fmt: skip
    lat = [i.r.latency_ms for i in items if i.r.latency_ms]
    if lat:
        out.metrics["latency_ms_p95"] = stats.percentile_nearest_rank(lat, 0.95)
    toks = [est_tokens(i.r.text) for i in items]
    out.metrics["tokens_est_mean"] = sum(toks) / len(toks)
    out.metrics["tokens_est_p95"] = stats.percentile_nearest_rank(toks, 0.95)
    # pooling bias: how much of each system's top 10 was never judged.
    judged = {i.q.qid: set(i.q.gold) | set(i.q.nonrel) for i in ans}
    for name, pick in (
        ("R", lambda i: i.r.R[:10]),
        *((leg, lambda i, leg=leg: i.r.legs.get(leg, [])[:10]) for leg in LEGS),
    ):
        tot = sum(len(pick(i)) for i in ans)
        if tot:
            out.metrics[f"unjudged_at10_{name}"] = (
                sum(g not in judged[i.q.qid] for i in ans for g in pick(i)) / tot
            )
    for name, runs in (baselines or {}).items():
        judged_paths = {i.q.qid: {_path(g) for g in judged[i.q.qid]} for i in ans}
        tot = sum(len(runs.get(i.q.qid, [])[:10]) for i in ans)
        if tot:
            out.metrics[f"unjudged_at10_path_{name}"] = (
                sum(p not in judged_paths[i.q.qid] for i in ans for p in runs.get(i.q.qid, [])[:10])
                / tot
            )
    return out


class _SupportsRecord(Protocol):
    def record(self) -> dict[str, Any]: ...


def item_records(items: Sequence[_SupportsRecord]) -> list[dict[str, Any]]:
    return [i.record() for i in items]
