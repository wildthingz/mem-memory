"""LongMemEval_S retrieval evaluation core (benchmarks-v2 B-R1, B-R2 (i), B-C2).

`run_lme` drives a backend over a split and returns per-item records; `score` turns the records
into metrics with CIs. Nothing here touches production code: the backend (bench/adapters/lme_ours.py
for the real pipeline, a fake in tests) answers `retrieve(question, haystack)` with the three lists
of benchmarks-v2 §0.3 as chunk refs `<session_id>#<round>`.

Unit of analysis (decision record, bench/datasets/README.md):
- Ranking metrics (Recall_any@5, Recall_all@10, nDCG@10 std, ndcg_any@10 upstream, MRR) are
  session-level: a chunk ranking becomes a session ranking by best chunk with sessions
  deduplicated (B-R1 step 3); gold = answer_session_ids.
- Delivered-context metrics (coverage, precision, IRA, FAR, F0.5) are chunk-level on S_q: gold =
  rounds with a `has_answer` turn. Negatives (B-R2 (i)) are in the same pool with empty gold.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from bench.datasets import lme
from bench.metrics import delivered as dv
from bench.metrics import ir, stats
from bench.metrics.tokens import est_tokens
from mem import listmode

LEGS = ("bm25", "dense", "rrf")
KINDS = ("answerable", "negative", "abs")
TOP8 = 8


@dataclass
class Retrieved:
    """One `recall`: F_q (top 100), R_q (top 50), S_q as chunk refs, plus what the gate saw."""

    F: list[str]
    R: list[str]
    S: list[str]
    text: str = ""  # the serialised tool result, as the client sees it
    latency_ms: float = 0.0
    p_scores: list[float] = field(default_factory=list)  # calibrated p of the pool, descending
    legs: dict[str, list[str]] = field(default_factory=dict)  # bm25 / dense / rrf (no reranker)
    bodies: list[str] = field(default_factory=list)  # full text of each delivered row (expand only)
    truncated: list[str] = field(default_factory=list)  # delivered rows `get` cut short (expand)
    ingest_ms: float = 0.0  # writing the haystack's sessions into the fresh store (ingest_session)
    paged: list[str] = field(
        default_factory=list
    )  # rows of pages 2.. of a paged answer (ADR 0024), when followed


class LmeBackend(Protocol):
    def retrieve(
        self, question: dict[str, Any], haystack: lme.Haystack, *, want_legs: bool
    ) -> Retrieved: ...


@dataclass
class Item:
    qid: str
    category: str
    kind: str  # answerable | negative | abs
    gold_sessions: frozenset[str]
    gold_rounds: frozenset[str]
    r: Retrieved
    top8_text: str = ""  # full text of R[:8] (for the B-C2 "vs top-8 full chunks" ratio)

    @property
    def abstained(self) -> bool:
        return not self.r.S

    def record(self) -> dict[str, Any]:
        return {"qid": self.qid, "category": self.category, "kind": self.kind,
                "F": self.r.F[:100], "R": self.r.R[:50], "S": self.r.S,
                "abstained": self.abstained, "gold": sorted(self.gold_rounds),
                "gold_sessions": sorted(self.gold_sessions), "latency_ms": self.r.latency_ms,
                "p_scores": self.r.p_scores[:5], "legs": {k: v[:50] for k, v in self.r.legs.items()},
                "response_chars": len(self.r.text)}  # fmt: skip


def run_lme(
    backend: LmeBackend,
    questions: Mapping[str, dict[str, Any]],
    manifest: Sequence[lme.ManifestRow],
    split: str,
    *,
    drop_negatives: frozenset[str] = frozenset(),
    include_abs: bool = True,
    want_legs: bool = True,
    on_item: Callable[[Item], None] | None = None,
) -> list[Item]:
    """Answerable (full haystack) + negative (evidence-removed) for the B-R1 set; ABS optionally."""
    items = []

    def one(m: lme.ManifestRow, kind: str) -> None:
        q = questions[m.question_id]
        gold_s = lme.gold_sessions(q) if kind == "answerable" else frozenset()
        gold_r = lme.gold_rounds(q) if kind == "answerable" else frozenset()
        hay = lme.haystack_of(q, exclude=lme.gold_sessions(q) if kind == "negative" else ())
        r = backend.retrieve(q, hay, want_legs=want_legs and kind == "answerable")
        text_of = {lme.ref(sid, rd.index): rd.user + "\n" + rd.assistant
                   for sid, _, rounds in hay.sessions for rd in rounds}  # fmt: skip
        it = Item(m.question_id, m.question_type, kind, frozenset(gold_s), frozenset(gold_r), r,
                  "\n".join(text_of.get(x, "") for x in r.R[:TOP8]))  # fmt: skip
        items.append(it)
        if on_item:
            on_item(it)

    br1 = lme.select(manifest, split, "br1")
    for m in br1:
        one(m, "answerable")
    for m in br1:
        if m.question_id not in drop_negatives:
            one(m, "negative")
    if include_abs:
        for m in lme.select(manifest, split, "abs"):
            one(m, "abs")
    return items


# ---- scoring ---------------------------------------------------------------------------------


@dataclass
class Scores:
    metrics: dict[str, float] = field(default_factory=dict)
    ci95: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    vectors: dict[str, list[float]] = field(
        default_factory=dict
    )  # per answerable item, for pairing


def _sess(refs: Sequence[str]) -> list[str]:
    return lme.dedupe_sessions(refs)


def ranking_vectors(items: Sequence[Item]) -> dict[str, list[float]]:
    """Per-answerable-item ranking metrics on R_q and F_q (session level)."""
    v: dict[str, list[float]] = defaultdict(list)
    for it in items:
        g = set(it.gold_sessions)
        rs, fs = _sess(it.r.R), _sess(it.r.F)
        v["recall_any5_R"].append(ir.recall_any(rs, g, 5))
        v["recall_any5_F"].append(ir.recall_any(fs, g, 5))
        v["recall_all10_R"].append(ir.recall_all(rs, g, 10))
        v["recall_all10_F"].append(ir.recall_all(fs, g, 10))
        v["ndcg_std10_R"].append(ir.ndcg_std(rs, dict.fromkeys(g, 1), 10))
        v["ndcg_any10_upstream_R"].append(ir.ndcg_any_upstream(rs, g, 10))
        v["mrr_R"].append(ir.mrr(rs, g))
        for leg, refs in it.r.legs.items():
            v[f"ndcg_std10_{leg}"].append(ir.ndcg_std(_sess(refs), dict.fromkeys(g, 1), 10))
    return dict(v)


def _delivered(items: Sequence[Item], rows: Callable[[Item], list[str]] = lambda i: i.r.S):
    return [dv.Delivered(rows(i), i.gold_rounds, i.kind == "answerable")
            for i in items if i.kind != "abs"]  # fmt: skip


def score(
    items: Sequence[Item], *, b: int = stats.B_DEFAULT, seed: int = stats.SEED,
    n_target: int | None = None,
) -> Scores:  # fmt: skip
    """All B-R1 / B-R2 (i) metrics. `n_target` = items of the test set, for the power estimate."""
    out = Scores()
    ans = [i for i in items if i.kind == "answerable"]
    neg = [i for i in items if i.kind == "negative"]
    absq = [i for i in items if i.kind == "abs"]
    out.n = {"answerable": len(ans), "negative": len(neg), "abs": len(absq)}
    if not ans:
        out.notes.append("no answerable items")
        return out
    strata = [i.category for i in ans]
    vec = ranking_vectors(ans)
    out.vectors = vec
    for name, vals in vec.items():
        out.metrics[name], *lohi = stats.bootstrap_mean(vals, strata, b, seed)
        out.ci95[name] = (lohi[0], lohi[1])
    out.metrics["n_answerable"] = float(len(ans))

    # delivered (S_q): answerable + negatives in one pool; ratio metrics bootstrap over queries.
    pool = [i for i in items if i.kind != "abs"]
    dl = _delivered(pool)
    pstrata = [i.category for i in pool]
    out.metrics["coverage_any"] = dv.coverage_any(dl)
    out.metrics["coverage_all"] = dv.coverage_all(dl)
    out.metrics["delivered_precision"] = dv.delivered_precision(dl)
    out.metrics["ira"] = dv.ira(dl)
    out.metrics["far"] = dv.far(dl)
    out.metrics["abstention_recall"] = dv.abstention_recall(dl)
    out.metrics["delivered_f05"] = dv.f05(out.metrics["delivered_precision"],
                                          out.metrics["coverage_any"])  # fmt: skip
    a = lambda i: float(i.kind == "answerable")
    out.ci95["coverage_any"] = stats.bootstrap_ratio(
        [float(bool(set(i.r.S) & i.gold_rounds)) for i in pool],
        [a(i) for i in pool],
        pstrata,
        b,
        seed,
    )[1:]
    out.ci95["coverage_all"] = stats.bootstrap_ratio(
        [float(i.kind == "answerable" and i.gold_rounds <= set(i.r.S)) for i in pool],
        [a(i) for i in pool], pstrata, b, seed)[1:]  # fmt: skip
    out.ci95["delivered_precision"] = stats.bootstrap_ratio(
        [len(set(i.r.S) & i.gold_rounds) for i in pool],
        [len(i.r.S) for i in pool],
        pstrata,
        b,
        seed,
    )[1:]
    out.ci95["ira"] = stats.bootstrap_ratio(
        [len(set(i.r.S) - i.gold_rounds) for i in pool], [float(bool(i.r.S)) for i in pool],
        pstrata, b, seed)[1:]  # fmt: skip
    out.ci95["far"] = stats.bootstrap_ratio(
        [float(i.kind == "answerable" and not i.r.S) for i in pool], [a(i) for i in pool],
        pstrata, b, seed)[1:]  # fmt: skip
    out.ci95["abstention_recall"] = stats.bootstrap_ratio(
        [float(i.kind == "negative" and not i.r.S) for i in pool],
        [float(i.kind == "negative") for i in pool], pstrata, b, seed)[1:]  # fmt: skip

    # baseline (c): RRF with no reranker returns its top 8, no gate.
    if all("rrf" in i.r.legs for i in ans):
        rrf8 = _delivered(pool, lambda i: i.r.legs.get("rrf", [])[:TOP8] if i.kind == "answerable"
                          else [])  # fmt: skip
        out.metrics["rrf8_coverage_any"] = dv.coverage_any(rrf8)
        out.metrics["rrf8_delivered_precision_answerable_only"] = dv.delivered_precision(
            [q for q in rrf8 if q.answerable])  # fmt: skip

    # superiority of R_q nDCG@10 (std) over the better single leg, paired on the same items.
    legs = [leg for leg in ("bm25", "dense") if f"ndcg_std10_{leg}" in vec]
    if len(legs) == 2:
        best = max(legs, key=lambda leg: sum(vec[f"ndcg_std10_{leg}"]))
        p = stats.paired_bootstrap(vec["ndcg_std10_R"], vec[f"ndcg_std10_{best}"], strata, b, seed)
        out.metrics |= {"ndcg_best_leg_is_bm25": float(best == "bm25"),
                        "ndcg_delta_vs_best_leg": p.delta, "ndcg_delta_ci_lo": p.lo,
                        "ndcg_delta_ci_hi": p.hi, "ndcg_delta_sd": p.sd,
                        "ndcg_mdd": stats.mdd_continuous(p.sd, p.n)}  # fmt: skip
        out.metrics["ndcg_power"] = stats.superiority_power(p.delta, p.sd, n_target or p.n)
        out.notes.append(f"best single leg by mean nDCG@10 (std): {best}")

    # B-R2 AUROC: p1 and p1 - p2 as "answer exists" scores, answerable vs negative.
    if ans and neg and all(i.r.p_scores for i in ans + neg):
        p1 = lambda i: i.r.p_scores[0]
        gap = lambda i: i.r.p_scores[0] - (i.r.p_scores[1] if len(i.r.p_scores) > 1 else 0.0)
        out.metrics["auroc_p1"] = stats.auroc([p1(i) for i in ans], [p1(i) for i in neg])
        out.metrics["auroc_p1_gap"] = stats.auroc([gap(i) for i in ans], [gap(i) for i in neg])
    else:
        out.metrics["auroc_p1"] = out.metrics["auroc_p1_gap"] = math.nan

    # ABS: descriptive only (CI half-width about +-0.16 at n = 20).
    if absq:
        out.metrics["abs_abstention_recall"] = sum(not i.r.S for i in absq) / len(absq)
        out.metrics["abs_n"] = float(len(absq))

    # macro beside micro, and per-category values for the headline metrics.
    for name in ("recall_any5_R", "ndcg_std10_R"):
        cats = defaultdict(list)
        for c, x in zip(strata, vec[name], strict=True):
            cats[c].append(x)
        out.per_category[name] = {c: sum(v) / len(v) for c, v in sorted(cats.items())}
        out.metrics[f"{name}_macro"] = sum(out.per_category[name].values()) / len(cats)
    # delivered metrics per question type (answerable + its negatives, as the headline), and the
    # official `_abs` questions as a category of their own (abstention recall; the rest is nan).
    by_cat: dict[str, list[dv.Delivered]] = defaultdict(list)
    for i in pool:
        by_cat[i.category].append(_delivered([i])[0])
    by_cat["_abs"] = [dv.Delivered(i.r.S, frozenset(), False) for i in absq]
    fns = {"coverage_any": dv.coverage_any, "delivered_precision": dv.delivered_precision,
           "ira": dv.ira, "far": dv.far, "abstention_recall": dv.abstention_recall}  # fmt: skip
    for name, fn in fns.items():
        out.per_category[name] = {c: fn(v) if v else math.nan for c, v in sorted(by_cat.items())}
    out.per_category["n"] = {c: float(len(v)) for c, v in sorted(by_cat.items())}
    lat = [i.r.latency_ms for i in items if i.r.latency_ms]
    if lat:
        out.metrics["latency_ms_p95"] = stats.percentile_nearest_rank(lat, 0.95)
    return out


# ---- tokens (B-C2) ---------------------------------------------------------------------------

TokenCounter = Callable[[Sequence[str]], list[int]]
SAMPLE_N = 300


def est_counter(texts: Sequence[str]) -> list[int]:
    return [est_tokens(t) for t in texts]


def token_sample(items: Sequence[Item], n: int = SAMPLE_N) -> list[Item]:
    """A fixed sample: items ordered by sha1(kind:qid), first n (B-R1 answerables and negatives).
    Normal-mode responses only: list-mode ones (ADR 0023) have their own cap and rows
    (`list_token_metrics`), so "B-C2 normal mode unchanged" reads off this sample with the mode on.
    With the mode off no response is a list response and the sample is as before."""
    pool = [i for i in items if i.kind != "abs" and not is_list_response(i.r.text)]
    key = lambda i: hashlib.sha1(f"{i.kind}:{i.qid}".encode()).hexdigest()
    return sorted(pool, key=key)[:n]


def is_list_response(text: str) -> bool:
    """A list-mode response (ADR 0023): the instruction line sits right under the header."""
    lines = text.splitlines()
    return len(lines) > 1 and lines[1] == listmode.INSTRUCTION


def list_token_metrics(items: Sequence[Item], counter: TokenCounter) -> dict[str, float]:
    """B-C2 list-mode rows: the tokens of every list-mode response in the run (not the 300 sample:
    they are few), against the mode's own cap. All zero when `read.list_mode` is off or no query
    was an aggregation (then `list_tokens_n` says so)."""
    texts = [i.r.text for i in items if i.kind != "abs" and is_list_response(i.r.text)]
    toks = counter(texts)
    return {
        "list_tokens_n": float(len(toks)),
        "list_tokens_mean": sum(toks) / len(toks) if toks else 0.0,
        "list_tokens_p95": float(stats.percentile_nearest_rank(toks, 0.95)) if toks else 0.0,
        "list_tokens_max": float(max(toks, default=0)),
        "list_cap": float(listmode.CAP_TOKENS),
    }


def token_metrics(items: Sequence[Item], counter: TokenCounter) -> dict[str, float]:
    sample = token_sample(items)
    toks = counter([i.r.text for i in sample])
    top8 = counter([i.top8_text for i in sample])
    empty = [t for t, i in zip(toks, sample, strict=True) if not i.r.S]
    full = sum(top8)
    return {
        "tokens_mean": sum(toks) / len(toks), "tokens_p95": stats.percentile_nearest_rank(toks, 0.95),
        "tokens_n": float(len(toks)), "empty_tokens_max": float(max(empty, default=0)),
        "n_empty": float(len(empty)),
        "reduction_vs_top8_full": 1 - sum(toks) / full if full else math.nan,
        "frac_responses_ge80pct_smaller": sum(t <= 0.2 * f for t, f in zip(toks, top8, strict=True) if f)
        / max(sum(1 for f in top8 if f), 1),
    }  # fmt: skip
