"""`bench gold check`: the no-answer checks of benchmarks-v2 §3.1 and advisory checks on drafts.

- **rg check** (no-answer drafts): no single chunk of the vault contains all of the draft's key
  terms. This is `rg -il -e t1 … ` per chunk with AND semantics, case-insensitive, substring match,
  done in Python so it is exact and testable. A hit means the answer may exist: the draft is
  flagged and `review` does not show it unless asked.
- **BM25-top-20 check** (no-answer drafts): `review` shows the BM25 top 20 for the question and the
  person confirms none answers it (the human half; recorded in review.jsonl).
- Answerable drafts: the gold ids must exist; the lexical overlap with the gold chunk is shown so a
  copied question is visible (the median control is enforced at freeze).
- **Pre-filter** (spec change 2026-10-24: Claude drafts, then filters before the human sees anything).
  A draft is dropped (flagged, so `review` never shows it) if it is *weak* (fewer than
  MIN_QUESTION_TOKENS content tokens in the question; an answerable draft with an empty short
  answer) or a *lexical-overlap failure*: the median over its rel-2 gold chunks of
  |T(question) ∩ T(chunk)| / |T(question)| is above pg.MAX_MEDIAN_OVERLAP (0.5). The dropped drafts
  and the reason are in checks.jsonl (`prefilter`), none is silently lost.
"""

from __future__ import annotations

import statistics
from typing import Any

from bench.gold import pg
from bench.gold.chunks import ChunkIndex

MIN_TERMS = 2
MIN_QUESTION_TOKENS = 3


def rg_hits(index: ChunkIndex, key_terms: list[str]) -> list[str]:
    """Gold ids of the chunks that contain every key term (case-insensitive substring)."""
    terms = [t.lower() for t in key_terms if t.strip()]
    if len(terms) < MIN_TERMS:
        raise ValueError(f"a no-answer draft needs at least {MIN_TERMS} key terms, got {terms}")
    return [c.gold_id for c in index.chunks if all(t in c.text.lower() for t in terms)]


def check_draft(draft: dict[str, Any], index: ChunkIndex) -> dict[str, Any]:
    """{"ok": bool, "problems": [...], "rg_hits": [...], "overlap": float | None}."""
    problems: list[str] = []
    out: dict[str, Any] = {"draft_id": draft["draft_id"], "rg_hits": [], "overlap": None}
    if draft["category"] in pg.NO_ANSWER:
        try:
            out["rg_hits"] = rg_hits(index, draft.get("key_terms", []))
        except ValueError as e:
            problems.append(str(e))
        if out["rg_hits"]:
            problems.append(f"{len(out['rg_hits'])} chunk(s) contain all key terms")
        if draft.get("gold"):
            problems.append("a no-answer draft has gold")
    else:
        missing = [g for g in draft.get("gold", {}) if g not in index.by_gold]
        if missing:
            problems.append(f"gold ids not in the vault: {missing}")
        if not any(r == 2 for r in draft.get("gold", {}).values()):
            problems.append("no rel-2 gold chunk")
        texts = [index.by_gold[g].text for g, r in draft.get("gold", {}).items()
                 if r == 2 and g in index.by_gold]  # fmt: skip
        if texts:
            out["overlap"] = max(pg.overlap(draft["question"], t) for t in texts)
    pre: list[str] = []
    if len(pg.content_tokens(draft.get("question", ""))) < MIN_QUESTION_TOKENS:
        pre.append(f"weak: fewer than {MIN_QUESTION_TOKENS} content tokens in the question")
    if draft["category"] not in pg.NO_ANSWER and not str(draft.get("answer", "")).strip():
        pre.append("weak: empty short answer")
    rel2 = [index.by_gold[g].text for g, r in draft.get("gold", {}).items()
            if r == 2 and g in index.by_gold]  # fmt: skip
    if draft["category"] not in pg.NO_ANSWER and rel2:
        med = statistics.median(pg.overlap(draft.get("question", ""), t) for t in rel2)
        out["overlap_median"] = med
        if med > pg.MAX_MEDIAN_OVERLAP:
            pre.append(
                f"lexical overlap: median {med:.2f} > {pg.MAX_MEDIAN_OVERLAP} with the gold chunk"
            )
    out["prefilter"] = pre
    problems += [f"pre-filter: {x}" for x in pre]
    out["problems"] = problems
    out["ok"] = not problems
    return out
