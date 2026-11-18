"""`bench gold judge`: Claude grades every pooled (question, chunk) pair 0 / 1 / 2 (§3.1).

Human-run (T2): it sends chunk text of the PG snapshot (personal/ and projects/ only, no work
notes) to the Anthropic API, needs `ANTHROPIC_API_KEY`, `uv run --extra llm` and a hard
`--budget-usd` (at most MAX_BUDGET_USD). Tests use a scripted fake with the same `LLM.create`
shape; nothing here touches the network.

One call per pair, so no grade anchors on another pair of the same question. The scale is the one
the human used to have: 0 not relevant, 1 supports the answer, 2 answers it. The grades are appended to
labels.jsonl with `judge: claude`; a pair the human graded is never overwritten (their label wins), and a
re-run only asks for pairs with no grade. A reply that does not parse is counted and left
ungraded, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bench import llm
from bench.gold import pg
from bench.gold.chunks import ChunkIndex
from bench.gold.session import Out, Work, record_label

MODEL = "claude-opus-5-5"
MAX_TOKENS = 400  # effort-low thinking included; the reply itself is one short JSON object
MAX_BUDGET_USD = 25.0
CHUNK_CHARS = 3000

SYSTEM = (
    "You grade whether a passage from a person's private notes answers a question, for a "
    "retrieval benchmark. Grades: 2 = the passage answers the question (the short answer, if "
    "given, can be read off it); 1 = the passage supports the answer or states part of it but "
    "does not answer it alone; 0 = not relevant, or on the topic but without the answer. Judge "
    "only the passage shown. A question with no expected answer is answered by no passage unless "
    "the passage really does state the answer: grade it 0 otherwise. "
    'Reply with one JSON object and nothing else: {"rel": 0|1|2}.'
)


def pair_prompt(q: pg.Question, chunk_text: str) -> str:
    text = chunk_text if len(chunk_text) <= CHUNK_CHARS else chunk_text[:CHUNK_CHARS] + " […]"
    answer = q.answer if q.answer else "(no answer expected: the notes may not cover this)"
    return f"Question: {q.question}\nShort answer: {answer}\n\nPassage:\n{text}"


def parse_rel(text: str) -> int | None:
    m = re.search(r'"rel"\s*:\s*([012])\b', text)
    return int(m.group(1)) if m else None


@dataclass
class JudgeReport:
    judged: int = 0
    unparsable: int = 0
    gone: int = 0  # pairs whose chunk is no longer in the import (not asked)
    voided: int = 0
    remaining: int = 0
    stopped: str = "done"  # done | budget
    cost: dict[str, Any] = field(default_factory=dict)


def todo_pairs(work: Work) -> list[dict[str, Any]]:
    """Pooled pairs of accepted questions with no grade from anyone yet."""
    live = {q.qid for q in work.accepted()}
    state = work.label_state()
    return [p for p in work.pool() if p["qid"] in live and (p["qid"], p["gold_id"]) not in state]


def judge_all(
    work: Work, index: ChunkIndex, client: llm.LLM, ledger: llm.Ledger, *, out: Out = print,
) -> JudgeReport:  # fmt: skip
    questions = {q.qid: q for q in work.accepted()}
    rep = JudgeReport()
    voided: set[str] = set()
    for p in todo_pairs(work):
        qid, gid = p["qid"], p["gold_id"]
        if qid in voided:
            continue
        chunk = index.by_gold.get(gid)
        if chunk is None:
            rep.gone += 1
            continue
        prompt = pair_prompt(questions[qid], chunk.text)
        try:
            ledger.check(MAX_TOKENS, len(SYSTEM + prompt) // 3)
        except llm.BudgetExceeded as e:
            out(f"STOPPED by the budget cap: {e}")
            rep.stopped = "budget"
            break
        reply = client.create(system=SYSTEM, messages=[{"role": "user", "content": prompt}],
                              tools=[], max_tokens=MAX_TOKENS)  # fmt: skip
        ledger.add(reply.usage, f"judge:{qid}:{gid}")
        text = "".join(b.get("text", "") for b in reply.content if b.get("type") == "text")
        rel = parse_rel(text)
        if rel is None:
            rep.unparsable += 1
            continue
        rep.judged += 1
        if record_label(work, questions[qid], gid, rel, judge="claude", source="pool", out=out):
            voided.add(qid)
            rep.voided += 1
    rep.remaining = len(todo_pairs(work))
    rep.cost = ledger.summary()
    return rep
