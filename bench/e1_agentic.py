"""B-E1 agentic arm (`bench e1-agentic`, MEM2-214, report §5 options E and H).

The answerer drives the mem read tools itself (`recall`, `get`; every argument the tool contract
declares passes through, so the `page` parameter of option B reaches the store once it lands) with the
skill as its system prompt, then a judge labels the final answer with the same per-type prompts as
`bench e1-run`. The answerer's settings are prompt-v2's (`AnthropicLLM(model, None)`: thinking
`between_tools`, which on claude-sonnet-5-5 is thinking off, no extended thinking; no effort, no
sampling parameters). Paging is the answerer's move: the loop only hands
tool results back, so whether the skill makes it page on cut rows is what the run measures.

The record keeps every tool call, the prompt sha and the skill sha, so a skill change is visible in the
run file; `compare` pairs a run with a baseline run file by question and `adoption` applies the
adoption rule (overall not below the baseline, no question type down more than 5 points).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from bench import e1
from bench.llm import LLM, BudgetExceeded
from bench.metrics import stats

CONDITION = "agentic"
MAX_TURNS = 12
MAX_TYPE_DROP = 0.05  # adoption: no question type down more than this
TOOLS = ("recall", "get")

AGENT_ANSWER = (
    "I will give you a question about the user's past chats with you. Your memory of those chats is "
    "behind your memory tools. Please answer the question based on what they return. If the memory "
    "does not contain the answer, say that you don't know.\n\n"
    "Current Date: {date}\nQuestion: {q}\nAnswer:"
)


class ToolBox(Protocol):
    schemas: list[dict[str, Any]]  # Anthropic tool definitions

    def call(self, name: str, args: dict[str, Any]) -> str: ...
    def close(self) -> None: ...


def agent_prompt(question: str, date: str) -> str:
    return AGENT_ANSWER.format(date=date, q=question)


def prompts_sha256() -> str:
    blob = json.dumps(
        [e1.PROMPT_VERSION, e1.JUDGE_TEMPLATES, AGENT_ANSWER, MAX_TURNS], sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _chars(*objs: Any) -> int:
    return sum(len(json.dumps(o, ensure_ascii=False)) for o in objs)


def _ask(q: dict[str, Any], box: ToolBox, answerer: LLM, system: str, spend: e1.Spend,
         max_tokens: int, max_turns: int) -> dict[str, Any]:  # fmt: skip
    """One question's tool dialogue. Raises BudgetExceeded before a call that could cross the cap."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": agent_prompt(q["question"], q["question_date"])}
    ]
    calls: list[dict[str, Any]] = []
    answer, stop, turns = "", "max_turns", 0
    for turns in range(1, max_turns + 1):
        spend.check("answer", max_tokens, len(system) + _chars(messages, box.schemas))
        reply = answerer.create(system=system, messages=messages, tools=box.schemas,
                                max_tokens=max_tokens)  # fmt: skip
        spend.ledgers["answer"].add(reply.usage, f"{q['question_id']}/t{turns}")
        uses = [b for b in reply.content if b.get("type") == "tool_use"]
        if not uses:
            answer, stop = e1._text(reply), "end_turn"
            break
        messages.append({"role": "assistant", "content": reply.content})
        results = []
        for b in uses:
            try:
                out, err = box.call(b["name"], dict(b["input"])), False
            except Exception as e:  # noqa: BLE001 - a tool error goes back to the answerer
                out, err = f"{type(e).__name__}: {e}", True
            calls.append({"turn": turns, "tool": b["name"], "input": b["input"], "text": out,
                          "error": err})  # fmt: skip
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out,
                            **({"is_error": True} if err else {})})  # fmt: skip
        messages.append({"role": "user", "content": results})
    return {"answer": answer, "stop": stop, "turns": turns, "calls": calls}


def run_agentic(
    questions: Sequence[dict[str, Any]],
    open_tools: Callable[[dict[str, Any]], ToolBox],
    answerer: LLM,
    judge: LLM,
    *,
    skill_text: str,
    budget_usd: float,
    answer_max_tokens: int = e1.ANSWER_MAX_TOKENS,
    judge_max_tokens: int = e1.JUDGE_MAX_TOKENS,
    max_turns: int = MAX_TURNS,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Question by question. Stops at the budget; a question cut off by it is not scored."""
    spend = e1.Spend(answerer, judge, budget_usd)
    items: list[dict[str, Any]] = []
    stopped = None
    try:
        for q in questions:
            kind, before = e1.judge_kind(q), spend.total
            box = open_tools(q)
            try:
                got = _ask(q, box, answerer, skill_text, spend, answer_max_tokens, max_turns)
            finally:
                box.close()
            jp = e1.judge_prompt(kind, q["question"], str(q["answer"]), got["answer"])
            spend.check("judge", judge_max_tokens, len(jp))
            jr = judge.create(system="", tools=[], max_tokens=judge_max_tokens,
                              messages=[{"role": "user", "content": jp}])  # fmt: skip
            spend.ledgers["judge"].add(jr.usage, f"{q['question_id']}/judge")
            raw = e1._text(jr)
            item = {"qid": q["question_id"], "type": q["question_type"], "judge_kind": kind,
                    "condition": CONDITION, **got, "judge_raw": raw, **e1.verdict_fields(raw),
                    "usd": round(spend.total - before, 6)}  # fmt: skip
            items.append(item)
            if on_item:
                on_item(item)
    except BudgetExceeded as e:
        stopped = str(e)
    return {"items": items, "planned": len(questions),
            "complete": stopped is None and len(items) == len(questions), "stopped": stopped,
            "cost": spend.summary(), "conditions": [CONDITION], "answerer": answerer.model,
            "judge": judge.model, "prompt_version": e1.PROMPT_VERSION,
            "prompts_sha256": prompts_sha256(), "max_turns": max_turns,
            "judge_version": e1.JUDGE_VERSION,
            "judge_flagged": sum(i["judge_flagged"] for i in items),
            "skill_sha256": hashlib.sha256(skill_text.encode()).hexdigest()}  # fmt: skip


# ---- scoring (offline) -----------------------------------------------------------------------


def score(run: dict[str, Any], b: int = 2000) -> tuple[dict[str, float], dict[str, Any]]:
    """`accuracy` (micro, CI resampling within type) and the per-type table, from the run file."""
    _, extra = e1.score_run(run, b)
    per = extra["per_condition"][CONDITION]
    return {"accuracy": per["accuracy"], "n": float(per["n"]), "macro": per["macro"]}, per


def _verdicts(run: dict[str, Any], cond: str) -> dict[str, tuple[float, str]]:
    return {i["qid"]: (float(e1.parse_verdict(i["judge_raw"])), e1._bucket(i))
            for i in run["items"] if i["condition"] == cond}  # fmt: skip


def baseline_condition(baseline: dict[str, Any]) -> str:
    """The arm to pair against: a skill-against-skill baseline is an agentic run, an `e1-run` file is `mem`."""
    conds = baseline.get("conditions", [])
    return CONDITION if CONDITION in conds and "mem" not in conds else "mem"


def compare(new: dict[str, Any], baseline: dict[str, Any], *, base_condition: str = "mem",
            new_condition: str = CONDITION, b: int = 2000) -> dict[str, Any]:  # fmt: skip
    """Both runs on the questions they share: accuracy of each, delta with a paired bootstrap CI
    (resampling questions within their type), overall and per type, and `questions` (right in the
    new run minus right in the baseline, a count). Runs on different splits are refused; whether
    the new run finished under its budget is carried for `adoption`. `new_condition="mem"` pairs
    two `e1-run` files (Round 10: an arm against the base arm)."""
    ns, bs = new.get("split"), baseline.get("split")
    if ns is not None and bs is not None and ns != bs:
        raise ValueError(f"the baseline is on split {bs!r}, the run on {ns!r}: not comparable")
    n, o = _verdicts(new, new_condition), _verdicts(baseline, base_condition)
    shared = sorted(set(n) & set(o))
    diffs = [n[q][0] - o[q][0] for q in shared]
    strata = [n[q][1] for q in shared]

    def block(ids: list[str]) -> dict[str, Any]:
        d = [n[q][0] - o[q][0] for q in ids]
        st = [n[q][1] for q in ids]
        _, lo, hi = stats.bootstrap_mean(d, st, b) if d else (0.0, float("nan"), float("nan"))
        return {"n": len(ids), "new": sum(n[q][0] for q in ids) / len(ids) if ids else float("nan"),
                "baseline": sum(o[q][0] for q in ids) / len(ids) if ids else float("nan"),
                "delta": sum(d) / len(d) if d else float("nan"), "delta_ci95": [lo, hi],
                "questions": round(sum(d))}  # fmt: skip

    by: dict[str, list[str]] = defaultdict(list)
    for q, s in zip(shared, strata, strict=True):
        by[s].append(q)
    return {"n_paired": len(shared), "split": ns, "new_complete": new.get("complete", True),
            "overall": block(shared),
            "per_type": {t: block(ids) for t, ids in sorted(by.items())},
            "new_only": sorted(set(n) - set(o)), "baseline_only": sorted(set(o) - set(n)),
            "discordant": {"new_only_right": sum(d > 0 for d in diffs),
                           "baseline_only_right": sum(d < 0 for d in diffs)}}  # fmt: skip


def adoption(cmp: dict[str, Any], max_drop: float = MAX_TYPE_DROP) -> dict[str, Any]:
    """The adoption rule for a skill on the agentic arm. (B-W5 unchanged is checked on its own run.)"""
    reasons = []
    if not cmp.get("new_complete", True):
        reasons.append("the run did not finish (cut by the budget cap): not scored for adoption")
    if cmp.get("n_paired") == 0 or math.isnan(cmp["overall"]["delta"]):
        reasons.append("no question is paired with the baseline")
    elif cmp["overall"]["delta"] < 0:
        reasons.append(f"overall {cmp['overall']['delta']:+.3f} is below the baseline")
    for t, row in cmp["per_type"].items():
        if row["delta"] < -max_drop - 1e-9:
            reasons.append(f"{t} is down {-row['delta']:.3f} (limit {max_drop})")
    return {"adopt": not reasons, "reasons": reasons}
