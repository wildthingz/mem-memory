"""B-LoCoMo: LoCoMo end-to-end QA through the memory pipeline (`bench locomo-run`; T2, MEM2-196).

Per conversation: a fresh store, every session written with `ingest_session` (its own date), then per
question the production `recall` text goes to an answerer model, whose answer is scored two ways:

- **Official LoCoMo scoring** (snap-research/locomo `task_eval/evaluation.py`, `eval_question_answering`),
  all five categories: token F1 with Porter stemming (categories 2, 3 and 4; category 3's gold cut at
  the first ';'), multi-answer F1 split on commas (category 1), and for the adversarial category 5 a hit
  iff the answer contains "no information available" or "not mentioned" (the paper treats these
  questions as unanswerable: a system must not give the distractor answer).
- **LLM judge** (categories 1-4): the CORRECT/WRONG prompt of Mem0 (Chhikara et al., 2025,
  arXiv 2504.19413, appendix) which the memory papers that report LoCoMo "J" scores (Mem0, Zep, ...)
  share. Those papers leave out category 5, and so does the judge here; category 5 is the rule above.
  `judge_accuracy` is the headline number, `f1` is next to it.

`score_run` recomputes every metric from the run file (F1 from the stored answer and gold, the judge
label re-parsed from its reply): scoring fixes never repeat the paid part. A hard `budget_usd` stops the
run BEFORE a call whose worst case would cross it (`bench.e1.Spend`); the file then says
`complete: false`. Nothing here is tuned on LoCoMo: the prompts below are fixed, one version.

The adversarial lure (`adversarial_answer`) is never read, so it cannot reach a prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, Protocol

from bench.datasets.locomo import ADVERSARIAL, CATEGORIES, Conversation
from bench.e1 import Spend, _text
from bench.llm import LLM, BudgetExceeded
from bench.metrics import stats

PROMPT_VERSION = 2
# thinking `between_tools` is claude-sonnet-5-5's thinking-off setting (no extended thinking; `disabled` 400s)
ANSWER_MAX_TOKENS, JUDGE_MAX_TOKENS = 1024, 200
# The LoCoMo paper's own answer wording (Maharana et al., ACL 2024, arXiv 2402.17753; code:
# snap-research/locomo task_eval/gpt_utils.py `QA_PROMPT`, `QA_PROMPT_CAT_5`, `get_input_context`):
# the context, a blank line, then the prompt. Category 2 (temporal) gets the repo's date suffix.
# Not the LME answer template, on purpose: published LoCoMo numbers use this prompt (MEM2-222 F16 ruling).
# Adversarial (cat 5) is not scored here, so only its prompt text is kept for the record.
ANSWER_PROMPT = (
    "{context}\n\nBased on the above context, write an answer in the form of a short phrase for "
    "the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {question} Short answer:"
)
TEMPORAL = 2
TEMPORAL_SUFFIX = " Use DATE of CONVERSATION to answer with an approximate date."
# The Mem0 paper's template (arXiv 2504.19413, appendix "Prompt Template for LLM as a Judge").
JUDGE_TEMPLATE = (
    'Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the '
    "following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground "
    "truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.\n\n"
    "The point of the question is to ask about something one user should know about the other user "
    "based on their prior conversations. The gold answer will usually be a concise and short answer "
    "that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as "
    "long as it touches on the same topic as the gold answer, it should be counted as CORRECT.\n\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The "
    "generated answer might be much longer or use relative time references (like 'last Tuesday' or "
    "'next month'), but you should be generous with your grading - as long as it refers to the same "
    "date or time period as the gold answer, it should be counted as CORRECT. Even if the format "
    "differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.\n\n"
    "Now it's time for the real question:\n"
    "Question: {question}\nGold answer: {gold_answer}\nGenerated answer: {generated_answer}\n\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT "
    "or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the "
    "evaluation script.\nJust return the label CORRECT or WRONG in a json format with the key as "
    '"label".'
)


def prompts_sha256() -> str:
    blob = json.dumps([PROMPT_VERSION, ANSWER_PROMPT, TEMPORAL_SUFFIX, JUDGE_TEMPLATE])
    return hashlib.sha256(blob.encode()).hexdigest()


# ---- official scoring (evaluation.py, ported) --------------------------------------------------


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


@cache
def _stemmer():
    from nltk.stem import PorterStemmer  # the stemmer evaluation.py uses

    return PorterStemmer()


def f1_score(prediction: str, ground_truth: str) -> float:
    ps = _stemmer()
    p = [ps.stem(w) for w in normalize_answer(prediction).split()]
    g = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def multi_f1(prediction: str, ground_truth: str) -> float:
    """Category 1: every gold sub-answer takes its best-matching predicted sub-answer; mean."""
    preds = [p.strip() for p in prediction.split(",")]
    golds = [g.strip() for g in ground_truth.split(",")]
    return sum(max(f1_score(p, g) for p in preds) for g in golds) / len(golds)


def adversarial_hit(prediction: str) -> bool:
    low = prediction.lower()
    return "no information available" in low or "not mentioned" in low


def score_item(category: int, prediction: str, gold: str | None) -> float:
    if category == ADVERSARIAL:
        return float(adversarial_hit(prediction))
    assert gold is not None
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, gold)
    if category == 1:
        return multi_f1(prediction, gold)
    raise ValueError(f"unknown LoCoMo category {category}")


def judge_label(text: str) -> bool | None:
    """The reply's label: the JSON `label`, else the one label word present (True CORRECT, False
    WRONG). Both words, or neither, is None: flagged (`judge_flagged` in `score_run`), not credited.
    This judge is Mem0's (a reasoning sentence, then a JSON label, 200 tokens), not B-E1's yes/no
    judge, so ADR 0029's cap and first-token rule do not apply here; its flagged count does."""
    m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper() == "CORRECT"
    words = {w.upper() for w in re.findall(r"\b(CORRECT|WRONG)\b", text, flags=re.IGNORECASE)}
    return words == {"CORRECT"} if len(words) == 1 else None


def parse_label(text: str) -> bool:
    """CORRECT iff the reply's label is CORRECT; a flagged reply is not credited."""
    return judge_label(text) is True


# ---- the run -----------------------------------------------------------------------------------


class Opened(Protocol):
    def recall(self, query: str) -> str: ...
    def close(self) -> None: ...


def run_locomo(
    conversations: Sequence[Conversation],
    open_conversation: Callable[[Conversation], Opened],
    answerer: LLM,
    judge: LLM,
    *,
    budget_usd: float,
    limit: int | None = None,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Conversation by conversation, question by question. Stops at the budget, keeping what finished."""
    spend = Spend(answerer, judge, budget_usd)
    planned = sum(len(c.qa) for c in conversations)
    if limit is not None:
        planned = min(planned, limit)
    items: list[dict[str, Any]] = []
    stopped = None
    try:
        for conv in conversations:
            if len(items) >= planned:
                break
            opened = open_conversation(conv)
            try:
                for q in conv.qa:
                    if len(items) >= planned:
                        break
                    ctx = opened.recall(q.question)
                    prompt = ANSWER_PROMPT.format(
                        context=ctx,
                        question=q.question + (TEMPORAL_SUFFIX if q.category == TEMPORAL else ""),
                    )
                    spend.check("answer", ANSWER_MAX_TOKENS, len(prompt))
                    before = spend.total
                    tag = f"{conv.sample_id}/{q.index}"
                    reply = answerer.create(system="", tools=[], max_tokens=ANSWER_MAX_TOKENS,
                                            messages=[{"role": "user", "content": prompt}])  # fmt: skip
                    spend.ledgers["answer"].add(reply.usage, tag)
                    response = _text(reply)
                    raw = None
                    if q.category != ADVERSARIAL:
                        jp = JUDGE_TEMPLATE.format(question=q.question, gold_answer=q.answer,
                                                   generated_answer=response)  # fmt: skip
                        spend.check("judge", JUDGE_MAX_TOKENS, len(jp))
                        jr = judge.create(system="", tools=[], max_tokens=JUDGE_MAX_TOKENS,
                                          messages=[{"role": "user", "content": jp}])  # fmt: skip
                        spend.ledgers["judge"].add(jr.usage, tag)
                        raw = _text(jr)
                    item = {"conv": conv.sample_id, "qi": q.index, "category": q.category,
                            "question": q.question, "gold": q.answer, "context_chars": len(ctx),
                            "answer": response, "judge_raw": raw,
                            "usd": round(spend.total - before, 6)}  # fmt: skip
                    items.append(item)
                    if on_item:
                        on_item(item)
            finally:
                opened.close()
    except BudgetExceeded as e:
        stopped = str(e)
    return {"items": items, "planned": planned, "complete": stopped is None and len(items) == planned,
            "stopped": stopped, "cost": spend.summary(), "answerer": answerer.model,
            "judge": judge.model, "prompt_version": PROMPT_VERSION,
            "prompts_sha256": prompts_sha256()}  # fmt: skip


# ---- scoring (offline) ---------------------------------------------------------------------------


def score_run(run: dict[str, Any], b: int = 2000) -> tuple[dict[str, float], dict[str, Any]]:
    nan = math.nan
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i in run["items"]:
        by_cat[i["category"]].append(i)
    per: dict[str, dict[str, Any]] = {}
    for cat in sorted(by_cat):
        its = by_cat[cat]
        scores = [score_item(cat, i["answer"], i["gold"]) for i in its]
        row: dict[str, Any] = {"n": len(its), "f1" if cat != ADVERSARIAL else "accuracy":
                               sum(scores) / len(scores)}  # fmt: skip
        if cat != ADVERSARIAL:
            ok = [float(parse_label(i["judge_raw"] or "")) for i in its]
            row["judge_accuracy"] = sum(ok) / len(ok)
        per[CATEGORIES[cat]] = row
    core = [i for i in run["items"] if i["category"] != ADVERSARIAL]
    ok = [float(parse_label(i["judge_raw"] or "")) for i in core]
    strata = [CATEGORIES[i["category"]] for i in core]
    acc, lo, hi = stats.bootstrap_mean(ok, strata, b) if core else (nan, nan, nan)
    f1s = [score_item(i["category"], i["answer"], i["gold"]) for i in core]
    adv = [score_item(ADVERSARIAL, i["answer"], None) for i in by_cat.get(ADVERSARIAL, [])]
    metrics = {
        "judge_accuracy": acc, "judge_accuracy_ci_lo": lo, "judge_accuracy_ci_hi": hi,
        "n_judged": float(len(core)),
        "f1": sum(f1s) / len(f1s) if f1s else nan,
        "adversarial_accuracy": sum(adv) / len(adv) if adv else nan, "n_adversarial": float(len(adv)),
        "judge_flagged": float(sum(judge_label(i["judge_raw"] or "") is None for i in core)),
        "run_complete": float(bool(run.get("complete"))), "cost_usd": float(run["cost"]["usd"]),
    }  # fmt: skip
    return metrics, {"per_category": per}
