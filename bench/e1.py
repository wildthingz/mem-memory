"""B-E1: LongMemEval_S end-to-end QA (benchmarks-v2 B-E1; T2, gated at P3 from MEM2-184).

`run_e1` (the paid part, `bench e1-run`) asks an answerer model each question under up to three
conditions and has a judge model label the answer with the type-specific LongMemEval prompt:

- mem     the answerer sees what `recall` returned for the question (the serialised tool result), or
          "no relevant memories" when the gate abstained. This is the arm the gate reads.
- none    the question only (no-memory baseline).
- oracle  the evidence sessions (`answer_session_ids`), whole (upper bound).

`score_run` recomputes every metric from the run file alone (judge replies are re-parsed), offline and
deterministic, so a scoring fix never repeats the paid part. A hard `budget_usd` stops the run BEFORE a
call whose worst case (its input uncached plus `max_tokens` of output) would cross it; the run file
then says `complete: false` and the gate row `run_complete` fails.

The judge prompts are the upstream LongMemEval `evaluate_qa.py` wording (one per question type, a
separate one for `_abs` questions), unchanged. The reply parsing differs from upstream (ADR 0029,
MEM2-269): upstream caps the judge at 10 tokens and credits `yes` anywhere in the reply; Sonnet is
sometimes verbose, and a correct item ("The response's final total is 3,000 ...") was cut before its
yes and scored wrong. Here the cap is `JUDGE_MAX_TOKENS` (64), the verdict is the FIRST word-bounded
yes/no token (`judge_verdict`), and a reply with neither is flagged (`judge_flagged` per item, a count
in the run record and the gate) and not credited, never silently scored. `JUDGE_VERSION` is a harness
field: runs made under the old cap never pool with runs made under this one. The same code judges
every arm (mem, none, oracle), `e1-agentic`, `--full` and LME_M (`e1_m` runs through `run_e1`).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from bench.llm import LLM, PRICES, BudgetExceeded, Ledger
from bench.metrics import stats

PROMPT_VERSION = 2
# The judge's reply handling (ADR 0029, MEM2-269). 1: upstream's rules (10 tokens, `yes` anywhere).
# 2: 64 tokens, the first yes/no token, a reply without one is flagged. A harness field: a run file
# without it was made under version 1.
JUDGE_VERSION = 2
ANSWERER_THINKING = "between_tools"  # what e1_qa.py ask() sends (MEM2-227 F5); a harness field
# Run validity (the owner, 2026-10-27, superseding the per-run 0.94 of 2026-10-26): the oracle arm is the same on
# every run of one harness, so its accuracy is pooled over those runs (answerer noise of +-2 questions in
# 125 must not void a run at random). VALID needs the pool >= ORACLE_POOL_MIN; a single run under
# ORACLE_RUN_FLOOR is a harness failure, INVALID whatever the pool says.
# The numbers: the owner, 2026-10-27, re-based on the strict judge (were 0.94 / 0.92). The 0.94 came from
# e1_qa.py's lenient judge; on the bench judge this answerer's oracle is about 0.94, so a 0.94 bar
# flaps by construction. 0.92 / 0.90 still catch a real harness break (the broken 26 Oct run: 0.896).
ORACLE_POOL_MIN = 0.92
ORACLE_RUN_FLOOR = 0.90
ORACLE_MIN = ORACLE_POOL_MIN
# Same harness: these run-record fields are identical. `split` too, except that a finals run (`--full`,
# split "full") pools with the TUNE runs of its harness: "the pool = the finals' run plus the tune runs
# on the same harness" (the owner, 2026-10-27).
HARNESS_FIELDS = (
    "prompt_version",
    "prompts_sha256",
    "answerer",
    "answerer_thinking",
    "judge",
    "split",
    "judge_version",
)
CONDITIONS = ("mem", "none", "oracle")
ABSTENTION = "abstention"  # the judge kind (and the per-type bucket) of `_abs` questions
ANSWER_MAX_TOKENS, JUDGE_MAX_TOKENS = 1024, 64  # judge: upstream's 10 cut Sonnet (ADR 0029)
CHARS_PER_TOKEN = 3  # a deliberately low ratio: the budget guard overestimates input, never under

_HEAD = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. "
)
_TAIL = (
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)
JUDGE_TEMPLATES: dict[str, str] = {
    "single-session-user": _HEAD + _TAIL,
    "single-session-assistant": _HEAD + _TAIL,
    "multi-session": _HEAD + _TAIL,
    "temporal-reasoning": _HEAD
    + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question "
        "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
        "(e.g., predicting 19 days when the answer is 18), the model's response is still correct."
    )
    + _TAIL,
    "knowledge-update": _HEAD
    + (
        "If the response contains some previous information along with an updated answer, the "
        "response should be considered as correct as long as the updated answer is the required "
        "answer."
    )
    + _TAIL,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\n"
        "Model Response: {response}\n\nIs the model response correct? Answer yes or no only."
    ),
    ABSTENTION: (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The "
        "model could say that the information is incomplete, or some other information is given "
        "but the asked information is not.\n\nQuestion: {question}\n\nExplanation: {answer}\n\n"
        "Model Response: {response}\n\nDoes the model correctly identify the question as "
        "unanswerable? Answer yes or no only."
    ),
}

# The answerer's prompts (prompt_version 2; the owner, 2026-10-26; MEM2-207/MEM2-227): `ANSWER` and
# `ANSWER_NONE` of the owner's `data/host-pass7/e1_qa.py` verbatim (LongMemEval's answer template plus
# the don't-know sentence); no system prompt. mem and oracle share `ANSWER` (the history is the memory
# text or the evidence sessions); none uses `ANSWER_NONE`. tests/fixtures/e1_qa_prompts.py pins both.
ANSWER = (
    "I will give you several history chats between you and a user. Please answer the question "
    "based on the relevant chat history. If the history does not contain the answer, say that you "
    "don't know.\n\nHistory Chats:\n{ctx}\n\nCurrent Date: {date}\nQuestion: {q}\nAnswer:"
)
ANSWER_NONE = "Current Date: {date}\nQuestion: {q}\nAnswer (say you don't know if you can't know):"


def judge_kind(q: dict[str, Any]) -> str:
    return ABSTENTION if str(q["question_id"]).endswith("_abs") else q["question_type"]


def judge_prompt(kind: str, question: str, answer: str, response: str) -> str:
    return JUDGE_TEMPLATES[kind].format(question=question, answer=answer, response=response)


def answer_prompt(condition: str, question: str, context: str, date: str) -> str:
    if condition == "none":
        return ANSWER_NONE.format(date=date, q=question)
    return ANSWER.format(ctx=context, date=date, q=question)


_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
UNPARSED = "judge replies without a yes/no token (judge_flagged; not credited; ADR 0029)"


def judge_verdict(text: str) -> str | None:
    """The judge's verdict: the FIRST word-bounded yes/no token of its reply, case-insensitive
    ("yes" | "no"), or None when the reply has neither (flagged by the caller). Upstream
    `evaluate_qa.py` credits `yes` anywhere in a 10-token reply; ADR 0029 says why this differs."""
    m = _YES_NO.search(text)
    return m.group(1).lower() if m else None


def parse_verdict(text: str) -> bool:
    """Correct iff the verdict is yes. A reply without a verdict is not credited (and flagged)."""
    return judge_verdict(text) == "yes"


def flagged(item: dict[str, Any]) -> bool:
    """Whether an item's judge reply has no yes/no token, re-parsed from the reply (old files too)."""
    return judge_verdict(item.get("judge_raw") or "") is None


def verdict_fields(raw: str) -> dict[str, Any]:
    """An item's verdict fields from the judge's reply: `correct`, `judge_verdict` (yes | no | None)
    and `judge_flagged` (no yes/no token). Every arm and runner writes these, from this function."""
    v = judge_verdict(raw)
    return {"correct": v == "yes", "judge_verdict": v, "judge_flagged": v is None}


def oracle_text(q: dict[str, Any]) -> str:
    """The evidence sessions whole, in haystack order, each with its date."""
    gold = set(q["answer_session_ids"])
    out, seen = [], set()
    for sid, date, turns in zip(
        q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True
    ):
        if sid in gold and sid not in seen:
            seen.add(sid)
            body = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
            out.append(f"[session {sid}, {date}]\n{body}")
    return "\n\n".join(out)


EXPANDED_HEADING = "## full text of the rows shown"


def mem_context(r: Any) -> str:
    """The mem arm's context (what an agent reads after `recall`, then `get`): recall's text, then
    the full text of every delivered row. An empty delivery is recall's text alone."""
    return expanded_context(r.text, r.bodies)


def expanded_context(text: str, bodies: Sequence[str]) -> str:
    """`mem_context` from its parts; LoCoMo's answerer reads the same (`locomo_ours`)."""
    if not bodies:
        return text
    return f"## mem recall\n{text}\n\n{EXPANDED_HEADING}\n" + "\n\n".join(bodies)


def context_for(condition: str, q: dict[str, Any], recalled: Callable[[], str]) -> str:
    """The memory text the answerer is given. `recalled` runs the read path only for `mem`."""
    if condition == "none":
        return ""
    if condition == "oracle":
        return oracle_text(q)
    if condition == "mem":
        return recalled()
    raise ValueError(f"unknown condition {condition!r} (want one of {CONDITIONS})")


def _text(reply: Any) -> str:
    return "".join(b.get("text", "") for b in reply.content if b.get("type") == "text").strip()


def prompts_sha256() -> str:
    blob = json.dumps([PROMPT_VERSION, JUDGE_TEMPLATES, ANSWER, ANSWER_NONE], sort_keys=True)  # fmt: skip
    return hashlib.sha256(blob.encode()).hexdigest()


class Spend:
    """One budget over two ledgers (answerer and judge may be different models)."""

    def __init__(self, answerer: LLM, judge: LLM, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.ledgers = {"answer": Ledger(answerer.model, budget_usd),
                        "judge": Ledger(judge.model, budget_usd)}  # fmt: skip

    @property
    def total(self) -> float:
        return sum(lg.total for lg in self.ledgers.values())

    def check(self, role: str, max_tokens: int, prompt_chars: int) -> None:
        inp, out = PRICES[self.ledgers[role].model]
        est_input = prompt_chars // CHARS_PER_TOKEN + 1
        worst = (est_input * inp * 1.25 + max_tokens * out) / 1e6
        if self.total + worst > self.budget_usd:
            raise BudgetExceeded(f"${self.total:.3f} spent, the next {role} call could cost up to "
                                 f"${worst:.3f}, budget ${self.budget_usd:.2f}")  # fmt: skip

    def summary(self) -> dict[str, Any]:
        return {"usd": round(self.total, 4), "budget_usd": self.budget_usd,
                **{role: lg.summary() for role, lg in self.ledgers.items()}}  # fmt: skip


def run_e1(
    questions: Sequence[dict[str, Any]],
    retrieve: Callable[[dict[str, Any]], str] | None,
    answerer: LLM,
    judge: LLM,
    *,
    conditions: Sequence[str] = CONDITIONS,
    budget_usd: float,
    answer_max_tokens: int = ANSWER_MAX_TOKENS,
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
    on_item: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Question by question, condition by condition. Stops at the budget, keeping what finished."""
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise ValueError(f"unknown conditions {bad} (want {CONDITIONS})")
    if "mem" in conditions and retrieve is None:
        raise ValueError("condition mem needs a retrieve function (the read path)")
    spend = Spend(answerer, judge, budget_usd)
    items: list[dict[str, Any]] = []
    stopped = None
    try:
        for q in questions:
            kind = judge_kind(q)
            cache: list[str] = []

            def recalled(q: dict[str, Any] = q, cache: list[str] = cache) -> str:
                if not cache:
                    cache.append(retrieve(q))  # type: ignore[misc]
                return cache[0]

            for cond in conditions:
                ctx = context_for(cond, q, recalled)
                prompt = answer_prompt(cond, q["question"], ctx, q["question_date"])
                spend.check("answer", answer_max_tokens, len(prompt))
                before = spend.total
                reply = answerer.create(system="", tools=[], max_tokens=answer_max_tokens,
                                        messages=[{"role": "user", "content": prompt}])  # fmt: skip
                spend.ledgers["answer"].add(reply.usage, f"{q['question_id']}/{cond}")
                response = _text(reply)
                jp = judge_prompt(kind, q["question"], str(q["answer"]), response)
                spend.check("judge", judge_max_tokens, len(jp))
                jr = judge.create(system="", tools=[], max_tokens=judge_max_tokens,
                                  messages=[{"role": "user", "content": jp}])  # fmt: skip
                spend.ledgers["judge"].add(jr.usage, f"{q['question_id']}/{cond}")
                raw = _text(jr)
                item = {"qid": q["question_id"], "type": q["question_type"], "judge_kind": kind,
                        "condition": cond, "context_chars": len(ctx), "answer": response,
                        "judge_raw": raw, **verdict_fields(raw),
                        "usd": round(spend.total - before, 6)}  # fmt: skip
                items.append(item)
                if on_item:
                    on_item(item)
    except BudgetExceeded as e:
        stopped = str(e)
    planned = len(questions) * len(conditions)
    return {"items": items, "planned": planned, "complete": stopped is None and len(items) == planned,
            "stopped": stopped, "cost": spend.summary(), "conditions": list(conditions),
            "answerer": answerer.model, "judge": judge.model, "prompt_version": PROMPT_VERSION,
            "prompts_sha256": prompts_sha256(), "judge_version": JUDGE_VERSION,
            "judge_flagged": sum(i["judge_flagged"] for i in items)}  # fmt: skip


# ---- scoring (offline) -----------------------------------------------------------------------


def _bucket(item: dict[str, Any]) -> str:
    return ABSTENTION if item.get("judge_kind") == ABSTENTION else item["type"]


# ---- run validity: the oracle arm pooled over the runs of one harness --------------------------


class PoolRefused(ValueError):
    """An `--oracle-pool` file that is not a run of the current run's harness, or counted twice."""


def harness_of(run: dict[str, Any]) -> dict[str, Any]:
    return {f: run.get(f) for f in HARNESS_FIELDS}


def pool_split(split: Any) -> Any:
    """The split a pooled run must have: the run's own, or tune for a finals run (split "full")."""
    return "tune" if split == "full" else split


def check_harness(current: dict[str, Any], other: dict[str, Any], name: str) -> None:
    """Refuses `other` (a pooled run's record) unless its harness is `current`'s, naming the field."""
    want = dict(current, split=pool_split(current.get("split")))
    for f in HARNESS_FIELDS:
        if other.get(f) != want[f]:
            raise PoolRefused(
                f"{name}: harness field {f} is {other.get(f)!r}, the current run's is {want[f]!r}"
                + (" (a finals run pools with tune runs)" if f == "split" and want[f] != current.get(f) else "")
            )  # fmt: skip


def run_digest(run: dict[str, Any]) -> str:
    """Which run a record is (its items), whatever the file name or the fields added later."""
    blob = json.dumps(run.get("items", []), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def oracle_counts(run: dict[str, Any]) -> tuple[int, int]:
    """(correct, n) of the oracle arm, the verdicts re-parsed from the judge's replies."""
    its = [i for i in run.get("items", []) if i.get("condition") == "oracle"]
    return sum(parse_verdict(i["judge_raw"]) for i in its), len(its)


def load_pool(paths: Sequence[Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    """The pooled run files, each checked against `current`'s harness (a record, or just its
    HARNESS_FIELDS) and against the others: a file counted twice is refused."""
    from pathlib import Path

    out: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        raw = path.read_bytes()
        rec = json.loads(raw)
        check_harness(harness_of(current), rec, str(p))
        key = str(path.resolve())
        for k in (key, run_digest(rec)):
            if k in seen:
                raise PoolRefused(f"{p}: the same run as {seen[k]} (a run counts once in the pool)")
            seen[k] = str(p)
        k_ok, n = oracle_counts(rec)
        if not n:
            raise PoolRefused(f"{p}: no oracle arm (a pooled run contributes its oracle answers)")
        out.append({"file": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
                    "digest": run_digest(rec), "correct": k_ok, "n": n})  # fmt: skip
    return out


def oracle_pool(run: dict[str, Any], pool: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """This run's oracle and the pooled oracle over it plus `pool` (from `load_pool`). Without a
    pool the pool is this run alone. The current run is always in the pool, once."""
    k, n = oracle_counts(run)
    me = run_digest(run)
    for m in pool:
        if m["digest"] == me:
            raise PoolRefused(
                f"{m['file']}: the current run itself (it is always in the pool, once)"
            )
    pk, pn = k + sum(m["correct"] for m in pool), n + sum(m["n"] for m in pool)
    return {"oracle_run": k / n if n else None, "oracle_run_n": n,
            "oracle_pooled": pk / pn if pn and n else None, "oracle_pool_n": pn if n else 0,
            "oracle_pool": [{"file": m["file"], "sha256": m["sha256"], "oracle": m["correct"] / m["n"],
                             "n": m["n"]} for m in pool]}  # fmt: skip


def pool_line(p: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    return (
        f"oracle run={f(p['oracle_run'])}, pooled={f(p['oracle_pooled'])} (n={p['oracle_pool_n']})"
    )


VALIDITY_RULE = f"pooled >= {ORACLE_POOL_MIN:.2f}, every run >= {ORACLE_RUN_FLOOR:.2f}"


def pool_invalid_reason(p: dict[str, Any]) -> str | None:
    """B-E1 is INVALID (neither pass nor fail) unless this run's oracle is >= ORACLE_RUN_FLOOR and
    the pooled oracle >= ORACLE_POOL_MIN. The reason names both numbers."""
    o, pooled = p["oracle_run"], p["oracle_pooled"]
    if o is None:
        return f"no oracle arm in the run (validity: {VALIDITY_RULE})"
    if o < ORACLE_RUN_FLOOR:
        return (f"oracle accuracy {o:.3f} < {ORACLE_RUN_FLOOR:.2f} on this run: a harness failure, "
                f"whatever the pool ({pool_line(p)}; validity: {VALIDITY_RULE})")  # fmt: skip
    if pooled < ORACLE_POOL_MIN:
        return (f"pooled oracle accuracy {pooled:.3f} < {ORACLE_POOL_MIN:.2f} ({pool_line(p)}; "
                f"validity: {VALIDITY_RULE}; the harness loses answers; the run proves nothing)")  # fmt: skip
    return None


def invalid_reason(metrics: dict[str, float]) -> str | None:
    """The validity of a scored run whose pool is itself alone (`score_run(run)` without a pool)."""
    o, n = metrics.get("accuracy_oracle", math.nan), int(metrics.get("n_oracle", 0) or 0)
    p = {"oracle_run": None if math.isnan(o) else o, "oracle_run_n": n,
         "oracle_pooled": metrics.get("oracle_pooled", None if math.isnan(o) else o),
         "oracle_pool_n": int(metrics.get("oracle_pool_n", n))}  # fmt: skip
    return pool_invalid_reason(p)


def score_run(
    run: dict[str, Any], b: int = 2000, pool: Sequence[dict[str, Any]] = ()
) -> tuple[dict[str, float], dict[str, Any]]:
    """Metrics from a run file (`pool`: the other runs of its harness, from `load_pool`). `accuracy` is the `mem` arm (micro: every question one vote), its 95%
    CI resamples questions within their type bucket. The other arms and the per-type values are
    reported next to it."""
    nan = math.nan
    metrics: dict[str, float] = {}
    per: dict[str, dict[str, Any]] = {}
    for cond in run["conditions"]:
        its = [i for i in run["items"] if i["condition"] == cond]
        # the verdict is re-parsed from the judge's reply, not trusted from the file
        ok = [float(parse_verdict(i["judge_raw"])) for i in its]
        strata = [_bucket(i) for i in its]
        acc, lo, hi = stats.bootstrap_mean(ok, strata, b) if its else (nan, nan, nan)
        by: dict[str, list[float]] = defaultdict(list)
        for v, s in zip(ok, strata, strict=True):
            by[s].append(v)
        per_type = {t: {"n": len(v), "accuracy": sum(v) / len(v),
                        "ci95": list(stats.bootstrap_mean(v, [t] * len(v), b)[1:])}
                    for t, v in sorted(by.items())}  # fmt: skip
        macro = sum(sum(v) / len(v) for v in by.values()) / len(by) if by else nan
        per[cond] = {"n": len(its), "accuracy": acc, "ci95": [lo, hi], "per_type": per_type,
                     "macro": macro, "judge_flagged": sum(flagged(i) for i in its)}  # fmt: skip
        suffix = "" if cond == "mem" else f"_{cond}"
        metrics[f"accuracy{suffix}"] = acc
        metrics[f"n{suffix}"] = float(len(its))
        if cond == "mem":
            metrics["accuracy_ci_lo"], metrics["accuracy_ci_hi"] = lo, hi
            metrics["accuracy_macro"] = per[cond]["macro"]
            metrics["abstention_accuracy"] = (
                per[cond]["per_type"][ABSTENTION]["accuracy"] if ABSTENTION in by else nan
            )
    metrics.setdefault("accuracy", nan)
    p = oracle_pool(run, pool)
    metrics["oracle_run"] = nan if p["oracle_run"] is None else p["oracle_run"]
    metrics["oracle_pooled"] = nan if p["oracle_pooled"] is None else p["oracle_pooled"]
    metrics["oracle_pool_n"] = float(p["oracle_pool_n"])
    metrics["oracle_valid"] = float(pool_invalid_reason(p) is None)
    # replies without a yes/no token, re-parsed (a 10-token file shows its cut replies here)
    metrics["judge_flagged"] = float(sum(flagged(i) for i in run["items"]))
    metrics["run_complete"] = float(bool(run.get("complete")))
    metrics["cost_usd"] = float(run["cost"]["usd"])
    return metrics, {"per_condition": per, "oracle": p}
