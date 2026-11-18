"""The writer model for T2 runs (benchmarks-v2 §0.5) and the B-C5 spend ledger.

`AnthropicLLM` wraps the official SDK (`uv run --extra llm ...`; ANTHROPIC_API_KEY). Tests use a
scripted fake with the same `.create()` shape. Every call's complete usage is appended to the
`Ledger`: input, cache_creation, cache_read and output tokens (thinking included) x list price.
A hard `budget_usd` raises `BudgetExceeded` BEFORE a call that could cross it (worst case: the
whole `max_tokens` is spent), so a run never silently overspends.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# $ per million tokens: (input, output). Cache reads 0.1x input; cache writes 1.25x input (5 min).
# Sonnet 5.5 $2/$10 (benchmarks-v2 §7, API model table). The spec names `claude-haiku-5-5` at
# $0.10/$0.50; the API model table (2026-09-25) lists `claude-haiku-4-5` at $1/$5. Both are kept;
# confirm the id before using the cheap variant for skill iteration.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
    # Opus 5.5 $4/$20 per MTok: third-party price pages (2026-10-24), not Anthropic's own table.
    # PG-v1 drafting (bench gold draft) uses it; confirm against the official table before relying
    # on the ledger's dollar figure (the --budget-usd cap is the guard either way).
    "claude-opus-5-5": (4.0, 20.0),
}
CACHE_READ, CACHE_WRITE = 0.1, 1.25


class BudgetExceeded(RuntimeError):
    pass


def call_cost(model: str, usage: dict[str, int]) -> float:
    inp, out = PRICES[model]
    return (usage.get("input_tokens", 0) * inp
            + usage.get("cache_creation_input_tokens", 0) * inp * CACHE_WRITE
            + usage.get("cache_read_input_tokens", 0) * inp * CACHE_READ
            + usage.get("output_tokens", 0) * out) / 1e6  # fmt: skip


@dataclass
class Ledger:
    model: str
    budget_usd: float
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(c["usd"] for c in self.calls)

    def check(self, max_tokens: int, est_input: int) -> None:
        """Worst case of the next call: its input uncached plus `max_tokens` of output."""
        inp, out = PRICES[self.model]
        worst = (est_input * inp * CACHE_WRITE + max_tokens * out) / 1e6
        if self.total + worst > self.budget_usd:
            raise BudgetExceeded(f"${self.total:.3f} spent, the next call could cost up to "
                                 f"${worst:.3f}, budget ${self.budget_usd:.2f}")  # fmt: skip

    def add(self, usage: dict[str, int], label: str) -> float:
        usd = call_cost(self.model, usage)
        self.calls.append({"label": label, **usage, "usd": round(usd, 6)})
        return usd

    def summary(self) -> dict[str, Any]:
        keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                "output_tokens")  # fmt: skip
        return {"model": self.model, "budget_usd": self.budget_usd, "calls": len(self.calls),
                "usd": round(self.total, 4), **{k: sum(c.get(k, 0) for c in self.calls)
                                                for k in keys}}  # fmt: skip


@dataclass
class Reply:
    content: list[dict[str, Any]]  # blocks as plain dicts, replayed unchanged
    stop_reason: str
    usage: dict[str, int]


class LLM(Protocol):
    model: str

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]], max_tokens: int) -> Reply: ...  # fmt: skip


class AnthropicLLM:
    """claude-sonnet-5-5 at effort `low`. `effort=None` is the B-E1 answerer: `thinking: between_tools`,
    no effort, no system prompt, no sampling parameters, exactly what the owner's `e1_qa.py` `ask()`
    sends (MEM2-227 F5). No sampling parameters anywhere (rejected by the model), so
    determinism comes from repeats (benchmarks-v2 §0.5). Thinking blocks are returned and replayed
    unchanged within a dialogue (the same conversation produced them)."""

    def __init__(self, model: str = "claude-sonnet-5-5", effort: str | None = "low") -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("a T2 run needs ANTHROPIC_API_KEY (and `uv run --extra llm`)")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("a T2 run needs `uv run --extra llm`") from e
        self.model, self.effort = model, effort
        self._client = anthropic.Anthropic()

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]], max_tokens: int) -> Reply:  # fmt: skip
        extra: dict[str, Any] = {"tools": tools} if tools else {}  # B-E1 calls pass none
        if system:
            extra["system"] = system
        if self.effort is None:  # the B-E1 answerer, as the owner's e1_qa.py ask() (MEM2-227 F5):
            extra["thinking"] = {"type": "between_tools"}  # `disabled` 400s on claude-sonnet-5-5
        else:
            extra["output_config"] = {"effort": self.effort}
        r = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, messages=messages,
            cache_control={"type": "ephemeral"}, **extra)  # fmt: skip
        u = r.usage
        usage = {"input_tokens": u.input_tokens or 0, "output_tokens": u.output_tokens or 0,
                 "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                 "cache_read_input_tokens": u.cache_read_input_tokens or 0}  # fmt: skip
        return Reply([b.model_dump(exclude_none=True) for b in r.content], r.stop_reason or "",
                     usage)  # fmt: skip
