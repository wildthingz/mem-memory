"""Token estimate (benchmarks-v2 §0.4). Server budget and every "est." number use this."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def est_tokens(text: str) -> int:
    """ceil(utf8_bytes / 3): deliberately conservative."""
    return math.ceil(len(text.encode("utf-8")) / 3)


def summarize(values: Sequence[int]) -> dict[str, float | int]:
    """Mean and nearest-rank p95 (no interpolation) over per-response token counts."""
    if not values:
        return {"n": 0, "mean": 0.0, "p95": 0}
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return {"n": len(ordered), "mean": statistics.fmean(ordered), "p95": ordered[rank - 1]}


def anthropic_counter(model: str = "claude-sonnet-5-5"):
    """Client-visible token counter (§0.4): Anthropic `count_tokens` on the serialised tool-result.

    Needs `ANTHROPIC_API_KEY` and `uv run --with anthropic`. The API counts a user message, which
    carries a small fixed overhead; it is measured once with a one-character message and subtracted,
    so an empty-result line is not charged for it. Raises RuntimeError without the package or key.
    """
    import os

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("count_tokens needs `uv run --with anthropic`") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("count_tokens needs ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()

    def raw(text: str) -> int:
        r = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}])
        return r.input_tokens

    overhead = raw("x") - 1

    def count(texts: Sequence[str]) -> list[int]:
        return [max(raw(t) - overhead, 0) if t else 0 for t in texts]

    return count
