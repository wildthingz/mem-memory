"""Pure helpers of the day-1 spike (bench/spike.py): statistics, parity checks, verdict."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

COSINE_MIN = 0.9999
LOGIT_MAX_DELTA = 1e-3
RERANK_P95_LIMIT_MS = 1000.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_ms(values: Sequence[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def cosine_parity(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Row-wise cosine between two embedding matrices (D1: min must be >= 0.9999)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    low = float(cos.min())
    return {"min_cosine": low, "threshold": COSINE_MIN, "ok": low >= COSINE_MIN}


def logit_parity(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Max absolute difference between two logit vectors (D1: must be <= 1e-3)."""
    delta = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    return {"max_abs_delta": delta, "threshold": LOGIT_MAX_DELTA, "ok": delta <= LOGIT_MAX_DELTA}


def verdict(
    gpu: dict[str, dict[str, float]] | None,
    cosine_ok: bool | None,
    logit_ok: bool | None,
) -> dict[str, Any]:
    """Map the measurements to the outcome the runbook explains.

    `gpu` holds the GPU rerank summaries, keys "avg" and "worst" (average-length and 512-token
    pairs). The limit is judged on the average-pair p95; a worst-case p95 over the limit is
    reported as a warning.
    """
    if gpu is None:
        return {"outcome": "no-gpu", "device_in_fingerprint": None, "warnings": []}
    warnings: list[str] = []
    if gpu["worst"]["p95"] > RERANK_P95_LIMIT_MS:
        warnings.append("worst-case (512-token) p95 above 1,000 ms")
    if not (cosine_ok and logit_ok):
        return {"outcome": "parity-fail", "device_in_fingerprint": True, "warnings": warnings}
    if gpu["avg"]["p95"] > RERANK_P95_LIMIT_MS:
        return {"outcome": "dr2-fallback", "device_in_fingerprint": False, "warnings": warnings}
    return {"outcome": "pass", "device_in_fingerprint": False, "warnings": warnings}
