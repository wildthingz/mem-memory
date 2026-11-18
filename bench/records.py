"""Run records, per-item files and the look ledger (benchmarks-v2 §6).

A run is `<run_id>.json` plus `<run_id>.items.jsonl` under bench/results/ (git-ignored). The
ledger `bench/ledger.jsonl` is committed and logs test looks (who, why, config hash, §5.4).
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mem import config as mem_config

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
LEDGER = BENCH_DIR / "ledger.jsonl"
BENCH_VERSION = 2

# T0 uses the stub embedder/reranker; it is recorded next to the production config fingerprint.
T0_STUBS = {"embedder": "HashEmbedder(256) (mem.testing)", "reranker": "none (P1 has no reranker)"}


def git_sha(cwd: Path = BENCH_DIR) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True)  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip()  # fmt: skip
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> dict[str, Any]:
    import os

    return {"machine": platform.machine(), "system": platform.platform(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}  # fmt: skip


def config_fingerprint(cfg: mem_config.Config | None = None) -> dict[str, Any]:
    cfg = cfg or mem_config.default_config()
    return {"fp_all": cfg.fp_all(), "fp_embed": cfg.fp_embed(), "device": cfg.device,
            "config": {"embedder": cfg.embedder, "reranker": cfg.reranker,
                       "retrieval": cfg.retrieval, "gate": cfg.gate}}  # fmt: skip


@dataclass
class RunRecord:
    run_id: str
    bench_id: str
    tier: str
    mode: str
    dataset: dict[str, Any]
    config: dict[str, Any]
    stubs: dict[str, str]
    git_sha: str
    hardware: dict[str, Any]
    metrics: dict[str, float]
    gates: list[dict[str, Any]]
    violations: list[str]
    n_items: int
    wall_s: float
    started_at: str
    bench_version: int = BENCH_VERSION
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # ci95, per_category, n, dataset, ...


def new_run_id(bench_id: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%dT%H%M%S%fZ}-{bench_id}"


def write_run(
    record: RunRecord, items: list[dict[str, Any]], results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{record.run_id}.json"
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (results_dir / f"{record.run_id}.items.jsonl").write_text(
        "".join(json.dumps(i, sort_keys=True, ensure_ascii=False) + "\n" for i in items),
        encoding="utf-8",
    )
    return path


def append_ledger(entry: dict[str, Any], ledger: Path = LEDGER) -> None:
    """One JSON line per test look: who, why, phase, split, config hash, git sha, time."""
    required = {"who", "why", "phase", "split", "config_hash"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"ledger entry lacks {sorted(missing)}")
    row = {"at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "git_sha": git_sha(), **entry}
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
