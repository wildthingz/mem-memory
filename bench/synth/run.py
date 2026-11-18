"""One oracle-client replay of a SYN world into a fresh store, shared by the P3 suites.

`SynSetup` (in `Context.options["syn"]`) names the seeds and size for the split being run and holds
a cache so B-W1/B-W2/B-W3/B-R5/B-R2 (ii) replay each seed once per `bench gate` run (the stores
live in `workdir`, which the caller keeps until the gate is done). Without a setup (T0 tests) the
suites use `DEFAULT_T0` and replay into their own temp dir.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.suites.base import BackendUnavailable, Context
from bench.synth import data as D
from bench.synth import score
from bench.synth.generator import Synth, generate
from bench.synth.replay import OpLog, Replay

OPT = "syn"


@dataclass
class SynSetup:
    seeds: tuple[int, ...]
    size: str
    split: str  # tune (dev seeds) | test (held-out seeds) | fixture
    workdir: Path | None = None
    cache: dict[tuple[int, str], SeedRun] = field(default_factory=dict)
    t1: dict[str, Any] = field(default_factory=dict)  # real-model wiring notes for the record
    cut: Callable[[list[dict[str, Any]]], list[str]] | None = None  # syn-score-dev --fit (Replay)


def default_t0() -> SynSetup:
    """T0: seed 1, size S, replayed into the suite's own temp dir (a fresh cache per Context)."""
    return SynSetup((1,), "S", "fixture")


def setup_of(ctx: Context) -> SynSetup:
    return ctx.options.setdefault(OPT, default_t0())


@dataclass
class SeedRun:
    syn: Synth
    replay: Replay
    db: Path
    logs: list[OpLog]
    history: list[str]  # violations of "memory count never decreases; superseded text unchanged"

    @property
    def mids(self) -> dict[int, int]:
        return score.saved_map(self.logs)


def _history_hook(db: Path, seen: dict[int, tuple[str, str]], problems: list[str]):
    prev_count = [0]

    def hook(log: OpLog) -> None:
        mem = score.read_memories(db)
        if len(mem) < prev_count[0]:
            problems.append(f"op {log.op['i']}: memories {prev_count[0]} -> {len(mem)}")
        prev_count[0] = len(mem)
        for i, row in mem.items():
            old = seen.setdefault(i, (row.content_hash, row.body))
            if old != (row.content_hash, row.body):
                problems.append(f"op {log.op['i']}: m{i} text changed")

    return hook


def replay_seed(ctx: Context, seed: int, size: str, tag: str = "syn", *, act: bool = True,
                keep: bool = True) -> SeedRun:  # fmt: skip
    setup = setup_of(ctx)
    key = (seed, size)
    keep = keep and setup.workdir is not None  # a cache needs stores that outlive one suite
    if act and keep and key in setup.cache:
        return setup.cache[key]
    syn = generate(seed, size)
    if setup.workdir is not None:
        db = setup.workdir / f"{tag}-{size}{seed}-{'act' if act else 'noact'}.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.unlink(missing_ok=True)
    else:
        n = ctx.options["_replays"] = ctx.options.get("_replays", 0) + 1
        db = ctx.store(f"{tag}-{size}{seed}-{n}.db")
    client = ctx.factory(db, scope="personal", tz=D.TZ, now=syn.ops[0]["now"])
    problems: list[str] = []
    rp = Replay(client, syn, act=act, on_op=_history_hook(db, {}, problems), cut=setup.cut)
    try:
        rp.run()
    finally:
        client.close()
    run = SeedRun(syn, rp, db, rp.logs, problems)
    if act and keep:
        setup.cache[key] = run
    return run


def require_p3(ctx: Context) -> None:
    """The P3 write path must exist: `link` is the first tool P1 refuses. UNAVAILABLE, not a pass."""
    if ctx.options.get("_p3_ok"):
        return
    probe = ctx.store("p3-probe.db")
    client = ctx.factory(probe, scope="personal")
    try:
        a = client.call(
            "remember",
            {"text": "Probe memory alpha", "type": "fact", "about": ["new:topic:ProbeA"]},
            now="2026-10-22T09:00:00Z",
        ).structured["id"]
        b = client.call(
            "remember",
            {"text": "Probe memory beta", "type": "fact", "about": ["new:topic:ProbeA"]},
            now="2026-10-22T09:01:00Z",
        ).structured["id"]
        try:
            client.call("link", {"src": b, "rel": "supersedes", "dst": a},
                        now="2026-10-22T09:02:00Z")  # fmt: skip
        except Exception as e:
            raise BackendUnavailable(f"the server has no working `link` yet: {e}") from e
    finally:
        client.close()
    ctx.options["_p3_ok"] = True


def per_seed(values: dict[int, dict[str, float]], metric: str, fn: str) -> float:
    xs = [v[metric] for v in values.values() if not math.isnan(v.get(metric, math.nan))]
    return (min if fn == "min" else max)(xs) if xs else math.nan
