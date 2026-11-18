"""Shared types for suites: a suite returns a SuiteResult; gates read only `metrics`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class BackendUnavailable(RuntimeError):
    """The code under test has not landed yet (e.g. `mem.serve`). Gates treat this as FAIL."""


@dataclass
class ToolResult:
    text: str
    structured: dict[str, Any]


class Client(Protocol):
    """One store, in process. `now` is the injected op date (ISO UTC) for that call."""

    db_path: Path

    def call(self, tool: str, args: dict[str, Any], *, now: str | None = None) -> ToolResult: ...
    def close(self) -> None: ...


# factory(db_path, scope=..., tz=..., now=...) -> Client
ClientFactory = Callable[..., Client]


@dataclass
class SuiteResult:
    bench_id: str
    tier: str = "t0"
    mode: str = "fixture"
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # ci95, per_category, n, dataset, ...
    # metric -> why it was not run (an explicit, scoped deferral: reported DEFERRED, never PASS)
    deferred: dict[str, str] = field(default_factory=dict)
    # why the whole run proves nothing (e.g. B-E1's oracle arm below its floor): reported INVALID
    invalid: str | None = None

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        """Record one named check; a failed check is a violation."""
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.violations.append(f"{name}: {detail}" if detail else name)
        return bool(ok)


@dataclass
class Context:
    """What a suite needs from the outside: where to put stores and how to open a client."""

    workdir: Path
    factory: ClientFactory
    list_tools: Callable[[], list[dict[str, Any]]] | None = None  # default: the live server's
    options: dict[str, Any] = field(default_factory=dict)  # P2+: {"lme": LmeSetup, ...}

    def store(self, name: str) -> Path:
        d = self.workdir / name
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            raise FileExistsError(d)
        return d
