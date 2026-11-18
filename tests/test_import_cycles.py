"""MEM2-240: every module imports on its own in a fresh interpreter, and `bench prf-check --help` runs.

In one pytest process the modules are already imported in an order that hides a cycle (on 0b483f4:
mem.gate -> mem.readguard -> mem.conflicts -> mem.read -> mem.gate), so each import here runs in
its own subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _modules() -> list[str]:
    paths = sorted((ROOT / "src" / "mem").glob("*.py")) + sorted((ROOT / "bench").glob("*.py"))
    names = []
    for p in paths:
        pkg = p.parent.name
        names.append(pkg if p.stem == "__init__" else f"{pkg}.{p.stem}")
    return names


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], check=False, cwd=ROOT, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("module", _modules())
def test_module_imports_alone_in_a_fresh_interpreter(module):
    r = _run("-c", f"import {module}")
    assert r.returncode == 0, r.stderr


def test_bench_prf_check_help_runs():
    r = _run("-c", "from bench.cli import app; app()", "prf-check", "--help")
    assert r.returncode == 0, r.stderr
    assert "prf-check" in r.stdout or "Usage" in r.stdout, r.stdout
