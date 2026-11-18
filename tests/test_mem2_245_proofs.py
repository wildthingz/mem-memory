"""MEM2-245 proofs: the round-2 re-check of MEM2-237's fixes. Each test fails on the fix tip it
names and pins the defect; Bosmang puts it on the fix as a regression test."""

from __future__ import annotations

from pathlib import Path

import pytest

from mem import harnesses as hx
from tests.test_setup_cmd import SetupFake, _setup

LINUX = {"env": {}, "platform": "linux"}


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


# ---- N1 (bosmang/mem2-243 @ b36e27a): a Codex entry mem cannot edit stops setup, not the harness --


def test_setup_skips_a_harness_whose_entry_mem_cannot_edit_and_goes_on(home):
    """docs/troubleshooting.md and docs/harnesses.md (MEM2-243) say that a file mem `cannot change
    safely` makes `mem setup` skip that harness with a `[warn]` line and do the others. It does
    not: `ConfigWriter.configured` parses the file fine while setup plans, `add` is planned, and
    `_save` refuses at apply time, so setup stops with `[FAIL] mcp: ... cannot change this file
    safely` (exit 1) after the service is started and before doctor and the self-test. Every
    later run stops at the same step until the user edits ~/.codex/config.toml by hand."""
    (home / ".codex").mkdir()
    cfg = home / ".codex" / "config.toml"
    junk = '[mcp_servers]\nmem-personal = { command = "x", args = [\n  "a",\n] }\n'
    cfg.write_text(junk)
    run = SetupFake()
    code, out = _setup(home, run, yes=True, writers=[hx.ConfigWriter("codex", **LINUX)])
    assert cfg.read_text() == junk, out  # never written: the fix keeps this promise
    assert "[warn] mcp: Codex CLI skipped:" in out, out
    assert code == 0 and run.selftests == 1, out
