"""MEM2-246: what the merge of the three MEM2-237 fix branches (242, 243, 244) adds: uninstall's
`installed` check is guarded like setup's `configured` (MEM2-245 merge note 1), and a harness file
mem cannot change is skipped with `[warn]` at apply time on uninstall too (N1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mem import harnesses as hx
from mem.setup_steps import Backups
from tests.test_setup_cmd import SetupFake, _setup

LINUX = {"env": {}, "platform": "linux"}


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


def test_uninstall_skips_a_broken_harness_file_with_no_backup_slot(home):
    """`ConfigWriter.installed` parses the file when no backup slot exists. A syntax-broken file
    with mem entries in it is skipped with `[warn]`; the service and the rest are still undone."""
    (home / ".gemini").mkdir()
    settings = home / ".gemini" / "settings.json"
    settings.write_text('{"mcpServers": {}}\n')
    run = SetupFake()
    writer = hx.ConfigWriter("gemini", **LINUX)
    code, out = _setup(home, run, yes=True, writers=[writer])
    assert code == 0 and "mem-personal" in settings.read_text(), out
    Backups(home / ".config" / "mem" / "backups").drop(settings)
    broken = settings.read_text() + "}{\n"
    settings.write_text(broken)
    code, out = _setup(home, run, yes=True, uninstall=True, writers=[writer])
    assert code == 0, out
    assert "[warn] mcp: Gemini CLI skipped:" in out, out
    assert run.enabled == set() and not run.mcp, out
    assert settings.read_text() == broken


def test_uninstall_skips_an_entry_mem_cannot_edit_and_goes_on(home):
    """A hand-written inline-table `mem-personal` in ~/.codex/config.toml: uninstall finds it, but
    `remove` refuses to rewrite that shape. The harness is skipped with `[warn]`, the file left
    as it is, and the uninstall still ends 0."""
    (home / ".codex").mkdir()
    cfg = home / ".codex" / "config.toml"
    run = SetupFake()
    writer = hx.ConfigWriter("codex", **LINUX)
    code, out = _setup(home, run, yes=True, writers=[writer])
    assert code == 0, out
    junk = '[mcp_servers]\nmem-personal = { command = "x", args = [\n  "a",\n] }\n'
    Backups(home / ".config" / "mem" / "backups").drop(cfg)
    cfg.write_text(junk)
    code, out = _setup(home, run, yes=True, uninstall=True, writers=[writer])
    assert code == 0, out
    assert "[warn] mcp: Codex CLI skipped:" in out, out
    assert run.enabled == set(), out
    assert cfg.read_text() == junk


def test_a_refused_add_leaves_no_backup_slot(home):
    """`add` refuses before it saves the file: no slot is left that would make uninstall think
    setup changed it."""
    (home / ".codex").mkdir()
    cfg = home / ".codex" / "config.toml"
    cfg.write_text('[mcp_servers]\nmem-personal = { command = "x", args = [\n  "a",\n] }\n')
    run = SetupFake()
    code, out = _setup(home, run, yes=True, writers=[hx.ConfigWriter("codex", **LINUX)])
    assert code == 0 and "[warn] mcp: Codex CLI skipped:" in out, out
    assert not Backups(home / ".config" / "mem" / "backups").has(cfg)
