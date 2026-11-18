"""MEM2-243: the MEM2-237 review findings I1, I3, I5 and M4 (the harness config writers and their
backups), from the proofs on bosmang-proof/mem/setup-review @ 671d426, plus the tests of the fix."""

from __future__ import annotations

import contextlib
import json
import os
import stat
import tomllib
from pathlib import Path

import pytest

from mem import harnesses as hx
from mem.setup_steps import Backups
from tests.test_setup_cmd import SetupFake, _setup

LINUX = {"env": {}, "platform": "linux"}


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


# ---- I1: the backup of a private config file is world-readable --------------------------------


def test_backup_slot_keeps_the_private_mode_of_the_config_it_copies(tmp_path, umask_022):
    """`Backups.save` copies ~/.cursor/mcp.json (0600, holding the user's other servers' tokens)
    with `shutil.copyfile`, which creates the copy with the umask: 0644 under the default 022.
    The slot and its folder must be no more readable than the file they hold."""
    secret = tmp_path / "mcp.json"
    secret.write_text('{"mcpServers": {"github": {"headers": {"Authorization": "Bearer ghp_x"}}}}')
    secret.chmod(0o600)
    backups = Backups(tmp_path / "backups")
    backups.save(secret)
    slot = backups._slot(secret)
    data = slot / "data"
    assert data.read_bytes() == secret.read_bytes()
    assert stat.S_IMODE(data.stat().st_mode) & 0o077 == 0, oct(data.stat().st_mode)
    assert stat.S_IMODE(slot.stat().st_mode) & 0o077 == 0, oct(slot.stat().st_mode)


# ---- I3: one harness config with a comment blocks setup and uninstall --------------------------


def _gemini(home: Path, text: str) -> Path:
    (home / ".gemini").mkdir(exist_ok=True)
    path = home / ".gemini" / "settings.json"
    path.write_text(text)
    return path


def test_uninstall_still_undoes_the_install_when_a_harness_config_has_a_comment(home):
    """After setup, the user adds a `//` comment to ~/.gemini/settings.json (Gemini CLI and VS
    Code read JSONC). `mem setup --uninstall` then raises in `ConfigWriter.configured` while it
    plans, before any step runs: the service stays enabled, Claude Code keeps the servers and
    the mem entries stay in the file. The same parse error stops every later `mem setup` run
    before doctor and the self-test (exit 1), and `--no-mcp` does not skip the writer."""
    settings = _gemini(home, '{"theme": "dark", "mcpServers": {}}\n')
    run = SetupFake()
    writer = hx.ConfigWriter("gemini", **LINUX)
    code, out = _setup(home, run, yes=True, writers=[writer])
    assert code == 0 and "mem-personal" in settings.read_text(), out
    settings.write_text("// my settings\n" + settings.read_text())
    code, out = _setup(home, run, yes=True, uninstall=True, writers=[writer])
    assert run.enabled == set() and not run.mcp, out
    assert "mem-personal" not in settings.read_text(), out


# ---- I5: the Codex writer can write a config.toml that Codex cannot parse ----------------------


def test_codex_writer_never_leaves_a_config_toml_that_does_not_parse(tmp_path):
    """A `mem-personal` entry written as an inline table under `[mcp_servers]` (valid TOML, the
    shape a hand edit or another tool produces) is not removed by `_strip`, which only recognises
    `[mcp_servers.mem-*]` headers, so `_merge` appends a second definition. The result does not
    parse: Codex CLI stops reading its config, and every later `mem setup` and `--uninstall`
    stops at `does not parse`. The writer must either replace the entry or refuse the file."""
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "o3"\n\n[mcp_servers]\nmem-personal = { url = "http://127.0.0.1:8765/personal/mcp",'
        ' http_headers = { "X-Mem-Host" = "box", "Authorization" = "Bearer OLD" } }\n'
    )
    spec = hx.Spec("box", 8765, "NEW", Path("/repo"), {})
    with contextlib.suppress(Exception):  # refusing is fine; writing junk is not
        hx.write_config(hx.get("codex"), spec, "http", path, Backups(tmp_path / "b"))
    doc = tomllib.loads(path.read_text())  # must still parse
    assert doc["mcp_servers"]["mem-personal"]["http_headers"]["Authorization"] in (
        "Bearer OLD", "Bearer NEW")  # fmt: skip


def test_restore_keeps_a_symlinked_config_a_symlink(tmp_path, home):
    """A dotfiles symlink (~/.cursor/mcp.json -> ~/dotfiles/cursor-mcp.json): `_commit` replaces
    the link with a regular file (`os.replace` of a temp file), and `restore` copies the bytes
    back into a regular file. The target is untouched but the link is gone: not byte for byte."""
    target = tmp_path / "dotfiles" / "cursor-mcp.json"
    target.parent.mkdir()
    target.write_text('{"mcpServers": {"other": {"url": "http://x"}}}\n')
    (home / ".cursor").mkdir()
    link = home / ".cursor" / "mcp.json"
    link.symlink_to(target)
    backups = Backups(home / ".config/mem/backups")
    spec = hx.Spec("box", 8765, "tok", Path("/repo"), {})
    hx.write_config(hx.get("cursor"), spec, "http", link, backups)
    hx.remove_config(hx.get("cursor"), spec, "http", link, backups)
    assert link.is_symlink() and os.readlink(link) == str(target)


# ---- the fix: beyond the proofs ------------------------------------------------------------------

SPEC = hx.Spec("box", 8765, "tok", Path("/repo"), {})
COMMENTED = """// my settings
{
  "theme": "dark", /* the dark one */
  "mcpServers": {
    "keep": {"httpUrl": "http://k"}, // a server of mine
  },
}
"""


def test_a_jsonc_config_keeps_its_comments_through_write_and_comes_back_byte_for_byte(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(COMMENTED)
    bk = Backups(tmp_path / "b")
    assert hx.write_config(hx.get("gemini"), SPEC, "http", path, bk) == "written"
    text = path.read_text()
    for note in ("// my settings", "/* the dark one */", "// a server of mine"):
        assert note in text
    doc = json.loads(hx._blank(text))
    assert doc["theme"] == "dark" and doc["mcpServers"]["keep"] == {"httpUrl": "http://k"}
    assert doc["mcpServers"]["mem-personal"]["headers"]["Authorization"] == "Bearer tok"
    assert hx.write_config(hx.get("gemini"), SPEC, "http", path, bk) == "unchanged"
    assert hx.remove_config(hx.get("gemini"), SPEC, "http", path, bk) == "restored"
    assert path.read_text() == COMMENTED


def test_a_comment_added_after_setup_survives_uninstall(tmp_path):
    """Restoring the backup would drop the user's new comment, so only our entries go."""
    path = tmp_path / "mcp.json"
    path.write_text('{"servers": {"other": {"type": "http", "url": "http://x"}}}\n')
    bk = Backups(tmp_path / "b")
    hx.write_config(hx.get("vscode"), SPEC, "http", path, bk)
    path.write_text(path.read_text().replace('"servers"', '// mine\n  "servers"'))
    assert hx.remove_config(hx.get("vscode"), SPEC, "http", path, bk) == "removed"
    text = path.read_text()
    assert "// mine" in text and "mem-" not in text and not bk.has(path)
    assert json.loads(hx._blank(text)) == {
        "servers": {"other": {"type": "http", "url": "http://x"}}
    }


def test_a_config_that_cannot_be_read_skips_that_harness_and_the_others_go_on(home):
    settings = _gemini(home, '{"mcpServers": {')  # broken: not even JSONC
    (home / ".cursor").mkdir()
    run = SetupFake()
    writers = [hx.ConfigWriter("gemini", **LINUX), hx.ConfigWriter("cursor", **LINUX)]
    code, out = _setup(home, run, yes=True, writers=writers)
    assert "[warn] mcp: Gemini CLI skipped:" in out and "settings.json does not parse" in out, out
    assert code == 0 and "mem-personal" in (home / ".cursor/mcp.json").read_text(), out
    assert settings.read_text() == '{"mcpServers": {'
    code, out = _setup(home, run, yes=True, uninstall=True, writers=writers)
    assert code == 0 and "[warn] mcp: Gemini CLI skipped:" in out, out
    assert not (home / ".cursor/mcp.json").exists()


def test_codex_dotted_keys_of_ours_are_replaced_not_duplicated(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[mcp_servers]\nmem-personal.url = "http://old"\nother = { url = "http://o" }\n'
    )
    hx.write_config(hx.get("codex"), SPEC, "http", path, Backups(tmp_path / "b"))
    servers = tomllib.loads(path.read_text())["mcp_servers"]
    assert servers["other"] == {"url": "http://o"}
    assert servers["mem-personal"]["url"] == "http://127.0.0.1:8765/personal/mcp"


def test_codex_entry_mem_cannot_edit_is_refused_and_the_file_left_alone(tmp_path):
    path = tmp_path / "config.toml"
    junk = '[mcp_servers]\nmem-personal = { command = "x", args = [\n  "a",\n] }\n'
    path.write_text(junk)
    with pytest.raises(hx.MemError, match="cannot change this file safely"):
        hx.write_config(hx.get("codex"), SPEC, "http", path, Backups(tmp_path / "b"))
    assert path.read_text() == junk


def test_adding_a_remote_to_codex_keeps_the_local_servers(tmp_path):
    from mem.setup_steps import Context, Endpoint

    token = tmp_path / "token"
    token.write_text("t\n")
    ctx = Context(runner=None, backups=Backups(tmp_path / "b"), home=tmp_path)  # type: ignore[arg-type]
    w = hx.ConfigWriter("codex", env={"CODEX_HOME": str(tmp_path / ".codex")}, platform="linux")
    w.h = hx.Harness(**{**w.h.__dict__, "skills": None})  # no skill copy in this test
    w.add(ctx, [Endpoint("mem-personal", "http://127.0.0.1:8765/personal/mcp", token)])
    w.add(ctx, [Endpoint("mem-work-personal", "https://work/personal/mcp", token)])
    servers = tomllib.loads((tmp_path / ".codex/config.toml").read_text())["mcp_servers"]
    assert set(servers) == {"mem-personal", "mem-work-personal"}


def test_backups_tighten_an_old_open_folder_and_writes_leave_no_readable_moment(
    tmp_path, umask_022
):
    root = tmp_path / "backups"
    root.mkdir(mode=0o755)
    src = tmp_path / "cfg.json"
    src.write_text("{}")
    src.chmod(0o644)
    Backups(root).save(src)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((Backups(root)._slot(src) / "data").stat().st_mode) == 0o644  # as its file
    leftover = tmp_path / ".new.json.mem"  # a temp file an interrupted run left, mode 0644
    leftover.write_text("old")
    hx._commit(tmp_path / "new.json", b"{}")
    assert stat.S_IMODE((tmp_path / "new.json").stat().st_mode) == 0o600
