"""Harness integrations (MEM2-235): MCP config writers, native skill install, generated snippets,
detection. Expected files are in tests/golden/harnesses; HOME is always a temp folder."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mem import cli
from mem import harnesses as hx
from mem import setup_steps as ss
from mem.common import MemError

REPO = Path(__file__).resolve().parents[1]
GOLD = Path(__file__).parent / "golden" / "harnesses"
SPEC = hx.Spec(host_label="testhost", port=8765, token="TESTTOKEN", repo=Path("/opt/mem"),
               stores={"personal": Path("/data/personal.db"), "work": Path("/data/work.db")})  # fmt: skip
NAMES = ["claude-code", "codex", "gemini", "cursor", "vscode", "claude-desktop"]
CASES = [(n, t) for n in NAMES for t in ("http", "stdio") if not (n == "claude-desktop" and t == "http")]  # fmt: skip
EXT = {"codex": "toml"}


def gold(name: str) -> bytes:
    return (GOLD / name).read_bytes()


def ext(name: str) -> str:
    return EXT.get(name, "json")


def target(tmp_path: Path, name: str) -> Path:
    p = hx.config_path(hx.get(name), tmp_path / "home", {}, "linux")
    assert p is not None
    return p


def start(tmp_path: Path, name: str, original: bytes | None) -> tuple[Path, ss.Backups]:
    path = target(tmp_path, name)
    path.parent.mkdir(parents=True)
    if original is not None:
        path.write_bytes(original)
        path.chmod(0o640)
    return path, ss.Backups(tmp_path / "bk")


# ---- config paths ----------------------------------------------------------------------------


@pytest.mark.parametrize("name,rel", [
    ("claude-code", ".claude.json"),
    ("codex", ".codex/config.toml"),
    ("gemini", ".gemini/settings.json"),
    ("cursor", ".cursor/mcp.json"),
    ("vscode", ".config/Code/User/mcp.json"),
    ("claude-desktop", ".config/Claude/claude_desktop_config.json"),
])  # fmt: skip
def test_config_path_on_linux(name, rel):
    assert hx.config_path(hx.get(name), Path("/h"), {}, "linux") == Path("/h") / rel


def test_config_path_follows_xdg_config_home_and_macos_layout():
    xdg = {"XDG_CONFIG_HOME": "/x"}
    assert hx.config_path(hx.get("vscode"), Path("/h"), xdg, "linux") == Path(
        "/x/Code/User/mcp.json"
    )
    mac = Path("/h/Library/Application Support")
    assert hx.config_path(hx.get("vscode"), Path("/h"), {}, "darwin") == mac / "Code/User/mcp.json"
    assert hx.config_path(hx.get("claude-desktop"), Path("/h"), {}, "darwin") == (
        mac / "Claude/claude_desktop_config.json")  # fmt: skip


def test_generic_has_no_config_file_and_unknown_names_are_refused():
    assert hx.config_path(hx.get("generic"), Path("/h"), {}, "linux") is None
    with pytest.raises(MemError, match="unknown harness 'emacs'.*claude-code"):
        hx.get("emacs")


# ---- writers: fresh, merged, second run, uninstall ---------------------------------------------


@pytest.mark.parametrize("name,transport", CASES)
def test_fresh_config_is_the_golden_file(tmp_path, name, transport):
    path, bk = start(tmp_path, name, None)
    assert hx.write_config(hx.get(name), SPEC, transport, path, bk) == "written"
    assert path.read_bytes() == gold(f"{name}.{transport}.fresh.{ext(name)}")
    assert path.stat().st_mode & 0o777 == 0o600  # it holds the bearer token


@pytest.mark.parametrize("name,transport", CASES)
def test_merge_keeps_the_other_servers_and_settings(tmp_path, name, transport):
    path, bk = start(tmp_path, name, gold(f"{name}.other.{ext(name)}"))
    hx.write_config(hx.get(name), SPEC, transport, path, bk)
    assert path.read_bytes() == gold(f"{name}.{transport}.merged.{ext(name)}")
    assert path.stat().st_mode & 0o777 == 0o640  # an existing file keeps its mode


@pytest.mark.parametrize("name,transport", CASES)
def test_second_run_changes_nothing(tmp_path, name, transport):
    path, bk = start(tmp_path, name, gold(f"{name}.other.{ext(name)}"))
    hx.write_config(hx.get(name), SPEC, transport, path, bk)
    first = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    assert hx.write_config(hx.get(name), SPEC, transport, path, bk) == "unchanged"
    assert path.read_bytes() == first and path.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("name,transport", CASES)
@pytest.mark.parametrize("existed", [True, False])
def test_uninstall_restores_the_original_byte_for_byte(tmp_path, name, transport, existed):
    original = gold(f"{name}.other.{ext(name)}") if existed else None
    path, bk = start(tmp_path, name, original)
    hx.write_config(hx.get(name), SPEC, transport, path, bk)
    hx.write_config(hx.get(name), SPEC, transport, path, bk)  # a second run must not lose it
    assert hx.remove_config(hx.get(name), SPEC, transport, path, bk) == "restored"
    if existed:
        assert path.read_bytes() == original and path.stat().st_mode & 0o777 == 0o640
    else:
        assert not path.exists()


@pytest.mark.parametrize("name", ["claude-code", "codex"])
def test_uninstall_after_the_user_edited_the_file_takes_out_only_ours(tmp_path, name):
    path, bk = start(tmp_path, name, gold(f"{name}.other.{ext(name)}"))
    hx.write_config(hx.get(name), SPEC, "http", path, bk)
    if name == "codex":
        path.write_text(path.read_text().replace('model = "gpt-5"', 'model = "gpt-6"'))
    else:
        data = json.loads(path.read_text())
        data["numStartups"] = 7
        path.write_text(json.dumps(data, indent=2) + "\n")
    assert hx.remove_config(hx.get(name), SPEC, "http", path, bk) == "removed"
    text = path.read_text()
    assert "mem-personal" not in text and "mem-work" not in text
    assert "other" in text and ("gpt-6" in text or "numStartups" in text)
    assert not bk.has(path)


def test_remove_when_nothing_was_written_is_absent(tmp_path):
    path, bk = start(tmp_path, "claude-code", gold("claude-code.other.json"))
    assert hx.remove_config(hx.get("claude-code"), SPEC, "http", path, bk) == "absent"
    assert path.read_bytes() == gold("claude-code.other.json")


def test_a_changed_token_replaces_our_entries_and_nothing_else(tmp_path):
    path, bk = start(tmp_path, "gemini", gold("gemini.other.json"))
    hx.write_config(hx.get("gemini"), SPEC, "http", path, bk)
    new = hx.Spec(SPEC.host_label, SPEC.port, "NEWTOKEN", SPEC.repo, SPEC.stores)
    assert hx.write_config(hx.get("gemini"), new, "http", path, bk) == "written"
    servers = json.loads(path.read_text())["mcpServers"]
    assert servers["mem-work"]["headers"]["Authorization"] == "Bearer NEWTOKEN"
    assert servers["other"] == {"command": "npx", "args": ["-y", "x"]}
    assert bk.has(path)  # the first backup (the true original) is still the one kept


def test_no_token_means_no_authorization_header(tmp_path):
    path, bk = start(tmp_path, "cursor", None)
    spec = hx.Spec("testhost", 8765, None, SPEC.repo, SPEC.stores)
    hx.write_config(hx.get("cursor"), spec, "http", path, bk)
    assert json.loads(path.read_text())["mcpServers"]["mem-work"]["headers"] == {
        "X-Mem-Host": "testhost"}  # fmt: skip


@pytest.mark.parametrize(
    "name,junk", [("claude-code", b'{"mcpServers": '), ("codex", b"[mcp_servers\n")]
)
def test_a_config_that_does_not_parse_is_left_alone_with_a_clear_error(tmp_path, name, junk):
    path, bk = start(tmp_path, name, junk)
    with pytest.raises(MemError, match="does not parse"):
        hx.write_config(hx.get(name), SPEC, "http", path, bk)
    assert path.read_bytes() == junk and not bk.has(path)


def test_codex_toml_written_is_valid_toml_with_the_right_tables(tmp_path):
    path, bk = start(tmp_path, "codex", gold("codex.other.toml"))
    hx.write_config(hx.get("codex"), SPEC, "http", path, bk)
    servers = tomllib.loads(path.read_text())["mcp_servers"]
    assert set(servers) == {"other", "mem-personal", "mem-work"}
    assert servers["mem-work"]["http_headers"]["Authorization"] == "Bearer TESTTOKEN"


def test_claude_desktop_cannot_use_http_from_its_config_file(tmp_path):
    path, bk = start(tmp_path, "claude-desktop", None)
    with pytest.raises(MemError, match="stdio"):
        hx.write_config(hx.get("claude-desktop"), SPEC, "http", path, bk)


def test_stdio_needs_a_store_path_per_server(tmp_path):
    path, bk = start(tmp_path, "cursor", None)
    spec = hx.Spec("testhost", 8765, None, SPEC.repo, {})
    with pytest.raises(MemError, match="stores"):
        hx.write_config(hx.get("cursor"), spec, "stdio", path, bk)


# ---- print forms -------------------------------------------------------------------------------


def test_print_hides_the_token_and_matches_what_is_written(tmp_path):
    text = hx.render_config(hx.get("claude-code"), SPEC, "http", redact=True)
    assert "TESTTOKEN" not in text and "<token>" in text
    shown = json.loads(text)["mcpServers"]["mem-work"]
    written = json.loads(gold("claude-code.http.fresh.json"))["mcpServers"]["mem-work"]
    assert shown["url"] == written["url"] and shown["type"] == written["type"]


def test_generic_prints_the_stdio_and_the_http_forms():
    text = hx.render_generic(SPEC, redact=True)
    assert "http://127.0.0.1:8765/personal/mcp" in text
    assert "Authorization" in text and "<token>" in text and "TESTTOKEN" not in text
    assert '"command": "uv"' in text and "personal=/data/personal.db" in text
    assert text.index("HTTP") < text.index("stdio")


# ---- native skills -----------------------------------------------------------------------------


@pytest.mark.parametrize("name,rel", [
    ("claude-code", ".claude/skills/mem"),
    ("codex", ".agents/skills/mem"),
    ("gemini", ".agents/skills/mem"),
    ("cursor", ".agents/skills/mem"),
    ("vscode", ".agents/skills/mem"),
])  # fmt: skip
def test_skill_goes_to_the_directory_the_harness_loads(name, rel):
    assert hx.skill_dir(hx.get(name), Path("/h")) == Path("/h") / rel


def test_claude_desktop_has_no_skill_directory_and_uses_pasted_text():
    h = hx.get("claude-desktop")
    assert hx.skill_dir(h, Path("/h")) is None and h.native_skill is False


def test_install_skill_copies_the_one_skill_file_and_is_idempotent(tmp_path):
    src = REPO / "skills" / "mem"
    assert hx.install_skill(hx.get("codex"), tmp_path, src) == "installed"
    dest = tmp_path / ".agents" / "skills" / "mem"
    assert (dest / "SKILL.md").read_bytes() == (src / "SKILL.md").read_bytes()
    assert hx.install_skill(hx.get("codex"), tmp_path, src) == "unchanged"


def test_install_skill_replaces_an_older_copy(tmp_path):
    dest = tmp_path / ".claude" / "skills" / "mem"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text("old")
    assert hx.install_skill(hx.get("claude-code"), tmp_path, REPO / "skills" / "mem") == "installed"
    assert (dest / "SKILL.md").read_bytes() == (REPO / "skills" / "mem" / "SKILL.md").read_bytes()


def test_remove_skill_deletes_ours_and_leaves_other_skills(tmp_path):
    hx.install_skill(hx.get("codex"), tmp_path, REPO / "skills" / "mem")
    other = tmp_path / ".agents" / "skills" / "other"
    other.mkdir()
    (other / "SKILL.md").write_text("x")
    assert hx.remove_skill(hx.get("codex"), tmp_path) == "removed"
    assert not (tmp_path / ".agents" / "skills" / "mem").exists() and other.exists()
    assert hx.remove_skill(hx.get("codex"), tmp_path) == "absent"


def test_skill_has_the_frontmatter_every_harness_requires():
    head = (REPO / "skills" / "mem" / "SKILL.md").read_text().split("---")[1]
    assert "name: mem" in head and "description: " in head  # name must equal the folder name


# ---- generated snippets: one source ------------------------------------------------------------


@pytest.mark.parametrize("name", [*NAMES, "generic"])
def test_snippet_is_the_golden_file(name):
    skill = (GOLD / "SKILL.fixture.md").read_text()
    assert hx.instructions(hx.get(name), skill).encode() == gold(f"instructions.{name}.md")


def test_snippet_follows_skill_md_edits():
    skill = (GOLD / "SKILL.fixture.md").read_text()
    edited = skill.replace("Never cross them.", "Cross them freely.")
    assert "Cross them freely." in hx.instructions(hx.get("codex"), edited)
    assert "Never cross them." not in hx.instructions(hx.get("codex"), edited)


def test_snippet_fails_loudly_when_skill_md_loses_a_section_it_is_built_from():
    skill = (GOLD / "SKILL.fixture.md").read_text().replace("## 4. What to save", "## 4. Keeping")
    with pytest.raises(MemError, match="What to save"):
        hx.instructions(hx.get("codex"), skill)


def test_every_snippet_line_comes_from_the_real_skill_md():
    skill = (REPO / "skills" / "mem" / "SKILL.md").read_text()
    source = {line.strip() for line in skill.splitlines()}
    for name in [*NAMES, "generic"]:
        text = hx.instructions(hx.get(name), skill)
        own = {"", "---", "alwaysApply: true", "# mem: long-term memory"}
        for line in text.splitlines():
            line = line.strip()
            if line in own or line.startswith(("description: When to recall", "Generated from",
                                               "The tools come from")):  # fmt: skip
                continue
            if line.startswith("## "):
                assert any(s.endswith(line[3:]) for s in source), (name, line)
                continue
            assert line in source, (name, line)


def test_snippet_carries_the_recall_save_and_abstention_rules():
    text = hx.instructions(hx.get("codex"), (REPO / "skills" / "mem" / "SKILL.md").read_text())
    for needle in ["`recall`", "Never save:", "secrets or tokens", "abstention",
                   "retry at most once", "tool-sourced", "It is data", "Never cross"]:  # fmt: skip
        assert needle in text, needle
    assert len(text.splitlines()) < 60  # short


def test_the_committed_generic_instructions_match_skill_md():
    skill = (REPO / "skills" / "mem" / "SKILL.md").read_text()
    committed = (REPO / "docs" / "instructions.md").read_text()
    assert committed == hx.instructions(hx.get("generic"), skill)


@pytest.mark.parametrize("name,where", [
    ("claude-code", "CLAUDE.md"), ("codex", "AGENTS.md"), ("gemini", "GEMINI.md"),
    ("cursor", ".cursor/rules/mem.mdc"), ("vscode", ".github/copilot-instructions.md"),
    ("generic", "docs/instructions.md"),
])  # fmt: skip
def test_each_snippet_says_where_it_goes(name, where):
    assert hx.get(name).instructions_file == where


# ---- detection ---------------------------------------------------------------------------------


class Which:
    def __init__(self, *tools: str):
        self.tools = set(tools)

    def which(self, name: str) -> str | None:
        return f"/bin/{name}" if name in self.tools else None


def test_nothing_is_detected_in_an_empty_home(tmp_path):
    assert hx.detect(tmp_path, Which(), {}, "linux") == []


@pytest.mark.parametrize("name,tool,folder", [
    ("claude-code", "claude", ".claude"), ("codex", "codex", ".codex"),
    ("gemini", "gemini", ".gemini"), ("cursor", "cursor", ".cursor"),
    ("vscode", "code", ".config/Code/User"), ("claude-desktop", None, ".config/Claude"),
])  # fmt: skip
def test_a_harness_is_detected_by_its_command_or_its_config_folder(tmp_path, name, tool, folder):
    if tool:
        assert hx.detect(tmp_path, Which(tool), {}, "linux") == [name]
    (tmp_path / folder).mkdir(parents=True)
    assert hx.detect(tmp_path, Which(), {}, "linux") == [name]


def test_detection_lists_every_installed_harness_in_a_fixed_order(tmp_path):
    (tmp_path / ".codex").mkdir()
    found = hx.detect(tmp_path, Which("claude", "gemini"), {}, "linux")
    assert found == ["claude-code", "codex", "gemini"]


# ---- the commands ------------------------------------------------------------------------------


def test_mcp_config_prints_by_default_and_writes_nothing(tmp_path):
    r = CliRunner().invoke(cli.app, ["mcp-config", "--harness", "cursor", "--home", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "mem-personal" in r.output and "<token>" in r.output
    assert not (tmp_path / ".cursor").exists()


def test_mcp_config_write_creates_the_file_and_a_backup_slot(tmp_path):
    token = tmp_path / ".config" / "mem" / "token"
    token.parent.mkdir(parents=True)
    token.write_text("SECRET1\n")
    args = ["mcp-config", "--harness", "cursor", "--write", "--home", str(tmp_path),
            "--host-label", "testhost"]  # fmt: skip
    r = CliRunner().invoke(cli.app, args, env={"XDG_CONFIG_HOME": "", "XDG_DATA_HOME": ""})
    assert r.exit_code == 0, r.output
    written = json.loads((tmp_path / ".cursor" / "mcp.json").read_text())
    assert written["mcpServers"]["mem-work"]["headers"]["Authorization"] == "Bearer SECRET1"
    assert "SECRET1" not in r.output
    again = CliRunner().invoke(cli.app, args, env={"XDG_CONFIG_HOME": "", "XDG_DATA_HOME": ""})
    assert "unchanged" in again.output


def test_mcp_config_uninstall_restores(tmp_path):
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    cfg.write_bytes(gold("cursor.other.json"))
    base = ["mcp-config", "--harness", "cursor", "--home", str(tmp_path)]
    env = {"XDG_CONFIG_HOME": "", "XDG_DATA_HOME": ""}
    assert CliRunner().invoke(cli.app, [*base, "--write"], env=env).exit_code == 0
    assert cfg.read_bytes() != gold("cursor.other.json")
    assert CliRunner().invoke(cli.app, [*base, "--uninstall"], env=env).exit_code == 0
    assert cfg.read_bytes() == gold("cursor.other.json")


def test_mcp_config_generic_prints_both_forms_and_refuses_write(tmp_path):
    r = CliRunner().invoke(cli.app, ["mcp-config", "--harness", "generic", "--home", str(tmp_path)])
    assert r.exit_code == 0 and "stdio" in r.output and "http://127.0.0.1" in r.output
    w = CliRunner().invoke(cli.app, ["mcp-config", "--harness", "generic", "--write"])
    assert w.exit_code != 0 and "print" in w.output


def test_mcp_config_rejects_an_unknown_harness_naming_the_valid_ones():
    r = CliRunner().invoke(cli.app, ["mcp-config", "--harness", "emacs"])
    assert r.exit_code != 0 and "claude-code" in r.output


def test_instructions_command_prints_the_snippet():
    r = CliRunner().invoke(cli.app, ["instructions", "--harness", "cursor"])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("---\ndescription: When to recall")
    assert "## Recall first" in r.output


def test_harnesses_doc_has_a_section_per_harness_with_a_url_and_a_date():
    doc = (REPO / "docs" / "harnesses.md").read_text()
    for title in ["Claude Code", "Codex CLI", "Gemini CLI", "Cursor", "VS Code", "Claude Desktop",
                  "claude.ai"]:  # fmt: skip
        assert f"\n## {title}" in doc, title
    for title in ["Claude Code", "Codex CLI", "Gemini CLI", "Cursor", "VS Code", "Claude Desktop",
                  "claude.ai"]:  # fmt: skip
        sec = doc.split(f"\n## {title}")[1].split("\n## ")[0]
        assert "https://" in sec and "2026-10-26" in sec, title
    assert "unsupported" in doc.split("\n## claude.ai")[1].lower()


# ---- the writers `mem setup` drives (MEM2-236) ----------------------------------------------------


def _writer_world(tmp_path):
    from mem import setup_steps as ss

    class Runner:
        def which(self, name):
            return None

    token = tmp_path / "token"
    token.write_text("tok\n")
    ctx = ss.Context(Runner(), ss.Backups(tmp_path / "backups"), tmp_path)
    eps = [ss.Endpoint(f"mem-{n}", f"http://127.0.0.1:8765/{n}/mcp", token, (("X-Mem-Host", "h"),))
           for n in ("personal", "work")]  # fmt: skip
    return ctx, eps


def test_the_setup_writers_cover_every_harness_with_an_http_config_file():
    assert [w.h.name for w in hx.writers()] == ["codex", "gemini", "cursor", "vscode"]


def test_a_config_writer_adds_once_and_removes_byte_for_byte(tmp_path):
    ctx, eps = _writer_world(tmp_path)
    cfg = tmp_path / ".cursor" / "mcp.json"
    cfg.parent.mkdir()
    original = '{"mcpServers": {"other": {"url": "http://x"}}}'
    cfg.write_text(original)
    w = hx.ConfigWriter("cursor", env={})
    assert w.detect(ctx) and not w.configured(ctx, eps)
    w.add(ctx, eps)
    w.add(ctx, eps)
    assert w.configured(ctx, eps)
    assert set(json.loads(cfg.read_text())["mcpServers"]) == {"other", "mem-personal", "mem-work"}
    w.remove(ctx)
    assert cfg.read_text() == original


def test_a_config_writer_is_not_configured_before_the_token_exists(tmp_path):
    ctx, eps = _writer_world(tmp_path)
    eps[0].token_file.unlink()
    assert hx.ConfigWriter("gemini", env={}).configured(ctx, eps) is False
