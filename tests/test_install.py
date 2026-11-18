"""`mem install` (MEM2-138): every system call goes through a fake runner, HOME is a temp folder."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from mem import install as inst
from mem.auth import TokenFile
from mem.doctor import Check

REPO = Path(__file__).resolve().parents[1]
QUERIES = {("systemctl", "--user", "is-enabled"), ("systemctl", "--user", "is-active"),
           ("claude", "mcp", "get")}  # fmt: skip


class FakeRunner:
    """A machine with uv, systemctl and claude on PATH; records every command it is given."""

    def __init__(self, gpu: bool = True, models: bool = False, tools=("uv", "systemctl", "claude")):
        self.gpu_ok, self.models, self.tools = gpu, models, set(tools)
        self.calls: list[list[str]] = []
        self.enabled: set[str] = set()
        self.active: set[str] = set()
        self.mcp: dict[str, list[str]] = {}
        self.project_mcp: set[str] = set()  # a .mcp.json in the working dir (this repo's plugin)
        self.home: Path | None = None
        self.fetches = 0

    def changes(self) -> list[list[str]]:
        return [c for c in self.calls if tuple(c[:3]) not in QUERIES]

    def which(self, name: str) -> str | None:
        return f"/opt/fake/bin/{name}" if name in self.tools else None

    def gpu(self, device: str) -> Check:
        if self.gpu_ok and device != "cpu":
            return Check("device", "ok", "cuda: Fake GPU, 7.5 GiB free of 8.0")
        return Check(
            "device", "fail", "torch sees no CUDA: mem serves only on an NVIDIA GPU (CUDA)"
        )

    def missing_models(self) -> list[str]:
        return [] if self.models else ["google/embeddinggemma-300m", "BAAI/bge-reranker-v2-m3"]

    def fetch_models(self) -> list[Check]:
        self.calls.append(["<fetch models>"])
        self.fetches += 1
        self.models = True
        return [Check("fetch", "ok", "fetched")]

    def run(self, cmd: list[str], cwd: Path | None = None) -> inst.Result:
        self.calls.append(list(cmd))
        match cmd:
            case ["systemctl", "--user", "is-enabled", unit]:
                return inst.Result(0 if unit in self.enabled else 1, "")
            case ["systemctl", "--user", "is-active", unit]:
                return inst.Result(0 if unit in self.active else 3, "")
            case ["systemctl", "--user", "enable", "--now", unit]:
                self.enabled.add(unit)
                self.active.add(unit)
            case ["systemctl", "--user", "disable", "--now", unit]:
                self.enabled.discard(unit)
                self.active.discard(unit)
            case ["systemctl", "--user", ("daemon-reload" | "restart"), *_]:
                pass
            case ["claude", "mcp", "get", name]:
                if name in self.mcp:  # `claude mcp get` searches every scope, user first
                    return inst.Result(
                        0, f"{name}:\n  Scope: User config (available in all your projects)\n"
                    )
                if name in self.project_mcp and cwd != self.home:
                    return inst.Result(
                        0, f"{name}:\n  Scope: Project config (shared via .mcp.json)\n"
                    )
                return inst.Result(1, f'No MCP server named "{name}".')
            case ["claude", "mcp", "add", *args]:
                self.mcp[args[args.index("--scope") + 2]] = args
            case ["claude", "mcp", "remove", *args]:
                self.mcp.pop(args[-1], None)
            case _:
                raise AssertionError(f"unexpected command {cmd}")
        return inst.Result(0, "")


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


def _install(home: Path, runner: FakeRunner, **kw) -> tuple[int, str]:
    out: list[str] = []
    runner.home = home
    opts = inst.Options(host_label="box", **kw)
    code = inst.main(opts, runner, inst.Paths.for_home(home, env={}), echo=out.append)
    return code, "\n".join(out)


def _files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in root.rglob("*")}


def test_fresh_install_does_every_step(home):
    run = FakeRunner()
    code, out = _install(home, run)
    assert code == 0, out
    cfg = home / ".config/mem"
    assert (cfg / "config.yaml").read_text().startswith("#")
    token = (cfg / "token").read_text().strip()
    assert token.startswith("mem_")
    assert stat.S_IMODE((cfg / "token").stat().st_mode) == 0o600
    assert TokenFile(cfg / "tokens").load().verify(token) == "mem-install"
    assert run.fetches == 1
    assert run.enabled == run.active == {"mem.service"}
    unit = (home / ".config/systemd/user/mem.service").read_text()
    assert f"WorkingDirectory={REPO}\n" in unit
    assert "Environment=PATH=/opt/fake/bin:/usr/local/bin:/usr/bin:/bin\n" in unit
    assert (
        f"ExecStart=/opt/fake/bin/uv run --extra models mem serve --http --host 127.0.0.1 "
        f"--port 8765 --host-label box --config {cfg}/config.yaml --token-file {cfg}/tokens "
        f"--store personal={home}/.local/share/mem/personal.db "
        f"--store work={home}/.local/share/mem/work.db\n"
    ) in unit
    assert "%h" not in unit and "code/mem" not in unit and "--tls" not in unit
    assert sorted(run.mcp) == ["mem-personal", "mem-work"]
    assert run.mcp["mem-work"] == [
        "--transport", "http", "--scope", "user", "mem-work", "http://127.0.0.1:8765/work/mcp",
        "--header", "X-Mem-Host: box", "--header", f"Authorization: Bearer {token}",
    ]  # fmt: skip
    skill = home / ".claude/skills/mem/SKILL.md"
    assert skill.read_bytes() == (REPO / "skills/mem/SKILL.md").read_bytes()
    assert token not in out  # the token is never printed


def test_second_run_changes_nothing_and_says_so(home):
    run = FakeRunner()
    _install(home, run)
    before, run.calls = _files(home), []
    code, out = _install(home, run)
    assert code == 0
    assert run.changes() == []
    assert _files(home) == before
    assert "nothing to do" in out


def test_dry_run_prints_every_action_and_touches_nothing(home):
    run = FakeRunner()
    code, out = _install(home, run, dry_run=True)
    assert code == 0, out
    assert list(home.iterdir()) == []
    assert run.changes() == []
    for shown in ("mem doctor --fetch", "config.yaml", "tokens", "mem.service",
                  "$ systemctl --user daemon-reload", "$ systemctl --user enable --now mem.service",
                  "$ claude mcp add --transport http --scope user mem-personal",
                  "Authorization: Bearer <token", ".claude/skills/mem"):  # fmt: skip
        assert shown in out, shown


def test_uninstall_reverses_and_keeps_config_and_stores(home):
    run = FakeRunner()
    _install(home, run)
    data = home / ".local/share/mem"
    data.mkdir(parents=True)
    (data / "personal.db").write_text("store")
    code, out = _install(home, run, uninstall=True)
    assert code == 0, out
    assert run.enabled == run.active == set() and run.mcp == {}
    assert not (home / ".config/systemd/user/mem.service").exists()
    assert not (home / ".claude/skills/mem").exists()
    assert (home / ".config/mem/config.yaml").exists() and (home / ".config/mem/tokens").exists()
    assert (data / "personal.db").exists()
    assert "--purge" in out
    run.calls = []
    code, out = _install(home, run, uninstall=True)
    assert code == 0 and run.changes() == [] and "nothing to undo" in out


def test_purge_also_deletes_config_tokens_and_stores(home):
    run = FakeRunner()
    _install(home, run)
    data = home / ".local/share/mem"
    data.mkdir(parents=True)
    for name in ("personal.db", "personal.db-wal", "work.db"):
        (data / name).write_text("store")
    code, out = _install(home, run, uninstall=True, purge=True)
    assert code == 0, out
    assert not (home / ".config/mem").exists()  # emptied, so removed with its empty parents
    assert not data.exists()


def test_refuses_without_a_gpu_and_touches_nothing(home):
    run = FakeRunner(gpu=False)
    code, out = _install(home, run)
    assert code == 1
    assert "--allow-cpu" in out
    assert list(home.iterdir()) == [] and run.changes() == []


def test_allow_cpu_installs_a_cpu_service(home):
    run = FakeRunner(gpu=False)
    code, out = _install(home, run, allow_cpu=True)
    assert code == 0, out
    assert (home / ".config/mem/config.yaml").read_text().endswith("device: cpu\n")
    unit = (home / ".config/systemd/user/mem.service").read_text()
    assert " --allow-cpu\n" in unit


def test_no_mcp_leaves_claude_code_to_the_plugin(home):
    run = FakeRunner(tools=("uv", "systemctl"))
    code, out = _install(home, run, mcp=False)
    assert code == 0, out
    assert not [c for c in run.calls if c[0] == "claude"]
    assert not (home / ".claude").exists()  # the plugin brings the skill: no second copy


def test_missing_tool_fails_before_any_change(home):
    run = FakeRunner(tools=("uv", "claude"))
    code, out = _install(home, run)
    assert code == 1
    assert "systemctl" in out
    assert list(home.iterdir()) == [] and run.changes() == []


def test_a_project_mcp_json_is_not_a_user_registration(home):
    """Run from this repo, `claude mcp get mem-work` finds the plugin's .mcp.json (project
    scope): install must still register user-scope servers and never remove the project ones."""
    run = FakeRunner()
    run.project_mcp = {"mem-personal", "mem-work"}
    code, out = _install(home, run)
    assert code == 0, out
    assert sorted(run.mcp) == ["mem-personal", "mem-work"]
    assert not [c for c in run.calls if c[:3] == ["claude", "mcp", "remove"]]
    _install(home, run, uninstall=True)
    assert run.mcp == {}
    run.calls = []
    _install(home, run, uninstall=True)
    assert run.changes() == []


def test_registration_check_reads_the_exit_code_not_the_wording(home):
    """MEM2-140 M5: `claude mcp get` has no machine-readable output, so a reworded scope line must
    not make install re-register the servers or make --uninstall skip them."""

    class Reworded(FakeRunner):
        def run(self, cmd, cwd=None):
            res = super().run(cmd, cwd)
            if cmd[:3] == ["claude", "mcp", "get"] and res.code == 0:
                return inst.Result(0, f"{cmd[3]}:\n  Type: http\n  Status: connected\n")
            return res

    run = Reworded()
    _install(home, run)
    run.calls = []
    code, out = _install(home, run)
    assert code == 0 and "nothing to do" in out, out
    assert not [c for c in run.calls if c[:3] == ["claude", "mcp", "add"]]
    code, out = _install(home, run, uninstall=True)
    assert code == 0 and run.mcp == {}, out


def test_changed_unit_is_rewritten_and_restarted(home):
    run = FakeRunner()
    _install(home, run)
    run.calls = []
    code, out = _install(home, run, port=8800)
    assert code == 0, out
    assert ["systemctl", "--user", "restart", "mem.service"] in run.calls
    assert "--port 8800" in (home / ".config/systemd/user/mem.service").read_text()
