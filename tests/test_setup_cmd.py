"""`mem setup` (MEM2-234): every system call goes through a fake runner, HOME is a temp folder."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mem import cli
from mem import install as inst
from mem import setup_cmd as sc
from mem import setup_steps as ss
from mem.common import MemError
from mem.doctor import Check
from tests.test_install import FakeRunner

GOOD = "mem_goodtoken"
EMBED_URL = "https://huggingface.co/google/embeddinggemma-300m"


class SetupFake(FakeRunner):
    """FakeRunner plus what `mem setup` asks beyond `mem install`."""

    def __init__(self, *, port_busy=False, systemd=True, driver_cuda="12.4", torch_cuda="12.4",
                 peer=None, **kw):  # fmt: skip
        super().__init__(**kw)
        self.port_busy, self.systemd = port_busy, systemd
        self.driver_cuda, self.torch_cuda = driver_cuda, torch_cuda
        self.peer = peer if peer is not None else {}  # host -> (token it accepts, ms)
        self.selftests = 0
        self.doctors = 0
        self.fetch_error: str | None = None

    def port_free(self, port: int) -> bool:
        return not self.port_busy

    def cuda_versions(self) -> tuple[str | None, str | None]:
        return self.driver_cuda, self.torch_cuda

    def fetch_models(self) -> list[Check]:
        if self.fetch_error:
            return [Check("fetch google/embeddinggemma-300m", "fail", self.fetch_error)]
        return super().fetch_models()

    def doctor(self, stores, device, cfg=None) -> list[Check]:
        self.doctors += 1
        return [Check("device", "ok", "cuda: Fake GPU")]

    def selftest(self, url: str, token: str, host_label: str) -> list[tuple[str, bool, str]]:
        self.selftests += 1
        return [("service", True, "up"), ("remember", True, "saved m1"),
                ("recall", True, "1 result"), ("forget", True, "retracted")]  # fmt: skip

    def probe(self, url: str, token: str) -> inst.Probe:
        host = url.split("/")[2]
        if host not in self.peer:
            return inst.Probe(None, 0.0, "name does not resolve")
        want, ms = self.peer[host]
        return inst.Probe(200 if token == want else 401, ms, "")

    def run(self, cmd, cwd=None):
        if cmd[:3] == ["systemctl", "--user", "show-environment"]:
            self.calls.append(list(cmd))
            return inst.Result(0 if self.systemd else 1, "" if self.systemd else "no bus")
        return super().run(cmd, cwd)


class FileWriter:
    """A harness whose config is one JSON file that setup edits (through Backups)."""

    name = "Filey"

    def __init__(self, path: Path):
        self.path = path

    def detect(self, ctx):
        return self.path.exists()

    def configured(self, ctx, endpoints):
        return all(e.name in self.path.read_text() for e in endpoints)

    def add(self, ctx, endpoints):
        ctx.backups.save(self.path)
        data = json.loads(self.path.read_text())
        data.setdefault("mcpServers", {}).update({e.name: {"url": e.url} for e in endpoints})
        self.path.write_text(json.dumps(data, indent=1))

    def installed(self, ctx):
        return ctx.backups.has(self.path)

    def remove(self, ctx):
        ctx.backups.restore(self.path)


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


def _setup(home, run, *, writers=(), answers=(), secret="", **kw):
    out: list[str] = []
    run.home = home
    opts = sc.SetupOptions(host_label="box", **kw)
    asked = iter(answers)
    code = sc.main(opts, run, inst.Paths.for_home(home, env={}), echo=out.append,
                   ask=lambda prompt: next(asked), ask_secret=lambda prompt: secret,
                   writers=list(writers))  # fmt: skip
    return code, "\n".join(out)


def _tree(root: Path) -> dict[str, bytes | None]:
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}  # fmt: skip


def test_fresh_setup_runs_every_stage_in_order_and_prints_pass(home):
    run = SetupFake()
    code, out = _setup(home, run, yes=True)
    assert code == 0, out
    stages = ["gpu", "models", "config", "token", "service", "mcp", "skill", "doctor", "self-test"]
    at = [out.index(f"] {s}:") for s in stages]
    assert at == sorted(at)
    assert "self-test: PASS" in out
    assert run.enabled == {"mem.service"} and run.doctors == 1 and run.selftests == 1
    assert (home / ".config/mem/token").exists()


def test_failing_self_test_prints_fail_and_exits_1(home):
    run = SetupFake()
    run.selftest = lambda url, token, host: [
        ("remember", True, "saved"),
        ("recall", False, "0 results"),
    ]
    code, out = _setup(home, run, yes=True)
    assert code == 1
    assert "self-test: FAIL" in out and "recall" in out and "0 results" in out


def test_second_run_is_a_no_op(home):
    run = SetupFake()
    _setup(home, run, yes=True)
    run.calls = []
    code, out = _setup(home, run, yes=True)
    assert code == 0, out
    assert run.changes() == [["systemctl", "--user", "show-environment"]] or not [
        c for c in run.changes() if c[:3] != ["systemctl", "--user", "show-environment"]
    ]
    assert "nothing to do" in out and "self-test: PASS" in out  # the check still runs


def test_dry_run_lists_every_stage_and_touches_nothing(home):
    run = SetupFake()
    code, out = _setup(home, run, dry_run=True)
    assert code == 0, out
    assert list(home.iterdir()) == []
    assert [c for c in run.changes() if c[:3] != ["systemctl", "--user", "show-environment"]] == []
    assert run.fetches == run.doctors == run.selftests == 0
    for word in ["models", "config", "token", "service", "mcp", "skill", "doctor", "self-test"]:
        assert f"[would] {word}" in out


def test_interactive_declined_changes_nothing(home):
    run = SetupFake()
    code, out = _setup(home, run, answers=["n"])
    assert code == 1 and "nothing was changed" in out and list(home.iterdir()) == []


def test_interactive_accepted_installs(home):
    run = SetupFake()
    code, _out = _setup(home, run, answers=[""])  # Enter = yes
    assert code == 0 and run.enabled == {"mem.service"}


def test_uninstall_restores_each_edited_config_byte_for_byte(home):
    cfg = home / ".filey.json"
    original = b'{"mcpServers":   {"other": {"url": "http://x"}},\n "theme": "dark"}'
    cfg.write_bytes(original)
    cfg.chmod(0o640)
    run = SetupFake()
    _setup(home, run, yes=True, writers=[FileWriter(cfg)])
    assert b"mem-personal" in cfg.read_bytes() and cfg.read_bytes() != original
    code, out = _setup(home, run, yes=True, uninstall=True, writers=[FileWriter(cfg)])
    assert code == 0, out
    assert cfg.read_bytes() == original and stat.S_IMODE(cfg.stat().st_mode) == 0o640
    assert not (home / ".config/mem/backups").exists() or not any(
        (home / ".config/mem/backups").iterdir()
    )


def test_uninstall_deletes_a_config_that_setup_created(home):
    cfg = home / ".filey.json"
    cfg.write_text("{}")  # detect() needs it to exist, so test the 'did not exist' path via Backups
    ctx = ss.Context(runner=SetupFake(), backups=ss.Backups(home / "bk"), home=home)
    new = home / "new.json"
    ctx.backups.save(new)
    new.write_text("made by setup")
    ctx.backups.restore(new)
    assert not new.exists()


def test_uninstall_keeps_config_tokens_and_stores_but_purge_deletes(home):
    run = SetupFake()
    _setup(home, run, yes=True)
    (home / ".local/share/mem").mkdir(parents=True)
    (home / ".local/share/mem/personal.db").write_text("data")
    code, out = _setup(home, run, yes=True, uninstall=True)
    assert code == 0, out
    assert (home / ".config/mem/config.yaml").exists()
    assert (home / ".local/share/mem/personal.db").read_text() == "data"
    assert run.enabled == set() and run.mcp == {}
    code, out = _setup(home, run, yes=True, uninstall=True, purge=True)
    assert code == 0, out
    assert not (home / ".config/mem/config.yaml").exists()
    assert not (home / ".local/share/mem/personal.db").exists()


def test_uninstall_dry_run_touches_nothing(home):
    run = SetupFake()
    _setup(home, run, yes=True)
    before = _tree(home)
    code, _out = _setup(home, run, dry_run=True, uninstall=True, purge=True)
    assert code == 0 and _tree(home) == before and run.enabled == {"mem.service"}


# ---- errors: each says what is wrong and how to fix it ----------------------------------------


def test_no_gpu_error(home):
    run = SetupFake(gpu=False, driver_cuda=None, torch_cuda=None)
    code, out = _setup(home, run, yes=True)
    assert code == 1 and list(home.iterdir()) == []
    assert "no NVIDIA GPU" in out and "Fix:" in out and "--allow-cpu" in out


def test_cuda_mismatch_error(home):
    run = SetupFake(gpu=False, driver_cuda="11.8", torch_cuda="12.4")
    code, out = _setup(home, run, yes=True)
    assert code == 1 and list(home.iterdir()) == []
    assert "driver supports CUDA 11.8" in out and "torch was built for CUDA 12.4" in out
    assert "Fix:" in out


def test_gated_model_error_has_the_exact_url_and_does_not_retry(home):
    run = SetupFake()
    run.fetch_error = "GatedRepoError: 401 Client Error. Cannot access gated repo"
    code, out = _setup(home, run, yes=True)
    assert code == 1
    assert EMBED_URL in out and "hf auth login" in out and "Fix:" in out


def test_port_in_use_error(home):
    run = SetupFake(port_busy=True)
    code, out = _setup(home, run, yes=True)
    assert code == 1 and list(home.iterdir()) == []
    assert "port 8765 is in use" in out and "--port" in out and "Fix:" in out


def test_port_in_use_by_our_own_running_service_is_fine(home):
    run = SetupFake()
    _setup(home, run, yes=True)
    run.port_busy = True
    code, out = _setup(home, run, yes=True)
    assert code == 0, out


def test_no_systemd_user_session_error(home):
    run = SetupFake(systemd=False)
    code, out = _setup(home, run, yes=True)
    assert code == 1 and list(home.iterdir()) == []
    assert "no systemd user session" in out and "loginctl enable-linger" in out


def test_failed_doctor_stops_with_exit_1(home):
    run = SetupFake()
    run.doctor = lambda stores, device, cfg: [Check("sqlite-vec", "fail", "not loadable")]
    code, out = _setup(home, run, yes=True)
    assert code == 1 and "sqlite-vec" in out and "self-test" not in out.split("doctor")[-1]


# ---- the download: progress, resume -----------------------------------------------------------


class FakeHub:
    """A download that dies once partway, and resumes from the bytes already on disk."""

    def __init__(self, size=100, die_at=60, error: Exception | None = None):
        self.size, self.die_at, self.error = size, die_at, error
        self.have: dict[str, int] = {}
        self.started_from: list[int] = []

    def __call__(self, repo, rev, progress):
        if self.error:
            raise self.error
        start = self.have.get(repo, 0)
        self.started_from.append(start)
        for done in range(start, self.size + 1, 20):
            progress(done, self.size)
            self.have[repo] = done
            if self.die_at is not None and done >= self.die_at:
                self.die_at = None
                raise ConnectionError("connection reset")
        return f"/cache/{repo}"


def test_interrupted_download_resumes_from_the_bytes_already_there():
    hub, out = FakeHub(), []
    path = sc.download(hub, "google/embeddinggemma-300m", "rev", echo=out.append, retries=2)
    assert path == "/cache/google/embeddinggemma-300m"
    assert hub.started_from == [0, 60]
    text = "\n".join(out)
    assert "100%" in text and "interrupted" in text and "resuming" in text


def test_download_gives_up_after_the_retries_with_a_fix():
    hub = FakeHub(die_at=0)
    hub.die_at = 0

    def always_dies(repo, rev, progress):
        raise ConnectionError("offline")

    with pytest.raises(MemError, match="offline.*rerun `mem setup`"):
        sc.download(always_dies, "r", "v", echo=lambda s: None, retries=1)


def test_gated_error_is_not_retried():
    calls = []

    def gated(repo, rev, progress):
        calls.append(1)
        raise PermissionError("GatedRepoError")

    with pytest.raises(MemError, match=EMBED_URL):
        sc.download(gated, "google/embeddinggemma-300m", "v", echo=lambda s: None, retries=3)
    assert len(calls) == 1


# ---- --remote ---------------------------------------------------------------------------------

REMOTE = "work.tail1234.ts.net"


def test_remote_with_a_code_registers_the_peers_servers_and_stores_the_token(home):
    run = SetupFake(peer={REMOTE: (GOOD, 12.0)})
    _setup(home, run, yes=True)
    run.calls = []
    code, out = _setup(home, run, yes=True, remote=REMOTE, code=sc.make_code(REMOTE, GOOD))
    assert code == 0, out
    assert sorted(run.mcp) == ["mem-personal", "mem-work", "mem-work-personal", "mem-work-work"]
    assert run.mcp["mem-work-work"] == [
        "--transport", "http", "--scope", "user", "mem-work-work", f"https://{REMOTE}/work/mcp",
        "--header", f"Authorization: Bearer {GOOD}",
    ]  # fmt: skip
    token_file = home / ".config/mem/remote-work.token"
    assert token_file.read_text().strip() == GOOD
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert GOOD not in out
    remotes = json.loads((home / ".config/mem/remotes.json").read_text())
    assert remotes == {"work": {"host": REMOTE, "url": f"https://{REMOTE}"}}


def test_remote_with_a_pasted_token(home):
    run = SetupFake(peer={REMOTE: (GOOD, 5.0)})
    code, out = _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)
    assert code == 0, out
    assert "mem-work-personal" in run.mcp


def test_remote_is_idempotent(home):
    run = SetupFake(peer={REMOTE: (GOOD, 5.0)})
    _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)
    run.calls = []
    code, _out = _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)
    assert code == 0 and not [c for c in run.calls if c[:3] == ["claude", "mcp", "add"]]


def test_remote_refused_token_changes_nothing_and_says_how_to_reissue(home):
    run = SetupFake(peer={REMOTE: ("mem_other", 5.0)})
    code, out = _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)
    assert code == 1 and run.mcp == {} and not (home / ".config/mem/remotes.json").exists()
    assert "401" in out and "mem token add" in out and "Fix:" in out


def test_remote_unreachable_names_the_host_and_the_fix(home):
    run = SetupFake(peer={})
    code, out = _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)
    assert code == 1 and run.mcp == {}
    assert REMOTE in out and "tailscale" in out and "Fix:" in out


def test_remote_dry_run_touches_nothing(home):
    run = SetupFake(peer={REMOTE: (GOOD, 5.0)})
    code, out = _setup(home, run, remote=REMOTE, dry_run=True, secret=GOOD)
    assert code == 0 and run.mcp == {} and list(home.iterdir()) == [] and "[would]" in out


def test_remote_is_written_to_other_harnesses_and_restored_on_uninstall(home):
    cfg = home / ".filey.json"
    original = b'{"mcpServers": {}}'
    cfg.write_bytes(original)
    run = SetupFake(peer={REMOTE: (GOOD, 5.0)})
    _setup(home, run, remote=REMOTE, yes=True, secret=GOOD, writers=[FileWriter(cfg)])
    assert f"https://{REMOTE}/work/mcp" in cfg.read_text()
    _setup(home, run, yes=True, uninstall=True, writers=[FileWriter(cfg)])
    assert cfg.read_bytes() == original and run.mcp == {}


def test_bad_code_is_a_clear_error(home):
    run = SetupFake(peer={REMOTE: (GOOD, 5.0)})
    code, out = _setup(home, run, remote=REMOTE, code="memc1_garbage", yes=True)
    assert code == 1 and "not a mem code" in out


def test_code_round_trips():
    assert sc.read_code(sc.make_code("h.ts.net", "mem_abc")) == ("h.ts.net", "mem_abc")


# ---- mem doctor on a remote -------------------------------------------------------------------


def _remember_remote(home, run):
    _setup(home, run, remote=REMOTE, yes=True, secret=GOOD)


def test_doctor_reports_remote_reachability_token_and_latency(home):
    run = SetupFake(peer={REMOTE: (GOOD, 12.0)})
    _remember_remote(home, run)
    checks = sc.check_remotes(inst.Paths.for_home(home, env={}), run)
    assert [(c.name, c.status) for c in checks] == [("remote work", "ok")]
    assert "12 ms" in checks[0].detail


def test_doctor_warns_on_slow_latency_and_fails_on_a_refused_token(home):
    run = SetupFake(peer={REMOTE: (GOOD, 900.0)})
    _remember_remote(home, run)
    paths = inst.Paths.for_home(home, env={})
    assert sc.check_remotes(paths, run)[0].status == "warn"
    run.peer = {REMOTE: ("mem_new", 5.0)}
    c = sc.check_remotes(paths, run)[0]
    assert c.status == "fail" and "401" in c.detail and "mem token add" in c.detail
    run.peer = {}
    c = sc.check_remotes(paths, run)[0]
    assert c.status == "fail" and "reach" in c.detail


def test_doctor_with_no_remotes_reports_nothing(home):
    assert sc.check_remotes(inst.Paths.for_home(home, env={}), SetupFake()) == []


# ---- cli --------------------------------------------------------------------------------------


def test_install_is_an_alias_of_setup_yes(home, monkeypatch):
    run = SetupFake()
    monkeypatch.setattr(inst, "SystemRunner", lambda: run)
    monkeypatch.setattr(inst.Paths, "here", classmethod(lambda c: c.for_home(home, env={})))
    res = CliRunner().invoke(cli.app, ["install", "--host-label", "box"])
    assert res.exit_code == 0, res.output
    assert "self-test: PASS" in res.output and run.enabled == {"mem.service"}


def test_cli_setup_wires_flags(home, monkeypatch):
    run = SetupFake()
    monkeypatch.setattr(inst, "SystemRunner", lambda: run)
    monkeypatch.setattr(inst.Paths, "here", classmethod(lambda c: c.for_home(home, env={})))
    res = CliRunner().invoke(cli.app, ["setup", "--dry-run", "--host-label", "box"])
    assert res.exit_code == 0 and "dry run" in res.output and list(home.iterdir()) == []
    res = CliRunner().invoke(cli.app, ["setup", "--purge"])
    assert res.exit_code != 0 and "--uninstall" in res.output
    res = CliRunner().invoke(cli.app, ["setup", "--code", "x"])
    assert res.exit_code != 0 and "--remote" in res.output


def test_token_add_code_prints_a_code_for_the_other_machine(tmp_path):
    res = CliRunner().invoke(cli.app, ["token", "add", "laptop", "--code", "--host", REMOTE,
                                       "--file", str(tmp_path / "tokens")])  # fmt: skip
    assert res.exit_code == 0, res.output
    code = res.stdout.strip().splitlines()[-1]
    host, token = sc.read_code(code)
    assert host == REMOTE and token.startswith("mem_")


# ---- `uv tool install` / `uvx`: no checkout, the files come from the wheel ----------------------


def test_setup_from_an_installed_wheel_uses_the_packaged_files_and_the_mem_binary(
    home, tmp_path, monkeypatch
):
    import shutil

    data = tmp_path / "site-packages/mem/_data"
    shutil.copytree(inst.REPO / "deploy", data / "deploy", ignore=shutil.ignore_patterns("*.timer"))
    shutil.copytree(inst.REPO / "skills", data / "skills")
    monkeypatch.setattr(inst, "DATA", data)
    paths = inst.Paths.for_home(home, env={})
    paths = type(paths)(**{**paths.__dict__, "repo": tmp_path / "site-packages"})
    out: list[str] = []
    run = SetupFake(tools=("uv", "systemctl", "claude", "mem"))
    run.home = home
    code = sc.main(sc.SetupOptions(host_label="box", yes=True), run, paths, echo=out.append)
    assert code == 0, "\n".join(out)
    unit = (home / ".config/systemd/user/mem.service").read_text()
    assert "ExecStart=/opt/fake/bin/mem serve --http" in unit
    assert "uv run" not in unit and f"WorkingDirectory={home}\n" in unit
    assert (home / ".claude/skills/mem/SKILL.md").exists()
