"""MEM2-242: the MEM2-237 review's setup findings I2, I4, I6, M3 and M5. The I2, I4 and M3 tests are
the review's proofs (tests/test_setup_review_proofs.py on bosmang-proof/mem/setup-review @ 671d426),
copied unchanged; I6 and M5 were code findings and get new tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mem import auth, cli, doctor, shipped
from mem import harnesses as hx
from mem import install as inst
from mem import setup_cmd as sc
from mem import setup_steps as ss
from mem.setup_steps import Backups, Context
from mem.store import Store
from mem.testing import HashEmbedder
from tests.helpers import serve_thread
from tests.test_install import FakeRunner
from tests.test_setup_cmd import SetupFake, _setup

LINUX = {"env": {}, "platform": "linux"}


@pytest.fixture
def home(tmp_path) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    return h


# ---- I2: `mem setup`'s doctor never looks at the gate --------------------------------------------


def _store_without_a_gate(path: Path) -> Path:
    Store.open(path, HashEmbedder(), scope="personal").close()
    return path


def test_setup_doctor_reports_a_store_that_no_gate_can_serve(tmp_path, monkeypatch):
    """`install.SystemRunner.doctor` (what `mem setup` runs at its end) calls `doctor.run` without
    `cfg`, so `check_gate` never runs: with no gate shipped and none fitted, setup ends with
    `self-test: PASS` while `mem serve` refuses the store. The same checks with the config
    (what the standalone `mem doctor --store` does) FAIL and name the `mem calibrate` command."""
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "none" / "serving_gate.json")
    db = _store_without_a_gate(tmp_path / "personal.db")
    from mem.config import default_config

    with_cfg = doctor.run(stores=[db], device="cpu", cfg=default_config())
    assert any(c.name.startswith("gate") and c.status == "fail" for c in with_cfg)
    checks = inst.SystemRunner().doctor([db], "cpu")
    gate = [c for c in checks if c.name.startswith("gate")]
    assert gate and gate[0].status == "fail" and "mem calibrate" in gate[0].detail, checks


def test_doctor_without_store_names_a_missing_gate(tmp_path, monkeypatch):
    """README: 'If your checkout has no src/mem/shipped/serving_gate.json, mem doctor says so.'
    Plain `mem doctor` (no --store, which is what the README and `mem setup` run) checks no gate
    at all: `doctor.run` guards `check_gate` with `stores`, and a fresh install has none yet."""
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "none" / "serving_gate.json")
    monkeypatch.setattr(inst.Paths, "here", classmethod(lambda c: c.for_home(tmp_path, env={})))
    res = CliRunner().invoke(cli.app, ["doctor", "--allow-unencrypted", "--opt-out-log",
                                       str(tmp_path / "optout.jsonl")])  # fmt: skip
    assert "gate" in res.output, res.output


def test_setup_doctor_fails_a_store_not_created_yet_that_no_gate_can_serve(tmp_path, monkeypatch):
    """A fresh install: the stores do not exist yet and no gate fits. `mem serve` refuses them
    (`serve.preflight`), so doctor must FAIL them too, not warn."""
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "none" / "serving_gate.json")
    checks = inst.SystemRunner().doctor([tmp_path / "personal.db"], "cpu")
    gate = [c for c in checks if c.name.startswith("gate")]
    assert gate and gate[0].status == "fail" and "mem calibrate" in gate[0].detail, checks


def test_setup_checks_the_gate_of_every_store_with_the_config_it_wrote(home):
    """`mem setup` hands its doctor every store of the install, present or not, and the config."""
    seen = []
    run = SetupFake()
    run.doctor = lambda stores, device, cfg=None: seen.append((stores, cfg)) or []
    code, out = _setup(home, run, yes=True)
    assert code == 0, out
    stores, cfg = seen[0]
    assert {p.name for p in stores} == {"personal.db", "work.db"} and cfg is not None


# ---- I4: uninstall takes out only an entry that matches today's port and label -----------------


def _gemini(home: Path, text: str) -> Path:
    (home / ".gemini").mkdir(exist_ok=True)
    path = home / ".gemini" / "settings.json"
    path.write_text(text)
    return path


def test_uninstall_removes_the_servers_setup_added_on_another_port(home):
    """`_plan_uninstall_extras` plans a writer's `remove` only when `configured(ctx, local)` is
    true, an exact match of url and headers for the port and host label of *this* run. After
    `mem setup --port 8766`, `mem setup --uninstall` (default port) leaves `mem-personal` and
    `mem-work` in Gemini CLI, Codex CLI and Cursor, and the backup slot behind. `remove` itself
    goes by name, so the plan should too."""
    settings = _gemini(home, '{"mcpServers": {"keep": {"httpUrl": "http://k"}}}\n')
    original = settings.read_bytes()
    run = SetupFake()
    writer = hx.ConfigWriter("gemini", **LINUX)
    code, out = _setup(home, run, yes=True, port=8766, writers=[writer])
    assert code == 0 and "mem-personal" in settings.read_text(), out
    code, out = _setup(home, run, yes=True, uninstall=True, writers=[writer])
    assert code == 0, out
    assert settings.read_bytes() == original, settings.read_text()
    assert not (home / ".config/mem/backups").exists()


# ---- M3: traces left by uninstall ---------------------------------------------------------------


class _Which(FakeRunner):
    def __init__(self, *tools):
        super().__init__(tools=tools)


def test_uninstall_removes_the_config_folder_the_writer_created(home):
    """Codex CLI found by `codex` on PATH with no ~/.codex yet: `add` creates ~/.codex/config.toml
    (`_commit` mkdirs), `remove` unlinks the file and leaves the empty ~/.codex behind."""
    ctx = Context(runner=_Which("codex"), backups=Backups(home / ".config/mem/backups"), home=home)
    token = home / "token"
    token.write_text("mem_t\n")
    eps = [ss.Endpoint("mem-personal", "http://127.0.0.1:8765/personal/mcp", token)]
    writer = hx.ConfigWriter("codex", **LINUX)
    writer.add(ctx, eps)
    assert (home / ".codex" / "config.toml").exists()
    writer.remove(ctx)
    assert not (home / ".codex").exists()


def test_uninstall_keeps_a_config_folder_the_user_had(home):
    (home / ".codex").mkdir()
    ctx = Context(runner=_Which("codex"), backups=Backups(home / ".config/mem/backups"), home=home)
    token = home / "token"
    token.write_text("mem_t\n")
    eps = [ss.Endpoint("mem-personal", "http://127.0.0.1:8765/personal/mcp", token)]
    writer = hx.ConfigWriter("codex", **LINUX)
    writer.add(ctx, eps)
    writer.remove(ctx)
    assert (home / ".codex").is_dir() and not (home / ".codex" / "config.toml").exists()


# ---- I6: the self-test goes through the running service -----------------------------------------


@pytest.fixture
def service(tmp_path):
    """A real `build_app` on 127.0.0.1 with bearer tokens, as mem.service runs it."""
    tf = auth.TokenFile(tmp_path / "tokens")
    token = tf.add("mem-install")
    url, stop = serve_thread(tmp_path / "srv", auth.Access(tokens=tf.load()))
    yield f"{url}/personal/mcp", token
    stop()


def test_selftest_remembers_recalls_and_forgets_through_the_service(service):
    url, token = service
    results = sc.selftest(url, token, "box")
    assert [(n, ok) for n, ok, _ in results] == [("service", True), ("remember", True),
                                                 ("recall", True), ("forget", True)], results  # fmt: skip


def test_selftest_fails_when_the_service_recalls_nothing(service, monkeypatch):
    recall = Store.recall
    monkeypatch.setattr(Store, "recall", lambda self, **kw: recall(self, **kw, since="2999-01-01"))
    results = sc.selftest(*service, "box")
    assert ("recall", False) in [(n, ok) for n, ok, _ in results], results


def test_selftest_fails_when_the_service_refuses_the_token(service):
    url, _token = service
    results = sc.selftest(url, "mem_wrong", "box", wait=2)
    assert results[0][0] == "service" and not results[0][1] and "token" in results[0][2], results
    assert len(results) == 1


def test_selftest_fails_when_nothing_answers_on_the_port(tmp_path):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    results = sc.selftest(f"http://127.0.0.1:{port}/personal/mcp", "mem_t", "box", wait=0.5)
    assert results[0][0] == "service" and not results[0][1], results
    assert "journalctl --user -u mem.service" in results[0][2] and len(results) == 1


def test_setup_self_test_targets_the_port_with_the_installed_token(home):
    seen = []
    run = SetupFake()
    run.selftest = lambda url, token, host: seen.append((url, token, host)) or [("x", True, "")]
    code, out = _setup(home, run, yes=True, port=8766)
    assert code == 0, out
    token = (home / ".config/mem/token").read_text().strip()
    assert seen == [("http://127.0.0.1:8766/personal/mcp", token, "box")]


def test_setup_loads_no_model_for_the_self_test(home, monkeypatch):
    """The self-test runs in mem.service; the setup process must not load the models next to it."""
    import mem.embedder

    def boom(*a, **kw):
        raise AssertionError("setup loaded the embedder")

    monkeypatch.setattr(mem.embedder, "SentenceTransformerEmbedder", boom)
    monkeypatch.setattr(sc, "selftest", lambda url, token, host: [("service", True, "up")])
    assert inst.SystemRunner().selftest("http://x", "t", "box") == [("service", True, "up")]


# ---- M5: Claude Desktop is named, and there is one setup interface ------------------------------


def test_setup_names_claude_desktop_and_the_text_to_paste(home):
    (home / ".config" / "Claude").mkdir(parents=True)
    code, out = _setup(home, SetupFake(), yes=True)
    assert code == 0, out
    assert "Claude Desktop" in out and "mem instructions --harness claude-desktop" in out


def test_setup_says_nothing_of_claude_desktop_when_it_is_not_there(home):
    code, out = _setup(home, SetupFake(), yes=True)
    assert code == 0 and "Claude Desktop" not in out, out


def test_the_unused_step_interface_is_gone():
    assert not any(hasattr(hx, n) for n in ("McpStep", "SkillStep"))
    assert not any(hasattr(ss, n) for n in ("Step", "default_steps", "run"))
