import os

# Typer forces a rich terminal under GITHUB_ACTIONS / FORCE_COLOR / PY_COLORS, and its ANSI codes
# split option names in CLI error text. Read at Typer's import, so it is set before anything loads it.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

import shutil
import socket
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mem import config as mem_config
from mem.store import Store
from mem.testing import HashEmbedder


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Scratch files go on disk, under the worktree, never in /tmp (MEM2-226: the sandbox's /tmp
    is RAM, and pytest tmp dirs there caused two OOM kills). Unless the caller passed
    `--basetemp`, `tmp_path` lives under `<rootdir>/.tmp/pytest-<pid>` and `tempfile` (what bench
    code called from tests, and subprocesses, use) under `<rootdir>/.tmp`. One basetemp per
    pytest process: pytest clears its basetemp at the start of a run, so a shared one let two
    pytest processes in one worktree delete each other's files (MEM2-222 F22); the dirs of
    processes that are gone are removed here. An explicit `--basetemp` wins and leaves TMPDIR
    alone. `tryfirst`: pytest's tmpdir plugin reads the option in its own `pytest_configure`."""
    if config.option.basetemp:
        return
    scratch = Path(config.rootpath) / ".tmp"
    scratch.mkdir(exist_ok=True)
    _prune_dead_basetemps(scratch)
    config.option.basetemp = str(scratch / f"pytest-{os.getpid()}")
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)


# MEM2-261 turned `ingest.resolve_dates` and `read.list_mode` on in the default config (Round 9's final
# config). The T0 suite was written for the read and write paths with every switch off and keeps testing
# them that way: from conftest's import on (before test modules build their module-level configs),
# `default_config()` returns the Round 8 sections, the `all_off_config()` ones, so a test's own gate needs
# no `list_model`. Tests of the release defaults are marked `release_defaults` and see the real sections;
# each switch has its own on-tests. Subprocesses are not patched: a test that starts `mem serve` in one
# passes the `round8_config` file. MEM2-274 made the default Round 10's `r10-all` (also `read.list_fill`,
# `read.window_boost` and the Qwen3 embedder); the Round 8 sections here have none of them, and the
# embedder is the Round 8 (Gemma) one.
_RELEASE_SECTIONS = (
    mem_config._read_section,
    mem_config._ingest_section,
    mem_config._embedder_section,
)


def _round8_read():
    return {"conflict_guard": False, "p3_model": False}


def _round8_ingest():
    return {"resolve_dates": False}


mem_config._read_section, mem_config._ingest_section = _round8_read, _round8_ingest
mem_config._embedder_section = mem_config.pins.embedder_section


@pytest.fixture(autouse=True)
def _release_defaults_when_marked(request, monkeypatch):
    if request.node.get_closest_marker("release_defaults"):
        monkeypatch.setattr(mem_config, "_read_section", _RELEASE_SECTIONS[0])
        monkeypatch.setattr(mem_config, "_ingest_section", _RELEASE_SECTIONS[1])
        monkeypatch.setattr(mem_config, "_embedder_section", _RELEASE_SECTIONS[2])


@pytest.fixture(scope="session")
def round8_config(tmp_path_factory):
    """A config file with the MEM2-261 and MEM2-274 defaults off, for a `mem serve` started as a subprocess."""
    path = tmp_path_factory.mktemp("cfg") / "round8.yaml"
    path.write_text(
        "models: {embedder: embeddinggemma-300m}\ningest: {resolve_dates: false}\n"
        "read: {list_mode: false, list_fill: 0, window_boost: false}\n"
    )
    return path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_dead_basetemps(scratch: Path) -> None:
    for d in scratch.glob("pytest-*"):
        pid = d.name.removeprefix("pytest-")
        if pid.isdigit() and int(pid) != os.getpid() and not _alive(int(pid)):
            shutil.rmtree(d, ignore_errors=True)


class Clock:
    def __init__(self, iso: str = "2026-10-22T09:00:00Z") -> None:
        self.t = datetime.fromisoformat(iso)

    def __call__(self) -> datetime:
        return self.t.astimezone(UTC)

    def set(self, iso: str) -> None:
        self.t = datetime.fromisoformat(iso)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def make_store(tmp_path, clock):
    opened = []

    def _make(name="personal", **kw):
        s = Store.open(tmp_path / f"{name}.db", HashEmbedder(), scope=name, now=clock, **kw)
        opened.append(s)
        return s

    yield _make
    for s in opened:
        s.close()


@pytest.fixture
def store(make_store):
    return make_store()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def free_port():
    """A port nothing listens on: CLI tests pass it as --port so a real `mem serve` on the
    default port (the person's host) is never called or mistaken for the test's."""
    return _free_port()


@pytest.fixture
def models_serve():
    """`mem serve --http --models-only --embedder hash` in a thread: the `/models` routes over
    the stub models and the stub config. Yields the port."""
    import uvicorn

    from mem.config import stub_config
    from mem.models_http import ModelHost
    from mem.serve import build_app
    from mem.testing import OverlapReranker

    port = _free_port()
    app = build_app({}, models=ModelHost(HashEmbedder(256), OverlapReranker(), stub_config()))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield port
    server.should_exit = True
    thread.join(5)
