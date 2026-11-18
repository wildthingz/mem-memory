"""MEM2-226: test scratch files land under the worktree's `.tmp/` (on disk), never in /tmp.

The sandbox's /tmp is RAM-backed; `tests/conftest.py::pytest_configure` makes the on-disk basetemp
the repo default so it no longer depends on each agent's shell."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest

ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp").resolve()


def _default_basetemp(config) -> Path:
    return Path(config.rootpath) / ".tmp" / f"pytest-{os.getpid()}"


def _skip_unless_defaulted(config) -> None:
    given = Path(config.option.basetemp)
    if given.resolve() != _default_basetemp(config).resolve():
        pytest.skip(f"--basetemp overridden by the caller: {given}")


def test_tmp_path_lives_under_the_worktree_scratch_dir_not_tmp(tmp_path, pytestconfig):
    _skip_unless_defaulted(pytestconfig)
    assert pytestconfig.rootpath == ROOT
    got = tmp_path.resolve()
    assert got.is_relative_to((ROOT / ".tmp" / f"pytest-{os.getpid()}").resolve()), got
    assert not got.is_relative_to(TMP), got


def test_tempfile_uses_the_same_scratch_dir(pytestconfig):
    """Bench code called from tests writes through `tempfile`; it must land in `.tmp/` too, and
    subprocesses inherit the same TMPDIR."""
    _skip_unless_defaulted(pytestconfig)
    scratch = (ROOT / ".tmp").resolve()
    assert Path(tempfile.gettempdir()).resolve() == scratch
    assert Path(os.environ["TMPDIR"]).resolve() == scratch
    with tempfile.TemporaryDirectory(prefix="bench-x-") as d:
        assert Path(d).resolve().parent == scratch
    with tempfile.NamedTemporaryFile() as f:
        assert Path(f.name).resolve().parent == scratch


def test_an_explicit_basetemp_wins_and_leaves_tmpdir_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/somewhere/else")
    monkeypatch.setattr(tempfile, "tempdir", "/somewhere/else")
    cfg = SimpleNamespace(
        option=SimpleNamespace(basetemp=str(tmp_path / "mine")), rootpath=tmp_path
    )
    conftest.pytest_configure(cfg)
    assert cfg.option.basetemp == str(tmp_path / "mine")
    assert os.environ["TMPDIR"] == "/somewhere/else" and tempfile.tempdir == "/somewhere/else"
    assert not (tmp_path / ".tmp").exists()


def test_without_basetemp_the_hook_points_everything_at_rootdir_dot_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/somewhere/else")
    monkeypatch.setattr(tempfile, "tempdir", "/somewhere/else")
    cfg = SimpleNamespace(option=SimpleNamespace(basetemp=None), rootpath=tmp_path)
    conftest.pytest_configure(cfg)
    assert cfg.option.basetemp == str(tmp_path / ".tmp" / f"pytest-{os.getpid()}")
    assert os.environ["TMPDIR"] == str(tmp_path / ".tmp") == tempfile.tempdir
    assert (tmp_path / ".tmp").is_dir()


def test_each_pytest_process_has_its_own_basetemp_and_dead_ones_are_pruned(tmp_path):
    """MEM2-222 F22: pytest clears its basetemp per run; a shared one broke a second process."""
    scratch = tmp_path / ".tmp"
    dead = subprocess.Popen(["true"])
    dead.wait()
    (scratch / f"pytest-{dead.pid}").mkdir(parents=True)
    (scratch / f"pytest-{os.getppid()}").mkdir()  # a live process: kept
    (scratch / "pytest-notapid").mkdir()  # not ours: kept
    cfg = SimpleNamespace(option=SimpleNamespace(basetemp=None), rootpath=tmp_path)
    conftest.pytest_configure(cfg)
    assert cfg.option.basetemp == str(scratch / f"pytest-{os.getpid()}")
    assert not (scratch / f"pytest-{dead.pid}").exists()
    assert (scratch / f"pytest-{os.getppid()}").is_dir() and (scratch / "pytest-notapid").is_dir()


def test_two_pytest_processes_in_one_worktree_do_not_clobber_each_other(tmp_path):
    """Two concurrent pytest runs, each writing to tmp_path and checking it after the other
    started: with a shared basetemp the second run's start-up wipes the first one's files."""
    t = tmp_path / "test_slow.py"
    t.write_text(
        "import os, time\n"
        "def test_keeps_its_file(tmp_path):\n"
        "    (tmp_path / 'x').write_text(str(os.getpid()))\n"
        "    time.sleep(1.5)\n"
        "    assert (tmp_path / 'x').read_text() == str(os.getpid())\n",
        encoding="utf-8",
    )
    (tmp_path / "conftest.py").write_text(
        (ROOT / "tests" / "conftest.py")
        .read_text(encoding="utf-8")
        .split("class Clock")[0]
        .replace("from mem.store import Store\n", "")
        .replace("from mem.testing import HashEmbedder\n", ""),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(tmp_path),
        str(t),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("TMPDIR", None)
    first = subprocess.Popen(cmd, cwd=tmp_path, env=env, stdout=subprocess.PIPE, text=True)
    time.sleep(0.7)
    second = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    out, _ = first.communicate(timeout=60)
    assert first.returncode == 0, out
    assert second.returncode == 0, second.stdout
