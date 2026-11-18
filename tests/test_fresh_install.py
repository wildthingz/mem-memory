"""The fresh-install check (MEM2-236) as a test: `mem setup --yes`, `mem doctor`, the self-test
and `--uninstall --purge` on a temp HOME with the network blocked, through the real CLI code.
The CI job and `scripts/fresh-install-test.sh` run the same driver from a fresh venv, without a
stand-in gate: there it fails until the package ships one (MEM2-237 M6). Scratch goes in the
checkout's .tmp/ (on disk), never in /tmp, and is gone after every run."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from mem import shipped

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "e2e" / "fresh_install.py"
SCRIPT = ROOT / "scripts" / "fresh-install-test.sh"


@pytest.fixture
def scratch():
    work = ROOT / ".tmp" / f"fresh-install-test-{uuid.uuid4().hex[:8]}"
    yield work
    shutil.rmtree(work, ignore_errors=True)


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEM_TEST_")}
    return env | extra


def _driver(work: Path, *, fake_gate: bool, no_shipped_gate: bool = False, **env: str):
    if fake_gate:
        env["MEM_TEST_FAKE_GATE"] = "1"
    hide = "from mem import shipped; shipped.GATE_PATH = shipped.GATE_PATH.with_name('none.json'); "
    code = (
        "import runpy, sys; "
        + (hide if no_shipped_gate else "")
        + f"sys.argv = [{str(DRIVER)!r}, {str(work)!r}]; "
        + f"runpy.run_path({str(DRIVER)!r}, run_name='__main__')"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300, check=False, env=_env(**env))  # fmt: skip


def test_a_fresh_install_works_and_leaves_no_trace_without_network(scratch):
    # this checkout's gate when one is committed; the stand-in only while none is
    done = _driver(scratch, fake_gate=not shipped.GATE_PATH.is_file())
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    assert "fresh install: PASS" in done.stdout
    assert "peak RSS" in done.stdout and "scratch" in done.stdout
    assert not scratch.exists()  # the driver deletes its scratch


def test_the_fresh_install_check_fails_when_the_package_ships_no_gate(scratch):
    done = _driver(scratch, fake_gate=False, no_shipped_gate=True)
    assert done.returncode == 1, done.stdout[-3000:] + done.stderr[-2000:]
    assert "fresh install: FAIL: the package ships no gate" in done.stdout
    assert "$ mem setup" not in done.stdout  # it stops before changing anything
    assert not scratch.exists()


def test_the_driver_stops_when_its_scratch_passes_the_limit(scratch):
    done = _driver(scratch, fake_gate=True, MEM_TEST_SCRATCH_LIMIT_MB="0")
    assert done.returncode == 1, done.stdout[-3000:] + done.stderr[-2000:]
    assert "over the 0 MB limit" in done.stdout
    assert "$ mem setup" not in done.stdout  # it stops at the first check, before setup
    assert not scratch.exists()  # and still cleans up


def test_the_driver_refuses_a_workdir_that_is_not_empty(scratch):
    scratch.mkdir(parents=True)
    (scratch / "keep").write_text("mine")
    done = _driver(scratch, fake_gate=True)
    assert done.returncode == 1 and "is not empty" in done.stdout
    assert (scratch / "keep").read_text() == "mine"


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)
def test_the_script_stops_when_its_scratch_passes_the_limit_and_cleans_up(scratch):
    done = subprocess.run([str(SCRIPT), str(scratch)], capture_output=True, text=True,
                          timeout=120, check=False, cwd=ROOT,
                          env=_env(MEM_TEST_SCRATCH_LIMIT_MB="0"))  # fmt: skip
    assert done.returncode == 1, done.stdout + done.stderr
    assert "after the source copy, over the 0 MB limit" in done.stdout
    assert not scratch.exists()  # the trap deletes it on failure too
