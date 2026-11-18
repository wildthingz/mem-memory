#!/bin/sh
# Fresh-install test (MEM2-236): a wheel built from what a user gets (`git archive HEAD`, so commit
# first), installed into a new venv, then that copy's tests/e2e/fresh_install.py on a clean HOME
# (see its docstring for what is faked; the models are always fakes here). Fails when the wheel
# ships no gate; MEM_TEST_FAKE_GATE=1 tests the rest with a stand-in gate.
# Usage: scripts/fresh-install-test.sh [WORKDIR]   (default .tmp/fresh-install in the checkout)
# Every byte of scratch (source copy, wheel, venv, uv cache, TMPDIR, the fake HOME) goes under
# WORKDIR, on disk and never in /tmp; WORKDIR is deleted on exit, pass or fail, and the run stops
# as soon as it passes MEM_TEST_SCRATCH_LIMIT_MB (default 1024).
set -eu
cd "$(dirname "$0")/.."
WORK="${1:-.tmp/fresh-install}"
LIMIT_MB="${MEM_TEST_SCRATCH_LIMIT_MB:-1024}"
rm -rf "$WORK"; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
trap 'rm -rf "$WORK"' EXIT
trap 'exit 130' INT TERM

guard() {
  used="$(du -sm "$WORK" | cut -f1)"
  if [ "$used" -gt "$LIMIT_MB" ]; then
    echo "fresh install: FAIL: scratch $WORK is $used MB after $1, over the $LIMIT_MB MB limit"
    exit 1
  fi
}

# the interpreter stays where uv keeps it (the venv links to it); everything else moves to WORK
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/python}"
export HOME="$WORK/home" TMPDIR="$WORK/tmp" UV_CACHE_DIR="$WORK/uv-cache"
export XDG_CACHE_HOME="$WORK/cache" XDG_CONFIG_HOME="$WORK/config" XDG_DATA_HOME="$WORK/data"
export XDG_STATE_HOME="$WORK/state"
mkdir -p "$HOME" "$TMPDIR" "$WORK/src"

git archive HEAD | tar -x -C "$WORK/src"
guard "the source copy"
uv build --quiet --wheel --out-dir "$WORK/dist" "$WORK/src"
guard "building the wheel"
uv venv --quiet --python 3.12 "$WORK/venv"
uv pip install --quiet --python "$WORK/venv/bin/python" "$WORK"/dist/*.whl
guard "installing the wheel"
"$WORK/venv/bin/python" -c "import mem; print('installed:', mem.__file__)"
# `mem` on PATH, as `uv tool install` leaves it: the service unit runs that executable
PATH="$WORK/venv/bin:$PATH" MEM_TEST_SCRATCH_ROOT="$WORK" MEM_TEST_SCRATCH_LIMIT_MB="$LIMIT_MB" \
  "$WORK/venv/bin/python" "$WORK/src/tests/e2e/fresh_install.py" "$WORK/run"
