"""Fresh-install check (MEM2-236): `mem setup --yes`, `mem doctor`, the self-test and
`mem setup --uninstall --purge` on a clean HOME with the network blocked, through the real CLI code.

Run it inside the venv the package was installed into: `python tests/e2e/fresh_install.py WORKDIR`
(`scripts/fresh-install-test.sh` makes the venv). What is faked, and only this: the machine has no
GPU, systemd user session or Claude Code, so `systemctl` and `claude` are small scripts on PATH that
keep their state in WORKDIR/state; the NVIDIA check says ok; the two pinned models are placeholder
files in a Hugging Face cache (so `doctor` finds them and nothing is downloaded); mem.service is
a uvicorn thread with the hash embedder while the self-test talks to it. The gate is not faked:
the installed package must ship one for its default config, or the check fails (a fresh install
could not serve). Only with `MEM_TEST_FAKE_GATE=1` is a stand-in gate written where the package
looks for it, for testing the rest of the flow on a tree with no gate committed. Exits 0 and
prints `fresh install: PASS`, or exits 1 naming the first thing that failed.

All scratch (the fake HOME, caches, TMPDIR) is under WORKDIR, which must be new or empty and is
deleted at the end, pass or fail (`MEM_TEST_KEEP=1` keeps it). The check fails as soon as the
scratch (`MEM_TEST_SCRATCH_ROOT`, default WORKDIR) passes `MEM_TEST_SCRATCH_LIMIT_MB` (1024).
"""

from __future__ import annotations

import hashlib
import os
import resource
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

WORK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".tmp/fresh-install").resolve()
HOME = WORK / "home"
STATE = WORK / "state"
BIN = WORK / "bin"
SCRATCH = Path(os.environ.get("MEM_TEST_SCRATCH_ROOT") or WORK).resolve()
LIMIT_MB = int(os.environ.get("MEM_TEST_SCRATCH_LIMIT_MB") or 1024)

SYSTEMCTL = """#!/bin/sh
# fake systemctl --user: remembers which units are enabled/active
S="$FAKE_STATE/units"; mkdir -p "$S"
shift  # --user
cmd="$1"; shift
unit="${*##* }"
case "$cmd" in
  show-environment|daemon-reload) exit 0 ;;
  enable) touch "$S/$unit.enabled"; [ "$1" = "--now" ] && touch "$S/$unit.active"; exit 0 ;;
  disable) rm -f "$S/$unit.enabled"; rm -f "$S/$unit.active"; exit 0 ;;
  restart|start) touch "$S/$unit.active"; exit 0 ;;
  stop) rm -f "$S/$unit.active"; exit 0 ;;
  is-active) [ -f "$S/$unit.active" ]; exit $? ;;
  is-enabled) [ -f "$S/$unit.enabled" ]; exit $? ;;
esac
exit 0
"""

CLAUDE = """#!/bin/sh
# fake `claude mcp add|get|remove`: one file per server in $FAKE_STATE/claude
S="$FAKE_STATE/claude"; mkdir -p "$S"
[ "$1" = "mcp" ] || exit 0
op="$2"; shift 2
case "$op" in
  add) while [ $# -gt 0 ]; do case "$1" in --transport|--scope|--header) shift 2 ;; -*) shift ;; *) break ;; esac; done
       echo "$*" > "$S/$1"; exit 0 ;;
  get) [ -f "$S/$3" ] || [ -f "$S/$1" ] ; exit $? ;;
  remove) while [ $# -gt 0 ]; do case "$1" in --scope) shift 2 ;; *) break ;; esac; done
          rm -f "$S/$1"; exit 0 ;;
esac
exit 0
"""


def block_network() -> None:
    """After the models are in the cache, `mem setup` must not need the network: any connection
    to an address that is not this machine raises."""
    import socket

    def local(host) -> bool:
        return isinstance(host, str) and host in ("127.0.0.1", "localhost", "::1", "")

    real_connect, real_lookup = socket.socket.connect, socket.getaddrinfo

    def connect(self, address):
        if self.family != socket.AF_UNIX and not local(address[0]):
            raise OSError(f"no network in this test: connect to {address[0]}")
        return real_connect(self, address)

    def lookup(host, *a, **kw):
        if not local(host):
            raise OSError(f"no network in this test: lookup of {host}")
        return real_lookup(host, *a, **kw)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.getaddrinfo = lookup  # type: ignore[assignment]


def fail(msg: str) -> NoReturn:
    print(f"fresh install: FAIL: {msg}")
    sys.exit(1)


def scratch_mb() -> int:
    return int(subprocess.run(["du", "-sm", str(SCRATCH)], capture_output=True, text=True,
                              check=True).stdout.split()[0])  # fmt: skip


def guard(after: str) -> None:
    """Stop before the scratch can fill the disk (or a RAM-backed /tmp)."""
    used = scratch_mb()
    if used > LIMIT_MB:
        fail(f"scratch {SCRATCH} is {used} MB after {after}, over the {LIMIT_MB} MB limit")


def snapshot(root: Path) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_dir():
            out[rel + "/"] = (stat.S_IMODE(p.stat().st_mode),)
        else:
            out[rel] = (stat.S_IMODE(p.stat().st_mode), hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def prepare() -> None:
    for d in (HOME, STATE, BIN):
        d.mkdir(parents=True, exist_ok=True)
    for name, body in (("systemctl", SYSTEMCTL), ("claude", CLAUDE)):
        (BIN / name).write_text(body)
        (BIN / name).chmod(0o755)
    # harness configs the person already has: must come back byte for byte
    (HOME / ".codex").mkdir()
    (HOME / ".codex" / "config.toml").write_text(
        'model = "o3"\n\n[mcp_servers.other]\nurl = "http://x/mcp"\n'
    )
    (HOME / ".gemini").mkdir()
    (HOME / ".gemini" / "settings.json").write_text(
        '{"theme": "dark",\n "mcpServers": {"keep": {"httpUrl": "http://k"}}}\n'
    )
    (HOME / ".gemini" / "settings.json").chmod(0o640)
    (HOME / ".cursor").mkdir()
    (HOME / ".cursor" / "mcp.json").write_text('{"mcpServers": {}}')
    for var in [
        v
        for v in os.environ
        if v.startswith(("CODEX_", "CLAUDE_", "GEMINI_", "XDG_", "HF_", "UV_CACHE"))
    ]:
        del os.environ[var]  # the person's own harness settings must not reach the check
    os.environ.update({
        "HOME": str(HOME), "FAKE_STATE": str(STATE), "HF_HOME": str(WORK / "hf"),
        "HF_HUB_OFFLINE": "1", "PATH": f"{BIN}:{os.environ['PATH']}",
        "XDG_CONFIG_HOME": str(HOME / ".config"), "XDG_DATA_HOME": str(HOME / ".local/share"),
        "XDG_STATE_HOME": str(HOME / ".local/state"), "XDG_CACHE_HOME": str(WORK / "cache"),
        "UV_CACHE_DIR": str(WORK / "cache" / "uv"), "TMPDIR": str(WORK / "tmp"),
    })  # fmt: skip
    (WORK / "tmp").mkdir(exist_ok=True)


def fake_machine() -> None:
    from mem import config, doctor, install, pins, shipped
    from mem.common import MemError
    from mem.gate import Gate

    for pin in doctor.configured_pins(config.default_config()):  # the models setup fetches
        snap = pins.snapshot_dir(pin)
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        (snap / doctor.WEIGHTS).write_bytes(b"placeholder")
    doctor.check_device = lambda device="cuda": doctor.Check("device", "ok", "cuda: fake GPU")
    install.SystemRunner.selftest = _selftest_on_a_stand_in_service(install.SystemRunner.selftest)
    cfg = config.default_config()
    if os.environ.get("MEM_TEST_FAKE_GATE") != "1":
        try:
            gate = shipped.gate_for(cfg)
        except MemError as err:
            fail(f"the package's shipped gate does not load: {err}")
        if gate is None:
            fail(f"the package ships no gate for its default config ({shipped.GATE_PATH}): "
                 "`mem serve` would refuse both stores on a fresh install. Commit the gate "
                 "(deliverables/runbooks/setup-host.md, step 1); MEM_TEST_FAKE_GATE=1 tests the "
                 "rest with a stand-in")  # fmt: skip
        print(f"shipped gate: {shipped.GATE_PATH} (fit {gate.fit_id})")
    else:
        print("shipped gate: none checked; MEM_TEST_FAKE_GATE=1 writes a stand-in")
        gate = Gate(platt_a=1.0, platt_b=0.0, tau_keep=0.5, tau_high=0.5, gap=0.0,
                    fit_id="fresh-install-test", fitted_at="2026-10-26T00:00:00Z",
                    fingerprint_hash=cfg.fp_all(), tune={"queries": 1})  # fmt: skip
        meta = shipped.metadata(
            gate,
            lme={"name": "LongMemEval-S cleaned", "version": "cleaned", "sha256": "a" * 64,
                 "split": "tune"},
            syn={"name": "SYN", "version": 1, "seeds": [1, 2, 3, 4, 5]},
        )  # fmt: skip
        shipped.GATE_PATH = WORK / "serving_gate.json"
        shipped.write(shipped.GATE_PATH, gate, meta)


def _selftest_on_a_stand_in_service(selftest):
    """The fake systemctl starts nothing, so for the span of each self-test mem.service is a
    uvicorn thread on the unit's port: the install's stores and token file, the hash embedder."""
    import threading
    import time
    from urllib.parse import urlsplit

    import uvicorn

    from mem import auth, install, serve
    from mem.testing import HashEmbedder

    def run(self, url: str, token: str, host_label: str):
        paths = install.Paths.here()
        access = auth.Access(tokens=auth.TokenFile(paths.tokens).load())
        server: list[uvicorn.Server] = []

        def service() -> None:
            paths.data.mkdir(parents=True, exist_ok=True)  # as `mem serve`'s store locks do
            stores = {n: serve.open_inprocess(p, scope=n, embedder=HashEmbedder(256))
                      for n, p in paths.stores().items()}  # fmt: skip
            config = serve.uvicorn_config(serve.build_app(stores, access=access), host="127.0.0.1",
                                          port=urlsplit(url).port or 80, log_level="error")  # fmt: skip
            server.append(uvicorn.Server(config))
            server[0].run()
            for s in stores.values():
                s.close()

        thread = threading.Thread(target=service, daemon=True)
        thread.start()
        try:
            return selftest(self, url, token, host_label)
        finally:
            while not server and thread.is_alive():
                time.sleep(0.05)
            if server:
                server[0].should_exit = True
            thread.join(10)

    return run


def mem(*args: str) -> tuple[int, str]:
    from typer.testing import CliRunner

    from mem import cli

    result = CliRunner().invoke(cli.app, list(args), catch_exceptions=False)
    print(f"$ mem {' '.join(args)}  -> exit {result.exit_code}\n{result.output}")
    guard(f"`mem {' '.join(args)}`")
    return result.exit_code, result.output


def main() -> None:
    if WORK.exists() and any(WORK.iterdir()):
        fail(f"{WORK} is not empty; the check deletes its WORKDIR at the end, so give it a new one")
    try:
        check()
    finally:
        if os.environ.get("MEM_TEST_KEEP") != "1":
            shutil.rmtree(WORK, ignore_errors=True)


def check() -> None:
    prepare()
    guard("making the fake HOME")
    before = snapshot(HOME)
    fake_machine()
    guard("faking the machine")
    block_network()
    import socket

    try:
        socket.create_connection(("192.0.2.1", 80), timeout=1)
        fail("the network block does not hold")
    except OSError as err:
        if "no network in this test" not in str(err):
            fail(f"the network block does not hold: {err}")

    code, out = mem("setup", "--yes")
    if code != 0 or "self-test: PASS" not in out:
        fail("`mem setup --yes` did not succeed with a passing self-test")
    if "nothing to do" not in mem("setup", "--yes")[1]:
        fail("a second `mem setup --yes` was not a no-op")
    code, out = mem("doctor", "--allow-unencrypted", "--opt-out-log", str(WORK / "optout.jsonl"))
    if code != 0 or "[FAIL]" in out or "[fail]" in out:
        fail("`mem doctor` is not green after setup")
    unit = HOME / ".config/systemd/user/mem.service"
    if not unit.is_file() or not (STATE / "units/mem.service.active").exists():
        fail("the service unit was not installed and started")
    if not (HOME / ".agents/skills/mem").is_dir() and not (HOME / ".claude/skills/mem").is_dir():
        fail("the skill was not installed")
    for rel in (".codex/config.toml", ".gemini/settings.json", ".cursor/mcp.json"):
        if "mem-personal" not in (HOME / rel).read_text():
            fail(f"{rel} does not list the mem servers")
    if not (STATE / "claude/mem-personal").exists():
        fail("Claude Code does not list mem-personal")

    code, out = mem("setup", "--uninstall", "--purge", "--yes")
    if code != 0:
        fail("`mem setup --uninstall --purge` failed")
    after = snapshot(HOME)
    if (STATE / "units").exists() and any((STATE / "units").glob("*.active")):
        fail("the service is still active")
    if list((STATE / "claude").glob("*")):
        fail("Claude Code still lists mem servers")
    left = {k for k in after if after[k] != before.get(k)} | (set(before) - set(after))
    if left:
        fail("not back to the starting point: " + ", ".join(sorted(left)))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"peak RSS {peak:.0f} MiB; scratch {scratch_mb()} MB at the end ({SCRATCH})")
    print("fresh install: PASS")


if __name__ == "__main__":
    main()
