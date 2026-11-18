"""MEM2-140 review proofs for the open-source release (oss-scrub, oss-install, oss-docs).

Each test fails on bosmang/oss-docs @ 4fa0cb2 and states the behaviour the finding asks for.
Denied words are decoded at run time so this file passes the public-clean scan itself."""

from __future__ import annotations

import base64
from pathlib import Path

from mem import install as inst
from mem.doctor import Check

REPO = Path(__file__).resolve().parents[1]
FOREIGN_UNIT = (
    "[Unit]\nDescription=mem, hand-installed from deploy/mem.service (p7-auth)\n\n[Service]\n"
    "WorkingDirectory=%h/code/mem\nEnvironment=PATH=%h/.local/bin:/usr/bin:/bin\n"
    "ExecStart=/usr/bin/env uv run --extra models mem serve --http --host 127.0.0.1 --port 8765 "
    "--host-label home --tls --token-file %h/.config/mem/tokens\n\n[Install]\nWantedBy=default.target\n"
)


class Runner:
    """uv, systemctl and claude on PATH; mem.service enabled and running (the host's state)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def which(self, name: str) -> str | None:
        return f"/opt/fake/bin/{name}"

    def gpu(self, device: str) -> Check:
        return Check("device", "ok", "cuda: Fake GPU")

    def missing_models(self) -> list[str]:
        return []

    def fetch_models(self) -> list[Check]:
        return []

    def run(self, cmd: list[str], cwd: Path | None = None) -> inst.Result:
        self.calls.append(list(cmd))
        if cmd[:3] in (["systemctl", "--user", "is-enabled"], ["systemctl", "--user", "is-active"]):
            return inst.Result(0, "")
        if cmd[:3] == ["claude", "mcp", "get"]:
            return inst.Result(1, "No MCP server named")
        return inst.Result(0, "")


def _run(home: Path, run: Runner, **kw) -> tuple[int, str]:
    out: list[str] = []
    paths = inst.Paths.for_home(home, env={})
    code = inst.main(inst.Options(host_label="box", **kw), run, paths, echo=out.append)
    return code, "\n".join(out)


def _host(tmp_path: Path) -> tuple[Path, Path]:
    """A home like the serving host's: a hand-copied unit, a config and a token, no install marker."""
    home = tmp_path / "home"
    unit = home / ".config/systemd/user/mem.service"
    unit.parent.mkdir(parents=True)
    unit.write_text(FOREIGN_UNIT)
    cfg = home / ".config/mem"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("device: cuda\n")
    (cfg / "tokens").write_text("{}\n")
    (cfg / "token").write_text("mem_x\n")
    return home, unit


def test_uninstall_leaves_a_unit_mem_install_did_not_write(tmp_path):
    """Finding I1: `plan_uninstall` removes ~/.config/systemd/user/mem.service whenever it exists,
    and stops the service, with no check that `mem install` wrote it (install.py `plan_uninstall`,
    `if paths.unit.exists()`). On the host the unit is the hand-copied deploy/mem.service with
    `--tls --host-label home` (p7-auth.md §3); oss-release.md step 6 runs exactly this command there
    and the TLS unit is gone for good (step 7 re-renders one without --tls)."""
    home, unit = _host(tmp_path)
    run = Runner()
    code, out = _run(home, run, uninstall=True)
    assert code == 0, out
    assert unit.read_text() == FOREIGN_UNIT, "a unit without the install marker must survive"
    assert ["systemctl", "--user", "disable", "--now", "mem.service"] not in run.calls
    assert "not written by mem install" in out or "kept" in out


def test_install_does_not_overwrite_a_unit_it_did_not_write(tmp_path):
    """Finding I1 (same root cause): a fresh `mem install` on a machine with a hand-written unit
    rewrites it (`_plan_service`: `changed = ... != text` -> write), dropping every operator flag
    (--tls, --tailscale, --oauth, --tz, --host-label) without a word."""
    home, unit = _host(tmp_path)
    run = Runner()
    code, out = _run(home, run)
    assert unit.read_text() == FOREIGN_UNIT, (
        "a unit without the install marker must not be rewritten"
    )
    assert code != 0 or "not written by mem install" in out


def test_public_clean_scan_covers_the_path_not_only_the_content(tmp_path):
    """Finding I2: tests/test_public_clean.py `_scan` looks at file content only, so a file or
    folder *named* with a denied word (docs/<owner>-notes.md, bench/<employer>/x.py) ships
    unnoticed; `mem release export` runs the same test, so it passes too."""
    from tests.test_public_clean import _DENY, _scan

    owner = base64.b64decode(_DENY[1]).decode()  # the owner's first name
    assert _scan(Path(f"docs/{owner}-notes.md"), b"# nothing personal in here\n"), (
        "a denied word in the file path must be a hit"
    )
    assert _scan(Path(f"bench/{owner}/cli.py"), b"x = 1\n")
