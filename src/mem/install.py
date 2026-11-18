"""`mem install`: set mem up on this machine in one command (MEM2-138).

Steps, each skipped when already in place: the GPU check (`mem doctor`'s device check), the pinned
models (`mem doctor --fetch`), a default config.yaml, a bearer token (its hash in `tokens` for
`mem serve --token-file`, the token itself in `token`, mode 0600, for the clients), the systemd user
unit rendered from deploy/mem.service for this machine, the two MCP servers in Claude Code (user
scope, with the bearer header) and the skill in ~/.claude/skills/mem. `--uninstall` undoes them in
reverse; the config, the token files and the stores stay unless `--purge`; the models stay in the
Hugging Face cache. Both touch only a unit this module wrote (first line `MARKER`): a hand-written
mem.service is kept, unchanged and running, and the output says so. Every system call goes through
one `Runner`, so tests use a fake one.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mem import auth, doctor, serve
from mem.common import MemError
from mem.config import Config, default_config

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "_data"  # deploy/ and skills/ inside an installed wheel
UNIT = "mem.service"
MARKER = "# Written by `mem install`"  # first line of a unit this module rendered
TOKEN_LABEL = "mem-install"
SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")


@dataclass(frozen=True)
class Result:
    code: int
    out: str


@dataclass(frozen=True)
class Probe:
    """One MCP `initialize` against a remote: the HTTP status (None: no answer) and the round trip."""

    status: int | None
    ms: float
    error: str


class Runner(Protocol):
    def run(self, cmd: list[str], cwd: Path | None = None) -> Result: ...
    def which(self, name: str) -> str | None: ...
    def gpu(self, device: str) -> doctor.Check: ...
    def missing_models(self) -> list[str]: ...
    def fetch_models(self) -> list[doctor.Check]: ...


class SystemRunner:
    """The real machine: subprocess, PATH, torch and the Hugging Face cache."""

    def run(self, cmd: list[str], cwd: Path | None = None) -> Result:
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False,
                                  cwd=cwd)  # fmt: skip
        except (OSError, subprocess.SubprocessError) as err:
            return Result(127, str(err))
        return Result(done.returncode, done.stdout + done.stderr)

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def gpu(self, device: str) -> doctor.Check:
        return doctor.check_device(device)

    def missing_models(self) -> list[str]:
        return [c.name for c in doctor.check_models() if c.status != "ok"]

    def fetch_models(self) -> list[doctor.Check]:
        from mem import setup_cmd

        return setup_cmd.fetch_models(print)

    # ---- what `mem setup` asks beyond `mem install` ----

    def port_free(self, port: int) -> bool:
        import socket

        with socket.socket() as sock:
            return sock.connect_ex(("127.0.0.1", port)) != 0

    def cuda_versions(self) -> tuple[str | None, str | None]:
        """(the CUDA version the NVIDIA driver supports, the one torch was built for)."""
        import re

        found = re.search(r"CUDA Version:\s*([\d.]+)", self.run(["nvidia-smi"]).out)
        try:
            import torch

            built = torch.version.cuda
        except ImportError:
            built = None
        return (found.group(1) if found else None), built

    def doctor(
        self, stores: list[Path], device: str, cfg: Config | None = None
    ) -> list[doctor.Check]:
        """`mem doctor` on `stores` (created yet or not) with the gate check under `cfg` (the
        default config when None), as `mem serve` would see them."""
        return doctor.run(stores=stores, device=device, cfg=cfg or default_config())

    def selftest(self, url: str, token: str, host_label: str) -> list[tuple[str, bool, str]]:
        from mem import setup_cmd

        return setup_cmd.selftest(url, token, host_label)

    def probe(self, url: str, token: str) -> Probe:
        from mem import setup_cmd

        return setup_cmd.probe(url, token)


@dataclass(frozen=True)
class Paths:
    home: Path
    config: Path  # $XDG_CONFIG_HOME/mem
    data: Path  # $XDG_DATA_HOME/mem: the stores
    units: Path  # systemd user units
    skill: Path  # ~/.claude/skills/mem
    repo: Path = REPO

    @classmethod
    def for_home(cls, home: Path, env: Mapping[str, str]) -> Paths:
        def xdg(var: str, default: str) -> Path:
            return Path(env.get(var) or home / default)

        cfg = xdg("XDG_CONFIG_HOME", ".config")
        return cls(home=home, config=cfg / "mem", data=xdg("XDG_DATA_HOME", ".local/share") / "mem",
                   units=cfg / "systemd" / "user", skill=home / ".claude" / "skills" / "mem")  # fmt: skip

    @classmethod
    def here(cls) -> Paths:
        return cls.for_home(Path.home(), os.environ)

    @property
    def checkout(self) -> bool:
        """Is `repo` a source checkout (deploy/ and skills/ beside the package)? Not under
        `uv tool install` or `uvx`: the wheel carries both folders in `mem/_data`."""
        return (self.repo / "deploy" / UNIT).is_file()

    @property
    def assets(self) -> Path:
        return self.repo if self.checkout else DATA

    @property
    def config_file(self) -> Path:
        return self.config / "config.yaml"

    @property
    def tokens(self) -> Path:
        return self.config / "tokens"

    @property
    def token(self) -> Path:
        return self.config / "token"

    @property
    def unit(self) -> Path:
        return self.units / UNIT

    def stores(self) -> dict[str, Path]:
        return {name: self.data / f"{name}.db" for name in serve.STORE_NAMES}

    def short(self, path: Path) -> str:
        return f"~/{path.relative_to(self.home)}" if path.is_relative_to(self.home) else str(path)


@dataclass(frozen=True)
class Options:
    host_label: str
    allow_cpu: bool = False
    dry_run: bool = False
    uninstall: bool = False
    purge: bool = False
    mcp: bool = True
    port: int = serve.DEFAULT_PORT


@dataclass
class Action:
    """One change. `cmd` runs through the runner; `apply` changes files. `shown` replaces `cmd`
    in the output when the command holds a secret."""

    step: str
    what: str
    cmd: list[str] | None = None
    apply: Callable[[], None] | None = None
    shown: list[str] | None = None


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    ok: list[tuple[str, str]] = field(default_factory=list)  # (step, why nothing to do)

    def do(self, step: str, what: str, **kw: object) -> None:
        self.actions.append(Action(step, what, **kw))  # type: ignore[arg-type]

    def fine(self, step: str, why: str) -> None:
        self.ok.append((step, why))


def default_label(hostname: str) -> str:
    """The machine's name as an X-Mem-Host label (`serve.HOST_LABEL`)."""
    label = "".join(c if c.isalnum() or c in "._-" else "-" for c in hostname)[:64]
    return label or "local"


# ---- rendering ----------------------------------------------------------------------------------


def _q(arg: str) -> str:
    """One systemd ExecStart word."""
    return arg if arg and not any(c.isspace() or c in "\"'\\%$" for c in arg) else shlex.quote(arg)


def render_unit(template: str, paths: Paths, opts: Options, uv: str, mem: str | None = None) -> str:
    """deploy/mem.service for this machine: the repo as working dir, uv's absolute path, the
    config, token file and stores spelled out. Loopback HTTP with a bearer token; no `--tls` (that
    needs `tailscale serve`, docs/multi-machine.md). Without a checkout (`uv tool install`) it runs
    the installed `mem` binary from the home directory; that install needs the `models` extra."""
    serve_args = ["serve", "--http", "--host", "127.0.0.1", "--port", str(opts.port),
                  "--host-label", opts.host_label, "--config", str(paths.config_file),
                  "--token-file", str(paths.tokens)]  # fmt: skip
    cmd = (
        [uv, "run", "--extra", "models", "mem", *serve_args] if mem is None else [mem, *serve_args]
    )
    cmd += [arg for n, p in paths.stores().items() for arg in ("--store", f"{n}={p}")]
    if opts.allow_cpu:
        cmd.append("--allow-cpu")
    path = list(dict.fromkeys([str(Path(uv).parent), *([str(Path(mem).parent)] if mem else []),
                               *SYSTEM_PATH]))  # fmt: skip
    out = [f"{MARKER} from deploy/{UNIT}; `mem install --uninstall` removes it."]
    seen = set()
    for line in template.splitlines():
        if line.startswith("#"):
            continue
        key = line.split("=", 1)[0]
        if key == "WorkingDirectory":
            line = f"WorkingDirectory={paths.repo if mem is None else paths.home}"
        elif line.startswith("Environment=PATH="):
            key, line = "PATH", f"Environment=PATH={':'.join(path)}"
        elif key == "ExecStart":
            line = "ExecStart=" + " ".join(_q(a) for a in cmd)
        seen.add(key)
        out.append(line)
    missing = {"WorkingDirectory", "PATH", "ExecStart"} - seen
    if missing:
        raise MemError(f"deploy/{UNIT}: no {', '.join(sorted(missing))} line to fill in")
    return "\n".join(out) + "\n"


def default_config_text(allow_cpu: bool) -> str:
    return (
        "# mem config (architecture-v2 §3.3). Every section left out takes the pinned defaults;\n"
        "# changing embedder, reranker or retrieval needs mem reembed / mem calibrate.\n"
        f"device: {'cpu' if allow_cpu else 'cuda'}\n"
    )


# ---- install ------------------------------------------------------------------------------------


def _write(path: Path, text: str, mode: int = 0o644) -> Callable[[], None]:
    def apply() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.chmod(mode)
        os.replace(tmp, path)

    return apply


def _new_token(paths: Paths) -> Callable[[], None]:
    def apply() -> None:
        tokens = auth.TokenFile(paths.tokens)
        if TOKEN_LABEL in tokens.labels():
            tokens.revoke(TOKEN_LABEL)  # ours, and its token file is gone: replace it
        _write(paths.token, tokens.add(TOKEN_LABEL) + "\n", 0o600)()

    return apply


def _skill_files(src: Path) -> dict[Path, bytes]:
    return {p.relative_to(src): p.read_bytes() for p in sorted(src.rglob("*")) if p.is_file()}


def _copy_skill(paths: Paths) -> Callable[[], None]:
    def apply() -> None:
        if paths.skill.exists():
            shutil.rmtree(paths.skill)
        shutil.copytree(paths.assets / "skills" / "mem", paths.skill)

    return apply


def _installed(run: Runner, query: str) -> bool:
    return run.run(["systemctl", "--user", query, UNIT]).code == 0


def _registered(run: Runner, paths: Paths, name: str) -> bool:
    """Is `name` a user-scope server? `claude mcp get` has no machine-readable output (no --json on
    `get` or `list` in Claude Code 2.1), so the exit code is the signal: 0 when a server of that
    name is configured in any scope, 1 when none is. Scope is handled by where it is asked from:
    `get` searches user, project and local scope, and run in this repo it finds the plugin's project
    .mcp.json, so ask from HOME, where only a user-scope registration (or a local-scope one made
    from HOME, which nobody does) can answer. The scope line it prints is prose and is not read."""
    return run.run(["claude", "mcp", "get", name], cwd=paths.home).code == 0


def plan_install(run: Runner, paths: Paths, opts: Options) -> Plan:
    plan = Plan()
    missing = run.missing_models()
    if missing:
        plan.do("models", f"fetch the pinned models: {', '.join(missing)} (mem doctor --fetch)",
                apply=lambda: _fetch(run))  # fmt: skip
    else:
        plan.fine("models", "pinned revisions in the Hugging Face cache")
    if paths.config_file.exists():
        plan.fine("config", f"{paths.short(paths.config_file)} exists (kept as is)")
    else:
        plan.do("config", f"write {paths.short(paths.config_file)}",
                apply=_write(paths.config_file, default_config_text(opts.allow_cpu)))  # fmt: skip
    new_token = not paths.token.exists()
    if new_token:
        plan.do("token", f"add token {TOKEN_LABEL!r} to {paths.short(paths.tokens)}, the token "
                f"itself to {paths.short(paths.token)} (mode 600)", apply=_new_token(paths))  # fmt: skip
    else:
        plan.fine("token", f"{paths.short(paths.token)} exists")
    _plan_service(run, paths, opts, plan)
    if not opts.mcp:
        plan.fine("mcp", "--no-mcp: the Claude Code plugin brings the MCP servers and the skill")
        return plan
    _plan_mcp(run, paths, opts, plan, replace=new_token)
    source = _skill_files(paths.assets / "skills" / "mem")
    if paths.skill.is_dir() and _skill_files(paths.skill) == source:
        plan.fine("skill", f"{paths.short(paths.skill)} up to date")
    else:
        plan.do("skill", f"copy skills/mem to {paths.short(paths.skill)}", apply=_copy_skill(paths))
    return plan


def _fetch(run: Runner) -> None:
    failed = [f"{c.name}: {c.detail}" for c in run.fetch_models() if c.status == "fail"]
    if failed:
        raise MemError("; ".join(failed))


def _ours(paths: Paths) -> bool:
    """Did `mem install` write the unit? Only when its first line is the MARKER `render_unit`
    puts there. A hand-written one (deploy/mem.service copied by an operator, with --tls or other
    flags of their own) has none, and install and uninstall leave it alone and say so."""
    try:
        with paths.unit.open(encoding="utf-8") as f:
            return f.readline().startswith(MARKER)
    except FileNotFoundError:
        return False


def _kept_note(paths: Paths) -> str:
    return (f"{paths.short(paths.unit)} was not written by mem install (no "
            f"{MARKER.lstrip('# ')!r} marker): kept as is")  # fmt: skip


def _plan_service(run: Runner, paths: Paths, opts: Options, plan: Plan) -> None:
    enabled, active = _installed(run, "is-enabled"), _installed(run, "is-active")
    if paths.unit.exists() and not _ours(paths):
        plan.fine("service", _kept_note(paths) + ("" if enabled and active else "; starting it"))
        if not (enabled and active):
            plan.do("service", "enable and start mem", cmd=["systemctl", "--user", "enable",
                                                           "--now", UNIT])  # fmt: skip
        return
    uv = run.which("uv") or "uv"
    mem = None
    if not paths.checkout:
        mem = run.which("mem")
        if mem is None:
            raise MemError("mem is not on PATH, so the service cannot run it. Fix: `uv tool "
                           "update-shell` and open a new shell, or install from a checkout")  # fmt: skip
    text = render_unit((paths.assets / "deploy" / UNIT).read_text(encoding="utf-8"), paths, opts,
                       uv, mem)  # fmt: skip
    changed = not paths.unit.exists() or paths.unit.read_text(encoding="utf-8") != text
    if changed:
        plan.do("service", f"write {paths.short(paths.unit)}", apply=_write(paths.unit, text))
        plan.do("service", "reload systemd", cmd=["systemctl", "--user", "daemon-reload"])
    if not (enabled and active):
        plan.do("service", "enable and start mem", cmd=["systemctl", "--user", "enable", "--now",
                                                       UNIT])  # fmt: skip
    elif changed:
        plan.do("service", "restart mem on the new unit", cmd=["systemctl", "--user", "restart",
                                                              UNIT])  # fmt: skip
    if not changed and enabled and active:
        plan.fine("service", f"{UNIT} up to date, enabled and running")


def _plan_mcp(run: Runner, paths: Paths, opts: Options, plan: Plan, replace: bool) -> None:
    for name in serve.STORE_NAMES:
        server = f"mem-{name}"
        there = _registered(run, paths, server)
        if there and not replace:
            plan.fine("mcp", f"{server} registered")
            continue
        if there:
            plan.do("mcp", f"drop {server} (it has the old token)",
                    cmd=["claude", "mcp", "remove", "--scope", "user", server])  # fmt: skip
        base = ["claude", "mcp", "add", "--transport", "http", "--scope", "user", server,
                f"http://127.0.0.1:{opts.port}/{name}/mcp", "--header",
                f"X-Mem-Host: {opts.host_label}", "--header"]  # fmt: skip
        plan.do("mcp", f"register {server} in Claude Code", cmd=None,
                apply=_register(run, base, paths.token),
                shown=[*base, f"Authorization: Bearer <token from {paths.short(paths.token)}>"])  # fmt: skip


def _register(run: Runner, base: list[str], token_file: Path) -> Callable[[], None]:
    """`claude mcp add` takes HTTP headers only as arguments (`--header`; `add-json` takes the same
    on the command line, and `--client-secret` prompts only for an OAuth secret), so the bearer
    token is in the argument list of that one short-lived process, readable by other processes of
    the same user (and root) while it runs. Not in the output: `Action.shown` hides it. The exposure
    is written down in docs/install.md; the fix would be a Claude Code flag that reads a header
    from the environment or a file."""

    def apply() -> None:
        token = token_file.read_text(encoding="utf-8").strip()  # written by an earlier action
        _check(run, [*base, f"Authorization: Bearer {token}"])

    return apply


# ---- uninstall ----------------------------------------------------------------------------------


def prune_empty(path: Path, home: Path) -> None:
    """Remove the folders above `path` that its removal left empty, stopping at the first one
    with anything in it and never touching `home` or above."""
    for parent in path.parents:
        if parent == home or home not in parent.parents:
            return
        try:
            parent.rmdir()
        except OSError:
            return


def _remove(*targets: Path, home: Path | None = None) -> Callable[[], None]:
    def apply() -> None:
        for t in targets:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink(missing_ok=True)
            if home:
                prune_empty(t, home)

    return apply


def plan_uninstall(run: Runner, paths: Paths, opts: Options) -> Plan:
    plan = Plan()
    if opts.mcp:
        for name in serve.STORE_NAMES:
            if _registered(run, paths, f"mem-{name}"):
                plan.do("mcp", f"unregister mem-{name}",
                        cmd=["claude", "mcp", "remove", "--scope", "user", f"mem-{name}"])  # fmt: skip
    if paths.skill.exists():
        plan.do(
            "skill",
            f"remove {paths.short(paths.skill)}",
            apply=_remove(paths.skill, home=paths.home),
        )
    if paths.unit.exists() and not _ours(paths):
        plan.fine("service", _kept_note(paths) + ", not stopped")
    else:
        if _installed(run, "is-enabled") or _installed(run, "is-active"):
            plan.do("service", "stop and disable mem",
                    cmd=["systemctl", "--user", "disable", "--now", UNIT])  # fmt: skip
        if paths.unit.exists():
            plan.do(
                "service",
                f"remove {paths.short(paths.unit)}",
                apply=_remove(paths.unit, home=paths.home),
            )
            plan.do("service", "reload systemd", cmd=["systemctl", "--user", "daemon-reload"])
    kept = [paths.config_file, paths.tokens, paths.token]
    kept += [p.with_name(p.name + s) for p in paths.stores().values() for s in ("", "-wal", "-shm")]
    kept = [p for p in kept if p.exists()]
    if kept and opts.purge:
        plan.do(
            "purge",
            "delete " + ", ".join(paths.short(p) for p in kept),
            apply=_remove(*kept, home=paths.home),
        )
    elif kept:
        plan.fine("kept", ", ".join(paths.short(p) for p in kept) + " (--purge deletes them)")
    return plan


# ---- running ------------------------------------------------------------------------------------


def _check(run: Runner, cmd: list[str]) -> None:
    res = run.run(cmd)
    if res.code != 0:
        raise MemError(f"`{cmd[0]} {' '.join(cmd[1:4])} …` exited {res.code}: {res.out.strip()}")


def preconditions(run: Runner, opts: Options) -> tuple[list[str], list[str]]:
    """(lines to show, problems that stop the install)."""
    shown, problems = [], []
    if not opts.uninstall:
        gpu = run.gpu("cpu" if opts.allow_cpu else "cuda")
        if gpu.status == "ok":
            shown.append(f"[ok  ] gpu: {gpu.detail}")
        elif opts.allow_cpu:
            shown.append(f"[warn] gpu: {gpu.detail}; --allow-cpu: serving on the CPU")
        else:
            problems.append(f"gpu: {gpu.detail}. Tests and CI only: pass --allow-cpu")
    tools = ["systemctl", *(["claude"] if opts.mcp else []), *([] if opts.uninstall else ["uv"])]
    found = {t: run.which(t) for t in tools}
    for tool, where in found.items():
        if where is None:
            hint = " (install Claude Code, or pass --no-mcp)" if tool == "claude" else ""
            problems.append(f"{tool}: not on PATH{hint}")
    if all(found.values()):
        shown.append("[ok  ] tools: " + ", ".join(f"{t} {w}" for t, w in found.items()))
    return shown, problems


def _show(action: Action) -> list[str]:
    lines = [f"{action.step}: {action.what}"]
    cmd = action.shown or action.cmd
    if cmd:
        lines.append("       $ " + shlex.join(cmd))
    return lines


def main(opts: Options, run: Runner, paths: Paths, echo: Callable[[str], None] = print) -> int:
    """Install (or with `opts.uninstall`, undo) mem; 0 when done, 1 when refused or failed."""
    verb = "uninstall" if opts.uninstall else "install"
    echo(f"mem {verb}" + (" (dry run: nothing is changed)" if opts.dry_run else ""))
    shown, problems = preconditions(run, opts)
    for line in shown:
        echo(line)
    for line in problems:
        echo(f"[FAIL] {line}")
    if problems and not opts.dry_run:
        echo(f"mem {verb} changed nothing.")
        return 1
    try:
        plan = (plan_uninstall if opts.uninstall else plan_install)(run, paths, opts)
    except (OSError, MemError) as err:
        echo(f"[FAIL] {err}")
        return 1
    for step, why in plan.ok:
        echo(f"[ok  ] {step}: {why}")
    tag = "[would]" if opts.dry_run else "[do  ]"
    for action in plan.actions:
        head, *rest = _show(action)
        echo(f"{tag} {head}")
        for line in rest:
            echo(line)
        if opts.dry_run:
            continue
        try:
            if action.cmd:
                _check(run, action.cmd)
            if action.apply:
                action.apply()
        except (OSError, MemError) as err:
            echo(f"[FAIL] {action.step}: {err}")
            echo("Fix that and run it again: the steps already done are skipped.")
            return 1
    if not plan.actions:
        echo("nothing to undo: mem is not installed" if opts.uninstall
             else "nothing to do: mem is installed and up to date")  # fmt: skip
    if not opts.uninstall:
        _serve_note(paths, echo)
    return 1 if problems else 0


def _serve_note(paths: Paths, echo: Callable[[str], None]) -> None:
    """mem serve refuses a store without a gate (§3.3); say so rather than leave a restart loop."""
    from mem import config

    try:
        cfg = config.load(paths.config_file) if paths.config_file.exists() else None
        problems = serve.preflight(paths.stores(), cfg or config.default_config())
    except (OSError, ValueError) as err:
        problems = [str(err)]
    for line in problems:
        echo(f"[note] mem serve will not start until: {line}")


__all__ = ["Options", "Paths", "Result", "Runner", "SystemRunner", "main", "render_unit"]
