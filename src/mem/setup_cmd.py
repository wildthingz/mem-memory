"""`mem setup`: `mem install` plus the checks around it (MEM2-234, docs/decisions/0018).

One idempotent command: GPU/CUDA check, pinned models (with progress and resume), config, token,
systemd user service, MCP servers and skill in each harness, `mem doctor` (the serving gate
included), then a self-test (remember, recall and forget through the service it started). The
installing itself is `mem install`'s plan (`install.plan_install` / `plan_uninstall`); this module
adds the error explanations (each with its fix), the confirmation, the harness writers
(`setup_steps.HarnessWriter`), `--remote <host>` and the remote checks `mem doctor` runs. `mem
install` is an alias of `mem setup --yes`. Every system call goes through the `Runner`; nothing is
sent anywhere except to a remote the user named.
"""

from __future__ import annotations

import base64
import getpass
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mem import config as config_mod
from mem import doctor, serve
from mem import install as inst
from mem import setup_steps as ss
from mem.common import MemError
from mem.doctor import Check

CODE_PREFIX = "memc1_"
LATENCY_WARN_MS = 500.0
RETRIES = 3
Download = Callable[[str, str, Callable[[int, int], None]], str]
Echo = Callable[[str], None]


@dataclass(frozen=True)
class SetupOptions(inst.Options):
    yes: bool = False
    remote: str | None = None
    code: str | None = None


# ---- the download: progress and resume ----------------------------------------------------------


def gated_message(repo: str) -> str:
    return (f"{repo} is gated and its terms are not accepted. Fix: open https://huggingface.co/"
            f"{repo}, accept the terms with your Hugging Face account, run `hf auth login`, then "
            "rerun `mem setup`")  # fmt: skip


def _is_gated(err: BaseException) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return any(w in text for w in ("gated", "401", "403", "unauthorized", "restricted"))


def download(dl: Download, repo: str, rev: str, echo: Echo, retries: int = RETRIES) -> str:
    """Fetch one pinned repo. A dropped connection resumes (the hub client keeps the partial file
    and continues from its size); a gated repo stops at once with the page to accept it on."""
    last = -1

    def progress(done: int, total: int) -> None:
        nonlocal last
        pct = 100 * done // total if total else 0
        if pct // 10 != last // 10 or pct == 100 > last:
            echo(f"       {repo}: {pct}%")
        last = pct

    for attempt in range(retries + 1):
        try:
            path = dl(repo, rev, progress)
        except Exception as err:
            if _is_gated(err):
                raise MemError(gated_message(repo)) from err
            if attempt == retries:
                raise MemError(f"downloading {repo} failed: {type(err).__name__}: {err}. Fix: check "
                               "the network, then rerun `mem setup`; it resumes where it "
                               "stopped") from err  # fmt: skip
            echo(f"       {repo}: interrupted ({err}); resuming ({attempt + 1}/{retries})")
            continue
        if last < 100:
            echo(f"       {repo}: 100%")
        return path
    raise AssertionError("unreachable")


def hub_download(repo: str, rev: str, progress: Callable[[int, int], None]) -> str:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class Bar(tqdm):  # type: ignore[type-arg]
        def update(self, n: float | None = 1) -> bool | None:
            shown = super().update(n)
            if self.total and self.unit == "B":
                progress(int(self.n), int(self.total))
            return shown

    return snapshot_download(repo, revision=rev, tqdm_class=Bar)


def fetch_models(echo: Echo, dl: Download = hub_download) -> list[Check]:
    out = []
    for pin in doctor.configured_pins(None):  # the default config's, as `mem install` writes it
        name = f"fetch {pin.id}"
        try:
            path = download(dl, pin.id, pin.revision, echo)
        except MemError as err:
            out.append(Check(name, "fail", str(err)))
            continue
        after = doctor.check_model(pin, name=name)
        out.append(
            after if after.status != "ok" else Check(name, "ok", f"{pin.revision[:8]} -> {path}")
        )
    return out


# ---- the self-test ------------------------------------------------------------------------------

SELFTEST_TEXT = "mem setup self-test: the rubber duck lives in the blue drawer."
SERVICE_WAIT_S = 180.0  # mem.service loads both models before it answers


def selftest(url: str, token: str, host_label: str,
             wait: float = SERVICE_WAIT_S) -> list[tuple[str, bool, str]]:  # fmt: skip
    """remember, recall, forget through the running service at `url` (its `/personal/mcp`) with
    the installed token: its models, reranker and gate, nothing loaded in this process. The memory
    is retracted at the end. (step, passed, detail) for each step run."""
    deadline = time.monotonic() + wait
    while (p := probe(url, token)).status is None and time.monotonic() < deadline:
        time.sleep(1.0)
    fix = "Fix: `systemctl --user status mem.service` and `journalctl --user -u mem.service`"
    if p.status is None:
        return [("service", False, f"no answer on {url} after {wait:.0f} s ({p.error}). {fix}")]
    if p.status in (401, 403):
        why = (f"{url} refused the token in the mem token file ({p.status}). Fix: `mem token "
               "list` should show mem-install; rerun `mem setup`")  # fmt: skip
        return [("service", False, why)]
    if p.status != 200:
        return [("service", False, f"{url} answered HTTP {p.status} to initialize. {fix}")]
    import asyncio

    tools = asyncio.run(_selftest_tools(url, token, host_label))
    return [("service", True, f"{url} answered in {p.ms:.0f} ms"), *tools]


async def _selftest_tools(url: str, token: str, host_label: str) -> list[tuple[str, bool, str]]:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    results: list[tuple[str, bool, str]] = []
    headers = {"Authorization": f"Bearer {token}", "X-Mem-Host": host_label}
    try:
        async with (
            httpx2.AsyncClient(headers=headers, timeout=60, trust_env=False) as http,
            Client(streamable_http_client(url, http_client=http), mode="legacy") as client,
        ):

            async def call(tool: str, **args: Any) -> dict[str, Any]:
                res = await client.call_tool(tool, args)
                if res.is_error:
                    raise MemError(" ".join(getattr(c, "text", "") for c in res.content))
                return dict(res.structured_content or {})

            async def recalled() -> list[str]:
                rows = (await call("recall", query="rubber duck blue drawer")).get("rows", [])
                return [r["id"] for r in rows]

            saved = await call("remember", text=SELFTEST_TEXT, type="fact",
                               about=["new:topic:Self-test"])  # fmt: skip
            mid = saved.get("id")
            results.append(("remember", saved.get("status") == "saved", f"saved {mid}"))
            found = mid in await recalled()
            results.append(
                ("recall", found, "found it" if found else "the memory did not come back")
            )
            gone = await call("forget", id=mid, reason="mem setup self-test")
            ok = gone.get("status") == "retracted" and mid not in await recalled()
            results.append(
                ("forget", ok, "retracted and no longer recalled" if ok else "still recalled")
            )
    except Exception as err:  # noqa: BLE001  any failure is a FAIL line, not a crash
        while isinstance(err, ExceptionGroup) and err.exceptions:  # the client's task groups
            err = err.exceptions[0]
        results.append(("error", False, f"{type(err).__name__}: {err}"))
    return results


# ---- diagnoses: what is wrong, and the fix ------------------------------------------------------


def _gpu_problem(run: Any, detail: str) -> str:
    driver, built = run.cuda_versions()
    if driver is None:
        return ("gpu: no NVIDIA GPU found (nvidia-smi is missing or lists no device). mem serves "
                "only on an NVIDIA GPU. Fix: run mem on a machine with one; tests and CI only: "
                "pass --allow-cpu")  # fmt: skip
    if built is None:
        return (f"gpu: the driver supports CUDA {driver} but torch is a CPU build. Fix: reinstall "
                "mem so torch matches the driver: `uv tool install --reinstall <repo>`")  # fmt: skip
    return (f"gpu: torch cannot use CUDA ({detail}): the driver supports CUDA {driver} but torch "
            f"was built for CUDA {built}. Fix: update the NVIDIA driver until `nvidia-smi` reports "
            f"CUDA {built} or newer, or reinstall torch for CUDA {driver}")  # fmt: skip


def diagnose(run: Any, opts: inst.Options) -> tuple[list[str], list[str]]:
    """`install.preconditions` with the GPU failure explained, plus the systemd user session and
    the port. (lines to show, problems that stop the setup)."""
    shown, problems = inst.preconditions(run, opts)
    problems = [_gpu_problem(run, p.partition("gpu: ")[2].split(". ")[0]) if p.startswith("gpu:")
                else p for p in problems]  # fmt: skip
    if opts.uninstall or run.which("systemctl") is None:
        return shown, problems
    if run.run(["systemctl", "--user", "show-environment"]).code != 0:
        problems.append("systemd: no systemd user session (`systemctl --user` cannot reach it). "
                        "Fix: log in with a full session (ssh or a desktop login, not `su`), or "
                        "run `sudo loginctl enable-linger $USER` and log in again")  # fmt: skip
    else:
        shown.append("[ok  ] systemd: user session reachable")
    if not run.port_free(opts.port) and not inst._installed(run, "is-active"):
        problems.append(f"port {opts.port} is in use by another process. Fix: stop it, or choose "
                        f"another port with `mem setup --port {opts.port + 1}`")  # fmt: skip
    return shown, problems


def explain(step: str, err: str) -> str:
    """Add the fix to an error from a step that came without one."""
    if step == "models" and "https://huggingface.co/" not in err and any(
        w in err.lower() for w in ("gated", "401", "403")):  # fmt: skip
        found = re.search(r"fetch (\S+?):", err)
        return gated_message(found.group(1) if found else doctor.configured_pins(None)[0].id)
    return err


# ---- remote codes and probes ---------------------------------------------------------------------


def make_code(host: str, token: str) -> str:
    raw = json.dumps({"h": host, "t": token}, separators=(",", ":")).encode()
    return CODE_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def read_code(code: str) -> tuple[str, str]:
    try:
        body = code.strip().removeprefix(CODE_PREFIX)
        if body == code.strip():
            raise ValueError("no prefix")
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return str(data["h"]), str(data["t"])
    except (ValueError, KeyError, TypeError) as err:
        raise MemError("not a mem code. Fix: on the other machine run "
                       "`mem token add <label> --code --host <its tailnet name>` and paste what "
                       "it prints") from err  # fmt: skip


def probe(url: str, token: str, timeout: float = 10.0) -> inst.Probe:
    """An MCP `initialize` against `url`, with the bearer token."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {},
        "clientInfo": {"name": "mem-setup", "version": "1"}}}).encode()  # fmt: skip
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"})  # fmt: skip
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as err:
        status = err.code
    except (urllib.error.URLError, OSError, ValueError) as err:
        return inst.Probe(None, 0.0, str(getattr(err, "reason", err)))
    return inst.Probe(status, (time.perf_counter() - start) * 1000, "")


def _problem(host: str, p: inst.Probe) -> str | None:
    if p.status is None:
        return (f"{host} is unreachable ({p.error}). Fix: is that machine up and on the tailnet? "
                f"Check `tailscale status`, `tailscale ping {host}`, and that `mem serve --tls` "
                "or `tailscale serve` runs there")  # fmt: skip
    if p.status in (401, 403):
        return (f"{host} refused the token ({p.status}). Fix: on that machine run "
                f"`mem token add <label> --code --host {host}` for a new code, then "
                f"`mem setup --remote {host}`")  # fmt: skip
    if p.status != 200:
        return f"{host} answered HTTP {p.status} to initialize. Fix: check `mem serve` there"
    return None


def _remotes(paths: inst.Paths) -> dict[str, dict[str, str]]:
    try:
        return json.loads((paths.config / "remotes.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except ValueError as err:
        raise MemError(f"{paths.short(paths.config / 'remotes.json')}: {err}") from err


def check_remotes(paths: inst.Paths, run: Any) -> list[Check]:
    """`mem doctor` for each remote: reachability, then token, then latency (ADR 0018 §4)."""
    out = []
    for name, remote in _remotes(paths).items():
        label = f"remote {name}"
        try:
            token = (paths.config / f"remote-{name}.token").read_text(encoding="utf-8").strip()
        except OSError:
            out.append(Check(label, "fail", f"no token file. Fix: `mem setup --remote "
                             f"{remote['host']}`"))  # fmt: skip
            continue
        p = run.probe(f"{remote['url']}/personal/mcp", token)
        if (why := _problem(remote["host"], p)) is not None:
            out.append(Check(label, "fail", why))
        elif p.ms > LATENCY_WARN_MS:
            out.append(Check(label, "warn", f"{remote['host']}: {p.ms:.0f} ms round trip "
                             f"(slow; over {LATENCY_WARN_MS:.0f} ms)"))  # fmt: skip
        else:
            out.append(Check(label, "ok", f"{remote['host']} reachable, token accepted, "
                             f"{p.ms:.0f} ms"))  # fmt: skip
    return out


# ---- Claude Code as a harness writer -------------------------------------------------------------


class ClaudeCodeWriter:
    """Claude Code through `claude mcp`, which owns ~/.claude.json (so no byte-for-byte backup:
    `remove` drops the servers by name)."""

    name = "Claude Code"

    def __init__(self, home: Path):
        self.home = home
        self.added: list[str] = []

    def detect(self, ctx: ss.Context) -> bool:
        return ctx.runner.which("claude") is not None

    def _get(self, ctx: ss.Context, name: str) -> bool:
        return ctx.runner.run(["claude", "mcp", "get", name], cwd=self.home).code == 0

    def configured(self, ctx: ss.Context, endpoints: list[ss.Endpoint]) -> bool:
        return all(self._get(ctx, e.name) for e in endpoints)

    def add(self, ctx: ss.Context, endpoints: list[ss.Endpoint]) -> None:
        for e in endpoints:
            if self._get(ctx, e.name):
                inst._check(ctx.runner, ["claude", "mcp", "remove", "--scope", "user", e.name])
            token = e.token_file.read_text(encoding="utf-8").strip()
            cmd = ["claude", "mcp", "add", "--transport", "http", "--scope", "user", e.name, e.url]
            for k, v in e.headers:
                cmd += ["--header", f"{k}: {v}"]
            inst._check(ctx.runner, [*cmd, "--header", f"Authorization: Bearer {token}"])

    def installed(self, ctx: ss.Context) -> bool:
        return any(self._get(ctx, f"mem-{n}") for n in serve.STORE_NAMES)

    def remove(self, ctx: ss.Context) -> None:
        """Nothing to restore: uninstall drops the servers by name (`install.plan_uninstall`,
        `_plan_uninstall_extras`)."""


# ---- main ---------------------------------------------------------------------------------------


def _configured(
    h: Any, ctx: ss.Context, endpoints: list[ss.Endpoint], skipped: list[str]
) -> bool | None:
    """`h.configured`, or None (noted in `skipped`) when its config file cannot be read: that one
    harness is left out and the others go on."""
    try:
        return h.configured(ctx, endpoints)
    except MemError as err:
        skipped.append(f"{h.name} skipped: {err}")
        return None


def _installed(h: Any, ctx: ss.Context, skipped: list[str]) -> bool:
    """`h.installed`, or False (noted in `skipped`) when its config file cannot be read, as
    `_configured` does for setup."""
    try:
        return h.installed(ctx)
    except MemError as err:
        skipped.append(f"{h.name} skipped: {err}")
        return False


def _or_skip(h: Any, change: Callable[[], None], echo: Echo) -> Callable[[], None]:
    """Run `change` (a harness's add or remove); when mem cannot change that harness's file, say
    so with a `[warn]` and go on with the other steps: the file is left as it is."""

    def apply() -> None:
        try:
            change()
        except MemError as err:
            echo(f"[warn] mcp: {h.name} skipped: {err}")

    return apply


def _local_endpoints(paths: inst.Paths, opts: inst.Options) -> list[ss.Endpoint]:
    return [ss.Endpoint(f"mem-{n}", f"http://127.0.0.1:{opts.port}/{n}/mcp", paths.token,
                        (("X-Mem-Host", opts.host_label),)) for n in serve.STORE_NAMES]  # fmt: skip


def _remote_name(host: str) -> tuple[str, str]:
    """(short name, base url) for `work.tail1234.ts.net` or `https://host:8443`."""
    url = host.rstrip("/") if "://" in host else f"https://{host}"
    name = (urlparse(url).hostname or host).split(".")[0]
    return "".join(c if c.isalnum() or c in "_-" else "-" for c in name) or "remote", url


def _remote_servers(paths: inst.Paths, name: str, url: str) -> list[ss.Endpoint]:
    token = paths.config / f"remote-{name}.token"
    return [ss.Endpoint(f"mem-{name}-{n}", f"{url}/{n}/mcp", token) for n in serve.STORE_NAMES]


def _confirm(opts: SetupOptions, plan: inst.Plan, ask: Callable[[str], str], echo: Echo) -> bool:
    if opts.yes or opts.dry_run or not plan.actions:
        return True
    echo("This will:")
    for action in plan.actions:
        echo(f"  - {action.step}: {action.what}")
    return ask("Proceed? [Y/n] ").strip().lower() in ("", "y", "yes")


def _run_actions(plan: inst.Plan, run: Any, opts: inst.Options, echo: Echo) -> bool:
    tag = "[would]" if opts.dry_run else "[do  ]"
    for action in plan.actions:
        head, *rest = inst._show(action)
        echo(f"{tag} {head}")
        for line in rest:
            echo(line)
        if opts.dry_run:
            continue
        try:
            if action.cmd:
                inst._check(run, action.cmd)
            if action.apply:
                action.apply()
        except (OSError, MemError) as err:
            echo(f"[FAIL] {action.step}: {explain(action.step, str(err))}")
            echo("Fix that and run it again: the steps already done are skipped.")
            return False
    return True


def _context(run: Any, paths: inst.Paths) -> ss.Context:
    return ss.Context(runner=run, backups=ss.Backups(paths.config / "backups"), home=paths.home)


def main(opts: SetupOptions, run: Any, paths: inst.Paths, echo: Echo = print,
         ask: Callable[[str], str] = input,
         ask_secret: Callable[[str], str] = getpass.getpass,
         writers: list[ss.HarnessWriter] | None = None) -> int:  # fmt: skip
    """Set mem up (or with `uninstall`, undo it; with `remote`, add that machine); 0 when done."""
    ctx = _context(run, paths)
    harnesses: list[Any] = [ClaudeCodeWriter(paths.home)] if opts.mcp else []
    harnesses += ss.harness_writers() if writers is None else writers
    verb = "uninstall" if opts.uninstall else "setup"
    echo(f"mem {verb}" + (f" --remote {opts.remote}" if opts.remote else "")
         + (" (dry run: nothing is changed)" if opts.dry_run else ""))  # fmt: skip
    try:
        if opts.remote:
            return _remote(opts, run, paths, ctx, harnesses, echo, ask, ask_secret)
        return _local(opts, run, paths, ctx, harnesses, echo, ask)
    except (OSError, MemError) as err:
        echo(f"[FAIL] {err}")
        echo(f"mem {verb} stopped. Fix that and run it again: the steps already done are skipped.")
        return 1


def _local(opts: SetupOptions, run: Any, paths: inst.Paths, ctx: ss.Context,
           harnesses: list[Any], echo: Echo, ask: Callable[[str], str]) -> int:  # fmt: skip
    verb = "uninstall" if opts.uninstall else "setup"
    shown, problems = diagnose(run, opts)
    for line in shown:
        echo(line)
    for line in problems:
        echo(f"[FAIL] {line}")
    if problems and not opts.dry_run:
        echo(f"mem {verb} changed nothing.")
        return 1
    local = _local_endpoints(paths, opts)
    others = [h for h in harnesses if not isinstance(h, ClaudeCodeWriter)]
    plan = (inst.plan_uninstall if opts.uninstall else inst.plan_install)(run, paths, opts)
    skipped: list[str] = []
    if opts.uninstall:
        _plan_uninstall_extras(plan, run, paths, opts, ctx, others, skipped, echo)
    else:
        for h in others:
            if not h.detect(ctx):
                continue
            done = _configured(h, ctx, local, skipped)
            if done:
                plan.fine("mcp", f"{h.name} already has the mem servers")
            elif done is not None:
                plan.do("mcp", f"add the mem servers to {h.name}",
                        apply=_or_skip(h, lambda h=h: h.add(ctx, local), echo))  # fmt: skip
    for step, why in plan.ok:
        echo(f"[ok  ] {step}: {why}")
    if not opts.uninstall and (paths.config.parent / "Claude").is_dir():
        echo("[note] Claude Desktop: setup does not configure it (its config takes only stdio "
             "servers; see docs/harnesses.md). For the recall and save rules, paste the output of "
             "`mem instructions --harness claude-desktop` into Settings > Profile")  # fmt: skip
    for why in skipped:
        echo(f"[warn] mcp: {why}")
    changed = bool(plan.actions)
    if not _confirm(opts, plan, ask, echo):
        echo("Cancelled: nothing was changed.")
        return 1
    failed = not _run_actions(plan, run, opts, echo)
    if failed:
        return 1
    if opts.uninstall:
        if not changed:
            echo("nothing to undo: mem is not installed")
        with_backups = paths.config / "backups"
        if with_backups.is_dir() and not any(with_backups.iterdir()) and not opts.dry_run:
            with_backups.rmdir()
            inst.prune_empty(with_backups, paths.home)
        return 1 if problems else 0
    if not changed:
        echo("nothing to do: mem is installed and up to date")
    inst._serve_note(paths, echo)
    ok = _verify(opts, run, paths, echo)
    return 0 if ok and not problems else 1


def _config(paths: inst.Paths) -> config_mod.Config | None:
    """The config `mem serve` runs on (None: the default)."""
    if not paths.config_file.exists():
        return None
    try:
        return config_mod.load(paths.config_file)
    except config_mod.ConfigError as err:
        raise MemError(f"{paths.short(paths.config_file)}: {err}") from err


def _verify(opts: SetupOptions, run: Any, paths: inst.Paths, echo: Echo) -> bool:
    """`mem doctor`, then the self-test. Both are checks, so they run on every setup."""
    tag = "[would]" if opts.dry_run else "[do  ]"
    url = f"http://127.0.0.1:{opts.port}/personal/mcp"
    echo(f"{tag} doctor: mem doctor")
    echo(f"{tag} self-test: remember, recall and forget through {url}, then retract the memory")
    if opts.dry_run:
        return True
    checks = run.doctor(list(paths.stores().values()), "cpu" if opts.allow_cpu else "cuda",
                        _config(paths))  # fmt: skip
    checks += check_remotes(paths, run)
    for c in checks:
        echo(f"[{ {'ok': 'ok  ', 'warn': 'warn', 'fail': 'FAIL'}[c.status] }] {c.name}: {c.detail}")
    if any(c.status == "fail" for c in checks):
        echo("doctor: FAIL. Fix the lines marked FAIL, then run `mem setup` again.")
        return False
    results = run.selftest(url, paths.token.read_text(encoding="utf-8").strip(), opts.host_label)
    for name, passed, detail in results:
        echo(f"       {name}: {'PASS' if passed else 'FAIL'} ({detail})")
    good = bool(results) and all(p for _, p, _ in results)
    echo(f"self-test: {'PASS' if good else 'FAIL'}")
    if not good:
        echo(
            "Fix: run `mem doctor` and `journalctl --user -u mem.service`, then `mem setup` again."
        )
    return good


def _plan_uninstall_extras(plan: inst.Plan, run: Any, paths: inst.Paths, opts: SetupOptions,
                           ctx: ss.Context, others: list[Any], skipped: list[str],
                           echo: Echo) -> None:  # fmt: skip
    remotes = _remotes(paths)
    for name, remote in remotes.items():
        for e in _remote_servers(paths, name, remote["url"]):
            if opts.mcp and inst._registered(run, paths, e.name):
                plan.do("mcp", f"unregister {e.name}",
                        cmd=["claude", "mcp", "remove", "--scope", "user", e.name])  # fmt: skip
    for h in others:
        if _installed(h, ctx, skipped):  # by name, whatever port, label or remote setup used
            plan.do("mcp", f"remove the mem servers from {h.name} (config restored from backup)",
                    apply=_or_skip(h, lambda h=h: h.remove(ctx), echo))  # fmt: skip
    extra = [paths.config / "remotes.json", *paths.config.glob("remote-*.token")]
    extra = [p for p in extra if p.exists()]
    if extra and opts.purge:
        plan.do("purge", "delete " + ", ".join(paths.short(p) for p in extra),
                apply=inst._remove(*extra, home=paths.home))  # fmt: skip
    elif extra:
        plan.fine("kept", ", ".join(paths.short(p) for p in extra) + " (--purge deletes them)")


def _remote(opts: SetupOptions, run: Any, paths: inst.Paths, ctx: ss.Context,
            harnesses: list[Any], echo: Echo, ask: Callable[[str], str],
            ask_secret: Callable[[str], str]) -> int:  # fmt: skip
    if opts.uninstall:
        raise MemError("--remote adds a machine; `mem setup --uninstall` removes all of them")
    active = [h for h in harnesses if h.detect(ctx)]
    if not active:
        raise MemError("no harness to add the server to: Claude Code is not on PATH and no other "
                       "harness was found. Fix: install one, then run this again")  # fmt: skip
    name, url = _remote_name(opts.remote or "")
    host = urlparse(url).hostname or url
    if opts.code:
        code_host, token = read_code(opts.code)
        if code_host != opts.remote:
            raise MemError(f"that code is for {code_host}, not {opts.remote}. Fix: run "
                           f"`mem setup --remote {code_host} --code ...` or ask for a new code")  # fmt: skip
    else:
        token = ask_secret(f"Token for {host} (`mem token add <label>` there): ").strip()
    p = run.probe(f"{url}/personal/mcp", token)
    problem = _problem(host, p)
    if problem:
        echo(f"[FAIL] remote: {problem}")
        echo("mem setup --remote changed nothing.")
        return 1
    echo(f"[ok  ] remote: {host} reachable, token accepted, {p.ms:.0f} ms")
    servers = _remote_servers(paths, name, url)
    token_file = paths.config / f"remote-{name}.token"
    remotes = _remotes(paths)
    entry = {"host": host, "url": url}
    same_token = token_file.exists() and token_file.read_text(encoding="utf-8").strip() == token
    plan = inst.Plan()
    if same_token and remotes.get(name) == entry:
        plan.fine("token", f"{paths.short(token_file)} up to date")
    else:
        plan.do("token", f"store the token in {paths.short(token_file)} (mode 600) and list {host} "
                f"in {paths.short(paths.config / 'remotes.json')}",
                apply=lambda: _store_remote(paths, name, entry, token))  # fmt: skip
    skipped: list[str] = []
    for h in active:
        done = _configured(h, ctx, servers, skipped)
        if done is None:
            continue
        if same_token and done:
            plan.fine("mcp", f"{h.name} already has {', '.join(e.name for e in servers)}")
        else:
            plan.do("mcp", f"add {', '.join(e.name for e in servers)} to {h.name}",
                    apply=_or_skip(h, lambda h=h: h.add(ctx, servers), echo))  # fmt: skip
    for step, why in plan.ok:
        echo(f"[ok  ] {step}: {why}")
    for why in skipped:
        echo(f"[warn] mcp: {why}")
    if not _confirm(opts, plan, ask, echo):
        echo("Cancelled: nothing was changed.")
        return 1
    if not _run_actions(plan, run, opts, echo):
        return 1
    if not plan.actions:
        echo(f"nothing to do: {host} is already set up")
    return 0


def _store_remote(paths: inst.Paths, name: str, entry: dict[str, str], token: str) -> None:
    inst._write(paths.config / f"remote-{name}.token", token + "\n", 0o600)()
    remotes = {**_remotes(paths), name: entry}
    inst._write(paths.config / "remotes.json", json.dumps(remotes, indent=1) + "\n")()


__all__ = ["SetupOptions", "check_remotes", "download", "main", "make_code", "read_code",
           "selftest"]  # fmt: skip
