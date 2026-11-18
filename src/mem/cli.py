"""`mem` command line (architecture-v2 §4.2). Thin wrappers over library code."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from mem import auth as auth_mod
from mem import backup as backup_mod
from mem import calibrate as calibrate_mod
from mem import catalog, guard, models_http, schema
from mem import config as config_mod
from mem import doctor as doctor_mod
from mem import hook as hook_mod
from mem import reembed as reembed_mod
from mem import release as release_mod
from mem import render as render_mod
from mem import review as review_mod
from mem import serve as serve_mod
from mem.common import MemError, fmt_ts, local_tz
from mem.gate import Gate, gate_json, write_gate
from mem.store import init_store
from mem.testing import Embedder, HashEmbedder, OverlapReranker, Reranker

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """mem: local long-term memory for Claude Code."""


DATA_HOME = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "mem"
CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "mem"
STATE_HOME = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "mem"

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="config.yaml. Default: ~/.config/mem/config.yaml if present, "
                 "else the built-in pinned defaults."),
]  # fmt: skip

_ICON = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}


@app.command()
def doctor(
    fetch: Annotated[
        bool, typer.Option("--fetch", help="Download pinned model revisions.")
    ] = False,
    store: Annotated[
        list[Path] | None, typer.Option("--store", help="Store file whose schema to check.")
    ] = None,
    allow_unencrypted: Annotated[
        bool,
        typer.Option(
            doctor_mod.OPT_OUT,
            help="Test machines only: an unencrypted store "
            "folder is a warning, not a failure. Logged to --opt-out-log.",
        ),
    ] = False,
    opt_out_log: Annotated[
        Path,
        typer.Option("--opt-out-log", help="Where --allow-unencrypted is logged (JSON lines)."),
    ] = STATE_HOME / "doctor.jsonl",
    config: ConfigOpt = None,
) -> None:
    """Check extensions, models, device, fingerprint, schema version, serving gate and encryption at
    rest.

    Every run is a serving-host check: it FAILS (exit 1) unless the device is CUDA with at least
    4 GB free VRAM (mem serves only on an NVIDIA GPU), and when the configured serving set
    (models.embedder + models.reranker, batch size, pool) needs more than 12 GiB of VRAM."""
    folder = store[0].parent if store else DATA_HOME
    cfg = _config(config, "st")
    _pinned(cfg, "Select a pinned model with models.embedder / models.reranker (ADR 0025).")
    results = doctor_mod.run(fetch=fetch, stores=store, at_rest=folder,
                             allow_unencrypted=allow_unencrypted, device=cfg.device,
                             at_rest_required=cfg.at_rest_required, cfg=cfg)  # fmt: skip
    from mem import install as install_mod
    from mem import setup_cmd

    paths = install_mod.Paths.here()
    if not store:  # the gates of the stores `mem serve` opens, created yet or not
        results += doctor_mod.check_gate(list(paths.stores().values()), cfg)
    results += setup_cmd.check_remotes(paths, install_mod.SystemRunner())
    for c in results:
        typer.echo(f"[{_ICON[c.status]}] {c.name}: {c.detail}")
    waived = [c for c in results if c.name == "at-rest" and c.status == "warn"]
    if allow_unencrypted and waived:
        log = auth_mod.AuditLog(opt_out_log)
        log.write({"ts": datetime.now(UTC).isoformat(timespec="seconds"),
                   "event": "at-rest-opt-out", "path": str(folder), "detail": waived[0].detail})  # fmt: skip
        log.close()
        typer.echo(f"at-rest opt-out logged to {log.path}", err=True)
    if any(c.status == "fail" for c in results):
        raise typer.Exit(1)


StubOpt = Annotated[
    str, typer.Option(help="st = the pinned models; hash = deterministic test stubs.")
]


def _config(path: Path | None, embedder: str) -> config_mod.Config:
    """The config in force: the file (or the default), with the stub sections for `hash`."""
    if embedder not in ("st", "hash"):
        raise typer.BadParameter("--embedder is st or hash")
    if path is None and (CONFIG_HOME / "config.yaml").exists():
        path = CONFIG_HOME / "config.yaml"
    try:
        cfg = config_mod.load(path) if path else config_mod.default_config()
    except (OSError, config_mod.ConfigError) as err:
        raise typer.BadParameter(str(err), param_hint="--config") from err
    return config_mod.stub_config(cfg) if embedder == "hash" else cfg


def _pinned(cfg: config_mod.Config, footer: str) -> None:
    """Exit with a [FAIL] line unless the config's model sections are pinned catalog entries."""
    try:
        catalog.embedder_entry(cfg.embedder)
        catalog.reranker_entry(cfg.reranker)
    except ValueError as err:
        _fail([str(err)], footer)


def _stores(store: list[str] | None) -> dict[str, Path]:
    try:
        return serve_mod.parse_stores(store) if store else {
            n: DATA_HOME / f"{n}.db" for n in serve_mod.STORE_NAMES
        }  # fmt: skip
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err


def _fail(lines: list[str], footer: str) -> NoReturn:
    for line in lines:
        typer.echo(f"[FAIL] {line}", err=True)
    typer.echo(footer, err=True)
    raise typer.Exit(1)


@app.command(name="import")
def import_(
    vault: Annotated[
        Path, typer.Argument(help="The Obsidian vault folder.", exists=True, file_okay=False)
    ],
    store: Annotated[
        list[str] | None,
        typer.Option(
            "--store",
            help="name=path, name = personal|work. Default: both in "
            "~/.local/share/mem/ (created if missing).",
        ),
    ] = None,
    render_dir: Annotated[
        str, typer.Option(help="Vault folder holding `mem render` output; never imported.")
    ] = "export",
    tz: Annotated[
        str | None,
        typer.Option(
            help="IANA zone for date-only `created` values. Default: TZ, else "
            "/etc/localtime, else UTC."
        ),
    ] = None,
    port: Annotated[
        int, typer.Option(help="The local mem serve whose embedder embeds new chunks (D5).")
    ] = serve_mod.DEFAULT_PORT,
    config: ConfigOpt = None,
    embedder: StubOpt = "st",
) -> None:
    """Import the vault into the stores: idempotent, one transaction per file (§2.4).

    Folder routing is `personal/` and `projects/` → personal, `work/` → work, plus the
    `vault.routes` overrides of config.yaml. Loads no model: new chunks are embedded by the
    running `mem serve` (D5; `--embedder hash` uses the test stub). Exit 1 if a file could
    not be read; the others are imported anyway."""
    from mem.importer import run_import
    from mem.store import Store

    cfg = _config(config, embedder)
    paths = _stores(store)
    if embedder == "st":
        try:
            emb: Embedder = models_http.connect(port, cfg)[0]
        except models_http.ServeUnavailable as err:
            _fail([str(err)], "Nothing was imported.")
    else:
        emb = HashEmbedder(int(cfg.embedder["dim"]))
    opened: dict[str, Store] = {}
    try:
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            opened[name] = Store.open(path, emb, scope=name, fp=cfg.fp_embed(),
                                      tz=tz or local_tz(), retrieval=cfg.retrieval,
                                      ingest=cfg.ingest)  # fmt: skip
        report = run_import(vault, opened, routes=cfg.vault_routes, render_dir=render_dir)
    except (MemError, sqlite3.Error, OSError) as err:
        _fail([str(err)], "The import stopped; re-running it resumes (every file is atomic).")
    finally:
        for s in opened.values():
            s.close()
    typer.echo("  ".join(f"{k} {getattr(report, k)}" for k in
                         ("added", "changed", "renamed", "deleted", "restored", "unchanged")))  # fmt: skip
    for rel in report.unrouted:
        typer.echo(f"[warn] not imported, folder has no route: {rel}", err=True)
    for warning in report.warnings:
        typer.echo(f"[warn] {warning}", err=True)
    for rel, why in report.errors:
        typer.echo(f"[FAIL] {rel}: {why}", err=True)
    if report.errors:
        raise typer.Exit(1)


@app.command()
def serve(
    http: Annotated[bool, typer.Option("--http", help="Serve HTTP (default: stdio).")] = False,
    host: Annotated[str, typer.Option(help="HTTP bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port.")] = serve_mod.DEFAULT_PORT,
    host_label: Annotated[
        str,
        typer.Option(
            help="Host label for stdio events. HTTP events take the client's X-Mem-Host "
            "header, else 'unknown'."
        ),
    ] = socket.gethostname(),
    store: Annotated[
        list[str] | None,
        typer.Option(
            "--store", help="name=path, name = personal|work. Default: both in ~/.local/share/mem/."
        ),
    ] = None,
    tz: Annotated[
        str | None,
        typer.Option(
            help="IANA zone for date-only inputs. Default: TZ, else /etc/localtime, else UTC."
        ),
    ] = None,
    device: Annotated[
        str | None, typer.Option(help="cuda, auto or cpu. Default: config.yaml `device`.")
    ] = None,
    allow_cpu: Annotated[
        bool,
        typer.Option("--allow-cpu", help="Tests and CI only: serve on the CPU (stub models)."),
    ] = False,
    tailscale: Annotated[
        bool,
        typer.Option(
            "--tailscale",
            help="HTTP: also listen on this machine's Tailscale address (P5); --host stays bound. "
            "If Tailscale is not up, warn and serve on --host only.",
        ),
    ] = False,
    require_tailscale: Annotated[
        bool,
        typer.Option(
            "--require-tailscale",
            help="As --tailscale, but refuse to start when the tailnet address is unavailable.",
        ),
    ] = False,
    allow_host: Annotated[
        list[str] | None,
        typer.Option(
            "--allow-host",
            help="HTTP: another Host name the MCP routes answer (a name, *.suffix or CIDR); "
            "repeatable. Loopback and --host always; --tailscale adds its address and name.",
        ),
    ] = None,
    token_file: Annotated[
        Path | None,
        typer.Option(
            "--token-file",
            help="HTTP: require `Authorization: Bearer <token>` on the MCP routes, tokens from "
            "this file (`mem token add`). P7.",
        ),
    ] = None,
    tls: Annotated[
        bool,
        typer.Option(
            "--tls",
            help="HTTP: the tailnet reaches mem only through `tailscale serve` HTTPS: bind loopback "
            "only, answer the MagicDNS name, refuse plaintext. Needs --token-file. P7.",
        ),
    ] = False,
    oauth: Annotated[
        bool,
        typer.Option(
            "--oauth",
            help="With --tls: also be an OAuth server for a claude.ai connector; the consent page "
            "asks for one of your mem tokens. P7, optional.",
        ),
    ] = False,
    oauth_state: Annotated[
        Path, typer.Option(help="OAuth clients and token hashes (--oauth).")
    ] = auth_mod.DEFAULT_OAUTH_STATE,
    rate_limit: Annotated[
        int, typer.Option(help="HTTP: MCP requests a minute per client (bursts as large); 0 = off.")
    ] = auth_mod.DEFAULT_RATE_LIMIT,
    audit_log: Annotated[
        Path, typer.Option(help="HTTP: one JSON line per request, every route and answer.")
    ] = auth_mod.DEFAULT_AUDIT_LOG,
    models_only: Annotated[
        bool,
        typer.Option(
            "--models-only",
            help="HTTP: serve only /models (for mem calibrate and the "
            "bench), no store; for a config no store is gated for yet.",
        ),
    ] = False,
    hook_settings: Annotated[
        Path,
        typer.Option(
            help="HTTP: the Claude Code settings file `mem install --hooks` wrote; the Stop-hook "
            "socket starts only when the hook is registered there (F4).",
        ),
    ] = hook_mod.DEFAULT_SETTINGS,
    config: ConfigOpt = None,
    embedder: StubOpt = "st",
) -> None:
    """Serve recall/get/remember over MCP: HTTP with both stores mounted, or stdio for one.
    HTTP also serves the loaded models to local tools under /models (D5, mem.models_http).

    Refuses to start unless every store's vectors and gate match the config (§3.3)."""
    tailscale = tailscale or require_tailscale
    if tailscale and not http:
        raise typer.BadParameter("--tailscale serves HTTP: add --http", param_hint="--tailscale")
    access = _access(http, tls, tailscale, oauth, host, token_file, oauth_state, rate_limit,
                     audit_log)  # fmt: skip
    ts_ip = ts_name = None
    if tls:
        ts_name = serve_mod.tailscale_dns_name()
        if ts_name is None:
            _fail(["`tailscale status --json` gives no MagicDNS name"],
                  "mem serve --tls will not start: run `tailscale up`, and enable MagicDNS and "
                  "HTTPS certificates for the tailnet (docs/multi-machine.md).")  # fmt: skip
        if oauth:
            access.oauth = auth_mod.OwnerOAuth(f"https://{ts_name}", access.tokens, oauth_state)
    if tailscale:
        try:
            ts_ip = serve_mod.tailscale_ip()
        except MemError as err:
            if require_tailscale:
                _fail([str(err)],
                      "mem serve will not start: --require-tailscale needs the tailnet address.")  # fmt: skip
            # the tailnet is optional: local clients keep their server (MEM2-41 F3)
            typer.echo(f"WARNING: {err}; serving on {host} only until mem serve restarts "
                       "(systemctl --user restart mem)", err=True)  # fmt: skip
        else:
            if access.tokens is None:
                typer.echo("WARNING: --tailscale without --token-file serves the tailnet "
                           "plaintext and unauthenticated; move to --tls --token-file "
                           "(docs/multi-machine.md)", err=True)  # fmt: skip
            ts_name = serve_mod.tailscale_dns_name()
            if ts_name is None:
                typer.echo("WARNING: no MagicDNS name from `tailscale status --json`: clients "
                           f"must use {ts_ip}, or pass --allow-host <name>", err=True)  # fmt: skip
    if models_only and not http:
        raise typer.BadParameter(
            "--models-only serves HTTP: add --http", param_hint="--models-only"
        )
    stores = {} if models_only else _stores(store)
    if not http and len(stores) != 1:
        raise typer.BadParameter(
            "stdio serves one store: pass --store personal=<path> (or work=<path>), or use --http",
            param_hint="--store",
        )
    cfg = _config(config, embedder)
    device = device or cfg.device
    tz = tz or local_tz()
    why = doctor_mod.cpu_reason(device) if not allow_cpu else None
    if why:
        _fail([why], "mem serve will not start on the CPU: it needs an NVIDIA GPU (CUDA). "
                     "Tests and CI only: pass --allow-cpu.")  # fmt: skip
    if embedder == "st":
        _pinned(cfg, "mem serve will not start on a model that is not pinned.")
        missing = [c for c in doctor_mod.check_models(cfg) if c.status == "fail"]
        vram = doctor_mod.check_vram(cfg)
        if vram.status == "fail":
            missing.append(vram)
        if missing:
            _fail([f"{c.name}: {c.detail}" for c in missing],
                  "mem serve will not start without the pinned models: run `mem doctor --fetch`.")  # fmt: skip
    problems = serve_mod.preflight(stores, cfg)
    if problems:
        _fail(problems, "mem serve will not start: the config and the stores' gate must match.")
    for warning in serve_mod.gate_warnings(stores):
        typer.echo(f"WARNING: {warning}", err=True)
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(serve_mod.store_locks(stores.values()))
        except serve_mod.StoreBusy as err:
            _fail([str(err)], "mem serve will not start while mem reembed writes the store.")
        if embedder == "st":
            os.environ["HF_HUB_OFFLINE"] = "1"  # D2: serving never touches the network
            from mem.embedder import SentenceTransformerEmbedder
            from mem.rerank import CrossEncoderReranker

            emb: Embedder = SentenceTransformerEmbedder(cfg.embedder, device)
            rr: Reranker = CrossEncoderReranker(cfg.reranker, device, cfg.rerank_batch_size)
        else:
            emb, rr = HashEmbedder(256), OverlapReranker()
        if http:
            serve_mod.run_http(stores, emb, rr, cfg, host=host, port=port, host_label=host_label,
                               tz=tz, tailscale=ts_ip,
                               allow=serve_mod.allowed_hosts(host, ts_ip, ts_name,
                                                             allow_host or []),
                               access=access,
                               hook_socket=hook_mod.socket_path()
                               if hook_mod.installed(hook_settings) else None)  # fmt: skip
        else:
            serve_mod.run_stdio(stores, emb, rr, cfg, host_label=host_label, tz=tz)


def _access(
    http: bool, tls: bool, tailscale: bool, oauth: bool, host: str, token_file: Path | None,
    oauth_state: Path, rate_limit: int, audit_log: Path,
) -> auth_mod.Access:  # fmt: skip
    """The P7 controls of `mem serve --http` from its flags (`mem.auth`); exits on a bad mix."""
    if (tls or oauth or token_file) and not http:
        _fail(["--tls, --oauth and --token-file are HTTP options"], "add --http.")
    if oauth and not tls:
        _fail(
            ["--oauth needs --tls: a connector must reach mem over https (tailscale funnel)"],
            "add --tls.",
        )
    if tls and tailscale:
        why = (
            "--tls and --tailscale: --tailscale binds the tailnet address in plaintext, --tls "
            "keeps the tailnet behind tailscale serve's https"
        )
        _fail([why], "drop --tailscale.")
    if tls and not serve_mod.host_allowed(host, serve_mod.LOOPBACK_HOSTS + ("127.0.0.0/8",)):
        why = (
            f"--tls with --host {host}: --tls binds loopback only (tailscale serve is the "
            "tailnet's way in)"
        )
        _fail([why], "use --host 127.0.0.1.")
    tokens = None
    if token_file is not None:
        try:
            tokens = auth_mod.TokenFile(token_file).load()
        except MemError as err:
            _fail([str(err)], "mem serve will not start with a broken token file.")
        if not len(tokens):
            _fail([f"{token_file}: no tokens"],
                  f"create one: mem token add <label> --file {token_file}")  # fmt: skip
    elif tls:
        _fail(["--tls without --token-file: the https endpoint would be open to the tailnet"],
              "create a token (mem token add <label>) and pass --token-file.")  # fmt: skip
    return auth_mod.Access(tokens=tokens, tls_only=tls,
                           limiter=auth_mod.RateLimiter(rate_limit) if rate_limit > 0 else None,
                           audit=auth_mod.AuditLog(audit_log) if http else None)  # fmt: skip


def _setup_command(
    allow_cpu: bool, dry_run: bool, uninstall: bool, purge: bool, mcp: bool,
    host_label: str | None, port: int, yes: bool, remote: str | None, code: str | None,
) -> None:  # fmt: skip
    from mem import install as install_mod
    from mem import setup_cmd

    if purge and not uninstall:
        raise typer.BadParameter("--purge deletes the stores: use it with --uninstall",
                                 param_hint="--purge")  # fmt: skip
    if code and not remote:
        raise typer.BadParameter("--code is the other machine's code: add --remote <host>",
                                 param_hint="--code")  # fmt: skip
    label = host_label or install_mod.default_label(socket.gethostname())
    if not serve_mod.HOST_LABEL.fullmatch(label):
        raise typer.BadParameter("1-64 characters of A-Z a-z 0-9 . _ -", param_hint="--host-label")
    opts = setup_cmd.SetupOptions(host_label=label, allow_cpu=allow_cpu, dry_run=dry_run,
                                  uninstall=uninstall, purge=purge, mcp=mcp, port=port, yes=yes,
                                  remote=remote, code=code)  # fmt: skip
    status = setup_cmd.main(opts, install_mod.SystemRunner(), install_mod.Paths.here(),
                            echo=typer.echo)  # fmt: skip
    if status:
        raise typer.Exit(status)


@app.command()
def setup(
    allow_cpu: Annotated[
        bool, typer.Option("--allow-cpu", help="Tests and CI only: set up without a GPU.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print every action; change nothing.")
    ] = False,
    uninstall: Annotated[
        bool,
        typer.Option("--uninstall", help="Undo each step; keeps the config, tokens and stores."),
    ] = False,
    purge: Annotated[
        bool,
        typer.Option(
            "--purge", help="With --uninstall: also delete the config, tokens and stores."
        ),
    ] = False,
    mcp: Annotated[
        bool,
        typer.Option(
            "--mcp/--no-mcp",
            help="Register the MCP servers and install the skill in Claude Code (needs `claude` "
            "on PATH); --no-mcp when the Claude Code plugin brings both, or without Claude Code "
            "(the other harnesses are still set up).",
        ),
    ] = True,
    host_label: Annotated[
        str | None, typer.Option(help="This machine's X-Mem-Host label. Default: the hostname.")
    ] = None,
    port: Annotated[int, typer.Option(help="The port mem serve listens on.")] = (
        serve_mod.DEFAULT_PORT
    ),
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Do not ask: run the whole plan.")
    ] = False,
    remote: Annotated[
        str | None,
        typer.Option(
            help="Add the mem servers of another machine (its tailnet name) to each "
            "harness. Asks for its token, or takes --code."
        ),
    ] = None,
    code: Annotated[
        str | None,
        typer.Option(help="With --remote: the code `mem token add <label> --code` printed there."),
    ] = None,
) -> None:
    """Set mem up here, end to end: GPU check, models, config, token, systemd user service, MCP
    servers and skill in each harness, `mem doctor`, then a self-test. Idempotent; --dry-run shows
    what it would do, --uninstall undoes it, --remote adds the other machine."""
    _setup_command(allow_cpu, dry_run, uninstall, purge, mcp, host_label, port, yes, remote, code)


@app.command()
def install(
    allow_cpu: Annotated[bool, typer.Option("--allow-cpu")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Say what would change; write nothing.")
    ] = False,
    uninstall: Annotated[bool, typer.Option("--uninstall", help="Undo it again.")] = False,
    purge: Annotated[bool, typer.Option("--purge")] = False,
    mcp: Annotated[bool, typer.Option("--mcp/--no-mcp")] = True,
    host_label: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int, typer.Option()] = serve_mod.DEFAULT_PORT,
    hooks: Annotated[
        bool,
        typer.Option(
            "--hooks",
            help="Only the auto-capture Stop hook (a thin client of mem serve); no setup.",
        ),
    ] = False,
    settings: Annotated[
        Path | None,
        typer.Option(
            help="With --hooks: the Claude Code settings file. Default: ~/.claude/settings.json."
        ),
    ] = None,
) -> None:
    """Alias of `mem setup --yes`. With --hooks it does only the optional extra: it registers (or,
    with --uninstall, removes) a Stop hook that has Claude save what is worth remembering before it
    stops. The hook talks to a running `mem serve --http` over a Unix socket and does nothing if
    serve is down."""
    if not hooks:
        if settings is not None:
            raise typer.BadParameter("--settings is the Stop hook's file: add --hooks",
                                     param_hint="--settings")  # fmt: skip
        _setup_command(
            allow_cpu, dry_run, uninstall, purge, mcp, host_label, port, True, None, None
        )
        return
    if allow_cpu or purge or not mcp or host_label is not None or port != serve_mod.DEFAULT_PORT:
        raise typer.BadParameter("--hooks takes only --uninstall, --dry-run and --settings",
                                 param_hint="--hooks")  # fmt: skip
    settings = settings or hook_mod.DEFAULT_SETTINGS
    try:
        if uninstall:
            result = hook_mod.uninstall(settings, dry_run=dry_run)
        else:
            result = hook_mod.install(settings, hook_mod.hook_command(), dry_run=dry_run)
    except hook_mod.SettingsError as err:
        _fail([str(err)], "fix or move the file, then run again.")
    typer.echo(f"{result}: Stop hook in {settings}")


@app.command("client-config")
def client_config(
    server: Annotated[str, typer.Option(help="The serving machine's tailnet name or address.")],
    host_label: Annotated[str, typer.Option(help="This machine's label, sent as X-Mem-Host.")],
    port: Annotated[int, typer.Option(help="The server's HTTP port.")] = serve_mod.DEFAULT_PORT,
    tls: Annotated[
        bool, typer.Option("--tls", help="The server runs --tls: https://<server>/ (P7).")
    ] = False,
    token: Annotated[
        bool | None,
        typer.Option(
            "--token/--no-token",
            help="Send the bearer token as $MEM_TOKEN, expanded by the shell that runs the lines "
            "(the token is never printed); default: on if $MEM_TOKEN is set.",
        ),
    ] = None,
) -> None:
    """Print the `claude mcp add` commands that connect this machine to a `mem serve --tailscale`
    (or, with --tls, to one behind `tailscale serve`)."""
    if not serve_mod.HOST_LABEL.fullmatch(host_label):
        raise typer.BadParameter("1-64 characters of A-Z a-z 0-9 . _ -", param_hint="--host-label")
    if token is None:
        token = bool(os.environ.get("MEM_TOKEN"))
    for line in serve_mod.client_config(server, host_label, port=port, tls=tls, token=token):
        typer.echo(line)


@app.command("mcp-config")
def mcp_config(
    harness: Annotated[
        str,
        typer.Option(help="claude-code, codex, gemini, cursor, vscode, claude-desktop, generic."),
    ],
    write: Annotated[
        bool,
        typer.Option("--write/--print", help="Merge into the harness's config file, or print."),
    ] = False,
    uninstall: Annotated[
        bool, typer.Option("--uninstall", help="Undo --write: restore the file as it was.")
    ] = False,
    transport: Annotated[
        str | None, typer.Option(help="http (default) or stdio; Claude Desktop reads only stdio.")
    ] = None,
    host_label: Annotated[
        str | None, typer.Option(help="X-Mem-Host for this machine. Default: the hostname.")
    ] = None,
    port: Annotated[int, typer.Option(help="The port mem serve listens on.")] = (
        serve_mod.DEFAULT_PORT
    ),
    home: Annotated[Path | None, typer.Option(help="Whose HOME to write under.")] = None,
) -> None:
    """Show or write the MCP servers' entry for a harness. Writing merges into its config file,
    leaves the other servers alone, saves the file first and does nothing the second time."""
    from mem import harnesses as hx
    from mem import install as install_mod
    from mem import setup_steps

    home = home or Path.home()
    try:
        h = hx.get(harness)
        paths = install_mod.Paths.for_home(home, os.environ)
        label = host_label or install_mod.default_label(socket.gethostname())
        if not serve_mod.HOST_LABEL.fullmatch(label):
            raise typer.BadParameter(
                "1-64 characters of A-Z a-z 0-9 . _ -", param_hint="--host-label"
            )
        token = paths.token.read_text(encoding="utf-8").strip() if paths.token.exists() else None
        spec = hx.Spec(label, port, token, paths.repo, paths.stores())
        tr = transport or ("http" if "http" in h.transports else "stdio")
        if h.name == "generic":
            if write or uninstall:
                raise typer.BadParameter("generic only prints: there is no file to write",
                                         param_hint="--write")  # fmt: skip
            typer.echo(hx.render_generic(spec), nl=False)
            return
        if not (write or uninstall):
            typer.echo(hx.render_config(h, spec, tr), nl=False)
            return
        path = hx.config_path(h, home, os.environ, sys.platform)
        backups = setup_steps.Backups(paths.config / "backups")
        if uninstall:
            outcome = hx.remove_config(h, spec, tr, path, backups)  # type: ignore[arg-type]
        else:
            outcome = hx.write_config(h, spec, tr, path, backups)  # type: ignore[arg-type]
            if token is None:
                typer.echo(f"no token at {paths.token}: written without an Authorization header",
                           err=True)  # fmt: skip
        typer.echo(f"{h.name}: {outcome} {path}")
    except MemError as err:
        raise typer.BadParameter(str(err)) from err


@app.command()
def instructions(
    harness: Annotated[str, typer.Option(help="A harness name, or generic.")],
) -> None:
    """Print the short "when to recall, when to save" text for a harness, generated from
    skills/mem/SKILL.md. For a harness that cannot load the skill (Claude Desktop, other clients)
    or an older version of one that can. Where it goes: docs/harnesses.md."""
    from mem import harnesses as hx

    try:
        text = hx.instructions(hx.get(harness), (hx.SKILL_SOURCE / "SKILL.md").read_text("utf-8"))
    except MemError as err:
        raise typer.BadParameter(str(err)) from err
    typer.echo(text, nl=False)


token_app = typer.Typer(no_args_is_help=True, help="Bearer tokens for mem serve --token-file (P7).")
app.add_typer(token_app, name="token")
TokenFileOpt = Annotated[
    Path, typer.Option("--file", help="The token file mem serve --token-file reads.")
]


@token_app.command("add")
def token_add(
    label: Annotated[str, typer.Argument(help="Who uses it, e.g. work-laptop; in the audit log.")],
    file: TokenFileOpt = auth_mod.DEFAULT_TOKEN_FILE,
    code: Annotated[
        bool,
        typer.Option("--code", help="Print one code (host and token) for `mem setup --remote`."),
    ] = False,
    host: Annotated[
        str | None, typer.Option(help="With --code: this machine's tailnet name.")
    ] = None,
) -> None:
    """Create a token for LABEL and print it. Only its hash is kept: copy it now."""
    from mem import setup_cmd

    if code and not host:
        raise typer.BadParameter("the code names this machine: pass --host <tailnet name>",
                                 param_hint="--host")  # fmt: skip
    try:
        token = auth_mod.TokenFile(file).add(label)
    except MemError as err:
        _fail([str(err)], "no token created.")
    typer.echo(f"token for {label} (shown once; mem keeps only its hash in {file}):", err=True)
    if code:
        typer.echo(f"on the other machine: mem setup --remote {host} --code <this code>", err=True)
    typer.echo(setup_cmd.make_code(host, token) if code and host else token)


@token_app.command("list")
def token_list(file: TokenFileOpt = auth_mod.DEFAULT_TOKEN_FILE) -> None:
    """The labels that have a token."""
    for label in auth_mod.TokenFile(file).labels():
        typer.echo(label)


@token_app.command("revoke")
def token_revoke(
    label: Annotated[str, typer.Argument()], file: TokenFileOpt = auth_mod.DEFAULT_TOKEN_FILE
) -> None:
    """Remove LABEL's token; a running mem serve refuses it from the next request."""
    if not auth_mod.TokenFile(file).revoke(label):
        _fail([f"no token labelled {label!r} in {file}"], "nothing revoked.")
    typer.echo(f"revoked {label}")


@app.command()
def calibrate(
    records: Annotated[
        Path | None,
        typer.Option("--records", help="Scored tune pools: raw logits and labels (JSONL)."),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option("--labels", help="Tune labels from `bench export-tune`; scored first."),
    ] = None,
    lme_file: Annotated[
        Path | None, typer.Option("--lme", help="longmemeval_s_cleaned.json the labels came from.")
    ] = None,
    records_out: Annotated[
        Path | None,
        typer.Option(
            "--records-out",
            help="With --labels: save the scored records (refit later with --records).",
        ),
    ] = None,
    store: Annotated[
        list[str] | None,
        typer.Option("--store", help="name=path to write meta.gate into (created if missing)."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Also write the gate as JSON (for mem switch).")
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the tune report (per category, frontier) as JSON."),
    ] = None,
    fit_id: Annotated[
        str, typer.Option("--fit-id", help="Gold-set versions + split, e.g. lme-s-tune-v1.")
    ] = "lme-s-tune",
    port: Annotated[
        int, typer.Option(help="--labels: the local mem serve whose models score the pools.")
    ] = serve_mod.DEFAULT_PORT,
    targets: Annotated[
        str,
        typer.Option(
            help="Floors targets: p2 (B-R1/B-R2, LME) or b-r3 (PG-v1 tune, P4 B-R3 rows)."
        ),
    ] = "p2",
    objective: Annotated[
        str,
        typer.Option(
            help="floors: Platt + the P2 floors rule (LME). p3: the cosine read model and the"
            " thresholds that target B-R2 (ii), B-R5 and B-R2 (i) FAR on SYN dev (needs records"
            " with cosines; architecture-v2 §3.3)."
        ),
    ] = "floors",
    require_masked: Annotated[
        bool,
        typer.Option(
            help="p3: refuse unless every pool item carries mcos/mz and pcos/pov (the masked,"
            " margin and relation-phrase feature sets are then always in the grid)."
        ),
    ] = False,
    config: ConfigOpt = None,
    embedder: StubOpt = "st",
) -> None:
    """Fit Platt, τ_keep, τ_high and g on tune pools; write the gate (§3.3).

    Either --records (already scored) or --labels with --lme: each tune query's pool is first
    scored by the production read path on a fresh store of its haystack (steps 1-5), with the
    config's models as the running mem serve holds them (D5), then fitted the same way. Loads no
    model either way. Prints the tune report (Platt inputs, what bound τ_keep, the per-category
    delivered metrics and the τ_keep frontier), then the gate as the last line."""
    cfg = _config(config, embedder)
    if targets not in ("p2", "b-r3"):
        raise typer.BadParameter("--targets is p2 or b-r3")
    if objective not in ("floors", "p3"):
        raise typer.BadParameter("--objective is floors or p3")
    if objective == "p3" and targets != "p2":
        raise typer.BadParameter("--objective p3 has its own targets (the P3 gate rows)")
    if (records is None) == (labels is None) or (labels is None) != (lme_file is None):
        raise typer.BadParameter("pass --records, or --labels with --lme")
    if records is not None:
        fp_all, pools = calibrate_mod.read_records(records)
        if fp_all != cfg.fp_all():
            _fail([(f"{records} was scored with fp_all {fp_all[:8]}, the config is "
                   f"{cfg.fp_all()[:8]}")], "Re-score the tune queries with this config, or pass "
                  "the config they were scored with.")  # fmt: skip
    else:
        assert labels is not None and lme_file is not None
        pools = _score_labels(labels, lme_file, cfg, port)
        if records_out:
            calibrate_mod.write_records(records_out, pools, fp_all=cfg.fp_all())
    stores = serve_mod.parse_stores(store) if store else {}
    wrong = [f"{n} ({p}): its vectors are not this config's embedder; write the gate with --out "
             "and apply it with mem switch after mem reembed"
             for n, p in stores.items() if p.exists() and not _embed_matches(p, cfg)]  # fmt: skip
    if wrong:
        _fail(wrong, "No gate was written.")
    p3_report: dict | None = None
    try:
        if objective == "p3":
            from mem import readfit

            gate, p3_report = readfit.fit_p3_gate(
                pools, fit_id=fit_id, fingerprint_hash=cfg.fp_all(),
                limit=cfg.retrieval["limit"], fitted_at=fmt_ts(datetime.now(UTC)),
                require_masked=require_masked)  # fmt: skip
        else:
            gate = calibrate_mod.fit_gate(pools, fit_id=fit_id, fingerprint_hash=cfg.fp_all(),
                                      limit=cfg.retrieval["limit"],
                                      fitted_at=fmt_ts(datetime.now(UTC)),
                                      targets=calibrate_mod.b_r3_targets() if targets == "b-r3"
                                      else None)  # fmt: skip
    except ValueError as err:
        if (unmet := getattr(err, "report", None)) is not None:  # readfit.GuardUnmet: the record
            from mem import readfit

            if report:
                report.write_text(json.dumps(unmet, indent=1, sort_keys=True, default=str) + "\n",
                                  encoding="utf-8")  # fmt: skip
            typer.echo(readfit.format_report(unmet))
        _fail([str(err)], "No gate was written.")
    for name, path in stores.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = init_store(path, name, cfg.fp_embed(), int(cfg.embedder["dim"]))
        write_gate(conn, gate)
        conn.close()
    if out:
        out.write_text(gate_json(gate) + "\n", encoding="utf-8")
    if p3_report is not None:
        from mem import readfit

        if report:
            report.write_text(json.dumps(p3_report, indent=1, sort_keys=True, default=str) + "\n",
                              encoding="utf-8")  # fmt: skip
        typer.echo(readfit.format_report(p3_report))
        typer.echo(gate_json(gate))
        return
    tune = calibrate_mod.tune_report(pools, gate, limit=cfg.retrieval["limit"],
                                     targets=calibrate_mod.b_r3_targets() if targets == "b-r3"
                                     else None)  # fmt: skip
    if report:
        report.write_text(json.dumps(tune, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(calibrate_mod.format_report(tune))
    typer.echo(gate_json(gate))


def _score_labels(
    labels: Path, lme_file: Path, cfg: config_mod.Config, port: int
) -> list[calibrate_mod.QueryPool]:
    """`bench export-tune` labels -> scored pools, through the bench's LME adapter (the same
    ingest and read path the gates use; `bench` ships in this package, imported only here) on
    the models of the mem serve at `port` (`mem.models_http`)."""
    from bench.adapters.lme_ours import OursLmeBackend
    from bench.datasets import calibration, lme
    from bench.suites.base import BackendUnavailable

    try:
        rows = calibration.read_tune_labels(labels)
        if not rows:
            raise ValueError(f"{labels}: no tune queries")
        questions = {q["question_id"]: q for q in lme.load_questions(lme_file)}
        if missing := sorted({r["qid"] for r in rows} - set(questions)):
            raise ValueError(f"{lme_file}: no question {missing[0]} (+{len(missing) - 1} more)")
        backend = OursLmeBackend(cfg=cfg, loaded=models_http.connect(port, cfg))
        try:
            return calibration.score_tune(rows, questions, backend)
        finally:
            backend.close()
    except models_http.ServeUnavailable as err:
        _fail([str(err)], "No gate was written: mem calibrate --labels scores through mem serve.")
    except (OSError, ValueError, KeyError, BackendUnavailable) as err:
        _fail([str(err)], "No gate was written.")


def _embed_matches(path: Path, cfg: config_mod.Config) -> bool:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
    finally:
        conn.close()
    return row is not None and json.loads(row[0]).get("embed") == cfg.fp_embed()


@app.command()
def reembed(
    store: Annotated[list[str], typer.Option("--store", help="name=path of a store to reembed.")],
    device: Annotated[
        str | None, typer.Option(help="auto, cpu or cuda. Default: config.yaml `device`.")
    ] = None,
    port: Annotated[
        int, typer.Option(help="The mem serve port; reembed refuses while it is in use.")
    ] = serve_mod.DEFAULT_PORT,
    config: ConfigOpt = None,
    embedder: StubOpt = "st",
) -> None:
    """Embed every chunk with the config's embedder into its own vec table, next to the old one
    (§3.3, A21). Resumable. Serving is unchanged until mem calibrate + mem switch.

    The one exception to D5 (docs/decisions/0006): mem serve holds the old embedder, so reembed
    loads the new one itself, and only while no mem serve runs (port free, stores not held)."""
    cfg = _config(config, embedder)
    stores = _stores(store)
    if missing := [f"{n} ({p}): no store" for n, p in stores.items() if not p.exists()]:
        _fail(missing, "Nothing was reembedded.")
    stop = ("Stop mem serve first (`systemctl --user stop mem`), re-run mem reembed, then "
            "calibrate on a `mem serve --http --models-only` with the new config.")  # fmt: skip
    if models_http.serve_listening(port):
        _fail([(f"127.0.0.1:{port} is in use, presumably by mem serve: it holds the old "
                "embedder and only one process may load models (D5)")], stop)  # fmt: skip
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(serve_mod.store_locks(stores.values(), exclusive=True))
        except serve_mod.StoreBusy as err:
            _fail([str(err)], stop)
        _reembed_stores(stores, cfg, embedder, device)
    typer.echo("Next: mem serve --http --models-only, then mem calibrate --labels <tune labels> "
               "--lme <LME file> --out gate.json, then mem switch --gate gate.json --store …, "
               "all with the same --config.")  # fmt: skip


def _reembed_stores(
    stores: dict[str, Path], cfg: config_mod.Config, embedder: str, device: str | None
) -> None:
    if embedder == "st":
        try:
            pin = catalog.embedder_entry(cfg.embedder).pin
        except ValueError as err:
            _fail([str(err)], "Nothing was reembedded.")
        if (c := doctor_mod.check_model(pin)).status == "fail":
            _fail([f"{c.name}: {c.detail}"], "Run `mem doctor --fetch` first.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        from mem.embedder import SentenceTransformerEmbedder

        emb: Embedder = SentenceTransformerEmbedder(cfg.embedder, device or cfg.device)
    else:
        emb = HashEmbedder(int(cfg.embedder["dim"]))
    for name, path in stores.items():
        conn = schema.connect(path)
        try:
            table = reembed_mod.reembed(conn, emb, cfg.fp_embed(), session_leg=cfg.session_leg)
            (n,) = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        finally:
            conn.close()
        typer.echo(f"{name}: {n} chunks in {table}")


@app.command()
def switch(
    gate: Annotated[Path, typer.Option("--gate", help="Gate JSON from mem calibrate --out.")],
    store: Annotated[list[str], typer.Option("--store", help="name=path of a store to switch.")],
    config: ConfigOpt = None,
    embedder: StubOpt = "st",
) -> None:
    """Point stores at the config's vectors and gate in one transaction (after mem reembed)."""
    cfg = _config(config, embedder)
    try:
        g = Gate(**json.loads(gate.read_text(encoding="utf-8")))
        for name, path in serve_mod.parse_stores(store).items():
            reembed_mod.switch(path, cfg, g)
            typer.echo(f"{name}: switched to {cfg.vec_table()} and gate {g.fit_id}")
    except (OSError, ValueError, TypeError) as err:
        _fail([str(err)], "Nothing was switched for the store above.")


@app.command()
def render(
    store: Annotated[list[Path], typer.Option("--store", help="Store file to render.")],
    out: Annotated[
        Path, typer.Option("--out", help="Render folder (written under <out>/<scope>/).")
    ],
    html: Annotated[bool, typer.Option("--html", help="Also write .html pages.")] = False,
    commit: Annotated[
        bool,
        typer.Option(
            "--commit",
            help="Commit the rendered scope folders (the git mirror); other files are left out.",
        ),
    ] = False,
) -> None:
    """Render stores to Markdown (and HTML), one-way and byte-identical for an identical store."""
    try:
        scopes: list[str] = []
        for path in store:
            r = render_mod.render_path(path, out, html=html)
            scopes.append(r.scope)
            typer.echo(
                f"{r.scope}: {r.written} written, {r.unchanged} unchanged, {r.removed} removed"
            )
        if commit:
            typer.echo(
                "committed" if render_mod.commit_mirror(out, scopes) else "nothing to commit"
            )
    except (MemError, sqlite3.Error) as err:
        _fail([str(err)], "Nothing was committed.")


@app.command()
def core(
    store: Annotated[Path, typer.Option("--store", help="Store file to read.")],
    budget: Annotated[int, typer.Option("--budget", help="Token budget.")] = 800,
) -> None:
    """Print the core index (pinned, then recent memories) for a `SessionStart` hook."""
    conn = _open_ro(store)
    typer.echo(review_mod.core_index(conn, budget))


@app.command()
def conflicts(
    store: Annotated[Path, typer.Option("--store", help="Store file to read.")],
    since: Annotated[str | None, typer.Option("--since", help="Only events at/after this.")] = None,
) -> None:
    """List `remember` results whose possible_conflicts were never linked or forgotten."""
    conn = _open_ro(store)
    for u in review_mod.unresolved_conflicts(conn, since):
        typer.echo(f"{u.new_id} (event {u.event_id}) conflicts with {', '.join(u.candidates)}")
    for h in review_mod.read_guard_hidden(conn, since):
        typer.echo(f"read guard hid {h.earlier} behind {h.later} ({h.times}x, last {h.last_at},"
                   f" q: {h.last_query}): link {h.later} supersedes {h.earlier} if it is an update,"
                   " duplicate_of if the same fact, relates_to if both hold")  # fmt: skip


@app.command("review")
def review_cmd(
    store: Annotated[Path, typer.Option("--store", help="Store file to read.")],
    day: Annotated[str, typer.Option("--day", help="UTC date, YYYY-MM-DD.")],
) -> None:
    """The evening sample: up to 10 of the day's recalls, chosen deterministically."""
    conn = _open_ro(store)
    for e in review_mod.daily_sample(conn, day, store=store.stem):
        typer.echo(f"#{e.id} {e.at} q: {e.query}")
        for c in e.context:
            typer.echo(f"    before: {c}")
        for r in e.rows or ["(empty)"]:
            typer.echo(f"    {r}")
    for h in review_mod.read_guard_hidden(conn, day):
        typer.echo(f"read guard hid {h.earlier} behind {h.later} ({h.times}x): offer `link"
                   f" {h.later} supersedes {h.earlier}` (or duplicate_of / relates_to)")  # fmt: skip
    typer.echo(f"tuning: {review_mod.tuning_counts(conn, day)}")


@app.command()
def quarantine(
    store: Annotated[Path, typer.Option("--store", help="Store file.")],
    release: Annotated[list[str] | None, typer.Option(
        "--release", help="Memory id to take out of quarantine (repeatable).")] = None,
) -> None:  # fmt: skip
    """List quarantined memories (prompt injection, never recalled), or release reviewed ones.
    Forget the rest with the `forget` tool."""
    if not store.exists():
        _fail([f"no store at {store}"], "Nothing was read.")
    conn = schema.connect(store)
    try:
        for ident in release or []:
            guard.release(conn, ident)
            typer.echo(f"released {ident}")
        if not release:
            rows = guard.quarantined(conn)
            for nid, reason, source, body in rows:
                typer.echo(f"m{nid} · {reason} · {source or 'no source'}\n    {body}")
            typer.echo(f"{len(rows)} quarantined")
    except MemError as err:
        _fail([str(err)], "Stopped there; earlier releases stand.")
    finally:
        conn.close()


def _open_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        _fail([f"no store at {path}"], "Nothing was read.")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


KeyFileOpt = Annotated[
    Path, typer.Option("--key-file", help="Backup key (64 hex chars, mode 600).")
]


@app.command("backup-key")
def backup_key(
    out: Annotated[Path, typer.Option("--out", help="Where to write the new key.")] = (
        CONFIG_HOME / "backup.key"
    ),
) -> None:
    """Create the backup key (once). Keep a copy OFF this machine: without it no backup restores."""
    try:
        backup_mod.new_key(out)
    except MemError as err:
        _fail([str(err)], "No key was written.")
    typer.echo(f"wrote {out} (mode 600). Copy it to your password manager now.")


@app.command()
def backup(
    store: Annotated[
        list[str] | None, typer.Option("--store", help="name=path. Default: personal and work.")
    ] = None,
    folder: Annotated[Path, typer.Option("--dir", help="Backup folder.")] = DATA_HOME / "backups",
    key_file: KeyFileOpt = CONFIG_HOME / "backup.key",
    keep: Annotated[int, typer.Option(help="Backups kept per store (older ones deleted).")] = 14,
) -> None:
    """Encrypted snapshot of each store (`VACUUM INTO`, safe while mem serve runs)."""
    try:
        key = backup_mod.load_key(key_file)
    except MemError as err:
        _fail([str(err)], "No backup was made.")
    failed = []
    for name, path in _stores(store).items():
        try:
            out = backup_mod.backup_store(path, folder, key, name=name)
            gone = backup_mod.prune(folder, name, keep)
        except MemError as err:
            failed.append(f"{name}: {err}")
            continue
        typer.echo(f"{name}: {out} ({out.stat().st_size} bytes), pruned {len(gone)}")
    if failed:
        _fail(failed, "Backup incomplete: older backups of the failed stores were kept.")


@app.command()
def restore(
    src: Annotated[Path, typer.Argument(help="A .mbk file from `mem backup`.", exists=True)],
    to: Annotated[Path, typer.Option("--to", help="New store file to write (must not exist).")],
    key_file: KeyFileOpt = CONFIG_HOME / "backup.key",
) -> None:
    """Decrypt a backup into a new store file and verify it (sha256, integrity_check)."""
    try:
        backup_mod.restore(src, backup_mod.load_key(key_file), to)
    except MemError as err:
        _fail([str(err)], "Nothing was restored.")
    typer.echo(f"restored {src.name} -> {to}: sha256 ok, integrity ok")


release_app = typer.Typer(help="Build the public copy of mem.", no_args_is_help=True)
app.add_typer(release_app, name="release")


@release_app.command("export")
def release_export(
    target: Annotated[Path, typer.Argument(help="A new or empty folder.")],
    repo: Annotated[Path, typer.Option(help="The mem repository to export from.")] = (
        release_mod.REPO
    ),
    ref: Annotated[str, typer.Option(help="The commit to export.")] = "HEAD",
) -> None:
    """Export the committed public paths of REF into TARGET as a fresh git repository with one
    commit, after the public-clean test passes there. Never pushes."""
    try:
        files = release_mod.export(target, repo, ref)
    except MemError as err:
        _fail([str(err)], "Nothing was exported.")
    typer.echo(f"exported {files} files to {target}: 1 commit, no remote")


if __name__ == "__main__":
    app()
