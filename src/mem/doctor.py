"""`mem doctor [--fetch]`: environment checks and the pinned model download (D2)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mem import catalog, shipped
from mem.common import MemError
from mem.config import Config, default_config
from mem.gate import read_gate
from mem.pins import EMBEDDER, Pin, fp_embed, snapshot_dir

Downloader = Callable[[str, str], str]

WEIGHTS = "model.safetensors"
WEIGHTS_INDEX = "model.safetensors.index.json"

GEMMA_HINT = (
    "accept the Gemma terms at https://huggingface.co/google/embeddinggemma-300m "
    "with your HF account, then run `hf auth login`"
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


def check_sqlite_vec() -> Check:
    try:
        import sqlite_vec

        con = sqlite3.connect(":memory:")
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        vec = con.execute("select vec_version()").fetchone()[0]
        con.execute("create virtual table t using fts5(x)")
    except (ImportError, AttributeError, sqlite3.Error) as err:
        return Check("sqlite-vec", "fail", f"{type(err).__name__}: {err}")
    return Check("sqlite-vec", "ok", f"vec {vec}, fts5 ok, sqlite {sqlite3.sqlite_version}")


MIN_FREE_VRAM_GIB = 4.0


def cpu_reason(device: str) -> str | None:
    """Why `device` (a config/--device value) serves on the CPU, or None when it is CUDA.
    Serving needs an NVIDIA GPU (architecture-v2 Host requirements, the owner 2026-10-24)."""
    if device == "cpu":
        return "device is cpu"
    try:
        import torch
    except ImportError:
        return "torch not installed (uv sync --extra models)"
    if not torch.cuda.is_available():
        return f"torch {torch.__version__} sees no CUDA"
    return None


def check_device(device: str = "cuda") -> Check:
    """Every doctor run is a serving-host check: CPU, or under 4 GB free VRAM, is a fail."""
    why = cpu_reason(device)
    if why:
        return Check("device", "fail", f"{why}: mem serves only on an NVIDIA GPU (CUDA)")
    import torch

    props = torch.cuda.get_device_properties(0)
    free, _total = torch.cuda.mem_get_info(0)
    detail = f"cuda: {props.name}, {free / 2**30:.1f} GiB free of {props.total_memory / 2**30:.1f}"
    if free / 2**30 < MIN_FREE_VRAM_GIB:
        return Check("device", "fail", f"{detail}; need >= {MIN_FREE_VRAM_GIB:g} GiB free VRAM")
    return Check("device", "ok", detail)


def missing_model_files(snap: Path) -> list[str]:
    """Files a snapshot still needs to load: config.json plus single or sharded weights.

    A gated 401 leaves a snapshot holding only README.md, so the folder existing proves nothing.
    `is_file()` follows the hub's blob symlinks, so a dangling link counts as missing.
    """
    missing = [] if (snap / "config.json").is_file() else ["config.json"]
    if (snap / WEIGHTS).is_file():
        return missing
    index = snap / WEIGHTS_INDEX
    if not index.is_file():
        return [*missing, WEIGHTS]
    try:
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [*missing, f"{WEIGHTS_INDEX} (unreadable)"]
    if not shards:
        return [*missing, f"{WEIGHTS_INDEX} (empty weight_map)"]
    return missing + [s for s in shards if not (snap / s).is_file()]


def _fix_hint(pin: Pin) -> str:
    licence = GEMMA_HINT if pin == EMBEDDER else "check the repo licence and run `hf auth login`"
    return f"{licence}; then run `mem doctor --fetch`"


def check_model(pin: Pin, name: str | None = None) -> Check:
    name = name or pin.id
    snap = snapshot_dir(pin)
    if not snap.is_dir():
        return Check(
            name, "fail", f"revision {pin.revision[:8]} not in cache; run `mem doctor --fetch`"
        )
    missing = missing_model_files(snap)
    if missing:
        return Check(
            name,
            "fail",
            f"revision {pin.revision[:8]} incomplete at {snap}, missing {', '.join(missing)}"
            f" ({_fix_hint(pin)})",
        )
    return Check(name, "ok", f"{pin.revision[:8]} at {snap}")


def configured_pins(cfg: Config | None) -> tuple[Pin, Pin]:
    """The (embedder, reranker) pins the config selects (`models:`, ADR 0025); the default config's
    without one."""
    cfg = cfg or default_config()
    return catalog.embedder_entry(cfg.embedder).pin, catalog.reranker_entry(cfg.reranker).pin


def check_models(cfg: Config | None = None) -> list[Check]:
    return [check_model(pin) for pin in configured_pins(cfg)]


VRAM_BUDGET_GIB = 12.0  # the work machine's GPU (ADR 0025): the serving set must fit under it


def check_vram(cfg: Config) -> Check:
    """The VRAM the configured serving set needs (weights at the section's dtype + activations +
    the process's CUDA context and allocator overhead, `catalog.vram_parts`), a fail over 12 GiB.
    An estimate: the overhead is fitted to one measured peak (the default set, p3-gate.md Round 9 F)."""
    parts = catalog.vram_parts(cfg)
    need = sum(g for _, g in parts)
    detail = (f"serving set needs ~{need:.1f} GiB ("
              + ", ".join(f"{name.rsplit('/', 1)[-1]} {g:.1f}" for name, g in parts)
              + f"), budget {VRAM_BUDGET_GIB:g} GiB")  # fmt: skip
    if need > VRAM_BUDGET_GIB:
        return Check("vram budget", "fail", f"{detail}: lower models.rerank_batch_size or retrieval.pool, or pick smaller models")  # fmt: skip
    return Check("vram budget", "ok", detail)


def check_fingerprint(cfg: Config | None = None) -> Check:
    if cfg is not None:
        return Check(
            "fingerprint", "ok", f"fp_embed {cfg.fp_embed()[:8]}, fp_all {cfg.fp_all()[:8]}"
        )
    return Check(
        "fingerprint",
        "ok",
        f"fp_embed {fp_embed()[:8]} (pinned section, provisional until config.yaml)",
    )


def check_schema(db: Path) -> Check:
    if not db.exists():
        return Check(f"schema {db.name}", "warn", "store does not exist yet")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute("select value from meta where key='schema_version'").fetchone()
        finally:
            con.close()
    except sqlite3.Error as err:
        return Check(f"schema {db.name}", "fail", f"{type(err).__name__}: {err}")
    if row is None:
        return Check(f"schema {db.name}", "fail", "meta.schema_version missing")
    return Check(f"schema {db.name}", "ok", f"schema_version {row[0]}")


def _store_name(path: Path, con: sqlite3.Connection | None) -> str:
    """The store's scope (what `--store name=path` wants), else the file's stem."""
    if con is not None:
        try:
            row = con.execute("select value from meta where key='scope'").fetchone()
        except sqlite3.Error:
            row = None
        if row and row[0]:
            return str(row[0])
    return path.stem


def check_gate(stores: list[Path], cfg: Config) -> list[Check]:
    """One check per store: which gate serves it. A store's own gate wins, else the shipped one
    (ADR 0017). A fingerprint that is not the config's FAILS and names the command that fixes it."""
    try:
        ship = shipped.load()
    except MemError as err:
        return [Check("gate", "fail", str(err))]
    out = []
    for path in stores:
        name = f"gate {path.name}"
        con, gate, store = None, None, path.stem
        if path.exists():
            try:
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                store = _store_name(path, con)
                gate = read_gate(con)
            except (sqlite3.Error, ValueError) as err:
                out.append(Check(name, "fail", f"{type(err).__name__}: {err}"))
                continue
            finally:
                if con is not None:
                    con.close()
        fix = shipped.calibrate_fix(store, path)
        if gate is not None:
            if gate.fingerprint_hash == cfg.fp_all():
                out.append(
                    Check(name, "ok", f"local gate {gate.fit_id or 'unnamed'} fits the config")
                )
            else:
                out.append(
                    Check(
                        name,
                        "fail",
                        f"local gate fitted for {gate.fingerprint_hash[:8]},"
                        f" the config is {cfg.fp_all()[:8]}: run `{fix}`",
                    )
                )
        elif ship is None:  # `mem serve` refuses the store, created yet or not (`preflight`)
            out.append(Check(name, "fail", f"no local gate and none shipped: run `{fix}`"))
        elif ship.gate.fingerprint_hash == cfg.fp_all():
            out.append(
                Check(
                    name,
                    "ok",
                    f"no local gate: serving on the shipped gate "
                    f"(fitted {ship.metadata['fit_date']}, mem {ship.metadata['mem_version']})",
                )
            )
        else:
            out.append(
                Check(
                    name, "fail", f"no local gate, and {shipped.mismatch(cfg, ship)}: run `{fix}`"
                )
            )
    return out


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10).stdout


ENCRYPTED_FS = ("fuse.gocryptfs", "fuse.cryfs", "ecryptfs")


RUNBOOK = "see docs/backups.md"
OPT_OUT = "--allow-unencrypted"


def check_at_rest(
    path: Path,
    run: Callable[[list[str]], str] = _run,
    allow_unencrypted: bool = False,
    required: bool = False,
) -> Check:
    """Is `path` (the stores' folder) on an encrypted volume: dm-crypt/LUKS below the mount, or
    a stacked encrypted filesystem? The stores themselves are plain SQLite (decision 0015).
    Not encrypted, or cannot tell, is `warn` ("deferred", the owner 2026-10-25) until the owner sets
    `security.at_rest_required: true`; then it is `fail`, and `allow_unencrypted` (test machines)
    makes it `warn` again."""
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        source, fstype = run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", str(target)]
        ).split()
        if fstype in ENCRYPTED_FS:
            return Check("at-rest", "ok", f"{path} on {fstype.removeprefix('fuse.')} ({source})")
        device = source.split("[", 1)[0]  # btrfs: /dev/mapper/luks-…[/@home]
        types = run(["lsblk", "-s", "-n", "-o", "TYPE", device]).split()
    except (OSError, subprocess.SubprocessError, ValueError) as err:
        problem = f"cannot tell whether {path} is encrypted at rest: {err}"
    else:
        if "crypt" in types:
            return Check("at-rest", "ok", f"{path} on dm-crypt ({device})")
        problem = f"{path} is NOT encrypted at rest ({device}, {fstype})"
    if not required:
        return Check("at-rest", "warn", f"{problem}: deferred (owner's backlog); see p7-backup.md")
    if allow_unencrypted:
        return Check("at-rest", "warn", f"{problem}; opted out ({OPT_OUT}), test machines only")
    return Check("at-rest", "fail", f"{problem}: {RUNBOOK}, or {OPT_OUT} on a test machine")


def _download(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision)


def fetch_models(downloader: Downloader | None = None, cfg: Config | None = None) -> list[Check]:
    download = downloader or _download
    out = []
    for pin in configured_pins(cfg):
        out.append(_fetch_one(pin, download))
    return out


def _fetch_one(pin: Pin, download: Downloader) -> Check:
    try:
        path = download(pin.id, pin.revision)
    except Exception as err:  # noqa: BLE001  network, auth and licence errors differ per hub version
        hint = f" ({GEMMA_HINT})" if pin == EMBEDDER else ""
        return Check(f"fetch {pin.id}", "fail", f"{type(err).__name__}: {err}{hint}")
    after = check_model(pin, name=f"fetch {pin.id}")
    if after.status != "ok":
        return after
    return Check(f"fetch {pin.id}", "ok", f"{pin.revision[:8]} -> {path}")


def run(
    fetch: bool = False,
    stores: list[Path] | None = None,
    at_rest: Path | None = None,
    allow_unencrypted: bool = False,
    device: str = "cuda",
    at_rest_required: bool = False,
    cfg: Config | None = None,
) -> list[Check]:
    results: list[Check] = []
    if fetch:
        results += fetch_models(cfg=cfg)
    results.append(check_sqlite_vec())
    results += check_models(cfg)
    results.append(check_device(device))
    if cfg is not None:
        results.append(check_vram(cfg))
    results.append(check_fingerprint(cfg))
    results += [check_schema(s) for s in stores or []]
    if cfg is not None and stores:
        results += check_gate(stores, cfg)
    if at_rest is not None:
        results.append(
            check_at_rest(at_rest, allow_unencrypted=allow_unencrypted, required=at_rest_required)
        )
    return results
