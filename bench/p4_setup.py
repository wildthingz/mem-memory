"""Builds the P4 gate's `P4Setup` and the helpers of the P4 commands (benchmarks-v2 §4 P4 row).

Kept out of `bench/cli.py` so tests can build the same setup with the stand-in models and importer.
The bench never touches the person's live stores: it extracts the snapshot to its own directory and
imports it into stores of its own (`<workdir>/<name>/personal.db`, `work.db`). One models serve is
shared by every backend of a gate run (the GPU holds one copy of the models, D5).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from bench.adapters import p4_ours
from bench.adapters.lme_ours import load_config, read_gate_file, start_models
from bench.datasets import vault as vault_ds
from bench.gold import pg
from bench.suites import b_p4
from bench.suites.base import BackendUnavailable

MINI_VAULT = Path("tests/fixtures/minivault/vault")
MINI_CONFIG = Path("tests/fixtures/minivault/config.yaml")
LIFECYCLE = Path(__file__).resolve().parent / "fixtures" / "vault-mini.lifecycle.yaml"
GOLD_DIR = Path(__file__).resolve().parent / "gold" / "personal" / "v1"
MTIME = (
    1768478400.0  # 2026-01-15T12:00:00Z, the fixture's mtime (notes with no date fall back to it)
)


def routes_from_config(path: Path | None) -> dict[str, str]:
    """`vault.routes` of a config.yaml (Q3 per-folder overrides); {} if the file has none."""
    if path is None or not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict((raw.get("vault") or {}).get("routes") or {})


def copy_fixture(src: Path, dst: Path) -> Path:
    """A working copy with the fixture's fixed mtime, so a re-import is a function of the files."""
    shutil.copytree(src, dst)
    for p in dst.rglob("*"):
        if p.is_file():
            os.utime(p, (MTIME, MTIME))
    return dst


def cli_renderer() -> Callable[[list[Path], Path], None]:
    """`mem render --store A --store B --out DIR` as a subprocess (the product command)."""
    import subprocess
    import sys

    def render(dbs: list[Path], out: Path) -> None:
        cmd = [sys.executable, "-m", "mem.cli", "render"]
        for db in dbs:
            cmd += ["--store", str(db)]
        p = subprocess.run([*cmd, "--out", str(out)], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            text = (p.stderr or p.stdout)[:300]
            if "No such command" in text or "No such option" in text:
                raise BackendUnavailable("`mem render` is not available (MEM2-33)")
            raise RuntimeError(f"mem render failed: {text}")

    return render


class P4Run:
    """Owns what a gate run starts (the models serve, temp dirs) so one `close()` cleans up."""

    def __init__(self) -> None:
        self.setup: b_p4.P4Setup | None = None
        self._serve: Any = None
        self._tmp = tempfile.TemporaryDirectory(prefix="bench-p4-")
        self.workdir = Path(self._tmp.name)
        self.backends: list[Any] = []
        self.cfg: Any = None

    @property
    def models(self) -> tuple[Any, Any] | None:
        """The (embedder, reranker) clients of the one serve this run started (shared with the
        LongMemEval backend of B-R1 / B-R2, so the GPU holds one copy of the models)."""
        return self._serve.models if self._serve is not None else None

    def close(self) -> None:
        if self.setup is not None:
            self.setup.close()
        for b in self.backends:
            b.close()
        if self._serve is not None:
            self._serve.stop()
        self._tmp.cleanup()


def build(
    *, snapshot: Path | None, snapshot_sha256: str | None, gold_dir: Path | None,
    pg_snapshot: Path | None = None, pg_snapshot_sha256: str | None = None,
    split: str, mini_vault: Path | None, mini_config: Path | None, lifecycle: Path | None,
    config_path: Path | None, gate_path: Path | None, embedder: str, device: str | None,
    serve_port: int | None, i5_sheet: Path | None = None, c4_run: Path | None = None,
    baselines: dict[str, Path] | None = None, latency_runs: dict[str, list[Path]] | None = None,
    c3_runs: dict[str, list[Path]] | None = None, render: Callable | None = None,
    importer: Any = None, loaded: tuple[Any, Any] | None = None, cfg: Any = None, gate: Any = None,
    bootstrap_b: int = 10_000,
) -> P4Run:  # fmt: skip
    """The setup of `bench gate P4`. Every input may be absent: the suites that need it then say
    UNAVAILABLE or leave their metric out (FAIL), never pass."""
    run = P4Run()
    try:
        cfg = cfg or load_config(config_path, embedder)
        run.cfg = cfg
        gate = gate if gate is not None else (read_gate_file(gate_path, cfg) if gate_path else None)
        if loaded is None:
            run._serve = start_models(cfg, embedder, config_path=config_path, device=device,
                                      port=serve_port)  # fmt: skip
            loaded = run._serve.models
        gold = pg.load(gold_dir) if gold_dir and (gold_dir / "SNAPSHOT").exists() else None
        snap_date = gold.snapshot["snapshot_date"] if gold else "2026-10-22"
        meta: dict[str, Any] = {}
        snap_vault = None
        if snapshot is not None:  # the full vault: B-I1, B-I3, B-C4, V5k / V50k
            sha = vault_ds.verify(snapshot, snapshot_sha256) if snapshot_sha256 else \
                vault_ds.sha256_file(snapshot)  # fmt: skip
            snap_vault = vault_ds.extract(snapshot, run.workdir / "snapshot")
            meta = {"snapshot_file": snapshot.name, "snapshot_sha256": sha,
                    "snapshot_date": snap_date, "gold_dir": str(gold_dir)}  # fmt: skip
        pg_vault = None
        if pg_snapshot is not None:  # PG-v1 was built on this one: personal/ + projects/ only
            sha = vault_ds.verify(pg_snapshot, pg_snapshot_sha256) if pg_snapshot_sha256 else \
                vault_ds.sha256_file(pg_snapshot)  # fmt: skip
            if gold and gold.snapshot.get("snapshot_sha256") != sha:
                raise ValueError(f"the frozen PG-v1 was built on snapshot sha256 "
                                 f"{gold.snapshot.get('snapshot_sha256')}, the PG snapshot is {sha}")  # fmt: skip
            pg_vault = vault_ds.extract(pg_snapshot, run.workdir / "pg-snapshot")
            meta |= {"pg_snapshot_file": pg_snapshot.name, "pg_snapshot_sha256": sha}
        if gold:
            meta |= {k: v for k, v in gold.snapshot.items() if k.endswith("_sha256")}
        fixture = copy_fixture(mini_vault, run.workdir / "mini") if mini_vault else None
        routes = routes_from_config(mini_config) if mini_vault else {}

        def new_backend(name: str) -> p4_ours.OursVaultBackend:
            b = p4_ours.OursVaultBackend(run.workdir / "stores" / name, snap_date, cfg=cfg,
                                         loaded=loaded, gate=gate, importer=importer)  # fmt: skip
            run.backends.append(b)
            return b

        s = b_p4.P4Setup(
            new_backend=new_backend, fixture_vault=fixture, snapshot_vault=snap_vault,
            lifecycle=lifecycle if lifecycle and Path(lifecycle).exists() else None, gold=gold,
            pg_vault=pg_vault,
            split=split, dataset_meta=meta, routes=routes,
            render=render if render is not None else cli_renderer(), i5_sheet=i5_sheet,
            c4_run=c4_run, baselines=baselines or {}, latency_runs=latency_runs or {},
            c3_runs=c3_runs or {}, bootstrap_b=bootstrap_b)  # fmt: skip
        if (
            snap_vault is not None or pg_vault is not None
        ):  # the vault's own routes (config.yaml), not the fixture's
            s.snapshot_routes = routes_from_config(config_path)
        run.setup = s
        return run
    except BaseException:
        run.close()
        raise
