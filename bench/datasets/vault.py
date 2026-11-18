"""The vault snapshot and the V5k / V50k load corpora (benchmarks-v2 §3.1, B-C).

- Snapshot: a tarball of the person's vault (`tar --zstd` or gzip/plain), sha256 recorded in
  `bench/fixtures/vault-<date>.sha256` and in `PG-v1/SNAPSHOT`. `verify` refuses a different
  file; `extract` refuses members that escape the target directory.
- V5k = the snapshot imported (≈ 4-6k chunks). V50k = V5k + LongMemEval_M filler chunks up to
  50,000 chunks total, chosen deterministically by sorted session id (`corpora.plan_corpus_file`),
  written with `ingest_session` under `lme:load:<sid>` source refs, which never are gold.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path

from bench.datasets import corpora

V50K_CHUNKS = 50_000
SKIP_ON_SNAPSHOT = (".git", ".trash")
PG_FOLDERS = ("personal", "projects")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(tarball: Path, expected_sha256: str) -> str:
    got = sha256_file(tarball)
    if got != expected_sha256:
        raise ValueError(f"{tarball}: sha256 {got} != expected {expected_sha256}")
    return got


def _check_members(names: list[str]) -> None:
    for n in names:
        parts = Path(n).parts
        if Path(n).is_absolute() or ".." in parts:
            raise ValueError(f"unsafe path in the snapshot: {n!r}")


def extract(tarball: Path, dest: Path) -> Path:
    """Extract into `dest` (created, must be empty or absent); returns the vault root."""
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    if str(tarball).endswith((".zst", ".zstd")):
        listing = subprocess.run(["tar", "--zstd", "-tf", str(tarball)], capture_output=True,
                                 text=True, check=True).stdout.splitlines()  # fmt: skip
        _check_members(listing)
        subprocess.run(["tar", "--zstd", "-xf", str(tarball), "-C", str(dest)], check=True)
    else:
        with tarfile.open(tarball) as tf:
            _check_members(tf.getnames())
            tf.extractall(dest, filter="data")
    entries = [p for p in dest.iterdir()]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else dest


def make_snapshot(vault: Path, out: Path) -> str:
    """Deterministic order, no owner or mtime noise in the member list; returns the sha256."""
    out.parent.mkdir(parents=True, exist_ok=True)
    excludes = [f"--exclude={x}" for x in SKIP_ON_SNAPSHOT]
    zst = str(out).endswith(".zst")
    cmd = ["tar", *(["--zstd"] if zst else []), "--sort=name", "--owner=0", "--group=0",
           "--numeric-owner", *excludes, "-cf", str(out), "-C", str(vault.parent), vault.name]  # fmt: skip
    subprocess.run(cmd, check=True)
    return sha256_file(out)


def pg_members(vault: Path, routes: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """(files kept, work-routed files left out) for the PG snapshot: files under `personal/` and
    `projects/` whose route (importer defaults + `vault.routes`) is the personal store. Paths are
    vault-relative posix, sorted; `.git` / `.trash` are skipped as in `make_snapshot`."""
    from mem.importer import route_of

    kept: list[str] = []
    dropped: list[str] = []
    for top in PG_FOLDERS:
        for root, dirs, files in os.walk(vault / top):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_ON_SNAPSHOT)
            for name in files:
                rel = (Path(root) / name).relative_to(vault).as_posix()
                (kept if route_of(rel, routes) == "personal" else dropped).append(rel)
    return sorted(kept), sorted(dropped)


def make_pg_snapshot(vault: Path, out: Path, routes: Mapping[str, str]) -> tuple[str, int, int]:
    """The PG snapshot (spec change 2026-10-24: PG-v1 holds no work-store content): only the files
    of `personal/` and `projects/` that are routed to the personal store, so no work chunk can
    reach a pool or qrels.tsv. Returns (sha256, files kept, work-routed files left out)."""
    kept, dropped = pg_members(vault, routes)
    if not kept:
        raise ValueError(f"no personal-routed file under {', '.join(PG_FOLDERS)} in {vault}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # a file, not stdin: `--zstd` reads stdin too
        names = Path(tmp) / "members"
        names.write_bytes(("\0".join(f"{vault.name}/{rel}" for rel in kept) + "\0").encode())
        cmd = ["tar", *(["--zstd"] if str(out).endswith(".zst") else []), "--owner=0",
               "--group=0", "--numeric-owner", "--no-recursion", "--null", "-cf", str(out),
               "-C", str(vault.parent), "-T", str(names)]  # fmt: skip
        subprocess.run(cmd, check=True)
    return sha256_file(out), len(kept), len(dropped)


def filler_plan(
    lme_m: Path, vault_chunks: int, total: int = V50K_CHUNKS
) -> list[corpora.PlannedSession]:
    """The LongMemEval_M sessions that bring a vault store of `vault_chunks` chunks to `total`."""
    if vault_chunks >= total:
        raise ValueError(
            f"the vault already has {vault_chunks} chunks (V50k needs fewer than {total})"
        )
    return corpora.plan_corpus_file(lme_m, total - vault_chunks)


def copy_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst
