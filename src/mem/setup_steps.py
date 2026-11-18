"""What `mem setup` builds on (MEM2-232, docs/decisions/0018): the `HarnessWriter` interface the
harness integrations implement, and `Backups`. A writer that edits a config file saves it first in
`Backups`; `restore` puts the exact bytes and mode back (or deletes a file that did not exist, and
the folders made for it). Every system call goes through one `Runner`, the one `mem install` uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mem.common import MemError
from mem.install import Result, Runner, SystemRunner


def write_private(path: Path, data: bytes, mode: int = 0o600) -> None:
    """Write `path` (through a symlink) created 0600, never readable by others on the way, then
    give it `mode`. The harness configs and their copies hold bearer tokens."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        os.fchmod(f.fileno(), 0o600)  # an old file left there keeps its mode otherwise
        f.write(data)
    os.chmod(path, mode)


def replace_file(path: Path, data: bytes, mode: int) -> None:
    """Swap in new bytes atomically. A symlink (a dotfiles repo) stays a link: its target is
    written."""
    real = path.resolve() if path.is_symlink() else path
    tmp = real.with_name(f".{real.name}.mem")
    write_private(tmp, data, mode)
    os.replace(tmp, real)


class Backups:
    """Copies of config files taken before a step edits them. One slot per file; the first save
    wins, so running setup twice never replaces the true original with an edited file. The
    folder is 0700 and each copy keeps the mode of its file (they hold bearer tokens)."""

    def __init__(self, root: Path):
        self.root = root

    def _slot(self, path: Path) -> Path:
        return self.root / hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:24]

    def has(self, path: Path) -> bool:
        return (self._slot(path) / "meta.json").exists()

    def save(self, path: Path) -> None:
        if self.has(path):
            return
        slot = self._slot(path)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        slot.mkdir(mode=0o700, exist_ok=True)
        slot.chmod(0o700)
        existed = path.exists()
        mode = path.stat().st_mode & 0o7777 if existed else None
        if existed:
            write_private(slot / "data", path.read_bytes(), mode)  # type: ignore[arg-type]
        missing = [d for d in path.absolute().parents if not d.exists()]
        meta = {"path": str(path.absolute()), "existed": existed, "mode": mode,
                "made": str(missing[-1]) if missing else None}  # fmt: skip
        (slot / "meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )  # written last: marks it complete

    def original(self, path: Path) -> bytes | None:
        """The saved bytes (None: the file did not exist when it was saved)."""
        if not self.has(path):
            raise MemError(f"no backup of {path}")
        slot = self._slot(path)
        meta = json.loads((slot / "meta.json").read_text(encoding="utf-8"))
        return (slot / "data").read_bytes() if meta["existed"] else None

    def drop(self, path: Path) -> None:
        """Forget the backup without restoring it (the step edited the file back by hand)."""
        if self.has(path):
            shutil.rmtree(self._slot(path))

    def restore(self, path: Path) -> None:
        if not self.has(path):
            raise MemError(f"no backup of {path}")
        slot = self._slot(path)
        meta = json.loads((slot / "meta.json").read_text(encoding="utf-8"))
        if meta["existed"]:
            replace_file(path, (slot / "data").read_bytes(), meta["mode"])
        else:
            path.unlink(missing_ok=True)
            # the folders the edit created go too, when nothing else is in them
            if meta.get("made"):
                for folder in path.absolute().parents:
                    try:
                        folder.rmdir()
                    except OSError:
                        break
                    if folder == Path(meta["made"]):
                        break
        shutil.rmtree(slot)


@dataclass
class Context:
    runner: Runner
    backups: Backups
    home: Path


@dataclass(frozen=True)
class Endpoint:
    """One mem MCP server a harness should know: `name` as the harness lists it."""

    name: str
    url: str
    token_file: Path  # the bearer token, mode 0600; read when the writer needs it
    headers: tuple[tuple[str, str], ...] = ()


class HarnessWriter(Protocol):
    """One coding harness's MCP config (setup-harness fills these in; Claude Code's lives in
    `setup_cmd`). `add` saves each file in `ctx.backups` before the first edit; `remove` undoes
    every edit of ours, so the file is back byte for byte. `installed` says whether there is
    anything of ours to remove: any mem server, whatever its port or label, or a backup."""

    name: str

    def detect(self, ctx: Context) -> bool: ...
    def configured(self, ctx: Context, endpoints: list[Endpoint]) -> bool: ...
    def add(self, ctx: Context, endpoints: list[Endpoint]) -> None: ...
    def installed(self, ctx: Context) -> bool: ...
    def remove(self, ctx: Context) -> None: ...


def harness_writers() -> list[HarnessWriter]:
    """The writers beyond Claude Code (`harnesses.writers`)."""
    from mem import harnesses

    return list(harnesses.writers())


__all__ = [
    "Backups",
    "Context",
    "Endpoint",
    "HarnessWriter",
    "Result",
    "Runner",
    "SystemRunner",
    "harness_writers",
]
