"""The optional auto-capture Stop hook (F4, MEM2-194). Off by default: `mem install --hooks`
registers `hookclient.py` as a Stop hook in the user's Claude Code settings. `mem serve` answers
it over a Unix socket (`HookServer`).

Contract matched: Claude Code hooks reference (code.claude.com/docs/en/hooks, read 2026-10-26;
CLI 2.1.289). Stop input is a JSON object with `hook_event_name`, `session_id`, `stop_hook_active`,
...; output on exit 0 is `{"decision": "block", "reason": ...}` (Claude continues, reason shown to
it), or nothing to allow the stop.
"""

from __future__ import annotations

import contextlib
import json
import os
import socketserver
import stat
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from mem import hookclient

REASON = "Before stopping, save anything worth remembering with the mem skill, then stop."
MARKER = "hookclient.py"  # what identifies our handler in the user's settings
MAX_INPUT = 1 << 20
DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.json"


def decide(raw: str | bytes) -> dict[str, str] | None:
    """The reply to one Stop hook input, or None to allow the stop (no output). Anything that is
    not a Stop event object, or that has `stop_hook_active` true (we already blocked once: no
    loops), allows."""
    try:
        event = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(event, dict) or event.get("hook_event_name") != "Stop":
        return None
    if event.get("stop_hook_active") is not False and "stop_hook_active" in event:
        return None  # true, or anything not strictly false: do not risk a loop
    return {"decision": "block", "reason": REASON}


def socket_path(env: Mapping[str, str] = os.environ) -> Path:
    return Path(hookclient.socket_path(env))


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(2)
        try:
            raw = self.rfile.read(MAX_INPUT)  # the client half-closes after sending
            out = decide(raw)
            if out is not None:
                self.wfile.write(json.dumps(out).encode())
        except OSError:
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def _private_dir(path: Path) -> None:
    """Make `path` a directory only this user can enter, or raise OSError. A pre-existing one is
    taken only if it is a real directory (not a symlink) owned by this user (the client's check,
    `hookclient.own_dir_fd`), then set to 0700 so the client accepts it (`hookclient.is_private_dir`):
    `/tmp/mem-<uid>` can be made by anyone first, and whoever controls it controls the reply."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, 0o700)
    fd = hookclient.own_dir_fd(path)
    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


class HookServer:
    """The socket endpoint of `mem serve`: 0600 socket in a 0700 directory, on a thread (the
    reply needs no store). A stale socket file is replaced; the file is removed on close. The
    directory must be this user's own (`_private_dir`), else `start` raises OSError."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._srv: _Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> HookServer:
        _private_dir(self.path.parent)
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        old = os.umask(0o177)  # created 0600 from the start, never briefly wider
        try:
            self._srv = _Server(os.fspath(self.path), _Handler)
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    __enter__ = start

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- registration in the user's Claude Code settings ---


class SettingsError(Exception):
    pass


class Files(Protocol):
    def read(self, path: Path) -> str | None: ...
    def write(self, path: Path, text: str) -> None: ...


class RealFiles:
    def read(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except FileNotFoundError:
            return None

    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".mem-tmp")
        tmp.write_text(text)
        tmp.replace(path)


REAL = RealFiles()


def hook_command() -> str:
    import sys

    return f"{sys.executable} -S -E {hookclient.__file__}"


def _load(path: Path, fs: Files) -> dict[str, Any]:
    text = fs.read(path)
    if text is None or not text.strip():
        return {}
    try:
        cfg = json.loads(text)
    except ValueError as err:
        raise SettingsError(f"{path}: not JSON ({err}); left untouched") from err
    if not isinstance(cfg, dict):
        raise SettingsError(f"{path}: not a JSON object; left untouched")
    return cfg


def _ours(handler: Any) -> bool:
    return isinstance(handler, dict) and MARKER in str(handler.get("command", ""))


def _strip(cfg: dict[str, Any]) -> bool:
    """Remove our handlers from cfg's Stop hooks, pruning what that empties; True if any went."""
    groups = cfg.get("hooks", {}).get("Stop") if isinstance(cfg.get("hooks"), dict) else None
    if not isinstance(groups, list):
        return False
    removed, kept = False, []
    for g in groups:
        hs = g.get("hooks") if isinstance(g, dict) else None
        if isinstance(hs, list) and any(_ours(h) for h in hs):
            removed = True
            g = {**g, "hooks": [h for h in hs if not _ours(h)]}
            if not g["hooks"]:
                continue
        kept.append(g)
    if removed:
        if kept:
            cfg["hooks"]["Stop"] = kept
        else:
            del cfg["hooks"]["Stop"]
            if not cfg["hooks"]:
                del cfg["hooks"]
    return removed


def _put(path: Path, cfg: dict[str, Any], fs: Files) -> None:
    fs.write(path, json.dumps(cfg, indent=2) + "\n")


def install(path: Path, command: str, *, fs: Files = REAL, dry_run: bool = False) -> str:
    """Register `command` as the Stop hook. Idempotent; a moved command replaces the old one."""
    cfg = _load(path, fs)
    entry = {"type": "command", "command": command, "timeout": 5}
    stop = cfg.get("hooks", {}).get("Stop", []) if isinstance(cfg.get("hooks"), dict) else []
    present = [h for g in stop if isinstance(g, dict) for h in g.get("hooks", []) if _ours(h)]
    if present == [entry]:
        return "unchanged"
    if dry_run:
        return "would install"
    _strip(cfg)
    cfg.setdefault("hooks", {}).setdefault("Stop", []).append({"hooks": [entry]})
    _put(path, cfg, fs)
    return "installed"


def installed(path: Path = DEFAULT_SETTINGS, *, fs: Files = REAL) -> bool:
    """Whether our Stop hook is registered in `path`. Unreadable settings count as not."""
    try:
        cfg = _load(path, fs)
    except (SettingsError, OSError):
        return False
    stop = cfg.get("hooks", {}).get("Stop") if isinstance(cfg.get("hooks"), dict) else None
    groups = stop if isinstance(stop, list) else []
    return any(_ours(h) for g in groups if isinstance(g, dict)
               for h in (g.get("hooks") if isinstance(g.get("hooks"), list) else []))  # fmt: skip


def uninstall(path: Path, *, fs: Files = REAL, dry_run: bool = False) -> str:
    cfg = _load(path, fs)
    if not _strip(cfg):
        return "unchanged"
    if dry_run:
        return "would remove"
    _put(path, cfg, fs)
    return "removed"
