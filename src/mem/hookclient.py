"""The Stop hook's thin client (F4, MEM2-194). Standard library only and no `mem` import, so it
runs as `python -S -E hookclient.py` without site-packages: it sends the hook input to the
running `mem serve` over its Unix socket and prints the reply. Fail-open: serve down, no socket,
a timeout or a garbled reply all print nothing and exit 0, and Claude stops as usual.

The reply is shown to Claude, so it is taken only from a socket inside this user's own private
directory (`is_private_dir`): `/tmp/mem-<uid>` can be made by anyone first, and serve refuses to
listen in such a directory (`hook._private_dir`, the same check), so the client refuses to read
from one (MEM2-206 F5)."""

import json
import os
import socket
import stat
import sys

TIMEOUT = 0.5  # seconds; the hook never holds Claude Code up longer than this
MAX_REPLY = 65536


def socket_path(env=os.environ) -> str:
    """`$XDG_RUNTIME_DIR/mem/hook.sock`; without a runtime dir, `/tmp/mem-<uid>/hook.sock`."""
    run = env.get("XDG_RUNTIME_DIR")
    return os.path.join(run, "mem", "hook.sock") if run else f"/tmp/mem-{os.getuid()}/hook.sock"


def own_dir_fd(path) -> int:
    """A descriptor on `path`, which must be a real directory (a symlink is refused, O_NOFOLLOW)
    owned by this user, else OSError. Checked on the descriptor, so what was checked is what the
    caller holds. The caller closes it."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
            raise PermissionError(f"{os.fspath(path)}: not a directory owned by this user")
    except BaseException:
        os.close(fd)
        raise
    return fd


def is_private_dir(path) -> bool:
    """True if `path` is this user's own real directory with mode 0700, as serve leaves it. A
    socket anywhere else may be anyone's, and its reply must not reach Claude."""
    try:
        fd = own_dir_fd(path)
    except OSError:
        return False
    try:
        return stat.S_IMODE(os.fstat(fd).st_mode) == 0o700
    finally:
        os.close(fd)


def dispatch(raw: str, path=None, timeout: float = TIMEOUT) -> str:
    """serve's reply to the hook input `raw`: a JSON object, or "" (allow the stop) on any failure,
    including a socket whose directory is not this user's private one."""
    try:
        target = os.fspath(path) if path is not None else socket_path()
        if not is_private_dir(os.path.dirname(target)):
            return ""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(target)
            s.sendall(raw.encode())
            s.shutdown(socket.SHUT_WR)
            chunks = []
            while chunk := s.recv(MAX_REPLY):
                chunks.append(chunk)
        reply = b"".join(chunks).decode().strip()
        return reply if reply and isinstance(json.loads(reply), dict) else ""
    except Exception:  # noqa: BLE001 - fail-open: nothing here may stop Claude from stopping
        return ""


def main() -> int:
    try:
        reply = dispatch(sys.stdin.read())
        if reply:
            sys.stdout.write(reply)
    except Exception:  # noqa: BLE001 - fail-open
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
