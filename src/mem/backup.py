"""Encrypted store backups and restore (architecture-v2 §6 P7, benchmarks-v2 B-H1).

`backup_store` takes a consistent snapshot with `VACUUM INTO` (safe while `mem serve` writes: it
reads one WAL snapshot), checks it, and encrypts it into `<name>-<UTC stamp>.mbk`. The plaintext
snapshot only exists inside a 0700 temp folder next to the output and is removed at once.

File format (`.mbk`, v1): `MAGIC`, a 4-byte big-endian header length, a JSON header (store name,
created, plaintext size and sha256, salt), then chunks of `CHUNK` plaintext bytes, each stored as
a 4-byte length and its AES-256-GCM ciphertext. The per-file key is HKDF-SHA256(key file, salt);
chunk i uses nonce i (11 bytes) + a last-chunk flag, and every chunk authenticates the header, so
a wrong key, a flipped bit, reordered chunks or a cut-off tail all fail to decrypt (STREAM).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import stat
import struct
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import sqlite_vec
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from mem.common import MemError

MAGIC = b"MEMBK1\n"
CHUNK = 1 << 20
SUFFIX = ".mbk"
_LEN = struct.Struct(">I")


class BackupError(MemError):
    pass


def new_key(path: Path) -> None:
    """Write 32 random bytes as hex to `path`, owner-only. Never overwrites an existing key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise BackupError(
            f"{path} exists; a new key would orphan every backup made with it"
        ) from None
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32) + "\n")


def load_key(path: Path) -> bytes:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        text = path.read_text(encoding="ascii").strip()
    except OSError as err:
        raise BackupError(f"cannot read key file {path}: {err}") from err
    if mode & 0o077:
        raise BackupError(f"{path} is readable by others (mode {mode:o}); chmod 600 {path}")
    try:
        key = bytes.fromhex(text)
    except ValueError:
        key = b""
    if len(key) != 32:
        raise BackupError(f"{path} is not a mem backup key (64 hex characters)")
    return key


def _new_private(path: Path) -> BinaryIO:
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _subkey(key: bytes, salt: bytes) -> AESGCM:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=b"mem-backup-v1")
    return AESGCM(hkdf.derive(key))


def _nonce(i: int, last: bool) -> bytes:
    return i.to_bytes(11, "big") + (b"\x01" if last else b"\x00")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _integrity(db: Path) -> str:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as err:
        return str(err)
    finally:
        conn.close()
    return "ok" if rows == [("ok",)] else "; ".join(r[0] for r in rows[:5])


def _snapshot(db: Path, dest: Path) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("VACUUM INTO ?", (str(dest),))
    finally:
        conn.close()


def _encrypt(src: Path, dest: BinaryIO, key: bytes, meta: dict[str, Any]) -> None:
    salt = secrets.token_bytes(16)
    header = json.dumps({**meta, "salt": salt.hex()}, sort_keys=True).encode("utf-8")
    dest.write(MAGIC + _LEN.pack(len(header)) + header)
    aead = _subkey(key, salt)
    with src.open("rb") as f:
        block, i = f.read(CHUNK), 0
        while True:
            nxt = f.read(CHUNK)
            ct = aead.encrypt(_nonce(i, not nxt), block, header)
            dest.write(_LEN.pack(len(ct)) + ct)
            if not nxt:
                return
            block, i = nxt, i + 1


def read_header(f: BinaryIO) -> tuple[bytes, dict[str, Any]]:
    if f.read(len(MAGIC)) != MAGIC:
        raise BackupError("not a mem backup (bad magic)")
    (n,) = _LEN.unpack(_read_exact(f, _LEN.size))
    header = _read_exact(f, n)
    try:
        return header, json.loads(header)
    except ValueError:
        raise BackupError("corrupted backup header") from None


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise BackupError("backup is truncated")
    return data


def _decrypt(f: BinaryIO, key: bytes) -> tuple[dict[str, Any], Iterator[bytes]]:
    header, meta = read_header(f)
    aead = _subkey(key, bytes.fromhex(meta["salt"]))

    def chunks() -> Iterator[bytes]:
        i = 0
        prefix = f.read(_LEN.size)
        while prefix:
            if len(prefix) != _LEN.size:
                raise BackupError("backup is truncated")
            ct = _read_exact(f, _LEN.unpack(prefix)[0])
            prefix = f.read(_LEN.size)
            try:
                yield aead.decrypt(_nonce(i, not prefix), ct, header)
            except InvalidTag:
                raise BackupError("cannot decrypt: wrong key or corrupted backup") from None
            i += 1
        if i == 0:
            raise BackupError("backup is truncated")

    return meta, chunks()


def backup_store(
    db: Path, out_dir: Path, key: bytes, *, name: str, now: datetime | None = None
) -> Path:
    """Snapshot `db`, check it, encrypt it to `out_dir/<name>-<stamp>.mbk`; return that path."""
    if not db.is_file():
        raise BackupError(f"no store at {db}")
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    when = (now or datetime.now(UTC)).astimezone(UTC)
    out = out_dir / f"{name}-{when:%Y%m%dT%H%M%SZ}{SUFFIX}"
    tmp = Path(tempfile.mkdtemp(prefix=".mem-backup-", dir=out_dir))  # 0700
    try:
        snap = tmp / "snapshot.db"
        _snapshot(db, snap)
        check = _integrity(snap)
        if check != "ok":
            raise BackupError(f"snapshot of {db} failed integrity_check: {check}")
        meta = {"format": 1, "store": name, "source": str(db), "created": when.isoformat(),
                "size": snap.stat().st_size, "sha256": _sha256(snap)}  # fmt: skip
        part = tmp / "out.part"
        with _new_private(part) as f:
            _encrypt(snap, f, key, meta)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(part, out)  # no-clobber
        except FileExistsError:
            raise BackupError(f"{out} exists") from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def restore(src: Path, key: bytes, dest: Path) -> Path:
    """Decrypt `src` into a new file `dest`, verified (size, sha256, integrity_check)."""
    if dest.exists():
        raise BackupError(f"{dest} exists; restore only writes a new file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f".{dest.name}.part"
    try:
        h, size = hashlib.sha256(), 0
        with src.open("rb") as f, _new_private(part) as out:
            meta, chunks = _decrypt(f, key)
            for block in chunks:
                h.update(block)
                size += len(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if (size, h.hexdigest()) != (meta["size"], meta["sha256"]):
            raise BackupError("restored bytes do not match the backup's size/sha256")
        check = _integrity(part)
        if check != "ok":
            raise BackupError(f"restored store failed integrity_check: {check}")
        try:
            os.link(part, dest)
        except FileExistsError:
            raise BackupError(f"{dest} exists; restore only writes a new file") from None
    except OSError as err:
        raise BackupError(f"restore failed: {err}") from err
    finally:
        part.unlink(missing_ok=True)
    return dest


def prune(out_dir: Path, name: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` backups of store `name`; return what was deleted."""
    if keep < 1:
        raise BackupError("keep must be at least 1")
    found = sorted(out_dir.glob(f"{name}-*{SUFFIX}"))
    old = found[:-keep]
    for p in old:
        p.unlink()
    return old
