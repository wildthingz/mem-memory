"""Store schema (architecture-v2 §2.3) and its numbered migrations.

Each migration is `migrations/NNNN_<name>.sql`; `meta.schema_version` holds the highest applied
number. Vector tables are per embedder fingerprint and are created at runtime.
"""

from __future__ import annotations

import re
import sqlite3
from importlib import resources
from pathlib import Path

import sqlite_vec

from mem.common import MemError

_MIGRATION = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def _migrations() -> list[tuple[int, str]]:
    found = []
    for entry in resources.files(__name__).joinpath("migrations").iterdir():
        m = _MIGRATION.match(entry.name)
        if m:
            found.append((int(m.group(1)), entry.read_text(encoding="utf-8")))
    return sorted(found)


def latest_version() -> int:
    return _migrations()[-1][0]


def schema_version(conn: sqlite3.Connection) -> int:
    has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'meta'").fetchone()
    if not has_meta:
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations in order; return the resulting schema version. A store whose version
    is newer than the code's latest migration is refused: older code must not write it (MEM2-222 F21)."""
    current = schema_version(conn)
    if current > (latest := latest_version()):
        raise MemError(
            f"store schema version {current} is newer than this code knows ({latest}); upgrade mem"
        )
    for number, sql in _migrations():
        if number <= current:
            continue
        conn.executescript(f"BEGIN;\n{sql}\nUPDATE meta SET value = '{number}' "
                           "WHERE key = 'schema_version';\nCOMMIT;")  # fmt: skip
        current = number
    return current


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a store: pragmas, sqlite-vec, migrations applied. Autocommit-off is up to callers."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


def vec_table_name(fp: str) -> str:
    return f"vec_{fp[:8]}"


def create_vec_table(conn: sqlite3.Connection, fp: str, dim: int) -> str:
    """Create the per-fingerprint vec0 table (A21); the app deletes vec rows with their chunk."""
    name = vec_table_name(fp)
    if not re.fullmatch(r"vec_[0-9a-f]{8}", name) or dim <= 0:
        raise ValueError(f"bad fingerprint or dimension: {fp!r}, {dim!r}")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
    )
    return name


def session_table_name(vec_table: str) -> str:
    """The session-vector table that goes with chunk table `vec_<fp8>`: `svec_<fp8>` (ADR 0022)."""
    return "s" + vec_table


def create_session_table(conn: sqlite3.Connection, vec_table: str, dim: int) -> str:
    """Create the per-fingerprint vec0 table of one vector per note/session node (ADR 0022)."""
    name = session_table_name(vec_table)
    if not re.fullmatch(r"svec_[0-9a-f]{8}", name) or dim <= 0:
        raise ValueError(f"bad vector table or dimension: {vec_table!r}, {dim!r}")
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0("
        f"node_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
    )
    return name
