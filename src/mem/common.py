"""Small shared pieces of the read and write paths: errors, ids, text normalisation, time."""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MEMORY_TYPES = ("fact", "preference", "decision", "event", "procedure", "insight")
ENTITY_TYPES = ("person", "project", "org", "topic", "place", "thing")
PREFIX = {"memory": "m", "entity": "e", "note": "n", "tag": "t"}
KIND_OF_PREFIX = {v: k for k, v in PREFIX.items()}

_ID = re.compile(r"^([mentc])(\d+)$")
_TS = "%Y-%m-%dT%H:%M:%SZ"


class MemError(ValueError):
    """A request the caller can fix; the message says how."""


@dataclass
class ToolResult:
    text: str
    structured: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Server-side token estimate (§3.2 step 8): ceil(utf8 bytes / 3), deliberately high."""
    return math.ceil(len(text.encode("utf-8")) / 3)


def norm(text: str) -> str:
    """NFKC, casefold, trimmed, single spaces, no [[ ]] (aliases and content hashes)."""
    t = unicodedata.normalize("NFKC", text).replace("[[", "").replace("]]", "")
    return " ".join(t.casefold().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def parse_id(ident: str) -> tuple[str, int] | None:
    m = _ID.match(ident.strip())
    return (m.group(1), int(m.group(2))) if m else None


def fmt_ts(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_TS)


def _zone_key(name: str) -> str | None:
    """`name` as a loadable IANA key, or None. Accepts a ':' prefix and zoneinfo file paths."""
    name = name.removeprefix(":")
    if "/zoneinfo/" in name:
        name = name.split("/zoneinfo/", 1)[1]
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    return name


def local_tz(
    environ: Mapping[str, str] = os.environ, localtime: Path = Path("/etc/localtime")
) -> str:
    """The system's IANA zone: `TZ`, else the `/etc/localtime` symlink target, else UTC."""
    if key := _zone_key(environ.get("TZ", "")):
        return key
    if localtime.is_symlink() and (key := _zone_key(str(localtime.resolve()))):
        return key
    return "UTC"


def normalize_ts(value: str, tz: str = "UTC") -> str:
    """UTC 'YYYY-MM-DDTHH:MM:SSZ'. Date-only input means 00:00 in the local zone `tz`."""
    value = value.strip()
    try:
        if len(value) == 10:
            return fmt_ts(datetime.combine(date.fromisoformat(value), time(), ZoneInfo(tz)))
        parsed = datetime.fromisoformat(value)
    except ValueError as err:
        raise MemError(f"bad date {value!r}: use YYYY-MM-DD or an ISO timestamp") from err
    return fmt_ts(parsed if parsed.tzinfo else parsed.replace(tzinfo=ZoneInfo(tz)))


def normalize_as_of(value: str, tz: str = "UTC") -> str:
    """The read instant of a read's `as_of`, UTC. Date-only input means the END of that day in the
    local zone `tz` (23:59:59 local, ADR 0028): everything that happened that day is visible.
    Timestamps are read as by `normalize_ts`."""
    value = value.strip()
    if len(value) == 10:
        try:
            day = date.fromisoformat(value)
        except ValueError as err:
            raise MemError(f"bad date {value!r}: use YYYY-MM-DD or an ISO timestamp") from err
        return fmt_ts(datetime.combine(day, time(23, 59, 59), ZoneInfo(tz)))
    return normalize_ts(value, tz)
