"""Small helpers shared by the P1 suites."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from bench.suites.base import Client
from mem import reldates

ROW = re.compile(
    r"^(?P<id>[mcent]\d+) · (?P<type>[^·]+) · (?P<date>\d{4}-\d{2}-\d{2}|-) · "
    r"(?P<title>[^·]+?) · about: (?P<about>[^·]*)"
    rf"(?: · dates: (?P<dates>{reldates.ROW_FIELD}))? · (?P<p>\d*\.?\d+)$"
)


def node_num(shown_id: str) -> int:
    return int(shown_id[1:])


def sql(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def sql_write(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> None:
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(query, params)
    finally:
        conn.close()


def run_ops(client: Client, ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply an op stream; returns each op with its `structured` and `text` result."""
    out = []
    for op in ops:
        r = client.call(op["tool"], op["args"], now=op["now"])
        out.append({**op, "structured": r.structured, "text": r.text})
    return out


def seed_corpus(client: Client, corpus: list[tuple[str, str, list[str]]], now: str) -> list[str]:
    ids = []
    for text, type_, about in corpus:
        r = client.call("remember", {"text": text, "type": type_, "about": about}, now=now)
        ids.append(r.structured["id"])
        assert r.structured["status"] == "saved", (text, r.structured)
    return ids
