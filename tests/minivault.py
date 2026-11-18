"""Helpers for the mini-vault fixture (tests/fixtures/minivault): a working copy of the frozen
vault, and `dump_store`, the name-keyed (id-free) view of a store the expected dumps are written in."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "minivault"
MTIME = 1768478400.0  # 2026-01-15T12:00:00Z: the mtime fallback for notes with no date

# Git ignores node_modules/, so the skipped dir is created in the working copy instead.
_UNTRACKED_SKIPPED = {"node_modules/pkg/README.md": "# pkg\n\nSkipped.\n"}


def make_vault(tmp: Path) -> Path:
    vault = tmp / "vault"
    shutil.copytree(FIXTURE / "vault", vault)
    for rel, text in _UNTRACKED_SKIPPED.items():
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text, encoding="utf-8")
    for path in vault.rglob("*"):
        if path.is_file():
            os.utime(path, (MTIME, MTIME))
    return vault


def expected(name: str) -> Any:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def label(row: sqlite3.Row | tuple) -> str:
    """source_ref for imported files, else `kind:type:name` (tags: `tag::name`)."""
    kind, type_, name, ref = row
    return ref or (f"tag::{name}" if kind == "tag" else f"{kind}:{type_}:{name}")


def dump_store(conn: sqlite3.Connection) -> dict[str, Any]:
    labels = {
        r[0]: label(r[1:])
        for r in conn.execute("SELECT id, kind, type, name, source_ref FROM nodes").fetchall()
    }
    nodes = [
        {
            "label": labels[r[0]],
            "kind": r[1],
            "type": r[2],
            "name": r[3],
            "date": r[4],
            "status": r[5],
        }
        for r in conn.execute(
            "SELECT id, kind, type, name, date, status FROM nodes WHERE origin='import'"
            " ORDER BY 2, 3, 4"
        )
    ]
    imported = {n["label"] for n in nodes}
    aliases: dict[str, list[str]] = {}
    for nid, alias in conn.execute(
        "SELECT node_id, alias FROM aliases WHERE origin='import' ORDER BY alias"
    ):
        aliases.setdefault(labels[nid], []).append(alias)
    props: dict[str, dict[str, str]] = {}
    for nid, key, value in conn.execute(
        "SELECT node_id, key, value FROM props WHERE origin='import' ORDER BY key"
    ):
        props.setdefault(labels[nid], {})[key] = value
    chunks: dict[str, list[dict[str, str]]] = {}
    for nid, key, hp, props_json in conn.execute(
        "SELECT node_id, chunk_key, heading_path, props FROM chunks ORDER BY node_id, ord"
    ):
        chunks.setdefault(labels[nid], []).append({"key": key, "heading_path": hp or ""})
    edges = []
    for eid, src, dst, rel, removed in conn.execute(
        "SELECT id, src, dst, rel, removed_at IS NULL FROM edges WHERE origin='import'"
    ):
        occ = [
            {"where": ck if ck else f"field:{field}", "anchor": anchor, "alias": alias}
            for ck, field, anchor, alias in conn.execute(
                "SELECT c.chunk_key, o.field, o.anchor, o.alias FROM link_occurrences o"
                " LEFT JOIN chunks c ON c.id = o.chunk_id WHERE o.edge_id = ?",
                (eid,),
            )
        ]
        occ.sort(key=lambda o: (o["where"], o["anchor"] or "", o["alias"] or ""))
        edges.append(
            {"src": labels[src], "dst": labels[dst], "rel": rel, "live": bool(removed), "occ": occ}
        )
    edges.sort(key=lambda e: (e["src"], e["rel"], e["dst"]))
    return {
        "nodes": nodes,
        "aliases": aliases,
        "props": props,
        "chunks": {k: v for k, v in chunks.items() if k in imported},
        "edges": edges,
    }
