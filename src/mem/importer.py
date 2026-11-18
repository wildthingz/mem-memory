"""`mem import`: Obsidian vault → graph (architecture-v2 §2.4, decisions 0010 and 0011).

One run scans the vault, classifies every note per store (unchanged / changed / added / renamed /
restored / vanished) and applies each file in its own `BEGIN IMMEDIATE` transaction. The vault owns
the notes: import-origin rows are rebuilt from the file, agent and user rows are never touched.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from mem import reldates
from mem.common import MemError, fmt_ts, normalize_ts
from mem.store import Store
from mem.vaultparse import (
    DATE_PREFIX,
    ChunkPlan,
    Link,
    ParsedNote,
    ParseError,
    file_hash,
    normalize,
    parse_note,
)

log = logging.getLogger(__name__)

DEFAULT_ROUTES = {"personal": "personal", "projects": "personal", "work": "work"}
SKIP_DIRS = {".obsidian", ".smart-env", ".stfolder", "Templates", "node_modules"}
SKIP_FILES = {"CLAUDE.md", "obsidian-instructions.md"}
_NEW_KEY = "\x00new"  # temporary chunk_key of a replacement, until the old row is deleted

EdgeKey = tuple[str, str]  # ("path" | "topic" | "tag", ref or casefolded name)


@dataclass
class ImportReport:
    added: int = 0
    changed: int = 0
    renamed: int = 0
    deleted: int = 0
    restored: int = 0
    unchanged: int = 0
    unrouted: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Note:
    rel: str
    path: Path
    mtime: float
    hash: str
    parsed: ParsedNote


def route_of(rel: str, routes: Mapping[str, str]) -> str | None:
    """The store for a vault-relative path: the longest matching folder prefix wins."""
    merged = {**DEFAULT_ROUTES, **{k.strip("/"): v for k, v in routes.items()}}
    best = max((f for f in merged if rel.startswith(f + "/")), key=len, default=None)
    return merged[best] if best else None


def _skipped(rel: str, render_dir: str) -> bool:
    parts = rel.split("/")
    return (
        parts[0] == render_dir.strip("/")
        or parts[0] == "Templates"
        or parts[-1] in SKIP_FILES
        or any(p in SKIP_DIRS for p in parts[:-1])
    )


def scan(vault: Path, render_dir: str) -> list[str]:
    """Every file of the vault that the skip list does not exclude, as sorted posix paths."""
    found = []
    for root, dirs, files in os.walk(vault):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        base = Path(root).relative_to(vault)
        for name in files:
            rel = (base / name).as_posix()
            if not _skipped(rel, render_dir):
                found.append(rel)
    return sorted(found)


class _Resolver:
    """Obsidian's link resolution inside one store: exact path, then unique name, then the
    shortest path. Notes whose file is gone (deleted nodes) answer only when no file does, so a
    link to a deleted note keeps pointing at it (§2.4 step 5)."""

    def __init__(self, files: list[str], ghosts: list[str], self_ref: str | None = None) -> None:
        self.files, self.ghosts = files, ghosts
        self.self_ref = self_ref

    @staticmethod
    def _pick(pool: list[str], target: str) -> str | None:
        t = _fold(target).removesuffix(".md")
        exact = [p for p in pool if _fold(_noext(p)) == t]
        suffix = [p for p in pool if _fold(_noext(p)).endswith("/" + t)]
        hits = exact or suffix
        return min(hits, key=lambda p: (len(p), p)) if hits else None

    def resolve(self, target: str) -> EdgeKey:
        t = target.strip().lstrip("/")
        for pool in (self.files, self.ghosts):
            md = [p for p in pool if p.endswith(".md")]
            if hit := self._pick(md, t) or self._pick_file(pool, t):
                return ("path", hit)
        return ("topic", t.casefold())

    @staticmethod
    def _pick_file(pool: list[str], target: str) -> str | None:
        hits = [p for p in pool if not p.endswith(".md")
                and (_fold(p) == _fold(target) or _fold(p).endswith("/" + _fold(target)))]  # fmt: skip
        return min(hits, key=lambda p: (len(p), p)) if hits else None


def _fold(name: str) -> str:
    """Case- and normalisation-insensitive form: a link typed NFC still finds a file named NFD."""
    return unicodedata.normalize("NFC", name).casefold()


def _noext(path: str) -> str:
    return path.removesuffix(".md")


def _desired(note: _Note, resolver: _Resolver) -> dict[tuple[EdgeKey, str], str]:
    """(dst key, rel) → display name of the target, for every edge the note should carry."""
    p, out = note.parsed, {}
    for tag in p.tags:
        out[(("tag", tag.casefold()), "tagged")] = tag
    for fl in p.fm_links:
        _add_link(out, resolver, note.rel, fl.link, fl.rel)
    for chunk in p.chunks:
        for link in chunk.links:
            _add_link(out, resolver, note.rel, link, "embeds" if link.embed else "links_to")
    return out


def _add_link(out: dict, resolver: _Resolver, own: str, link: Link, rel: str) -> None:
    if not link.target:
        return  # [[#Heading]]: a link to the note itself
    key = resolver.resolve(link.target)
    if key == ("path", own):
        return
    out.setdefault((key, rel), link.target.strip())


def _stored_keys(conn: sqlite3.Connection, node_id: int) -> set[tuple[EdgeKey, str]]:
    keys = set()
    for rel, kind, name, ref in conn.execute(
        "SELECT e.rel, n.kind, n.name, n.source_ref FROM edges e JOIN nodes n ON n.id = e.dst"
        " WHERE e.src = ? AND e.origin = 'import' AND e.removed_at IS NULL",
        (node_id,),
    ):
        key = ("tag", name.casefold()) if kind == "tag" else (
            ("path", ref) if ref else ("topic", name.casefold()))  # fmt: skip
        keys.add((key, rel))
    return keys


def _stamp(value: object, tz: str) -> str | None:
    try:
        if isinstance(value, datetime):
            return normalize_ts(value.isoformat(), tz)
        return normalize_ts(value.isoformat() if isinstance(value, date) else str(value), tz)
    except (MemError, ValueError):
        return None


def _stated_date(note: _Note, tz: str) -> str | None:
    """`created`, else the date in the filename; None when the note states neither."""
    for value in (note.parsed.created, note.parsed.filename_date):
        if value is not None and (stamp := _stamp(value, tz)):
            return stamp
    return None


def _note_date(note: _Note, tz: str) -> str:
    """The node's date (ADR 0010): the stated date, else the file's mtime."""
    return _stated_date(note, tz) or fmt_ts(datetime.fromtimestamp(note.mtime, UTC))


def _annotations(store: Store, note: _Note) -> dict[str, str | None]:
    """chunk key -> the date annotation of its text (ADR 0021), against the note's date in the
    store's zone; empty with the switch off or when the note states no date. Never the mtime: it
    is the clock of whoever last copied the vault, not the note's date, so anchoring to it would
    guess (ADR 0021) and change on every clone or sync."""
    if not store.resolve_dates or (stated := _stated_date(note, store.tz)) is None:
        return {}
    ref = store.ref_date(stated)
    return {c.key: reldates.annotation(c.text, ref) for c in note.parsed.chunks}


def _annotated(conn: sqlite3.Connection, store: Store, node_id: int, note: _Note) -> bool:
    """The stored chunks carry exactly the annotations the switch calls for."""
    want = _annotations(store, note)
    have = {k: d for k, d in conn.execute(
        "SELECT chunk_key, dates FROM chunks WHERE node_id = ?", (node_id,)) if d is not None}  # fmt: skip
    return have == {k: d for k, d in want.items() if d is not None}


def _created_warning(note: _Note, tz: str) -> str | None:
    created = note.parsed.created
    if created is None or _stamp(created, tz):
        return None
    return f"{note.rel}: created {str(created)!r} is not a date; the note is dated by its filename or mtime"


def run_import(
    vault: Path,
    stores: Mapping[str, Store],
    *,
    routes: Mapping[str, str],
    render_dir: str = "export",
) -> ImportReport:
    """Import `vault` into the stores ("personal", "work"); the report sums both stores."""
    vault = Path(vault)
    report = ImportReport()
    files = scan(vault, render_dir)
    by_store: dict[str, list[str]] = {name: [] for name in stores}
    for rel in files:
        name = route_of(rel, routes)
        if name is None:
            if rel.endswith(".md"):
                report.unrouted.append(rel)
        elif name in by_store:
            by_store[name].append(rel)
    for name, store in stores.items():
        _import_store(vault, store, by_store[name], report)
    return report


def _read(vault: Path, rel: str) -> _Note:
    path = vault / rel
    text = normalize(path.read_text(encoding="utf-8"))
    return _Note(rel, path, path.stat().st_mtime, file_hash(text), parse_note(rel, text))


def _import_store(vault: Path, store: Store, files: list[str], report: ImportReport) -> None:
    conn = store.conn
    md = [f for f in files if f.endswith(".md")]
    notes: dict[str, _Note] = {}
    broken: set[str] = set()
    for rel in md:
        try:
            notes[rel] = _read(vault, rel)
        except (ParseError, yaml.YAMLError, UnicodeDecodeError, OSError) as err:
            log.warning("skipping %s: %s", rel, err)
            report.errors.append((rel, str(err)))
            broken.add(rel)
    synced = {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT path, node_id, content_hash FROM source_files")}  # fmt: skip
    ghosts = [r[0] for r in conn.execute(
        "SELECT source_ref FROM nodes WHERE origin = 'import' AND source_ref LIKE '%.md'")
        if r[0] not in files]  # fmt: skip
    resolver = _Resolver(files, ghosts)

    todo: list[tuple[str, str, int | None]] = []  # (mode, rel, node id)
    fresh: list[str] = []
    for rel, note in notes.items():
        if rel in synced:
            node_id, old_hash = synced[rel]
            if (old_hash == note.hash and _annotated(conn, store, node_id, note)
                    and _stored_keys(conn, node_id) == set(_desired(note, resolver))):  # fmt: skip
                report.unchanged += 1
            else:
                todo.append(("changed", rel, node_id))
        else:
            fresh.append(rel)
    vanished = {r: v for r, v in synced.items() if r not in notes and r not in broken}
    pool = sorted(
        [(nid, h, r) for r, (nid, h) in vanished.items()]
        + [(r[0], r[1], r[2]) for r in conn.execute(
            "SELECT id, content_hash, source_ref FROM nodes WHERE status = 'deleted'"
            " AND origin = 'import' AND content_hash IS NOT NULL AND source_ref LIKE '%.md'")
           if r[2] not in files],
    )  # fmt: skip
    renames: list[tuple[str, str, int | None]] = []
    for rel in fresh:
        at_path = conn.execute("SELECT id, status FROM nodes WHERE origin = 'import' AND source_ref = ?",
                               (rel,)).fetchone()  # fmt: skip
        if at_path:
            mode = "restored" if at_path[1] == "deleted" else "added"
            todo.append((mode, rel, at_path[0]))
            continue
        match = next((p for p in pool if p[1] == notes[rel].hash), None)
        if match:
            pool.remove(match)
            renames.append(("renamed", rel, match[0]))
            vanished.pop(match[2], None)
        else:
            todo.append(("added", rel, None))
    for mode, rel, node_id in [*renames, *sorted(todo, key=lambda t: t[1])]:
        resolver.self_ref = rel
        if warning := _created_warning(notes[rel], store.tz):
            report.warnings.append(warning)
        _apply(store, vault, notes[rel], node_id, mode, resolver)
        setattr(report, mode, getattr(report, mode) + 1)
    for rel, (node_id, _) in sorted(vanished.items()):
        _delete(store, rel, node_id)
        report.deleted += 1


def _delete(store: Store, rel: str, node_id: int) -> None:
    conn = store.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("UPDATE nodes SET status = 'deleted', status_at = ? WHERE id = ?",
                     (store.now_ts(), node_id))  # fmt: skip
        conn.execute("DELETE FROM source_files WHERE path = ?", (rel,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _embed_needed(
    store: Store, node_id: int | None, plans: list[ChunkPlan]
) -> dict[tuple[str, str], list[float]]:
    """Vectors for the chunks whose (text, ctx) the store does not already hold for this node,
    computed outside the write transaction (§2.4 step 3)."""
    have: dict[str, tuple[str, str]] = {}
    if node_id is not None:
        have = {k: (h, c) for k, h, c in store.conn.execute(
            "SELECT chunk_key, content_hash, ctx FROM chunks WHERE node_id = ?", (node_id,))}  # fmt: skip
    need = {(c.content_hash, c.ctx): c for c in plans if have.get(c.key) != (c.content_hash, c.ctx)}
    rows = [(None, c.ctx, c.text) for c in need.values()]
    vecs = store.embed_chunks(rows) if rows else []
    return dict(zip(need, vecs, strict=True))


def _apply(
    store: Store, vault: Path, note: _Note, node_id: int | None, mode: str, resolver: _Resolver
) -> None:
    conn, now, p = store.conn, store.now_ts(), note.parsed
    vectors = _embed_needed(store, node_id, p.chunks)
    desired = _desired(note, resolver)
    conn.execute("BEGIN IMMEDIATE")
    try:
        nid = _upsert_node(conn, store, note, node_id, mode, now)
        _reconcile_labels(conn, nid, p)
        chunk_ids = _reconcile_chunks(store, nid, p.chunks, vectors, _annotations(store, note))
        store.refresh_session(nid)
        _reconcile_edges(conn, nid, note, resolver, desired, chunk_ids, now)
        conn.execute("DELETE FROM source_files WHERE node_id = ? AND path <> ?", (nid, note.rel))
        conn.execute(
            "INSERT OR REPLACE INTO source_files(path, vault, node_id, content_hash, mtime,"
            " imported_at) VALUES (?, ?, ?, ?, ?, ?)",
            (note.rel, str(vault), nid, note.hash, note.mtime, now),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _upsert_node(
    conn: sqlite3.Connection, store: Store, note: _Note, node_id: int | None, mode: str, now: str
) -> int:
    p = note.parsed
    date_ = _note_date(note, store.tz)
    fields = (p.kind, p.type, p.title, date_, date_, note.hash, note.rel)
    if node_id is None:
        row = conn.execute("SELECT id FROM nodes WHERE origin = 'import' AND source_ref = ?",
                           (note.rel,)).fetchone()  # fmt: skip
        node_id = row[0] if row else None
    if node_id is None:
        return conn.execute(
            "INSERT INTO nodes(kind, type, name, date, valid_from, content_hash, source_ref,"
            " recorded_at, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'import')",
            (*fields, now),
        ).lastrowid  # type: ignore[return-value]
    conn.execute(
        "UPDATE nodes SET kind = ?, type = ?, name = ?, date = ?, valid_from = ?, content_hash = ?,"
        " source_ref = ?, status = 'active', status_at = ?, valid_to = NULL WHERE id = ?",
        (*fields, now if mode == "restored" else None, node_id),
    )
    return node_id


def _reconcile_labels(conn: sqlite3.Connection, nid: int, p: ParsedNote) -> None:
    """Aliases and props of origin 'import' are replaced; other origins stay (step 3)."""
    from mem.common import norm

    conn.execute("DELETE FROM aliases WHERE node_id = ? AND origin = 'import'", (nid,))
    conn.executemany(
        "INSERT OR IGNORE INTO aliases(alias, alias_norm, node_id, origin) VALUES (?, ?, ?, 'import')",
        [(a, norm(a), nid) for a in p.aliases],
    )
    conn.execute("DELETE FROM props WHERE node_id = ? AND origin = 'import'", (nid,))
    conn.executemany(
        "INSERT OR IGNORE INTO props(node_id, key, value, origin) VALUES (?, ?, ?, 'import')",
        [(nid, k, v) for k, v in p.props],
    )


def _reconcile_chunks(
    store: Store,
    nid: int,
    plans: list[ChunkPlan],
    vectors: dict[tuple[str, str], list[float]],
    dates: dict[str, str | None],
) -> dict[str, int]:
    """Diff by chunk_key (step 4). A kept chunk keeps its id; a replaced one is inserted before
    the old row is deleted, so SQLite cannot hand the old id to the new row (a stale id must stay
    gone, §2.3). Body occurrences of every chunk of the node are dropped here and re-inserted from
    the new plan, which is what makes kept chunks safe against `occ_body` (A3, step 4.5)."""
    conn, vec = store.conn, store.vec_table
    old = {k: (i, h, c) for i, k, h, c in conn.execute(
        "SELECT id, chunk_key, content_hash, ctx FROM chunks WHERE node_id = ?", (nid,))}  # fmt: skip
    conn.execute("DELETE FROM link_occurrences WHERE chunk_id IN"
                 " (SELECT id FROM chunks WHERE node_id = ?)", (nid,))  # fmt: skip
    ids: dict[str, int] = {}
    inserted: list[tuple[int, ChunkPlan]] = []
    replaced = {k for k in old if k in {c.key for c in plans}
                and not any(c.key == k and c.content_hash == old[k][1] for c in plans)}  # fmt: skip
    for c in plans:
        if c.key in old and c.key not in replaced:
            cid, _, ctx = old[c.key]
            conn.execute("UPDATE chunks SET ord = ?, heading_path = ?, props = ?, tokens = ?,"
                         " ctx = ?, dates = ? WHERE id = ?", (c.ord, c.heading_path, c.props,
                                                          c.tokens, c.ctx, dates.get(c.key), cid))  # fmt: skip
            if ctx != c.ctx:
                conn.execute(f"DELETE FROM {vec} WHERE chunk_id = ?", (cid,))
                inserted.append((cid, c))
            ids[c.key] = cid
            continue
        key = c.key + _NEW_KEY if c.key in old else c.key
        cid = conn.execute(
            "INSERT INTO chunks(node_id, chunk_key, ord, key_kind, heading_path, ctx, text,"
            " tokens, content_hash, props, dates) VALUES (?, ?, ?, 'section', ?, ?, ?, ?, ?, ?, ?)",
            (
                nid,
                key,
                c.ord,
                c.heading_path,
                c.ctx,
                c.text,
                c.tokens,
                c.content_hash,
                c.props,
                dates.get(c.key),
            ),
        ).lastrowid
        ids[c.key] = cid  # type: ignore[assignment]
        inserted.append((cid, c))  # type: ignore[arg-type]
    gone = [i for k, (i, _, _) in old.items() if k in replaced or k not in ids]
    for cid in gone:
        conn.execute(f"DELETE FROM {vec} WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE id = ?", (cid,))
    for k in replaced:
        conn.execute("UPDATE chunks SET chunk_key = ? WHERE id = ?", (k, ids[k]))
    missing = [(cid, c) for cid, c in inserted if (c.content_hash, c.ctx) not in vectors]
    if missing:
        vectors = {**vectors, **dict(zip(
            [(c.content_hash, c.ctx) for _, c in missing],
            store.embed_chunks([(cid, c.ctx, c.text) for cid, c in missing]), strict=True))}  # fmt: skip
    store.add_vectors([(cid, c.ctx, c.text) for cid, c in inserted],
                      [vectors[(c.content_hash, c.ctx)] for _, c in inserted])  # fmt: skip
    return ids


def _reconcile_edges(
    conn: sqlite3.Connection, nid: int, note: _Note, resolver: _Resolver,
    desired: dict[tuple[EdgeKey, str], str], chunk_ids: dict[str, int], now: str,
) -> None:  # fmt: skip
    """Reuse the live edge for each desired (dst, rel), insert the missing ones, end the live
    import edges no longer desired, then insert the occurrences (step 5)."""
    conn.execute("DELETE FROM link_occurrences WHERE field IS NOT NULL AND edge_id IN"
                 " (SELECT id FROM edges WHERE src = ?)", (nid,))  # fmt: skip
    for rel in {fl.rel for fl in note.parsed.fm_links}:
        conn.execute("INSERT OR IGNORE INTO rels(rel, inverse, src_kinds, dst_kinds, description)"
                     " VALUES (?, NULL, '*', '*', 'frontmatter link (import)')", (rel,))  # fmt: skip
    edge_of: dict[tuple[EdgeKey, str], int] = {}
    for (key, rel), name in desired.items():
        dst = _target_node(conn, key, name, note, now)
        row = conn.execute("SELECT id FROM edges WHERE src = ? AND dst = ? AND rel = ?"
                           " AND removed_at IS NULL", (nid, dst, rel)).fetchone()  # fmt: skip
        edge_of[(key, rel)] = row[0] if row else conn.execute(
            "INSERT INTO edges(src, dst, rel, origin, recorded_at) VALUES (?, ?, ?, 'import', ?)",
            (nid, dst, rel, now)).lastrowid  # fmt: skip
    keep = set(edge_of.values())
    for (eid,) in conn.execute("SELECT id FROM edges WHERE src = ? AND origin = 'import'"
                               " AND removed_at IS NULL", (nid,)).fetchall():  # fmt: skip
        if eid not in keep:
            conn.execute("UPDATE edges SET removed_at = ? WHERE id = ?", (now, eid))

    def edge_id(link: Link, rel: str) -> int | None:
        key = resolver.resolve(link.target) if link.target else None
        return edge_of.get((key, rel)) if key else None

    for fl in note.parsed.fm_links:
        if (eid := edge_id(fl.link, fl.rel)) is not None:
            conn.execute("INSERT INTO link_occurrences(edge_id, field, pos, anchor, alias)"
                         " VALUES (?, ?, ?, ?, ?)",
                         (eid, fl.field, fl.link.pos, fl.link.anchor, fl.link.alias))  # fmt: skip
    for c in note.parsed.chunks:
        for link in c.links:
            if (eid := edge_id(link, "embeds" if link.embed else "links_to")) is not None:
                conn.execute("INSERT INTO link_occurrences(edge_id, chunk_id, pos, anchor, alias)"
                             " VALUES (?, ?, ?, ?, ?)",
                             (eid, chunk_ids[c.key], link.pos, link.anchor, link.alias))  # fmt: skip


def _target_node(conn: sqlite3.Connection, key: EdgeKey, name: str, note: _Note, now: str) -> int:
    """The node an edge points at, created when it does not exist yet: a tag (with its `part_of`
    chain), a topic stub for an unresolved name, a file stub, or a shell for a note that a later
    file of this run will fill in."""
    kind, ref = key
    if kind == "path":
        row = conn.execute("SELECT id FROM nodes WHERE origin = 'import' AND source_ref = ?",
                           (ref,)).fetchone()  # fmt: skip
        if row:
            return row[0]
        is_note = ref.endswith(".md")
        label = ref.rsplit("/", 1)[-1]
        return _insert(conn, "note", "note" if is_note else "file",
                       DATE_PREFIX.sub("", label.removesuffix(".md")) if is_note else label,
                       ref, now)  # fmt: skip
    if kind == "topic":
        return _find_or_insert(conn, "entity", "topic", name, now)
    return _tag_node(conn, name, now)


def _find_or_insert(conn: sqlite3.Connection, kind: str, type_: str | None, name: str,
                    now: str) -> int:  # fmt: skip
    """The active unowned node of this kind and name (any origin, case-insensitive), or a new one."""
    row = conn.execute(
        "SELECT id FROM nodes WHERE kind = ? AND type IS ? AND name = ? COLLATE NOCASE"
        " AND status = 'active' AND source_ref IS NULL ORDER BY id",
        (kind, type_, name),
    ).fetchone()
    return row[0] if row else _insert(conn, kind, type_, name, None, now)


def _tag_node(conn: sqlite3.Connection, name: str, now: str) -> int:
    tid = _find_or_insert(conn, "tag", None, name, now)
    if "/" in name:
        parent = _tag_node(conn, name.rsplit("/", 1)[0], now)
        if not conn.execute("SELECT 1 FROM edges WHERE src = ? AND dst = ? AND rel = 'part_of'"
                            " AND removed_at IS NULL", (tid, parent)).fetchone():  # fmt: skip
            conn.execute("INSERT INTO edges(src, dst, rel, origin, recorded_at)"
                         " VALUES (?, ?, 'part_of', 'import', ?)", (tid, parent, now))  # fmt: skip
    return tid


def _insert(conn: sqlite3.Connection, kind: str, type_: str | None, name: str, ref: str | None,
            now: str) -> int:  # fmt: skip
    return conn.execute(
        "INSERT INTO nodes(kind, type, name, valid_from, source_ref, recorded_at, origin)"
        " VALUES (?, ?, ?, ?, ?, ?, 'import')",
        (kind, type_, name, now, ref, now),
    ).lastrowid  # type: ignore[return-value]
