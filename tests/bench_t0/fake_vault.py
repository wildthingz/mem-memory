"""A stand-in importer for the T0 tests of the P4 harness (not the product importer).

It implements architecture-v2 §2.4 in the small, on the real schema and the real `Store`, so the
harness can be tested end to end (B-I1..B-I5, B-R3, the PG tool) before `mem.importer` exists:
path-keyed nodes, `chunk_key = heading path#n/p0`, the per-file lifecycle (unchanged / changed /
new / renamed / deleted / restored), aliases, props, tags, wikilinks with occurrences. Anything the
product importer does beyond this (entity routing, size splitting, callouts) is not modelled.
The tests that use it prove the HARNESS: every suite check must fail when this module is broken on
purpose (tests/bench_t0/test_p4_suites.py).

`run_import(vault, stores, *, routes, render_dir)` is the contract `bench.adapters.p4_ours.resolve_importer`
expects (docs/decisions/0010 of the mini-vault task): the vault goes to personal.db / work.db by
folder, `routes` (longest prefix wins) overrides the default.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from mem.write import content_hash

SKIP_DIRS = {".obsidian", ".smart-env", ".stfolder", "Templates", "node_modules", "export"}
SKIP_FILES = {"CLAUDE.md", "obsidian-instructions.md"}
WIKI = re.compile(r"(!?)\[\[([^\]\n|#^]+)(?:#(\^?[^\]\n|]*))?(?:\|([^\]\n]*))?\]\]")
HEAD = re.compile(r"^(#{1,3})\s+(.*?)\s*$")
TAG = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]*)")
COMMENT = re.compile(r"%%.*?%%", re.DOTALL)
PERSON_TAGS = {"person": ("entity", "person"), "project": ("entity", "project"),
               "team": ("entity", "org")}  # fmt: skip


def normalise(text: str) -> str:
    return COMMENT.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            fm = yaml.safe_load(text[4:end]) or {}
            return (fm if isinstance(fm, dict) else {}), text[end + 5 :]
    return {}, text


def _singular(key: str) -> str:
    return key[:-1] if key.endswith("s") and len(key) > 4 else key


def sections(body: str) -> list[tuple[str, str]]:
    """(heading path, text) per H1-H3 section; text before the first heading has path ''."""
    out: list[tuple[str, list[str]]] = [("", [])]
    stack: list[tuple[int, str]] = []
    fence = False
    for line in body.split("\n"):
        if line.startswith("```"):
            fence = not fence
        m = None if fence else HEAD.match(line)
        if m:
            level = len(m[1])
            stack = [s for s in stack if s[0] < level] + [(level, m[2])]
            out.append((" > ".join(t for _, t in stack), [line]))
        else:
            out[-1][1].append(line)
    return [(p, "\n".join(ls).strip("\n")) for p, ls in out if "\n".join(ls).strip()]


def keyed_chunks(body: str) -> list[tuple[str, str, str]]:
    """(chunk_key, heading_path, text); n = occurrence of the heading path in the note."""
    seen: dict[str, int] = {}
    out = []
    for path, text in sections(body):
        seen[path] = seen.get(path, 0) + 1
        out.append((f"{path}#{seen[path]}/p0", path, text))
    return out


def scan(root: Path) -> dict[str, str]:
    files = {}
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root)
        if set(rel.parts[:-1]) & SKIP_DIRS or rel.name in SKIP_FILES:
            continue
        files[rel.as_posix()] = p.read_text(encoding="utf-8")
    return files


def other_files(root: Path) -> list[str]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and p.suffix != ".md" and not (set(rel.parts[:-1]) & SKIP_DIRS):
            out.append(rel.as_posix())
    return out


def resolve(target: str, paths: list[str]) -> str | None:
    """The Obsidian way: exact path, then unique basename, then shortest path."""
    t = target.removesuffix(".md")
    exact = [p for p in paths if p.removesuffix(".md") == t]
    if exact:
        return exact[0]
    base = sorted((p for p in paths if p.removesuffix(".md").rsplit("/", 1)[-1] == t.rsplit("/", 1)[-1]),
                  key=lambda p: (len(p), p))  # fmt: skip
    return base[0] if base else None


def _node_kind(fm: dict[str, Any]) -> tuple[str, str]:
    tags = fm.get("tags") or []
    tags = [tags] if isinstance(tags, str) else tags
    for t in tags:
        if t in PERSON_TAGS:
            return PERSON_TAGS[t]
    return "note", "note"


def route_of(path: str, routes: dict[str, str]) -> str:
    best = max((k for k in routes if path == k or path.startswith(k.rstrip("/") + "/")), key=len,
               default=None)  # fmt: skip
    if best is not None:
        return routes[best]
    return "work" if path.startswith("work/") else "personal"


def run_import(vault: Path, stores: dict[str, Any], *, routes: dict[str, str] | None = None,
               render_dir: str = "export") -> dict[str, int]:  # fmt: skip
    total: dict[str, int] = {}
    for name, store in stores.items():
        rep = import_vault(store, vault, include=lambda p, n=name: route_of(p, routes or {}) == n)
        for k, v in rep.items():
            total[k] = total.get(k, 0) + v
    return total


def import_vault(store: Any, root: Path, *, dry_run: bool = False,
                 include: Any = None) -> dict[str, int]:  # fmt: skip
    conn = store.conn
    files = {p: normalise(t) for p, t in scan(Path(root)).items() if include is None or include(p)}
    hashes = {p: content_hash(t) for p, t in files.items()}
    known = {r[0]: r[1:] for r in conn.execute(
        "SELECT n.source_ref, n.id, n.content_hash, n.status FROM nodes n"
        " WHERE n.origin = 'import' AND n.source_ref IS NOT NULL AND n.kind IN ('note', 'entity')"
        " AND coalesce(n.type, '') <> 'file'")}  # fmt: skip
    rep = dict.fromkeys(("added", "changed", "renamed", "deleted", "restored", "unchanged"), 0)
    vanished = {p: v for p, v in known.items() if p not in files and v[2] != "deleted"}
    plan: list[tuple[str, str, int | None]] = []  # (path, action, node id)
    for p, h in hashes.items():
        if p in known:
            nid, old_h, status = known[p]
            if status == "active" and old_h == h:
                rep["unchanged"] += 1
            else:
                plan.append((p, "restored" if status == "deleted" else "changed", nid))
            continue
        twin = next((q for q, v in vanished.items() if v[1] == h), None)
        deleted_twin = next((q for q, v in known.items() if v[2] == "deleted" and v[1] == h
                             and q not in files), None)  # fmt: skip
        src = twin or deleted_twin
        if src:
            vanished.pop(src, None)
            plan.append((p, "renamed", known[src][0]))
        else:
            plan.append((p, "added", None))
    if dry_run:
        for _, a, _ in plan:
            rep[a] += 1
        rep["deleted"] = len(vanished)
        return rep
    now = store.now_ts()
    paths = sorted(files) + other_files(Path(root))
    written = []
    for p, action, nid in plan:
        written.append((p, _write_note(store, p, files[p], hashes[p], nid, paths, now)))
        rep[action] += 1
    for p, nid in written:
        _link_note(store, p, files[p], nid, paths, now)
    for p, (nid, _, _) in vanished.items():
        conn.execute("UPDATE nodes SET status='deleted', status_at=? WHERE id=?", (now, nid))
        rep["deleted"] += 1
    return rep


def _write_note(store: Any, path: str, text: str, chash: str, nid: int | None, paths: list[str],
                now: str) -> int:  # fmt: skip
    """Pass 1 of a note: node, aliases, props, chunks. Edges come in pass 2 (`_link_note`), when every
    file of the run has its node, so a forward link resolves to the real note, not a stub."""
    conn = store.conn
    fm, body = split_frontmatter(text)
    stem = Path(path).stem
    title = next((m[2] for ln in body.split("\n") if (m := HEAD.match(ln)) and m[1] == "#"),
                 re.sub(r"^\d{4}-\d{2}-\d{2}[_ -]", "", stem))  # fmt: skip
    kind, typ = _node_kind(fm)
    date = str(fm.get("created") or "")[:10] or now[:10]
    valid_from = f"{date}T00:00:00Z"
    conn.execute("BEGIN IMMEDIATE")
    try:
        if nid is None:
            nid = conn.execute(
                "INSERT INTO nodes(kind, type, name, date, valid_from, recorded_at, origin,"
                " source_ref, content_hash, host) VALUES (?,?,?,?,?,?,'import',?,?,?)",
                (kind, typ, title, date, valid_from, now, path, chash, store.host),
            ).lastrowid
        else:
            conn.execute(
                "UPDATE nodes SET kind=?, type=?, name=?, date=?, valid_from=?, content_hash=?,"
                " source_ref=?, status='active', status_at=NULL WHERE id=?",
                (kind, typ, title, date, valid_from, chash, path, nid),
            )
        assert nid is not None
        _aliases_props(conn, nid, fm, title, stem)
        _chunks(store, nid, title, body)
        conn.execute("COMMIT")
        return nid
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _link_note(store: Any, path: str, text: str, nid: int, paths: list[str], now: str) -> None:
    conn = store.conn
    fm, body = split_frontmatter(text)
    conn.execute("BEGIN IMMEDIATE")
    try:
        kept = {
            k: i
            for i, k in conn.execute("SELECT id, chunk_key FROM chunks WHERE node_id=?", (nid,))
        }
        _edges(conn, nid, path, fm, body, kept, paths, now)
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _norm_alias(a: str) -> str:
    return " ".join(str(a).casefold().split())


def _aliases_props(conn: Any, nid: int, fm: dict[str, Any], title: str, stem: str) -> None:
    conn.execute("DELETE FROM aliases WHERE node_id=? AND origin='import'", (nid,))
    conn.execute("DELETE FROM props WHERE node_id=? AND origin='import'", (nid,))
    names = [title, stem, *(fm.get("aliases") or [])]
    for a in dict.fromkeys(str(n) for n in names):
        conn.execute("INSERT OR IGNORE INTO aliases(alias, alias_norm, node_id, origin)"
                     " VALUES (?,?,?,'import')", (a, _norm_alias(a), nid))  # fmt: skip
    for k, v in fm.items():
        if k in ("aliases", "tags", "created") or isinstance(v, (list, dict)) or "[[" in str(v):
            continue
        conn.execute("INSERT OR IGNORE INTO props(node_id, key, value, origin)"
                     " VALUES (?,?,?,'import')", (nid, k, str(v)))  # fmt: skip


def _chunks(store: Any, nid: int, title: str, body: str) -> dict[str, int]:
    conn = store.conn
    have = {k: (i, h) for i, k, h in conn.execute(
        "SELECT id, chunk_key, content_hash FROM chunks WHERE node_id=?", (nid,))}  # fmt: skip
    want = keyed_chunks(body)
    keys = {k for k, _, _ in want}
    for k, (cid, _) in have.items():
        if k not in keys:
            _drop_chunk(store, cid)
    kept: dict[str, int] = {}
    add = []
    for i, (key, hp, text) in enumerate(want):
        h = content_hash(text)
        ctx = f"{title} > {hp}" if hp else title
        if key in have and have[key][1] == h:
            kept[key] = have[key][0]
            conn.execute("UPDATE chunks SET ord=? WHERE id=?", (i, have[key][0]))
            continue
        if key in have:
            _drop_chunk(store, have[key][0])
        cid = conn.execute(
            "INSERT INTO chunks(node_id, chunk_key, ord, key_kind, heading_path, ctx, text, tokens,"
            " content_hash) VALUES (?,?,?,'section',?,?,?,?,?)",
            (nid, key, i, hp, ctx, text, len(text.split()), h),
        ).lastrowid
        assert cid is not None
        kept[key] = cid
        add.append((cid, ctx, text))
    store.add_vectors(add)
    return kept


def _drop_chunk(store: Any, cid: int) -> None:
    store.conn.execute(f"DELETE FROM {store.vec_table} WHERE chunk_id=?", (cid,))
    store.conn.execute("DELETE FROM chunks WHERE id=?", (cid,))


def _target_node(conn: Any, target: str, paths: list[str], now: str, store_host: str | None) -> int:
    hit = resolve(target, paths)
    if hit:
        row = conn.execute(
            "SELECT id FROM nodes WHERE origin='import' AND source_ref=?", (hit,)
        ).fetchone()
        if row:
            return row[0]
        if not hit.endswith(".md"):  # an unindexed file (.png, .canvas): a file stub with its path
            return conn.execute(
                "INSERT INTO nodes(kind, type, name, valid_from, recorded_at, origin, source_ref)"
                " VALUES ('note','file',?,?,?,'import',?)",
                (hit.rsplit("/", 1)[-1], now, now, hit),
            ).lastrowid
    row = conn.execute(
        "SELECT id FROM nodes WHERE kind='entity' AND type='topic'"
        " AND name=? COLLATE NOCASE AND source_ref IS NULL",
        (target,),
    ).fetchone()
    if row:
        return row[0]
    return conn.execute(
        "INSERT INTO nodes(kind, type, name, valid_from, recorded_at, origin, host)"
        " VALUES ('entity','topic',?,?,?,'import',?)",
        (target, now, now, store_host),
    ).lastrowid


def _edges(conn: Any, nid: int, path: str, fm: dict[str, Any], body: str, kept: dict[str, int],
           paths: list[str], now: str) -> None:  # fmt: skip
    host = None
    conn.execute(
        "DELETE FROM link_occurrences WHERE chunk_id IN"
        " (SELECT id FROM chunks WHERE node_id=?) OR edge_id IN"
        " (SELECT id FROM edges WHERE src=? AND origin='import')",
        (nid, nid),
    )
    desired: dict[
        tuple[int, str], list[tuple[int | None, str | None, int, str | None, str | None]]
    ] = {}

    def want(target: str, rel: str, occ: tuple) -> None:
        dst = _target_node(conn, target.strip(), paths, now, host)
        if dst != nid:
            desired.setdefault((dst, rel), []).append(occ)

    for key, _, text in keyed_chunks(body):
        for m in WIKI.finditer(text):
            want(m[2], "embeds" if m[1] else "links_to", (kept[key], None, m.start(), m[3], m[4]))
    for k, v in fm.items():
        vals = v if isinstance(v, list) else [v]
        for i, item in enumerate(vals):
            m = WIKI.search(str(item))
            if m and k not in ("aliases", "tags"):
                conn.execute("INSERT OR IGNORE INTO rels(rel, inverse, src_kinds, dst_kinds,"
                             " description) VALUES (?, NULL, '*', '*', 'frontmatter link')",
                             (_singular(k),))  # fmt: skip
                want(m[2], _singular(k), (None, k, i, m[3], m[4]))
    tags = fm.get("tags") or []
    tags = [tags] if isinstance(tags, str) else list(tags)
    tags += [
        m[1]
        for _, _, t in keyed_chunks(body)
        for m in TAG.finditer(_strip_code(t))
        if not re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8}", m[1])
    ]
    for tag in dict.fromkeys(tags):
        tid = _tag_node(conn, tag, now)
        desired.setdefault((tid, "tagged"), [])
    live = {(d, r): e for e, d, r in conn.execute(
        "SELECT id, dst, rel FROM edges WHERE src=? AND origin='import' AND removed_at IS NULL",
        (nid,))}  # fmt: skip
    for key_, occs in desired.items():
        eid = (
            live.get(key_)
            or conn.execute(
                "INSERT INTO edges(src, dst, rel, origin, recorded_at) VALUES (?,?,?,'import',?)",
                (nid, key_[0], key_[1], now),
            ).lastrowid
        )
        for chunk_id, field, pos, anchor, alias in occs:
            conn.execute("INSERT OR IGNORE INTO link_occurrences(edge_id, chunk_id, field, pos,"
                         " anchor, alias) VALUES (?,?,?,?,?,?)",
                         (eid, chunk_id, field, pos, anchor, alias))  # fmt: skip
    for key_, eid in live.items():
        if key_ not in desired:
            conn.execute("UPDATE edges SET removed_at=? WHERE id=?", (now, eid))


def _strip_code(text: str) -> str:
    return re.sub(r"`[^`]*`|```.*?```", "", text, flags=re.DOTALL)


def _tag_node(conn: Any, tag: str, now: str) -> int:
    row = conn.execute(
        "SELECT id FROM nodes WHERE kind='tag' AND name=? COLLATE NOCASE AND source_ref IS NULL",
        (tag,),
    ).fetchone()
    if row:
        return row[0]
    return conn.execute(
        "INSERT INTO nodes(kind, type, name, valid_from, recorded_at, origin)"
        " VALUES ('tag', NULL, ?, ?, ?, 'import')",
        (tag, now, now),
    ).lastrowid


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
