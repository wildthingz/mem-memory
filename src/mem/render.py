"""`mem render`: store → Markdown/HTML, one-way (architecture-v2 §2, DR-5, A14).

The output is a pure function of the store: pages are ordered by id, nothing carries the render time,
and a file is only written when its bytes change, so an unchanged store leaves no diff and one
changed memory touches only its own pages. Only `<out>/<scope>/` is written (and pruned)."""

from __future__ import annotations

import json
import posixpath
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup, escape

from mem.common import MemError

_PLAIN = re.compile(r"^[A-Za-z0-9_./:@+-][A-Za-z0-9_./:@ +-]*$")


@dataclass
class Item:
    label: str
    target: str | None = None  # scope-relative page path without extension
    extra: str | None = None


@dataclass
class Page:
    path: str  # scope-relative, without extension
    title: str
    front: list[tuple[str, str]]
    text: str = ""
    raw_body: str | None = None  # notes: the chunk texts, verbatim
    sections: list[tuple[str, list[Item]]] = field(default_factory=list)


@dataclass
class RenderResult:
    scope: str
    written: int = 0
    unchanged: int = 0
    removed: int = 0


def _scalar(value: object) -> str:
    text = "" if value is None else str(value)
    if _PLAIN.match(text) and _yaml_str(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _yaml_str(text: str) -> bool:
    """True when `text` as a plain YAML scalar loads back as exactly that string."""
    try:
        return yaml.safe_load(f"k: {text}\n") == {"k": text}
    except yaml.YAMLError:
        return False


def _env(ext: str) -> Environment:
    def href(page_path: str, target: str) -> str:
        rel = posixpath.relpath(target + ext, posixpath.dirname(page_path) or ".")
        return quote(rel)

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=ext == ".html",
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["scalar"] = _scalar
    env.filters["link"] = lambda item: ""  # replaced per page: the href depends on the page
    env.globals["_href"] = href
    return env


def _link_filter(env: Environment, page_path: str, html: bool):
    def link(item: Item):
        extra = f" — {item.extra}" if item.extra else ""
        if item.target is None:
            label = f"{item.label}{extra}"
            return escape(label) if html else label
        href = env.globals["_href"](page_path, item.target)
        if html:
            return Markup(f'<a href="{escape(href)}">{escape(item.label)}</a>{escape(extra)}')
        return f"[{item.label}]({href}){extra}"

    return link


def _safe_note_path(ref: str) -> str:
    parts = [p for p in ref.split("/") if p not in ("", ".", "..")]
    name = "/".join(parts) or "note"
    return name.removesuffix(".md")


def _collect(conn: sqlite3.Connection) -> tuple[str, list[Page], dict[str, list[Item]]]:
    scope = conn.execute("SELECT value FROM meta WHERE key='scope'").fetchone()
    if scope is None:
        raise MemError("not a mem store: meta has no scope")
    # Quarantined memories (P7, `mem.guard`) are never rendered: the mirror is committed and read
    # as trusted. A store the P7 migration has not reached yet has none.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(nodes)")}
    held = " AND quarantine IS NULL" if "quarantine" in cols else ""
    nodes = {
        r[0]: r
        for r in conn.execute(
            "SELECT id, kind, type, name, summary, body, date, valid_from, valid_to, recorded_at,"
            f" origin, source_ref, status, pinned FROM nodes WHERE kind <> 'tag'{held} ORDER BY id"
        )
    }

    def is_note(n: tuple) -> bool:  # a vault note or import stub; a memory's `source` is a URL etc.
        return bool(n[11]) and n[1] != "memory"

    note_paths: dict[int, str] = {}
    taken: set[str] = set()
    for n in nodes.values():  # by id: the lowest id keeps a contested path, later ones get ~n<id>
        if is_note(n) and n[12] != "deleted":
            path = "notes/" + _safe_note_path(n[11])
            if path.lower() in taken:
                path = f"{path}~n{n[0]}"
            taken.add(path.lower())
            note_paths[n[0]] = path

    def page_path(node_id: int) -> str | None:
        n = nodes.get(node_id)
        if n is None or n[12] == "deleted" and is_note(n):
            return None
        if is_note(n):
            return note_paths[node_id]
        return {"memory": f"memories/m{n[0]}", "entity": f"entities/e{n[0]}"}.get(n[1])

    edges = conn.execute(
        "SELECT src, dst, rel FROM edges WHERE removed_at IS NULL"
        " AND rel IN ('about','supersedes') ORDER BY src, dst"
    ).fetchall()
    about: dict[int, list[int]] = {}
    about_rev: dict[int, list[int]] = {}
    supersedes: dict[int, list[int]] = {}
    superseded_by: dict[int, list[int]] = {}
    for src, dst, rel in edges:
        if src not in nodes or dst not in nodes:
            continue
        if rel == "about":
            about.setdefault(src, []).append(dst)
            about_rev.setdefault(dst, []).append(src)
        else:
            supersedes.setdefault(src, []).append(dst)
            superseded_by.setdefault(dst, []).append(src)

    def refs(ids: list[int], prefix: str | None = None) -> list[Item]:
        out = []
        for i in sorted(ids):
            if i in nodes:
                tag = f"{nodes[i][1][0]}{i}" if nodes[i][1] != "note" else f"n{i}"
                out.append(Item(nodes[i][3] if prefix is None else tag, page_path(i), None))
        return out

    chunks: dict[int, list[str]] = {}
    for node_id, text in conn.execute("SELECT node_id, text FROM chunks ORDER BY node_id, ord, id"):
        chunks.setdefault(node_id, []).append(text)
    aliases: dict[int, list[str]] = {}
    for node_id, alias in conn.execute(
        "SELECT node_id, alias FROM aliases ORDER BY node_id, alias"
    ):
        aliases.setdefault(node_id, []).append(alias)

    pages: list[Page] = []
    listing: dict[str, list[Item]] = {"Memories": [], "Entities": [], "Notes": []}
    for n in nodes.values():
        nid, kind, type_, name, summary, body, date, vfrom, vto, rec, origin, ref, status, pin = n
        path = page_path(nid)
        if path is None:
            continue
        if is_note(n):
            front = [("id", f"n{nid}"), ("kind", kind), ("type", type_), ("title", name),
                     ("source_ref", ref), ("date", date), ("status", status)]  # fmt: skip
            pages.append(Page(path, name, [(k, v) for k, v in front if v],
                              raw_body="".join(chunks.get(nid, []))))  # fmt: skip
            listing["Notes"].append(Item(name, path))
        elif kind == "memory":
            front = [("id", f"m{nid}"), ("type", type_), ("status", status),
                     ("valid_from", vfrom), ("valid_to", vto), ("recorded_at", rec),
                     ("origin", origin), ("source", ref), ("pinned", "true" if pin else None),
                     ("supersedes", ", ".join(f"m{i}" for i in sorted(supersedes.get(nid, [])))),
                     ("superseded_by", ", ".join(f"m{i}" for i in sorted(superseded_by.get(nid, []))))]  # fmt: skip
            page = Page(path, name, [(k, v) for k, v in front if v], text=body or "")
            page.sections = [("About", i)] if (i := refs(about.get(nid, []))) else []
            pages.append(page)
            listing["Memories"].append(Item(f"m{nid}", path, name))
        elif kind == "entity":
            front = [("id", f"e{nid}"), ("type", type_), ("status", status),
                     ("aliases", "; ".join(aliases.get(nid, [])))]  # fmt: skip
            page = Page(path, name, [(k, v) for k, v in front if v], text=summary or "")
            mems = [Item(f"m{m}", page_path(m), nodes[m][3]) for m in sorted(about_rev.get(nid, []))
                    if m in nodes and nodes[m][1] == "memory"]  # fmt: skip
            page.sections = [("Memories", mems)] if mems else []
            pages.append(page)
            listing["Entities"].append(Item(f"e{nid}", path, name))
    pages = [p for p in pages if p.path]
    pages.sort(key=lambda p: p.path)
    return scope[0], pages, listing


def _render_pages(
    scope: str, pages: list[Page], listing: dict[str, list[Item]], html: bool
) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for ext in (".md", ".html") if html else (".md",):
        env = _env(ext)
        page_t = env.get_template(f"page{ext}.j2")
        index_t = env.get_template(f"index{ext}.j2")
        for page in pages:
            env.filters["link"] = _link_filter(env, page.path, ext == ".html")
            text = page_t.render(page=page, front=page.front)
            out[page.path + ext] = text.encode("utf-8")
        env.filters["link"] = _link_filter(env, "index", ext == ".html")
        sections = [(h, items) for h, items in listing.items() if items]
        out["index" + ext] = index_t.render(scope=scope, sections=sections).encode("utf-8")
    return out


def _sync(root: Path, files: dict[str, bytes]) -> tuple[int, int, int]:
    """Write changed files under `root`, delete the ones no longer rendered."""
    root_resolved = root.resolve()
    written = unchanged = removed = 0
    for rel, data in files.items():
        target = root / rel
        if root_resolved not in target.resolve().parents:
            raise MemError(f"refusing to write outside the render dir: {rel}")
        if target.is_file() and target.read_bytes() == data:
            unchanged += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
        written += 1
    if root.is_dir():
        for path in sorted(root.rglob("*"), reverse=True):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in files:
                path.unlink()
                removed += 1
            elif path.is_dir() and not any(path.iterdir()):
                path.rmdir()
    return written, unchanged, removed


def render_store(conn: sqlite3.Connection, out: str | Path, *, html: bool = False) -> RenderResult:
    """Render one store into `<out>/<scope>/`."""
    scope, pages, listing = _collect(conn)
    files = _render_pages(scope, pages, listing, html)
    written, unchanged, removed = _sync(Path(out) / scope, files)
    return RenderResult(scope, written, unchanged, removed)


def render_path(path: str | Path, out: str | Path, *, html: bool = False) -> RenderResult:
    """Like `render_store`, reading the store file read-only."""
    if not Path(path).is_file():
        raise MemError(f"no store at {path}")
    conn = sqlite3.connect(f"file:{quote(str(Path(path).resolve()))}?mode=ro", uri=True)
    try:
        return render_store(conn, out, html=html)
    finally:
        conn.close()


def commit_mirror(out: str | Path, scopes: list[str]) -> bool:
    """Commit only `<out>/<scope>/` for the given scopes (initialising the repo if needed).

    Foreign files in `out`, even ones already staged, are never part of the commit.
    False if nothing changed."""
    out = Path(out)

    def git(*args: str) -> str:
        res = subprocess.run(
            ["git", "-C", str(out), "-c", "user.name=mem", "-c", "user.email=mem@localhost",
             "-c", "commit.gpgsign=false", *args],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if res.returncode:
            raise MemError(f"git {args[0]} failed: {res.stderr.strip()}")
        return res.stdout

    if not (out / ".git").exists():
        git("init", "-q")
    git("add", "-A", "--", *scopes)
    if not git("status", "--porcelain", "--", *scopes).strip():
        return False
    git("commit", "-q", "-m", "mem render", "--only", "--", *scopes)
    return True
