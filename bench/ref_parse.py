"""Independent reference parser for B-I3 / B-I4 (benchmarks-v2 B-I): shares no code with the importer.

It reads the vault files directly and reports what any correct import must contain: the notes,
the frontmatter notes, the link occurrences (body + frontmatter), the unique logical edges per rel
and the tag edges. Definitions (decision record 0010; the importer's mapping table is
architecture-v2 §2.4):

- Notes: `*.md` files outside the skip list (`.obsidian`, `.smart-env`, `.stfolder`, `Templates`,
  `node_modules`, the render folder; `CLAUDE.md`, `obsidian-instructions.md`).
- Normalised body: LF endings, frontmatter removed, `%%comments%%` stripped (what is indexed).
- Link occurrence: one `[[X]]`, `[[X|a]]`, `[[X#H]]`, `[[X#^b]]` (rel links_to) or `![[X]]` (rel
  embeds) in the body outside code spans and fenced blocks, plus one per wikilink value of a
  frontmatter key (rel = the key, singular; one per list item). Empty targets and links that
  resolve to the note itself are not occurrences (a self edge cannot be stored).
- Resolution, the Obsidian way: exact path (with or without `.md`), then unique basename, then the
  shortest path. A target with a non-`.md` extension (`.png`, `.canvas`, `.base`) is resolved the same
  way among the vault's other files: it is a file stub carrying its path. Anything else unresolved ->
  `stub:<target casefolded>` (the importer's topic stub).
- Logical edge: distinct (note, destination, rel). Tag edge: (note, tag) from `tags:` and from
  inline `#tag` outside code, not hex colours, not pure numbers.
- Frontmatter-derived notes: notes whose frontmatter has a prop or link (any key other than
  `tags`, `created`, `aliases`) or an alias that is not the title or the file stem (`created` is the
  node date, §2.4): the store keeps no marker for a note whose frontmatter is only `tags`,
  those tags are compared as tag edges. `frontmatter_notes` (any non-empty frontmatter) is
  reported next to it.
Uses PyYAML for the frontmatter (python-frontmatter is a thin wrapper over the same call and is
not a project dependency).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKIP_DIRS = frozenset({".obsidian", ".smart-env", ".stfolder", "Templates", "node_modules"})
SKIP_FILES = frozenset({"CLAUDE.md", "obsidian-instructions.md"})
RENDER_DIR = "export"

_WIKI = re.compile(r"(!?)\[\[([^\]\n|#^]*)(?:#[^\]\n|]*)?(?:\|[^\]\n]*)?\]\]")
_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_COMMENT = re.compile(r"%%.*?%%", re.DOTALL)
_HEX = re.compile(r"^(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_TAG = re.compile(r"(?<![\w/&#])#([\w/-]+)")


def normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    text = normalise_newlines(text)
    if text.startswith("---\n"):
        end = text.find("\n---", 3)
        if end != -1 and text[end + 4 : end + 5] in ("", "\n"):
            try:
                fm = yaml.safe_load(text[4 : end + 1]) or {}
            except yaml.YAMLError:
                return {}, text
            return (fm if isinstance(fm, dict) else {}), text[end + 5 :]
    return {}, text


def normalised_body(text: str) -> str:
    return _COMMENT.sub("", split_frontmatter(text)[1])


def _code_free(body: str) -> str:
    return _INLINE_CODE.sub("", _FENCE.sub("", body))


def singular(key: str) -> str:
    if key.endswith("ies"):
        return key[:-3] + "y"
    return key[:-1] if key.endswith("s") and not key.endswith("ss") else key


def _fold(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [] if v is None else [v]


@dataclass
class RefNote:
    path: str
    frontmatter: dict[str, Any]
    body: str
    occurrences: list[tuple[str, str, str | None]] = field(
        default_factory=list
    )  # (target, rel, field)
    tags: set[str] = field(default_factory=set)

    @property
    def has_frontmatter(self) -> bool:
        return bool(self.frontmatter)

    @property
    def has_frontmatter_rows(self) -> bool:
        """Notes that give the store a props / extra-alias / frontmatter-link row. `created` is the
        node date (§2.4), `tags` are edges; an alias equal to the title or the file stem is implicit."""
        fm = self.frontmatter
        if any(k not in ("tags", "created", "aliases") and _as_list(v) for k, v in fm.items()):
            return True
        stem = Path(self.path).stem
        implicit = {
            _fold(self._title()),
            _fold(stem),
            _fold(re.sub(r"^\d{4}-\d{2}-\d{2}[_ -]", "", stem)),
        }
        return any(
            _fold(str(a)) not in implicit for a in _as_list(fm.get("aliases")) if str(a).strip()
        )

    def _title(self) -> str:
        m = re.search(r"^# +(.+?)\s*#*\s*$", _code_free(self.body), re.MULTILINE)
        return m[1] if m else re.sub(r"^\d{4}-\d{2}-\d{2}_", "", Path(self.path).stem)


def parse_note(path: str, text: str) -> RefNote:
    fm, _ = split_frontmatter(text)
    body = normalised_body(text)
    note = RefNote(path, fm, body)
    clean = _code_free(body)
    for m in _WIKI.finditer(clean):
        note.occurrences.append((m[2].strip(), "embeds" if m[1] else "links_to", None))
    for key, value in fm.items():
        if key in ("aliases", "tags"):
            continue
        for item in _as_list(value):
            for m in _WIKI.finditer(str(item)):
                note.occurrences.append((m[2].strip(), singular(str(key)), str(key)))
    raw_tags = fm.get("tags")
    if isinstance(raw_tags, str):
        raw_tags = re.split(r"[,\s]+", raw_tags)
    note.tags |= {str(t).lstrip("#") for t in _as_list(raw_tags) if str(t).strip()}
    for m in _TAG.finditer(clean):
        t = m[1]
        if _HEX.match(t) or t.replace("/", "").isdigit():
            continue
        note.tags.add(t)
    return note


def scan(root: Path, render_dir: str = RENDER_DIR) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*.md")):
        rel = p.relative_to(root)
        if (set(rel.parts[:-1]) & (SKIP_DIRS | {render_dir})) or rel.name in SKIP_FILES:
            continue
        out[rel.as_posix()] = p.read_text(encoding="utf-8")
    return out


def all_files(root: Path, render_dir: str = RENDER_DIR) -> list[str]:
    """Every file outside the skip list: the link targets that are not notes."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts[:-1]) & (SKIP_DIRS | {render_dir})):
            out.append(rel.as_posix())
    return out


def resolve(target: str, paths: list[str]) -> str | None:
    t = target.removesuffix(".md")
    for p in paths:
        if p.removesuffix(".md") == t:
            return p
    base = t.rsplit("/", 1)[-1]
    cands = sorted((p for p in paths if p.removesuffix(".md").rsplit("/", 1)[-1] == base),
                   key=lambda p: (len(p), p))  # fmt: skip
    return cands[0] if cands else None


def resolve_file(target: str, files: list[str]) -> str | None:
    """A non-note target: exact path, then unique basename, then the shortest path."""
    if target in files:
        return target
    base = target.rsplit("/", 1)[-1]
    cands = sorted((f for f in files if f.rsplit("/", 1)[-1] == base), key=lambda f: (len(f), f))
    return cands[0] if cands else None


@dataclass
class RefVault:
    notes: dict[str, RefNote]
    occurrences: list[tuple[str, str, str]]  # (src, dst key, rel), one per occurrence
    edges: set[tuple[str, str, str]]  # distinct (src, dst key, rel)
    tag_edges: set[tuple[str, str]]

    def counts(self) -> dict[str, int]:
        return {"notes": len(self.notes),
                "frontmatter_notes": sum(n.has_frontmatter for n in self.notes.values()),
                "frontmatter_row_notes": sum(n.has_frontmatter_rows for n in self.notes.values()),
                "link_occurrences": len(self.occurrences), "logical_edges": len(self.edges),
                "tag_edges": len(self.tag_edges)}  # fmt: skip

    def edges_by_rel(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, _, rel in self.edges:
            out[rel] = out.get(rel, 0) + 1
        return dict(sorted(out.items()))


def parse_vault(root: Path, render_dir: str = RENDER_DIR) -> RefVault:
    texts = scan(root, render_dir)
    paths = sorted(texts)
    others = [f for f in all_files(root, render_dir) if not f.endswith(".md")]
    notes = {p: parse_note(p, t) for p, t in texts.items()}
    occurrences: list[tuple[str, str, str]] = []
    for p, n in notes.items():
        for target, rel, _ in n.occurrences:
            if not target:
                continue
            hit = resolve(target, paths)
            if hit is None and Path(target).suffix not in ("", ".md"):
                hit = resolve_file(target, others)
            if hit == p:
                continue
            occurrences.append((p, hit if hit else f"stub:{target.casefold()}", rel))
    return RefVault(notes, occurrences, set(occurrences),
                    {(p, t) for p, n in notes.items() for t in n.tags})  # fmt: skip
