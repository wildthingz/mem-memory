"""Obsidian note → desired graph state (architecture-v2 §2.4), with no database access.

`parse_note` turns one normalised note into a `ParsedNote`: node fields, alias and prop sets, tags,
frontmatter links, and chunks by `chunk_key` with their wikilink occurrences. The importer
(`mem.importer`) diffs that against the store. Frozen readings: docs/decisions/0010 and 0011.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import yaml
from markdown_it import MarkdownIt

from mem.common import estimate_tokens, norm

MIN_TOKENS = 40  # a section under this merges into the next
MAX_TOKENS = 400  # a section over this splits at block boundaries

_COMMENT = re.compile(r"%%.*?%%", re.DOTALL)
_WIKILINK = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
_INLINE_CODE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_TAG = re.compile(r"(?<![\w/#&\[])#([\w/-]*[^\W\d][\w/-]*)")
_HEX_COLOUR = re.compile(r"(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")
_CALLOUT = re.compile(r"^\s*>\s*\[!([\w-]+)\]", re.MULTILINE)
_BLOCK_ID = re.compile(r"(?:^|\s)\^([\w-]+)\s*$", re.MULTILINE)
DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
_ENTITY_TAGS = (("person", "person"), ("project", "project"), ("team", "org"))
_CORE_RELS = {"about", "supersedes", "duplicate_of", "same_as", "relates_to", "links_to", "embeds",
              "tagged", "part_of"}  # fmt: skip

_MD = MarkdownIt("commonmark")


class ParseError(ValueError):
    """The note cannot be read (bad frontmatter YAML)."""


@dataclass(frozen=True)
class Link:
    target: str
    anchor: str | None
    alias: str | None
    embed: bool
    pos: int


@dataclass(frozen=True)
class FmLink:
    field: str
    rel: str
    link: Link  # pos = index of the value in its list (0 for a scalar)


@dataclass(frozen=True)
class ChunkPlan:
    key: str
    heading_path: str
    ord: int
    ctx: str
    text: str
    tokens: int
    content_hash: str
    props: str | None
    links: tuple[Link, ...]


@dataclass
class ParsedNote:
    title: str
    stem: str
    created: Any
    aliases: list[str]
    props: list[tuple[str, str]]
    tags: list[str]
    fm_links: list[FmLink]
    chunks: list[ChunkPlan]
    kind: str = "note"
    type: str = "note"
    filename_date: str | None = None
    body: str = field(default="", repr=False)


def normalize(raw: str) -> str:
    """LF endings, no BOM, `%%comments%%` stripped: the text that is hashed and indexed."""
    text = raw.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")
    return _COMMENT.sub("", text)


def file_hash(normalised: str) -> str:
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 3)
    while end != -1 and text[end + 4 : end + 5] not in ("", "\n"):
        end = text.find("\n---", end + 1)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[4:end])
    except yaml.YAMLError as err:
        raise ParseError(f"frontmatter is not valid YAML: {err}") from err
    body = text[end + 5 :]
    return (meta if isinstance(meta, dict) else {}), body


def _code_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of fenced / indented code blocks and inline code spans in `text`."""
    starts = [0]
    for line in text.split("\n")[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    lines = text.count("\n") + 1
    ranges = []
    for tok in _MD.parse(text):
        if tok.type in ("fence", "code_block") and tok.map:
            a, b = tok.map
            ranges.append((starts[a], starts[b] if b < lines else len(text)))
    for m in _INLINE_CODE.finditer(text):
        if not any(a <= m.start() < b for a, b in ranges):
            ranges.append((m.start(), m.end()))
    return ranges


def _outside(ranges: list[tuple[int, int]], pos: int) -> bool:
    return not any(a <= pos < b for a, b in ranges)


def _link(m: re.Match[str]) -> Link:
    inner = m.group(2)
    target, _, alias = inner.partition("|")
    target = target.rstrip("\\").strip()
    name, _, anchor = target.partition("#")
    return Link(name.strip(), anchor.strip() or None, alias.strip() or None, bool(m.group(1)),
                m.start())  # fmt: skip


def scan_links(text: str) -> list[Link]:
    code = _code_ranges(text)
    return [_link(m) for m in _WIKILINK.finditer(text) if _outside(code, m.start())]


def scan_tags(text: str) -> list[str]:
    code = _code_ranges(text)
    wikilinks = [(m.start(), m.end()) for m in _WIKILINK.finditer(text)]
    tags = []
    for m in _TAG.finditer(text):
        if not _outside(code, m.start()) or not _outside(wikilinks, m.start()):
            continue
        tag = m.group(1).strip("/")
        if tag and not _HEX_COLOUR.fullmatch(tag):
            tags.append(tag)
    return tags


@dataclass
class _Section:
    path: str
    n: int
    blocks: list[tuple[int, int]]  # line ranges [start, end) of its top-level blocks
    head: int | None = None  # first line of its heading, which opens its first chunk


def _sections(body: str) -> tuple[list[_Section], str | None]:
    """Split at H1–H3; the second value is the first H1's text. Duplicate numbers `n` count every
    section with that heading path, merged ones included, so merging never shifts a key."""
    tokens = _MD.parse(body)
    sections = [_Section("", 1, [])]
    stack: list[tuple[int, str]] = []
    seen: dict[str, int] = {"": 1}
    first_h1 = None
    for i, tok in enumerate(tokens):
        if tok.level != 0 or tok.map is None:
            continue
        if tok.type == "heading_open" and tok.tag in ("h1", "h2", "h3"):
            level = int(tok.tag[1])
            text = tokens[i + 1].content.strip()
            if level == 1 and first_h1 is None:
                first_h1 = text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            path = " > ".join(t for _, t in stack)
            seen[path] = seen.get(path, 0) + 1
            sections.append(_Section(path, seen[path], [], tok.map[0]))
        elif tok.nesting >= 0 and not tok.type.endswith("_close"):
            sections[-1].blocks.append((tok.map[0], tok.map[1]))
    if not sections[0].blocks:
        sections.pop(0)
    return sections, first_h1


def _slice(lines: list[str], blocks: list[tuple[int, int]], head: int | None = None) -> str:
    """The lines from the first block (or the heading line `head`) to the last block's end."""
    if not blocks:
        return "\n".join(lines[head : head + 1]).strip() if head is not None else ""
    return "\n".join(lines[blocks[0][0] if head is None else head : blocks[-1][1]]).strip()


def _split_blocks(lines: list[str], blocks: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Greedy groups of whole blocks, each up to MAX_TOKENS; an oversized block stays alone."""
    groups: list[list[tuple[int, int]]] = []
    used = 0
    for block in blocks:
        size = estimate_tokens("\n".join(lines[block[0] : block[1]]))
        if groups and used + size <= MAX_TOKENS:
            groups[-1].append(block)
            used += size
        else:
            groups.append([block])
            used = size
    return groups


def _chunks(body: str, title: str, sections: list[_Section]) -> list[ChunkPlan]:
    lines = body.split("\n")
    chunks: list[ChunkPlan] = []
    carry: list[str] = []
    for i, sec in enumerate(sections):
        own = _slice(lines, sec.blocks, sec.head)
        pending = [*carry, own] if own else carry
        if i < len(sections) - 1 and estimate_tokens("\n\n".join(pending)) < MIN_TOKENS:
            carry = pending
            continue
        groups = _split_blocks(lines, sec.blocks) or [[]]
        for part, group in enumerate(groups):
            piece = _slice(lines, group, sec.head if part == 0 else None)
            text = "\n\n".join([*carry, piece] if part == 0 else [piece])
            text = text.strip()
            if not text:
                continue
            chunks.append(_plan(title, sec, part, len(chunks), text))
        carry = []
    return chunks


def _plan(title: str, sec: _Section, part: int, ordinal: int, text: str) -> ChunkPlan:
    props: dict[str, list[str]] = {}
    code = _code_ranges(text)
    callouts = [m.group(1) for m in _CALLOUT.finditer(text) if _outside(code, m.start())]
    block_ids = [m.group(1) for m in _BLOCK_ID.finditer(text) if _outside(code, m.start())]
    if callouts:
        props["callouts"] = callouts
    if block_ids:
        props["block_ids"] = block_ids
    return ChunkPlan(
        key=f"{sec.path}#{sec.n}/p{part}",
        heading_path=sec.path,
        ord=ordinal,
        ctx=f"{title} > {sec.path}" if sec.path else title,
        text=text,
        tokens=estimate_tokens(text),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        props=json.dumps(props, ensure_ascii=False) if props else None,
        links=tuple(scan_links(text)),
    )


def _as_str(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _names(value: Any) -> list[str]:
    """`aliases:` / `tags:` as a list of strings (a scalar may be comma or space separated)."""
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[,\s]+", str(value))
    return [_as_str(v).strip() for v in items if _as_str(v).strip()]


def _fm_rel(key: str) -> str:
    base = rel = re.sub(r"\W+", "_", key.strip().lower()).strip("_")
    if rel.endswith("ies"):
        rel = rel[:-3] + "y"
    elif rel.endswith("s") and not rel.endswith("ss"):
        rel = rel[:-1]
    return f"fm_{base}" if {base, rel} & _CORE_RELS else rel


def _flatten(value: Any) -> list[Any]:
    return [x for v in value for x in (_flatten(v) if isinstance(v, list) else [v])]


def _frontmatter_fields(meta: dict[str, Any]) -> tuple[list[tuple[str, str]], list[FmLink]]:
    """Scalar values → props; wikilink values → edges. YAML reads an unquoted `[[X]]` as a nested
    list, so a list inside a list is one link target."""
    props: list[tuple[str, str]] = []
    links: list[FmLink] = []
    for key, value in meta.items():
        if key in ("created", "aliases", "tags") or value is None or isinstance(value, dict):
            continue
        rel = _fm_rel(str(key))
        items = value if isinstance(value, list) else [value]
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                continue
            if isinstance(item, list):
                found = [Link(_as_str(t).strip(), None, None, False, idx) for t in _flatten(item)]
            else:
                found = [Link(link.target, link.anchor, link.alias, link.embed, idx)
                         for link in scan_links(_as_str(item))]  # fmt: skip
            if found:
                links += [FmLink(str(key), rel, link) for link in found]
            else:
                props.append((str(key), _as_str(item)))
    return list(dict.fromkeys(props)), links


def parse_note(rel_path: str, normalised: str) -> ParsedNote:
    """`rel_path` is the vault-relative posix path, `normalised` the output of `normalize`."""
    meta, body = split_frontmatter(normalised)
    stem = rel_path.rsplit("/", 1)[-1].removesuffix(".md")
    stem_title = DATE_PREFIX.sub("", stem)
    sections, h1 = _sections(body)
    title = h1 or stem_title
    props, fm_links = _frontmatter_fields(meta)
    tags = list(dict.fromkeys(
        t.lstrip("#") for t in [*_names(meta.get("tags")), *scan_tags(body)] if t.lstrip("#")
    ))  # fmt: skip
    aliases: dict[str, str] = {}
    for alias in [*_names(meta.get("aliases")), title, stem_title]:
        aliases.setdefault(norm(alias), alias)
    lowered = {t.lower() for t in tags}
    kind, type_ = next(
        (("entity", t) for tag, t in _ENTITY_TAGS if tag in lowered), ("note", "note")
    )
    m = DATE_PREFIX.match(stem)
    return ParsedNote(
        title=title,
        stem=stem_title,
        created=meta.get("created"),
        aliases=list(aliases.values()),
        props=props,
        tags=tags,
        fm_links=fm_links,
        chunks=_chunks(body, title, sections),
        kind=kind,
        type=type_,
        filename_date=m.group(1) if m else None,
        body=body,
    )
