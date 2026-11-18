"""Harness integrations (MEM2-235; docs/harnesses.md): the MCP servers and the mem skill in
Claude Code, Codex CLI, Gemini CLI, Cursor, VS Code (Copilot agent) and Claude Desktop.

The MCP servers are the interface everywhere. `skills/mem/SKILL.md` is the one source of the
"when to recall, when to save" behaviour: a harness that loads Agent Skills gets that file in its
skills directory; `instructions()` makes a short snippet from the same file for the rest (and for
older versions). Config writers merge into the harness's file, leave other servers alone, save the
file in `Backups` before the first edit, and do nothing on a second run. Removal restores the saved
bytes when the file is otherwise as we left it, and takes out only our entries when it is not.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mem.common import MemError
from mem.install import prune_empty
from mem.serve import STORE_NAMES
from mem.setup_steps import Backups, Context, replace_file

_CHECKOUT_SKILL = Path(__file__).resolve().parents[2] / "skills" / "mem"
SKILL_SOURCE = (
    _CHECKOUT_SKILL
    if _CHECKOUT_SKILL.is_dir()
    else Path(__file__).parent / "_data" / "skills" / "mem"
)
TRANSPORTS = ("http", "stdio")
PLACEHOLDER = "<token>"


@dataclass(frozen=True)
class Spec:
    """What the entries say: the local server's address and label, the bearer token (None: no
    Authorization header), and for stdio the repo to run `mem` from and one store file per name."""

    host_label: str
    port: int
    token: str | None
    repo: Path
    stores: Mapping[str, Path]


@dataclass(frozen=True)
class Harness:
    name: str
    label: str
    fmt: str  # "json" | "toml"
    key: str  # the table that holds the servers
    http_shape: (
        str | None
    )  # "typed" (type+url), "plain" (url), "httpUrl"; None: no HTTP in the file
    typed_stdio: bool
    skills: str | None  # ".claude" | ".agents"; None: no native skills directory
    instructions_file: str
    commands: tuple[str, ...] = ()
    folder: str = ""  # relative to HOME, unless `config_dir` is used

    @property
    def native_skill(self) -> bool:
        return self.skills is not None

    @property
    def transports(self) -> tuple[str, ...]:
        return TRANSPORTS if self.http_shape else ("stdio",)


HARNESSES: dict[str, Harness] = {h.name: h for h in [
    Harness("claude-code", "Claude Code", "json", "mcpServers", "typed", True, ".claude",
            "CLAUDE.md", ("claude",), ".claude"),
    Harness("codex", "Codex CLI", "toml", "mcp_servers", "toml", True, ".agents",
            "AGENTS.md", ("codex",), ".codex"),
    Harness("gemini", "Gemini CLI", "json", "mcpServers", "httpUrl", False, ".agents",
            "GEMINI.md", ("gemini",), ".gemini"),
    Harness("cursor", "Cursor", "json", "mcpServers", "plain", True, ".agents",
            ".cursor/rules/mem.mdc", ("cursor",), ".cursor"),
    Harness("vscode", "VS Code (Copilot agent)", "json", "servers", "typed", True, ".agents",
            ".github/copilot-instructions.md", ("code",)),
    Harness("claude-desktop", "Claude Desktop", "json", "mcpServers", None, False, None,
            "Settings > Profile > personal preferences (paste the text)"),
    Harness("generic", "Any other MCP client", "json", "mcpServers", "typed", True, None,
            "docs/instructions.md"),
]}  # fmt: skip
CONFIGURED = [n for n in HARNESSES if n != "generic"]  # the ones with a config file here


def get(name: str) -> Harness:
    if name not in HARNESSES:
        raise MemError(f"unknown harness {name!r}: one of {', '.join(HARNESSES)}")
    return HARNESSES[name]


def _config_base(home: Path, env: Mapping[str, str], platform: str) -> Path:
    if platform == "darwin":
        return home / "Library" / "Application Support"
    return Path(env.get("XDG_CONFIG_HOME") or home / ".config")


def config_path(h: Harness, home: Path, env: Mapping[str, str], platform: str) -> Path | None:
    match h.name:
        case "claude-code":
            return home / ".claude.json"
        case "codex":
            return Path(env.get("CODEX_HOME") or home / ".codex") / "config.toml"
        case "gemini":
            return home / ".gemini" / "settings.json"
        case "cursor":
            return home / ".cursor" / "mcp.json"
        case "vscode":
            return _config_base(home, env, platform) / "Code" / "User" / "mcp.json"
        case "claude-desktop":
            return _config_base(home, env, platform) / "Claude" / "claude_desktop_config.json"
    return None


# ---- entries -----------------------------------------------------------------------------------


def _headers(spec: Spec, redact: bool) -> dict[str, str]:
    headers = {"X-Mem-Host": spec.host_label}
    if redact or spec.token:
        headers["Authorization"] = f"Bearer {PLACEHOLDER if redact else spec.token}"
    return headers


def entries(h: Harness, spec: Spec, transport: str, redact: bool = False) -> dict[str, dict]:
    """mem-personal and mem-work as this harness spells them."""
    if transport not in h.transports:
        raise MemError(
            f"{h.label} reads only {' and '.join(h.transports)} servers from its config file"
            + (
                " (remote servers go through its Connectors screen, which needs public HTTPS)"
                if h.name == "claude-desktop"
                else ""
            )
        )
    out: dict[str, dict] = {}
    for name in STORE_NAMES:
        if transport == "http":
            url = f"http://127.0.0.1:{spec.port}/{name}/mcp"
            head = _headers(spec, redact)
            match h.http_shape:
                case "httpUrl":
                    e: dict[str, Any] = {"httpUrl": url, "headers": head}
                case "plain" | "toml":
                    e = {"url": url, "headers": head}
                case _:
                    e = {"type": "http", "url": url, "headers": head}
        else:
            if name not in spec.stores:
                raise MemError(f"stdio needs a path for each of the stores; none for {name}")
            e = {"command": "uv", "args": ["run", "--project", str(spec.repo), "mem", "serve",
                                           "--store", f"{name}={spec.stores[name]}"]}  # fmt: skip
            if h.typed_stdio:
                e = {"type": "stdio", **e}
        out[f"mem-{name}"] = e
    return out


def _toml_table(name: str, e: dict) -> str:
    q = json.dumps
    lines = [f"[mcp_servers.{name}]"]
    if "url" in e:
        lines.append(f"url = {q(e['url'])}")
        pairs = ", ".join(f"{q(k)} = {q(v)}" for k, v in e["http_headers"].items())
        lines.append(f"http_headers = {{ {pairs} }}")
    else:
        lines.append(f"command = {q(e['command'])}")
        lines.append(f"args = [{', '.join(q(a) for a in e['args'])}]")
    return "\n".join(lines) + "\n"


def _toml_body(es: dict[str, dict]) -> str:
    return "\n".join(_toml_table(n, e) for n, e in es.items())


def _codex_entries(spec: Spec, transport: str, redact: bool) -> dict[str, dict]:
    es = entries(HARNESSES["codex"], spec, transport, redact)
    for e in es.values():
        e.pop("type", None)
        if "headers" in e:
            e["http_headers"] = e.pop("headers")
    return es


def _entries_for(h: Harness, spec: Spec, transport: str, redact: bool = False) -> dict[str, dict]:
    return _codex_entries(spec, transport, redact) if h.fmt == "toml" else entries(
        h, spec, transport, redact)  # fmt: skip


# ---- documents: JSON as a dict, TOML as text ---------------------------------------------------

_STORES = "|".join(STORE_NAMES)
_NAME = re.compile(rf"^mem-(?:[\w-]+-)?(?:{_STORES})$")  # mem-work, or mem-<remote>-work
_DEC = json.JSONDecoder()


def _blank(text: str, commas: bool = True) -> str:
    """JSONC as plain JSON at the same offsets: comments and (with `commas`) trailing commas
    become spaces (newlines stay). Gemini CLI and VS Code read their files as JSONC."""
    out, i, n = list(text), 0, len(text)
    comma = None  # a ',' with only blanks after it so far
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            i, comma = j + 1, None
            continue
        if text.startswith(("//", "/*"), i):
            j = text.find("\n" if c == "/" and text[i + 1] == "/" else "*/", i + 2)
            j = n if j < 0 else j if text[i + 1] == "/" else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in text[i:j]]
            i = j
            continue
        if c in "}]" and comma is not None and commas:
            out[comma] = " "
        comma = i if c == "," else comma if c.isspace() else None
        i += 1
    return "".join(out)


def _notes(h: Harness, text: str) -> str:
    """What JSONC adds to the JSON (its comments): restoring a backup must not lose them."""
    plain = _blank(text, commas=False)
    return "" if h.fmt == "toml" else "".join(c for c, b in zip(text, plain) if c != b)


def _load(h: Harness, raw: bytes | None, path: Path) -> Any:
    text = (raw or b"").decode("utf-8", errors="replace")
    try:
        if h.fmt == "toml":
            tomllib.loads(text)
            return text
        plain = _blank(text)
        data = json.loads(plain) if plain.strip() else {}
    except ValueError as err:  # JSONDecodeError and TOMLDecodeError
        raise MemError(f"{path} does not parse: {err}. Fix: correct it, or merge the entries "
                       f"`mem mcp-config --harness {h.name}` prints by hand") from err  # fmt: skip
    if not isinstance(data, dict) or not isinstance(data.get(h.key, {}), dict):
        raise MemError(f"{path} does not parse as a config with a {h.key!r} table")
    return data


def _read(h: Harness, path: Path) -> tuple[str, Any]:
    """The file's text and its document (an empty one when the file does not exist)."""
    raw = path.read_bytes() if path.exists() else None
    return (raw or b"").decode("utf-8", errors="replace"), _load(h, raw, path)


def _parsed(h: Harness, doc: Any) -> dict:
    return tomllib.loads(doc) if h.fmt == "toml" else doc


def _servers(h: Harness, doc: Any) -> dict:
    return _parsed(h, doc).get(h.key, {})


def _has(h: Harness, doc: Any, es: dict[str, dict]) -> bool:
    have = _servers(h, doc)
    return all(have.get(n) == e for n, e in es.items())


def _merge(h: Harness, doc: Any, es: dict[str, dict]) -> Any:
    if h.fmt == "toml":
        text = _strip(h, doc, set(es))
        text = text if text.endswith("\n") or not text else text + "\n"
        return text + ("\n" if text else "") + _toml_body(es)
    return {**doc, h.key: {**doc.get(h.key, {}), **es}}


def _toml_names(names: set[str] | None) -> str:
    return "|".join(map(re.escape, sorted(names))) if names else rf"mem-(?:[\w-]+-)?(?:{_STORES})"


def _strip(h: Harness, doc: Any, names: set[str] | None = None) -> Any:
    """`doc` without our entries (all of them, or the ones in `names`). In TOML an entry is a
    `[mcp_servers.mem-*]` table, or a `mem-* = {...}` / `mem-*.url = ...` line in `[mcp_servers]`
    (`mcp_servers.mem-*... =` at the top)."""
    ours = (lambda k: k in names) if names else _NAME.match
    if h.fmt == "json":
        return {**doc, h.key: {k: v for k, v in doc.get(h.key, {}).items() if not ours(k)}}
    name = rf"""["']?(?:{_toml_names(names)})["']?\s*"""
    table = re.compile(rf"^\[\s*mcp_servers\s*\.\s*{name}[.\]]")
    in_servers = re.compile(rf"^{name}[.=]")
    at_top = re.compile(rf"^mcp_servers\s*\.\s*{name}[.=]")
    out: list[str] = []
    skipping, where = False, None
    for line in doc.splitlines(keepends=True):
        s = line.strip()
        if s.startswith("["):
            skipping, where = bool(table.match(s)), re.sub(r"\s", "", s)
        elif (where is None and at_top.match(s)) or (
                where == "[mcp_servers]" and in_servers.match(s)):  # fmt: skip
            continue
        if not skipping:
            out.append(line)
    text = "".join(out)
    return text[:-1] if text.endswith("\n\n") else text


# ---- writing: only what reads back as intended -------------------------------------------------


def _ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _members(plain: str, i: int) -> tuple[list[tuple[str, int, int, int]], int]:
    """The object opening at `plain[i]`: (name, key start, value start, value end) per member,
    and where its `}` is."""
    out = []
    i = _ws(plain, i + 1)
    while plain[i] != "}":
        name, j = _DEC.raw_decode(plain, i)
        v = _ws(plain, _ws(plain, j) + 1)
        _value, end = _DEC.raw_decode(plain, v)
        out.append((name, i, v, end))
        i = _ws(plain, end)
        i = _ws(plain, i + 1) if plain[i] == "," else i
    return out, i


def _line(text: str, i: int) -> tuple[int, int]:
    """Start and end (at the newline) of the line holding `text[i]`."""
    end = text.find("\n", i)
    return text.rfind("\n", 0, i) + 1, len(text) if end < 0 else end


def _indent(text: str, i: int) -> str:
    return re.match(r"[ \t]*", text[_line(text, i)[0] :]).group()  # type: ignore[union-attr]


def _object_edits(text: str, plain: str, at: int, want: dict) -> list[tuple[int, int, str]]:
    """Edits that make the object at `at` hold `want`: changed values replaced, members not in
    `want` cut (with their line), new ones added at the end. The rest of the text stays."""
    members, close = _members(plain, at)
    marks = _blank(text, commas=False)  # where the commas are, trailing ones too
    have = {m[0]: json.loads(plain[m[2] : m[3]]) for m in members}
    edits: list[tuple[int, int, str]] = []
    kept = [m for m in members if m[0] in want]
    ind = _indent(text, members[0][1]) if members else _indent(text, at) + "  "

    def dumps(v: Any) -> str:
        return json.dumps(v, indent=2, ensure_ascii=False).replace("\n", "\n" + ind)

    for name, k, v, end in members:
        if name not in want:
            start = _line(plain, k)[0]
            whole = not plain[start:k].strip()  # the member starts its line: cut the line
            q = _ws(marks, end)
            stop = q + 1 if marks[q] == "," else end
            eol = _line(plain, stop)[1]
            if whole and not plain[stop:eol].strip() and eol < len(plain):
                stop = eol + 1
            edits.append((start if whole else k, stop, ""))
        elif have[name] != want[name]:
            edits.append((v, end, dumps(want[name])))
    new = [n for n in want if n not in have]
    after = None  # where the last member we keep ends, and whether a ',' follows it
    if kept:
        q = _ws(marks, kept[-1][3])
        after = (q + 1, True) if marks[q] == "," else (kept[-1][3], False)
        if after[1] and not new and kept[-1] is not members[-1]:
            edits.append((q, q + 1, ""))  # its ',' led to members that are gone
    if not new:
        return edits
    block = ",".join(f"\n{ind}{json.dumps(n, ensure_ascii=False)}: {dumps(want[n])}" for n in new)
    pos = after[0] if after else at + 1
    eol = _line(plain, pos)[1]
    put = eol if not plain[pos:eol].strip() else pos
    if after and not after[1]:
        block = "," + block if put == pos else block
        if put != pos:
            edits.append((pos, pos, ","))
    if not members:
        block += "\n" + _indent(text, close)
    edits.append((put, put, block))
    return edits


def _edit_jsonc(text: str, key: str, want: dict) -> str:
    """`text` (JSONC with comments) changed to `want` by editing only the members that change, so
    the comments and the layout of the rest survive."""
    plain = _blank(text)
    top = _ws(plain, 0)
    have = json.loads(plain)
    edits: list[tuple[int, int, str]] = []
    if key in have:
        at = next(m[2] for m in _members(plain, top)[0] if m[0] == key)
        edits = _object_edits(text, plain, at, want[key])
    elif want[key]:
        edits = _object_edits(text, plain, top, {**have, key: want[key]})
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def _same(h: Harness, got: dict, want: dict) -> bool:
    rest = lambda d: {k: v for k, v in d.items() if k != h.key}
    return rest(got) == rest(want) and got.get(h.key, {}) == want.get(h.key, {})


def _save(h: Harness, path: Path, text: str, after: Any, want: dict) -> None:
    """Write `after` (for JSON: `want`), but only when it reads back as `want`. A file with
    comments is edited in place so they stay; a file that would not parse is never written."""
    _commit(path, _render(h, path, text, after, want))


def _render(h: Harness, path: Path, text: str, after: Any, want: dict) -> bytes:
    """The bytes `_save` writes; raises MemError, writing nothing, when they would not read back
    as `want`."""
    if h.fmt == "toml":
        out = after
    elif _blank(text) != text:
        out = _edit_jsonc(text, h.key, want)
    else:
        out = json.dumps(after, indent=2, ensure_ascii=False) + "\n"
    try:
        good = _same(h, _parsed(h, _load(h, out.encode(), path)), want)
    except MemError:
        good = False
    if not good:
        raise MemError(f"{path}: mem cannot change this file safely (it holds the servers in a "
                       f"shape mem does not edit); it is left as it is. Fix: change the entries by "
                       f"hand as `mem mcp-config --harness {h.name}` prints them")  # fmt: skip
    return out.encode()


def _want_merged(h: Harness, doc: Any, es: dict[str, dict]) -> dict:
    d = _parsed(h, doc)
    return {**d, h.key: {**d.get(h.key, {}), **es}}


def _want_stripped(h: Harness, doc: Any) -> dict:
    d = _parsed(h, doc)
    return {**d, h.key: {k: v for k, v in d.get(h.key, {}).items() if not _NAME.match(k)}}


def _commit(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o7777 if path.exists() else 0o600  # holds the bearer token
    replace_file(path, data, mode)


def _unchanged_since(h: Harness, text: str, doc: Any, saved: bytes | None, path: Path,
                     ours: dict[str, dict]) -> bool:  # fmt: skip
    """The file is the saved one plus `ours` and nothing else (no comment added either)."""
    orig = _load(h, saved, path)
    return (path.exists() and _strip(h, doc) == _strip(h, _merge(h, orig, ours))
            and _notes(h, text) == _notes(h, (saved or b"").decode("utf-8", errors="replace")))  # fmt: skip


def write_config(h: Harness, spec: Spec, transport: str, path: Path, backups: Backups) -> str:
    """'written' or 'unchanged'. The file is saved in `backups` before its first edit."""
    es = _entries_for(h, spec, transport)
    text, doc = _read(h, path)
    if _has(h, doc, es):
        return "unchanged"
    want = _want_merged(h, doc, es)
    after = _merge(h, doc, es)
    backups.save(path)
    _save(h, path, text, after, want)
    return "written"


def remove_config(h: Harness, spec: Spec, transport: str, path: Path, backups: Backups) -> str:
    """'restored' (the saved bytes are back), 'removed' (only our entries went: the file had been
    edited since) or 'absent'."""
    es = _entries_for(h, spec, transport)
    text, doc = _read(h, path)
    if backups.has(path) and _unchanged_since(h, text, doc, backups.original(path), path, es):
        backups.restore(path)
        return "restored"
    if not any(n in _servers(h, doc) for n in es):
        backups.drop(path)
        return "absent"
    _save(h, path, text, _strip(h, doc), _want_stripped(h, doc))
    backups.drop(path)
    return "removed"


def render_config(h: Harness, spec: Spec, transport: str, redact: bool = True) -> str:
    es = _entries_for(h, spec, transport, redact)
    if h.fmt == "toml":
        return _toml_body(es)
    return json.dumps({h.key: es}, indent=2, ensure_ascii=False) + "\n"


def render_generic(spec: Spec, redact: bool = True) -> str:
    h = HARNESSES["generic"]
    return (
        "# HTTP: the local server `mem install` starts; one entry per store.\n"
        "# Clients that name the type differently use `streamable-http`, or `httpUrl`, or no type.\n"
        f"{render_config(h, spec, 'http', redact)}\n"
        "# stdio: one process per store; it loads the models itself, so prefer HTTP.\n"
        f"{render_config(h, spec, 'stdio', redact)}"
    )


# ---- native skills -----------------------------------------------------------------------------


def skill_dir(h: Harness, home: Path) -> Path | None:
    return home / h.skills / "skills" / "mem" if h.skills else None


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def install_skill(h: Harness, home: Path, src: Path = SKILL_SOURCE) -> str:
    dest = skill_dir(h, home)
    if dest is None:
        raise MemError(
            f"{h.label} has no skills directory: use `mem instructions --harness {h.name}`"
        )
    if dest.is_dir() and _tree(dest) == _tree(src):
        return "unchanged"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return "installed"


def remove_skill(h: Harness, home: Path) -> str:
    dest = skill_dir(h, home)
    if dest is None or not dest.exists():
        return "absent"
    shutil.rmtree(dest)
    prune_empty(dest, home)
    return "removed"


# ---- generated instructions --------------------------------------------------------------------

# (section title, which of its items): None = all, else a test on the item text.
_FROM_SKILL: list[tuple[str, Any]] = [
    ("Stores", None),
    ("Recall first", None),
    ("Read cheaply", lambda item: "tool-sourced" in item or "abstention" in item),
    ("What to save", None),
    ("How to save", lambda item: item.startswith("- ") and item.startswith("- One atomic")),
]
_HEADING = re.compile(r"^## (?:\d+\. )?(.*)$")
_CURSOR_HEAD = (
    "---\ndescription: When to recall from and save to the mem long-term memory "
    "(generated from skills/mem/SKILL.md)\nalwaysApply: true\n---\n"
)


def _sections(skill: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in skill.splitlines():
        m = _HEADING.match(line)
        if m:
            current = out.setdefault(m.group(1), [])
        elif current is not None:
            current.append(line)
    return {t: _trim(b) for t, b in out.items()}


def _trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def _items(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        if line.startswith((" ", "\t")) and items:
            items[-1] += "\n" + line
        elif line.strip():
            items.append(line)
    return items


def instructions(h: Harness, skill: str) -> str:
    """The short recall/save rules for `h`, cut from the text of SKILL.md. Fails when SKILL.md no
    longer has a section this is built from, so the snippet cannot drift without a test failing."""
    sections = _sections(skill)
    blocks: list[str] = []
    for title, keep in _FROM_SKILL:
        full = next((t for t in sections if t.startswith(title)), None)
        if full is None:
            raise MemError(f"SKILL.md has no section '{title}': update harnesses._FROM_SKILL")
        body = sections[full]
        if keep is not None:
            body = [i for i in _items(body) if keep(i)]
            if not body:
                raise MemError(
                    f"SKILL.md section '{title}' lost the rules the snippet is built from"
                )
            body = "\n".join(body).splitlines()
        blocks.append("\n".join([f"## {full}", *body]))
    head = (
        "# mem: long-term memory\n\n"
        "Generated from skills/mem/SKILL.md by `mem instructions`; edit that file, not this one.\n"
        "The tools come from the `mem-personal` and `mem-work` MCP servers.\n"
    )
    text = head + "\n" + "\n\n".join(blocks) + "\n"
    return _CURSOR_HEAD + text if h.name == "cursor" else text


# ---- detection ---------------------------------------------------------------------------------


def detect(home: Path, runner: Any, env: Mapping[str, str], platform: str) -> list[str]:
    """The harnesses installed here: their command is on PATH (through `runner.which`) or their
    config folder exists under `home`. In a fixed order."""
    found = []
    for name in CONFIGURED:
        h = HARNESSES[name]
        folder = home / h.folder if h.folder else config_path(h, home, env, platform).parent  # type: ignore[union-attr]
        if any(runner.which(c) for c in h.commands) or folder.is_dir():
            found.append(name)
    return found


# ---- the writers `mem setup` drives (setup_steps.HarnessWriter) --------------------------------


def _endpoint_entries(h: Harness, endpoints: list[Any]) -> dict[str, dict]:
    """One entry per setup Endpoint, in this harness's spelling (HTTP, with the bearer token)."""
    if not h.http_shape:
        raise MemError(f"{h.label} reads only stdio servers from its config file")
    out: dict[str, dict] = {}
    for ep in endpoints:
        token = ep.token_file.read_text(encoding="utf-8").strip()
        head = {**dict(ep.headers), "Authorization": f"Bearer {token}"}
        key = "headers"
        match h.http_shape:
            case "httpUrl":
                e: dict[str, Any] = {"httpUrl": ep.url, key: head}
            case "plain":
                e = {"url": ep.url, key: head}
            case "toml":
                e = {"url": ep.url, "http_headers": head}
            case _:
                e = {"type": "http", "url": ep.url, key: head}
        out[ep.name] = e
    return out


class ConfigWriter:
    """A harness's MCP config file as a `HarnessWriter`: merge the servers in (file saved in
    `Backups` first), leave the rest alone, and on `remove` put the saved bytes back."""

    def __init__(
        self, name: str, env: Mapping[str, str] | None = None, platform: str | None = None
    ):
        self.h = get(name)
        self.name = self.h.label
        self.env = os.environ if env is None else env
        self.platform = sys.platform if platform is None else platform

    def _path(self, ctx: Context) -> Path:
        return config_path(self.h, ctx.home, self.env, self.platform)  # type: ignore[return-value]

    def detect(self, ctx: Context) -> bool:
        return self.h.name in detect(ctx.home, ctx.runner, self.env, self.platform)

    def _doc(self, ctx: Context) -> Any:
        return _read(self.h, self._path(ctx))[1]

    def configured(self, ctx: Context, endpoints: list[Any]) -> bool:
        """Raises MemError when the file cannot be read, before setup plans anything for it."""
        doc = self._doc(ctx)
        if not all(ep.token_file.is_file() for ep in endpoints):
            return False  # setup has not written the token yet, so nothing can match it
        return _has(self.h, doc, _endpoint_entries(self.h, endpoints))

    def add(self, ctx: Context, endpoints: list[Any]) -> None:
        path, h = self._path(ctx), self.h
        es = _endpoint_entries(h, endpoints)
        text, doc = _read(h, path)
        if h.native_skill:
            install_skill(h, ctx.home)
        if _has(h, doc, es):
            return
        out = _render(h, path, text, _merge(h, doc, es), _want_merged(h, doc, es))
        ctx.backups.save(path)  # only once the file can be changed: a refusal leaves no slot
        _commit(path, out)

    def installed(self, ctx: Context) -> bool:
        path = self._path(ctx)
        if ctx.backups.has(path):
            return True
        return path.exists() and any(_NAME.match(k) for k in _servers(self.h, self._doc(ctx)))

    def remove(self, ctx: Context) -> None:
        path, h = self._path(ctx), self.h
        if h.native_skill:
            remove_skill(h, ctx.home)
        text, doc = _read(h, path)
        ours = {k: v for k, v in _servers(h, doc).items() if _NAME.match(k)}
        if ctx.backups.has(path) and _unchanged_since(
                h, text, doc, ctx.backups.original(path), path, ours):  # fmt: skip
            ctx.backups.restore(path)
            return
        if ours:
            _save(h, path, text, _strip(h, doc), _want_stripped(h, doc))
        ctx.backups.drop(path)


def writers() -> list[ConfigWriter]:
    """The writers beyond Claude Code: every harness with an HTTP-capable config file (Claude Desktop is paste-only)."""
    return [ConfigWriter(n) for n in CONFIGURED if n != "claude-code" and HARNESSES[n].http_shape]
