"""Write-path filters and provenance (P7 hardening, benchmarks-v2 B-H1).

- **Secrets** in any text field of `remember` reject the call: nothing is written or logged, and
  the error names the kind of secret, never its value.
- **Prompt injection** (text that addresses the assistant or forges a prompt) puts the memory in
  quarantine: it is stored with its reason, never recalled and kept out of the core index, until
  the person releases it (`mem quarantine --release`) or forgets it.
- **Tool-sourced** memories (`source` is a URL or `tool:<name>`: a fetched page, a file, command
  output, another MCP server) are second-hand. They rank below first-hand memories
  (`TOOL_WEIGHT`), carry a `tool-sourced` flag in recall, and are also quarantined on directive
  text aimed at an AI, which first-hand memories (procedures, preferences) may legitimately hold.

Matching runs on an NFKC, casefolded, zero-width-free, single-spaced copy of the text.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Iterable

from mem.common import MemError, parse_id

TOOL_WEIGHT = 0.8  # provenance weight of tool-sourced rows (first-hand rows: 1.0)
TOOL_FLAG = "tool-sourced"
_TOOL_SOURCE = re.compile(r"^(?:(?:https?|ftp)://|tool:)", re.IGNORECASE)
_INVISIBLE = re.compile(r"[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

# (name, pattern) on the raw text: secrets are case-sensitive tokens.
SECRETS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx))
    for name, rx in (
        ("private key", r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"),
        ("AWS access key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        ("GitHub token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
        ("GitLab token", r"\bglpat-[A-Za-z0-9_-]{20,}"),
        ("Slack token", r"\bxox[abpors]-[A-Za-z0-9-]{10,}"),
        ("API key", r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"),
        ("Google API key", r"\bAIza[0-9A-Za-z_-]{35}"),
        ("Stripe key", r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}"),
        ("JWT", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        ("credentials in a URL", r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s:/@]+@[^\s/@]"),
        ("bearer token", r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    )
)
# `password: hunter2!` and the like: a secret-ish key, then a value with ≥ 2 character classes,
# so `password is required` or `token: none` pass. The key may end a longer name (`DB_PASSWORD=`,
# `AWS_SECRET_ACCESS_KEY=`, `export X_API_KEY=`, YAML `postgres_password: ...`): only a letter or
# digit before it stops the match.
_ASSIGNED = re.compile(
    r"(?i)(?<![a-z0-9])(?:password|passwd|passphrase|pwd|secret"
    r"|(?:api|access|secret|private)[_ -]?key|auth[_ -]?token|client[_ -]?secret|token)\b\s*(?:[:=]|is)\s*[\"'`]?([^\s\"'`]{8,})"
)
_CLASSES = (re.compile(r"[a-z]"), re.compile(r"[A-Z]"), re.compile(r"\d"), re.compile(r"[^\w\s]"))


def _any(*alternatives: str) -> str:
    return "|".join(alternatives)


_AI = r"(?:ai|assistant|llm|agent|claude|chatbot|language model)"
_NOUN = r"(?:instructions?|prompts?|rules?|guidelines?|directions?|context|messages?)"
# Nouns for the anchored `everything before/earlier/prior ...` form only. Not `conversation` or
# `context` on their own: `forget the previous conversation` and `forget everything earlier in
# this conversation` are things a person does, so they only count tied to an instruction word
# (`the instructions in this conversation`, `the conversation rules`). Bare `message(s)` stays:
# `everything before this message` points at the prompt (F15). Keep the two lists apart:
# widening the shared one for one alternative is what produced F8, F13 and F14 in turn.
_PROMPT_NOUN = (
    r"(?:(?:conversation|context) )?(?:instructions?|prompts?|rules?|guidelines?|directions?"
    r"|(?:system )?messages?)(?: (?:in|from|of) (?:this |that |the |your |our )*(?:conversation|context))?"
)
# (name, pattern) on the normalised text: any provenance.
INJECTION: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx))
    for name, rx in (
        ("override", _any(
            (r"\b(?:ignore|disregard|forget|override|bypass) (?:all |any |the |your |of )*"
             rf"(?:previous|prior|above|earlier|preceding|existing|system|original) {_NOUN}\b"),
            # Noun-free only on the shapes that point at a prompt: `everything above`, `whatever
            # you were told`. `forget everything before 2020` addresses nobody, so `before`,
            # `earlier`, `prior`, `previously` need an instruction noun after them.
            (r"\b(?:ignore|disregard|forget) (?:all |of )*(?:everything|anything|what(?:ever)?)"
             r" (?:above|you (?:were|have been) told)\b"),
            (r"\b(?:ignore|disregard|forget) (?:all |of )*(?:everything|anything|what(?:ever)?)"
             r"(?: (?:said|written|stated|given))? (?:before|earlier|prior|previously)(?: to| in)?"
             rf" (?:this |these |that |those |the |your |my |our )*{_PROMPT_NOUN}\b"))),
        ("role reassignment", _any(
            r"\byou are (?:now|no longer)\b",
            r"\bfrom now on,? you (?:are|will|must|should)\b",
            r"\bact as (?:an?|the) (?:unrestricted|jailbroken)")),
        ("forged prompt", _any(
            r"<\|(?:im_start|im_end|system|endoftext)\|>", r"\[/?inst\]", r"</?system>",
            r"</?(?:function_calls|tool_call|invoke)\b", r"^\s*(?:system|assistant)\s*:",
            r"\b(?:new|updated|real) (?:system )?instructions\s*:")),
        ("system prompt probe",
         r"\b(?:reveal|print|show|repeat|leak) (?:your |the )?(?:system prompt|hidden instructions)"),
        ("hide from user", _any(
            r"\b(?:do not|don't|never) (?:tell|inform|mention|show|alert) (?:this to )?the user\b",
            r"\bwithout (?:telling|asking|informing|notifying) the user\b")),
        ("exfiltration", (r"\b(?:send|post|upload|forward|exfiltrate|leak) (?:all |the |any |your )*"
         r"(?:secrets|credentials|api keys|tokens|passwords|private keys|ssh keys|\.env"
         r"|env(?:ironment)?(?: variables)?)\b")),
    )
)  # fmt: skip
# Tool-sourced text only: text addressed to an AI reader, and pipe-to-shell commands.
DIRECTIVE: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rx))
    for name, rx in (
        ("addressed to an AI", _any(
            rf"\b(?:{_AI}|model)s? (?:reading|processing|summari[sz]ing|parsing|seeing) this\b",
            rf"\b(?:dear|attention|hey|note to(?: the)?) {_AI}s?\b",
            rf"\bif you are an? {_AI}\b",
            r"\bwhen you (?:read|see|recall|find|load) this\b")),
        ("pipe to shell", r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b"),
    )
)  # fmt: skip


def normalise(text: str) -> str:
    t = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return " ".join(t.casefold().split())


def is_tool_source(source: str | None) -> bool:
    return bool(source and _TOOL_SOURCE.match(source.strip()))


def find_secret(text: str) -> str | None:
    """The kind of the first secret found in `text`, else None."""
    for name, rx in SECRETS:
        if rx.search(text):
            return name
    for m in _ASSIGNED.finditer(text):
        if sum(bool(c.search(m.group(1))) for c in _CLASSES) >= 2:
            return "password or key assignment"
    return None


def find_injection(text: str, *, tool_sourced: bool) -> str | None:
    """The reason to quarantine `text`, else None."""
    flat = normalise(text)
    for name, rx in INJECTION + (DIRECTIVE if tool_sourced else ()):
        if rx.search(flat):
            return name
    return None


def check(fields: Iterable[str | None], *, tool_sourced: bool) -> str | None:
    """Reject secrets (MemError); return the quarantine reason for injection, else None."""
    texts = [f for f in fields if f]
    for text in texts:
        if kind := find_secret(text):
            raise MemError(
                f"rejected: the memory contains what looks like a secret ({kind}). Secrets are "
                "never stored: remove it (say where the secret lives instead) and save again."
            )
    for text in texts:
        if reason := find_injection(text, tool_sourced=tool_sourced):
            return reason
    return None


def weight(tool_sourced: bool) -> float:
    return TOOL_WEIGHT if tool_sourced else 1.0


def quarantined(conn: sqlite3.Connection) -> list[tuple[int, str, str | None, str]]:
    """(id, reason, source, body) of every quarantined active memory, oldest first."""
    return conn.execute(
        "SELECT id, quarantine, source_ref, coalesce(body, '') FROM nodes WHERE kind = 'memory'"
        " AND status = 'active' AND quarantine IS NOT NULL ORDER BY id"
    ).fetchall()  # fmt: skip


def release(conn: sqlite3.Connection, ident: str) -> None:
    """Take memory `ident` (m<id>) out of quarantine: the person read it and keeps it."""
    ref = parse_id(ident)
    if ref is None or ref[0] != "m":
        raise MemError(f"{ident!r}: use a memory id like m412")
    with conn:
        cur = conn.execute("UPDATE nodes SET quarantine = NULL WHERE id = ? AND kind = 'memory'"
                           " AND quarantine IS NOT NULL", (ref[1],))  # fmt: skip
    if cur.rowcount != 1:
        raise MemError(f"{ident} is not a quarantined memory")
