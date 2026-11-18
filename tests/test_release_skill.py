"""MEM2-261: the release skill is the mem2-202 skill (v1). The owner's pooled 9b-E2 verdict (2026-10-28): skill v4
(MEM2-248 + MEM2-257: v1 plus save rules for `about`, `supersedes` and `needs_choice`) does not ship; over two
B-W5 replays pooled it gave conflict_action 0.881 (< 0.926, tune-r6) and about_fill 0.961 (< 0.98). v4 stays
on `bosmang/mem2-257` as the record. The text here is the mem2-202 one, byte for byte, but for the two
version lines (frontmatter `metadata.version` and the heading) that say it is v1. Option E's v2 read rules
(counting, a later update, paging) are not in it either. ADR 0027's `needs_choice` code stays in the server.
MEM2-274: the open-source release (MEM2-246) words the user they/them; those pronoun edits, listed in
`PRONOUNS`, are the only other difference allowed."""

import hashlib
import re
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1] / "skills" / "mem" / "SKILL.md"
TEXT = SKILL.read_text(encoding="utf-8")

# sha256 of `git show bosmang/mem2-202:skills/mem/SKILL.md` (5,708 bytes)
V1_SHA256 = "47b6d3231c4082ba54b225d046fbfe7a17c411f44733ea0087140cfef62b8ceb"
VERSION_LINES = ('metadata:\n  version: "1"\n', "")
V1_HEADING = ("# Memory (mem skill v1, P3 write path)\n", "# Memory (mem v1, P3 write path)\n")
# (release text, mem2-202 text): the neutral wording of MEM2-246, each once
PRONOUNS = (
    ("Before stating their preferences", "Before stating his preferences"),
    ("If they ask about the older value", "If he asks about the older value"),
    ("only when they said it themselves", "only when he said it himself"),
    ("recall until they review it", "recall until he reviews it"),
    ("Tell them in one line; do not", "Tell him in one line; do not"),
    ("or ask them, or pass", "or ask him, or pass"),
    ("even if they never said", "even if he never said"),
    ("Tell them in one line what you did", "Tell him in one line what you did"),
)

# One phrase per rule that is not in v1: skill v4's three save rules and Option E's v2 read rules.
_NOT_V1 = (
    r"`about` = every named entity",
    r"older memory.s same subject and attribute",
    r"`relates_to` only when both stay true",
    r"`needs_choice` for an `about` name",
    r"Counting, ordering",
    r"Count the lines you wrote",
    r"for a later update",
    r"latest `valid_from`",
    r"page\s+through them",
    r"continuation token or page offer",
)


def _as_mem2_202(text: str) -> str:
    text = text.replace(*VERSION_LINES, 1).replace(*V1_HEADING, 1)
    for neutral, v1 in PRONOUNS:
        assert text.count(neutral) == 1, neutral
        text = text.replace(neutral, v1)
    return text


def _later_rules(text: str) -> list[str]:
    return [p for p in _NOT_V1 if re.search(p, text)]


def test_the_release_skill_is_the_mem2_202_text_but_for_the_version_lines():
    assert hashlib.sha256(_as_mem2_202(TEXT).encode("utf-8")).hexdigest() == V1_SHA256


def test_the_release_skill_does_not_call_the_user_he():
    assert not re.search(r"\b(he|him|his|himself)\b", TEXT)


def test_the_skill_version_says_v1():
    assert re.search(r"^# Memory \(mem skill v1\b", TEXT, re.MULTILINE)
    assert re.search(r'^  version: "1"$', TEXT, re.MULTILINE)


def test_no_v4_save_rule_and_no_e_v2_read_rule_is_in_the_skill():
    assert _later_rules(TEXT) == []
    assert not re.search(r"^## 3a\.", TEXT, re.MULTILINE)


def test_v1_keeps_its_own_about_line():
    assert "Name every subject the text mentions" in TEXT


def test_the_checks_are_falsifiable():
    """They fail on the v4 text (its save rules) and on v2's read rules; the sha check fails on any edit."""
    v4 = (
        "- `about` = every named entity the memory states a fact about\n"
        "- A new value for an older memory's same subject and attribute → `supersedes`, not `relates_to`.\n"
        "- `remember` answered `needs_choice` for an `about` name → resend with the `#e<id>` it names.\n"
    )
    assert len(_later_rules(v4)) == 3
    assert _later_rules("## 3a. Counting, ordering\nCount the lines you wrote, not the rows.")
    edited = _as_mem2_202(TEXT).replace("atomic", "atomic ", 1)
    assert hashlib.sha256(edited.encode("utf-8")).hexdigest() != V1_SHA256


def test_every_generated_snippet_carries_no_later_rule():
    hx = pytest.importorskip("mem.harnesses")  # on bosmang/setup-integrate; runs once merged
    for name in hx.HARNESSES:
        assert _later_rules(hx.instructions(hx.get(name), TEXT)) == [], name
