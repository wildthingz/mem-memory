"""MEM2-222 F23: the skill's rules checked against the code they describe, not only for their text.

Every tool, parameter, enum value, link relation and reply status the skill tells the model to use must
be one the server accepts; the version lives in the frontmatter and agrees with the heading."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from mem import common, contracts, link
from mem.gate import Gate
from tests.test_recall_p2 import _save, gated, logit  # noqa: F401 (fixture)

SKILL = Path(__file__).resolve().parents[1] / "skills" / "mem" / "SKILL.md"
TEXT = SKILL.read_text(encoding="utf-8")
_, FRONT, BODY = TEXT.split("---\n", 2)
META = yaml.safe_load(FRONT)
TOOLS = {t["name"]: t for t in contracts.mcp_tools()["tools"]}


def _props(tool: str) -> set[str]:
    return set(TOOLS[tool]["inputSchema"]["properties"])


def _section(title: str) -> str:
    m = re.search(rf"^## (?:\d+[a-z]?\. )?{re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)",
                  BODY, re.MULTILINE | re.DOTALL)  # fmt: skip
    assert m, f"SKILL.md has no section {title!r}"
    return m.group(1)


def test_the_version_is_in_the_frontmatter_and_matches_the_heading():
    assert META["name"] == "mem"
    version = str(META["metadata"]["version"])
    assert re.fullmatch(r"\d+", version)
    assert re.search(rf"^# Memory \(mem skill v{version}\b", BODY, re.MULTILINE)


def test_the_tools_the_skill_lists_are_exactly_the_servers():
    listed = re.search(r"Tools on each: (.*?)\.\n", BODY, re.DOTALL)[1]
    assert re.findall(r"`(\w+)`", listed) == list(TOOLS)


def test_every_recall_filter_the_skill_names_is_a_recall_parameter():
    body = _section("Recall first")
    named = {"about", "type", "kinds", "since", "until", "as_of", "include_history"}
    assert all(f"`{n}" in body for n in named)
    assert named <= _props("recall")
    kinds = set(
        re.search(r"`kinds`\s*\(([^)]*)\)", body)[1].replace("`", "").replace(",", " ").split()
    )
    assert kinds == set(TOOLS["recall"]["inputSchema"]["properties"]["kinds"]["items"]["enum"])


def test_every_remember_field_the_skill_names_is_a_remember_parameter():
    body = _section("How to save")
    named = set(re.findall(r"`(\w+)`(?: = | is required|\s*\(| and |:)", body)) | {"stated_by_user"}
    named -= {"rejected", "quarantined", "saved", "exists"}
    assert named and named <= _props("remember"), named - _props("remember")
    assert set(TOOLS["remember"]["inputSchema"]["required"]) - {"text"} <= named | {"type"}


def test_the_memory_types_the_skill_offers_are_the_stores():
    line = re.search(r"`type`: ([^\n]+)", _section("How to save"))[1]
    assert tuple(t.strip(" .") for t in line.split("·")) == common.MEMORY_TYPES


def test_the_entity_types_the_skill_offers_are_the_stores():
    types = re.search(r"\(types: ([^)]*)\)", _section("How to save"))[1]
    assert tuple(t.strip() for t in types.split(",")) == common.ENTITY_TYPES


def test_the_reply_statuses_the_skill_handles_are_the_servers():
    statuses = set(TOOLS["remember"]["structuredContent"]["properties"]["status"]["enum"])
    for s in statuses:
        assert f"`{s}`" in BODY, s


def test_every_link_relation_the_skill_uses_is_accepted_by_the_store(store):
    rels = set(re.findall(r'rel="(\w+)"', BODY)) | set(re.findall(r"link m\d+ (\w+) m\d+", BODY))
    assert rels == {"supersedes", "duplicate_of", "relates_to", "same_as"}
    assert rels <= set(re.findall(r"\w+", link._valid_rels(store.conn)))
    for args in re.findall(r"link\(([^)]*)\)", BODY):
        assert {k for k in re.findall(r"(\w+)=", args)} <= _props("link")
    assert re.search(r"`forget\((\w+), (\w+)\)`", BODY).groups() == tuple(
        TOOLS["forget"]["inputSchema"]["required"]
    )


def test_the_abstention_line_the_skill_quotes_is_what_recall_prints(gated):  # noqa: F811
    """The skill teaches `no relevant memories (best .22)` as the whole answer: one line, that
    wording and that p format (the store's scope follows it)."""
    quoted = re.search(r"`(no relevant memories \(best \.\d\d\))`", BODY)[1]
    s = gated({"alpha": logit(0.22)}, gate=Gate(1.0, 0.0, tau_keep=0.5, tau_high=0.8, gap=0.2,
                                               fingerprint_hash="0" * 64))  # fmt: skip
    _save(s, "Chess opening alpha: the Sicilian.")
    res = s.recall(query="chess opening")
    assert res.structured["abstained"] is True and "\n" not in res.text
    assert res.text == f"{quoted} · personal"
