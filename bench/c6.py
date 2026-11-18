"""B-C6 session overhead (benchmarks-v2 §2 B-C6): fixed tokens a Claude Code session pays for mem.

`count_tokens` on one prompt, without and with the two MCP servers' tool schemas (named as Claude
Code names them, `mcp__mem-personal__recall` ...) and the skill, and optionally the `SessionStart`
core block. The overhead is the difference. The tool list comes from the frozen contract
(`mem.contracts.mcp_tools()`, the same JSON `mem serve` advertises: B-S1 pins the two together) or,
with `--tools-url`, from a live MCP endpoint.

R5 (architecture-v2 R5: one server with a `store` param) is a shape of the tool list, not a claim:
with `r5_applied` the tools are counted under ONE server only if every tool has a `store` param;
otherwise the record keeps the two servers / 12 tools it actually counted and the gate fails it.

Skill: Claude Code always loads the skill's name + description and loads the body when it is
invoked. The gate counts the whole `SKILL.md` as the system text (the worst case); the
description-only figure is recorded too.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mem import contracts

SERVERS = ("mem-personal", "mem-work")
R5_SERVERS = ("mem",)
PROMPT = "Hello."
BUDGET = 2500
ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "mem" / "SKILL.md"

# counter(messages=[...], tools=[...] | None, system=str | None) -> input_tokens
Counter = Callable[..., int]


def has_store_param(tools: list[dict[str, Any]]) -> bool:
    """The R5 shape: every tool takes a `store` argument."""
    return bool(tools) and all("store" in t["inputSchema"].get("properties", {}) for t in tools)


def tool_defs(
    tools: list[dict[str, Any]] | None = None, servers: tuple[str, ...] = SERVERS
) -> list[dict[str, Any]]:
    """The tools of the servers as the API sees them: name, description, input_schema."""
    tools = tools if tools is not None else contracts.mcp_tools()["tools"]
    return [{"name": f"mcp__{server}__{t['name']}", "description": t["description"],
             "input_schema": t["inputSchema"]} for server in servers for t in tools]  # fmt: skip


def r5_shape_ok(rec: dict[str, Any]) -> bool:
    """A record may claim R5 only if it counted one server whose tools all take `store`."""
    return len(rec.get("servers", ())) == 1 and bool(rec.get("store_param"))


def fetch_tools(url: str) -> list[dict[str, Any]]:
    """The tool list a live MCP endpoint advertises (full URL, e.g. http://127.0.0.1:8765/mcp)."""
    import asyncio

    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async def go() -> list[dict[str, Any]]:
        http = httpx2.AsyncClient(timeout=60)
        async with Client(streamable_http_client(url, http_client=http), mode="legacy") as c:
            res = await c.list_tools()
        return [t.model_dump(by_alias=True, exclude_none=True) for t in res.tools]

    return asyncio.run(go())


def skill_description(text: str) -> str:
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def anthropic_counter(model: str = "claude-sonnet-5-5") -> Counter:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("count_tokens needs `uv run --with anthropic`") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("count_tokens needs ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()

    def count(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
              system: str | None = None) -> int:  # fmt: skip
        kw: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kw["tools"] = tools
        if system:
            kw["system"] = system
        return client.messages.count_tokens(**kw).input_tokens

    return count


def measure(
    count: Counter, *, tools: list[dict[str, Any]] | None = None, skill_text: str | None = None,
    core_block: str | None = None, r5_applied: bool = False, method: str = "count_tokens",
    model: str = "",
) -> dict[str, Any]:  # fmt: skip
    """The B-C6 record. `method` other than `count_tokens` (an estimate) cannot pass the gate."""
    skill_text = skill_text if skill_text is not None else SKILL.read_text(encoding="utf-8")
    tools = tools if tools is not None else contracts.mcp_tools()["tools"]
    store_param = has_store_param(tools)
    servers = R5_SERVERS if r5_applied and store_param else SERVERS
    defs = tool_defs(tools, servers)
    msgs = [{"role": "user", "content": PROMPT}]
    base = count(messages=msgs)
    with_tools = count(messages=msgs, tools=defs)
    desc = skill_description(skill_text)
    system_full = skill_text + (("\n\n" + core_block) if core_block else "")
    full = count(messages=msgs, tools=defs, system=system_full)
    only_skill = count(messages=msgs, system=skill_text) - base
    desc_only = count(messages=msgs, tools=defs, system=desc) - base if desc else with_tools - base
    return {
        "kind": "c6-run", "method": method, "model": model, "prompt": PROMPT,
        "tools_n": len(defs), "servers": list(servers), "store_param": store_param,
        "r5_applied": r5_applied,
        "baseline_tokens": base,
        "tools_tokens": with_tools - base,
        "skill_tokens": only_skill,
        "core_block_tokens": (full - count(messages=msgs, tools=defs, system=skill_text))
        if core_block else 0,
        "session_tokens": full - base,
        "session_tokens_description_only": desc_only,
        "budget": BUDGET,
    }  # fmt: skip
