"""B-W5 agent replay (benchmarks-v2 B-W5): a writer model with the production skill saves what it
decides to save from each dialogue, through the real tools (`recall`, `remember`, `link`,
`subjects`). The harness answers the writer's questions to the user from gold
(`bench.synth.dialogue.gold_answer`). Everything is recorded: per call usage and cost (B-C5), every
tool call with its result, and the final store, so every metric can be recomputed offline from the
run file (`bench.suites.b_w5.score_run`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bench.llm import LLM, BudgetExceeded, Ledger
from bench.suites.base import Client
from bench.synth.dialogue import WRAPPER, Dialogue, gold_answer
from bench.synth.generator import Synth
from mem import contracts

TOOLS = ("recall", "remember", "link", "subjects")
MAX_TURNS = 14
MAX_QUESTIONS = 2
MAX_TOKENS = 4096
SKILL = Path(__file__).resolve().parents[1] / "skills" / "mem" / "SKILL.md"


def tool_schemas() -> list[dict[str, Any]]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]}
            for t in contracts.mcp_tools()["tools"] if t["name"] in TOOLS]  # fmt: skip


def skill_text() -> tuple[str, str]:
    text = SKILL.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()


def snapshot_store(db: Path) -> dict[str, Any]:
    """What the writer left behind: memories (text, status, interval), entities with their live
    `about` edges and aliases, and the live `supersedes` / `duplicate_of` edges."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        mem = [dict(zip(("id", "text", "status", "valid_from", "valid_to", "recorded_at", "type"),
                        r, strict=True)) for r in conn.execute(
            "SELECT id, body, status, valid_from, valid_to, recorded_at, type FROM nodes "
            "WHERE kind='memory' ORDER BY id")]  # fmt: skip
        about: dict[int, list[int]] = {}
        for src, dst in conn.execute("SELECT src, dst FROM edges WHERE rel='about' AND "
                                     "removed_at IS NULL ORDER BY src, dst"):  # fmt: skip
            about.setdefault(src, []).append(dst)
        ents = [{"id": r[0], "type": r[1], "name": r[2]} for r in conn.execute(
            "SELECT id, type, name FROM nodes WHERE kind='entity' ORDER BY id")]  # fmt: skip
        aliases = [{"node": r[0], "alias": r[1]} for r in conn.execute(
            "SELECT node_id, alias FROM aliases ORDER BY node_id, alias")]  # fmt: skip
        links = [{"src": r[0], "dst": r[1], "rel": r[2]} for r in conn.execute(
            "SELECT src, dst, rel FROM edges WHERE rel IN ('supersedes','duplicate_of') "
            "AND removed_at IS NULL ORDER BY id")]  # fmt: skip
    finally:
        conn.close()
    for m in mem:
        m["about"] = about.get(m["id"], [])
    return {"memories": mem, "entities": ents, "aliases": aliases, "links": links}


def _est_tokens(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode()) // 3


def run_dialogue(
    dl: Dialogue, syn: Synth, client: Client, llm: LLM, ledger: Ledger, system: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:  # fmt: skip
    """One dialogue: the writer loop until it stops (or MAX_TURNS). Returns the dialogue's record."""
    from datetime import datetime, timedelta

    start = datetime.strptime(dl.start, "%Y-%m-%dT%H:%M:%SZ")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"<conversation>\n{dl.transcript}\n</conversation>\n{WRAPPER.format(date=dl.date)}"
            ),
        }
    ]
    calls: list[dict[str, Any]] = []
    step = questions = 0
    stop = "max_turns"
    error = None
    for turn in range(MAX_TURNS):
        try:
            ledger.check(MAX_TOKENS, _est_tokens(messages) + _est_tokens(tools))
        except BudgetExceeded as e:
            stop, error = "budget", str(e)
            break
        reply = llm.create(system=system, messages=messages, tools=tools, max_tokens=MAX_TOKENS)
        ledger.add(reply.usage, f"d{dl.index}.t{turn}")
        messages.append({"role": "assistant", "content": reply.content})
        uses = [b for b in reply.content if b.get("type") == "tool_use"]
        if reply.stop_reason == "refusal":
            stop = "refusal"
            break
        if not uses:
            text = " ".join(b.get("text", "") for b in reply.content if b.get("type") == "text")
            if "?" in text and questions < MAX_QUESTIONS:
                questions += 1
                messages.append({"role": "user", "content": gold_answer(text, dl, syn)})
                continue
            stop = "end_turn"
            break
        results = []
        for b in uses:
            step += 1
            now = (start + timedelta(seconds=step)).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                r = client.call(b["name"], b["input"], now=now)
                out, structured, is_err = r.text, r.structured, False
            except Exception as e:  # noqa: BLE001 - a tool error goes back to the writer
                out, structured, is_err = f"{type(e).__name__}: {e}", {}, True
            calls.append({"turn": turn, "tool": b["name"], "input": b["input"], "now": now,
                          "text": out, "structured": structured, "error": is_err})  # fmt: skip
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out,
                            **({"is_error": True} if is_err else {})})  # fmt: skip
        messages.append({"role": "user", "content": results})
    return {"dialogue": dl.index, "stop": stop, "error": error, "turns": turn + 1,
            "questions": questions, "calls": calls}  # fmt: skip


def run_all(
    dialogues: list[Dialogue], syn: Synth, factory: Callable[..., Client], db: Path, llm: LLM,
    budget_usd: float, tz: str,
) -> dict[str, Any]:  # fmt: skip
    """All dialogues in date order into ONE fresh store; returns the run record."""
    system, skill_sha = skill_text()
    tools = tool_schemas()
    ledger = Ledger(llm.model, budget_usd)
    client = factory(db, scope="personal", tz=tz, now=dialogues[0].start)
    runs = []
    try:
        for dl in dialogues:
            rec = run_dialogue(dl, syn, client, llm, ledger, system, tools)
            runs.append(rec)
            if rec["stop"] == "budget":
                break
    finally:
        client.close()
    return {"snapshot": snapshot_store(db), "model": llm.model, "skill_sha256": skill_sha, "tools": [t["name"] for t in tools],
            "dialogues": runs, "ledger": ledger.calls, "cost": ledger.summary(),
            "complete": len(runs) == len(dialogues) and all(r["stop"] != "budget" for r in runs)}  # fmt: skip
