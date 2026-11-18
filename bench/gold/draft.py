"""`bench gold draft`: a Claude model drafts candidate questions per note or note pair.

Human-run: it sends note text from the vault to the Anthropic API (needs `ANTHROPIC_API_KEY`,
`uv run --extra llm`, a hard `--budget-usd`; B-C5 ledger). Tests use a scripted fake with the same
`LLM.create` shape. A draft is a proposal: the person accepts, edits or rejects it in
`bench gold review`, and nothing here reaches the frozen files without that.

Each draft: draft_id, unit, category, question, answer (short), gold [{id, rel}] chosen only from
the chunk ids shown in the prompt, filters, and for no-answer drafts the `key_terms` that the
no-answer check scans for. Drafts that do not parse or cite an unknown chunk are counted in the
report, never silently dropped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import llm
from bench.gold import pg
from bench.gold.chunks import Chunk, ChunkIndex
from bench.gold.sampling import Unit

MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000
CHUNK_CHARS = 1800  # a chunk longer than this is cut in the prompt (the gold id is still whole)

CATEGORY_PROMPTS = {
    "single-fact": "one question whose answer is a single fact stated in ONE chunk. Reword it: do "
    "not copy the chunk's phrases.",
    "multi-note": "one question that needs facts from BOTH notes (one chunk of each) to answer.",
    "temporal": "one question about WHEN something happened or how long ago / how long it lasted, "
    "answerable from one chunk; give the dates in the answer.",
    "knowledge-update": "one question about the CURRENT value of something that the vault states "
    "both in an older and a newer statement (two chunks, in one or both notes). gold: the newer "
    "chunk rel 2, the older rel 1. Skip this slot (return no draft for it) if the notes hold no "
    "such pair.",
    "alias-entity": "one question that names a person, project or thing by an alias, nickname or "
    "partial name that the note itself does not use in the answer chunk (it appears in the note's "
    "title, aliases or another section).",
    "relational": "one question about how the two notes' subjects are related (who reports to "
    "whom, which project a person is on, what links them); gold: the chunks that state the link.",
    "no-answer-near-miss": "one question that is on the topic of the note and sounds answerable "
    "but whose answer is NOT in the vault (a detail the note never gives). answer: ''. gold: []. "
    "key_terms: 2-4 specific terms an answering passage would contain.",
    "no-answer-off-topic": "one question in the person's domain that the vault does not cover at "
    "all (a different topic from these notes). answer: ''. gold: []. key_terms: 2-4 specific "
    "terms an answering passage would contain.",
}
SYSTEM = (
    "You write evaluation questions for a retrieval benchmark over a person's private Obsidian "
    "vault. Questions are what the owner would ask their own notes, in their own voice, short and "
    "natural. Never quote a chunk verbatim as a question. Use ONLY the chunk ids you are shown. "
    "Reply with one JSON object and nothing else: "
    '{"drafts": [{"slot": <int>, "category": str, "question": str, "answer": str, '
    '"gold": [{"id": str, "rel": 2|1}], "filters": {"about": [str], "type": [str]}, '
    '"key_terms": [str]}]}. rel 2 = the chunk answers the question, rel 1 = it supports it. '
    "filters.about is allowed only for an entity or note title that the question names. "
    "Leave filters empty unless the question names one."
)


def unit_prompt(unit: Unit, index: ChunkIndex) -> str:
    parts = []
    for path in unit.paths:
        parts.append(f"## Note: {path}")
        for c in index.of_path(path):
            body = c.text if len(c.text) <= CHUNK_CHARS else c.text[:CHUNK_CHARS] + " […]"
            parts.append(f"### chunk id: {c.gold_id}\n{body}")
    slots = "\n".join(f"{i}. category `{cat}`: {CATEGORY_PROMPTS[cat]}"
                      for i, cat in enumerate(unit.slots))  # fmt: skip
    return "\n\n".join(parts) + f"\n\nDraft one question for each slot:\n{slots}"


@dataclass
class DraftReport:
    drafts: list[dict[str, Any]]
    parse_failures: int
    unknown_chunk: int
    skipped_slots: int
    by_category: dict[str, int]
    cost: dict[str, Any]


def _json_object(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


def parse_reply(unit: Unit, text: str, shown: set[str]) -> tuple[list[dict[str, Any]], int, int]:
    """(valid drafts, parse failures, drafts citing an unknown chunk)."""
    try:
        raw = _json_object(text)["drafts"]
    except (ValueError, KeyError, TypeError):
        return [], 1, 0
    out, bad = [], 0
    for d in raw:
        try:
            slot = int(d["slot"])
            cat = unit.slots[slot]
            if d.get("category", cat) != cat:
                raise ValueError("category differs from the slot")
            gold = {g["id"]: int(g["rel"]) for g in d.get("gold", [])}
            q = str(d["question"]).strip()
        except (KeyError, ValueError, TypeError, IndexError):
            return out, 1, bad
        if set(gold) - shown:
            bad += 1
            continue
        if cat in pg.NO_ANSWER:
            gold = {}
        filters = {k: v for k, v in (d.get("filters") or {}).items()
                   if k in pg.FILTER_KEYS and v}  # fmt: skip
        out.append({"unit": unit.uid, "slot": slot, "category": cat, "question": q,
                    "answer": str(d.get("answer", "")).strip(), "gold": gold, "filters": filters,
                    "key_terms": [str(t) for t in d.get("key_terms", [])],
                    "source": list(unit.paths), "store": unit.store})  # fmt: skip
    return out, 0, bad


def draft_all(
    units: list[Unit], index: ChunkIndex, client: llm.LLM, ledger: llm.Ledger,
    *, existing: list[dict[str, Any]] | None = None,
) -> DraftReport:  # fmt: skip
    """One call per unit (resumable: units with drafts in `existing` are kept, not re-asked)."""
    done = {d["unit"] for d in existing or []}
    drafts = list(existing or [])
    fails = unknown = skipped = 0
    for unit in units:
        if unit.uid in done:
            continue
        prompt = unit_prompt(unit, index)
        ledger.check(MAX_TOKENS, len(SYSTEM + prompt) // 3)
        reply = client.create(system=SYSTEM, messages=[{"role": "user", "content": prompt}],
                              tools=[], max_tokens=MAX_TOKENS)  # fmt: skip
        ledger.add(reply.usage, f"draft:{unit.uid}")
        text = "".join(b.get("text", "") for b in reply.content if b.get("type") == "text")
        shown = {c.gold_id for p in unit.paths for c in index.of_path(p)}
        got, f, u = parse_reply(unit, text, shown)
        fails, unknown = fails + f, unknown + u
        skipped += len(unit.slots) - len(got)
        for d in got:
            d["draft_id"] = f"{unit.uid}-{d['slot']}"
        drafts += got
    counts = Counter(d["category"] for d in drafts)
    return DraftReport(drafts, fails, unknown, skipped, dict(counts), ledger.summary())


def write_drafts(path: Path, drafts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n" for d in drafts),
                    encoding="utf-8")  # fmt: skip


def read_drafts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def chunk_of(index: ChunkIndex, gid: str) -> Chunk:
    return index.by_gold[gid]
