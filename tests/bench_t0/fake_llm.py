"""Scripted writers for the B-W5 harness tests (no network). They play the writer model's side of
`bench.llm.LLM.create`: given the transcript they decide what to save, so a test can build a run
file for a perfect, a lazy, a noisy or a conflict-blind writer and check what B-W5 reports."""

from __future__ import annotations

import itertools
from typing import Any

from bench.llm import Reply
from bench.synth import data as D
from bench.synth.dialogue import Dialogue

_ids = itertools.count(1)


def _use(name: str, **inp: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"tu_{next(_ids)}", "name": name, "input": inp}


class Writer:
    """mode: perfect | lazy | noisy | no_link | wrong_rel | paraphrase_lossy"""

    model = "claude-sonnet-5-5"

    def __init__(self, dialogues: list[Dialogue], syn: Any, mode: str = "perfect") -> None:
        self.dialogues, self.mode = dialogues, mode
        self.ents = {e["key"]: e for e in syn.expected["entities"]}
        self.calls = 0

    def _dialogue(self, messages: list[dict[str, Any]]) -> Dialogue:
        first = messages[0]["content"]
        for dl in self.dialogues:
            if dl.facts[0]["user_text"] in first:
                return dl
        raise AssertionError("unknown dialogue")

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]], max_tokens: int) -> Reply:  # fmt: skip
        self.calls += 1
        usage = {"input_tokens": 3000, "output_tokens": 500, "cache_read_input_tokens": 6000,
                 "cache_creation_input_tokens": 0}  # fmt: skip
        dl = self._dialogue(messages)
        turn = sum(m["role"] == "assistant" for m in messages)
        if self.mode == "lazy":
            return Reply([{"type": "text", "text": "Nothing worth saving."}], "end_turn", usage)
        if turn == 0:
            blocks = []
            for f in dl.facts:
                ent = self.ents[f["subject"]]
                text = f["user_text"]
                if self.mode == "paraphrase_lossy":
                    text = f"{ent['name']} is somehow connected to {f['value']}."
                blocks.append(
                    _use(
                        "remember",
                        text=text,
                        type=D.RELATIONS[f["relation"]].mem_type,
                        about=[f"new:{ent['type']}:{ent['name']}"],
                    )
                )
            if self.mode == "noisy":
                for m in dl.ignores:
                    blocks.append(
                        _use(
                            "remember",
                            text=f"The user mentioned the {m} thing.",
                            type="insight",
                            about=["new:topic:Smalltalk"],
                        )
                    )
            return Reply(blocks, "tool_use", usage)
        if turn == 1 and self.mode != "no_link":
            uses = [b for b in messages[-2]["content"] if b["type"] == "tool_use"]
            results = messages[-1]["content"]
            blocks = []
            for f, res in zip(dl.facts, results, strict=False):
                if not f["conflict_with"]:
                    continue
                words = res["content"].replace("\n", " ").split()
                ids = [w for w in words if w.startswith("m") and w[1:].isdigit()]
                if len(ids) >= 2:
                    rel = "relates_to" if self.mode == "wrong_rel" else "supersedes"
                    blocks.append(_use("link", src=ids[0], rel=rel, dst=ids[1]))
            del uses
            if blocks:
                return Reply(blocks, "tool_use", usage)
        return Reply([{"type": "text", "text": "Saved."}], "end_turn", usage)
