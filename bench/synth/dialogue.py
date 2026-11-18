"""B-W5 data: 40 dialogues rendered from a SYN-M world (benchmarks-v2 B-W5).

Eligible ops: `new_fact`, `compatible_add`, and `update` of sub-type `normal` (value_returns are
excluded by the spec, so every value maps to one gold fact; late, future-dated and same-instant
updates cannot be expressed by a dialogue's single date and are left out; dup/retract/ambiguous ops
are not save-worthy facts). Facts are grouped into consecutive dialogues of 3-12 facts, with small
talk (`ignore` items, each with a unique marker word) between them.

The gold of the dialogue world is recomputed from the included facts only: an update whose
predecessor was not included is a plain new fact. `conflict` marks an update whose predecessor is a
gold fact of an earlier dialogue (the writer is expected to `link ... supersedes`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from bench.synth import data as D
from bench.synth.align import Fact
from bench.synth.generator import Synth, tokens

N_DIALOGUES = 40
FACTS_MIN, FACTS_MAX = 3, 12
WRAPPER = "This conversation just ended on {date}. Follow your memory skill."
SMALL_TALK = (
    "By the way, I finally tried the {m} sandwich place near the station.",
    "Unrelated, but my {m} playlist has been on repeat all week.",
    "I keep meaning to repaint the shed in {m} green.",
    "Funny story: the {m} bus was late again this morning.",
)
ASSISTANT_TURNS = ("Got it.", "Understood.", "Thanks for telling me.", "Okay, noted.")


@dataclass
class Dialogue:
    index: int
    date: str  # YYYY-MM-DD
    start: str  # UTC timestamp the harness clock starts at
    facts: list[
        dict[str, Any]
    ]  # {op, subject, relation, value, conflict_with (fact index or None)}
    ignores: list[str]
    transcript: str
    entities: list[str] = field(default_factory=list)


def _markers(rng: random.Random, taken: set[str], n: int) -> list[str]:
    out = []
    while len(out) < n:
        w = "".join(rng.choice(D.SYLLABLES) for _ in range(3))
        if w not in taken and w.capitalize() not in taken:
            taken.add(w)
            out.append(w)
    return out


def build_dialogues(syn: Synth, n: int = N_DIALOGUES, seed: int | None = None) -> list[Dialogue]:
    rng = random.Random((syn.seed * 7 + 1) if seed is None else seed)
    ents = {e["key"]: e for e in syn.expected["entities"]}
    eligible = [op for op in syn.ops if (
        op["kind"] in ("new_fact", "compatible_add")
        or (op["kind"] == "update" and op["gold"]["subtype"] == "normal"))]  # fmt: skip
    taken = {t for e in ents.values() for t in tokens(e["name"])}
    taken |= {t for m in syn.expected["memories"].values() for t in tokens(m["value"])}
    # Whole entities, so slot timelines (and with them the updates) stay intact: entities are drawn
    # in random order until >= n * 7 facts, the stream is cut to <= n * FACTS_MAX facts, then split
    # into n dialogues of FACTS_MIN..FACTS_MAX facts.
    by_ent: dict[str, list[dict[str, Any]]] = {}
    for op in eligible:
        by_ent.setdefault(syn.expected["memories"][str(op["i"])]["entity"], []).append(op)
    keys = sorted(by_ent)
    rng.shuffle(keys)
    chosen: list[dict[str, Any]] = []
    for k in keys:
        if len(chosen) >= n * 7:
            break
        chosen += by_ent[k]
    chosen.sort(key=lambda o: o["i"])
    chosen = chosen[: n * FACTS_MAX]
    if len(chosen) < n * FACTS_MIN:
        raise ValueError(f"SYN world has {len(chosen)} usable facts, {n * FACTS_MIN} needed")
    sizes = [FACTS_MIN] * n
    extra = len(chosen) - n * FACTS_MIN
    while extra > 0:
        i = rng.randrange(n)
        if sizes[i] < FACTS_MAX:
            sizes[i] += 1
            extra -= 1
    eligible = chosen
    last_in_slot: dict[str, tuple[int, int]] = {}  # slot -> (dialogue index, fact number)
    out: list[Dialogue] = []
    pos = 0
    day_count: dict[str, int] = {}
    for d, size in enumerate(sizes):
        ops = eligible[pos : pos + size]
        pos += size
        date = ops[0]["now"][:10]
        k = day_count.get(date, 0)
        day_count[date] = k + 1
        facts: list[dict[str, Any]] = []
        lines: list[tuple[str, str]] = []
        for op in ops:
            g = op["gold"]
            slot = syn.expected["memories"][str(op["i"])]
            ent = ents[slot["entity"]]
            rel = D.RELATIONS[slot["rel"]]
            cue = rng.choice(rel.cues)
            prefix = ""
            conflict = None
            if (
                not rel.multi and g["slot"] in last_in_slot
            ):  # an update, or a re-save after a retract
                conflict = list(last_in_slot[g["slot"]])
                prefix = rng.choice(("Update: ", "Now ", "Actually, "))
            subject = (
                f"the {ent['type']} {ent['name']}"  # a cross-kind pair shares one name
                if (ent["group"] or "").startswith("X")
                else ent["name"]
            )
            sentence = f"{prefix}{subject} {cue} {slot['value']}."
            facts.append({"op": op["i"], "subject": ent["key"], "relation": slot["rel"],
                          "value": slot["value"], "conflict_with": conflict,
                          "user_text": sentence})  # fmt: skip
            last_in_slot[g["slot"]] = (d, len(facts) - 1)
            lines.append(("fact", sentence))
        ignores = _markers(rng, taken, rng.randint(1, 2))
        for m in ignores:
            lines.append(("ignore", rng.choice(SMALL_TALK).format(m=m)))
        rng.shuffle(lines)
        transcript = "\n".join(
            f"User: {text}\nAssistant: {rng.choice(ASSISTANT_TURNS)}" for _, text in lines
        )
        out.append(
            Dialogue(
                d,
                date,
                f"{date}T{9 + k:02d}:00:00Z",
                facts,
                ignores,
                transcript,
                sorted({f["subject"] for f in facts}),
            )
        )
    return out


def gold_facts(dialogues: list[Dialogue]) -> list[tuple[int, int, Fact]]:
    return [(dl.index, i, Fact(f["subject"], f["relation"], f["value"]))
            for dl in dialogues for i, f in enumerate(dl.facts)]  # fmt: skip


def timelines(dialogues: list[Dialogue]) -> dict[tuple[str, str], set[str]]:
    tl: dict[tuple[str, str], set[str]] = {}
    for dl in dialogues:
        for f in dl.facts:
            tl.setdefault((f["subject"], f["relation"]), set()).add(f["value"])
    return tl


def gold_current(dialogues: list[Dialogue]) -> set[tuple[int, int]]:
    """(dialogue, fact) of the gold-current facts: the last value of each single-valued slot, and
    every fact of a multi-valued slot."""
    last: dict[tuple[str, str], tuple[int, int]] = {}
    cur: set[tuple[int, int]] = set()
    for dl in dialogues:
        for i, f in enumerate(dl.facts):
            key = (f["subject"], f["relation"])
            if D.RELATIONS[f["relation"]].multi:
                cur.add((dl.index, i))
            else:
                last[key] = (dl.index, i)
    return cur | set(last.values())


def gold_answer(question: str, dl: Dialogue, syn: Synth) -> str:
    """The harness answers a writer's question to the user from gold: it names the entities of this
    dialogue that the question mentions (canonical name and type); else 'use your judgement'."""
    ents = {e["key"]: e for e in syn.expected["entities"]}
    q = set(tokens(question))
    hits = []
    for k in dl.entities:
        e = ents[k]
        names = [e["name"], e.get("alias") or "", e.get("first") or ""]
        if any(n and set(tokens(n)) <= q for n in names):
            hits.append(f"{e['name']} is the {e['type']} I mean")
    return ("; ".join(hits) + ". Please go ahead and save." if hits
            else "Use your best judgement and go ahead.")  # fmt: skip
