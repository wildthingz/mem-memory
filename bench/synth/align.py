"""B-W5 alignment (benchmarks-v2 B-W5, A17): is a saved memory the gold fact (s, r, v)?

Deterministic, relation- and negation-aware. Text is normalised (NFKC, casefold, punctuation ->
space; `n't` is expanded to `not` first so contractions are negations) and split into sentences and
tokens. A memory aligns to gold fact (s, r, v) iff ALL hold:

1. Subject: the memory's `about` contains entity s.
2. Value: some sentence contains v as a whole-token match.
3. Relation: that same sentence contains a cue c of cues(r) and no cue of another relation.
4. No negation: no phrase of NEGATIONS occurs between c and v, or within the 3 tokens before
   whichever of c and v comes first. (c = the cue occurrence nearest to v.)
5. Exclusive: the text contains no other value of the same slot's timeline.

The versioned lists live in `bench.synth.data` (cues, NEGATIONS); the audit sheet of the P3 gate
(`bench.suites.b_w5`) is how a human checks them (more than 1 wrong of 30 aligned fails the gate).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from bench.synth import data as D

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
NEGATION_WINDOW = 3


@dataclass(frozen=True)
class Fact:
    subject: str  # entity key (e.g. "P3")
    relation: str
    value: str


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"n't\b", " not", t)


def sentences(text: str) -> list[list[str]]:
    """Normalised token lists, one per sentence."""
    out = []
    for s in _SENT.split(text.strip()):
        toks = _WORD.findall(_norm(s))
        if toks:
            out.append(toks)
    return out


def tokens(text: str) -> list[str]:
    return _WORD.findall(_norm(text))


def find(hay: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(hay) - n + 1) if hay[i : i + n] == needle] if n else []


def _phrase_tokens(phrases: Iterable[str]) -> list[list[str]]:
    out = []
    for p in phrases:
        t = tokens(p)
        if t and t != ["n", "t"]:  # `n't` is covered by the contraction expansion
            out.append(t)
    return out


_NEG = _phrase_tokens(D.NEGATIONS)
_CUES = {rel: _phrase_tokens(cs) for rel, cs in D.all_cues().items()}


def _has_phrase(window: list[str], phrases: list[list[str]]) -> bool:
    return any(find(window, p) for p in phrases)


def aligns(
    text: str, about: Collection[str], fact: Fact, timeline: Collection[str] = (),
) -> bool:  # fmt: skip
    """`about`: the entity keys on the memory's `about` edges. `timeline`: every value of the
    slot (s, r), including fact.value (rule 5)."""
    if fact.subject not in about:  # rule 1
        return False
    v_tok = tokens(fact.value)
    others = [tokens(o) for o in timeline if o != fact.value]
    all_sent = sentences(text)
    flat = [t for s in all_sent for t in s]
    if any(find(flat, o) for o in others if o):  # rule 5
        return False
    for toks in all_sent:
        vpos = find(toks, v_tok)
        if not vpos:  # rule 2
            continue
        cue_pos = [(i, len(c)) for c in _CUES[fact.relation] for i in find(toks, c)]
        if not cue_pos:  # rule 3
            continue
        if any(find(toks, c) for rel, cs in _CUES.items() if rel != fact.relation for c in cs):
            continue
        v = vpos[0]
        c, _ = min(cue_pos, key=lambda p: abs(p[0] - v))
        first, second = (c, v) if c < v else (v, c)
        window = toks[max(0, first - NEGATION_WINDOW) : second]  # rule 4
        if _has_phrase(window, _NEG):
            continue
        return True
    return False


def aligned_facts(
    text: str, about: Collection[str], facts: Iterable[Fact], timelines: Mapping[tuple[str, str],
                                                                               Collection[str]],
) -> list[Fact]:  # fmt: skip
    return [f for f in facts if aligns(text, about, f, timelines.get((f.subject, f.relation), ()))]


def mentions_marker(text: str, marker: str) -> bool:
    """`ignore` items of a dialogue (small talk with a unique marker string)."""
    return bool(find(tokens(text), tokens(marker)))
