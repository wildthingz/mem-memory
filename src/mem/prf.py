"""Option C, pseudo-relevance expansion terms (MEM2-216): the nouns and names the first pass's top
chunks share, to add to the query for a second FTS and vec pass.

Nouns (common and proper) count once per chunk and are kept when at least `min_chunks` of the top
chunks have them; NER names are kept from any one chunk. Terms already in the query are dropped.
Order: nouns by chunk count (descending) then alphabetically, then names the same way; the cap
keeps the head. No stop list and no cue list: the tagger decides what a noun is, and `fts_match`
drops the generic stopwords when the terms reach the FTS leg.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["MODEL", "NlpUnavailable", "Tagged", "expand_query", "expansion_terms", "spacy_tagger"]

MODEL = "en_core_web_sm"
_WORD = re.compile(r"\w+")


class NlpUnavailable(RuntimeError):
    """spaCy or its pinned model is not installed (extra `nlp`); nothing is downloaded at runtime."""


@dataclass(frozen=True)
class Tagged:
    nouns: list[str]  # lemmas of NOUN and PROPN tokens, lower-case
    names: list[str]  # named-entity texts, as written


Tagger = Callable[[str], Tagged]


def spacy_tagger() -> Tagger:
    """`en_core_web_sm` as a tagger; NlpUnavailable if spaCy or the model wheel is not installed."""
    try:
        import spacy

        nlp = spacy.load(MODEL, disable=["parser"])
    except (ImportError, OSError) as e:
        raise NlpUnavailable(
            f"spaCy model {MODEL} is not installed: `uv sync --extra nlp` (no runtime download)"
        ) from e

    def tag(text: str) -> Tagged:
        doc = nlp(text)
        return Tagged(
            nouns=[t.lemma_.casefold() for t in doc if t.pos_ in ("NOUN", "PROPN") and t.is_alpha],
            names=[e.text.strip() for e in doc.ents if e.text.strip()],
        )

    return tag


def _ranked(counts: Counter[str]) -> list[str]:
    return [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].casefold(), kv[0]))]


def expansion_terms(
    query: str, chunks: Sequence[str], tagger: Tagger, *, top: int = 10, min_chunks: int = 2,
    cap: int = 10,
) -> list[str]:  # fmt: skip
    """Expansion terms from the first `top` of `chunks` (pool order), at most `cap`."""
    in_query = {w.casefold() for w in _WORD.findall(query)}
    noun_df: Counter[str] = Counter()
    name_df: Counter[str] = Counter()
    for text in chunks[:top]:
        tagged = tagger(text)
        noun_df.update(set(tagged.nouns))
        name_df.update(set(tagged.names))
    nouns = Counter({t: n for t, n in noun_df.items() if n >= min_chunks and t not in in_query})
    names = Counter({t: n for t, n in name_df.items()
                     if not {w.casefold() for w in _WORD.findall(t)} <= in_query})  # fmt: skip
    out = _ranked(nouns)
    seen = {t.casefold() for t in out}
    for name in _ranked(names):
        if name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out[:cap]


def expand_query(query: str, terms: Sequence[str]) -> str:
    return " ".join([query, *terms])
