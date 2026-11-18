"""Relation-phrase match (MEM2-171, B-R2 (ii)): does the memory say something about the question's
*predicate*, not merely about its subject?

The subject-masked signals (`mem.subjectmask`) still compare whole sentences, so a memory about the
right person with the wrong predicate keeps a high score from everything around the name. This
module reduces both texts to their predicate phrase by generic text processing: the subject's
names are removed, then question words, determiners, pronouns, auxiliaries, prepositions and
conjunctions (`STOP`, English function words only: no cue list, template, relation id or value
list anywhere), and numbers. What is left ("favourite food"; "works dentist") is compared two ways:

- `phrase_cosine`: the embedder's cosine of the question phrase (query prompt) and the memory
  phrase (document prompt, its `ctx` and text each reduced), embedded at scoring time;
- `phrase_overlap`: the share of the question phrase's stems found among the memory phrase's
  stems (`stem`: a light suffix stemmer so "lives", "lived", "living" and "live" agree).

A question with no content words left (nothing but function words around the name) has no
phrase: the cosine falls back to the subject-masked texts and the overlap is 0. Deterministic,
and the same functions run at fit time (`calibrate.score_pool`) and at serve time
(`read.pool_probs`), both through `subjectmask.pool_masked`. The subject rule and this module's
constants are part of the fingerprint of a model that uses a phrase feature (`readmodel`).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from mem import catalog, subjectmask
from mem.common import norm

if TYPE_CHECKING:
    from mem.store import Store

VERSION = 1
MIN_WORD_CHARS = 2
# English function words: determiners, pronouns, question words, auxiliaries and copulas,
# prepositions, conjunctions, contraction remnants. Content words (nouns, verbs, adjectives,
# adverbs) are never listed here.
STOP = frozenset((
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "either",
    "neither", "no", "much", "many", "my", "your", "his", "her", "its", "our",
    "their", "mine", "yours", "hers", "ours", "theirs", "own", "i", "me", "you", "he", "him",
    "she", "it", "we", "us", "they", "them", "who", "whom", "whose", "what", "which", "where",
    "when", "why", "how", "whoever", "whatever", "whichever", "am", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "doing", "done", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "of", "in", "on",
    "at", "to", "for", "from", "by", "with", "about", "into", "onto", "over", "under", "between",
    "among", "through", "during", "before", "after", "above", "below", "up", "down", "off", "out",
    "as", "than", "per", "via", "and", "or", "but", "nor", "so", "yet", "if", "then", "because",
    "while", "whether", "although", "though", "unless", "until", "not", "also", "just", "very",
    "too", "only", "same", "such", "there", "here", "s", "t", "d", "m", "re", "ve", "ll",
))  # fmt: skip
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_VOWELS = set("aeiou")


def stem(word: str) -> str:
    """A light, deterministic suffix stemmer: possessive, plural, -ing, -ed, a doubled final
    consonant, a final -e. Enough to join the inflections of one predicate; never a dictionary."""
    w = word.casefold()
    for suf in ("'s", "’s"):
        w = w.removesuffix(suf)
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in _VOWELS and w[-1] not in "lsz":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def strip_names(text: str, names: Iterable[str]) -> str:
    """`text` without any whole-word, case-insensitive occurrence of `names` (longest first)."""
    uniq = sorted({n.strip() for n in names if len(n.strip()) >= subjectmask.MIN_NAME_CHARS},
                  key=lambda n: (-len(n), n.casefold(), n))  # fmt: skip
    if not uniq:
        return text
    body = "|".join(re.escape(n) for n in uniq)
    return re.sub(rf"(?<!\w)(?:{body})(?!\w)", " ", text, flags=re.IGNORECASE)


def content_words(text: str, names: Iterable[str]) -> list[str]:
    """The predicate phrase of `text` as words, in order: subject names removed, normalised,
    function words and tokens without a letter dropped."""
    out = []
    for w in _WORD.findall(norm(strip_names(text, names))):
        if len(w) < MIN_WORD_CHARS or w in STOP or not any(c.isalpha() for c in w):
            continue
        out.append(w)
    return out


def phrase(text: str, names: Iterable[str]) -> str:
    return " ".join(content_words(text, names))


def overlap(query_words: Sequence[str], doc_words: Sequence[str]) -> float:
    """Share of the question's stems present among the memory's stems (0 with no question
    phrase)."""
    q = {stem(w) for w in query_words}
    if not q:
        return 0.0
    d = {stem(w) for w in doc_words}
    return len(q & d) / len(q)


def rule() -> list:
    """What the phrase features depend on besides the model's numbers (fingerprint body)."""
    stop = hashlib.sha256("\n".join(sorted(STOP)).encode("utf-8")).hexdigest()
    return [VERSION, MIN_WORD_CHARS, stop]


# ---- from the store -----------------------------------------------------------------------------


def query_phrase(store: Store, query: str) -> str:
    return phrase(query, _query_names(store, query))


def _query_names(store: Store, query: str) -> list[str]:
    return subjectmask.entity_names(store.conn, subjectmask.query_subject_ids(store.conn, query))


def _doc_words(store: Store, chunk_ids: Sequence[int]) -> list[tuple[list[str], list[str]]]:
    """(ctx words, text words) of each chunk with its own subjects' names removed."""
    conn, out = store.conn, []
    cache: dict[int, list[str]] = {}
    for cid in chunk_ids:
        node_id, ctx, text = conn.execute(
            "SELECT node_id, coalesce(ctx, ''), text FROM chunks WHERE id = ?", (cid,)
        ).fetchone()  # fmt: skip
        if node_id not in cache:
            cache[node_id] = subjectmask.entity_names(
                conn, subjectmask.chunk_subject_ids(conn, node_id)
            )
        out.append((content_words(ctx, cache[node_id]), content_words(text, cache[node_id])))
    return out


def pool_phrase(
    store: Store, query: str, chunk_ids: Sequence[int], want: Iterable[str]
) -> dict[str, list[float]]:
    """`phrase_cosine` and/or `phrase_overlap` of one pool."""
    from mem.store import doc_text

    want = set(want)
    out: dict[str, list[float]] = {}
    if not chunk_ids:
        return {k: [] for k in ("phrase_cosine", "phrase_overlap") if k in want}
    q_words = content_words(query, _query_names(store, query))
    docs = _doc_words(store, chunk_ids)
    if "phrase_cosine" in want:
        q_text = " ".join(q_words) or subjectmask.masked_query(store, query)
        q = np.asarray(
            store.embedder.embed_query(catalog.query_prompt(store.embedder) + q_text),
            dtype=np.float64,
        )  # fmt: skip
        masked = subjectmask.masked_docs(store, chunk_ids)
        texts = [
            doc_text(" ".join(c), " ".join(t), store.embedder)
            if c or t
            else doc_text(mc, mt, store.embedder)
            for (c, t), (mc, mt) in zip(docs, masked, strict=True)
        ]
        cos = []
        for v in store.embedder.embed_docs(texts):
            v = np.asarray(v, dtype=np.float64)
            d = float(np.linalg.norm(q) * np.linalg.norm(v))
            cos.append(float(np.dot(q, v)) / d if d else 0.0)
        out["phrase_cosine"] = cos
    if "phrase_overlap" in want:
        out["phrase_overlap"] = [overlap(q_words, [*c, *t]) for c, t in docs]
    return out
