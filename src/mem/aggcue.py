"""The aggregation-cue detector (MEM2-193, ADR 0023): does the query ask to count, total or list
things that are spread over several memories ("how many", "how often", "which", "list", "all",
"total", "in total", "count" and close variants)?

Generic English function words only (rule 4): no dataset cue list, no question id, no template.
A cue is a word or phrase of the query, matched on word boundaries, so "account", "listen" and
"totally" are not cues. Date arithmetic ("how many days ago/between ...") is a single-answer
question, not an aggregation, and "at all" / "after all" / "all right" are idioms."""

from __future__ import annotations

import re

_UNIT = r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
# "which <plural noun>" or "which of": a set is asked for; "which was / which is / which book" is not
_PLURAL_WHICH = r"which\s+(?:of\b|(?!(?:is|was|has|does|this|its|us|class|business)\b)\w{2,}s\b)"

_CUES = re.compile(
    r"\b(?:"
    # "how many days ago / between / since / have passed": elapsed time, not a count of things
    r"how many(?! "
    + _UNIT
    + r" (?:have |had |has )?(?:ago|between|before|after|since|until|passed|elapsed|apart|from)\b)"
    r"|how often"
    r"|in total|total(?:s|ed|ing|ling)?"
    r"|altogether|combined"
    r"|count(?:s|ed|ing)?"
    r"|number of"
    r"|list(?:s|ed|ing)?"
    r"|" + _PLURAL_WHICH + r")\b"
    r"|(?<!\bat )(?<!\bafter )(?<!\babove )\ball\b(?! right\b)",
    re.IGNORECASE,
)


def cues(query: str) -> list[str]:
    """The cues found in `query`, lower-cased, in order of appearance (empty: not an aggregation)."""
    return [m.group(0).lower() for m in _CUES.finditer(query)]


def is_aggregation(query: str) -> bool:
    return bool(_CUES.search(query))
