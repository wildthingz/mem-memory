"""MEM2-114 re-review proofs for the MEM2-113 fixes (F7, F13 of MEM2-112).

F14 (new, minor): the F13 fix shares one `_NOUN` list between the first `override` alternative
(`(ignore|forget|...) (previous|prior|earlier|original|...) NOUN`) and the new anchored
`before/earlier` alternative. Adding `conversation` to that list for the second use also widens
the first: `forget the previous conversation`, `ignore the earlier conversation` now quarantine
first-hand text that addresses no assistant and forges no prompt. On 82540d1 (before the fix)
these were saved. Fails on bosmang/mem2-113 e49eeef.
"""

import pytest

from mem import guard

ATLAS = "new:project:Atlas"


@pytest.mark.parametrize(
    "text",
    [
        "Alice wants to forget the previous conversation with his landlord.",
        "Ana would rather forget the earlier conversation about the move.",
        "Alice wants to ignore the previous conversation about rent and renegotiate.",
    ],
)
def test_first_hand_recollection_of_a_conversation_is_not_quarantined(make_store, text):
    """F14: a person forgetting or ignoring a past conversation is a memory about them, not an
    instruction to the assistant (`guard` module docstring); the widened `_NOUN` quarantines it."""
    assert guard.find_injection(text, tool_sourced=False) is None, text
    assert (
        make_store().remember(text=text, type="fact", about=[ATLAS]).structured["status"] == "saved"
    )
