"""B-W5 alignment rules (benchmarks-v2 B-W5, A17): the spec's exact cases, one test per rule."""

import pytest

from bench.synth.align import Fact, aligned_facts, aligns, mentions_marker, sentences

LONDON = Fact("P1", "lives_in", "London")
TIMELINE = {"London", "Paris"}


def al(text, about=("P1",), fact=LONDON, timeline=TIMELINE):
    return aligns(text, set(about), fact, timeline)


def test_spec_positive_case():
    assert al("Ana lives in London")


@pytest.mark.parametrize(
    "text",
    [
        "Ana has never lived in London",
        "Ana no longer lives in London",
        "Ana moved away from London",
        "Ana's favourite city is London",
        "Ana used to live in London",
        "Ana lives in Paris",
        "Ana left Paris and now lives in London",
    ],
)
def test_spec_negative_cases(text):
    assert not al(text)


def test_rule1_subject_must_be_in_about():
    assert not al("Ana lives in London", about=("P2",))
    assert not al("Ana lives in London", about=())


def test_rule2_value_is_a_whole_token_match():
    assert not al("Ana lives in Londonderry")
    assert al("Ana lives in LONDON.")  # casefold
    assert al("Ana lives in London")  # NFKC folds the no-break space


def test_rule3_relation_cue_in_the_same_sentence_as_the_value():
    assert not al("Ana is nice. She lives in town. London is big.")
    assert al("Ana is nice. She lives in London.")


def test_rule3_a_cue_of_another_relation_in_the_sentence_blocks():
    assert not al("Ana works at Brightwell and lives in London")  # works_at cue present


@pytest.mark.parametrize(
    "cue", ["lives in", "lived in", "living in", "based in", "moved to", "home is", "resident of"]
)
def test_all_cues_of_the_relation_align(cue):
    assert al(f"Ana {cue} London")


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Ana isn't living in London", False),  # n't -> not
        ("Ana formerly lived in London", False),
        ("Ana is an ex resident of London", False),
        ("Ana previously lived in London", False),
        ("Ana quit living in London", False),
        ("Ana lives in London until May", True),  # `until` after the value is outside the window
        ("Ana lives in London", True),
    ],
)
def test_rule4_negation_window(text, ok):
    assert al(text) is ok


def test_rule4_window_is_three_tokens_before_the_first_of_cue_and_value():
    assert not al("Not really, Ana lives in London")  # `not` is 3 tokens before the cue
    assert al("Not at all really, Ana lives in London")  # 5 tokens before: outside


def test_rule5_exclusive_over_the_slot_timeline():
    assert not al("Ana lives in London, Paris is nice", timeline={"London", "Paris"})
    assert al("Ana lives in London, Paris is nice", timeline={"London"})


def test_aligned_facts_returns_every_matching_gold_fact():
    works = Fact("P1", "works_at", "Brightwell")
    tl = {("P1", "lives_in"): {"London"}, ("P1", "works_at"): {"Brightwell"}}
    got = aligned_facts("Ana lives in London.", {"P1"}, [LONDON, works], tl)
    assert got == [LONDON]
    both = aligned_facts(
        "Ana lives in London. Ana works at Brightwell.", {"P1"}, [LONDON, works], tl
    )
    assert both == [LONDON, works]  # non-atomic memory: two facts


def test_sentences_split_and_normalise():
    assert sentences("Ana lives in London. It's NICE!") == [
        ["ana", "lives", "in", "london"], ["it", "s", "nice"]]  # fmt: skip


def test_ignore_marker_is_a_token_match():
    assert mentions_marker("I tried the zorbali sandwich place", "zorbali")
    assert not mentions_marker("I tried the zorbalis", "zorbali")
