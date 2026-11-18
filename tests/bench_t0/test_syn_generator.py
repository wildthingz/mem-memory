"""SYN generator (benchmarks-v2 §3.2): determinism, the golden file, and every constraint the spec
lists, checked on the generated worlds (dev and held-out seeds)."""

import itertools
import json
from collections import Counter
from pathlib import Path

import pytest

from bench.synth import data as D
from bench.synth.generator import generate, lev_within, local_midnight_utc, tokens

GOLDEN = Path(__file__).resolve().parents[2] / "bench" / "synth" / "golden" / "seed1_S.sha256"
SEEDS = (*D.DEV_SEEDS, *D.HELD_OUT_SEEDS)


@pytest.fixture(scope="module")
def worlds():
    return {(s, "S"): generate(s, "S") for s in SEEDS} | {(101, "M"): generate(101, "M")}


def test_dev_and_held_out_seeds_are_disjoint_and_as_specified():
    assert D.DEV_SEEDS == (1, 2, 3, 4, 5)
    assert D.HELD_OUT_SEEDS == (101, 102, 103, 104, 105)


def test_generation_is_deterministic_and_seeds_differ():
    a, b = generate(7, "S"), generate(7, "S")
    assert a.files() == b.files()
    assert generate(8, "S").sha256() != a.sha256()


def test_golden_file_seed1_size_S():
    """Drift catcher: any change to the data lists, templates or generator changes these hashes.
    Regenerate with `uv run bench syn-generate --seed 1 --size S` and commit WITH a version bump."""
    want = dict(line.split()[::-1] for line in GOLDEN.read_text().splitlines() if line.strip())
    assert generate(1, "S").sha256() == want


def test_world_sizes_and_op_counts(worlds):
    for (_, size), syn in worlds.items():
        spec = D.SIZES[size]
        assert len(syn.ops) == spec["ops"]
        types = Counter(e["type"] for e in syn.expected["entities"])
        assert types == {"person": spec["people"], "project": spec["projects"],
                         "org": spec["orgs"], "place": spec["places"]}  # fmt: skip
        counts = syn.expected["counts"]
        for kind, share in D.SHARES.items():
            if kind != "new_fact":
                assert counts.get(kind, 0) == int(share * spec["ops"]), kind
        assert counts["new_fact"] == spec["ops"] - sum(
            v for k, v in counts.items() if k != "new_fact")  # fmt: skip


def test_collisions_and_cross_kind_pairs(worlds):
    for (_, size), syn in worlds.items():
        spec = D.SIZES[size]
        ents = syn.expected["entities"]
        firsts = Counter(e["first"] for e in ents if e["type"] == "person")
        assert sum(1 for n in firsts.values() if n == 2) == spec["first_collisions"]
        assert all(n <= 2 for n in firsts.values())
        names = Counter(e["name"] for e in ents)
        assert sum(1 for n in names.values() if n == 2) == spec["cross_kind"]
        cross = [n for n, c in names.items() if c == 2]
        for n in cross:
            assert {e["type"] for e in ents if e["name"] == n} == {"project", "org"}


def test_no_two_canonical_names_within_levenshtein_2(worlds):
    for syn in worlds.values():
        strings = sorted({" ".join(tokens(e["name"])) for e in syn.expected["entities"]}
                         | {" ".join(tokens(e["alias"])) for e in syn.expected["entities"]
                            if e["alias"]})  # fmt: skip
        for i, a in enumerate(strings):
            for b in strings[i + 1 :]:
                assert not lev_within(a, b, 2), (a, b)


def test_typos_only_on_names_of_5_chars_or_more_and_one_edit(worlds):
    seen = 0
    for syn in worlds.values():
        names = {e["key"]: e["name"] for e in syn.expected["entities"]}
        for op in syn.ops:
            for m in op["mentions"]:
                if m["style"] == "typo":
                    seen += 1
                    assert len(names[m["entity"]]) >= 5
                    assert lev_within(m["surface"].lower(), names[m["entity"]].lower(), 1)
                    assert m["surface"] != names[m["entity"]]
    assert seen > 0


def test_slot_values_are_world_unique_and_absent_from_filler_and_cues(worlds):
    base = set()
    for r in D.RELATIONS.values():
        for c in r.cues:
            base |= set(tokens(c))
    for f in (*D.FILLER, *D.UPDATE_PREFIXES, *D.NEGATIONS):
        base |= set(tokens(f))
    for syn in worlds.values():
        names = {t for e in syn.expected["entities"] for t in tokens(e["name"])}
        mems = syn.expected["memories"]
        owner: dict[str, tuple[str, str]] = {}
        for m in mems.values():
            tok = tokens(m["value"])
            assert len(tok) == 1
            assert tok[0] not in base and tok[0] not in names, m["value"]
            # the same value may repeat only in the same slot (value_returns, paraphrase)
            assert owner.setdefault(tok[0], (m["slot"], m["value"]))[0] == m["slot"], m["value"]


def test_every_value_repeat_is_value_returns_or_a_duplicate(worlds):
    for syn in worlds.values():
        by_slot_value: dict[tuple[str, str], list[int]] = {}
        for k, m in syn.expected["memories"].items():
            by_slot_value.setdefault((m["slot"], m["value"]), []).append(int(k))
        for ops in by_slot_value.values():
            for later in ops[1:]:
                op = syn.ops[later]
                assert op["kind"] == "dup_paraphrase" or (
                    op["kind"] == "update" and op["gold"]["subtype"] == "value_returns"
                ), op


def test_value_returns_reuse_the_exact_earlier_text(worlds):
    n = 0
    for syn in worlds.values():
        for op in syn.ops:
            if op["kind"] == "update" and op["gold"]["subtype"] == "value_returns":
                n += 1
                earlier = syn.ops[op["gold"]["returns_to"]]
                assert op["args"]["text"] == earlier["args"]["text"]
                assert op["gold"]["value"] == earlier["gold"]["value"]
    assert n > 0


def test_update_subtypes_follow_the_spec_shares(worlds):
    for syn in worlds.values():
        n = syn.expected["counts"]["update"]
        sub = syn.expected["update_subtypes"]
        assert sub["value_returns"] == round(0.20 * n)
        assert sub["future"] == round(0.10 * n)
        assert sub["late"] == max(1, round(0.05 * n))
        assert sub["same_instant"] == max(1, round(0.05 * n))
        cue = sum(op["gold"]["has_cue"] for op in syn.ops if op["kind"] == "update")
        assert abs(cue - n / 2) <= 1  # half carry a change cue


def test_future_late_and_same_instant_dates(worlds):
    for syn in worlds.values():
        mems = syn.expected["memories"]
        for op in syn.ops:
            if op["kind"] != "update":
                continue
            sub = op["gold"]["subtype"]
            as_of = op["args"]["as_of"]
            counterpart = mems[str(op["gold"]["counterpart"])]
            cp_day = syn.ops[counterpart["op"]]["args"]["as_of"]
            if sub == "future":
                assert as_of > op["now"][:10]
            elif sub == "late":
                assert as_of < max(
                    m["valid_from"][:10] for m in mems.values() if m["slot"] == op["gold"]["slot"]
                )  # precedes a known value
            elif sub == "same_instant":
                assert as_of == cp_day or any(
                    m["slot"] == op["gold"]["slot"]
                    and syn.ops[m["op"]]["args"]["as_of"] == as_of
                    and m["op"] != op["i"]
                    for m in mems.values()
                )


def test_chains_are_total_orders_with_derived_intervals(worlds):
    for syn in worlds.values():
        mems = syn.expected["memories"]
        for key, s in syn.expected["slots"].items():
            if s["multi"]:
                assert all(mems[str(o)]["valid_to"] is None for o in s["chain"])
                continue
            chain = s["chain"]
            for a, b in itertools.pairwise(chain):
                assert mems[str(a)]["valid_to"] == mems[str(b)]["valid_from"], key
            if chain:
                assert mems[str(chain[-1])]["valid_to"] is None
                assert s["current"] == [chain[-1]]
            vf = [mems[str(o)]["valid_from"] for o in chain]
            assert vf == sorted(vf)


def test_gold_valid_from_is_local_midnight_in_the_configured_zone():
    assert local_midnight_utc("2026-06-03") == "2026-06-02T22:00:00Z"  # CEST
    assert local_midnight_utc("2026-01-05") == "2026-01-04T23:00:00Z"  # CET


def test_cue_lists_are_disjoint_across_relations():
    assert D.cue_conflicts() == []
    assert all(len(r.cues) >= 2 for r in D.RELATIONS.values())


def test_filler_and_prefixes_contain_no_cue_or_negation_phrase():
    from bench.synth.align import find

    phrases = [tokens(c) for r in D.RELATIONS.values() for c in r.cues]
    phrases += [tokens(n) for n in D.NEGATIONS if tokens(n) != ["n", "t"]]
    for text in (*D.FILLER, *D.UPDATE_PREFIXES):
        toks = tokens(text)
        assert not [p for p in phrases if find(toks, p)], text


def test_every_op_text_carries_a_cue_of_its_relation(worlds):
    for syn in worlds.values():
        for op in syn.ops:
            if op["tool"] != "remember":
                continue
            rel = (
                syn.expected["memories"][str(op["i"])]["rel"]
                if str(op["i"]) in syn.expected["memories"]
                else None
            )
            if rel is None:
                continue
            text = " ".join(tokens(op["args"]["text"]))
            assert any(" ".join(tokens(c)) in text for c in D.RELATIONS[rel].cues), op


def test_queries_cover_current_asof_and_missing(worlds):
    for syn in worlds.values():
        kinds = Counter(q["type"] for q in syn.queries)
        assert kinds["current"] > 0 and kinds["missing_slot"] > 0
        for q in syn.queries:
            if q["type"] == "missing_slot":
                assert q["gold"] == []
                assert q["slot"] not in syn.expected["slots"]
            if q["type"] == "as_of":
                m = syn.expected["memories"][str(q["gold"][0])]
                assert m["valid_from"] <= q["as_of"] < m["valid_to"]


def test_ops_are_chronological_and_json_clean(worlds):
    for syn in worlds.values():
        nows = [op["now"] for op in syn.ops]
        assert nows == sorted(nows) and len(set(nows)) == len(nows)
        json.dumps(syn.ops)  # plain data only


def test_too_small_world_is_an_error_not_a_silent_shortfall():
    spec = dict(D.SIZES["S"])
    D.SIZES["_tiny"] = spec | {"people": 2, "projects": 1, "orgs": 1, "places": 1,
                               "first_collisions": 1, "cross_kind": 1, "ops": 400}  # fmt: skip
    try:
        with pytest.raises(RuntimeError, match="too small"):
            generate(1, "_tiny")
    finally:
        del D.SIZES["_tiny"]


def test_value_returns_never_repeat_the_current_value(worlds):
    """FIX A: a value_returns op must not duplicate the memory it supersedes (the last-written one),
    nor the latest-by-valid-time one; returns_to is an earlier, different memory."""
    seen = 0
    for syn in worlds.values():
        for op in syn.ops:
            g = op["gold"]
            if op["kind"] != "update" or g["subtype"] != "value_returns":
                continue
            seen += 1
            assert g["returns_to"] != g["counterpart"]
            cur = syn.ops[g["counterpart"]]
            assert op["args"]["text"] != cur["args"]["text"]
            assert g["value"] != cur["gold"]["value"]
            slot_ops = [o for o in syn.ops[: op["i"]]
                        if o["gold"].get("slot") == g["slot"] and "as_of" in o["args"]
                        and o["kind"] in ("new_fact", "update")]  # fmt: skip
            gone = {o["gold"]["target"] for o in syn.ops[: op["i"]] if o["kind"] == "retract"}
            slot_ops = [o for o in slot_ops if o["i"] not in gone]
            latest = max(slot_ops, key=lambda o: (o["args"]["as_of"], o["i"]))
            assert op["args"]["text"] != latest["args"]["text"]
    assert seen


def test_update_counterpart_is_the_member_valid_at_its_as_of(worlds):
    """MEM2-156: the gold counterpart of an update is the slot's latest active member by valid time
    (the value the new one supersedes), not the member written last. The two differ after a `late`
    update: the late member is already closed in the gold chain, so labelling it the counterpart
    turned the scorer's correct list (the current member, surfaced) into a B-W3 miss, the oracle
    linked nothing, and the slot leaked in B-R5 (11 of the 22 gated misses on the dev seeds)."""
    late_then_update = 0
    for syn in worlds.values():
        for op in syn.ops:
            if op["kind"] != "update":
                continue
            slot = op["gold"]["slot"]
            gone = {o["gold"]["target"] for o in syn.ops[: op["i"]] if o["kind"] == "retract"}
            members = [o for o in syn.ops[: op["i"]] if o["gold"].get("slot") == slot
                       and "as_of" in o["args"] and o["kind"] != "dup_exact"
                       and o["i"] not in gone]  # fmt: skip
            latest = max(members, key=lambda o: (o["args"]["as_of"], o["now"], o["i"]))
            assert op["gold"]["counterpart"] == latest["i"], (syn.seed, op["i"])
            last_written = max(members, key=lambda o: o["i"])
            if last_written["gold"].get("subtype") == "late":
                late_then_update += 1
                assert last_written is not latest
    assert late_then_update  # the worlds contain the case that was mislabelled
