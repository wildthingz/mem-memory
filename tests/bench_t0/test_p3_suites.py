"""The P3 suites against the stand-in server (tests/bench_t0/fake_p3.py): each must PASS on a correct
server and FAIL, on the named metric or check, on every deliberately broken one (red-green for the
gates). The same suites on the real store and server: test_p3_real.py."""

import math

import pytest

from bench.suites import b_r5_p3, b_w6_p3, b_w123
from bench.suites.base import BackendUnavailable, Context
from bench.synth.run import SynSetup
from tests.bench_t0.fake_p3 import P3Flags, factory_p3
from tests.bench_t0.fake_serve import list_tools


def ctx(tmp_path, seeds=(1,), size="S", split="fixture", **flags):
    flags.setdefault("oracle_recall", True)
    return Context(tmp_path, factory_p3(P3Flags(**flags)), list_tools,
                   {"syn": SynSetup(seeds, size, split)})  # fmt: skip


def failed(res):
    return {i["check"] for i in res.items if "check" in i and not i["ok"]}


# ---- a correct server passes ---------------------------------------------------------------------


@pytest.mark.parametrize("seeds", [(1,), (2, 3), (101,)])
def test_w1_w2_w3_pass_on_a_correct_server(tmp_path, seeds):
    c = ctx(tmp_path, seeds)
    w1, w2, w3 = b_w123.run_b_w1(c), b_w123.run_b_w2(c), b_w123.run_b_w3(c)
    assert w1.metrics["wrong_merge_rate"] == 0
    assert w1.metrics["ask_back_recall"] == 1
    assert w1.metrics["ask_rate"] <= 0.10 and w1.metrics["split_rate"] <= 0.02
    assert w1.metrics["final_link_accuracy"] >= 0.98 and w1.metrics["table_case_failures"] == 0
    assert w2.metrics["exists_rate"] == 1 and w2.metrics["paraphrase_surfacing"] >= 0.9
    assert w2.metrics["final_duplicate_rate"] <= 0.03 and w2.metrics["case_failures"] == 0
    assert w2.metrics["wrong_invalidation_compat"] == 0
    for m in ("surfacing_recall", "chain_correctness", "slot_accuracy", "history"):
        assert w3.metrics[m] >= 0.95, m
    assert w3.metrics["wrong_invalidation"] == 0 and w3.metrics["graph_diffs"] == 0
    assert w3.metrics["list_precision"] >= 0.5 and w3.metrics["case_failures"] == 0


def test_every_seed_is_replayed_once_per_setup_and_shared_across_suites(tmp_path):
    c = ctx(tmp_path, (1, 2))
    c.options["syn"].workdir = tmp_path / "kept"  # what `bench gate` provides
    b_w123.run_b_w1(c)
    cache = c.options["syn"].cache
    assert set(cache) == {(1, "S"), (2, "S")}
    first = {k: v.db for k, v in cache.items()}
    b_w123.run_b_w3(c)
    assert {k: v.db for k, v in c.options["syn"].cache.items()} == first


def test_w6_all_items_pass_on_a_correct_server(tmp_path):
    res = b_w6_p3.run(ctx(tmp_path))
    assert res.violations == [] and res.metrics["violations"] == 0
    names = {i["check"] for i in res.items}
    for n in range(1, 12):
        assert any(x.startswith(f"{n}_") for x in names), f"item {n} not run"


def test_r5_cases_and_syn_probes_pass_on_a_correct_server(tmp_path):
    res = b_r5_p3.run_b_r5(ctx(tmp_path))
    assert res.violations == [] and res.metrics["case_failures"] == 0
    assert res.metrics["leakage"] == 0
    assert res.metrics["as_of_accuracy"] >= 0.95 and res.metrics["current_recall_at_1"] >= 0.90
    names = {i["check"] for i in res.items}
    for case in "abcdef":
        assert any(x.startswith(f"{case}_") for x in names), f"case ({case}) not run"


def test_r2_ii_missing_slot_abstention(tmp_path):
    res = b_r5_p3.run_b_r2_ii(ctx(tmp_path, (101, 102), "M", "test"))
    assert res.metrics["missing_slot_abstention"] >= 0.90 and res.metrics["missing_n_min"] > 10


# ---- every mutant fails the right thing ----------------------------------------------------------

W1_MUTANTS = [
    ("never_ask", "ask_back_recall", lambda v: v < 0.90),
    ("never_ask", "wrong_merge_rate", lambda v: v > 0.005),
    ("always_ask", "ask_rate", lambda v: v > 0.10),
    ("no_typo_link", "split_rate", lambda v: v > 0),
    ("ask_gold_missing", "ask_offered_gold", lambda v: v < 1),
]


@pytest.mark.parametrize(("flag", "metric", "bad"), W1_MUTANTS)
def test_w1_mutants(tmp_path, flag, metric, bad):
    res = b_w123.run_b_w1(ctx(tmp_path, (1, 2, 3), **{flag: True}))
    value = res.metrics.get(metric)
    if value is None:  # diagnostics are per seed only
        value = min(s[metric] for s in res.extra["per_seed"].values())
    assert bad(value), (flag, metric, value)


@pytest.mark.parametrize(
    ("flag", "metric", "bad"),
    [
        ("no_exists", "exists_rate", lambda v: v < 1),
        ("no_conflicts", "paraphrase_surfacing", lambda v: v < 0.9),
        ("dup_not_hidden", "final_duplicate_rate", lambda v: v > 0.03),
        ("invalidate_on_add", "wrong_invalidation_compat", lambda v: v > 0),
    ],
)
def test_w2_mutants(tmp_path, flag, metric, bad):
    res = b_w123.run_b_w2(ctx(tmp_path, (1, 2, 3), **{flag: True}))
    assert bad(res.metrics[metric]), (flag, metric, res.metrics[metric])


def test_w2_n2_case_fails_when_a_superseded_text_is_reported_exists(tmp_path):
    res = b_w123.run_b_w2(ctx(tmp_path, exists_for_closed=True))
    assert "n2_resave_superseded_text_is_saved_not_exists" in failed(res)


def test_w2_concurrency_fails_when_both_writers_save(tmp_path):
    res = b_w123.run_b_w2(ctx(tmp_path, no_exists=True))
    assert "concurrent_same_text_one_row_other_exists" in failed(res)


@pytest.mark.parametrize(
    ("flag", "metric", "bad"),
    [
        ("no_conflicts", "surfacing_recall", lambda v: v < 0.95),
        ("no_derive", "chain_correctness", lambda v: v < 0.95),
        ("no_derive", "graph_diffs", lambda v: v > 0),
        ("lossy_history", "history", lambda v: v < 1),
        ("invalidate_on_add", "wrong_invalidation", lambda v: v > 0),
    ],
)
def test_w3_mutants(tmp_path, flag, metric, bad):
    res = b_w123.run_b_w3(ctx(tmp_path, (1, 2, 3), **{flag: True}))
    assert bad(res.metrics[metric]), (flag, metric, res.metrics[metric])


def test_w3_a22_case_fails_without_candidates(tmp_path):
    res = b_w123.run_b_w3(ctx(tmp_path, no_conflicts=True))
    assert "a22_old_attribute_update_after_62_writes_surfaces" in failed(res)


def test_w3_t0_graph_equality_check_is_a_named_check(tmp_path):
    res = b_w123.run_b_w3(ctx(tmp_path, no_derive=True))
    assert "t0_seed1_graph_equals_expected_state" in failed(res)


@pytest.mark.parametrize(
    ("flag", "check"),
    [
        ("link_not_idempotent", "8_repeated_link_is_ok"),
        ("needs_choice_writes", "7_needs_choice_writes_nothing_incl_new_subjects"),
        ("same_as_crash", "9_same_as_shared_neighbour_and_alias_succeeds"),
        ("link_imported", "10_supersedes_imported_is_refused"),
        ("busy", "11_no_sqlite_busy_reaches_the_client"),
        ("no_derive", "4_s1_supersedes_components_are_chains"),
        ("nondeterministic", "1_s1_dumps_byte_identical"),
        ("no_about", "3_s1_every_memory_has_live_about_edge"),
        ("rewrite", "2_no_row_deleted_or_rewritten"),
    ],
)
def test_w6_mutants(tmp_path, flag, check):
    res = b_w6_p3.run(ctx(tmp_path, **{flag: True}))
    assert check in failed(res), sorted(failed(res))
    assert res.metrics["violations"] > 0


@pytest.mark.parametrize(
    ("flag", "check"),
    [
        ("no_derive", "b_chain_rederives"),
        ("no_swap", "c_reply_reports_the_swap"),
        ("forget_keeps", "d_chain_rederives_without_the_retracted"),
        ("no_derive", "f_predecessors_get_empty_intervals"),
        ("utc_date_only", "e_write_2026-11-01"),
        ("utc_date_only", "e_read_date_only_ends_at_local_midnight"),
        ("start_of_day_as_of", "e_read_date_only_covers_the_local_day"),
    ],
)
def test_r5_case_mutants(tmp_path, flag, check):
    res = b_r5_p3.run_b_r5(ctx(tmp_path, **{flag: True}))
    assert check in failed(res), sorted(failed(res))
    assert res.metrics["case_failures"] > 0


@pytest.mark.parametrize("flag", ["ignore_valid_to", "ignore_valid_from"])
def test_r5_syn_leakage_is_caught(tmp_path, flag):
    res = b_r5_p3.run_b_r5(ctx(tmp_path, (1, 2, 3), **{flag: True}))
    assert res.metrics["leakage"] > 0 or res.metrics["as_of_accuracy"] < 0.95


def test_r2_ii_fails_when_the_server_does_not_abstain(tmp_path):
    res = b_r5_p3.run_b_r2_ii(ctx(tmp_path, (1, 2), no_abstain=True))
    assert res.metrics["missing_slot_abstention"] < 0.90


def test_w1_baselines_show_the_a8_failure(tmp_path):
    res = b_w123.run_b_w1(ctx(tmp_path, (1, 2, 3)))
    assert res.metrics["wrong_merge_rate"] == 0
    per = res.extra["per_seed"]
    v1 = sum(s["baseline_v1_rule_wrong_merge"] for s in per.values())
    exact = sum(s["baseline_exact_alias_wrong_merge"] for s in per.values())
    assert v1 > exact  # the v1 fuzzy rule merges "Ana" into the first Ana
    assert all(not math.isnan(s["baseline_exact_only_split"]) for s in per.values())


def test_suites_report_unavailable_when_the_server_has_no_link(tmp_path):
    from tests.bench_t0.fake_serve import factory_with

    c = Context(tmp_path, factory_with(), list_tools, {"syn": SynSetup((1,), "S", "fixture")})
    with pytest.raises(BackendUnavailable, match="link"):
        b_w123.run_b_w3(c)
