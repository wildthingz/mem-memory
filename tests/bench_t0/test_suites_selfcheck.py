"""The P1 suites against a tiny fake server: they must pass on a correct one and fail,
for the right named check, on each deliberately broken one (red-green for the gates)."""

import pytest

from bench.suites import REGISTRY
from bench.suites.base import Context
from tests.bench_t0.fake_serve import Flags, factory_with, list_tools


def run(tmp_path, suite, **flags):
    return REGISTRY[suite](Context(tmp_path, factory_with(Flags(**flags)), list_tools))


@pytest.mark.parametrize("suite", ["B-S1", "B-R6", "B-W6", "B-R5", "B-C1", "B-C2"])
def test_suites_pass_on_a_correct_server(tmp_path, suite):
    res = run(tmp_path, suite)
    assert res.violations == []
    assert res.items


@pytest.mark.parametrize(
    ("suite", "flag", "failing"),
    [
        ("B-R5", "ignore_valid_from", "a_as_of_2026-10-22T09:00:00Z"),
        ("B-R5", "ignore_valid_to", "a_as_of_2026-11-01T00:00:00Z"),
        ("B-R5", "closed_boundary", "a_as_of_2026-11-01T00:00:00Z"),
        ("B-R5", "utc_date_only", "e_write_2026-11-01"),
        ("B-R5", "utc_date_only", "e_read_date_only_ends_at_local_midnight"),
        ("B-R5", "start_of_day_as_of", "e_read_date_only_covers_the_local_day"),
        ("B-R6", "lossy_get", "get_continuation_byte_identical"),
        ("B-R6", "chatty_empty", "empty_is_one_no_results_line"),
        ("B-R6", "no_p1_header", "q0_header"),
        ("B-R6", "edge_overlap", "edges_each_live_edge_exactly_once"),
        ("B-W6", "cross_store", "6_id_from_other_store_is_not_found"),
        ("B-W6", "rewrite", "2_no_row_deleted_or_rewritten"),
        ("B-W6", "no_about", "3_every_memory_has_live_about_edge"),
        ("B-W6", "clock_recorded", "1_memory_recorded_at_equals_its_op_date"),
        ("B-W6", "nondeterministic", "1_dumps_byte_identical"),
    ],
)
def test_suites_fail_on_a_broken_server(tmp_path, suite, flag, failing):
    res = run(tmp_path, suite, **{flag: True})
    failed = {i["check"] for i in res.items if not i["ok"]}
    assert failing in failed, failed
