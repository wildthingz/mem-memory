"""`mem calibrate`'s thresholds search (MEM2-129): the fast search agrees with the exhaustive one it
replaced (tests/calibrate_ref.py) on small seeded sets, feasible and infeasible, and a host-sized
set (3,500 pools) finishes in seconds within a memory cap."""

import subprocess
import sys

import numpy as np
import pytest

from mem import calibrate
from tests.calibrate_ref import search_exhaustive

LENIENT = (
    calibrate.Target("delivered_precision", "precision", ">=", 0.3),
    calibrate.Target("ira", "ira", "<=", 0.8),
    calibrate.Target("coverage_any", "coverage_any", ">=", 0.5),
    calibrate.Target("abstention_recall", "abstention_recall", ">=", 0.5),
    calibrate.Target("far", "far", "<=", 0.4),
)


def _small_pools(seed):
    """Few, short, tied pools: ties, empty pools and single-item pools all occur."""
    rng = np.random.default_rng(seed)
    pools = []
    for q in range(int(rng.integers(20, 45))):
        items = [
            calibrate.PoolItem(
                f"c{i}",
                float(rng.choice([-3.0, -1.5, -0.5, 0.0, 0.7, 1.5, 3.0])),
                int(rng.random() < 0.3),
            )
            for i in range(int(rng.integers(0, 6)))
        ]
        ans = bool(rng.random() < 0.65)
        if seed % 2 and items:  # a separable set: gold on top for answerables, negatives low
            items = [calibrate.PoolItem(i.key, i.z - 4 * (not ans) + 5 * (i.rel and ans), i.rel)
                     for i in items]  # fmt: skip
        pools.append(calibrate.QueryPool(f"q{q}", ans, items))
    return pools


def _same(new, old, *, feasible):
    assert (new.tau_keep, new.tau_high, new.gap, new.feasible, new.rule) == (
        old.tau_keep, old.tau_high, old.gap, old.feasible, old.rule)  # fmt: skip
    assert new.front == old.front
    assert new.exact
    assert new.targets.keys() == old.targets.keys()
    for name, t in new.targets.items():
        o = old.targets[name]
        assert {k: v for k, v in t.items() if k != "best"} == {
            k: v for k, v in o.items() if k != "best"
        }
        better = max if t["op"] == ">=" else min
        # `best` is over the evaluated settings: all of them when infeasible; when feasible the
        # settings above the bisected τ_keep are skipped, so it may fall short on precision, IRA
        # and abstention recall, never exceed the exhaustive best, and is exact for the monotone two
        assert better(t["best"], o["best"]) == o["best"]
        if not feasible or name in ("coverage_any", "far"):
            assert t["best"] == o["best"], name


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("which", ["p2", "lenient"])
def test_fast_search_equals_the_exhaustive_search(seed, which):
    targets = calibrate.p2_targets() if which == "p2" else LENIENT
    for limit in (1, 3):
        table = calibrate._Table(_small_pools(seed), 1.0, 0.0, limit)
        old = search_exhaustive(table, targets)
        _same(calibrate._search(table, targets), old, feasible=old.feasible)


def test_the_equivalence_seeds_cover_every_outcome():
    seen = set()
    for seed in range(40):
        for targets in (calibrate.p2_targets(), LENIENT):
            c = calibrate._search(calibrate._Table(_small_pools(seed), 1.0, 0.0, 3), targets)
            seen.add(c.rule)
    assert {"strictest", "fallback_coverage_far", "fallback_coverage_ceiling"} <= seen


def test_a_thinned_search_still_gives_the_exact_fallback_and_a_valid_front(monkeypatch):
    monkeypatch.setattr(calibrate, "EXACT_CELLS", 0)
    monkeypatch.setattr(calibrate, "FRONT_TAUS", 7)
    monkeypatch.setattr(calibrate, "FRONT_CUTS", 6)
    monkeypatch.setattr(calibrate, "FRONT_LEVELS", 3)
    targets = calibrate.p2_targets()
    for seed in range(15):
        table = calibrate._Table(_small_pools(seed), 1.0, 0.0, 3)
        old = search_exhaustive(table, targets)
        new = calibrate._search(table, targets)
        if old.feasible:
            continue  # the thinned scan may find a (flagged) setting or give up; no claim
        assert (new.tau_keep, new.tau_high, new.gap, new.rule) == (
            old.tau_keep, old.tau_high, old.gap, old.rule)  # fmt: skip
        for pt in new.front:  # every point is a real setting with its real numbers
            m = table.metrics(pt["tau_keep"], pt["tau_high"], pt["gap"])
            for k in calibrate.FRONT_KEYS:
                assert pt[k] == pytest.approx(float(np.nan_to_num(m[k])), abs=1e-6)


def test_a_thinned_feasible_answer_meets_every_target(monkeypatch):
    monkeypatch.setattr(calibrate, "EXACT_CELLS", 0)
    for seed in range(15):
        table = calibrate._Table(_small_pools(seed), 1.0, 0.0, 3)
        new = calibrate._search(table, LENIENT)
        if new.rule == "strictest":
            assert new.exact is False and new.feasible
            m = table.metrics(new.tau_keep, new.tau_high, new.gap)
            assert all(t.met(float(m[t.metric])) for t in LENIENT)


_PERF = """
import resource, sys, time
from mem import calibrate
from tests.calibrate_ref import synth_pools
pools = synth_pools(3500, 1, shift={shift}, miss=0.02, neg_shift={neg})
t0 = time.time()
c = calibrate.choose_thresholds(pools, a=1.0, b=0.0, limit=8)
print(time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, c.feasible, c.rule)
"""


@pytest.mark.parametrize(("shift", "neg", "feasible"), [(5.0, 0.0, False), (12.0, -4.0, True)])
def test_t0_3500_pools_finish_in_under_60s_and_2gb(shift, neg, feasible):
    """The host size (MEM2-1, 25 Oct: 3,356 pools ran 31 min and hit a 7 GB cap), infeasible (the
    whole candidate range is searched and the front built) and feasible. Peak RSS is read in a
    fresh process, with the pools in it."""
    out = subprocess.run(
        [sys.executable, "-c", _PERF.format(shift=shift, neg=neg)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    seconds, rss_mb, got = float(out[0]), float(out[1]), out[2] == "True"
    print(f"3,500 pools, feasible={got}: {seconds:.1f} s, peak RSS {rss_mb:.0f} MB")
    assert got is feasible
    assert seconds < 60 and rss_mb < 2048
