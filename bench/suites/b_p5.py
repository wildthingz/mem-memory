"""The P5 gate suites (benchmarks-v2 §4 P5 row): B-E5 (two-machine parity) and B-C6 (session overhead).

Inputs are the records of the human-run legs, passed in `ctx.options["p5"]` (`P5Setup`): one
`bench e5-run` record per machine/transport (exactly one with role `local`, the baseline of the
+p95 gate), `bench e5-reconnect` records, and one `bench c6-run` record. A missing input leaves the
metric out ("not produced": FAIL), never a pass. B-S1 over Tailscale is the `smoke_failures` of the
remote records.

`home_only` (`bench gate P5 --home-only`; the owner's scope change of 24 Oct: the work machine joins
later) reports the home-side rows from the local record and marks the rows that need a remote run
(parity, added p95) or a reconnect record as DEFERRED when those records are absent: not run,
neither PASS nor FAIL. Records that are supplied are always evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bench import c6, e5
from bench.suites.base import BackendUnavailable, Context, SuiteResult

OPT = "p5"
TOL = 1e-6


@dataclass
class P5Setup:
    e5_runs: list[Path] = field(default_factory=list)
    reconnects: list[Path] = field(default_factory=list)
    c6_run: Path | None = None
    home_only: bool = False


CROSS_MACHINE = ("parity_id_mismatches", "parity_max_p_diff", "added_p95_ms")
DEFER_WHY = "home-only checkpoint, no work-machine run yet"


def setup_of(ctx: Context) -> P5Setup:
    setup = ctx.options.get(OPT)
    if setup is None:
        raise BackendUnavailable("P5 needs the recorded runs: --e5-run, --e5-reconnect, --c6-run")
    return setup


def run_b_e5(ctx: Context) -> SuiteResult:
    setup = setup_of(ctx)
    res = SuiteResult("B-E5", mode="recorded")
    recs = [e5.load_record(p, "e5-run") for p in setup.e5_runs]
    locals_ = [r for r in recs if r["role"] == "local"]
    remotes = [r for r in recs if r["role"] == "remote"]
    res.check("one_local_baseline", len(locals_) == 1, f"{len(locals_)} local records")
    deferring = setup.home_only and not remotes
    if not deferring:
        res.check("at_least_one_remote", bool(remotes), "no remote record (work machine)")
    if len(locals_) != 1 or not (remotes or deferring):
        return res
    base = locals_[0]
    for r in recs:
        res.check(f"{r['label']}: same fixtures",
                  (r["queries_sha256"], r["ops_sha256"]) == (base["queries_sha256"], base["ops_sha256"]))  # fmt: skip
        res.check(f"{r['label']}: runs >= 3", r["runs"] >= 3, f"runs={r['runs']}")
        res.check(f"{r['label']}: scratch legs ran", bool(r["writes"]) and r["scratch"] is not None)
        for s in e5.STORES:
            res.check(f"{r['label']}/{s}: stable across runs", not r["stores"][s]["unstable_qids"],
                      str(r["stores"][s]["unstable_qids"][:5]))  # fmt: skip
    mismatches, max_dp, write_diffs = 0, 0.0, 0
    for r in remotes:
        cmp = e5.compare_runs(base, r, TOL)
        mismatches += cmp["id_mismatches"] + len(cmp["write_diffs"])
        write_diffs += len(cmp["write_diffs"])
        max_dp = max(max_dp, cmp["max_p_diff"])
        for line in cmp["notes"] + cmp["write_diffs"][:5]:
            res.check(f"{base['label']} vs {r['label']}", False, line)
        res.notes.append(f"{r['label']}: ids_mismatch={cmp['id_mismatches']} "
                         f"max_p_diff={cmp['max_p_diff']:.3g} write_diffs={len(cmp['write_diffs'])} "
                         f"added_p95={e5.added_p95(base, r):.1f} ms")  # fmt: skip
    if remotes:
        res.metrics["parity_id_mismatches"] = float(mismatches)
        res.metrics["parity_max_p_diff"] = max_dp
        res.metrics["added_p95_ms"] = max(e5.added_p95(base, r) for r in remotes)
    else:
        res.deferred.update({m: DEFER_WHY for m in CROSS_MACHINE})
    res.metrics["cross_store_ids"] = float(sum(r["cross_store_ids"] or 0 for r in recs))
    res.metrics["smoke_failures"] = float(sum(r["smoke_failures"] or 0 for r in recs))
    res.metrics["local_p95_ms"] = e5.p95(
        [v for s in e5.STORES for v in base["stores"][s]["latencies_ms"]]
    )
    recon = [e5.load_record(p, "e5-reconnect") for p in setup.reconnects]
    if recon:
        res.metrics["reconnect_failures"] = float(sum(not (r["ok"] and (r["retries"] or 0) <= 1)
                                                      for r in recon))  # fmt: skip
        for r in recon:
            res.check(f"reconnect {r['label']}", r["ok"] and (r["retries"] or 0) <= 1, r["detail"])
    elif setup.home_only:
        res.deferred["reconnect_failures"] = DEFER_WHY
    res.extra["records"] = [r["label"] for r in recs]
    return res


def run_b_c6(ctx: Context) -> SuiteResult:
    setup = setup_of(ctx)
    res = SuiteResult("B-C6", mode="recorded")
    if setup.c6_run is None:
        raise BackendUnavailable("B-C6 needs --c6-run (`bench c6-run`, count_tokens)")
    rec = e5.load_record(setup.c6_run, "c6-run")
    res.check(
        "counted_with_count_tokens", rec["method"] == "count_tokens", f"method={rec['method']}"
    )
    if rec["method"] != "count_tokens":
        return res  # an estimate is not the gate's measurement
    res.metrics["session_tokens"] = float(rec["session_tokens"])
    r5 = bool(rec["r5_applied"])
    if r5:  # the flag counts only if the record shows the R5 shape (one server, `store` param)
        res.check("r5_shape_in_record", c6.r5_shape_ok(rec),
                  f"r5_applied but servers={rec.get('servers')} tools_n={rec.get('tools_n')} "
                  f"store_param={rec.get('store_param')}")  # fmt: skip
        r5 = c6.r5_shape_ok(rec)
    res.metrics["overhead_ok"] = float(rec["session_tokens"] <= rec["budget"] or r5)
    res.notes.append(f"tools={rec['tools_tokens']} skill={rec['skill_tokens']} "
                     f"core_block={rec['core_block_tokens']} r5_applied={rec['r5_applied']} "
                     f"servers={rec.get('servers')} tools_n={rec.get('tools_n')}")  # fmt: skip
    return res
