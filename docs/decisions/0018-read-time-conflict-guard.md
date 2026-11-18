# 0018: A read-time conflict guard hides the earlier of a conflicting same-subject pair in recall's final set

- **Status:** Accepted (the maintainer's ruling, MEM2-159, 2026-10-25); **off by default; revisit** (the owner, 2026-10-25, MEM2-164)
- **Date:** 2026-10-25 · **Task:** MEM2-159 · **By:** Core (Harness) on the maintainer's ruling

## Update 2026-10-25 (MEM2-164): off by default; revisit
`read.conflict_guard` now defaults to `false`. The code, the θ_read fit (`bench syn-fit-read-guard`) and its tests stay; a test or
a run that wants the guard sets `read.conflict_guard: true` explicitly. **Reason (host pass 5):** on SYN dev the 5 residual stale
pairs score below the best co-valid pair (6.0), so no cut reaches leakage 0 without wrongly hiding co-valid rows. With the guard off
a P3 gate file without a `read_guard` section is accepted; with it on it is still refused. B-R5's SYN rows are known misses
(benchmarks-v2 B-R5). Revisit on PG-v1 or when a scorer feature separates those 5 pairs. The switch is in `fp_all`.

## Context
B-R5's SYN gate is **leakage = 0**: no current-value read may deliver a slot memory that is not valid at
t. Round 4 of MEM2-1 (host, 25 Oct, commit 6d0efc2) reached B-W3 surfacing 0.967 / list precision 0.552
/ chain 0.961 and B-R5 leakage 0.00887 (as_of 0.935, current@1 0.856). Every one of the 18 leaks sat in a
slot with a **missed update**: the write path surfaced ~97% of updates, the counterpart of the other 3%
was never linked, its interval never closed, and the stale value came back beside the current one.
Detection alone cannot reach 100%: the fit's nested worst fold is 0.96-0.97 and the write-time list is
cut for precision, not recall. In real use the same thing happens whenever Claude skips a surfaced
`possible_conflicts` line (B-W5 conflict action is gated at ≥ 0.85, not 1).

## Decision
`recall` gets a second, read-time use of the write path's conflict model (`meta.gate.conf_model`):

1. **Where.** In the final set, after the gate and the limit/budget cut (`mem.read._recall_gated`
   step 7b). It only removes rows; it never adds one (A2) and never changes what the gate decided.
   `include_history` reads are not guarded.
2. **Rule.** For two memories of the set that share a subject (an `about` edge), the later one in its
   chain (later `valid_from`; tie: later `recorded_at`, then id) is the "new text" and the earlier one
   the candidate; the pair gets the write path's generic features (`mem.confmodel.FEATURES`, computed by
   `mem.conflicts._features`) and the conflict model's score. At score ≥ **θ_read** the earlier one is
   **hidden**: left out of the rows, named in `structured.hidden` as
   `{id, possibly_superseded_by, score}` and in one text line (`hidden m12 · possibly superseded by m45
   · get m12 to see it`). The recall event logs the pair (`result.hidden`); `mem.review.read_guard_hidden`
   lists the pairs nobody has linked yet so the daily review can offer a real `supersedes` (or
   `duplicate_of` / `relates_to`). `mem conflicts` and `mem review` print them.
3. **Nothing is written.** The hidden memory stays active and open, in `get`, in `include_history`, in
   the core index. A later real `link` closes the interval and the guard has nothing left to do.
4. **θ_read.** Fitted on SYN dev only, by nested leave-one-seed-out CV (`bench syn-fit-read-guard`,
   `bench/synth/readguard_fit.py`): the dev seeds are replayed under the gate's own conflict cut (so
   missed updates leave stale members open, as in B-R5), every read probe goes through the production
   `recall` with the guard off, and every same-subject pair of the delivered set is scored as the guard
   scores it and labelled from the gold only: the earlier row is *stale* (not valid at t: hiding it is
   right) or *co-valid* (valid at t: two relations of one subject, two values of a multi-valued relation,
   a B-W2 paraphrase and its original; hiding it is wrong). θ_read on the training seeds is the smallest
   cut that hides **zero** co-valid pairs (just above the best co-valid score, the midpoint to the next
   stale score when one lies above), plus a safety margin chosen by the inner folds, **never below
   θ_conf** (the read cut is strict). The fit is `feasible` only if every outer fold hides zero co-valid
   rows **and** reaches leakage 0. The final θ_read is the same rule on all five seeds (flat, optimistic,
   labelled). Nothing is relaxed: an infeasible fit is reported with the numbers and no gate is made.
5. **Fingerprints and refusals.** θ_read lives in `meta.gate.read_guard` with a fingerprint over
   `(version, theta_read, conf_model fingerprint)`; a `read_guard` whose fingerprint does not match, or
   that was fitted against another `conf_model`, is refused at load. The config switch
   **`read.conflict_guard`** (default on) is **inside `fp_all`**: a gate fitted with the guard on is void
   with it off (and the other way round), like every other read-path setting. With the switch on, a
   P3 gate (a `conf_model` in force) without a `read_guard` is refused by `mem serve` preflight, by
   `bench gate P3`, by the bench's P3 backend and by `recall` itself: there is no default read cut.
6. **Generic signals only.** The guard adds no feature; the conflict model's features are the write
   path's, and none of them knows a SYN cue list, template, relation id or value list (rule 4).

## Alternatives considered
- **Reach leakage 0 at write time only.** Nested worst surfacing is 0.96-0.97 after three rounds of
  scorer work (MEM2-150/154/156); the remaining misses are first-name-only counterparts and counterparts
  under a non-slot memory, and the list cut trades recall for the precision row. And a perfect detector
  does not help when the writer ignores the line. The read guard is the only place that sees the pair
  *after* the gate has already selected both.
- **Close the stale interval at read time (write `supersedes` from recall).** Rejected: a read that
  writes breaks A8 (read-time lookups never create anything), would act on a score, not a judgement, and
  could not be undone by the daily review. Hiding is reversible by construction.
- **Rank the stale row lower instead of hiding it.** Leakage counts presence, not rank; a lower stale
  row is still a leak and still a wrong fact in the context window.
- **A separate read-time model.** More parameters fitted on the same 750 dev updates; the write path's
  model already scores exactly this pair, and one scorer means one fingerprint to refit.

## Consequences
- B-R5 leakage can reach 0 with a missed update still in the store; `as_of_accuracy` and
  `current_recall_at_1` can only rise from hiding a stale row above the gold one, and fall only through
  a wrongly hidden co-valid row, which the fit counts as infeasible.
- Cost: pairs inside the final set only (≤ `limit` rows, 8 by default; one reranker call per member that
  has same-subject predecessors in the set, usually one). With the stub models on SYN-M (863 memories,
  626 probes, every pair forced to hide) the `guard` stage is p95 0.65 ms, max 1.7 ms; total recall
  p95 2.02 → 2.53 ms. On the host the cost is one extra reranker call of ≤ 7 pairs on the reads whose
  set has same-subject members; B-C1 reports it as the `guard` stage (`bench.latency.STAGES`).
- `fp_all` changes (the `read` section joined it): every gate and record made before this is void and
  the host round refits everything (`deliverables/runbooks/p3-gate.md` §2b, round 5).
- `structured.hidden` joins the `recall` contract (optional field; rows are unchanged).
- Supersedes nothing: 0007 (resolver, conflicts, chains) still defines the chain and the write path.

## Revisit if
The nested fit cannot reach leakage 0 without hiding a co-valid memory on dev (then the numbers are
reported and the scorer, not the rule, is the thing to fix); the daily review shows hidden pairs that
were not updates (wrongly hidden in live use); or the `guard` stage costs more than the B-C1 regression
rule allows on the host.
