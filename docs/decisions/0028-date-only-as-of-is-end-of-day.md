# 0028. A date-only `as_of` on a read means the end of that day (MEM2-265)

Status: accepted. Spec: architecture-v2 §2 (timestamps), §3.2 step 0 (`t = as_of or now`), the
`recall` contract (§4), benchmarks-v2 B-R5 (e). Evidence: `data/temporal-r10/` (research note §0 and
§C option 0). Requested by the owner (MEM2-1, 28 Oct, Round 10 task 1).

## Context

The LME harness passed `as_of` = the question's timestamp with its time of day
(`parse_lme_date("2023/05/20 (Sat) 02:21")` = `2023-05-20T02:21:00Z`). Reads keep a memory iff
`valid_from <= t < valid_to`. LongMemEval stamps some gold sessions later on the same day as the
question. Those sessions were dropped before the legs ran. The owner counted it on
`longmemeval_s_cleaned.json`: 44/500 questions are affected (43 temporal-reasoning, 1
knowledge-update), and in 20 of them every gold session is after the question time. No question has
gold on a later day. On the tune split this explains 5 of the 7 temporal pool misses on the final
config (gpt4_213fd887, gpt4_0a05b494, gpt4_d9af6064, c9f37c46, gpt4_2f56ae70), plus one partly
(gpt4_8c8961ae) and a missing endpoint in three delivered-but-wrong answers. Upstream LongMemEval
does not time-filter at all: the haystack is the history before the question.

The product had the same ambiguity. A date-only `as_of` was read as 00:00 local (architecture-v2
§2), the same rule as on the write side. So `recall(as_of="2026-11-01")` did not see anything that
happened on 1 November, which is not what a person means by "as of 1 November".

## Decision

1. **Reads.** A date-only `as_of` (YYYY-MM-DD) passed to a read means the **end of that day**:
   23:59:59 in the store's configured zone (`store.tz`: `mem serve --tz`, else the system zone,
   `local_tz`), converted to UTC
   (`mem.common.normalize_as_of`). Everything that happened that day, at any time, is visible, and
   nothing from the next local day is. Timestamps are stored to the second, so `valid_from <=
   23:59:59` covers the whole day. This applies to `recall(as_of=)` and to
   `calibrate.score_pool(as_of=)`, which must see the same eligible set as recall.
2. **Zone: the configured local zone, not UTC.** The person thinks in their own calendar day.
   Write-side dates and `since`/`until` are already read in that zone. With UTC, a Europe/Amsterdam
   user asking "as of 1 November" would see things from 00:30 on 2 November (local) and, in summer,
   miss the last two hours of 1 November.
3. **Timestamped `as_of` is unchanged.** It keeps the half-open predicate at that instant.
4. **Writes are unchanged.** A date-only `remember(as_of=)` is still local midnight: a fact said to
   hold "from 1 November" holds from the start of that day. Same-day corrections (N1) behave as
   before: both members have `valid_from` = local midnight, and the end-of-day read sees the last
   one.
5. **`since`/`until` are unchanged** (they filter `nodes.date`, a separate decision).
6. **Harness.** Every LME path reads at the question's day at 23:59:59Z (`lme.question_as_of`).
   LME has no zone, and the harness stores are UTC. Session dates keep their time of day
   (`parse_lme_date`). The paths are the `OursLmeBackend` methods `score` (tune scoring for
   `mem calibrate --labels`, `score_tune`), `retrieve` (B-R1, pool recall, `e1-run`, `--full`,
   LME_M, `delivered-dump`) and `open_tools` (`e1-agentic`, whose store clock is that instant).
   They also include `bench prf-check` and the `as_of` field of `bench export-tune`. The P4 adapter's
   direct read-path `t` and `bench gold review`'s `t` now use recall's reading of the same
   date-only snapshot. LoCoMo already reads one day after its last session, which is later than any
   session, so it has no same-day loss and is unchanged.

## Consequences

- No retrieval or read section of the config changes, and neither does the default fingerprint
  (`fp_all`). The golden test still passes. Gates and records fitted before this change were
  computed on smaller eligible sets for the affected questions. Round 10 task 1
  (`deliverables/runbooks/p3-gate.md`) re-calibrates and re-gates on the final config before any
  number is quoted.
- LME numbers before and after are not comparable on the 44 affected questions. Report this as a
  harness fix, not a retrieval gain (research note §C option 0, "Honesty").
- B-R5 case (e) changes its read-side checks. `e_read_date_only_is_local_midnight` becomes
  `e_read_date_only_covers_the_local_day` (a memory at 23:30 local on the day is visible) and
  `e_read_date_only_ends_at_local_midnight` (00:30 local on the next day is not). The T0 fake has a
  mutant for each wrong reading: `utc_date_only` and `start_of_day_as_of`.
- The tool contract (`mcp_tools.json`) does not change. `as_of` stays a free string. The skill
  already says "`as_of` (what was true then)".

## Alternatives rejected

- **Harness only, product unchanged.** That fixes LME but leaves the product's date-only `as_of`
  excluding the day the person named. The research note (§C option 0) flags this as needing an
  ADR if adopted. It is adopted because the old reading is surprising for people too.
- **Next local midnight with a strict `<`.** Same meaning, but it needs a second predicate shape in
  `build_allowed`. 23:59:59 with the existing `<=` is identical at one-second resolution.
- **UTC day.** Rejected for people (point 2). The harness uses UTC only because LME's stores and
  dates are UTC.
