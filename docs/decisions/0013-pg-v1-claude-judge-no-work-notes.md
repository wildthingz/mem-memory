# 0013: PG-v1 is judged by Claude with a blind kappa check, and holds no work-store content

**Status:** accepted; spec change approved by the owner (comment on MEM2-1, 2026-10-24 17:29), implemented in MEM2-99.
Spec: benchmarks-v2 §3.1 and §7 (rewritten with a dated note). Quotas, split rule, seed 20261022 and the B-R3 gate
(`bench/configs/gates.yaml` `P4`) are unchanged.

## What changed and why
- **Who labels.** Claude grades every pooled (question, chunk) pair 0/1/2 (`bench gold judge`, T2, hard budget ≤ $25).
  The human grades a seeded random 50 of those judgments blind (`bench gold kappa-sample`). Why: their labelling time falls
  from ≈ 4 h to ≈ 1.25 h, and the human check measures whether the shortcut is safe instead of assuming it.
- **No work notes.** PG-v1 is drafted, pooled and scored on a separate PG snapshot (`personal/` and `projects/`, minus
  folders routed to `work`). Why: a gold set, a pool or a `qrels.tsv` that holds client text must never be committed
  or sent to an API; the way to guarantee it is that the import the tool works on has no such chunk.

## Definitions
- **Kappa:** Cohen's, graded 0/1/2, **linear** weights w = |i-j|/2; kappa = 1 - Σw·o / Σw·e. Reported with n.
  Undefined (both raters used one grade) is `nan` and never passes. Threshold **0.7, inclusive** (the task's number).
  Hand-computed pins in the tests: perfect = 1, chance table = 0, [[20,5,0],[5,10,5],[0,0,5]] = 23/38 = 0.605.
- **The sample:** `random.Random(20261024).sample` over the sorted (qid, gold id) keys of Claude's *pooled* judgments of
  accepted questions, n = min(50, available). The first draw is saved in `kappa_sample.json`, so a resumed sitting sees the
  same pairs. The sitting shows question, short answer and chunk; never Claude's grade, the system that found the chunk, or
  a hint. 20261024 is my choice (the date of the change); the spec fixes no seed for this sample.
- **Judge column:** every `qrels.tsv` row (rel 0 included) has `judge` = `claude` | `human`. Their label wins over Claude's on
  a pair whatever the row order (their 50 blind labels, and any they give later). `freeze` refuses a missing/unknown judge, and
  any `claude` row unless the recorded kappa ≥ 0.7; `load` (so `verify` and B-R3) re-checks both against `SNAPSHOT`
  (`judge_kappa`, `judge_kappa_n`, `judge_kappa_weights: linear`). The `(rel, content_hash)` tuples B-R3 reads are unchanged.
- **Draft gold** (the chunks Claude proposed with a draft, rel 1/2) is recorded as `human`: they accept or edit the draft
  with the chunk and answer on screen, as before. Only **pooled** pairs are Claude's. (Reading: confirm if the draft gold
  should count as Claude's instead; then the < 0.7 path could not make every row theirs.)
- **kappa < 0.7:** `kappa-sample` prints "DO NOT stand" and the number of pairs left; `label` then offers every pooled pair
  whose current judge is not them (their 50 are not asked again). `label` before the check, or after a pass, does nothing and says why.
- **Voiding** is unchanged and applies to Claude too: a no-answer question that gets a 1 or 2 is rejected and its quota reopens.
- **Pre-filter** (`bench gold check`): a draft is dropped (flagged, never shown in review, reason kept in `checks.jsonl`) if it
  is weak (< 3 content tokens in the question, or an answerable draft with an empty short answer) or a lexical-overlap failure:
  the **median over the draft's rel-2 gold chunks** of |T(q) ∩ T(chunk)| / |T(q)| > 0.5 (same T as the freeze control). The
  freeze control is unchanged (median over questions of the best chunk ≤ 0.5), so both still hold. ≥ 50% reworded unchanged.
- **Sampling:** the eligible notes are those in `personal/` and `projects/` held by the personal store, with a section of ≥ 40
  words. The 40 notes are split over the two folders ∝ eligible notes, largest remainder (ties to the larger folder, then the
  name). The 10 linked pairs are drawn from linked pairs whose notes are both eligible (a pair may cross the two folders, so
  there is no per-folder quota for pairs). `draft --slots-per-unit N` (default 4) raises the drafts per unit; targets scale
  as 5/3 × quota × N/4 with largest remainder.
- **Guard:** `freeze` and `verify` refuse any question source, gold or pooled (rel 0) path whose route (`vault.routes` of
  `--config` over the importer defaults) is `work`, or that sits in a work store. Without `--config` only the default
  `work/` route and the import's own stores are checked: the runbook always passes `--config`.
- **Two snapshots in the gate:** `bench gate P4 --pg-snapshot …` (B-R3, the PG import) next to `--vault-snapshot …` (B-I1,
  B-I3, B-C4, V5k/V50k). The gold's `SNAPSHOT` sha256 is checked against the PG snapshot. B-R3 never falls back to the full vault.

## Assumptions and what would change them
- The ≈ 1.25 h assumes ≈ 140 drafts shown at ≈ 25 s (the pre-filter removes the weak and copied ones) plus 50 pairs at ≈ 8 s.
  `bench gold status` reports the real pace; if the review pace is 35 s the total is ≈ 1.6 h. The 25 s is my estimate, not a measurement.
- Claude's labels are only as good as the 50-pair check: with n = 50 the kappa CI is about ±0.1, so a kappa of 0.7-0.8 is
  weak evidence the labels are better than 0.6. The task fixes the 0.7 bar; I report n and the table with every value.
- Cost of `judge` (Opus 5.5, $4/$20 per MTok; unverified price, see `bench/llm.py`): ≈ 700-900 calls × (≈ 700 input +
  ≈ 200 output tokens) ≈ $4-8; the cap is ≤ $25 and the ledger stops before a call that could cross it.
