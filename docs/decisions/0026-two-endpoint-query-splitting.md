# 0026. Two-endpoint query splitting: a pool, a rerank and a gate per sub-query (MEM2-213)

Status: accepted, switch default **off** until the adoption rule below is met on TUNE. (Option A, the
date-window leg, was dropped by the owner on 2026-10-26: the dry run falsified it, 6/18.)

## Context

An ordering or duration question names two events ("how many days between A and B", "which came
first, A or B", "how long had I been A when B"). One query for both events is a poor match for either:
the pool and the reranker favour the event the wording leans on, and the other endpoint's rows fall
under the gate or out of the pool. The dry run (MEM2-209, `deliverables/research/option-ad-dryrun.md`)
counted how often a deterministic pattern finds the two events: **18 of 38** tune temporal questions
(threshold 10; 17 genuine, one misfire "with a friend or not"), with 10 false splits on the other tune
questions ("music albums or EPs", "my previous goal ... before I updated my goal").

## Decision

`read.split_endpoints` (config.yaml, default `false`). When on, and the store has a reranker, and the
question is not one list mode takes (ADR 0023 wins), `mem.splitq.split_endpoints` looks for two events:

1. `between A and B`; 2. `how long/many .. had/did I been A when/before/until/after B`;
3. `since A when/before/until B`; 4. `<clause> before/after/when <clause>` (sides of at least 4 and 3
words); 5. `A or B`, in that order, first match wins. The patterns are the dry run's, **frozen**: no
re-tuning, no type labels, no dataset cue lists, no LLM; no pattern, no split.

For a match, recall runs **per sub-query**: the whole candidate pool (eligible set, both legs, RRF,
per-node cap, `pool` = 30 chunks), the rerank against the sub-query, p (Platt or read model, tool
weighting) and the gate (`select`, so each sub-query abstains or keeps by its own p). Then
`merge_endpoints`: a chunk both kept counts once (the higher p); at most `per_node_cap` chunks per
node; taken round-robin from each sub-query's kept list up to `limit`, so each endpoint's best row
survives the cut; **ordered by date** (undated last). Steps 7-8 (flags, token budget, conflict guard)
run on that list unchanged. If neither sub-query keeps a row the answer abstains. The result carries
`structured["split"] = {pattern, parts, kept: [n1, n2]}` (also in the events log), and
`timings["split"]` is the whole split stage in ms (B-C1: two pools and two reranks instead of one).

Everything else takes the normal path unchanged.

## Switch off

The key is absent from the default `read` section; the split is never tried and `recall` returns
byte for byte what it did (tests: `tests/test_split_endpoints.py`, matching questions included).

## Fingerprint

`read` gains `split_endpoints: true` **only when on**, so off is the existing golden (`55f53d2e…`);
on, `fp_all` is `72fe0e95…` and a gate has to be fitted for it (golden test in `tests/test_config.py`).
`fp_embed` is unchanged. The sub-queries change which p the gate sees, so the gate is refitted rather
than reused.

## Adoption rule (the owner's; for the host pass)

Round 9, on TUNE only, paired-bootstrap CIs: B-E1 tune at or above the baseline (the no-regression
floor) with no question type down more than 5 points; P2 hard rows, B-C1, B-C2 and P3 rows unchanged.
**Main proxy:** recall-all@8 on the matched temporal tune questions (both endpoints' gold rows in the
delivered 8) up against the switch off. B-C1 is counted with the extra rerank (warm p95 as in ADR 0022:
L5k ≤ 1,500 ms, L50k ≤ 1,200 ms). Only then does the default change, with its own fingerprint and gate.
Watch the false splits: the 10 above run an extra pool and rerank for nothing and may drop a row the
single query kept.

## Consequences

- Cost: a split question runs the pool stage and the reranker twice.
- A state question ("my previous goal ... before I updated") splits though it needs the older value,
  not an ordering; the per-sub-query gate keeps it harmless where both pools agree, and the false-split
  count is part of the adoption read.
- `bench/temporal_dryrun.py` imports the patterns from `mem.splitq`, one copy.
