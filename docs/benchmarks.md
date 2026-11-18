# Benchmarks

What has been measured, how far to trust it, and how to reproduce it with public data.

## Read this first

- Every number below was measured on **one machine** (an RTX 4080 SUPER desktop, 4 CPU threads for the
  CPU runs), by the project's author, in October 2026.
- The [final results](#final-results-end-to-end-question-answering) are one run each on the full public
  benchmarks, at the shipped configuration. Everything under
  [component measurements](#component-measurements) comes from the **tune** split and from
  informational runs while the system was being built; those thresholds are provisional floors, not
  promises, and rows marked *provisional* may move as the code changes.
- Where a target was missed it is listed as missed.

## Final results: end-to-end question answering

Each question is answered by `claude-sonnet-5-5` from what `recall` returns (the production read path,
shipped configuration and gate), then graded by `claude-sonnet-5-5` with the benchmark's own judge
prompt. Every run is a single pass; 95% confidence intervals in brackets.

| Benchmark | Score | Reference arms | Date |
|---|---|---|---|
| **LongMemEval-S**, 500 questions (~50 sessions each) | **82.6%** (413/500) [79.2–85.6]; 84.0% with the 22 judge replies that were cut off re-judged | no memory 7.6%; oracle (gold sessions only) 92.4% | 2026-10-29 |
| **LoCoMo**, categories 1–4 (1,540 questions), LLM judge | **82.4%** [80.5–84.2]; token F1 0.505 | | 2026-10-29 |
| **LongMemEval-M**, 500 questions (~500 sessions each) | **73.8%** (369/500) [70.0–77.2]; 75.6% with cut-off judge replies re-judged; recall_any@5 0.943 | | 2026-10-30 |

By question type:

| LongMemEval type | S | M |
|---|---|---|
| single-session (assistant) | 96.4% | 94.6% |
| single-session (user) | 94.3% | 90.0% |
| knowledge update | 89.7% | 92.3% |
| multi-session | 78.2% | 58.6% |
| temporal reasoning | 74.4% | 64.7% |
| single-session (preference) | 66.7% | 56.7% |
| abstention (correctly says "not known") | 0.967 | 0.933 |

LoCoMo by category: single-hop 89.8%, temporal 83.5%, multi-hop 67.7%, open-domain 57.3%. Category 5
(adversarial, 446 questions) is left out of the judged score, as in the published papers: under the
official string rule it scores 0.9%, because the answerer rejects the false premise in its own words
instead of the exact phrases the rule matches.

What to take from it:

- **Weakest areas:** preference questions, and evidence spread over many sessions. Going from S to M
  (10× more sessions per question) costs 8–9 points overall, almost all of it in multi-session (−20)
  and temporal (−10) questions.
- **The judge fix.** The judge reasons before its verdict and sometimes hit the 64-token reply cap;
  those replies count as wrong in the official number. Re-judging only them with a 512-token cap (same
  judge, prompt and answers) gives the second figure. The official number is the first one.
- On the tune split the end-to-end score went from 0.752 to 0.832 over the last two tuning rounds
  (relative dates resolved at ingest, list mode with list fill, the date-window boost, and the
  Qwen3-Embedding-0.6B embedder: decisions 0021, 0023, 0028, 0030, 0031).
- Recall latency at the shipped configuration (pool 30, 4 chunks per node), warm p95, one client:
  **452 ms** on a 5,000-chunk store and **540 ms** on 50,000 chunks (2026-10-29).

## Component measurements

Measured while the system was being built, before the final configuration; kept for the record.

| What | Number | Date | Status |
|---|---|---|---|
| Recall latency, warm p95, 5,000-chunk store, one client, at candidate pool 50, 4 chunks per node (target ≤ 1,500 ms) | 738 ms; rerank 710 ms | 2026-10-25 | provisional; one B-C1 run, recorded in [decision 0016](decisions/0016-per-node-cap-4.md); 50,000-chunk latency is recorded separately and not summarised here |
| Recall latency at the earlier defaults, pool 30 and 2 chunks per node, warm p95, 5,000-chunk store, one client (3 fresh-process runs) | 451 ms in all three runs; full-text leg median 1.2 ms, p95 2.7 ms; rerank about 425 ms. Four concurrent clients: total p95 about 1,797 ms | 2026-10-24 | superseded; the full-text and four-client figures were measured only at pool 30 |
| Candidate pool recall (answerable LongMemEval-S tune queries, n = 115: the share whose evidence is in the 50 candidates the reranker sees) | 0.939 (108/115, 95% CI 0.880–0.970); the same at 30 candidates | 2026-10-25 | provisional |
| Reranker quality (AUROC of the reranker score, answerable vs. not, LongMemEval-S tune) | 0.969 | 2026-10-24 | provisional |
| Recall answer size (300 sampled responses, tune): mean tokens, p95 tokens; reduction vs. returning the top 8 full chunks | 91, 196; 0.987 | 2026-10-24 | provisional; token counting by estimate unless an Anthropic key is set |
| Write-path latency (`remember`), warm p95, dev seeds | 70 ms | 2026-10-25 | provisional |
| Spike, reranking 30 pairs: GPU p95 | under 1,000 ms (the run passed its limit; the exact figure was not recorded) | 2026-10-23 | measured once |
| Spike, reranking 30 pairs on 4 CPU threads | 10.2 s p50 / 10.25 s p95 (average-length pairs); embedding 1.58k tokens/s | 2026-10-23 | measured once. About 11 minutes for a 1.04M-token vault import on the CPU |
| Gate fit time, LongMemEval-S tune (230 queries) | 551 s (about 9 minutes) | 2026-10-24 | measured once |
| Test suite on the merged tree | 1,267 passed, 4 skipped | 2026-10-25 | on the tree this copy was cut from |

### Targets that were missed

These are stated here because the docs would be misleading without them.

- **Delivered precision on LongMemEval.** The relevance-gate targets (precision ≥ 0.70 together with
  coverage ≥ 0.85) cannot both be met on the LongMemEval tune split: the best fit gave precision
  0.71–0.79 at coverage 0.30–0.39, and precision 0.14 at coverage 0.85 (2026-10-24). LongMemEval labels
  whole rounds, so on-topic neighbouring chunks count as wrong. Those rows are now reported as
  descriptive, and precision is to be judged on a personal question set instead (not yet done).
- **Synthetic read-path checks** (benchmark B-R5, dev seeds, 2026-10-25), accepted as known misses:
  current-fact recall@1 0.856 (target ≥ 0.90), as-of-date accuracy 0.935 (target ≥ 0.95), stale-fact
  leakage about 0.004 (target 0). A diagnosis found the limit is ranking, not the gate: for 10.6% of
  current-fact queries the right memory ranks below the first 237, so no gate can fix it.
- **Missing-slot abstention** (answering "no" when the memory exists for the subject but not for the
  asked relation; B-R2 (ii)): best 0.10 live, 0.30 under nested cross-validation, target ≥ 0.90.
  Still open when this was written.
- Of the nine write-path gates run informationally on the dev seeds on 2026-10-25, five passed (B-W1,
  B-W2, B-W6, B-R1, B-C3) and four did not.

## What the benchmarks are

The harness is `bench/` (run it with `uv run bench ...`). Its ids:

| Id | Question |
|---|---|
| B-R1, B-R2 | Retrieval and abstention on LongMemEval-S: recall@5, nDCG@10, delivered precision and coverage, and how often a question with no answer correctly gets no rows |
| B-R5 | Current and as-of recall, and stale-fact leakage, on synthetic (SYN) histories |
| B-R6 | The response format: every reply obeys its shape and token limits |
| B-W1, B-W2, B-W3, B-W5 | The write path: merging, duplicates, supersession, and the skill's behaviour in scripted dialogues |
| B-C1, B-C2, B-C3 | Cost: recall latency, tokens per answer, write latency |
| B-S1, B-W6 | Fixtures: the store and write contracts |

## Reproduce

Everything is run from a checkout, with `uv run --extra models` on a machine that meets the
[requirements](install.md).

1. **Unit tests and the first gate, on any machine, no GPU or network** (stand-in models):

   ```bash
   uv run pytest -q
   uv run bench gate P1            # PASS / FAIL / UNAVAILABLE per gate id
   ```

   `UNAVAILABLE` counts as a failure, never a pass. `bench/README.md` describes the layout.

2. **Get the public data.** LongMemEval (MIT, see [THIRD_PARTY.md](../THIRD_PARTY.md)):

   ```bash
   uv run --extra models bench fetch --which s --which m --dest data/lme
   uv run --extra models bench build-lme data/lme/longmemeval_s_cleaned.json
   ```

   `bench fetch` pins each file's sha256 and size and fails on a mismatch. `bench build-lme` freezes the
   tune/test split and the evidence-removed negatives and checks the counts (500 questions, 151 tune,
   349 test, 115 tune and 269 test answerable in B-R1).
   A 30-pair hand audit of the negatives is part of the full procedure (`bench audit-apply`); without
   it the numbers differ slightly from the ones above.

3. **Fit the gate** on the tune split, as in [install.md](install.md#fit-the-relevance-gate-once).
   Take the time of the first `mem calibrate` as your gate-fit time.

4. **Latency at 5,000 chunks**, three fresh-process runs (stop your own `mem serve` first: the bench
   starts its own, and one copy of the models fits in about 5 GB):

   ```bash
   uv run --extra models bench build-corpus data/lme/longmemeval_m_cleaned.json --size L5k
   for i in 1 2 3; do
     uv run --extra models bench latency-run --lme-file data/lme/longmemeval_s_cleaned.json \
       --lme-m data/lme/longmemeval_m_cleaned.json --corpus L5k --gate data/lme/gate.json \
       --out data/lat/L5k-$i.json
   done
   ```

   The pass condition is the median of the three runs' warm p95. `--size L50k` builds the large corpus.

5. **Retrieval on the tune split** (informational, not a test look). Set `ANTHROPIC_API_KEY` for exact
   token counts, or pass `--tokens est`:

   ```bash
   uv run --with anthropic --extra models bench gate P2 --split tune \
     --lme-file data/lme/longmemeval_s_cleaned.json --gate data/lme/gate.json \
     --latency-run data/lat/L5k-1.json --latency-run data/lat/L5k-2.json --latency-run data/lat/L5k-3.json
   ```

   It prints one line per gate id and a per-question-type table. `bench/results/` holds the run
   records; `bench/ledger.jsonl` logs looks at the test split.

Do not tune against the `test` split: each look at it is logged, and the project allows at most three
per phase. Fix things on tune, and look at test once.

The write-path and synthetic checks (`bench gate P3`, `bench syn-*`) need the synthetic generator's
dev seeds and, for the skill replay (B-W5), an Anthropic API key and a budget; `bench --help` lists the
commands and `bench/README.md` the seeds and rules.
