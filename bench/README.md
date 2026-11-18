# bench: benchmark harness (benchmarks-v2 §6)

Re-run everything the P1 gate needs, from a clean checkout, CPU only, no network:

```bash
uv run pytest -q                 # includes tests/bench_t0: metrics, harness, suite self-checks, P1 gates
uv run bench gate P1             # PASS / FAIL / UNAVAILABLE per gate id; writes bench/results/*.json
uv run bench run --suite B-R6    # one suite
```

- `suites/` one module per benchmark id; each returns a `SuiteResult` (named checks → violations → metrics).
- `configs/gates.yaml` phase → gate rows. Only thresholds written in benchmarks-v2 appear there.
- `adapters/ours.py` the single seam to production code (`mem.serve.open_inprocess`, `list_tool_schemas`).
  If `mem.serve` cannot be imported the gates report `UNAVAILABLE` (a failure, never a pass).
- `fixtures/streams.py` hand-written op streams and corpora. `dump.py` canonical store dump.
- `metrics/` std nDCG and the upstream `ndcg_any` kept apart (A6), recall/MRR, delivered-context
  metrics, token estimate. `records.py` run records and `ledger.jsonl` (test looks, `--ledger "<why>"`).
- Human-run legs (spike, HTTP/stdio, `claude -p`): the project's private P1 runbook (not included).

## P2 (read quality) additions

```bash
uv run pytest -q                      # T0: manifest, metrics, P2 suites on fake models, freeze/looks
uv run bench gate P2 --split tune ... # informational T1 run; --split test is a logged look
```

- `datasets/` LongMemEval layout, frozen manifest, negatives + audit, L5k/L50k plans, tune labels (README there).
- `lme_eval.py` run + score (B-R1, B-R2 (i), B-C2 metrics, CIs); `latency.py` the B-C1 process protocol.
- `suites/b_r1.py b_r2.py b_p2_cost.py` the P2 forms of B-R1, B-R2, B-R6, B-C1, B-C2 (`PHASE_REGISTRY`).
- `adapters/lme_ours.py` the single seam to the P2 pipeline (`ingest_session`, `recall_lists`); without it
  the gates report `UNAVAILABLE`.
- `freeze.py` `bench freeze P2` (freeze-on-tune) and the ≤ 3 looks rule; a test look needs frozen rows
  and `--ledger`.
- Human-run legs (download, calibrate, hand audit, GPU, `count_tokens`): the project's private P2 runbook (not included).

## P3 (cleaner saving) additions

```bash
uv run pytest -q                  # T0: SYN generator + golden file, alignment rules, P3 suites on the stand-in
                                  # server (red-green per property) and the exact ones on the real store
                                  # (test_p3_real.py), B-W5 harness with scripted writers
uv run bench gate P3 --split tune --embedder hash --gate <stub gate.json>  # runs on the stub models; the
                                  # SYN-quality gates FAIL there by design (real models: runbooks/p3-gate.md)
```

- `synth/` SYN (benchmarks-v2 §3.2): `data.py` versioned relations, cue lists (disjoint), negations, filler,
  name/value lists; `generator.py` `generate(seed, size)` → ops, expected_state, queries (golden:
  `synth/golden/seed1_S.sha256`); `replay.py` the oracle client; `score.py` the B-W1/W2/W3 metrics (each states its
  numerator and denominator); `align.py` the B-W5 alignment rules; `dialogue.py` the 40 B-W5 dialogues;
  `baselines.py` the B-W1 baselines; `calibration.py` SYN dev pools and the θ_conf fit; `run.py` the shared replay.
- `suites/b_w123.py` B-W1/2/3, `b_w5.py` B-W5 (scores a run file), `b_w6_p3.py` B-W6 items 1-11, `b_r5_p3.py` B-R5
  cases (a)-(f) + SYN probes and B-R2 (ii), `b_c3.py` B-C3 (`PHASE_REGISTRY["P3"]`).
- `agent_replay.py` + `llm.py` the writer loop, the Anthropic client and the B-C5 spend ledger with a hard budget.
- `adapters/p3_ours.py` the real-store seam (`P3Backend`, `OursWriteServer` for B-C3).
- Human / paid legs (calibration, B-C3 runs, B-W5 run and audit): the project's private P3 runbook (not included).
  Decisions and definitions: `docs/decisions/0007-p3-benchmarks-syn-and-definitions.md`.

SYN gates use the worst of the held-out seeds 101-105 (§5.1: the pipeline is deterministic per seed, so there is no
bootstrap over dependent ops); dev seeds 1-5 feed calibration and iteration. Looks: a `w5-run --split test` is data
collection (no metric is printed) and is logged as `test-data`; the gate that scores it is the look.

## P4 (bring in your vault) additions

```bash
uv run pytest -q                  # T0: PG tool, the P4 suites on a stand-in importer (every check red-green via
                                  # mutants), B-R3 scoring with hand-worked values, `bench gold` through the CLI
uv run bench gate P4 --split tune ...   # informational T1 run; --split test is a logged look (inputs: p4-gate.md)
```

- `gold/` PG-v1 (benchmarks-v2 §3.1): `pg.py` model, exact quotas, sha1 split, validate, freeze, stale-gold check;
  `sampling.py` 40 notes + 10 linked pairs and the 200-draft slot plan; `draft.py` the Claude drafting call (budget cap);
  `check.py` the no-answer `rg` check and the pre-filter; `session.py` resumable review, blind kappa and label sittings;
  `pool.py` pooling; `judge.py` Claude grades the pooled pairs (T2, hard budget); `kappa.py` linear-weighted Cohen's kappa and the
  seeded sample; `bm25.py`; `cli.py` = `bench gold draft|check|review|pool|judge|kappa-sample|label|status|freeze|verify`.
  PG-v1 holds no work notes: `bench snapshot-make --pg` builds the PG snapshot (decision 0013).
- `pg_eval.py` B-R3 (nDCG@10 std graded on R_q, delivered metrics on S_q, CIs, path-level DR-4, unjudged@10);
  `baselines_p4.py` qmd and `rg -il`; `ref_parse.py` the independent parser of B-I3 / B-I4; `i5.py` B-I5.
- `suites/b_p4.py` B-I1…B-I5, B-C4, B-R3 and the V5k forms of B-C1 / B-C3 (`PHASE_REGISTRY["P4"]`; B-R1 / B-R2 reuse the P2
  forms on the re-calibrated gate). `p4_setup.py` builds the gate's `P4Setup`; `p4_cli.py` the other commands (`snapshot-make`,
  `vault-import`, `pg-score-tune`, `i5-sample|label`, `c4-run`, `qmd-run`, `rg-run`, `vault-latency-run`, `vault-c3-run`).
- `adapters/p4_ours.py` the single seam to the importer (`mem.importer.run_import`, two stores) and the read path;
  `adapters/p4_load.py` the V5k / V50k load stores; `datasets/vault.py` snapshot verify / extract / V50k filler plan.
- `fixtures/vault-mini.lifecycle.yaml` the B-I2 descriptor for the mini-vault of MEM2-31 (`tests/fixtures/minivault`).
- Human / paid legs (snapshot, import, drafting, sittings, qmd, B-I5, B-C4, calibration, latency, the gate):
  the project's private P4 runbook (not included). Decisions and definitions: `docs/decisions/0012-p4-benchmarks-pg-and-importer-checks.md`.

## P5 (two machines)

```bash
uv run bench e5-queries --out data/p5/queries.jsonl; uv run bench e5-ops --out data/p5/ops.jsonl
uv run bench e5-run ... ; uv run bench e5-reconnect ... ; uv run bench c6-run --out data/p5/c6.json
uv run bench gate P5 --e5-run <home.json> --e5-run <work.json> --e5-reconnect <rc.json> --c6-run <c6.json>
```

`e5.py` (client legs, parity, reconnect), `c6.py` (count_tokens overhead), `suites/b_p5.py` (the gate suites over the
recorded runs). Human-run legs: the project's private P5 runbook (not included).

## Round 10 additions

- `bench list-fill-dryrun RECORDS --delivered delivered.jsonl --gate gate.json --lme-file longmemeval_s_cleaned.json --out dryrun.json [--ns 12,16,20] [--cap 1000] [--no-refit] [--e1 e1-tune.json]`
  (MEM2-268, ADR 0030): the round-level list fill on recorded **tune** pools, no models. Per arm (today, `fill_N`
  for each N, the `tau_list` refit on gold rounds, and the research's `sim_top_N` / `sim_tau_T` for reproduction):
  every-gold-round delivery on the answerable multi-session questions, rows and cost-model tokens, abstention
  changes on the cued pools, the effect on the other list-mode questions. Question texts are streamed from the LME
  file for the pool ids only; a pool of a non-tune question or a gate with another fingerprint is refused.
- `bench window-dryrun --records tune_records.jsonl --delivered delivered.jsonl --gate gate-list.json --dataset longmemeval_s_cleaned.json --out DIR [--deltas 0.5,1,2,3] [--slacks 0.1,0.2,0.3,0.5]`
  (MEM2-266, ADR 0031): the anchored-window gate boost on recorded **tune** pools, no models: per (slack, Δ) cell the temporal gate
  cuts rescued, gold pushed out, net, non-gold rows admitted and negatives newly answered; `Best: … → pass|drop` (net < 2 drops).
- `bench e1-pair ARM.json BASE.json [--b 2000]` (MEM2-270): two `e1-run` files paired by question on their `mem` arms: accuracy,
  delta with a 95% paired-bootstrap CI and the question count, overall and per type; names every type down by more than 1 question.
