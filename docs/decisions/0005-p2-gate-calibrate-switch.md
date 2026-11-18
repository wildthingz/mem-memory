# 0005: P2 read path: reranker, gate, calibrate, refuse-on-mismatch, reembed/switch

Status: accepted for review · Task: MEM2-17 · Spec: architecture-v2 §3.2 steps 4–9, §3.3, §6 P2

## Decisions
1. **Where the gate lives.** `mem.gate.Gate` is `meta.gate` (JSON): `platt_a`, `platt_b`, `tau_keep`,
   `tau_high`, `gap`, `theta_conf` (null until P3), `fit_id`, `fitted_at`, `fingerprint_hash` (= `fp_all`)
   and `tune` (achieved F0.5, precision, coverage_any, IRA, FAR, abstention recall, queries, §3.3 step 4).
   `Gate.select(ps)` is the one place the rule is written; `recall` and the calibrator both call it.
2. **Abstention edge cases** (not spelled out in §3.2): a one-row pool has `p2 = 0`; an empty pool
   abstains with `no relevant memories · <scope>` (no `best`). The abstention text is exactly that line
   (B-R6), so dropped `about` filters go only to `structured.filters_dropped` there.
3. **Store with or without a reranker.** `Store.open(reranker=None)` keeps the P1 slice (RRF top 5,
   `uncalibrated`) for library callers: unit tests and the P1 T0 bench suite, which MEM2-16 moves to the
   P2 form. With a reranker, `recall` refuses without a gate (`run mem calibrate`). `mem serve` always
   passes a reranker and runs `preflight` first, so nothing served is ever ungated (A2).
   Rejected: removing the P1 path now (breaks the P1 T0 gate before its P2 replacement exists).
4. **Refuse on mismatch** = `mem.serve.preflight(stores, cfg)`, read-only, before any model loads:
   missing store, `meta.fingerprint.embed ≠ fp_embed`, missing/incomplete gate,
   `gate.fingerprint_hash ≠ fp_all`, or a non-null `gate:` value in config.yaml that differs from
   `meta.gate`. Each line names the fix. `open_inprocess(cfg=…)` applies the same check for bench.
5. **Stub config.** `config.stub_config()` gives the CPU stand-ins (HashEmbed-256, OverlapReranker) their
   own `fp_embed`/`fp_all`, so `mem serve --embedder hash` goes through exactly the same refusal path.
6. **Calibrate is split in two (open question to the maintainer, comment on MEM2-17).** `calibrate.score_pool`
   (steps 1–5, gate off, no event, never an MCP tool) is called in-process by the harness;
   `mem calibrate --records` fits from saved pools and loads no model (D5). Records are JSONL with a
   header `{"fp_all": …}`; calibrate refuses records scored under another config. It writes `meta.gate`
   only into stores whose vectors match the config; otherwise `--out gate.json` for `mem switch`.
7. **Platt.** scikit-learn `LogisticRegression` (DR-1 lists it), unpenalised (`C=inf`), with Platt's
   smoothed targets (N₊+1)/(N₊+2), 1/(N₋+2) via sample weights, so a separable tune set still gives finite
   coefficients. It was already in `uv.lock` (sentence-transformers pulls it); now a direct dependency.
   Rejected: a hand-written Newton solver in numpy (less code to review is the spec's library).
8. **Threshold search.** τ_keep: candidates 0 and every observed p (`p ≥ τ`), max F0.5 under the 0.02
   coverage-loss bound, ties to the lower τ. τ_high and g: cut points midway between observed p1 / gap
   values (`<` tests), max abstention recall under FAR ≤ 0.05, ties to higher precision then the milder
   rule. S_q in calibration = kept rows, top `limit`; the budget is not simulated (8 rows fit 600 tokens).
   Vectorised with numpy so 230 tune queries × ~230² (τ_high, g) pairs run in well under a second.
9. **Flags.** Memories only. Current reads: `until <date> → m<next>` where `next` is the following member
   of the `supersedes` component ordered by (`valid_from`, `recorded_at`, `id`), as §3.1 derives chains.
   History reads: `[from – to]`, open end `[from – ]`, empty interval `superseded same instant` (N1).
   Dates in flags are local dates in the store's zone.
10. **Budget.** The whole response text (header, rows, dropped-filter lines) is estimated with
    `ceil(utf8/3)` before a row is added. If not even one row fits: `budget N too small for one row`.
11. **Reranker loading** follows the day-1 spike (MEM2-2, verdict `pass`): fp32, `device: auto` (CUDA on
    the desktop), TF32 off, raw logits (`Identity` activation), `max_length` 512. `device` stays outside
    the fingerprint (parity held).
12. **reembed/switch.** `reembed(conn, embedder, fp)` fills `vec_<fp8>` next to the old table (resumable).
    `switch(path, cfg, gate)` refuses a gate for another `fp_all` and a table missing any chunk's vector,
    then writes `vec_table`, `fingerprint`, `gate` in one `BEGIN IMMEDIATE`. CLI: `mem switch` and
    `mem reembed --store … [--config] [--device]` (added in MEM2-64 for review finding F5).
    **Deviation from D5's letter:** `mem reembed` loads the new embedder in-process, because the running
    `mem serve` holds the old one and refuses to start on the new fingerprint, so there is nothing to
    call. It is a rare, one-off, resumable step; `--device cpu` keeps it off the GPU while serve runs.
    `--embedder st` always loads the pinned model, so it refuses a config whose `embedder` section
    differs from the pins (writing pinned vectors under another fingerprint would mix vector spaces).
    Rejected: a store-free, gate-free model endpoint on `mem serve` (new serving surface for a one-off).

## Not changed
No schema migration (meta is key/value). P1 tests unchanged except the four stdio `mem serve` tests,
which now pre-gate their store (P2 refuses an ungated store by design).
