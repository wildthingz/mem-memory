# 0012: P4 benchmarks: PG-v1 definitions, B-R3 / DR-4, the importer checks, V5k

**Status:** accepted for the harness (MEM2-34). Spec: benchmarks-v2 §3.1, B-R3, B-I1–B-I5, B-C4, B-C1/B-C3 at V5k, §4 P4 row.
Thresholds are the spec's and nothing else: `bench/configs/gates.yaml` `P4`, pinned to the spec by `tests/bench_t0/test_p4_gate.py`.

## Decisions

**PG-v1 (§3.1)**
- *Quotas exact by construction.* A category's total is 3 × its tune count, and within a category the questions are
  sorted by `sha1(qid)`, first third tune. `bench gold review` stops showing a category at its total, `validate`
  refuses to freeze any other count, and `split.tsv` is re-derived on every load (a hand edit fails).
- *Gold* = `{path::chunk_key: rel}`, rel 2 answers, 1 supports. `qrels.tsv` also keeps the pooled chunks judged
  **0** (explicit rel 0 rows with the chunk hash): they are not gold, they are what lets us report how much of each
  system's top 10 nobody judged (pooling bias). A rel-0 row never makes a question stale.
- *Lexical overlap* of a question = |T(q) ∩ T(chunk)| / |T(q)|, best over its rel-2 chunks; T = lower-cased `\w+`
  tokens of 2+ characters minus a fixed stop list, no stemming. Control: median over answerable ≤ 0.5.
- *Reworded* = the person edited the drafted question text (a diff against the draft, recorded at review). ≥ 50%.
- *Two stores (Q3, decision 0010).* The vault lands in `personal.db` and `work.db`; a question is asked of the store its
  source notes were routed to (`Question.store`), as the two MCP servers are asked separately. Linked note pairs are
  sampled inside one store (an edge cannot cross stores). The three delivered/ranking metrics are micro averages over
  all 120 questions, each run on its own store.
- *No-answer checks:* `rg`-equivalent AND-of-key-terms per chunk in Python (exact, testable) + the person's BM25 top-20
  confirmation at review. A no-answer question that later gets a rel ≥ 1 label at pooling is voided and its quota reopens.
- *Pooling:* top 5 of R_q from ours, BM25 and dense, deduplicated, minus labelled; pair order is shuffled by
  `sha1(qid, id)` and the source system is never shown.
- *Draft model:* `claude-opus-5-5` at $4/$20 per MTok in `bench/llm.py` (**third-party price pages, 2026-10-24, not
  Anthropic's table**: confirm before trusting the ledger's dollars; the `--budget-usd` cap is the guard either way).

**B-R3 and DR-4**
- nDCG@10 (std, graded) only; the upstream `ndcg_any` is LME-only (A6) and not computed for PG.
- Delivered metrics are §2.0 verbatim over answerable + no-answer; coverage_all needs every rel-2 chunk in S_q.
- Baselines (BM25, dense, RRF no reranker, qmd, `rg -il`) are descriptive. qmd and rg return files or their own
  chunks, so the **DR-4 comparison is at path level for every system** (a path's rel = best rel of its gold chunks,
  ranking deduplicated by path in rank order, nDCG@10 std). qmd's point estimate ≥ ours → `dr4_reopen = 1`, recorded
  (a `record` row, never a pass/fail threshold). Ours is also reported at chunk level (the gate row `ndcg_std10_R`).
- qmd runs in its own named index (`--index pgv1`), hybrid `qmd query --json -n 10`, over the extracted snapshot.

**Importer checks (B-I)**
- *Seam.* The importer is `mem.importer.run_import(vault, stores, *, routes, render_dir)` (decision 0010 of MEM2-31);
  `bench/adapters/p4_ours.py:resolve_importer` is the one place that knows it. Missing → UNAVAILABLE.
- *B-I1:* canonical dump of nodes, edges, link_occurrences, chunks, aliases, props, **source_files** (sessions dropped)
  + sha256 of each fp32 embedding, of both stores; diff lines of two fresh imports. Run on the mini-vault and on the
  snapshot, each its own gate row.
- *B-I2:* cases (a)–(i) driven by a descriptor (`bench/fixtures/vault-mini.lifecycle.yaml`: the file to edit, the
  section's chunk key, the alias/tag/prop/link to remove…). "0 writes" = the importer's report is only `unchanged`
  and the full dump is identical. Agent rows (alias, prop, edge) are seeded before (g) and (h) and compared after.
- *B-I3 (independent parser):* `bench/ref_parse.py` (PyYAML + regex; python-frontmatter is not a dependency and is a
  thin wrapper over the same call). Definitions are in its docstring. Compared separately: notes, link occurrences
  (per (note, destination, rel) and in total), logical edges, tag edges, and frontmatter-derived notes. A note whose
  frontmatter is only `tags` leaves no store marker; its tags are covered by the tag-edge comparison. Links to
  non-note files resolve to a file stub carrying its path (decision 0010). On the mini-vault the parser gives 14
  occurrences / 11 edges / 9 tag edges, the same as the hand-worked `expected_s1.json`.
- *B-I4:* (1) chunk texts in `ord` order, joined, equal the normalised body **with whitespace runs collapsed**
  (chunk boundaries trim blank lines; any other difference fails); (2) `mem render` twice → byte-identical trees;
  (3) after `forget` of one memory the tree differs in **exactly one file**, that memory's.
- *B-I5:* 100 live import edges over both stores, seed 7, proportional to rel with a floor of min(5, n) per rel, the
  rest by largest remainder. A rel with 5 edges fails the 0.95 bar at one wrong edge: that is the spec's bar.
  The sheet is checked against a fresh seed-7 sample of the store (a stale sheet fails).
- *B-C4:* cold = import into fresh stores with the models already loaded (serve start is outside the clock),
  embeddings included; warm = the second import, which must report only `unchanged`. The device and threads are
  recorded; the spec says CPU.

**V5k / V50k (B-C1, B-C3)**
- V5k = the whole snapshot in **one** store (every folder routed to `personal`), because the cost unit is chunks in
  one store (A19); the live split gives two smaller stores, so V5k is the worst case. V50k = V5k + LongMemEval_M
  filler chunks to 50,000 in total, sorted session ids, `lme:load:<sid>` source refs (never gold). Queries: the 120 PG
  questions in sha1 order, cycled (n < 321, so the last calls repeat; the read path has no result cache).

## Open questions (asked of the maintainer)
1. **Cross-store links** (a `work/` note linking a `personal/` note) are unspecified. The reference parser resolves such a
   link to the target's path; if the importer makes a stub in the source store instead, B-I3 lists it. Decide what the
   importer does; the parser follows.
2. Confirm the readings above where the spec is silent: B-I4 (1) whitespace-insensitive, B-I4 (3) "exactly one file",
   DR-4 at path level, V5k as one store.
