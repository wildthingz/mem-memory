# 0016: Default per-node cap 2 → 4 (new fingerprint)

- **Status:** Accepted (requested and approved by the owner, MEM2-1 comment of 2026-10-25 05:52, 1/2, T2)
- **Date:** 2026-10-25 · **Task:** MEM2-145 · **By:** Core
- Spec: architecture-v2 §3.2 step 4, §3.3. Code: `src/mem/config.py` (`retrieval.per_node_cap`), `src/mem/read.py` `candidate_pool`.

## Context
Step 4 collapsed the fused list to at most 2 chunks per node before taking the top 50. On the LME tune (115 answerables)
this capped out the gold round (often round 0): of the 16 pool misses at cap 2, 11 had the gold session in the pool with
only 2 of its chunks.

## Decision
`retrieval.per_node_cap` defaults to 4. Pool stays 50 (the owner's decision). With cap 4, pool 30 gives the same LME recall,
and B-C1 at pool 50 measured warm p95 738 ms with rerank 710 ms.

## Evidence (host, LME tune, 115 answerables)
| cap | pool recall (same at pool 30 and 50) | best coverage_any |
|---|---|---|
| 2 | 0.861 | 0.852 |
| 3 | 0.922 | 0.904 |
| 4 | 0.939 | 0.913 |
| 6 | 0.948 | 0.922 |

P2 tune gate at cap 2 / 4 / 6 (pool 50, exact tokens): recall_any5 0.930 / 0.939 / 0.948; nDCG superior at all three;
B-R5 0 failures; B-R6 0 violations; B-C2 mean tokens 180 / 152 / 157, all passing.

## Alternatives considered
- **Keep 2:** loses 11 of 16 misses to the cap alone.
- **Cap 6:** +0.009 pool recall over 4, but one long vault note could fill 6 of 50 slots. Revisit on PG-v1 tune
  (`deliverables/runbooks/p4-gate.md` §4b).
- **Cap 3:** 0.922, below 4 at no benefit.

## Consequences
- `per_node_cap` is in `fp_all`: golden fingerprint is now `1ed3420b…85c6` (was `1d78f44e…40a3`). **Every gate fit and
  result made at cap 2 is void; the gate must be re-fitted and re-gated** (`p2-pool.md`, `p3-gate.md` §2 re-score).
  `fp_embed` is unchanged, so stores and vec tables need no re-embedding.
- More chunks per node reach the reranker; final output still collapses to one row per node.

## Addendum (2026-10-25, MEM2-164): default pool 50 → 30
- **Decision (the owner, 2026-10-25):** `retrieval.pool` defaults to 30. At per-node cap 4, pool 30 gives the same LME recall as 50
  (0.939) and about 285 ms less rerank time. This confirms the table above ("same at pool 30 and 50") and reverses the
  2026-10-24 raise to 50 (MEM2-128) now that the cap, not the pool size, carries the recall.
- **Consequence:** `pool` is in `fp_all`: another new fingerprint (golden fingerprint in `tests/test_config.py` updated, with the
  read-guard default of ADR 0018). **Every gate fit and result made at pool 50 is void; the gate must be re-fitted and
  re-gated** (`deliverables/runbooks/p3-gate.md`, Round 6). `fp_embed` is unchanged: no re-embedding.
- Spec: architecture-v2 §3.2 step 4. B-C1 / B-C3 at L50k were measured at pool 50 and are re-measured at the final config in Round 6.
