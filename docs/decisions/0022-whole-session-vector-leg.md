# 0022. Whole-session vectors and a session leg in RRF (MEM2-195)

Status: accepted, switch default **off** until the adoption rule below is met on TUNE.

## Context

A chat session or a long note runs far past the embedder's 512 tokens, so no single chunk vector
stands for it, and a question about the session's overall subject can match no one round well enough
to enter the pool while the session as a whole is plainly the right place. The dense leg only sees
chunks.

## Decision

`retrieval.session_leg` (config.yaml, default `false`). When on:

- **One vector per note node** (`nodes.kind = 'note'`: imported notes and chat sessions; a memory is
  one chunk already and gets none), in a new vec0 table `svec_<fp8>` next to `vec_<fp8>` (same
  fingerprint suffix, cosine, keyed by node id).
- **The vector is the L2-normalised mean of the node's chunk vectors** (chunks of kind body, section,
  fact, as stored in `vec_<fp8>`). Why the mean and not a re-embed of the whole text: the whole text
  would be truncated at 512 tokens (the tail of a long session would never be seen), a second embedder
  pass over every session doubles the import cost, and the mean needs only vectors already stored, so
  it is exact, deterministic, rebuilt from the database alone and cheap to keep current. Chunk
  vectors are unit length, so the mean's direction is the centroid; a node whose chunk vectors cancel
  to the zero vector gets none.
- **Kept current:** `write.ingest_session` and the importer recompute the node's vector in the same
  transaction as its chunks (so a changed or removed chunk changes it); `mem reembed` rebuilds all of
  them in the new space (`reembed(..., session_leg=True)`; `switch` refuses a config with the leg on
  and no `svec` table); opening a store with the leg on brings them up to date. Once the leg has
  been on, triggers on `chunks` record the node of every chunk inserted, updated or deleted in
  `svec_<fp8>_dirty`, with the switch on or off; opening with the leg on recomputes those nodes and
  any note without a vector, so a note changed while the leg was off is not ranked by the vector of
  text it no longer has. A store whose `svec` predates the tracking is rebuilt in full once.
- **The leg** (`mem.sessionvec.session_leg`): top `leg_k` sessions by cosine to the query among those
  with at least one chunk in the eligible set `allowed` (applied inside the scan, so before `LIMIT`,
  the same rule as the other legs). A hit contributes its eligible chunks, nearest to the query
  first, **at most `per_node_cap` per session**; the list is cut to `leg_k` chunks. It joins
  `[lexical, dense]` as a third list in the same RRF (`rrf_k` unchanged); the existing per-node cap
  and pool cut then apply as before. Scope, `since/until/type/kinds/about`, validity and quarantine
  are all in `allowed`, so the leg cannot surface a chunk the other legs could not.
- Recall timings gain a `session` stage only when the leg is on.

## Switch off

Nothing is created and nothing is read: no `svec` table (and, unless the leg has been on before, no
tracking table or triggers), the candidate pool and recall output are
byte for byte what they were (tests: `tests/test_session_leg.py`).

## Fingerprint

`retrieval` gains the key `session_leg: true` **only when on**, so off is the existing golden
(`55f53d2e…`) and stores and gates fitted before stay valid; on, `fp_all` is `d4a85f93…` and a gate
has to be fitted for it (`mem calibrate` on the leg-on config). `fp_embed` is unchanged, so the
chunk vectors are reused; `mem reembed` is needed only to build `svec`, and opening the store does
that anyway.

## Adoption rule (the owner's; for the host pass)

The same as F1 (ADR 0021): on TUNE only, B-E1 tune overall at or above the Round 8 baseline with no
question type down more than 5 points; P2 hard rows hold; B-C1 L5k warm p95 ≤ 1,500 ms and L50k ≤
1,200 ms; B-C2 holds for normal recall; P3 rows unchanged (accepted misses only). **Plus:** LME tune
R@5 ≥ 0.948 and pool recall ≥ 0.939 (the baselines), with the leg on. Only then does the default
change, with its own fingerprint and gate.

How to measure with the leg on: put `retrieval: {session_leg: true}` in the config used for the
run; `mem calibrate --labels … --records-out rec.jsonl` writes the pools, `bench pool-recall
--records rec.jsonl` reports pool recall, and the B-R1 run reports R@5 and lists the leg as
`legs.session` beside bm25/dense/rrf. (T0 proof with the hash models: `test_pool_recall_and_br1_run_with_the_session_leg_on`.)

## Consequences

- Schema unchanged (the table is per fingerprint, created at runtime like `vec_<fp8>`). A store with
  the leg on holds one extra float vector per note; sessions are few next to chunks.
- A session close to the query pulls in rounds the other legs ranked low; the per-session cap and the
  pool/rerank stages bound the cost. Whether it helps is empirical: the numbers above decide.
