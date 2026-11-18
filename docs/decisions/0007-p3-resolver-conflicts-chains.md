# 0007 · P3 write path: resolver, conflict candidates, chains

Status: accepted · Spec: architecture-v2 §3.1 (A1, A5, A6, A8, A22, N1, N2)

## Decisions
- **Entity name chunks.** A created entity gets a `key_kind='name'` chunk (`chunk_key='name'`) with a
  vector, so the resolver's name-embedding leg (§3.1 step 5) has something to search. Recall already
  excludes `name` chunks. Entities made before P3 have none: they are found by exact alias and
  trigram only. The P4 importer must add the chunk for entity notes.
- **`THETA_NAME = 0.85`** (`mem/resolve.py`) is provisional: the spec names θ_name but gives no value
  and no calibration set. It only widens "ask" (never auto-links), so a wrong value costs an extra
  question, not a wrong merge.
- **Typo auto-link** exactly as §3.1: distance ≤ 1 (≥ 5 chars) / ≤ 2 (≥ 12 chars), unique, nothing
  else within 3. An auto-linked mention is not learned as an alias (only `mention → #id` learns).
- **Conflict candidates** run in phase A over `status='active'` memories at any time (closed
  intervals included, so an old attribute still surfaces). Memories with the same `content_hash` as
  the new text are never listed. Without a reranker or `gate.theta_conf` there are no candidates
  (an unfitted store); `mem calibrate` fits `theta_conf` later in P3.
- **Chains** live in `mem/chain.py` (`component`, `rederive`, `supersede`) so the `link` and `forget`
  tools reuse them. Edges are stored newer → older by (`valid_from`, `recorded_at`, `id`); a call
  in the other direction is swapped and reported. Reopen collisions (N2) raise a `MemError` naming
  both ids.
- **Vectors computed in phase A** (`Store.embed_chunks`), inserted in phase B; only name chunks of
  new entities are embedded under the lock.

## Rejected
- Fuzzy ratio libraries (`rapidfuzz`): v1's `token_set_ratio` is the bug A8 removes, and a 10-line
  Levenshtein avoids a dependency.
- Ranking conflict candidates globally only: crowds out old same-subject facts (A22).
