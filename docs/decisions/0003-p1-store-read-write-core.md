# 0003: P1 store, remember, recall and get

Status: accepted · Task: MEM2-9

## Decisions
- **Layout.** `mem.store.Store` owns the connection, embedder, clock and `events` log. Tool logic is
  in `mem.write.remember` and `mem.read.recall` / `mem.read.get`; `Store` exposes them as methods
  returning `ToolResult(text, structured)`, which `mem serve` (p1-serve) will wrap.
- **Transactions.** `Store` sets `isolation_level=None` and the write path issues `BEGIN IMMEDIATE`
  itself; the exact-duplicate check is the `nodes_mem_hash` conflict inside a savepoint (B1).
- **Unmatched plain `about` names** are created as `topic` entities (§3.1 step 6; the §4.1 example
  `1:1s → e88 (new, topic)`), not rejected. Two or more exact alias matches → `needs_choice`, nothing written.
- **`supersedes` on `remember`** raises a clear error until P3 (chain re-derivation is P3).
- **P1 recall** = RRF top `min(limit, 5)`, header `uncalibrated (P1 build)`, `p` carries the RRF score.
  `about` with several entries is an OR (provisional; §3.2 does not say). Empty result: `no results · <scope>`.
- **Time zone.** `Store(tz=...)` (default UTC) converts date-only input; no config key exists yet.
- **Vec table fingerprint** for fake embedders = sha256(class name + dim); real runs pass `fp_embed`.
- **get continuation** offsets are character offsets (`c812@1840`); edges use `e45@edges:10`.

## Rejected
- Applying scope after LIMIT (v1): misses eligible rows (A7); both legs use `chunk_id IN allowed`.
- Auto-rejecting unknown plain names in P1: would make the first `remember` of any subject need `new:`.
