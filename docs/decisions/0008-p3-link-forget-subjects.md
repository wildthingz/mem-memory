# 0008 · P3 `link`, `forget`, `subjects`

Status: accepted · Spec: architecture-v2 §3.1 (A1, A11, N1, N2), §4.1, §4.3 · Code: `mem/link.py`, `mem/subjects.py`

## Decisions
- **Errors raise `MemError`** (MCP `is_error`, the message says the fix), like `remember`. The contract's
  `link` result keeps its `ok`/`error` fields, but `ok` is only ever `true`: a failed call is an error reply
  and the transaction is rolled back, so there is one error channel for all tools. No event is logged for a failed call.
- **Agent-callable rels** are every row of `rels` except `part_of` (importer only). Wrong endpoint kinds and unknown
  rels list the valid rels.
- **Imported nodes (`origin='import'`)**: may be the `dst` of `about`/`relates_to`/…, never touched otherwise.
  `supersedes`/`duplicate_of` refuse an imported end; any other rel refuses an imported `src`; `forget` refuses
  imported nodes; `remove` refuses imported edges. All say "edit the vault". `same_as` swaps when only `src` is imported.
  The check runs before the endpoint-kind check, so `supersedes` onto a note says "edit the vault".
- **`same_as` leaves a live `same_as` edge `src → dst`** after the merge (not in the spec's list, which only
  removes edges that existed). It records the merge and makes a repeat a no-op (`src` is then `duplicate`, so the idempotency
  check must find an edge, not look at status). Imported edges on `src` are re-pointed like any other (soft-removed only if
  they would duplicate a live `dst` edge or become a self-loop).
- **`remove=true`** soft-deletes the edge (`removed_at`) and re-derives the chains on both ends; a missing edge is an `ok` no-op.
  Removing a `duplicate_of`/`same_as` is refused: the merge also moved aliases and edges, which `remove` cannot undo faithfully.
- **`forget`** sets `status='retracted'`, `status_at`, stores the reason in `props('retracted_reason')`, re-derives the
  chain without it; repeating is an `ok` no-op. `valid_to` of the retracted row is left as it was.
- **`subjects`** lists active entities and tags; `memories` = active memories with a live `about` edge, `notes` = active notes
  with any live edge to it. A query matches names and aliases by normalised substring, ranked exact > prefix > contains,
  then by memory count.
- **Contract snapshot**: only `forget` changed (description; the `phase: P3` marker is gone). No parameter changed. The
  skill moves v0 → v1 anyway (§4.3), and `LATER_PHASE` gating in `mem serve` is removed.

## Rejected
- `ok:false` results for rule violations: two error channels for callers to handle, and half-applied merges
  would need hand-written undo instead of a rollback.
- Un-merging on `remove` of `same_as`/`duplicate_of`: needs a record of every moved alias/edge; a new `same_as`/`relates_to` plus `forget` covers real cases.

## Not done here
`mem calibrate` re-run with SYN dev (θ_conf) and the P3 T0 gate tests of the SYN generator step: `p3-syn` has no
commits beyond `main` yet, so neither exists to run. See `deliverables/runbooks/p3-link.md`.
