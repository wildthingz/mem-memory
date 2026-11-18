# 0019: The conflict scorer gets `same_type`, the store's own memory type, as a feature

- **Status:** Accepted (offline evidence; the live tune gate of round 5 confirms)
- **Date:** 2026-10-25 · **Task:** MEM2-156 · **By:** Core

## Context
After round 4 (SYN_VERSION 3, gate from `base`) B-W3 passes on the live tune gate (surfacing 0.967,
chain 0.961) and B-R5 leakage is 0.0089, from 10 missed updates on the dev seeds. Read against the
generator, 9 of the 10 share one shape: the list's best is a same-subject memory of *another slot* that
shares the filler sentence with the new text (`jaccard` 0.45–0.50, reranker logit −3 to +2.5), while the
counterpart states the same slot with another cue phrase (`jaccard` 0.1, cosine ≈ 0.5, reranker −4 to −6).
Every recorded text feature, including both models, prefers the wrong candidate; the tenth miss is the
`exists` cascade behind one of them. The cut cannot be loosened: in the round-4 fit's own grid, `delta` 10
at `top_k` 5 gives worst-fold surfacing 0.947 at precision 0.50 against 0.973 / 0.547 for `delta` 5,
because theta has to rise to hold the precision floor. The remaining lever is a feature that separates a
same-slot counterpart from a same-subject memory of another kind.

## Decision
Add `same_type` to `mem.confmodel.FEATURES`: 1 when the candidate's `nodes.type` (fact | preference |
event | decision | procedure | insight, set by the caller of `remember`) equals the new memory's type.
Computed in `mem.conflicts._features` from the store; the write path passes the new memory's type.
Offline nested CV on the round-4 host records (same rules, `base`), with candidate types aligned from the
generator by subject and valid date (7268 aligned; the 13442 candidates about other subjects were given
`same_type` 1, so the estimate is a lower bound on the live feature's separation):

| records | before (host round 4) | with `same_type` | in-sample misses |
|---|---|---|---|
| gated (g1) | 0.973 / 0.545 | 0.986 / 0.536 | 10 → 5 |
| clean | 0.973 / 0.532 | 0.980 / 0.535 | 10 → 4 |

(worst outer fold surfacing / precision; every outer fold ≥ 0.980.) Of the 10 round-4 misses the feature
takes 1/217, 2/818, 2/947, 3/820, 5/882; the `exists` cascade 1/544 goes with 1/217 in a live replay.

## Alternatives considered
- **A wider `delta` or no relative cut:** already in the grid (`delta` 10); it loses surfacing because theta
  rises to hold precision. No rule change reaches these rows.
- **Document-frequency-weighted overlap** (down-weight words common across the store's memories, so a
  shared filler sentence counts less): the fillers are a one-in-ten vocabulary, so the weighted overlap
  still favours the filler-sharing memory in every one of the 9 rows; it narrows the gap, flips none.
- **Chain position / chain length as a signal:** both candidates are open singletons or chain heads in
  these rows; nothing to separate.
- **A reranker prompt that asks "does the second statement replace the first?":** a model change, host
  only, not testable offline from the records; kept for the case the live gate shows `same_type` short.

## Consequences
- Generic by construction: the signal is a store schema field the caller sets, like the subjects behind
  `shared_subjects`; no cue list, template, relation id or value list. Said plainly: in SYN the type is a
  function of the relation table (fact 16, preference 3, event 2 of 21 relations), so on SYN it carries
  coarse slot information, exactly as a real caller's type does for a real slot. It separates kinds, not
  slots: 3111 of the aligned non-same-slot candidates are of the new memory's type.
- The feature set changes the fingerprint: dev records scored before this commit lack the feature and the
  fit refuses them by name (`fit_conf_model`); round 5 re-scores both record sets. The worlds do not change
  (SYN_VERSION 3), so no golden check and no read-gate refit.
- Four misses stay with every signal the store has: 2/748 and 2/915 (the counterpart names the subject by
  first name only, written from an `ambiguous_mention`), 3/424 and 3/800 (filler-dominated, same type). So
  B-R5 leakage falls but is not expected to reach 0 from this change alone.

## Revisit if
The live round-5 gate does not reproduce the offline gain (then the other-subject candidates' types, which
the prototype could not see, moved the fit), or a live store's callers do not set memory types consistently
enough for the signal to hold outside SYN.
