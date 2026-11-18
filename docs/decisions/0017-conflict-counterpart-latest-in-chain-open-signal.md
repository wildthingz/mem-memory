# 0017: An update's gold counterpart is the latest member of its chain, and `open` is that store signal

- **Status:** Accepted
- **Date:** 2026-10-25 · **Task:** MEM2-156 · **By:** Core

## Context
Round 3 of MEM2-156 showed 11 of the 22 gated B-W3 misses were updates whose gold counterpart was
closed in the store (`open` = 0) while the scorer surfaced the open member. Two readings were on the
table: the owner's (the `open` feature is the wrong store signal; replace or augment it with "valid at the
new fact's valid_from" and/or "latest in its supersedes chain") and Core's (the SYN_VERSION 2 gold
named the member *written last*, which after a `late` update is a member the chain had already closed:
a label bug). The two candidate gold definitions disagree only after a future-dated update: "valid at
the new valid_from" names the closed-now predecessor, "latest in the chain" names the open future member.
Either link target yields the same chain, since `rederive` orders the whole component by valid time.

## Decision
The gold counterpart of an update is the slot's latest active member by valid time (SYN_VERSION 3,
commit 6d0efc2), i.e. the latest in its chain. In a correctly linked store that member is the one with
`valid_to` NULL, so the existing `open` feature *is* the generic "latest in its chain" signal; no
`valid_at` feature is added. The dev export carries each candidate's `valid_from`/`valid_to` and the
op's `as_of` so store-time signals can be tried offline from the records.

## Alternatives considered
- **Gold = member valid at the new valid_from, plus a `valid_at` feature**: the label would name a closed
  member after every future-dated update (75 of 750 dev updates are `future`), the feature would have to
  fight `open` there, and the records could not supply it offline (a closed candidate's `valid_to`
  against the new `valid_from` is not recorded). On the host records relabelled under the chain rule the
  gold counterpart is open in 1345 of 1345 gated records, so the signal already exists.
- **Add `chain_latest` as a second feature next to `open`**: identical to `open` in every correctly
  linked store; it only differs when a chain is mislinked, which the oracle replay never produces.

## Consequences
- The 18 relabelled dev records turn 11 gated misses into hits; nested worst fold on the gated host
  records goes from 0.960 / 0.551 to 0.973 / 0.535 (surfacing / precision), `same_instant` 0.875 → 1.000.
- Cost: the worlds changed (SYN_VERSION 3), so round 4 regenerates every record and refits; and `open`
  cannot separate a stale member left open by an earlier miss from the true counterpart (both open).
- Supersedes nothing: 0007-p3-resolver-conflicts-chains defines the chain, not the gold label.

## Revisit if
A live tune run shows an update whose gold counterpart is closed in a correctly linked store, or the
remaining hard rows (first-name-only counterparts, counterparts under a non-slot memory) need a
store-time signal that the recorded `valid_from`/`valid_to` fields can now be used to prototype.
