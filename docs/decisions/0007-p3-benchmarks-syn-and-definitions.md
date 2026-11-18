# 0007: P3 write benchmarks: SYN, metric definitions and what the harness does not assume

**Status:** accepted for the harness (MEM2-23). Product-side consequences are flagged as questions at the end.

**Context.** benchmarks-v2 §3.2 and B-W1…B-W6, B-R5, B-R2 (ii), B-C3 leave several definitions open. A metric
that others build gates on needs its numerator, denominator and unit written down before it is computed. These
are the choices, each with the reason. Everything is implemented in `bench/synth/` and `bench/suites/`.

**Decisions**

1. **Scenario ops, not tool calls.** `ops.jsonl` holds scenario ops with mention *styles*; the oracle driver
   (`bench/synth/replay.py`) turns them into `about` strings once it knows server ids. The gold
   (`expected_state.json`) is computed by simulating the spec's semantics (chain re-derivation ordered by
   `valid_from, recorded_at, id`; half-open intervals) independently of any server. Cross-check: the T0 stand-in
   server (`tests/bench_t0/fake_p3.py`, written from architecture-v2 §3.1/§4.1) reproduces the gold exactly on SYN-S
   seeds 1-5, 101-105 and SYN-M 101-103 (0 graph differences).
2. **"Half of the updates carry a cue"** = a change prefix (`Update:`, `Now`, …); every text has its relation cue
   regardless (B-W5 alignment needs it). Exactly half, alternating.
3. **Sub-type quotas** (value_returns 20%, future 10%, late 5%, same-instant 5%) are `round(share × updates)`,
   at least 1 for late and same-instant, so a small seed still has them. Op-kind quotas are exact when feasible;
   an infeasible kind becomes `new_fact` and `counts` records what was generated. A world too small for its op
   count raises (it never silently produces fewer ops).
4. **More relations than the spec names** (12 → 19 over four entity types) so that 520 `new_fact` ops (SYN-M) have
   free (subject, relation) slots. Cue lists are disjoint (no cue equals or contains another relation's cue).
5. **Introduction ops** are written `alias → #e<id>` directly, not "bare alias, wait for the ask, then answer".
   Unique first names are never used alone (a server cannot know them without a fuzzy subset rule, which A8
   forbids auto-accepting); first-name-only mentions exist only for colliding first names and cross-kind names,
   as `ambiguous_mention` ops the server must ask about. The oracle answers an ask with `#e<id>` (no alias taught).
6. **B-W1 scored mentions** = resolver decisions (canonical, alias, typo, ambiguous). Explicit `new:`, `#id` and
   introductions are not scored. `ask_rate` therefore excludes introductions as the spec says.
7. **Surfacing counterpart** = the slot's most recently *written* active member (not the neighbour in valid time),
   so late and same-instant updates are judged on the same footing. **List precision** = candidates sharing the
   gold subject AND relation / candidates returned, over `update` and `dup_paraphrase` ops (the ops with a gold
   counterpart); the all-ops variant is reported, not gated.
8. **Typos** are 1-edit on names ≥ 5 chars as the spec says; a deletion can leave a 4-char surface of a 5-char
   name. The harness does not soften this (a resolver with its own length floor on the *mention* will split).
9. **B-W5 is scored offline from a run file.** `bench w5-run` is the only paid step; `bench gate` recomputes
   every metric from the file (deterministic). Dialogues use whole entities so slot timelines and updates stay
   intact; late/future/same-instant updates (not expressible with one dialogue date) and value_returns are excluded.
   Definitions: *capture* = gold facts with an aligned memory recorded at/after their dialogue; *noise* = memories
   aligned to no fact, or carrying an `ignore` marker; *about-fill* = memories that name a gold entity in their text
   and have it on a live `about` edge / memories that name one; *conflict_action* = surfaced true conflicts acted on
   with `supersedes` (either direction or inline); *wrong invalidation* = gold-current facts with an aligned
   memory but no current aligned memory. A missing audit sheet makes `alignment_audit_wrong` NaN, so the gate
   fails rather than skips.
10. **No real result is ever simulated.** Against the real server every P3 gate prints `UNAVAILABLE` until `link`
    works (`require_p3` probes it). The stand-in server exists only to prove each suite can pass and can fail
    (46 mutant tests). Gate rows beyond the spec's numbers are exact-case rows (`*_case_failures`, `violations`,
    `table_case_failures` = 0) and `run_complete` = 1; no threshold is invented or relaxed.
11. **`anthropic` is an optional extra** (`--extra llm`), imported lazily; the T0 suite never needs it.

**Questions for the maintainer (product side, not blocking the harness)**

- `mem calibrate` has no way to take SYN pools or `theta_conf` yet: the runbook merges record files
  (`bench merge-records`) and writes θ_conf with `bench gate-add-theta`. Production could absorb both.
- The P3 server must expose unfiltered conflict scores when `theta_conf` is unset (the θ fit needs them).
- Skill v1 must exist before the B-W5 run; the repo's `skills/mem/SKILL.md` still says `link` is unavailable.
- benchmarks-v2 §0.5 names `claude-haiku-5-5`; the API model table lists `claude-haiku-4-5`. Prices for both are in
  `bench/llm.py`.
