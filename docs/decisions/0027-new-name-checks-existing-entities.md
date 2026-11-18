# 0027. `new:<type>:<Name>` checks existing entities of any type first (MEM2-256)

Status: accepted. Spec: architecture-v2 §3.1 resolver step 3 and A8 (cross-kind → ask). A8 is unchanged.

## Context

B-W5 (host replay, skill v3 save rules, seed 1/M/tune, gate-serving.json): about_fill **0.940**
(17 misses) against the 0.98 threshold. The writer now tags places and projects, but the tags do not
land on the existing gold nodes: `about ["new:topic:Juniper"]` for the existing place Juniper,
`["Aurora", "new:thing:Fenove"]`, `["Falcon", "new:person:Carrow"]`, with `about_types` empty in
every miss.

Cause (557287a): `write._resolve` handled `new:<type>:<Name>` by looking only for an active,
unowned entity with the **same type** and name. It never consulted the aliases. So
`new:topic:Juniper` created a second node next to place Juniper. Bare names go through
`resolve.lookup` (aliases, typo rule) and were fine. Spec step 3 said "`new:` → create in phase B",
which assumed the writer only says `new:` for a name nobody holds.

## Decision

For `new:<type>:<Name>`, phase A first runs `resolve.exact` on the normalised name. That is an exact
alias match (case-insensitive, NFKC), any type, active entities, owned or not. Vault pages carry
their title as an alias.

| exact hits | result |
|---|---|
| exactly one of `<type>` (other types may also be present) | reuse it, as before. The subject line is `matched` with the stored name. |
| several of `<type>` | `needs_choice` with all hits |
| none of `<type>`, one or more of other types | `needs_choice` naming each one (`e<id> · <type> · <Name>`), nothing written. The writer resends with `#e<id>`. |
| none | create in phase B, as before |

Across types it **asks** and never auto-links, which is A8's cross-kind rule applied to `new:`
names as it already applies to bare names.

**Escape: `remember(distinct_new=false)`**, a new optional boolean and the only contract change.
Without it, a writer that really means a different thing ("Juniper" the band next to the place) had
no way out: the `needs_choice` reply's own advice, `new:<type>:<Name>`, would ask again forever.
With `distinct_new: true`, a `new:` name whose hits are all of other types is created. A
same-type hit is still reused, several same-type hits still ask, and bare names are untouched. The
`needs_choice` text says when to use it. The event log records it only when it is true.

One exact same-type hit is reused even when another type shares the name. The writer named the type,
so this is the old behaviour, and asking would only make it resend the same choice.

## Consequences

- `new:` reuse now also covers aliases ("new:place:Juni" → place Juniper, alias Juni) and owned
  (vault) entities. Before, it created a duplicate unowned node next to an imported page.
- SYN worlds hold project/org pairs that share one name. The replay oracle's `new:` for the second
  one now gets `needs_choice` without the gold entity on offer, and it answers `new:` with
  `distinct_new` (`bench.synth.replay`). The B-W1 T0 table's Atlas setup passes `distinct_new` for
  the org. The checks themselves are unchanged.
- `bench w5-run`'s writer loop returns every tool result, `needs_choice` included, to the writer
  model, which can resend. A scripted test proves the round trip on the real server.
- The skill (skills/mem/SKILL.md) does not mention `distinct_new` yet. The next Vault task covers
  the skill.
- The stand-in server `tests/bench_t0/fake_p3.py` keeps the old `new:` rule and ignores
  `distinct_new`. The B-W5 fake-client tests do not depend on the rule.

## Alternatives rejected

- **Auto-link across types** (`new:topic:Juniper` → place Juniper). It would fix the misses
  silently, but it changes A8. A cross-kind name is exactly the case A8 refuses to guess.
- **Fuzzy (`lookup`) for `new:`**. `new:` is the writer's explicit statement. A typo or
  token-subset neighbour would turn every explicit creation into a possible ask (for example
  "new:person:Ana Lopez" when Ana exists). Exact name/alias is the evidence the misses show.
- **A per-entry escape form** (for example `new!:type:Name`). It is a new `about` grammar where an
  optional field is enough. The call is resent after a choice anyway, by which time the other
  entries are `#e<id>`.

## Switch (MEM2-272, the owner 2026-10-29, "option 1")

Host B-W5 (v1 skill mem2-202 on 2d50067, gate-serving.json, two replays pooled) against tune-r6
without this ADR (one replay):

| Measure | With ADR 0027 (run 1 / run 2 / pooled) | Without (tune-r6, one replay) |
|---|---|---|
| conflict_action | 23/26 / 27/31 / 0.877 | 0.926 |
| about_fill | 0.968 / 0.965 / mean 0.966 | 0.947 |
| capture | 0.912 / 0.933 | 0.880 |
| noise | 0.134 / 0.111 | 0.153 |
| wrong_invalidation | 0.009 / 0.013 | n/a |

Skill v4's pooled conflict_action was also 0.881, so the drop follows this code, not the skill.
The owner keeps the rule **on** for the capture, noise and about_fill gains and wants a way to turn it
off if daily use shows missed contradictions.

**`ingest.new_checks_existing`** (bool, default `true`).

- `true` (or absent): this ADR. Phase A runs `resolve.exact` over any type, asks across types, and
  `distinct_new` is the escape.
- `false`: the rule before this ADR (557287a). `new:<type>:<Name>` looks only for an active, unowned
  entity of the same type and name (case-insensitive), else it creates one. Aliases are not used.
  `distinct_new` is accepted and ignored.

**Name and section.** `write._resolve` reads its settings from `Store.ingest_cfg`, the `ingest`
section, as `resolve_dates` does. That section holds the write-time switches. The name says what
`true` does: a `new:` name checks the existing entities first.

**Fingerprint.** `fp_all` must stay `4b73eb54…` for the default, and the stores and gates on it must
keep loading. `config.OPTIONAL` drops a switch that is written `false`, because those switches
default to off. This one defaults to on, so it is listed in `config.OPTIONAL_ON`, which drops the
key when it is written `true`. Absent and `true` therefore give the same section and the same
`fp_all`. Written `false`, the key stays in `ingest`. Since the switch changes which subjects get
written, it is part of `fp_all`. `Config.fp_all` includes the `ingest` section when
`resolve_dates` is on, as before, or when this switch is off, so turning it off with
`resolve_dates` off also gives a new fingerprint. Over the default, `false` gives `dac44d3c…`
(`tests/test_config.py`, `GOLDEN_FP_ALL_NEW_CHECKS_OFF`). A gate fitted on the default is refused
with the switch off and must be re-fitted, or a gate fitted with it off must be installed.
