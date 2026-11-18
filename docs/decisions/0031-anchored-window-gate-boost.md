# 0031. Anchored-window boost at the gate: a soft date signal, never a filter (MEM2-266)

Status: **adopted (MEM2-274): `read.window_boost: {delta: 3.0, slack: 0.1}` is the default**, the host's tuned cell, as part of
the final config Round 10 `r10-all` (`fp_all` 3f8c9c11). Round 10 tune evidence (the host's runs, reported in MEM2-274 from the owner's MEM2-1 comment of 30 Oct): B-E1 tune **0.832** for `r10-all` against the Round 10 base arm 0.792 and Round 9's 9c 0.776. `window_boost: true` still means the dry run's
{delta: 1.0, slack: 0.1}. Was: accepted as a switch, default **off** until the Round 10 bars below hold on the host's
TUNE run (runbook `p3-gate.md` "Round 10 (combined)", step 3; numbered 0028 on the branch, renumbered at the merge, MEM2-270). Not adopted by this ADR. Option 2 of the temporal-memory research note (28 Oct, §C, there
"option 1"). Option A, a date-window *leg*, was dropped on 26 Oct (6/18): it changed the pool, but the
windowed losses it could reach were gate cuts whose gold was already in the pool.

## Context

On the 9c final config (F1 + F2+F3), 4 of the 11 tune temporal questions that delivered no gold are
**gate cuts**: the gold round is in the 30-pool but its reranker logit is under tau_keep (z ≈ -7.49):
71017277 "jewelry last Saturday" z -7.51, 0bc8ad93 "museum two months ago" -7.69, gpt4_1e4a8aec
"gardening two weeks ago" -8.45, gpt4_4929293b "relative's life event a week ago" -10.64. Each names a
relative date that the `mem.reldates` rules (ADR 0021) already resolve, and each gold session falls
inside the resolved window. Published systems that do well on temporal questions add time as a boost or
an ordering key at read, not as a hard filter: hard date filters have hurt temporal recall elsewhere
(the note's finding 6).

## Decision

`read.window_boost` (config.yaml, default absent = off). Values: `false`; `true` = the tuned
`{delta: 1.0, slack: 0.1}`; or an explicit `{delta: <logits, > 0>, slack: <per day, >= 0>}`. `true` is
written out as the mapping when loaded, so `fp_all` names the numbers.

When on, and the store has a reranker:

1. **Parse** (`mem.window.windows`): `mem.reldates.resolve(query, ref)`, `ref` = the as_of day in the
   store's zone (today when there is no as_of). No new parser and no model. Each resolved value (a day,
   an ISO-week range, a month, a year) is an anchor. Anchors that start on or after the question day
   ("today", "tomorrow", "next Friday") give no window: the question day is the edge of the store, and
   nothing in the future can be recalled.
2. **Window** = the anchor widened by `max(1 day, ceil(slack × distance))` on each side, `distance` =
   days from the anchor's midpoint to `ref`, the end clipped at `ref`. The slack grows with distance in
   the Weber style: people date distant events less precisely.
3. **Boost**: before the gate, `delta` is added to the raw reranker logit z of every pool member whose
   node's `valid_from` day falls in a window. p is computed from the boosted z, by the Platt logit or the
   read model. The gate (`select`: tau_keep, tau_high and gap) and list mode's tau_list then decide on
   that p as usual. Out-of-window members keep their score. **Nothing is filtered and nothing joins the
   pool.**
4. **Order**: the kept in-window rows are delivered first, each part in p order. Then the row limit and
   token budget (gated path) or the token cap (list mode) cut as before. A row shows the p the gate
   used, which is the boosted p. The events log keeps the **raw** z (`feat`), so records for a later fit
   stay unboosted, and adds `window: {phrases: [[phrase, start, end]], in_window: n}`. `timings["window"]`
   is the date lookup.

The split path (ADR 0026) is not boosted: it already orders by date. P1 stores (no reranker) are
unchanged.

**The gate invariant holds.** Every delivered row is in `gate.select(ps)` (or above tau_list) on the
p it shows. The boost only changes which p the gate sees, and the ordering only changes which kept rows
fall within the limit. T0: `tests/test_window_boost.py`.

## Switch off

The key is absent from the default `read` section, and `false` drops it (`config.OPTIONAL`). Off,
`recall` runs exactly today's code path: no parse, no lookup, no log key. A query that does not parse
with the switch on is also unchanged. Both are tested byte for byte (text, structured result and events
log).

## Fingerprint

`read` gains `window_boost: {delta, slack}` **only when on**, so the default `fp_all` (`4b73eb54…`) and
the all-off golden (`55f53d2e…`) are unchanged. All-off plus the tuned values gives `8563682d…`
(`tests/test_config.py`). A new fingerprint needs its own gate. The calibration records are raw z
(`calibrate.score_pool` is not boosted), so the refit on the host gives the same Platt fit and
tau_keep, under the new `fp_all`.

## The dry run (tune only, recorded pools, no models): `bench window-dryrun`

The 9c records (`tune_records.jsonl`, 115 answerable + 115 evidence-removed negatives), `delivered.jsonl`
and `gate-list.json`. Session dates and row text come from the pinned LongMemEval_S file, and the split is
re-derived so that only tune questions are read. The replay of the read path **without** the boost
reproduces the recorded delivery on **115/115** answerable questions. The parser fires on 10/115 tune
questions (8 temporal, 1 multi-session, 1 single-session-user), and on the same 10 negatives. Full
table: `deliverables/research/round10-window-dryrun.md`.

| slack 0.1 | Δ 0.5 | **Δ 1** | Δ 2 | Δ 3 |
|---|---|---|---|---|
| temporal gate cuts rescued (of 4) | 2 | **3** | 3 | 3 |
| gold pushed out of the 8 rows / 600 tokens | 0 | **0** | 0 | 0 |
| net rescues | 2 | **3** | 3 | 3 |
| non-gold rows admitted: questions / rows / est. tokens | 5 / 10 / 240 | **6 / 17 / 401** | 8 / 34 / 792 | 8 / 37 / 876 |
| negatives that newly deliver rows (rows) | 3 (7) | **4 (13)** | 5 (30) | 5 (40) |

Δ = 1 is the knee: it rescues 71017277, 0bc8ad93 and gpt4_1e4a8aec (all three answered wrong in the
9c e1 run). gpt4_4929293b needs Δ ≥ 3.15 (3 still misses it), and at Δ ≥ 2 the admitted noise doubles.
The slack values 0.1, 0.2 and 0.3 rescue the same 3. The selection rule, fixed before the table was
read, takes the most net rescues, then the fewest negatives newly answered, then the fewest admitted
rows, then the smallest Δ and slack. That picks slack 0.1, one row fewer than 0.2. 0.5 loses
gpt4_1e4a8aec, because more in-window rows crowd it past the 8-row limit. Every tune gold date is
within a day of its anchor, so tune cannot tell 0.1 from 0.3. A personal store with looser "two months
ago" wording may want the wider slack, and the config takes it.

**Risks the dry run shows.** (a) 4 more evidence-removed negatives deliver rows (71017277,
gpt4_b5700ca9, b46e15ee, eac54add). Every one of them is a temporal question whose window holds other
sessions, and that is the cost of a soft date signal. P2's FAR line (≤ 0.05) is the guard. (b) The margin
on gpt4_1e4a8aec is thin: -8.45 + 1 = -7.45 against -7.49. (c) On 2ebe6c92 and b46e15ee, which were
answered correctly, in-window rows displace other non-gold rows. Gold stays delivered.

## Bars for the host run (Round 10 (combined), step 3: the arm against the base arm)

The dry run is re-run on task 1's records first, because day-granular as_of changes the pools. The
same stop point applies (net < 2 drops the option). Then, with the switch on, against task 1's run:
**B-E1** VALID (pooled oracle ≥ 0.92, this run ≥ 0.90) and mem accuracy not below task 1's; **no
question type down by more than 1 question**; **P2 6/6**; **B-R6 0** violations; **latency** p95 up by
≤ 5 ms at L5k and L50k (median of three). The commands are in `deliverables/runbooks/p3-gate.md`
"Round 10 (combined)" (the base arm is task 1 + task 4 with every switch off; the Round 10 oracle pool). Only then does the default change, with its own gate.

## Consequences

- Latency: one regex pass over the query (about 9 µs) and, when it parses, one indexed lookup of
  the pool's node dates (30 rows). Nothing is added when nothing parses.
- A non-temporal question that mentions a date ("last month" in a multi-session question) is boosted
  too. On tune that is 2 of 115, and neither changed its delivery.
- `bench window-dryrun` replays the read path's selection on records with `mem.window`, so the dry run
  and the switch share one parser and one window rule.
