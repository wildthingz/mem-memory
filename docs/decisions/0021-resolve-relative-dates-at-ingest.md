# 0021. Relative dates resolved at ingest as a derived annotation (MEM2-192)

Status: accepted, switch default **off** until the adoption rule below is met on TUNE.
**Adopted (Round 9, the owner 2026-10-28): default `true` since MEM2-261** (final config F1 + F2+F3; new default `fp_all`, gate fitted on it: the host's 9c gate).

## Context

Notes and chat sessions say "yesterday", "last Tuesday", "two weeks ago". Read months later the
words point nowhere, and a question asking "what did I do on March 9?" shares no token with a chunk
that says "yesterday". B-E1 temporal sits at about 53%. The fix has to be local and deterministic
(no LLM, no network) and must not touch the stored text (a note is evidence, ADR 0010).

## Decision

`ingest.resolve_dates` (config.yaml, default `false`). When on, at `mem import`, the LME/bench
`ingest_session` path and `remember` *with* `as_of`, each chunk's text is scanned by
`mem.reldates` against the note or session date, read as a calendar date **in the store's zone**
(`tz`; a UTC timestamp late in the evening in New York is still that evening's date). The result is
stored in a new column `chunks.dates` (migration 0003), e.g.
`yesterday [= 2023-03-09]; last month [= 2023-02]`:

- `chunks.text` stays verbatim; embeddings and the reranker input are unchanged (ctx + text).
- `dates` is its own `chunks_fts` column (default bm25 weight 1.0), so the ISO date is findable.
- `recall` shows it on the row and in the structured row as `dates`, rendered short with the
  absolute date first (`... · dates: 2023-03-09 (yesterday) · p`; amended by MEM2-254, below). With
  the switch off, or no expression resolved, the key and the segment are absent and recall output is
  byte for byte what it was. `get` carries it too (below).
- `remember` without `as_of` has no known reference date (the clock is not the note's date): no
  annotation.
- At import the note's date is its stated date: `created`, else the date in the filename. A note
  that states neither gets no annotation. The file's mtime (the node's date fallback, ADR 0010) is
  never the anchor: it is the clock of whoever last copied the vault and changes on clone or sync.
- Re-import is idempotent. The importer treats a file as unchanged only if its stored annotations
  equal what the switch calls for, so switching off and re-importing drops them (and on restores
  them) even though the file did not change.

Fingerprint: `fp_all` includes the `ingest` section **only when `resolve_dates` is on**. Off, the
fingerprint is the existing golden (`55f53d2e…`), so gates fitted before this change stay valid;
on, it is `1001d404…` and a gate has to be fitted for it. `fp_embed` is unchanged.

## Amendment (MEM2-254): the row field is part of the contract, short, and in `get`

Round 9 (P2 hard row B-R6 with the switch on: `violations=148`, baseline 0) showed the `dates:`
segment was outside the pointer-row grammar the B-R6 checker enforces, and the structured `dates`
key outside the frozen recall row schema (`additionalProperties: false`). The cut-off fields in the
finding's examples (`dates: last week [= `) were the checker's own 80-character excerpt of each bad
row, not a cut in recall (recall has no per-row length cap); the rows were whole but illegal.

**Decision: amend the contract** to allow one optional `dates:` segment, rather than put the dates
where it already allows them. The only free-text places are the title and `about`. The title is the
node's name, ≤ 12 words, and B-R6 counts its words: dates there would eat the title's words and blur
"what is this" with "when". `about` holds subject names, and a client resolving them (`#id`,
`subjects`) would trip over dates. A separate, strictly typed segment keeps both meanings clean, and
with the switch off nothing changes.

- **Row grammar**: `id · type · date · title[ · about: …][ · dates: F][ · flags…] · p`, where
  `F = E(, E)*`, `E = V( \(words\))?`, `V` = `YYYY`, `YYYY-MM`, `YYYY-MM-DD` or a week
  `YYYY-MM-DD..YYYY-MM-DD`, `words` = letters, digits and spaces (`mem.reldates.ROW_FIELD`; the
  B-R6 checkers in `bench/suites/b_p2_cost.py` and `bench/suites/helpers.py` use that regex). The
  segment now comes before the flags (it came after them); only switch-on output moves.
- **Short, never cut inside a date**: the full form is `2023-05-22..2023-05-28 (last week),
  2023-05-27 (last Saturday)`. Over `ROW_CHARS` = 40 characters the words go first (the dates
  alone, distinct), then whole trailing dates; the first date is always shown whole. B-C2 had
  measured ~13 tokens/call for the old form (mean 159 vs 146); 40 characters is ≤ 14 est. tokens
  per row, and the usual one- or two-date field is far less.
- **Only resolved entries** are shown: an entry without a date (`last Saturday` with no `[= …]`,
  which the importer never stores but an old or hand-edited row could hold) is dropped, never shown
  as if it had one.
- **Structured** `rows[].dates` (string, optional) is in the frozen recall schema
  (`src/mem/contracts/mcp_tools.json`) and equals the row's segment.
- **`get`** carries the dates in full (no cap): a chunk's header ends `· dates: F`, a memory's
  header the dates of its chunks (`remember` with `as_of`). The text stays verbatim, so B-R6's
  byte-identical continuation check is untouched; the header's tokens count in `max_tokens` as
  before. A note's or entity's section list does not repeat them (get the chunk). The B-E1
  answerer reads `get`, so the absolute dates now reach it.
- **Stored and indexed form unchanged** (`chunks.dates` = `yesterday [= 2023-03-09]`, FTS as
  before): no re-import, and the fingerprints (`fp_all` golden off `55f53d2e…`, on `1001d404…`)
  are unchanged. The rendered form is read-side only.

## Rules (`mem.reldates`)

A rule change needs no version stamp: the importer compares the stored annotations, values
included, with what the rules give now, so re-importing rewrites any that changed.

Reference date D. Case-insensitive, whole words.

| Expression | Value |
|---|---|
| today, tonight, this morning/afternoon/evening/night | D |
| yesterday, last night, last evening | D−1 |
| tomorrow | D+1 |
| (the) day before yesterday / day after tomorrow | D−2 / D+2 |
| N days / weeks ago (N digits, one…twelve, a, an) | D−N / D−7N |
| N months / years ago | same day-of-month, clamped to the month's last day (31 Mar − 1 month = 28 Feb) |
| last / this / next week | Monday..Sunday range of that ISO week, `a..b` |
| last / this / next month | `YYYY-MM` |
| last / this / next year | `YYYY` |
| last *Weekday* | the nearest such day strictly before D (a Sunday's "last Sunday" is D−7) |
| next *Weekday* | the nearest such day strictly after D |
| bare *Weekday* ("on Tuesday") | only if the sentence's tense is plain: a past cue (was, were, had, did, went, met …) and no future cue gives the nearest earlier such day; a future cue (will, 'll, going to, scheduled, tomorrow …; not "should", which is as often counterfactual, nor "next", which belongs to its own phrase such as "next week") and no past cue gives the nearest later one |

A phrase repeated with the same value is annotated once; at most 16 annotations per chunk.

## Never resolved (no annotation, never a guess)

Vague: recently, a while ago, the other day, soon, a few / a couple of / many days ago, any unit
not listed (fortnight, decade). Ambiguous: "this Friday", "every/each Tuesday", "Tuesdays", a bare
weekday with no tense cue or with both; "in two weeks" (a duration as often as a future date);
"last weekend"; absolute but partial dates ("March 9", "the 5th", "in March"). Matching is on whole
words, so "yesterdays" and "someday" are untouched.

## Adoption rule (the owner's; for the host pass)

Default stays off. On TUNE only: B-E1 tune overall ≥ the Round 8 baseline (`mem2-187`)
with no question type down more than 5 points; P2 hard rows hold; B-C1 L5k warm p95 ≤ 1,500 ms and
L50k ≤ 1,200 ms; B-C2 holds for normal recall; P3 rows unchanged (accepted misses only). Target:
B-E1 temporal (baseline about 53%). Only then does the switch default change, with its own
fingerprint and gate.

## Consequences

- Schema 3: `chunks.dates`, and `chunks_fts` rebuilt with a third column (one `rebuild` on first
  open of an existing store; text/ctx ranking unchanged because the new column is empty).
- A wrong rule here would put a wrong date next to true text; hence the narrow rules above and the
  tense gate on bare weekdays. A month-name form ("March 9") is not indexed; the ISO form is. A
  follow-up can add it if TUNE shows questions phrased that way miss.
