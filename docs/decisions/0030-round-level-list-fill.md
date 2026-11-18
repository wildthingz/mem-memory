# 0030: Round-level list fill in list mode (`read.list_fill`)

- **Status:** **Adopted (MEM2-274): `read.list_fill: 16` is the default**, as part of the host's final config Round 10 `r10-all`
  (`fp_all` 3f8c9c11). Round 10 tune evidence (the host's runs, reported in MEM2-274 from the owner's MEM2-1 comment of 30 Oct): B-E1 tune **0.832** for `r10-all` against the Round 10 base arm 0.792 and Round 9's 9c 0.776. Was: proposed, off by default.
- **Date:** 2026-10-28 · **Task:** MEM2-268 (Round 10, task 3) · **By:** Vault, from the owner's MEM2-1 comment of
  28 Oct 18:06 and `multi-session-research.md` (Part 0, option 1)

## Context
List mode (ADR 0023) delivers every pool row with `p >= tau_list`, and `tau_list` is fitted so that >= 0.95 of the
gold **sessions** have a row in the list (§4: "a list answer is complete when every gold session has a row in it").
The answerer, though, reads the text of the delivered **rounds**: a session "covered" by its small-talk round
delivers none of the evidence. On the Round 9c final config (tune, 36 answerable multi-session questions): gold
sessions covered 32/36, every gold round delivered 22/36. Of the 11 wrong multi-session tune answers, 6 have the
missing gold round in the 30-row pool, cut by the `tau_list` **score** (p 0.02289, z about -7.17), none by tokens
(those responses used 86 to 632 of the 1,000-token cap).

## Dry run (`bench list-fill-dryrun`, no models; recorded Round 9c tune pools and deliveries)
Inputs: `tune_records.jsonl`, `delivered.jsonl`, `gate-list.json` (Platt a 0.51256, b -0.08146; `tau_list`
0.022890), the LME question texts for the detector (`mem.aggcue`), tune only (the manifest check refuses other
pools). 115 answerable tune questions, 39 cued (list mode); multi-session 36, of which 22 cued. Tokens are the
list-mode cost model (`listmode.tokens_for`, 40 + 16/session + 30/row) over the non-empty deliveries.

**Reproduction of the research's `sim.py` first** (its rule: delivered ∪ top N, or ∪ p >= 0.005, on *every*
multi-session question, no cue test, no `tau_list` test, no cap): as delivered 22/36; top-12 25/36; top-16 29/36;
top-20 30/36; p >= 0.005 30/36; sessions 32 / 34 / 36 / 36 / 36; rows mean 8.4 / 12.5 / 16.1 / 20.0 / 21.9.
**Identical to `sim_out.txt`.** The research table's "top 16 → 28/36" is the switch's rule (below), not the sim's:
the one-question gap is `gpt4_5501fe77`, a multi-session question the detector does not flag (normal path), which
the sim fills and the switch never touches.

| arm (the switch's rule: cued, `tau_list` test, cap) | every gold round delivered | gold sessions | rows mean / p95 (MS) | tokens p50 / p95 / max (all 39 list-mode) | abstention changes (cued pools) | wrong qids whose gold is recovered |
|---|---|---|---|---|---|---|
| today (as delivered) | **22/36** | 32/36 | 8.4 / 16 | 312 / 632 / 768 | 0 | – |
| fill to 12 | 24/36 | 33/36 | 10.6 / 16 | 480 / 632 / 768 | 0 | 81507db6, a9f6b44c |
| **fill to 16** | **28/36** | 35/36 | 12.7 / 16 | 648 / **696** / 768 | **0** | 81507db6, bf659f65, 3a704032, a9f6b44c, d3ab962e, ba358f49 |
| fill to 20 | 29/36 | 35/36 | 15.1 / 20 | 784 / 832 / 864 | 0 | + 0a995998 |
| `tau_list` refit on gold rounds (0.011305, floor 0.95, feasible, held-out mean 0.951 worst 0.893) | 26/36 | 33/36 | 12.3 / 22 | 436 / 876 / 996 | **9** (negatives abstaining 12 → 3) | 81507db6, bf659f65, a9f6b44c, 0a995998 |

Other list-mode questions (cued, not multi-session; 17, all complete today): rows mean 7.5 → 16.0 at N=16 (20.0 at
N=20), tokens p95 570 → 696 (832); every gold round still delivered (17/17) under every arm. Report:
`data/temporal-r10/dryrun/list-fill.{json,txt}` (git-ignored, with the evidence).

## Decision (tune only; the choice was made on these counts before any model run)
1. **`read.list_fill: N`**, default off (0; the key is absent and `fp_all` is unchanged; set, it joins the `read`
   section and so `fp_all`). It needs `read.list_mode`. With it off, recall is byte for byte today's.
2. **Rule.** In list mode, for a cued query: the `tau_list` abstention test is unchanged (no row at or above
   `tau_list`: the usual `no relevant memories` line, byte-identical). Otherwise the rows above `tau_list` are
   followed by the next-ranked rows **by p** (the gated path's p: the Platt of the rerank logit with provenance
   weighting, the same order the matches already use; equal to the z order when the weights are equal) until
   **N rows in all** (`max(N, matches)`: never fewer than today's matches). Everything after is as today: grouped
   by session in date order, the instruction line, the 1,000-token cap cut from the bottom, paging (ADR 0024),
   the ADR 0023 grammar (B-R6 passes; the cap is on the whole response). The over-cap line counts only the
   **matches** (rows above `tau_list`) the cap dropped, never fill rows, so its wording stays true; when the cap
   drops fill rows only, there is no over-cap line. `structured.matches` stays the rows above `tau_list`;
   `structured.filled` (only with the switch on) is the number of fill rows shown.
3. **N = 16** on tune: +6 questions with every gold round (22 → 28 of 36), gold recovered in 6 of the 11 wrong
   answers, tokens p95 696 of the 1,000 cap, abstentions unchanged by construction. N = 20 adds one question
   (0a995998, gold at rank 19) whose answer the research classes as a reader/dedupe loss (its cut gold was in the
   answer anyway through sibling rows) for 136 more tokens at p95 and four more non-gold rows on every list
   answer; N = 12 recovers two. The research's recommendation is 16 as well.
4. **The round-unit refit is not a mode of the switch.** Fitted with the gold round as the unit (`listfit`
   `unit="round"`, kept for the dry run) the cut lands at p 0.0113 (z about -8.5): it recovers fewer questions
   than the fill at 16 (26 vs 28), costs more tokens (p95 876, max 996), and, being the abstention test as well,
   **stops abstaining on 9 of the 12 cued tune negatives** that abstain today: ADR 0023's "abstention unchanged"
   property and the 10/10 `_abs` score are the first things the fill was asked to keep. Nothing else was
   implemented for it, deliberately (no second mode to calibrate and gate).
5. Non-list queries, the normal path, the gate file and the gate's `list_model` are untouched; no LLM, no dataset
   cue list, nothing keyed on the question-type label (the dry run reads the LME type only to report by it).

## Host pass lines (runbook `p3-gate.md` "Round 10 (combined)", step 3; this ADR was numbered 0028 on the branch and renumbered at the merge, MEM2-270)
`read.list_fill: 16` on the serving config only if, on the owner's host with the 9c gate, all hold: B-E1 not below
the task-1 base arm; no category down by more than 1 question; P2 6/6 and B-R6 0 violations; list-mode tokens
p95 <= 1,000 (`list_tokens_p95`); latency +<= 5 ms p95. Otherwise it stays off.

## Not done here / honest limits
- The counts are retrieval counts on recorded pools: whether Sonnet answers the 6 recovered questions is the
  B-E1 run's to show (the research expects +3 to 4 of the 11: a9f6b44c's oracle is wrong too, ba358f49 needs a
  two-hop sum, bf659f65 depends on counting a signed vinyl). The 25 questions already right get up to 8 more
  non-gold rows each (about 2.3k characters of round text per row once fetched).
- Tokens are the cost model, not the rendered responses (pools carry no text); the fit's test shows the model
  never undercounts a rendered response.
- The dry run is arithmetic over the recorded pools (no model loaded, in the sandbox or on the host); the switch's
  tests are T0 on the hash models. The host run that decides adoption is the Round 10 B-E1 mem arm.
