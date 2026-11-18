# 0023: List-everything-then-count mode: its own cut `tau_list` and its own token cap

- **Status:** Accepted, **adopted in Round 9 (the owner, 2026-10-28): `read.list_mode` default `true` since MEM2-261** (final config F1 + F2+F3; tau_list from the gate's `list_model`, the host's 9c gate). Was: proposed, OFF by default.
- **Date:** 2026-10-26 · **Task:** MEM2-193 (F2 + F3) · **By:** Core (Harness), from the owner's MEM2-1 comment of 26 Oct 02:17
- **See also:** ADR 0030 (MEM2-268): `read.list_fill`, the round-level fill after the `tau_list` test (the fit's unit is the session; the reader needs the rounds).

## Context
B-E1 (ADR 0020) puts multi-session at about 72% under the P2 gate. Multi-session questions are mostly
aggregations ("how many…", "which…", "in total…"): the answer needs *every* matching session, and the normal recall
returns the top 8 above `tau_keep` inside a 600-token budget, so the reader sees some of the matches and counts those.

## Decision
1. **A generic aggregation-cue detector** (`mem.aggcue`): word-boundary English function words only: *how many, how
   often, in total, total(s), altogether, combined, count(s/ed/ing), number of, list(s/ed/ing), all, which of /
   which + a plural noun*. Date arithmetic (*how many days ago/between/since…*) and idioms (*at all, after all, all
   right*) are not cues; *account, listen, totally* are not matched. No dataset cue list, question id or template.
2. **List mode** (`read.list_mode`, default **false**, accepted in `config.yaml` but absent from the default `read`
   section, so with it off `fp_all` and every output are exactly what they were). With it on, a query the detector
   flags is answered by `_recall_list` instead of the gated top-k:
   - every pool member with `p >= tau_list` (p as in the gated path, provenance weighting included); no `limit`, no
     `tau_high` / gap abstention; none above the cut: the usual `no relevant memories` line;
   - **grouped by session** (the node, e.g. a chat session), groups in date order, each opened by `<date> · <name>`;
     rows under it as today (`id · type · date · title · p`);
   - one line under the header: *List every match below before you count; one line per match.*
   - the mode's own **token cap** (rule 3). Over it, the lowest-p rows are dropped and one line says how many
     (*+N more matches above tau_list not shown*); an explicit `budget` argument still wins.
   - non-aggregation queries take the normal path, byte for byte. The pool is the normal pool (`retrieval.pool`):
     "every row" means every reranked candidate above the cut.
3. **Token cap: 1,000 tokens** (server estimate, ceil(bytes/3)) per list response, so **p95 <= 1,000** by
   construction in the server, and the B-C2 report checks it on the real run. Normal-mode B-C2 (mean <= 250, p95 <=
   600, empty <= 30) is unchanged and the gate rows are not touched. The B-C2 record gains `list_tokens_n / mean /
   p95 / max / cap` over every list-mode response of the run (all zero with the mode off); they are report rows, not
   a `gates.yaml` row, because the gates file is frozen.
4. **`tau_list` is calibrated, not guessed** (`mem.listfit`, `bench fit-tau-list`, `bench gate-add-tau-list`):
   LME **tune** only (the command refuses a pool whose question is not tune in the manifest), the answerable pools
   whose question the detector flags and that hold a gold row. Unit: the session. Rule (floors style): the strictest
   cut whose training mean coverage of gold sessions, **after the cap has dropped rows**, is >= the floor; the floor
   comes from {0.80, 0.85, 0.90, 0.95} by an inner CV on the training folds only (highest floor whose held-out
   coverage stays within 0.03); no cut reaching it: the best-coverage cut, flagged `floor_met: false`, **infeasible,
   and `gate-add-tau-list` refuses it**. **Nested CV:** 5 outer folds assigned by a hash of the question id; each
   refits floor and cut on the other four and is scored once on its own. The outer folds, the final cut (flat CV,
   labelled optimistic), the cap and the Platt it is a cut of go in `meta.gate.list_model` under **one fingerprint**.
5. **Gate file rule:** a gate without `list_model` is accepted when `read.list_mode` is off and **refused when it is
   on**: `recall` (every query), `mem serve` preflight. A `list_model` whose fingerprint does not match its fields,
   or whose Platt (a, b) differs from the gate's, is refused at load. `read.list_mode` joins `fp_all` only when set,
   so turning it on voids older gates (re-gate: fit, then `gate-add-tau-list`).

## Response grammar (pinned MEM2-259, as the server emits it since MEM2-193; B-R6 checks it)
A list-mode response is, line by line (`·` is U+00B7 with one space on each side; `listmode.layout`, `read._recall_list`):
1. **Header:** `<n> results (of <c> checked) · <scope> · list mode · <g> session[s]`, then, with paging on and more
   than one page, the ADR 0024 tail. `n` = the rows listed, `g` = the group lines (always `results`, `1 session`).
2. **Instruction line:** exactly `List every match below before you count; one line per match.` (`listmode.INSTRUCTION`).
3. **Groups**, one per session (node), in date order (undated groups, `-`, first; ties by node id): a group line
   `<YYYY-MM-DD|-> · <session name, first 12 words>`, unindented, followed by at least one row.
4. **Rows**, each indented by exactly two spaces under its group, in the normal row grammar
   (`id · type · date · title · [about: …] · [dates: …] · p`, title <= 12 words, p in [0, 1]; `dates` per ADR 0021).
5. **Over-cap line** (optional, last): `+<N> more matches above tau_list not shown (token cap <cap>); narrow the query`.
   Never together with the paging tail (ADR 0024 replaces it).
6. The whole response <= the mode's cap (1,000 est. tokens). An answer with nothing above `tau_list` is the normal
   abstention line, not a list response. Not in the B-R6 grammar (never produced on the bench, which passes no
   filters and runs under the default cap): the `about '…': …, not applied` filter notes, and the one-row-over-the-cap
   trailer (`<id> is over the budget …`).
B-R6 (`bench/suites/b_p2_cost.py`, P2 form) takes a response as list mode when its second line is the instruction line
or its header carries `· list mode ·`, checks it against this grammar, and reports `violations_list` / `violations_normal`
(and `responses_list` / `responses_normal`) next to the total.

## Adoption rule (written before the run; TUNE only)
Turn `read.list_mode` on in the serving config only if, on the owner's host with the fitted gate, **all** hold:
1. B-E1 overall >= the Round 8 baseline (`mem2-187`, ecfd84c) and **no question type down by more than 5
   points**; **target: multi-session** (baseline about 72%) goes up;
2. P2 hard rows hold; B-C1 L5k <= 1,500 ms and L50k <= 1,200 ms;
3. B-C2 normal mode unchanged; P3 rows unchanged;
4. **list-mode tokens p95 <= 1,000** (the cap above, from the B-C2 `list_tokens_p95` row), with `list_tokens_n` > 0;
5. the `fit-tau-list` fit is feasible (every outer fold met its floor).
Otherwise it stays off, as with MEM2-171 (ADR 0020).

## Not done here / honest limits
- The numbers (the fitted `tau_list`, outer-fold coverage, B-E1, B-C1) need the real reranker and Sonnet on the
  host; the code here is tested on the hash models only (T0). No result is claimed.
- Coverage is of the gold sessions *inside the pool*; the pool ceiling is B-R1's.
- Pools carry no row text, so the fit prices a response with a cost model (`listmode.tokens_for`, 40 + 16/session +
  30/row); a test checks it never undercounts a rendered response.
