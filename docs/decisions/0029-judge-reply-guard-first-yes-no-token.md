# 0029. The B-E1 judge: 64 tokens, the first yes/no token, unparsed replies flagged (MEM2-269)

Status: accepted. Spec: benchmarks-v2 B-E1 (the judge and run validity). Evidence: the owner's
Round 10 research note `data/temporal-r10/multi-session-research.md`, row 6c49646a (git-ignored).
Requested by the owner (MEM2-1, 28 Oct 18:06, Round 10 task 4). Harness only.

## Context

B-E1 judges every answer with upstream LongMemEval's `evaluate_qa.py` prompts, and until now with
upstream's reply handling too: `n_tokens=10` and "correct iff `yes` occurs anywhere in the reply"
(MEM2-188 F4 adopted both to match upstream). Upstream's judge is GPT-4o, which answers the prompt's
"Answer yes or no only" with a bare word. Ours is Sonnet 5.5, which sometimes explains first: on the
tune runs 13 of 375 judge replies were verbose. Item 6c49646a (gold "3,000 miles") was answered
correctly, the judge began "The response's final total is 3,000", the 10-token cap cut it there, no
`yes` was present, and the item scored wrong. Nothing in the record showed that the verdict was a
truncation rather than a judgement.

Two things were wrong: the cap was sized for another model's habits, and a reply without any verdict
was scored (as `no`) in silence.

## Decision

1. **Cap.** `JUDGE_MAX_TOKENS` is 64 (was 10). The budget guard counts the new worst case.
2. **Parse.** The verdict is the **first** word-bounded `yes` or `no` token of the reply,
   case-insensitive (`e1.judge_verdict`; `\b(yes|no)\b`). "Yes, the answer ..." is yes; "No." is no;
   "The response's final total is 3,000. Yes." is yes; "Nothing matches", "not correct" and
   "Yesterday" contain no token. Upstream's "yes anywhere" is gone: "No, it is not yes" is now no.
3. **Flag, don't guess.** A reply with neither token is recorded per item (`judge_verdict: null`,
   `judge_flagged: true`), counted in the run record (`judge_flagged`), printed by `e1-run`,
   `e1-run --lme-size M` and `locomo-run`, carried as the gate metric `judge_flagged` with a note on
   the B-E1 gate row, and **not credited** (`correct: false`). The count is re-derived from the raw
   replies at scoring time, so an old run file re-scored today shows its cut replies as flagged
   instead of as quiet noes.
4. **The prompt text is unchanged**, so `prompts_sha256` and `prompt_version` (2) do not move.
5. **`judge_version`** (`e1.JUDGE_VERSION`, now 2) is a new run-record field and a member of
   `HARNESS_FIELDS`. `--oracle-pool` therefore refuses a file of the other judge version naming the
   field, and a run file from before the field existed reads as `None` and is refused the same way.
   The pooled oracle rule (pooled >= 0.92, every run >= 0.90, a finals run pools with `tune` runs of
   its harness) is unchanged among runs of one judge version. `e1-run --lme-size M` adds
   `judge_version` to the fields a resume must match (`e1_m.CHECKED`).
6. **Every arm, every runner.** `run_e1` writes the verdict fields through one function
   (`e1.verdict_fields`) for mem, none and oracle; `e1-agentic`, `--full` and LME_M use the same
   function and the same cap (`e1_m` runs through `run_e1`; `e1_agentic` imports both).
7. **LoCoMo is not the same judge.** `bench/locomo.py` uses the Mem0 CORRECT/WRONG prompt, which
   asks for a reasoning sentence and then a JSON label, under a 200-token cap; its parser already
   credits only an unambiguous CORRECT. Its cap and parse stay. It gains the same disclosure: a reply
   with neither label, or both, is counted as `judge_flagged` in its metrics and printed.

## Consequences

- **The finals disclose it:** "judge reply parsing differs from upstream LongMemEval: 64 output
  tokens and the first yes/no token, unparsed replies flagged and not credited (ADR 0029)". Numbers
  judged under version 1 and version 2 are not the same measurement; a verbose-but-correct reply
  that version 1 scored wrong can score right under version 2.
- **The oracle pool restarts.** `data/e1/tune-r8c.json` and the Round 9 `e1-tune.json` files are
  judge version 1 and are refused in a version 2 pool. The finals (`--full`) pool with the Round 10
  tune runs made after this change (the re-gate of Round 10 task 1 onward); the runbook says which
  files. Until two or more such tune runs exist, a finals run's pool is smaller than before, and the
  per-run floor of 0.90 still applies to it.
- **A non-zero `judge_flagged` is a signal, not a failure.** It does not void a run. The gate row
  prints the count so a reviewer can read the flagged items (`judge_raw` in the run file) before
  quoting a number. If flagged replies recur at 64 tokens, raise the cap again rather than parse
  harder.
- **Known limit of "first token".** A judge that quotes the model's own "yes" before its verdict
  ("The response says no, which is wrong. Yes.") is read as no. The owner asked for the first
  token; it is the simplest rule that survives a cut-off tail, and the 64-token cap makes a
  reasoning preamble short. The alternative, the last token, is right for that case but wrong when
  the cap cuts a reply after a quoted token. Either way the prompt says "yes or no only", so a
  preamble is the exception.

## Alternatives rejected

- **Keep 10 tokens, score "no yes" as wrong (upstream).** Right for GPT-4o, wrong for Sonnet: it
  turns verbosity into misses, invisibly.
- **Change the prompt to force a bare verdict.** It would change `prompts_sha256` and the
  comparability of the judge with upstream's; the owner asked for the prompt text to stay.
- **Count flagged items out of the denominator.** Hides the judge's failure rate inside the
  accuracy and makes n vary between runs; flagged items stay in, as not credited, and are counted.
- **Fold the judge version into `prompts_sha256`.** The sha names the prompt text and nothing
  else; a separate field says what changed and keeps old files readable.
