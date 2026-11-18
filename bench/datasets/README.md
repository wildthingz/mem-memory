# LongMemEval datasets, manifest and negatives (benchmarks-v2 §1, B-R1, B-R2 (i), B-C)

Everything here is reproducible from the official files with `uv run bench …`; the checked-in
files are `lme_split.tsv` (frozen manifest), `lme_negatives.tsv`, `lme_negatives_audit.tsv`,
`checksums.sha256` (written by the first `bench fetch`). Tests use the tiny slices in
`bench/fixtures/lme/` (regenerate with `uv run python -m bench.fixtures.lme.make_fixture`).

## Definitions chosen where benchmarks-v2 leaves room (decision record)

| Choice | Rule | Why / effect |
|---|---|---|
| Tune size | `round(0.3 n)` half-up in integers, `(30 n + 50) // 100`, per (type, `_abs`) stratum | Python's `round` is banker's rounding; no official stratum has an exact .5, so counts are identical (151/349) either way. |
| `_abs` | `question_id` ends in `_abs` | Official convention. |
| B-R1 set | not `_abs`, type ∈ {knowledge-update, temporal-reasoning, multi-session, single-session-user} | Gives 115 / 269. SSA and SSP are in the manifest but not in B-R1/B-R2 (spec). |
| Round | starts at each `user` turn and takes the assistant turn(s) after it; assistant turns before the first user turn are a round of their own; a round is relevant if **either** turn has `has_answer` | A6 mapping. |
| Chunk ref | `<session_id>#<round_index>` | Adapter maps product chunk ids to it through `source_ref` + `chunk_key`. |
| Ranking metrics | **session level** (best chunk per session, sessions deduplicated in rank order); gold = `answer_session_ids` | B-R1 step 3. Recall_any@5, Recall_all@10, nDCG@10 (std), ndcg_any@10 (upstream), MRR. |
| Delivered metrics | **chunk level** on S_q; gold = rounds with a `has_answer` turn; negatives (B-R2 (i)) are in the same pool with empty gold | §2.0 and the A16 note ("LME relevance is round-level"). `coverage_all` = every gold round delivered (LME evidence is binary, so rel 2 = rel 1). |
| ABS | run on its own haystack, abstention recall only, descriptive | Spec: CI half-width ≈ ±0.16 at n = 20. Excluded from the delivered pool. |
| Haystack dedup | a repeated `session_id` inside one haystack keeps its first occurrence; the count is recorded (`Haystack.dropped_duplicates`) | Spec: "session dedup". |
| B-C2 "≥ 80% fewer than top-8 full chunks" | **aggregate**: `1 - Σ tokens(response) / Σ tokens(full text of R_q[:8])` ≥ 0.80 on the 300-response sample; the per-response share is reported too | Spec wording is ambiguous (aggregate vs per response). Both numbers are in the record; the gate row uses the aggregate. |
| Audit sample | the 30 B-R1 pairs with the smallest `sha1("audit7:<qid>")` | Deterministic ("seed 7"). `auto_hit_sessions` is a hint for the auditor (normalised answer string found in a remaining session; skipped for answers < 4 chars), never the verdict. |
| L5k / L50k | LME_M sessions deduplicated by id (a conflicting duplicate is an error), sorted by id, rounds taken in order until exactly N chunks; the session that crosses N is truncated to the rounds that fit | "First N chunks" made exact. `corpus_fingerprint` pins the plan. |
| ndcg_any (upstream) | `bench/metrics/upstream_lme.py` (vendored, commit `6a92d1a…`) and the dependency-free `ir.ndcg_any_upstream`; a test asserts they agree on 300 random rankings | Kept apart from nDCG@10 (std) (A6). SHA as reported by the GitHub API on 2026-10-23: re-check on `bench fetch`. |

## Files

| File | Made by | Content |
|---|---|---|
| `lme_split.tsv` | `bench build-lme` | question_id, question_type, is_abs, split, rank-in-stratum. Frozen; `bench build-lme` re-derives it and fails on any difference. |
| `lme_negatives.tsv` | `bench build-lme`, then `bench audit-apply` | one row per B-R1 question: removed answer sessions, remaining count, auto hint, `audit_dropped`. |
| `lme_negatives_audit.tsv` | `bench build-lme` | the 30-pair sheet; the auditor fills `still_contains_answer` (y/n) and `notes`. |
| `checksums.sha256` | `bench fetch` | sha256 of the official files; later fetches must match. |

## LoCoMo (MEM2-196)

`bench locomo-fetch` downloads `data/locomo10.json` from github.com/snap-research/locomo at commit
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` and checks it against the pinned sha256
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` and size 2,805,274 bytes
(`bench/datasets/locomo.py`). 10 conversations, 272 sessions, 1,986 questions (categories 1-5:
282 / 321 / 96 / 841 / 446). **Licence: CC BY-NC 4.0** (the repo's `LICENSE.txt`; GitHub shows "other"):
non-commercial use only, attribution to Maharana et al. (ACL 2024, arXiv 2402.17753). The file is
downloaded, never checked in; the checked-in fixture `bench/fixtures/locomo/locomo_mini.json` is
hand-written, LoCoMo-shaped, and contains no LoCoMo text. There is no `THIRD_PARTY.md` on this branch to
add it to. Round mapping, scoring and judge: `bench/locomo.py` docstring, runbook p3-gate.md Round 9.
