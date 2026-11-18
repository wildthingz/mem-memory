# 0024: Coverage mode: list mode paged behind a continuation token

- **Status:** Proposed (switch `read.list_paging`, default off; adoption rule below is for the owner to accept)
- **Date:** 2026-10-26 · **Task:** MEM2-212 (option B) · builds on ADR 0023

## Decision
1. `read.list_paging` (default false, absent from the default `read` section so `fp_all` is unchanged; on, it
   joins `fp_all`). It needs `read.list_mode`: the config loader refuses it alone, and the store ignores it
   without list mode. Turning it on changes the fingerprint (with list mode on, `fp_all` is `8d21307d…`, golden
   test in `tests/test_config.py`), so it voids the fitted gate: a gate has to be fitted for the paging
   fingerprint before the adoption run below.
2. With it on, an aggregation query's list-mode answer is split into **pages**: the gated rows (`p >= tau_list`,
   same set and same order as ADR 0023, lowest p last) are packed greedily, each page under the mode's cap
   (1,000 tokens, or an explicit `budget`). Each page is grouped by session with the session date, as in 0023.
   The header prices the longest paging text, so no page exceeds the cap. Nothing is added after the gate.
3. Page 1 header: `8 of 14 shown · more: recall(…, page=2-<tag>)`. Middle pages the same with the running count;
   the last page: `14 of 14 shown · page 3 of 3 · last page`. If everything fits on one page the answer is
   plain list mode (no paging text). The "+N more matches not shown" line is not used when paging is on.
   **Grammar (pinned MEM2-259):** the tail is appended to the ADR 0023 list header, on the same line, never a line
   of its own: ` · <a> of <t> shown · more: recall(…, page=<k>-<10 lowercase hex>)` on page 1 and middle pages,
   ` · <t> of <t> shown · page <n> of <n> · last page` on the last; `a` = rows on this and earlier pages (<= `t`),
   `…` is U+2026. B-R6 checks it with the rest of the list grammar (ADR 0023 "Response grammar").
4. **Token** = `<page>-<tag>` (a bare `page=2` could not be bound). `tag` = first 10 hex of sha256 over the query
   and filters, `retrieval` + `read` config, the list model fingerprint, and the ordered gated row ids. A changed
   query, filter, config, or store content (rows would repeat or vanish) is refused as stale; malformed tokens and
   pages past the end are refused. `page` without paging or on a non-aggregation query is refused.
5. MCP contract: one new optional `page` (string) on `recall`; snapshot updated. Structured result gains `page`,
   `pages`, `next_page` on multi-page answers.
6. Switch off: output byte for byte today's (tests: `tests/test_list_paging.py`).

## Adoption rule (as ADR 0023, TUNE only)
On the owner's host with the fitted gate: B-E1 overall >= baseline and no type down > 5 points; **main proxy:
recall-all@8 on the multi-session tune questions with paged rows counted** goes up; list-mode tokens p95 <= the cap;
B-C2 normal mode and P3 unchanged. Otherwise it stays off.

## Not measured here
This host has no LongMemEval data, so the number of tune questions that trigger the mode and recall-all@8 were not
run. The bench side now counts paged rows (MEM2-259): `bench delivered-dump` follows every `next_page` token
and `bench delivered-pair` prints a **whole delivery** column (recall-any, recall-all, gold-round fraction over every
delivered row of every page, paired CI) next to @5/@8/@10; the proxy above reads that column. T0 only.
