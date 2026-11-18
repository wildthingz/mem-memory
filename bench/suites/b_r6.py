"""B-R6, P1 form (benchmarks-v2 B-R6): the response contract on fixtures.

P1 has no reranker, so the `p` column carries `rrf` and the header carries
`uncalibrated (P1 build)`; the empty case is a query over an empty eligible set that returns one
`no results` line of <= 30 tokens. The `p` column and the abstention line are checked from P2.
Token counts are the server estimate (ceil(bytes/3), "est."); `count_tokens` needs the network.
"""

from __future__ import annotations

import re

from bench.contract_check import structured_violations
from bench.fixtures import streams as fx
from bench.metrics.tokens import est_tokens
from bench.suites.base import Client, Context, SuiteResult
from bench.suites.helpers import ROW, node_num, seed_corpus, sql, sql_write

P1_HEADER = "uncalibrated (P1 build)"
BUDGET = 600
P1_TOP = 5


def check_response(res: SuiteResult, tag: str, query: str, r) -> None:
    lines = r.text.splitlines()
    bad = structured_violations("recall", r.structured)
    res.check(f"{tag}_structured_contract", not bad, "; ".join(bad))
    res.check(f"{tag}_header", bool(lines) and P1_HEADER in lines[0], lines[0] if lines else "")
    body = lines[1:]
    parsed = [ROW.match(ln) for ln in body]
    res.check(f"{tag}_row_format", all(parsed),
              "; ".join(ln for ln, m in zip(body, parsed, strict=True) if not m))  # fmt: skip
    res.check(f"{tag}_titles_le_12_words",
              all(len(m["title"].split()) <= 12 for m in parsed if m), query)  # fmt: skip
    res.check(f"{tag}_rows_le_top5", len(body) <= P1_TOP, f"{len(body)} rows")
    text_ids = [m["id"] for m in parsed if m]
    res.check(f"{tag}_text_matches_structured", text_ids == [x["id"] for x in r.structured["rows"]],
              f"{text_ids} vs {[x['id'] for x in r.structured['rows']]}")  # fmt: skip
    res.check(f"{tag}_tokens_le_budget", est_tokens(r.text) <= BUDGET, f"{est_tokens(r.text)} est.")


def _check_empty(res: SuiteResult, ctx: Context) -> None:
    client = ctx.factory(ctx.store("b_r6_empty.db"), scope="personal")
    try:
        r = client.call("recall", {"query": "anything at all"}, now="2026-10-22T09:00:00Z")
        res.check(
            "empty_is_one_no_results_line",
            len(r.text.splitlines()) == 1 and r.text.startswith("no results"),
            repr(r.text),
        )
        res.check("empty_le_30_tokens", est_tokens(r.text) <= 30, f"{est_tokens(r.text)} est.")
        res.check("empty_structured", r.structured.get("rows") == []
                  and not structured_violations("recall", r.structured), str(r.structured))  # fmt: skip
    finally:
        client.close()


def _get_pages(client: Client, first: str, max_tokens: int, now: str) -> tuple[list, list[str]]:
    """Follow continuation tokens from `first`; returns (responses, item texts)."""
    out, texts, token = [], [], first
    for _ in range(200):
        r = client.call("get", {"ids": [token], "max_tokens": max_tokens}, now=now)
        out.append(r)
        item = r.structured["items"][0]
        texts.append(item.get("text", ""))
        token = item.get("continuation")
        if not token:
            return out, texts
    raise AssertionError("continuation did not terminate in 200 pages")


def _insert_long_memory(client: Client, body: str, now: str) -> str:
    """`remember` caps text at 1,200 chars (A1), so a 9 kB item is written straight to the store,
    as a long imported note would be (P4); the path under test is `get`'s paging of it."""
    sql_write(client.db_path,
              "INSERT INTO nodes(kind, type, name, body, date, valid_from, recorded_at, origin,"
              " content_hash, session, host) VALUES ('memory', 'procedure', 'Long code block', ?,"
              " ?, ?, ?, 'agent', 'long-fixture', 's1', 'bench')",
              (body, now[:10], now, now))  # fmt: skip
    return f"m{sql(client.db_path, 'SELECT max(id) FROM nodes')[0][0]}"


def _check_get(res: SuiteResult, ctx: Context) -> None:
    now = "2026-10-22T09:00:00Z"
    client = ctx.factory(ctx.store("b_r6_get.db"), scope="personal")
    try:
        code = fx.code_block()
        res.check("fixture_is_about_3000_tokens", 2800 <= est_tokens(code) <= 3200,
                  f"{est_tokens(code)} est.")  # fmt: skip
        mid = _insert_long_memory(client, code, now)
        pages, texts = _get_pages(client, mid, 500, now)
        res.check("get_continuation_pages_gt_1", len(pages) > 1, f"{len(pages)} page(s)")
        res.check("get_continuation_byte_identical", "".join(texts) == code,
                  f"{len(''.join(texts).encode())} bytes vs {len(code.encode())}")  # fmt: skip
        res.check("get_max_tokens_respected_per_page",
                  all(est_tokens(p.text) <= 500 for p in pages),
                  str([est_tokens(p.text) for p in pages]))  # fmt: skip
        whole = client.call("get", {"ids": [mid], "max_tokens": 4000}, now=now)
        res.check("get_whole_text_when_it_fits", whole.structured["items"][0].get("text") == code
                  and not whole.structured["items"][0].get("continuation"), "")  # fmt: skip

        # max_tokens covers the WHOLE response: ten ids share it.
        ids = seed_corpus(client, fx.CORPUS, now)
        r = client.call("get", {"ids": ids[:9] + [mid], "max_tokens": 300}, now=now)
        res.check("get_total_max_tokens_respected", est_tokens(r.text) <= 300,
                  f"{est_tokens(r.text)} est. > 300")  # fmt: skip
        bad = structured_violations("get", r.structured)
        res.check("get_structured_contract", not bad, "; ".join(bad))

        # Stale chunk id: the chunk was replaced; never return a different section (A10, A12).
        chunk = sql(client.db_path, "SELECT id FROM chunks WHERE node_id = ?",
                    (node_num(ids[0]),))[0][0]  # fmt: skip
        sql_write(client.db_path, "DELETE FROM chunks WHERE id = ?", (chunk,))
        stale = client.call("get", {"ids": [f"c{chunk}"]}, now=now)
        item = stale.structured["items"][0]
        res.check("stale_chunk_no_longer_exists", "no longer exists" in stale.text
                  and fx.CORPUS[0][0] not in stale.text, stale.text)  # fmt: skip
        res.check("stale_chunk_not_found", item.get("found") is False, str(item))
    finally:
        client.close()


def _check_edges(res: SuiteResult, ctx: Context) -> None:
    now = "2026-10-22T09:00:00Z"
    client = ctx.factory(ctx.store("b_r6_edges.db"), scope="personal")
    try:
        mems = []
        for i in range(fx.EDGE_COUNT):
            about = [fx.EDGE_ENTITY] if i == 0 else ["EdgeHub"]
            r = client.call("remember", {"text": f"Edge fixture memory number {i} alpha{i}",
                            "type": "fact", "about": about}, now=now)  # fmt: skip
            mems.append(r.structured["id"])
        ent = sql(client.db_path, "SELECT id FROM nodes WHERE kind = 'entity' AND name = 'EdgeHub'")
        eid = f"e{ent[0][0]}"
        seen: list[str] = []
        token, pages = eid, 0
        while token and pages < 20:
            r = client.call("get", {"ids": [token], "max_tokens": 2000}, now=now)
            seen += re.findall(r"\bm\d+\b", r.text)
            token = r.structured["items"][0].get("continuation")
            pages += 1
        res.check("edges_paginated", pages > 1, f"{pages} page(s)")
        res.check("edges_each_live_edge_exactly_once",
                  sorted(seen) == sorted(mems),
                  f"missing {sorted(set(mems) - set(seen))}, repeated "
                  f"{sorted({x for x in seen if seen.count(x) > 1})}")  # fmt: skip
    finally:
        client.close()


def run(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-R6", mode="fixture")
    client = ctx.factory(ctx.store("b_r6_recall.db"), scope="personal")
    try:
        seed_corpus(client, fx.CORPUS, "2026-10-01T09:00:00Z")
        for i, q in enumerate(fx.CORPUS_QUERIES):
            r = client.call("recall", {"query": q}, now="2026-10-22T09:00:00Z")
            check_response(res, f"q{i}", q, r)
            res.check(f"q{i}_nonempty", bool(r.structured["rows"]), q)
        r = client.call("recall", {"query": fx.CORPUS_QUERIES[0], "budget": 40},
                        now="2026-10-22T09:00:00Z")  # fmt: skip
        res.check("small_budget_respected", est_tokens(r.text) <= 40, f"{est_tokens(r.text)} est.")
    finally:
        client.close()
    _check_empty(res, ctx)
    _check_get(res, ctx)
    _check_edges(res, ctx)
    res.metrics["violations"] = float(len(res.violations))
    return res
