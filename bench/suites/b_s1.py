"""B-S1 (T0 form): tool list snapshot, then remember -> recall -> get over the in-process client.

The HTTP, stdio and `claude -p` legs are human-run at the gate: deliverables/runbooks/p1-gate.md.
"""

from __future__ import annotations

from bench.adapters import ours
from bench.contract_check import structured_violations
from bench.suites.base import Context, SuiteResult
from bench.suites.helpers import ROW
from mem import contracts

TOKEN_TEXT = "the bench token is 7f3c"


def run(ctx: Context) -> SuiteResult:
    res = SuiteResult("B-S1", mode="in_memory")
    res.check(
        "tool_list_equals_frozen_contract",
        (ctx.list_tools or ours.list_tool_schemas)() == contracts.mcp_tools()["tools"],
    )
    client = ctx.factory(ctx.store("b_s1.db"), scope="personal")
    try:
        saved = client.call(
            "remember",
            {"text": TOKEN_TEXT, "type": "fact", "about": ["new:topic:Bench"]},
            now="2026-10-22T09:00:00Z",
        )
        s = saved.structured
        res.check("remember_contract", not structured_violations("remember", s),
                  "; ".join(structured_violations("remember", s)))  # fmt: skip
        res.check("remember_saved", s.get("status") == "saved" and bool(s.get("id")), str(s))
        res.check("remember_subjects", any("Bench" in x for x in s.get("subjects", [])),
                  str(s.get("subjects")))  # fmt: skip
        mem_id = s.get("id") or ""

        rec = client.call("recall", {"query": "bench token"}, now="2026-10-22T10:00:00Z")
        bad = structured_violations("recall", rec.structured)
        res.check("recall_contract", not bad, "; ".join(bad))
        rows = rec.structured.get("rows", [])
        row = next((r for r in rows if r.get("id") == mem_id), None)
        res.check("recall_returns_saved_memory", row is not None, f"{mem_id} not in {rows}")
        if row:
            res.check("recall_row_fields", (row["kind"], row["type"], row["date"], row["about"],
                      row["flags"]) == ("memory", "fact", "2026-10-22", ["Bench"], []), str(row))  # fmt: skip
        text_rows = [ln for ln in rec.text.splitlines() if ROW.match(ln)]
        res.check("recall_text_has_row_for_id", any(ln.startswith(mem_id + " · ")
                  for ln in text_rows), rec.text)  # fmt: skip

        got = client.call("get", {"ids": [mem_id]}, now="2026-10-22T10:05:00Z")
        bad = structured_violations("get", got.structured)
        res.check("get_contract", not bad, "; ".join(bad))
        items = got.structured.get("items", [])
        res.check("get_text_verbatim", len(items) == 1 and items[0].get("found") is True
                  and items[0].get("text") == TOKEN_TEXT, str(items))  # fmt: skip
    finally:
        client.close()
    res.metrics["failures"] = float(len(res.violations))
    return res
