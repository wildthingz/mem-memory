"""B-C1 (latency at L5k) and B-C2 (tokens per recall) as gated from P2, plus the P2 form of B-R6.

B-C1 aggregates `bench latency-run` outputs (3 fresh-process runs; median of the runs' warm p95).
B-C2 counts the fixed 300-response sample with the setup's token counter; the gate also requires
the counter to be Anthropic `count_tokens` (`counter_exact`), because §0.4 says reported numbers
are client-visible tokens, and an estimate is labelled "est." and does not gate.
B-R6 checks every B-R1/B-R2 response against the contract, then the fixture `get`/edge checks: a
list-mode response (ADR 0023) against the list grammar (ADR 0023 "Response grammar", the paging tail
of ADR 0024), every other response against the pointer-row grammar; violations are counted per grammar.
"""

from __future__ import annotations

import re

from bench import latency, lme_eval
from bench.metrics.tokens import est_tokens
from bench.suites import b_r6
from bench.suites.base import BackendUnavailable, Context, SuiteResult
from bench.suites.lme_setup import fill_result, setup_of, shared_run
from mem import listmode, reldates

BUDGET = 600
EMPTY_MAX = 30
# `id · type · date · title · [about: …] · [dates: …] · p`; `about` is omitted when the node has none
# (chat sessions); `dates` only with `ingest.resolve_dates` on (ADR 0021), whole dates only.
ROW = re.compile(
    r"^(?P<id>[mcent]\d+) · (?P<type>[^·]+) · (?P<date>\d{4}-\d{2}-\d{2}|-) · (?P<title>[^·]+?)"
    r"(?: · about: (?P<about>[^·]*))?"
    rf"(?: · dates: (?P<dates>{reldates.ROW_FIELD}))? · (?P<p>\.?\d*\.?\d+)$"
)
HEADER = re.compile(r"^\d+ results? \(of \d+ checked\) · \w+")
# The P2 abstention line (architecture-v2 §3.2 step 6): `no relevant memories (best .22) · personal`;
# `(best …)` is absent when nothing was eligible. P1's `no results` is not a P2 response.
ABSTENTION = re.compile(r"^no relevant memories(?: \(best \d?\.\d\d\))? · (?:personal|work)$")


# List mode (ADR 0023 "Response grammar"; the paging tail is ADR 0024's): the header, then the instruction
# line, then per session `<date> · <name>` (date order, undated first) with its rows indented two spaces in
# the row grammar above; last, at most one over-cap line, never with the paging tail.
LIST_HEADER = re.compile(
    r"^(?P<n>\d+) results \(of \d+ checked\) · \w+ · list mode · (?P<g>\d+) sessions?"
    r"(?: · (?P<shown>\d+) of (?P<total>\d+) shown"
    r" · (?:more: recall\(…, page=(?P<next>\d+)-[0-9a-f]{10}\)|page (?P<k>\d+) of (?P<pages>\d+) · last page))?$"
)
LIST_MARK = re.compile(r"^\d+ results \(of \d+ checked\) · \w+ · list mode · ")
GROUP = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2}|-) · (?P<name>\S.*)$")
OVER_CAP = re.compile(
    r"^\+\d+ more matches above tau_list not shown \(token cap \d+\); narrow the query$"
)
INDENT = "  "


def is_list_text(text: str) -> bool:
    """A list-mode response: the instruction line under the header (`lme_eval.is_list_response`), or
    a list-mode header without it (then the list check says the instruction line is missing)."""
    return lme_eval.is_list_response(text) or bool(LIST_MARK.match(text))


def response_violations(text: str, abstained: bool) -> list[str]:
    if not abstained and is_list_text(text):
        return list_violations(text)
    return normal_violations(text, abstained)


def _row_violation(ln: str) -> str | None:
    m = ROW.match(ln)
    if not m:
        return f"row format: {ln[:80]!r}"
    if len(m["title"].split()) > 12:
        return f"title > 12 words: {ln[:80]!r}"
    if not 0 <= float(m["p"]) <= 1:
        return f"p outside [0,1]: {ln[:80]!r}"
    return None


def list_violations(text: str) -> list[str]:
    lines = text.splitlines()
    head = LIST_HEADER.match(lines[0]) if lines else None
    out = [] if head else [f"bad list header: {lines[0][:100] if lines else ''!r}"]
    if head and head["shown"] and int(head["shown"]) > int(head["total"]):
        out.append(f"list header: {head['shown']} shown of {head['total']}")
    if head and head["k"] and int(head["k"]) != int(head["pages"]):
        out.append(f"list header: last page {head['k']} of {head['pages']}")
    body = lines[1:]
    if body[:1] == [listmode.INSTRUCTION]:
        body = body[1:]
    else:
        out.append("missing instruction line under the header")
    if body and OVER_CAP.match(body[-1]):
        if head and head["shown"]:
            out.append("over-cap line with the paging tail (ADR 0024: paging replaces it)")
        body = body[:-1]
    rows = 0
    groups: list[tuple[str, int]] = []  # (date, rows under it)
    for ln in body:
        if ln.startswith(INDENT):
            if not groups:
                out.append(f"row before any group header: {ln[:80]!r}")
            else:
                groups[-1] = (groups[-1][0], groups[-1][1] + 1)
            rows += 1
            if v := _row_violation(ln[len(INDENT) :]):
                out.append(v)
        elif m := GROUP.match(ln):
            if len(m["name"].split()) > 12:
                out.append(f"group name > 12 words: {ln[:80]!r}")
            groups.append(("" if m["date"] == "-" else m["date"], 0))
        elif OVER_CAP.match(ln):
            out.append(f"over-cap line is not the last line: {ln[:80]!r}")
        else:
            out.append(f"not a group header or an indented row: {ln[:80]!r}")
    dates = [d for d, _ in groups]
    if dates != sorted(dates):
        out.append(f"groups not in date order: {[d or '-' for d in dates]}")
    out += [f"group {d or '-'} has no rows" for d, n in groups if n == 0]
    if head and int(head["n"]) != rows:
        out.append(f"list header says {head['n']} results, {rows} rows listed")
    if head and int(head["g"]) != len(groups):
        out.append(f"list header says {head['g']} sessions, {len(groups)} group headers")
    if est_tokens(text) > listmode.CAP_TOKENS:
        out.append(f"{est_tokens(text)} est. tokens > list cap {listmode.CAP_TOKENS}")
    return out


def normal_violations(text: str, abstained: bool) -> list[str]:
    lines = text.splitlines()
    if abstained:
        out = []
        if len(lines) != 1 or not ABSTENTION.match(lines[0]):
            out.append(f"empty result is not one abstention line: {text[:80]!r}")
        if est_tokens(text) > EMPTY_MAX:
            out.append(f"empty result {est_tokens(text)} est. tokens > {EMPTY_MAX}")
        return out
    out = []
    if not lines or not HEADER.match(lines[0]):
        out.append(f"bad header: {lines[0][:80] if lines else ''!r}")
    if lines and "uncalibrated" in lines[0]:
        out.append("P1 'uncalibrated' header in a P2 response")
    for ln in lines[1:]:
        if v := _row_violation(ln):
            out.append(v)
    if est_tokens(text) > BUDGET:
        out.append(f"{est_tokens(text)} est. tokens > budget {BUDGET}")
    return out


def run_b_r6(ctx: Context) -> SuiteResult:
    _, items, _ = shared_run(ctx)
    res = SuiteResult("B-R6", tier="t1", mode="ingest_verbatim")
    count = {"list": 0, "normal": 0}
    bad = {"list": 0, "normal": 0}
    for it in items:
        if it.kind == "abs":
            continue
        grammar = "list" if not it.abstained and is_list_text(it.r.text) else "normal"
        count[grammar] += 1
        for v in response_violations(it.r.text, it.abstained):
            bad[grammar] += 1
            res.check(f"{it.kind}:{it.qid}:{grammar}", False, v)
    res.items.append({"check": "responses", "ok": not res.violations, "n": len(items)})
    b_r6._check_empty(res, ctx)
    b_r6._check_get(res, ctx)
    b_r6._check_edges(res, ctx)
    res.metrics["violations"] = float(len(res.violations))
    res.metrics["responses_checked"] = float(sum(count.values()))
    for g in (
        "normal",
        "list",
    ):  # the list grammar (ADR 0023) on list-mode responses, the row grammar else
        res.metrics[f"responses_{g}"] = float(count[g])
        res.metrics[f"violations_{g}"] = float(bad[g])
    res.metrics["violations_fixture"] = float(len(res.violations) - sum(bad.values()))
    return res


def run_b_c2(ctx: Context) -> SuiteResult:
    s, items, sc = shared_run(ctx)
    res = SuiteResult("B-C2")
    fill_result(res, s, items, sc)
    res.metrics |= lme_eval.token_metrics(items, s.token_counter)
    res.metrics |= lme_eval.list_token_metrics(items, s.token_counter)  # ADR 0023, own cap
    res.metrics["counter_exact"] = float(s.token_counter_exact)
    res.notes.append("tokens: Anthropic count_tokens" if s.token_counter_exact
                     else "tokens: server estimate ceil(bytes/3) (est.), not count_tokens")  # fmt: skip
    return res


def run_b_c1(ctx: Context) -> SuiteResult:
    s = setup_of(ctx)
    res = SuiteResult("B-C1", tier="t1", mode="ingest_verbatim")
    if len(s.latency_runs) < latency.RUNS_REQUIRED:
        raise BackendUnavailable(
            f"B-C1 needs {latency.RUNS_REQUIRED} `bench latency-run --corpus L5k` outputs, "
            f"got {len(s.latency_runs)}"
        )
    runs = latency.load_runs(s.latency_runs)
    res.metrics |= latency.aggregate(runs)
    res.items = [{"run": str(p), **r} for p, r in zip(s.latency_runs, runs, strict=True)]
    res.notes.append("B-C1 at L5k on the serving host; median of 3 fresh-process runs' warm p95")
    return res
