"""`bench window-dryrun` (MEM2-266): the anchored-window gate boost replayed on recorded pools. No models.

For each recorded tune pool (`mem calibrate --records-out`: chunk ref and raw reranker logit z per
member) the query's relative date is resolved with the `mem.reldates` rules against the question day
(`mem.window`), each member's session date comes from the LongMemEval file, and the read path's
selection is replayed: the gate (Platt p, tau_keep, tau_high/gap) and the 8-row / 600-token budget,
or list mode's tau_list and token cap for an aggregation question. The boost adds `delta` to z of the
in-window members and delivers the in-window rows first; nothing is filtered. The replay without the
boost is checked against the recorded delivery (`delivered.jsonl`), so the table is about the boost,
not about the replay.

Only tune questions are read: the split is re-derived from the file (`lme.build_manifest`) and a
record of a test question is refused.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer

from bench.datasets import lme
from mem import aggcue, config, listmode, reldates, window
from mem.common import estimate_tokens
from mem.gate import Gate

MIN_NET = 2  # the stop point: fewer net rescues (rescued gold - gold pushed out) drops the option
GATE_KEYS = ("platt_a", "platt_b", "tau_keep", "tau_high", "gap")
DELTAS = (0.5, 1.0, 2.0, 3.0)
SLACKS = (0.1, 0.2, 0.3, 0.5)
SCOPE = "personal"


@dataclass(frozen=True)
class Query:
    qid: str
    kind: str  # answerable | negative (the B-R2 haystack, gold sessions removed)
    category: str
    question: str
    ref: date  # the question day
    keys: list[str]  # pool members, `<session>#<round>`, in pool order
    zs: list[float]
    days: list[date]  # each member's session date (valid_from, UTC)
    lines: list[str]  # each member's recall row without its p
    gold: frozenset[str]
    recorded: list[str] | None  # the recorded delivery (answerable only)


def _p_text(p: float) -> str:
    return f"{p:.2f}".removeprefix("0")


def flags(q: Query, slack: float) -> list[bool]:
    ws = window.windows(q.question, q.ref, slack)
    return [window.inside(d, ws) for d in q.days]


def deliver(q: Query, gate: Gate, *, delta: float, flags: Sequence[bool]) -> list[str]:
    """The keys recall would deliver, in order, with `delta` added to the in-window z."""
    defaults = config.default_config().retrieval
    ps = [gate.platt(z + (delta if f else 0.0)) for z, f in zip(q.zs, flags, strict=True)]
    line = lambda i: f"{q.lines[i]} · {_p_text(ps[i])}"
    if aggcue.is_aggregation(q.question):
        lm = gate.lister
        assert lm is not None, "list mode needs the gate's list_model"
        order = sorted(
            (i for i in range(len(ps)) if ps[i] >= lm.tau_list), key=lambda i: (-ps[i], i)
        )
        order = window.in_window_first(order, flags)
        chosen: list[int] = []
        for i in order:
            if (
                estimate_tokens(_render_list(q, [*chosen, i], line, len(order), lm.cap_tokens))
                > lm.cap_tokens
            ):
                break
            chosen.append(i)
        return [q.keys[i] for i in chosen]
    kept, abstained = gate.select(ps)
    if abstained:
        return []
    out: list[int] = []
    for i in window.in_window_first(kept, flags):
        if len(out) >= defaults["limit"]:
            break
        trial = [
            f"{len(out) + 1} results (of {len(ps)} checked) · {SCOPE}",
            *(line(j) for j in out),
            line(i),
        ]
        if estimate_tokens("\n".join(trial)) > defaults["budget"]:
            break
        out.append(i)
    return [q.keys[i] for i in out]


def _render_list(q: Query, chosen: list[int], line: Any, matches: int, cap: int) -> str:
    """List mode's response as `read._recall_list` lays it out (no paging)."""
    by_day: dict[date, list[str]] = {}
    for i in chosen:
        by_day.setdefault(q.days[i], []).append(line(i))
    groups = [
        (f"{d.isoformat()} · chat session {d.isoformat()}", ls) for d, ls in sorted(by_day.items())
    ]
    n = len(groups)
    header = (f"{len(chosen)} results (of {len(q.keys)} checked) · {SCOPE} · list mode ·"
              f" {n} {'session' if n == 1 else 'sessions'}")  # fmt: skip
    more = listmode.MORE.format(n=matches - len(chosen), cap=cap) if len(chosen) < matches else ""
    return listmode.layout(header, groups, more=more)


def compare(q: Query, gate: Gate, *, delta: float, slack: float) -> dict[str, Any]:
    ws = window.windows(q.question, q.ref, slack)
    f = [window.inside(d, ws) for d in q.days]
    base = deliver(q, gate, delta=0.0, flags=[False] * len(q.keys))
    new = deliver(q, gate, delta=delta, flags=f) if ws else base
    ps = {
        k: gate.platt(z + (delta if fl else 0.0)) for k, z, fl in zip(q.keys, q.zs, f, strict=True)
    }
    lines = dict(zip(q.keys, q.lines, strict=True))
    admitted = [k for k in new if k not in base and k not in q.gold]
    gold_base, gold_new = q.gold & set(base), q.gold & set(new)
    cut = None
    in_pool = [i for i, k in enumerate(q.keys) if k in q.gold]
    if q.kind == "answerable" and q.category == "temporal-reasoning" and in_pool and not gold_base:
        best = max(in_pool, key=lambda i: q.zs[i])
        cut = {"z": round(q.zs[best], 2), "in_window": f[best], "rescued": bool(gold_new)}
    return {
        "qid": q.qid, "kind": q.kind, "category": q.category, "delta": delta, "slack": slack,
        "window": [[w.phrase, w.value, w.start.isoformat(), w.end.isoformat()] for w in ws],
        "in_window": sum(f), "base": base, "new": new,
        "rescued": q.kind == "answerable" and not gold_base and bool(gold_new),
        "pushed_out": sorted(gold_base - gold_new),
        "admitted": admitted,
        "admitted_tokens": sum(estimate_tokens(f"\n{lines[k]} · {_p_text(ps[k])}") for k in admitted),
        "newly_answered": q.kind == "negative" and not base and bool(new),
        "gate_cut": cut,
        "replay_ok": None if q.recorded is None else base == q.recorded,
    }  # fmt: skip


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["kind"] == "answerable"]
    neg = [r for r in rows if r["kind"] == "negative"]
    rescued = [r["qid"] for r in ans if r["rescued"]]
    pushed = [r["qid"] for r in ans if r["pushed_out"]]
    adm = [r for r in ans if r["admitted"]]
    checked = [r for r in rows if r["replay_ok"] is not None]
    cuts = [r for r in ans if r["gate_cut"]]
    return {
        "fires": {"answerable": sum(1 for r in ans if r["window"]),
                  "negative": sum(1 for r in neg if r["window"])},
        "fires_by_category": _count(r["category"] for r in ans if r["window"]),
        "rescued": len(rescued), "rescued_qids": rescued,
        "pushed_out": len(pushed), "pushed_qids": pushed,
        "net": len(rescued) - len(pushed),
        "gate_cuts": {"n": len(cuts), "rescued": sum(1 for r in cuts if r["gate_cut"]["rescued"]),
                      "detail": {r["qid"]: r["gate_cut"] for r in cuts}},
        "admitted": {"questions": len(adm), "rows": sum(len(r["admitted"]) for r in adm),
                     "tokens": sum(r["admitted_tokens"] for r in adm), "qids": [r["qid"] for r in adm]},
        "negatives": {"newly_answered": sum(1 for r in neg if r["newly_answered"]),
                      "newly_answered_qids": [r["qid"] for r in neg if r["newly_answered"]],
                      "rows": sum(len(r["admitted"]) for r in neg)},
        "replay": {"checked": len(checked), "match": sum(1 for r in checked if r["replay_ok"]),
                   "mismatch": [r["qid"] for r in checked if not r["replay_ok"]]},
    }  # fmt: skip


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def verdict(net: int) -> str:
    return "pass" if net >= MIN_NET else "drop"


# ---- loading ------------------------------------------------------------------------------------


def _rows(q: dict[str, Any], exclude: set[str]) -> dict[str, tuple[date, str]]:
    """Chunk ref -> (session date, recall row without p) as `ingest_session` would store them: chunk
    ids in import order, the `dates:` field from the round text against its session day."""
    out, cid = {}, 0
    for sid, iso, rounds in lme.haystack_of(q, exclude=exclude).sessions:
        day = date.fromisoformat(iso[:10])
        for r in rounds:
            cid += 1
            text = "\n".join(t for t in (r.user, r.assistant) if t)
            parts = [f"c{cid}", "chat_session", day.isoformat(), f"chat session {day.isoformat()}"]
            if shown := reldates.render(reldates.annotation(text, day), reldates.ROW_CHARS):
                parts.append("dates: " + shown)
            out[lme.ref(sid, r.index)] = (day, " · ".join(parts))
    return out


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load(records: Path, delivered: Path, dataset: Path) -> list[Query]:
    recs = [r for r in _jsonl(records) if "qid" in r]  # the first line is the {"fp_all"} header
    dl = {r["qid"]: r for r in _jsonl(delivered)}
    want = {r["qid"].split(":", 1)[0] for r in recs}
    strata, raw = [], {}
    for q in lme.iter_questions(dataset):  # streamed: one question in memory at a time
        strata.append({"question_id": q["question_id"], "question_type": q["question_type"]})
        if q["question_id"] in want:
            raw[q["question_id"]] = q
    split = {m.question_id: m.split for m in lme.build_manifest(strata)}
    if bad := sorted(w for w in want if split.get(w) != "tune"):
        raise ValueError(
            f"records name questions outside the tune split (the dry run reads tune only): {bad[:5]}"
        )
    out = []
    for r in recs:
        qid, kind = r["qid"].split(":", 1)
        q = raw[qid]
        rows = _rows(q, set(q["answer_session_ids"]) if kind == "negative" else set())
        keys = [it["key"] for it in r["items"]]
        gold = (
            dl[qid]["gold_rounds"] if qid in dl else [it["key"] for it in r["items"] if it["rel"]]
        )
        out.append(Query(
            qid=qid, kind=kind, category=q["question_type"], question=q["question"],
            ref=date.fromisoformat(lme.parse_lme_date(q["question_date"])[:10]), keys=keys,
            zs=[float(it["z"]) for it in r["items"]], days=[rows[k][0] for k in keys],
            lines=[rows[k][1] for k in keys],
            gold=frozenset(gold if kind == "answerable" else ()),
            recorded=dl[qid]["delivered"] if kind == "answerable" and qid in dl else None,
        ))  # fmt: skip
    return out


def load_gate(path: Path) -> Gate:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Gate(**{k: v for k, v in data.items() if k in Gate.__dataclass_fields__})


# ---- the run ------------------------------------------------------------------------------------


def run(queries: Sequence[Query], gate: Gate, deltas: Sequence[float], slacks: Sequence[float]
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # fmt: skip
    grid, per_query = [], []
    for slack in slacks:
        for delta in deltas:
            rows = [compare(q, gate, delta=delta, slack=slack) for q in queries]
            grid.append({"slack": slack, "delta": delta, **summarise(rows)})
            per_query += rows
    # the best cell: most net rescues, then fewest abstention false accepts, fewest admitted rows,
    # the smallest delta and slack (the least change that does it)
    best = min(grid, key=lambda g: (-g["net"], g["negatives"]["newly_answered"], g["admitted"]["rows"],
                                    g["delta"], g["slack"]))  # fmt: skip
    summary = {"n": {"answerable": sum(q.kind == "answerable" for q in queries),
                     "negative": sum(q.kind == "negative" for q in queries)},
               "replay": grid[0]["replay"], "grid": grid,
               "best": {"slack": best["slack"], "delta": best["delta"], "net": best["net"]},
               "verdict": verdict(best["net"]), "min_net": MIN_NET}  # fmt: skip
    return summary, per_query


def table(summary: dict[str, Any]) -> str:
    n = summary["n"]
    lines = [
        (
            f"Replay without the boost = recorded delivery on {summary['replay']['match']}/"
            f"{summary['replay']['checked']} answerable tune queries."
        ),
        "",
        (
            "| slack | Δ | parser fires (ans / neg) | temporal gate cuts rescued | rescued | gold pushed out | net"
            " | non-gold admitted: questions / rows / tokens | negatives newly answered (rows) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in summary["grid"]:
        c, a, ng = g["gate_cuts"], g["admitted"], g["negatives"]
        lines.append(
            f"| {g['slack']} | {g['delta']} | {g['fires']['answerable']} / {g['fires']['negative']}"
            f" (of {n['answerable']} / {n['negative']}) | {c['rescued']} of {c['n']} | {g['rescued']}"
            f" | {g['pushed_out']} | {g['net']} | {a['questions']} / {a['rows']} / {a['tokens']}"
            f" | {ng['newly_answered']} ({ng['rows']}) |"
        )
    b = summary["best"]
    lines += ["", (f"Best: slack {b['slack']}, Δ {b['delta']}: net {b['net']} → **{summary['verdict']}**"
                   f" (stop point: net < {summary['min_net']} drops the option).")]  # fmt: skip
    return "\n".join(lines) + "\n"


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def register(app: typer.Typer) -> None:
    @app.command("window-dryrun")
    def window_dryrun(
        records: Annotated[Path, typer.Option(help="tune_records.jsonl (mem calibrate --records-out).")],
        delivered: Annotated[Path, typer.Option(help="delivered.jsonl (bench delivered-dump on the same config and gate).")],
        gate: Annotated[Path, typer.Option(help="The gate JSON the records were served with.")],
        dataset: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json (tune questions only are kept).")],
        out: Annotated[Path, typer.Option(help="Output dir.")] = Path("data/window-dryrun"),
        deltas: Annotated[str, typer.Option(help="Comma-separated Δ (logits).")] = ",".join(map(str, DELTAS)),
        slacks: Annotated[str, typer.Option(help="Comma-separated slack per day of distance.")] = ",".join(map(str, SLACKS)),
    ) -> None:  # fmt: skip
        """The anchored-window gate boost on recorded tune pools (MEM2-266, ADR 0031): no models."""
        summary, rows = run(
            load(records, delivered, dataset), load_gate(gate), _floats(deltas), _floats(slacks)
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out / "per_query.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        text = table(summary)
        (out / "table.md").write_text(text)
        typer.echo(text)
