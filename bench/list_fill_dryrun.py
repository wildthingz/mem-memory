"""`bench list-fill-dryrun` (MEM2-268): the round-level list fill, dry run on the recorded pools.

No models. Over the LME **tune** pool records (`mem calibrate`), the delivered-rows dump (`bench
delivered-dump`) and the gate's Platt + `tau_list`, for the list-mode (cued) questions, per arm:
how many of the answerable multi-session questions get *every* gold round delivered (the answerer
reads delivered rounds, not sessions: research `multi-session-research.md` Part 0), rows and
estimated tokens (the list-mode cost model, `listmode.tokens_for`), abstention changes on the cued
pools (answerable and negative), and the effect on the cued questions of the other types.

Arms: `today` (the delivery as recorded); `fill_N` (the switch `read.list_fill`: the `tau_list`
abstention test unchanged, then the top `max(N, rows above tau_list)` rows by p, cut from the bottom
to the token cap); `refit_round` (`tau_list` refitted with the gold *round* as the coverage unit,
`listfit.fit_list_model(unit="round")`, the cut as the test too); and, to reproduce the research's
`sim.py`, `sim_top_N` / `sim_tau_T` (delivered ∪ top N, or ∪ p >= T, on every multi-session
question, no cue test, no cap). Question texts come from the LME file (as `fit-tau-list`), streamed,
for the pool ids only; pools of non-tune questions are refused.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from bench.metrics import stats
from mem import aggcue, listfit, listmode
from mem.calibrate import QueryPool

NS = (12, 16, 20)
SIM_TAUS = (0.005,)  # the research's variant (b) lands about here (round floor 0.95)
MULTI_SESSION = "multi-session"


@dataclass(frozen=True)
class Delivered:
    """One line of the delivered dump: the rows recall showed (every page), and the gold."""

    qid: str
    category: str
    rows: list[str]
    gold_rounds: set[str]
    gold_sessions: set[str]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Delivered:
        return cls(d["qid"], str(d.get("category", "")), [*d["delivered"], *d.get("paged", [])],
                   set(d["gold_rounds"]), set(d.get("gold_sessions", ())))  # fmt: skip


def read_delivered(path: Path) -> dict[str, Delivered]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = Delivered.from_dict(json.loads(line))
            out[d.qid] = d
    return out


@dataclass(frozen=True)
class Question:
    qid: str
    category: str
    cued: bool
    keys: list[str]  # pool keys by p descending
    p: list[float]
    d: Delivered

    def above(self, tau: float) -> list[str]:
        return [k for k, p in zip(self.keys, self.p, strict=True) if p >= tau]


Arm = Callable[[Question], list[str]]


def _questions(
    pools: Sequence[QueryPool], delivered: Mapping[str, Delivered], queries: Mapping[str, str],
    platt: tuple[float, float],
) -> list[Question]:  # fmt: skip
    a, b = platt
    out = []
    for pool in pools:
        qid = listfit.question_of(pool.qid)
        if not pool.answerable or qid not in delivered:
            continue
        items = sorted(enumerate(pool.items), key=lambda t: (-(a * t[1].z + b), t[0]))
        out.append(Question(qid, delivered[qid].category or pool.category,
                            aggcue.is_aggregation(queries.get(qid, "")),
                            [i.key for _, i in items], [listfit.sigmoid(a * i.z + b) for _, i in items],
                            delivered[qid]))  # fmt: skip
    return out


def _groups(keys: Iterable[str]) -> int:
    return len({listfit.session_of(k) for k in keys})


def tokens(keys: Sequence[str]) -> int:
    """The list-mode cost model of a delivery (0 for an abstention)."""
    return listmode.tokens_for(len(keys), _groups(keys)) if keys else 0


def capped(keys: Sequence[str], cap: int) -> list[str]:
    """As `_recall_list` does: the lowest-p rows are dropped until the response fits the cap."""
    rows = list(keys)
    while rows and tokens(rows) > cap:
        rows.pop()
    return rows


def fill_arm(n: int, tau_list: float, cap: int) -> Arm:
    def arm(q: Question) -> list[str]:
        if not q.cued:
            return q.d.rows  # the normal path: untouched
        above = q.above(tau_list)
        if not above:
            return q.d.rows  # the abstention test is unchanged
        return capped(q.keys[: max(n, len(above))], cap)

    return arm


def tau_arm(tau: float, cap: int) -> Arm:
    """`tau` as the cut and the abstention test (the refit arm), cued questions only."""

    def arm(q: Question) -> list[str]:
        return capped(q.above(tau), cap) if q.cued else q.d.rows

    return arm


def sim_top_arm(n: int) -> Arm:
    return lambda q: list(dict.fromkeys([*q.d.rows, *q.keys[:n]]))


def sim_tau_arm(tau: float) -> Arm:
    return lambda q: list(dict.fromkeys([*q.d.rows, *q.above(tau)]))


def _summary(qs: Sequence[Question], arm: Arm, today: Arm) -> dict[str, Any]:
    rows, complete, sessions, toks, recovered, lost = {}, 0, 0, [], [], []
    for q in qs:
        got = arm(q)
        rows[q.qid] = len(got)
        full = bool(q.d.gold_rounds) and q.d.gold_rounds <= set(got)
        was = bool(q.d.gold_rounds) and q.d.gold_rounds <= set(today(q))
        complete += full
        sessions += bool(q.d.gold_sessions) and q.d.gold_sessions <= {
            listfit.session_of(k) for k in got
        }
        if got:
            toks.append(tokens(got))
        if full and not was:
            recovered.append(q.qid)
        if was and not full:
            lost.append(q.qid)
    n = len(qs)
    counts = list(rows.values())
    return {"complete": complete, "sessions_covered": sessions, "n": n, "rows": rows,
            "rows_mean": round(sum(counts) / n, 2) if n else None,
            "rows_p95": stats.percentile_nearest_rank(counts, 0.95) if n else None,
            "tokens_p50": stats.percentile_nearest_rank(toks, 0.5) if toks else 0,
            "tokens_p95": stats.percentile_nearest_rank(toks, 0.95) if toks else 0,
            "tokens_max": max(toks) if toks else 0,
            "recovered": sorted(recovered), "lost": sorted(lost)}  # fmt: skip


def _abstentions(pools: Sequence[QueryPool], cued: set[str], platt: tuple[float, float],
                 tau: float) -> dict[str, int]:  # fmt: skip
    """Over the cued pools: how many abstain (nothing at or above `tau`), answerable and negative."""
    a, b = platt
    out = {"answerable": 0, "negative": 0}
    for pool in pools:
        if listfit.question_of(pool.qid) not in cued:
            continue
        if all(listfit.sigmoid(a * i.z + b) < tau for i in pool.items):
            out["answerable" if pool.answerable else "negative"] += 1
    return out


def run(
    pools: Sequence[QueryPool], delivered: Mapping[str, Delivered], queries: Mapping[str, str],
    platt: tuple[float, float], tau_list: float, *, ns: Sequence[int] = NS,
    cap: int = listmode.CAP_TOKENS, refit: bool = True, wrong: Iterable[str] = (),
) -> dict[str, Any]:  # fmt: skip
    qs = _questions(pools, delivered, queries, platt)
    cued = {q.qid for q in qs if q.cued}
    today: Arm = lambda q: q.d.rows
    arms: dict[str, Arm] = {"today": today}
    for n in ns:
        arms[f"fill_{n}"] = fill_arm(n, tau_list, cap)
    taus: dict[str, float] = {"today": tau_list, **{f"fill_{n}": tau_list for n in ns}}
    refit_report: dict[str, Any]
    if refit:
        try:
            lm, rep = listfit.fit_list_model(pools, queries, platt, cap=cap, unit="round")
        except ValueError as e:  # too few aggregation questions for the nested CV
            refit_report = {"skipped": str(e)}
        else:
            arms["refit_round"] = tau_arm(lm["tau_list"], cap)
            taus["refit_round"] = lm["tau_list"]
            refit_report = {"tau_list": lm["tau_list"], "floor": lm["floor"], "n": rep["n"],
                            "feasible": lm["nested"]["feasible"],
                            "mean_heldout_coverage": lm["nested"]["mean_heldout_coverage"],
                            "worst_heldout_coverage": lm["nested"]["worst_heldout_coverage"],
                            "outer_folds": lm["nested"]["outer_folds"]}  # fmt: skip
    else:
        refit_report = {"skipped": "not asked (--no-refit)"}
    sim: dict[str, Arm] = {**{f"sim_top_{n}": sim_top_arm(n) for n in ns},
                           **{f"sim_tau_{t}": sim_tau_arm(t) for t in SIM_TAUS}}  # fmt: skip
    ms = [q for q in qs if q.category == MULTI_SESSION]
    other = [q for q in qs if q.cued and q.category != MULTI_SESSION]
    listed = [q for q in qs if q.cued]
    abst = {name: _abstentions(pools, cued, platt, tau) for name, tau in taus.items()}
    wrong = set(wrong)
    out = {
        "n_questions": len(qs), "cued": sorted(cued), "ns": list(ns), "cap": cap,
        "tau_list": tau_list, "platt": list(platt),
        "multi_session": {"n": len(ms), "n_cued": sum(q.cued for q in ms),
                          "arms": {name: _summary(ms, arm, today) for name, arm in {**arms, **sim}.items()}},
        "other_list_mode": {"n": len(other),
                            "arms": {name: _summary(other, arm, today) for name, arm in arms.items()}},
        "list_mode": {"n": len(listed),
                      "arms": {name: _summary(listed, arm, today) for name, arm in arms.items()}},
        "abstentions": abst,
        "abstention_changes": {name: sum(abs(abst[name][k] - abst["today"][k]) for k in abst[name])
                               for name in abst},
        "refit_round": refit_report,
    }  # fmt: skip
    if wrong:  # the research's "recovered in wrong qids": gold rounds newly delivered per wrong qid
        for name, arm in {**arms, **sim}.items():
            out["multi_session"]["arms"][name]["recovered_in_wrong"] = {
                q.qid: len((q.d.gold_rounds & set(arm(q))) - set(q.d.rows))
                for q in ms
                if q.qid in wrong
            }
    return out


def table(rep: dict[str, Any]) -> str:
    ms, other = rep["multi_session"], rep["other_list_mode"]
    lines = [(f"{rep['n_questions']} answerable tune questions, {len(rep['cued'])} cued (list mode);"
              f" cap {rep['cap']} tokens; tau_list {rep['tau_list']:.6f}"),
             f"multi-session: {ms['n']} answerable ({ms['n_cued']} cued)",
             (f"{'arm':14s} {'complete':>9s} {'sessions':>9s} {'rows mean/p95':>14s}"
              f" {'tokens p50/p95/max':>19s} {'abst. changes':>13s}  recovered / lost")]  # fmt: skip
    for name, s in ms["arms"].items():
        ch = rep["abstention_changes"].get(name)
        lines.append(f"{name:14s} {s['complete']:>4d}/{ms['n']:<4d} {s['sessions_covered']:>4d}/{ms['n']:<4d}"
                     f" {s['rows_mean']:>7.1f}/{s['rows_p95']:<6d} {s['tokens_p50']:>6d}/{s['tokens_p95']:<5d}/{s['tokens_max']:<6d}"
                     f" {('-' if ch is None else str(ch)):>13s}  {','.join(s['recovered']) or '-'} / {','.join(s['lost']) or '-'}")  # fmt: skip
        if s.get("recovered_in_wrong"):
            lines.append(f"{'':14s} recovered in wrong: {s['recovered_in_wrong']}")
    lines.append(f"other list-mode questions (cued, not multi-session): {other['n']}")
    for name, s in other["arms"].items():
        lines.append(f"{name:14s} {s['complete']:>4d}/{other['n']:<4d} {s['sessions_covered']:>4d}/{other['n']:<4d}"
                     f" {s['rows_mean']:>7.1f}/{s['rows_p95']:<6d} {s['tokens_p50']:>6d}/{s['tokens_p95']:<5d}/{s['tokens_max']:<6d}"
                     f" {'':>13s}  {','.join(s['recovered']) or '-'} / {','.join(s['lost']) or '-'}")  # fmt: skip
    lm = rep["list_mode"]
    lines.append(f"all list-mode questions: {lm['n']}; tokens p95 per arm: "
                 + ", ".join(f"{n} {s['tokens_p95']}" for n, s in lm["arms"].items()))  # fmt: skip
    lines.append("abstentions on the cued pools (answerable / negative): "
                 + ", ".join(f"{n} {a['answerable']}/{a['negative']}" for n, a in rep["abstentions"].items()))  # fmt: skip
    r = rep["refit_round"]
    lines.append("refit on gold rounds: " + (r["skipped"] if "skipped" in r else
                 f"tau_list {r['tau_list']:.6f} (floor {r['floor']}, n {r['n']}, feasible {r['feasible']},"
                 f" held-out coverage mean {r['mean_heldout_coverage']:.3f} worst {r['worst_heldout_coverage']:.3f})"))  # fmt: skip
    return "\n".join(lines)


def question_texts(path: Path, ids: Iterable[str]) -> dict[str, str]:
    """The question text of each id, streamed from the LME file (one question in memory)."""
    from bench.datasets import lme

    want, out = set(ids), {}
    for q in lme.iter_questions(path):
        if q["question_id"] in want:
            out[q["question_id"]] = q["question"]
            if len(out) == len(want):
                break
    return out


def register(app: typer.Typer) -> None:
    @app.command("list-fill-dryrun")
    def list_fill_dryrun(
        records_file: Annotated[Path, typer.Argument(help="LME tune pool records (`mem calibrate`).")],
        delivered: Annotated[Path, typer.Option(help="delivered.jsonl (`bench delivered-dump`).")],
        gate: Annotated[Path, typer.Option(help="The gate (Platt and list_model.tau_list).")],
        lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json (the query texts).")],
        out: Annotated[Path, typer.Option(help="Write the report as JSON (table next to it, .txt).")],
        ns: Annotated[str, typer.Option(help="Fill sizes, comma-separated.")] = ",".join(map(str, NS)),
        cap: Annotated[int, typer.Option(help="List-mode token cap (ADR 0023).")] = listmode.CAP_TOKENS,
        refit: Annotated[bool, typer.Option(help="Also fit tau_list on gold rounds (arm c).")] = True,
        e1: Annotated[Path | None, typer.Option(help="e1-tune.json: name the wrong multi-session"
                                                     " questions whose gold an arm recovers.")] = None,
    ) -> None:  # fmt: skip
        """Round-level list fill (MEM2-268), dry run on recorded tune pools: no model, no API."""
        from bench.cli import MANIFEST
        from bench.datasets import lme
        from mem import calibrate

        fp_all, pools = calibrate.read_records(records_file)
        g = json.loads(gate.read_text(encoding="utf-8"))
        if g.get("fingerprint_hash") != fp_all:
            typer.echo(f"REFUSED: the records' fp_all {fp_all[:12]} is not the gate's fingerprint_hash"
                       f" {str(g.get('fingerprint_hash'))[:12]}")  # fmt: skip
            raise typer.Exit(2)
        if not (g.get("list_model") or {}).get("tau_list"):
            typer.echo("REFUSED: the gate has no list_model.tau_list (ADR 0023)")
            raise typer.Exit(2)
        split = {r.question_id: r.split for r in lme.read_manifest(MANIFEST)}
        ids = {listfit.question_of(p.qid) for p in pools}
        bad = sorted(q for q in ids if split.get(q) != "tune")
        if bad:
            typer.echo(f"REFUSED: {len(bad)} pool question(s) are not LME tune (first: {bad[0]}):"
                       " the dry run is tune only")  # fmt: skip
            raise typer.Exit(2)
        queries = question_texts(lme_file, ids)
        wrong: list[str] = []
        if e1 is not None:
            items = json.loads(e1.read_text(encoding="utf-8"))["items"]
            wrong = [i["qid"] for i in items if i["condition"] == "mem" and not i["correct"]
                     and i["type"] == MULTI_SESSION and not lme.is_abs(i["qid"])]  # fmt: skip
        rep = run(pools, read_delivered(delivered), queries, (g["platt_a"], g["platt_b"]),
                  float(g["list_model"]["tau_list"]), ns=[int(x) for x in ns.split(",") if x],
                  cap=cap, refit=refit, wrong=wrong)  # fmt: skip
        text = table(rep)
        typer.echo(text)
        out.write_text(json.dumps(rep, sort_keys=True, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")  # fmt: skip
        out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
