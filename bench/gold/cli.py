"""`bench gold ...`: build PG-v1 (benchmarks-v2 §3.1). Order of use, with the time each takes:

    draft        (Claude drafts ~200 candidates; paid, sends PG-snapshot text to the API)
    check        (no-answer checks and the pre-filter: weak and copied drafts are dropped, instant)
    review       (accept / edit / reject drafts: ~45 min, you reword at least half)
    pool         (top 5 of ours, BM25, dense per question; needs the models)
    judge        (Claude grades every pooled pair 0 / 1 / 2; paid, hard --budget-usd)
    kappa-sample (you grade 50 of Claude's pairs blind: ~7 min; kappa >= 0.7 = Claude's labels stand)
    label        (only if kappa < 0.7: you grade the rest of the pairs yourself)
    status (counts, shortfalls, kappa, your real pace) / freeze / verify

PG-v1 holds no work-store content: everything works on a bench-owned import of the PG snapshot
(`bench snapshot-make --pg`, then `bench vault-import`), never on your live stores. State is in `--work` (default data/gold/v1/, git-ignored); the frozen files go to
`bench/gold/personal/v1/`.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer

from bench.gold import check as check_mod
from bench.gold import draft as draft_mod
from bench.gold import judge as judge_mod
from bench.gold import kappa as kappa_mod
from bench.gold import pg, pool, sampling, session
from bench.gold.bm25 import bm25_top
from bench.gold.chunks import ChunkIndex

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

StoresOpt = Annotated[
    Path, typer.Option(help="Dir with the bench-owned import (personal.db, work.db).")
]
WorkOpt = Annotated[Path, typer.Option(help="Work dir (drafts, decisions, labels).")]
ConfigOpt = Annotated[
    Path | None, typer.Option(help="config.yaml (vault.routes): work-routed folders are refused.")
]
DEFAULT_STORES, DEFAULT_WORK = Path("data/pg/stores"), Path("data/gold/v1")
FROZEN = Path(__file__).resolve().parent / "personal" / "v1"


def open_conns(stores_dir: Path) -> dict[str, sqlite3.Connection]:
    found = {
        n: stores_dir / f"{n}.db" for n in ("personal", "work") if (stores_dir / f"{n}.db").exists()
    }
    if not found:
        raise typer.BadParameter(
            f"no personal.db / work.db in {stores_dir}: run `bench vault-import`"
        )
    return {n: sqlite3.connect(f"file:{p}?mode=ro", uri=True) for n, p in found.items()}


def _state(
    stores_dir: Path, work: Path
) -> tuple[dict[str, sqlite3.Connection], ChunkIndex, session.Work]:
    conns = open_conns(stores_dir)
    return conns, ChunkIndex.from_conns(conns), session.Work(work)


@app.command("draft")
def draft_cmd(
    budget_usd: Annotated[float, typer.Option(help="Hard spend cap (B-C5 ledger).")],
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
    model: Annotated[str, typer.Option(help="Drafting model.")] = draft_mod.MODEL,
    effort: Annotated[str, typer.Option(help="low | medium | high")] = "medium",
    more: Annotated[
        int, typer.Option(help="Round k > 0: the same slots dealt to other notes.")
    ] = 0,
    yes: Annotated[
        bool, typer.Option(help="Do not ask before sending note text to the API.")
    ] = False,
    slots_per_unit: Annotated[
        int, typer.Option(help="Drafts per note or pair: raise it if a quota cannot be filled.")
    ] = sampling.SLOTS_PER_UNIT,
) -> None:
    """Draft ≈ 200 candidate questions (4 per note or note pair) with Claude."""
    from bench import llm

    conns, index, w = _state(stores_dir, work)
    units = sampling.sample_units(index, conns, slots_per_unit=slots_per_unit)
    if more:
        units = sampling.plan_slots([sampling.Unit(u.uid, u.paths, store=u.store) for u in units],
                                    shift=more, slots_per_unit=slots_per_unit)  # fmt: skip
    chars = sum(len(c.text) for u in units for p in u.paths for c in index.of_path(p))
    typer.echo(f"{len(units)} units, {sum(len(u.slots) for u in units)} drafts, "
               f"≈ {chars // 3:,} input tokens of note text.")  # fmt: skip
    if not yes:
        typer.confirm(f"This sends the text of those notes from your vault to the Anthropic API "
                      f"({model}, cap ${budget_usd:.2f}). Continue?", abort=True)  # fmt: skip
    client = llm.AnthropicLLM(model, effort)
    ledger = llm.Ledger(model, budget_usd)
    existing = w.drafts()
    try:
        rep = draft_mod.draft_all(units, index, client, ledger, existing=existing)
    except llm.BudgetExceeded as e:
        typer.echo(f"STOPPED by the budget cap: {e}")
        raise typer.Exit(1) from e
    draft_mod.write_drafts(w.path("drafts.jsonl"), rep.drafts)
    typer.echo(f"{len(rep.drafts)} drafts ({rep.parse_failures} unit(s) unparsable, "
               f"{rep.unknown_chunk} citing an unknown chunk, {rep.skipped_slots} slots skipped); "
               f"cost ${ledger.total:.2f}")  # fmt: skip
    typer.echo("  " + ", ".join(f"{c} {n}" for c, n in sorted(rep.by_category.items())))


@app.command("check")
def check_cmd(stores_dir: StoresOpt = DEFAULT_STORES, work: WorkOpt = DEFAULT_WORK) -> None:
    """No-answer rg check and advisory checks on every draft; flagged drafts are not shown in review."""
    _, index, w = _state(stores_dir, work)
    drafts = w.drafts()
    rows = [check_mod.check_draft(d, index) for d in drafts]
    for r in rows:
        w.log_event("checks.jsonl", r)
    bad = [r for r in rows if not r["ok"]]
    typer.echo(f"{len(rows)} drafts checked, {len(bad)} flagged")
    pre = [r for r in rows if r.get("prefilter")]
    typer.echo(f"  pre-filter dropped {len(pre)}: "
               f"{sum(any(x.startswith('weak') for x in r['prefilter']) for r in pre)} weak, "
               f"{sum(any(x.startswith('lexical') for x in r['prefilter']) for r in pre)} "
               f"lexical-overlap failures")  # fmt: skip
    for r in bad[:20]:
        typer.echo(f"  {r['draft_id']}: {'; '.join(r['problems'])}")


@app.command("review")
def review_cmd(
    minutes: Annotated[float, typer.Option(help="Stop after this long (one sitting).")] = 45,
    snapshot_date: Annotated[str, typer.Option(help="t of the BM25 check (YYYY-MM-DD).")] = "",
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
) -> None:
    """One sitting: accept / edit / reject drafts. Stops at --minutes; run it again to continue."""
    from mem.common import normalize_as_of

    conns, index, w = _state(stores_dir, work)
    t = normalize_as_of(snapshot_date or "9999-12-31", "UTC")  # as recall reads it (ADR 0028)

    def bm25(d: dict[str, Any], k: int) -> list[str]:
        return bm25_top(conns, index, d.get("store", "personal"), d["question"],
                        d.get("filters", {}), k, t)  # fmt: skip

    session.review_sitting(w, index, bm25, minutes=minutes)


@app.command("pool")
def pool_cmd(
    snapshot_date: Annotated[str, typer.Option(help="t of the retrieval (YYYY-MM-DD).")],
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
    config: Annotated[Path | None, typer.Option(help="config.yaml")] = None,
    embedder: Annotated[
        str, typer.Option(help="st = pinned models; hash = test stand-ins.")
    ] = "st",
    device: Annotated[str | None, typer.Option()] = None,
    serve_port: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Pool: top 5 of ours, BM25 and dense per accepted question, minus judged chunks."""
    from bench.adapters.p4_ours import OursVaultBackend

    w = session.Work(work)
    qs = w.accepted()
    if not qs:
        raise typer.BadParameter("no accepted questions: run `bench gold review` first")
    be = OursVaultBackend(stores_dir, snapshot_date, config_path=config, models=embedder,
                          device=device, serve_port=serve_port)  # fmt: skip
    try:
        st = pool.build(w, qs, be)
    finally:
        be.close()
    typer.echo(f"{st.questions} questions, {st.pairs} pooled pairs ({st.new_pairs} new); "
               f"budget {pool.MAX_PAIRS}{'  OVER BUDGET' if st.over_budget else ''}")  # fmt: skip


@app.command("judge")
def judge_cmd(
    budget_usd: Annotated[
        float, typer.Option(help=f"Hard spend cap, at most {judge_mod.MAX_BUDGET_USD:g}.")
    ],
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
    model: Annotated[str, typer.Option(help="Judging model.")] = judge_mod.MODEL,
    effort: Annotated[str, typer.Option(help="low | medium | high")] = "low",
    yes: Annotated[
        bool, typer.Option(help="Do not ask before sending chunk text to the API.")
    ] = False,
) -> None:
    """Claude grades every pooled (question, chunk) pair 0 / 1 / 2 (T2: paid, needs the API)."""
    from bench import llm

    if not 0 < budget_usd <= judge_mod.MAX_BUDGET_USD:
        raise typer.BadParameter(f"--budget-usd must be in (0, {judge_mod.MAX_BUDGET_USD:g}]")
    _, index, w = _state(stores_dir, work)
    todo = judge_mod.todo_pairs(w)
    if not todo:
        typer.echo("no pooled pair is waiting for a grade")
        return
    typer.echo(f"{len(todo)} pooled pairs to judge ({model}, cap ${budget_usd:.2f}).")
    if not yes:
        typer.confirm("This sends the text of those chunks of your PG snapshot (personal/ and "
                      "projects/, no work notes) to the Anthropic API. Continue?", abort=True)  # fmt: skip
    client = llm.AnthropicLLM(model, effort)
    ledger = llm.Ledger(model, budget_usd)
    rep = judge_mod.judge_all(w, index, client, ledger, out=typer.echo)
    w.log_event("judge_runs.jsonl", {"model": model, "judged": rep.judged, "cost": rep.cost})
    typer.echo(f"judged {rep.judged} pairs ({rep.unparsable} unparsable, {rep.gone} chunk gone, "
               f"{rep.voided} no-answer question(s) voided); {rep.remaining} still without a grade; "
               f"cost ${ledger.total:.2f} of ${budget_usd:.2f}")  # fmt: skip
    if rep.stopped == "budget" or rep.unparsable:
        raise typer.Exit(1)


@app.command("kappa-sample")
def kappa_sample_cmd(
    n: Annotated[int, typer.Option(help="Pairs in the blind sample.")] = kappa_mod.SAMPLE_N,
    seed: Annotated[int, typer.Option(help="Seed of the sample.")] = kappa_mod.SAMPLE_SEED,
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
) -> None:
    """Grade 50 of Claude's judgments blind; print Cohen's kappa (linear weights) with n."""
    _, index, w = _state(stores_dir, work)
    if not w.judgments("claude", "pool"):
        raise typer.BadParameter("Claude has judged no pooled pair yet: run `bench gold judge`")
    session.kappa_sitting(w, index, n=n, seed=seed)
    result, left = w.kappa_result()
    if left:
        typer.echo(f"{left} sampled pair(s) still to grade: run `bench gold kappa-sample` again.")
        return
    assert result is not None
    typer.echo(result.text())
    if result.stands:
        typer.echo(f"kappa >= {kappa_mod.KAPPA_MIN}: Claude's labels stand. Your {result.n} "
                   f"grades are recorded as human and win on those pairs.")  # fmt: skip
    else:
        rest = sum(1 for p in w.pool() if w.label_state().get((p["qid"], p["gold_id"]),
                                                              (0, ""))[1] != "human")  # fmt: skip
        typer.echo(f"kappa < {kappa_mod.KAPPA_MIN} (or undefined): Claude's labels DO NOT stand. "
                   f"`bench gold label` now asks you for the other {rest} pooled pair(s) "
                   f"(your {result.n} count as done).")  # fmt: skip


@app.command("label")
def label_cmd(
    minutes: Annotated[float, typer.Option(help="Stop after this long (one sitting).")] = 45,
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
) -> None:
    """Only after kappa < 0.7: judge the pooled (question, chunk) pairs 0 / 1 / 2 yourself."""
    _, index, w = _state(stores_dir, work)
    result, left = w.kappa_result()
    if result is None or left:
        typer.echo("run `bench gold kappa-sample` first: your own labelling is only needed when "
                   "Claude's blind kappa is below 0.7")  # fmt: skip
        raise typer.Exit(1)
    if result.stands:
        typer.echo(f"kappa {result.kappa:.3f} (n = {result.n}) >= {kappa_mod.KAPPA_MIN}: Claude's "
                   f"labels stand, nothing for you to label.")  # fmt: skip
        return
    session.label_sitting(w, index, minutes=minutes, relabel_claude=True)


@app.command("status")
def status_cmd(work: WorkOpt = DEFAULT_WORK) -> None:
    """Counts per category, shortfalls, pooled pairs judged and your real pace."""
    typer.echo(session.status_text(session.Work(work)))


@app.command("freeze")
def freeze_cmd(
    snapshot: Annotated[Path, typer.Option(help="The snapshot tarball PG-v1 was built on.")],
    snapshot_date: Annotated[str, typer.Option(help="YYYY-MM-DD, the t of every B-R3 query.")],
    out: Annotated[
        Path, typer.Option(help="Version directory (v1; a v1.1 is a new directory).")
    ] = FROZEN,
    version: Annotated[str, typer.Option()] = pg.VERSION,
    stores_dir: StoresOpt = DEFAULT_STORES,
    work: WorkOpt = DEFAULT_WORK,
    config: ConfigOpt = None,
) -> None:
    """Validate (exact quotas, gold resolves, overlap, reworded, judge column, no work paths)
    and write the hashed files."""
    from bench.datasets import vault as vault_ds
    from bench.p4_setup import routes_from_config

    _, index, w = _state(stores_dir, work)
    qs = w.accepted()
    result, left = w.kappa_result()
    kappa = result.kappa if result is not None and not left else None
    try:
        hashes = pg.freeze(qs, index, out, snapshot_sha256=vault_ds.sha256_file(snapshot),
                           snapshot_name=snapshot.name, snapshot_date=snapshot_date,
                           version=version, routes=routes_from_config(config), kappa=kappa,
                           kappa_n=result.n if result is not None else 0)  # fmt: skip
    except (ValueError, pg.Frozen) as e:
        typer.echo(f"NOT FROZEN: {e}")
        raise typer.Exit(1) from e
    typer.echo(f"froze {len(qs)} questions into {out}")
    for k, v in hashes.items():
        typer.echo(f"  {k} sha256 {v}")


@app.command("verify")
def verify_cmd(gold_dir: Annotated[Path, typer.Option()] = FROZEN,
               stores_dir: StoresOpt = DEFAULT_STORES, config: ConfigOpt = None) -> None:  # fmt: skip
    """Hashes, split rule, judge column, no work-routed path and the stale-gold check against an
    import (run after every import)."""
    from bench.p4_setup import routes_from_config

    gold = pg.load(gold_dir)
    index = ChunkIndex.from_conns(open_conns(stores_dir))
    stale = pg.stale_gold(gold, index)
    work_paths = pg.work_routed(gold.questions, index, routes_from_config(config))
    by_judge = Counter(gold.judges.values())
    typer.echo(
        f"{len(gold.questions)} questions, hashes ok, split rule ok; judge column ok "
        f"(claude {by_judge['claude']}, human {by_judge['human']}; kappa "
        f"{gold.snapshot.get('judge_kappa')} n={gold.snapshot.get('judge_kappa_n')}); "
        f"work-routed paths: {len(work_paths)}; stale gold: {len(stale)}"
    )
    for line in [*work_paths, *stale][:20]:
        typer.echo(f"  {line}")
    if stale or work_paths:
        raise typer.Exit(1)
