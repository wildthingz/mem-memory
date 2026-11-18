"""P6 commands of `bench` (benchmarks-v2 §4 P6 row); registered on the main app by `bench.cli`.

daily-review   the 2-minute evening review: up to 10 of the day's recalls, marked by you
incident       log a crash / data-loss / wrong-store incident (B-E4 blocking incidents)
dup-scan       the nightly duplicate scan (pairs of current memories that look like one)
e4-report      B-E4 metrics over the last 10 working days, the gate, one PASS/FAIL line per row
pgv2-sample    PG-v2 step 1: 60 real recall events -> sheet.jsonl + labels.tsv (you fill `rel`)
pgv2-freeze    PG-v2 step 2: the filled labels.tsv -> a frozen bench/gold/personal/v2
pgv2-run       B-R3 on PG-v2 (recorded, not gated), on a copy of the live stores
weekly         the weekly T1 suite: pytest + every phase's gate on tune, from bench/configs/weekly.yaml
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from bench import gates, p6_use, pgv2, records
from bench.gold import pg
from bench.suites import b_p6
from bench.suites.base import BackendUnavailable

DATA = Path("data/p6")
StoresOpt = Annotated[
    Path,
    typer.Option(
        help="Directory with personal.db and work.db (~/.local/share/mem on the serving host)."
    ),
]
ReviewsOpt = Annotated[Path, typer.Option(help="Review marks (JSONL, append-only).")]
TzOpt = Annotated[str, typer.Option(help="IANA zone whose midnight ends a day.")]
GOLD_V2 = b_p6.GOLD_V2
WEEKLY_PLAN = Path(__file__).resolve().parent / "configs" / "weekly.yaml"


def default_stores() -> Path:
    return Path.home() / ".local" / "share" / "mem"


def _today(tz: str) -> str:
    return p6_use.day_of(datetime.now(UTC).isoformat(), tz)


def _conns(stores: Path) -> dict[str, sqlite3.Connection]:
    return {
        n: sqlite3.connect(f"file:{stores / (n + '.db')}?mode=ro", uri=True)
        for n in p6_use.STORES
        if (stores / f"{n}.db").exists()
    }


e4_result = b_p6.e4_result


_DEFAULT_STORES = default_stores()


def register(app: typer.Typer) -> None:
    @app.command("daily-review")
    def daily_review(
        stores: StoresOpt = _DEFAULT_STORES,
        reviews: ReviewsOpt = DATA / "reviews.jsonl",
        day: Annotated[str | None, typer.Option(help="YYYY-MM-DD, default today.")] = None,
        n: Annotated[int, typer.Option(help="Spec: up to 10.")] = p6_use.SAMPLE_PER_DAY,
        tz: TzOpt = "UTC",
        show_only: Annotated[bool, typer.Option("--show-only", help="Print, ask nothing.")] = False,
        redo: Annotated[bool, typer.Option(help="Ask again for already marked events.")] = False,
    ) -> None:
        """Show the day's sampled recalls and mark each: u useful, i irrelevant, s should-have-found
        (then the id it should have returned, or `not stored`), Enter to skip it for now."""
        day = day or _today(tz)
        events = p6_use.load_events(stores)
        sample = p6_use.review_sample(events, day, n, tz)
        if not sample:
            typer.echo(f"no recall events on {day} in {stores}")
            raise typer.Exit(0)
        done = p6_use.load_reviews(reviews)
        conns = _conns(stores)
        todo = 0
        for k, e in enumerate(sample, 1):
            old = done.get(e.ref)
            typer.echo(f"\n[{k}/{len(sample)}] {e.ref} · {e.at} · {e.host} · {e.store}")
            for c in p6_use.context_of(events, e):
                typer.echo(f"   before: {c}")
            typer.echo(f"   QUERY: {e.args.get('query')}")
            if e.result.get("abstained") or not e.ids:
                typer.echo("   returned: nothing")
            for i in e.ids:
                typer.echo(
                    f"   returned: {pgv2.describe(conns[e.store], pgv2.normalise_id(conns[e.store], i))}"
                )
            if old:
                typer.echo(f"   marked: {old.mark}{' ' + old.found if old.found else ''}")
            if show_only or (old and not redo):
                continue
            choice = (
                typer.prompt("   u / i / s / Enter=skip", default="", show_default=False)
                .strip()
                .lower()
            )
            if not choice:
                todo += 1
                continue
            mark = {"u": "useful", "i": "irrelevant", "s": "should-have-found"}.get(choice[:1])
            if mark is None:
                typer.echo("   not u/i/s: skipped")
                todo += 1
                continue
            found = None
            if mark == "should-have-found":
                found = typer.prompt(
                    "   id it should have found (e.g. m12) or 'not stored'"
                ).strip()
            p6_use.append_review(
                reviews,
                p6_use.Review(e.ref, mark, found, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )
        typer.echo(f"\n{len(sample)} sampled for {day}; {todo} left unmarked")

    @app.command("incident")
    def incident(
        kind: Annotated[str, typer.Argument(help="crash | data-loss | wrong-store")],
        note: Annotated[str, typer.Argument(help="What happened, in a line.")],
        store: Annotated[str, typer.Option(help="personal | work")] = "",
        incidents: Annotated[Path, typer.Option()] = DATA / "incidents.jsonl",
        day: Annotated[str | None, typer.Option(help="Default today.")] = None,
        tz: TzOpt = "UTC",
    ) -> None:
        """Log a blocking incident. data-loss and wrong-store fail the gate; crashes are reported."""
        try:
            p6_use.append_incident(
                incidents, date=day or _today(tz), kind=kind, note=note, store=store
            )
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e
        typer.echo(f"logged {kind} in {incidents}")

    @app.command("dup-scan")
    def dup_scan(
        stores: StoresOpt = _DEFAULT_STORES,
        scans: Annotated[Path, typer.Option()] = DATA / "dupscan.jsonl",
        port: Annotated[
            int, typer.Option(help="The running `mem serve` (D5: it holds the models).")
        ] = 8765,
        config: Annotated[Path | None, typer.Option(help="config.yaml of that serve.")] = None,
        theta: Annotated[
            float | None, typer.Option(help="θ_conf; default: the store's gate.")
        ] = None,
        day: Annotated[str | None, typer.Option()] = None,
        tz: TzOpt = "UTC",
    ) -> None:
        """Nightly: current memories that share an `about` subject, whose reranker logit is >= θ_conf,
        with no duplicate_of / supersedes path between them. Appends one line per store to --scans.
        Read-only on the stores; the reranker is the one of the running `mem serve`."""
        from mem import config as mem_config
        from mem import gate as gate_mod
        from mem import models_http

        cfg = mem_config.load(config) if config else mem_config.default_config()
        try:
            reranker = models_http.connect(port, cfg)[1]
        except models_http.ServeUnavailable as e:
            typer.echo(f"UNAVAILABLE {e}")
            raise typer.Exit(2) from e
        for name, conn in _conns(stores).items():
            g = gate_mod.read_gate(conn)
            th = theta if theta is not None else (g.theta_conf if g else None)
            if th is None:
                typer.echo(f"{name}: no θ_conf in the store's gate: pass --theta")
                raise typer.Exit(2)
            pairs = p6_use.duplicate_pairs(conn, reranker.score, th)
            (n_cur,) = conn.execute(
                "SELECT count(*) FROM nodes WHERE kind='memory' AND "
                "status='active' AND valid_to IS NULL"
            ).fetchone()
            p6_use.append_scan(
                scans, date=day or _today(tz), store=name, pairs=pairs, theta=th, n_current=n_cur
            )
            typer.echo(
                f"{name}: {len(pairs)} pair(s) at θ_conf {th:g} among {n_cur} current memories"
            )
            for a, b, z in pairs:
                typer.echo(
                    f"  {a} ~ {b}  logit {z:g}: link one `duplicate_of` the other, or `forget` one"
                )

    @app.command("e4-report")
    def e4_report(
        stores: StoresOpt = _DEFAULT_STORES,
        reviews: ReviewsOpt = DATA / "reviews.jsonl",
        scans: Annotated[Path, typer.Option()] = DATA / "dupscan.jsonl",
        incidents: Annotated[Path, typer.Option()] = DATA / "incidents.jsonl",
        since: Annotated[
            str | None, typer.Option(help="First day; default: the last 10 working days.")
        ] = None,
        until: Annotated[str | None, typer.Option()] = None,
        tz: TzOpt = "UTC",
        work_host: Annotated[
            list[str] | None,
            typer.Option(
                help="Host label prefix of the work machine (repeatable; default `work`)."
            ),
        ] = None,
        work_overhead_ms: Annotated[
            float | None,
            typer.Option(
                help="p95 the network adds over local "
                "for the work machine (`added_p95_ms` of its B-E5 record)."
            ),
        ] = None,
        out: Annotated[
            Path | None, typer.Option(help="Write the report (markdown + JSON beside).")
        ] = None,
    ) -> None:
        """B-E4: metrics and the gate, one line per row. Exit 1 unless every row passes;
        exit 2 when the stores hold no live events (there is nothing to measure)."""
        try:
            rep, start = b_p6.e4_report(
                b_p6.P6Setup(
                    stores,
                    reviews,
                    scans,
                    incidents,
                    since,
                    until,
                    tz,
                    work_host or ["work"],
                    work_overhead_ms,
                )
            )
        except (BackendUnavailable, OSError, ValueError) as e:
            typer.echo(f"UNAVAILABLE e4-report: {e}")
            raise typer.Exit(2) from e
        typer.echo(p6_use.render_report(rep))
        res = e4_result(rep)
        specs = [s for s in gates.load_gates()["P6"] if s.id == "B-E4"]
        failed = 0
        for s in specs:
            ok, detail = gates.evaluate(s, res)
            failed += not ok
            typer.echo(f"{'PASS' if ok else 'FAIL':<5} {detail}")
        typer.echo(
            f"{len(specs) - failed}/{len(specs)} B-E4 rows pass (window {start} .. {until or 'now'})"
        )
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(p6_use.render_report(rep), encoding="utf-8")
            out.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "window": [start, until],
                        "metrics": rep.metrics,
                        "unresolved": rep.unresolved,
                        "git_sha": records.git_sha(),
                        "at": datetime.now(UTC).isoformat(),
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        if failed:
            raise typer.Exit(1)

    @app.command("pgv2-sample")
    def pgv2_sample(
        stores: StoresOpt = _DEFAULT_STORES,
        reviews: ReviewsOpt = DATA / "reviews.jsonl",
        out: Annotated[Path, typer.Option(help="Sheet directory.")] = DATA / "pgv2",
        n: Annotated[int, typer.Option(help="Spec: 60.")] = pgv2.N_QUESTIONS,
        pool: Annotated[
            bool,
            typer.Option(
                "--pool/--no-pool",
                help="Add the top-`depth` of ours, "
                "BM25 and dense per query (starts the bench's own mem serve).",
            ),
        ] = False,
        depth: Annotated[int, typer.Option()] = pgv2.POOL_DEPTH,
        config: Annotated[Path | None, typer.Option()] = None,
        gate_file: Annotated[Path | None, typer.Option("--gate")] = None,
        embedder: Annotated[
            str, typer.Option(help="st = pinned models; hash = test stand-ins.")
        ] = "st",
        device: Annotated[str | None, typer.Option()] = None,
        serve_port: Annotated[int | None, typer.Option()] = None,
    ) -> None:
        """60 real recall events -> `sheet.jsonl` and `labels.tsv`. Fill the `rel` column of
        labels.tsv (0 not relevant, 1 supports, 2 answers; all 0 = nothing stored) in ~1 h.
        Contains your queries and memory text: private, never copy it off the host."""
        events = p6_use.load_events(stores)
        revs = p6_use.load_reviews(reviews)
        sample = pgv2.sample_events(events, revs, n)
        if len(sample) < n:
            typer.echo(f"only {len(sample)} distinct live recall queries (need {n}): use it longer")
            raise typer.Exit(2)
        conns = _conns(stores)
        backend, pool_fn = None, None
        if pool:
            backend = LiveBackend.copy_of(stores, config, gate_file, embedder, device, serve_port)
            pool_fn = backend.systems
        try:
            qs, lines = pgv2.make_sheet(sample, revs, conns, pool=pool_fn, depth=depth)
        finally:
            if backend:
                backend.close()
        pgv2.write_sheet(out, qs, lines)
        typer.echo(
            f"{len(qs)} questions, {len(lines)} lines to label: fill `rel` in {out / 'labels.tsv'}"
        )

    @app.command("pgv2-freeze")
    def pgv2_freeze(
        sheet: Annotated[Path, typer.Option(help="Directory of pgv2-sample.")] = DATA / "pgv2",
        out: Annotated[Path, typer.Option(help="Frozen directory.")] = GOLD_V2,
    ) -> None:
        """Check the labels and freeze PG-v2 (questions.yaml, qrels.tsv, SNAPSHOT). Never overwrites."""
        qs = [
            json.loads(ln)
            for ln in (sheet / "sheet.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        window = {"first": min(q["at"] for q in qs), "last": max(q["at"] for q in qs)}
        try:
            hashes = pgv2.freeze(
                sheet, out, built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), window=window
            )
        except (ValueError, pg.Frozen) as e:
            typer.echo(str(e))
            raise typer.Exit(2) from e
        typer.echo(f"frozen {out}: " + ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()))
        typer.echo(
            "Private: this directory holds your queries. It is git-ignored; do not commit it."
        )

    @app.command("pgv2-run")
    def pgv2_run(
        stores: StoresOpt = _DEFAULT_STORES,
        gold_dir: Annotated[Path, typer.Option()] = GOLD_V2,
        config: Annotated[Path | None, typer.Option()] = None,
        gate_file: Annotated[Path | None, typer.Option("--gate")] = None,
        embedder: Annotated[str, typer.Option()] = "st",
        device: Annotated[str | None, typer.Option()] = None,
        serve_port: Annotated[int | None, typer.Option()] = None,
        results_dir: Annotated[Path, typer.Option()] = records.RESULTS_DIR,
    ) -> None:
        """B-R3 on PG-v2: recorded, not gated (benchmarks-v2 §4 P6). Runs on a COPY of the live
        stores, so the 60 queries write no events into them; the copy is taken now, which is the
        state the labels are judged against (a gold id since forgotten is reported stale)."""
        from bench import pg_eval

        questions = pgv2.load(gold_dir)
        snap = pg.read_snapshot(gold_dir / "SNAPSHOT")
        backend = LiveBackend.copy_of(stores, config, gate_file, embedder, device, serve_port)
        try:
            stale = backend.stale_gold(questions)
            if stale:
                typer.echo(
                    f"FAIL {len(stale)} gold id(s) no longer in the stores (first: "
                    f"{'; '.join(stale[:4])}): a memory was forgotten or an import changed it"
                )
                raise typer.Exit(1)
            t0 = time.perf_counter()
            items = pg_eval.run_pg(backend, questions)  # type: ignore[arg-type]
            sc = pg_eval.score_pg(items)
            wall = time.perf_counter() - t0
        finally:
            backend.close()
        started = datetime.now(UTC)
        rec = records.RunRecord(
            run_id=records.new_run_id("B-R3-PGv2", started),
            bench_id="B-R3-PGv2",
            tier="t1",
            mode="live-copy",
            dataset={"name": "PG-v2", "version": snap["version"], **snap},
            config=records.config_fingerprint(backend.cfg),
            stubs={},
            git_sha=records.git_sha(),
            hardware=records.hardware(),
            metrics=sc.metrics,
            gates=[],
            violations=[],
            n_items=len(items),
            wall_s=wall,
            started_at=started.isoformat(),
            notes=sc.notes + ["recorded, not gated: PG-v2 is a gate only from the next version"],
            extra={
                "ci95": {k: list(v) for k, v in sc.ci95.items()},
                "n": sc.n,
                "per_category": sc.per_category,
            },
        )
        path = records.write_run(rec, [i.record() for i in items], results_dir)
        floors = {
            "coverage_any": (">=", 0.85),
            "far": ("<=", 0.10),
            "delivered_precision": (">=", 0.70),
            "ira": ("<=", 0.5),
            "abstention_recall": (">=", 0.70),
            "ndcg_std10_R": (">=", 0.70),
        }
        typer.echo(f"B-R3 on PG-v2: n={sc.n} (the B-R3 floors, for reading only):")
        for k, (op, th) in floors.items():
            lo, hi = sc.ci95.get(k, (float("nan"),) * 2)
            v = sc.metrics.get(k, float("nan"))
            ok = v >= th if op == ">=" else v <= th
            typer.echo(
                f"  {k:<20} {v:.3f}  CI [{lo:.3f}, {hi:.3f}]  floor {op} {th}  "
                f"{'meets' if ok else 'below'}"
            )
        typer.echo(f"record: {path}")

    @app.command("weekly")
    def weekly(
        plan: Annotated[Path, typer.Option(help="Steps and variables.")] = WEEKLY_PLAN,
        var: Annotated[
            list[str] | None, typer.Option(help="name=value, overrides the plan's vars.")
        ] = None,
        results_dir: Annotated[Path, typer.Option()] = records.RESULTS_DIR,
        only: Annotated[list[str] | None, typer.Option(help="Run only these step names.")] = None,
    ) -> None:
        """The weekly T1 suite: every step in order, none stopping the rest. One line per step
        (PASS / FAIL / UNAVAILABLE), a summary file in the results dir, a note of any step that
        passed last week and does not now. Exit 1 unless every step passes."""
        spec = yaml.safe_load(plan.read_text(encoding="utf-8"))
        vals = {k: str(v) for k, v in (spec.get("vars") or {}).items()}
        for item in var or []:
            k, _, v = item.partition("=")
            vals[k] = v
        outcomes: list[dict[str, Any]] = []
        for step in spec["steps"]:
            if only and step["name"] not in only:
                continue
            argv, missing = _fill(step["argv"], vals)
            if missing:
                outcomes.append(
                    {
                        "step": step["name"],
                        "status": "UNAVAILABLE",
                        "detail": f"unset: {', '.join(missing)} (pass --var name=value)",
                    }
                )
                typer.echo(f"UNAVAILABLE {step['name']}: unset {', '.join(missing)}")
                continue
            t0 = time.perf_counter()
            typer.echo(f"--- {step['name']}: {' '.join(argv)}")
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
            status = "PASS" if proc.returncode == 0 else "FAIL"
            outcomes.append(
                {
                    "step": step["name"],
                    "status": status,
                    "returncode": proc.returncode,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "tail": tail,
                }
            )
            typer.echo("\n".join(f"    {ln}" for ln in tail))
            typer.echo(f"{status} {step['name']}")
        prev = _last_weekly(results_dir)
        regress = [
            o["step"] for o in outcomes if o["status"] != "PASS" and prev.get(o["step"]) == "PASS"
        ]
        at = datetime.now(UTC)
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"weekly-{at:%Y%m%dT%H%M%SZ}.json"
        path.write_text(
            json.dumps(
                {
                    "at": at.isoformat(),
                    "git_sha": records.git_sha(),
                    "plan": str(plan),
                    "outcomes": outcomes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        bad = [o for o in outcomes if o["status"] != "PASS"]
        if regress:
            typer.echo(f"REGRESSION since the last weekly run: {', '.join(regress)}")
        typer.echo(f"{len(outcomes) - len(bad)}/{len(outcomes)} weekly steps pass; summary {path}")
        if bad:
            raise typer.Exit(1)


def _fill(argv: list[str], vals: dict[str, str]) -> tuple[list[str], list[str]]:
    """Substitute `{name}`. An empty value drops the flag before it (`--config ''` means no flag)
    only for the vars the plan marks optional by leaving them empty AND the flag is a `--option`;
    a path-like required var that is empty is reported missing."""
    out, missing = [], []
    for tok in argv:
        if tok.startswith("{") and tok.endswith("}"):
            name = tok[1:-1]
            v = vals.get(name, "")
            if v == "":
                if name in OPTIONAL_VARS and out and out[-1].startswith("--"):
                    out.pop()
                else:
                    missing.append(name)
                continue
            out.append(v)
        else:
            out.append(tok)
    return out, missing


OPTIONAL_VARS = frozenset({"config"})


def _last_weekly(results_dir: Path) -> dict[str, str]:
    files = sorted(Path(results_dir).glob("weekly-*.json")) if Path(results_dir).exists() else []
    if not files:
        return {}
    return {o["step"]: o["status"] for o in json.loads(files[-1].read_text())["outcomes"]}


class LiveBackend:
    """B-R3 over a copy of the live stores. Thin wrapper on `OursVaultBackend` whose chunk-id ->
    gold-id map also names memory chunks `m<node id>` (the ids the events log and PG-v2 holds)."""

    @classmethod
    def copy_of(
        cls,
        stores: Path,
        config: Path | None,
        gate_file: Path | None,
        embedder: str,
        device: str | None,
        serve_port: int | None,
    ) -> Any:
        from bench.adapters.lme_ours import read_gate_file
        from bench.adapters.p4_ours import OursVaultBackend
        from mem import gate as gate_mod

        root = Path(tempfile.mkdtemp(prefix="pgv2-stores-"))
        for name in p6_use.STORES:
            src = stores / f"{name}.db"
            if src.exists():
                _backup(src, root / f"{name}.db")
        names = tuple(n for n in p6_use.STORES if (root / f"{n}.db").exists())

        class _B(OursVaultBackend):
            def _gid(self, store: str, cid: int) -> str:
                c = self.index.by_id.get((store, cid))
                if c:
                    return c.gold_id
                row = (
                    self.store(store)
                    .conn.execute(
                        "SELECT n.id FROM chunks c JOIN nodes n ON n.id = c.node_id WHERE c.id = ?"
                        " AND n.kind = 'memory'",
                        (cid,),
                    )
                    .fetchone()
                )
                return f"m{row[0]}" if row else f"c{cid}"

            def stale_gold(self, questions: list[pg.Question]) -> list[str]:
                bad = []
                for q in questions:
                    conn = self.store(q.store).conn
                    for g in sorted(q.gold):
                        ok = (
                            conn.execute(
                                "SELECT 1 FROM nodes WHERE id = ? AND status = 'active'",
                                (int(g[1:]),),
                            ).fetchone()
                            is not None
                            if g.startswith("m") and g[1:].isdigit()
                            else g in self.index.by_gold
                        )
                        if not ok:
                            bad.append(f"{q.qid} {g}")
                return bad

            def close(self) -> None:
                super().close()
                shutil.rmtree(root, ignore_errors=True)

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        b = _B(
            root,
            today,
            config_path=config,
            models=embedder,
            device=device,
            serve_port=serve_port,
            stores=names,
        )
        if gate_file:
            b.gate = read_gate_file(gate_file, b.cfg)
        else:
            conn = sqlite3.connect(f"file:{root / (names[0] + '.db')}?mode=ro", uri=True)
            b.gate = gate_mod.read_gate(conn)
            conn.close()
        return b


def _backup(src: Path, dst: Path) -> None:
    """A consistent copy of a live (WAL) database without locking out the server."""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
