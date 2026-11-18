"""`bench` command line (benchmarks-v2 §6): P1 T0 gates, P2 LongMemEval data, freeze, gate P2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated

import typer

from bench import freeze as freeze_mod
from bench import (
    gates,
    list_fill_dryrun,
    lme_eval,
    p4_cli,
    p5_cli,
    p6_cli,
    prf_check,
    records,
    temporal_dryrun,
    window_dryrun,
)
from bench import p4_setup as p4_setup_mod
from bench.adapters import ours
from bench.datasets import corpora, fetch, lme, negatives
from bench.datasets import locomo as locomo_data
from bench.gold import cli as gold_cli
from bench.suites.base import BackendUnavailable

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(gold_cli.app, name="gold")
p4_cli.register(app)
p5_cli.register(app)
p6_cli.register(app)
temporal_dryrun.register(app)
list_fill_dryrun.register(app)
window_dryrun.register(app)
prf_check.register(app)


@app.callback()
def main() -> None:
    """bench: benchmark harness for mem."""


DATASETS = Path(__file__).resolve().parent / "datasets"
MANIFEST = DATASETS / "lme_split.tsv"
NEGATIVES = DATASETS / "lme_negatives.tsv"
AUDIT = DATASETS / "lme_negatives_audit.tsv"
# The spec's counts (benchmarks-v2 §1), reproduced by `bench build-lme` or it refuses to write.
EXPECTED_COUNTS = {"total": 500, "tune": 151, "test": 349, "br1_tune": 115, "br1_test": 269,
                   "c_ans_tune": 96, "c_ans_test": 224, "abs_tune": 10, "abs_test": 20}  # fmt: skip


EmbedderOpt = Annotated[
    str, typer.Option(help="st = the pinned models; hash = the deterministic test stand-ins.")
]
DeviceOpt = Annotated[
    str | None,
    typer.Option(help="The bench's own mem serve: auto, cpu or cuda. Default: config `device`."),
]
ServePortOpt = Annotated[
    int | None,
    typer.Option(help="Port of the bench's own pinned mem serve (D5; refused if in use). "
                 "Default: a free port."),
]  # fmt: skip


def _lme_options(
    lme_file: Path, split: str, tune_run: Path | None, latency_runs: list[Path], tokens: str,
    cfg, gate_file: Path | None, embedder: str, device: str | None, config: Path | None = None,
    serve_port: int | None = None, loaded=None,
) -> dict:  # fmt: skip
    from bench.adapters.lme_ours import OursLmeBackend
    from bench.metrics import tokens as tk
    from bench.suites.lme_setup import LmeSetup, load_tune_record

    manifest = lme.read_manifest(MANIFEST)
    questions = lme.load_questions(lme_file)
    lme.verify_manifest(questions, MANIFEST)
    exact = tokens == "anthropic"
    counter = tk.anthropic_counter() if exact else lme_eval.est_counter
    n_test = len(lme.select(manifest, "test", "br1"))
    backend = OursLmeBackend(cfg=cfg, config_path=config, gate_path=gate_file, models=embedder,
                             device=device, serve_port=serve_port, loaded=loaded)  # fmt: skip
    gate_meta = gate_record(backend.gate, gate_file) if gate_file else None
    return {lme_setup_key(): LmeSetup(
        backend=backend, questions={q["question_id"]: q for q in questions},
        manifest=manifest, negatives=negatives.read_negatives(NEGATIVES), split=split,
        dataset_meta={"name": "LongMemEval_S cleaned", "file": lme_file.name,
                      "sha256": lme.sha256_file(lme_file),
                      "manifest_sha256": lme.sha256_file(MANIFEST), "version": 2,
                      "gate": gate_meta},
        token_counter=counter, token_counter_exact=exact,
        tune_record=load_tune_record(tune_run) if tune_run else None, n_target=n_test,
        latency_runs=latency_runs)}  # fmt: skip


def gate_record(gate, gate_file: Path) -> dict:
    """What the run record keeps about the gate (architecture-v2 §3.3 steps 4-5): the file, its
    fit id, whether the fit met the five P2 tune targets, the rule that chose the setting, the
    thresholds, the five tune numbers and, for an infeasible fit, the Pareto front on tune."""
    tune = gate.tune or {}
    return {"file": gate_file.name, "sha256": lme.sha256_file(gate_file), "fit_id": gate.fit_id,
            "feasible": tune.get("feasible"), "rule": tune.get("rule"),
            "tau_keep": gate.tau_keep, "tau_high": gate.tau_high, "gap": gate.gap,
            "tune": {k: tune.get(k) for k in ("precision", "ira", "coverage_any",
                                              "abstention_recall", "far")},
            **({"front": tune["front"]} if "front" in tune else {})}  # fmt: skip


def lme_setup_key() -> str:
    from bench.suites.lme_setup import OPT

    return OPT


def _p3_setup(
    split,
    syn_size,
    syn_seeds,
    w5_run,
    w5_audit,
    c3_run,
    config,
    gate_file,
    embedder,
    device,
    serve_port,
    e1_run=None,
    oracle_pool=None,
):
    """The P3 context: the backend (real stores over the bench's own pinned serve), the SYN setup
    for the split (dev seeds on tune, held-out seeds on test), B-W5's recorded run, B-C3's runs."""
    import tempfile

    from bench.adapters.p3_ours import P3Backend
    from bench.suites import b_c3, b_w5
    from bench.synth import data as syn_data
    from bench.synth.run import OPT as SYN_OPT
    from bench.synth.run import SynSetup

    backend = P3Backend(config_path=config, gate_path=gate_file, models=embedder, device=device,
                        serve_port=serve_port)  # fmt: skip
    if (backend.gate is not None and backend.read.get("p3_model", False)
            and backend.gate.read_model is None):  # fmt: skip
        backend.close()
        raise ValueError(
            "read.p3_model is on but this gate file has no read_model (rerank, cosine, gap;"
            " MEM2-155), and there is no fallback. Refit with `mem calibrate --objective p3` on"
            " records with cosines (deliverables/runbooks/p3-gate.md §2), or set read.p3_model:"
            " false (the default, ADR 0020)"
        )
    if (backend.gate is not None and backend.read.get("conflict_guard", False)
            and backend.gate.needs_read_guard()):  # fmt: skip
        backend.close()
        raise ValueError(
            "the P3 gate needs a read_guard (theta_read, ADR 0018) while read.conflict_guard is"
            " on: this gate file has none, and there is no default read cut. Fit it with `bench"
            " syn-fit-read-guard --gate <gate>` and add it with `bench gate-add-read-guard`"
            " (deliverables/runbooks/p3-gate.md §2b)"
        )
    seeds = (tuple(int(x) for x in syn_seeds.split(",")) if syn_seeds
             else syn_data.HELD_OUT_SEEDS if split == "test" else syn_data.DEV_SEEDS)  # fmt: skip
    if syn_size not in syn_data.SIZES:
        raise ValueError(f"--syn-size must be one of {sorted(syn_data.SIZES)}")
    tmp = tempfile.TemporaryDirectory(prefix="bench-syn-")
    options: dict = {SYN_OPT: SynSetup(seeds, syn_size, split, workdir=Path(tmp.name))}
    if w5_run:
        options[b_w5.OPT] = b_w5.W5Setup(
            w5_run, w5_audit, split, seed=101 if split == "test" else 1
        )
    if e1_run:
        from bench.suites import b_e1

        options[b_e1.OPT] = b_e1.E1Setup(e1_run, split, oracle_pool)
    runs: dict[str, list[Path]] = {}
    for item in c3_run or []:
        corpus, _, path = item.partition("=")
        if corpus not in ("L5k", "L50k") or not path:
            raise ValueError(f"--c3-run wants L5k=path or L50k=path, got {item!r}")
        runs.setdefault(corpus, []).append(Path(path))
    if runs:
        options[b_c3.OPT] = runs
    return backend, options, tmp


@app.command()
def gate(
    phase: Annotated[str, typer.Argument(help="Phase id, e.g. P1.")],
    results_dir: Annotated[Path, typer.Option(help="Where run records go.")] = records.RESULTS_DIR,
    ledger: Annotated[
        str | None, typer.Option(help="Log this run as a test look with this reason.")
    ] = None,
    who: Annotated[str, typer.Option(help="Who is looking (ledger).")] = "bench",
    only: Annotated[list[str] | None, typer.Option(help="Run only these gate ids.")] = None,
    split: Annotated[
        str, typer.Option(help="P2+: tune (informational) or test (a look).")
    ] = "tune",
    lme_file: Annotated[Path | None, typer.Option(help="longmemeval_s_cleaned.json")] = None,
    tune_run: Annotated[
        Path | None, typer.Option(help="B-R1 tune run record (power, FAR drift) for a test look.")
    ] = None,
    latency_run: Annotated[
        list[Path] | None, typer.Option(help="`bench latency-run` outputs (3, for B-C1).")
    ] = None,
    tokens: Annotated[str, typer.Option(help="anthropic (count_tokens) or est.")] = "anthropic",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    gate_file: Annotated[
        Path | None, typer.Option("--gate", help="P2+: gate JSON from `mem calibrate --out`.")
    ] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
    syn_size: Annotated[
        str, typer.Option(help="P3: SYN size (S, M, L); M is the gate size.")
    ] = "M",
    syn_seeds: Annotated[
        str | None, typer.Option(help="P3: comma list, default dev 1-5 (tune) / held-out 101-105.")
    ] = None,
    w5_run: Annotated[
        Path | None, typer.Option(help="P3: the `bench w5-run` file B-W5 scores.")
    ] = None,
    w5_audit: Annotated[Path | None, typer.Option(help="P3: the filled B-W5 audit sheet.")] = None,
    e1_run: Annotated[
        Path | None, typer.Option(help="P3: the `bench e1-run` file B-E1 scores.")
    ] = None,
    oracle_pool: Annotated[
        list[Path] | None,
        typer.Option(
            "--oracle-pool",
            help="P3 B-E1 validity: another `bench e1-run` file of the --e1-run's harness whose "
            "oracle answers pool with it (repeat). Default: the pool the run file recorded.",
        ),
    ] = None,
    c3_run: Annotated[
        list[str] | None, typer.Option(help="P3: `bench c3-run` outputs as L5k=path / L50k=path.")
    ] = None,
    vault_snapshot: Annotated[
        Path | None, typer.Option(help="P4: vault-YYYY-MM-DD.tar.zst the PG-v1 was built on.")
    ] = None,
    snapshot_sha256: Annotated[
        str | None, typer.Option(help="P4: the recorded sha256 of the snapshot.")
    ] = None,
    pg_snapshot: Annotated[
        Path | None, typer.Option(help="P4: pgvault-YYYY-MM-DD.tar.zst (no work notes): B-R3.")
    ] = None,
    pg_snapshot_sha256: Annotated[
        str | None, typer.Option(help="P4: the recorded sha256 of the PG snapshot.")
    ] = None,
    gold_dir: Annotated[
        Path, typer.Option(help="P4: frozen PG-v1 directory.")
    ] = p4_setup_mod.GOLD_DIR,
    mini_vault: Annotated[
        Path, typer.Option(help="P4: the mini-vault fixture (vault dir).")
    ] = p4_setup_mod.MINI_VAULT,
    mini_config: Annotated[
        Path, typer.Option(help="P4: the mini-vault's config.yaml (routes).")
    ] = p4_setup_mod.MINI_CONFIG,
    lifecycle: Annotated[
        Path, typer.Option(help="P4: B-I2 descriptor for the mini-vault.")
    ] = p4_setup_mod.LIFECYCLE,
    i5_sheet: Annotated[Path | None, typer.Option(help="P4: the labelled B-I5 sheet.")] = None,
    c4_run: Annotated[Path | None, typer.Option(help="P4: `bench c4-run` output (B-C4).")] = None,
    baseline: Annotated[
        list[str] | None, typer.Option(help="P4: qmd=path / rg=path (`bench qmd-run`, `rg-run`).")
    ] = None,
    v_latency_run: Annotated[
        list[str] | None,
        typer.Option(help="P4: `vault-latency-run` outputs as V5k=path / V50k=path."),
    ] = None,
    v_c3_run: Annotated[
        list[str] | None, typer.Option(help="P4: `vault-c3-run` outputs as V5k=path / V50k=path.")
    ] = None,
    e5_run: Annotated[
        list[Path] | None,
        typer.Option(help="P5: `bench e5-run` records, one per machine/transport."),
    ] = None,
    e5_reconnect: Annotated[
        list[Path] | None, typer.Option(help="P5: `bench e5-reconnect` records.")
    ] = None,
    c6_run: Annotated[Path | None, typer.Option(help="P5: `bench c6-run` record (B-C6).")] = None,
    stores_dir: Annotated[
        Path | None, typer.Option(help="P6: directory with personal.db and work.db.")
    ] = None,
    work_overhead_ms: Annotated[
        float | None, typer.Option(help="P6: `added_p95_ms` of the work machine's B-E5 record.")
    ] = None,
    home_only: Annotated[
        bool,
        typer.Option(
            "--home-only",
            help="P5 checkpoint on the home desktop only: rows that need the work "
            "machine's records are DEFERRED (not run), not FAIL.",
        ),
    ] = False,
) -> None:
    """Run a phase's gates; print PASS/FAIL per gate id; exit 1 unless all pass (with P5
    --home-only: unless every gate is PASS or DEFERRED)."""
    from bench.adapters.lme_ours import load_config

    options, label, cfg, e1_pool = None, "", None, None
    factory, p3_backend, syn_tmp, p4_run = ours.open_client, None, None, None
    if oracle_pool and not (phase == "P3" and e1_run):
        raise typer.BadParameter("--oracle-pool belongs to P3 with --e1-run")
    if phase == "P5":  # recorded human-run legs: no data split, no test look
        from bench.suites import b_p5

        options = {b_p5.OPT: b_p5.P5Setup(e5_run or [], e5_reconnect or [], c6_run, home_only)}
        label = " [P5 recorded runs" + (", home-only: cross-machine rows deferred]" if home_only
                                        else "]")  # fmt: skip
    elif phase == "P6":  # the live stores and the human-kept files: no data split, no test look
        from bench.suites import b_p6

        options = {
            b_p6.OPT: b_p6.P6Setup(
                stores=stores_dir or b_p6.P6Setup().stores, work_overhead_ms=work_overhead_ms
            )
        }
        label = " [P6 live stores]"
    elif phase != "P1":
        if split not in ("tune", "test"):
            raise typer.BadParameter("--split must be tune or test")
        if split == "test":
            if not ledger:
                typer.echo('a test look needs --ledger "<why>" (benchmarks-v2 §5.4)')
                raise typer.Exit(2)
            try:
                freeze_mod.check_test_look_allowed(phase)
            except freeze_mod.Refused as e:
                typer.echo(f"REFUSED {e}")
                raise typer.Exit(2) from e
        label = f" [{phase} {split}{'' if split == 'test' else ': informational, not a look'}]"

        if phase == "P3" and e1_run:
            e1_pool = _e1_gate_pool(e1_run, split, oracle_pool)
        if phase == "P3":
            try:
                p3_backend, options, syn_tmp = _p3_setup(
                    split, syn_size, syn_seeds, w5_run, w5_audit, c3_run, config, gate_file,
                    embedder, device, serve_port, e1_run, oracle_pool)  # fmt: skip
            except (RuntimeError, ValueError, OSError) as e:
                typer.echo(f"UNAVAILABLE setup: {e}")
                raise typer.Exit(2) from e
            factory, cfg = p3_backend.factory, p3_backend.cfg
        if phase == "P4":
            try:
                p4_run = _p4_run(
                    vault_snapshot, snapshot_sha256, pg_snapshot, pg_snapshot_sha256, gold_dir, split,
                    mini_vault, mini_config,
                    lifecycle, config, gate_file, embedder, device, serve_port, i5_sheet, c4_run,
                    baseline, v_latency_run, v_c3_run)  # fmt: skip
            except (RuntimeError, ValueError, OSError, BackendUnavailable) as e:
                typer.echo(f"UNAVAILABLE setup: {e}")
                raise typer.Exit(2) from e
            options = {p4_setup_mod.b_p4.OPT: p4_run.setup}
            cfg = p4_run.cfg
        if lme_file is not None:
            try:
                cfg = load_config(config, embedder)
                options = (options or {}) | _lme_options(
                    lme_file, split, tune_run, latency_run or [], tokens, cfg, gate_file, embedder,
                    device, config, serve_port, loaded=p4_run.models if p4_run else None)  # fmt: skip
            except (RuntimeError, ValueError, OSError) as e:
                typer.echo(f"UNAVAILABLE setup: {e}")
                raise typer.Exit(2) from e
            except Exception as e:
                typer.echo(f"UNAVAILABLE setup: {type(e).__name__}: {e}")
                raise typer.Exit(2) from e
    try:
        outcomes = gates.run_phase(phase, factory, results_dir=results_dir,
                                   only=set(only) if only else None, options=options,
                                   split=split if phase not in ("P1", "P5", "P6") else None, cfg=cfg)  # fmt: skip
    finally:  # stops the bench's own mem serve, if a backend started one
        for setup in (options or {}).values():
            getattr(getattr(setup, "backend", None), "close", lambda: None)()
        if p3_backend is not None:
            p3_backend.close()
        if p4_run is not None:
            p4_run.close()
        if syn_tmp is not None:
            syn_tmp.cleanup()
    for o in outcomes:
        typer.echo(f"{o.status:<11} {o.id:<5} {o.detail}")
    if e1_pool is not None and any(o.id == "B-E1" for o in outcomes):
        _echo_e1_validity(e1_pool)
    if phase not in ("P1", "P5", "P6") and split == "tune":
        for o in outcomes:
            if o.run_file is not None:
                typer.echo(per_category_table(o.id, o.run_file), nl=False)
    if ledger or (phase not in ("P1", "P5", "P6") and split == "test"):
        fp = records.config_fingerprint(cfg)
        records.append_ledger({"who": who, "why": ledger or "", "phase": phase,
                               "split": "fixture" if phase in ("P1", "P5", "P6") else split,
                               "config_hash": fp["fp_all"],
                               "outcomes": {o.id: o.status for o in outcomes}})  # fmt: skip
    n_def = sum(o.deferred for o in outcomes)
    typer.echo(
        f"{sum(o.passed for o in outcomes)}/{len(outcomes)} gates pass"
        + (f", {n_def} deferred (not a pass)" if n_def else "")
        + f" for {phase}{label}"
    )
    if not all(o.passed or o.deferred for o in outcomes):
        raise typer.Exit(1)


def _e1_gate_pool(e1_run: Path, split: str, oracle_pool: list[Path] | None) -> dict:
    """B-E1's oracle pool for `gate P3`, checked before anything starts: a pooled file of another
    harness, or a run counted twice, is REFUSED (exit 2) with the cause named."""
    from bench import e1
    from bench.suites import b_e1

    try:
        rec, pool = b_e1.load(b_e1.E1Setup(e1_run, split, oracle_pool))
    except (e1.PoolRefused, OSError, ValueError, KeyError) as e:
        typer.echo(f"REFUSED --oracle-pool: {e}")
        raise typer.Exit(2) from e
    return e1.oracle_pool(rec, pool)


def _echo_e1_validity(p: dict) -> None:
    from bench import e1

    typer.echo(e1.pool_line(p))
    why = e1.pool_invalid_reason(p)
    typer.echo(f"INVALID: {why}" if why else "VALID")


def _pairs(items: list[str] | None, what: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for item in items or []:
        key, _, path = item.partition("=")
        if not path:
            raise ValueError(f"{what} wants NAME=path, got {item!r}")
        out.setdefault(key, []).append(Path(path))
    return out


def _p4_run(vault_snapshot, snapshot_sha256, pg_snapshot, pg_snapshot_sha256, gold_dir, split,
            mini_vault, mini_config, lifecycle,
            config, gate_file, embedder, device, serve_port, i5_sheet, c4_run, baseline,
            v_latency_run, v_c3_run):  # fmt: skip
    base = {k: v[0] for k, v in _pairs(baseline, "--baseline").items()}
    return p4_setup_mod.build(
        snapshot=vault_snapshot,
        snapshot_sha256=snapshot_sha256,
        pg_snapshot=pg_snapshot,
        pg_snapshot_sha256=pg_snapshot_sha256,
        gold_dir=gold_dir,
        split=split,
        mini_vault=mini_vault if mini_vault.exists() else None,
        mini_config=mini_config,
        lifecycle=lifecycle,
        config_path=config,
        gate_path=gate_file,
        embedder=embedder,
        device=device,
        serve_port=serve_port,
        i5_sheet=i5_sheet,
        c4_run=c4_run,
        baselines=base,
        latency_runs=_pairs(v_latency_run, "--v-latency-run"),
        c3_runs=_pairs(v_c3_run, "--v-c3-run"),
    )


PER_CATEGORY_COLS = ("n", "coverage_any", "delivered_precision", "ira", "far", "abstention_recall")


def per_category_table(bench_id: str, run_file: Path) -> str:
    """The tune run's per-category delivered metrics (benchmarks-v2 §2.0 restricted to each LME
    question type, `_abs` on its own) from a run record; "" when the record has none."""
    rec = json.loads(Path(run_file).read_text(encoding="utf-8"))
    pc = rec.get("extra", {}).get("per_category") or {}
    if not all(c in pc for c in PER_CATEGORY_COLS):
        return ""
    cats = sorted(pc["n"], key=lambda c: (c == "_abs", c))
    lines = [
        f"  {bench_id} per category ({rec['dataset'].get('split', '?')}):"
        + "".join(f"{c[:9]:>11}" for c in ("n", "cov_any", "precision", "ira", "far", "abst"))
    ]
    for cat in cats:
        cells = []
        for col in PER_CATEGORY_COLS:
            v = pc[col].get(cat)
            if v is None or math.isnan(v):
                cells.append(f"{'-':>11}")
            else:
                cells.append(f"{int(v):>11}" if col == "n" else f"{v:>11.3f}")
        lines.append(f"    {cat[:26]:<26}" + "".join(cells))
    return "\n".join(lines) + "\n"


@app.command("freeze")
def freeze_cmd(
    phase: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option(help="Why, based on tune data only.")],
    who: Annotated[str, typer.Option()] = "bench",
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="ID.metric=value, once (e.g. B-R1.ira=0.4).")
    ] = None,
) -> None:
    """Freeze the phase's test thresholds (unchanged, or changed once on tune data)."""
    over = {}
    for item in set_ or []:
        key, _, val = item.partition("=")
        bid, _, metric = key.partition(".")
        over[(bid, metric)] = float(val)
    try:
        changes = freeze_mod.freeze(phase, who=who, reason=reason,
                                    config_hash=records.config_fingerprint()["fp_all"],
                                    overrides=over)  # fmt: skip
    except freeze_mod.Refused as e:
        typer.echo(f"REFUSED {e}")
        raise typer.Exit(2) from e
    typer.echo(f"frozen {phase}" + "".join(f"\n  {c}" for c in changes))


@app.command()
def run(
    suite: Annotated[str, typer.Option(help="Bench id, e.g. B-R6.")],
    results_dir: Annotated[Path, typer.Option()] = records.RESULTS_DIR,
    phase_: Annotated[str | None, typer.Option("--phase", help="Phase form, e.g. P3.")] = None,
) -> None:
    """Run one T0 suite and write its run record."""
    for phase, specs in gates.load_gates().items():
        if phase_ and phase != phase_:
            continue
        for spec in specs:
            if spec.id == suite:
                outcomes = gates.run_phase(phase, ours.open_client, results_dir=results_dir,
                                           only={suite})  # fmt: skip
                typer.echo(json.dumps([{"id": o.id, "status": o.status, "detail": o.detail}
                                       for o in outcomes], indent=2))  # fmt: skip
                raise typer.Exit(0 if all(o.passed for o in outcomes) else 1)
    typer.echo(f"unknown suite {suite!r}")
    raise typer.Exit(2)


@app.command("fetch")
def fetch_cmd(
    dest: Annotated[Path, typer.Option(help="Where to put the files.")] = Path("data/lme"),
    which: Annotated[list[str] | None, typer.Option(help="s and/or m (default s).")] = None,
) -> None:
    """Download LongMemEval (HF xiaowu0162/longmemeval-cleaned) and pin sha256 in checksums.sha256."""
    for name, digest in fetch.fetch(which or ["s"], dest).items():
        typer.echo(f"{digest}  {name}")


@app.command("build-lme")
def build_lme(
    lme_file: Annotated[Path, typer.Argument(help="longmemeval_s_cleaned.json")],
    force: Annotated[bool, typer.Option(help="Overwrite a frozen manifest.")] = False,
) -> None:
    """Write the frozen manifest, negatives and the 30-pair audit sheet; check the spec's counts."""
    fetch.verify(lme_file.name, lme_file)
    qs = lme.load_questions(lme_file)
    rows = lme.build_manifest(qs)
    counts = lme.manifest_counts(rows)
    wrong = {k: (counts[k], v) for k, v in EXPECTED_COUNTS.items() if counts[k] != v}
    if wrong:
        typer.echo(f"counts differ from benchmarks-v2 §1 (got, want): {wrong}")
        raise typer.Exit(1)
    if MANIFEST.exists() and not force:
        lme.verify_manifest(qs, MANIFEST)
        typer.echo(f"{MANIFEST.name} already frozen and reproduced from this file")
    else:
        typer.echo(f"{MANIFEST.name} sha256 {lme.write_manifest(rows, MANIFEST)}")
    neg = negatives.build_negatives(qs, rows)
    negatives.write_negatives(neg, NEGATIVES)
    negatives.write_audit_sheet(negatives.audit_sample(neg), qs, AUDIT)
    typer.echo(f"counts {counts}\nwrote {NEGATIVES.name} ({len(neg)} pairs) and {AUDIT.name} (30)")


@app.command("audit-apply")
def audit_apply(
    sheet: Annotated[Path, typer.Argument(help="The filled lme_negatives_audit.tsv.")] = AUDIT,
) -> None:
    """Read the hand-filled audit sheet; drop any pair that still contains the answer; report."""
    neg, report = negatives.apply_audit(negatives.read_negatives(NEGATIVES), sheet)
    negatives.write_negatives(neg, NEGATIVES)
    typer.echo(json.dumps(report))


@app.command("build-corpus")
def build_corpus_cmd(
    lme_m: Annotated[Path, typer.Argument(help="longmemeval_m_cleaned.json")],
    size: Annotated[str, typer.Option(help="L5k or L50k")] = "L5k",
) -> None:
    """Plan a load corpus (first N chunks, sorted session ids) and print its fingerprint.

    Streams the M file in two passes (`corpora.plan_corpus_file`); it is never loaded whole."""
    plan = corpora.plan_corpus_file(lme_m, corpora.SIZES[size])
    typer.echo(json.dumps({"size": size, "chunks": corpora.n_chunks(plan), "sessions": len(plan),
                           "truncated_last": plan[-1].truncated,
                           "fingerprint": corpora.corpus_fingerprint(plan)}))  # fmt: skip


@app.command("latency-run")
def latency_run(
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json (query source)")],
    lme_m: Annotated[Path, typer.Option(help="longmemeval_m_cleaned.json (corpus source)")],
    out: Annotated[Path, typer.Option(help="Output JSON for this process run.")],
    corpus: Annotated[str, typer.Option(help="L5k or L50k")] = "L5k",
    gate_file: Annotated[
        Path | None, typer.Option("--gate", help="Gate JSON from `mem calibrate --out`.")
    ] = None,
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """One fresh-process B-C1 run (1 cold + 20 warm-up + 300 queries; 1 then 4 clients).

    Queries: the B-R1 test questions in sha1 order, cycled (`bench.latency.schedule`). Run it 3x."""
    import hashlib

    from bench import latency
    from bench.adapters import lme_ours
    from bench.suites.base import BackendUnavailable

    manifest = lme.read_manifest(MANIFEST)
    qs = {q["question_id"]: q for q in lme.load_questions(lme_file)}
    ordered = sorted((r.question_id for r in lme.select(manifest, "test", "br1")),
                     key=lambda i: hashlib.sha1(i.encode()).hexdigest())  # fmt: skip
    queries = latency.schedule([qs[i]["question"] for i in ordered])
    del qs  # only the question texts are needed: free the S file's haystacks before the corpus
    plan = corpora.plan_corpus_file(lme_m, corpora.SIZES[corpus])  # streamed, never loaded whole
    try:
        cfg = lme_ours.load_config(config, embedder)
        server = lme_ours.OursLoadServer(plan, cfg=cfg, config_path=config, gate_path=gate_file,
                                         models=embedder, device=device, serve_port=serve_port)  # fmt: skip
    except (BackendUnavailable, OSError, ValueError) as e:
        typer.echo(f"UNAVAILABLE latency-run: {e}")
        raise typer.Exit(2) from e
    try:
        res = latency.run_process(server.timed_recall, queries)
    finally:
        getattr(server, "close", lambda: None)()
    res |= {"corpus": corpus, "corpus_fingerprint": corpora.corpus_fingerprint(plan),
            "schedule": {"questions": len(ordered), "calls": len(queries),
                         "rule": "B-R1 test questions in sha1 order, call i = question i mod n"},
            "config": records.config_fingerprint(cfg), "hardware": records.hardware(),
            "git_sha": records.git_sha(), "bench_version": records.BENCH_VERSION}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}: cold {res['cold_ms']} ms, "
               f"warm p95 {res['one_client']['total']['p95']:.0f} ms")  # fmt: skip


@app.command("auth-overhead")
def auth_overhead_cmd(
    base: Annotated[str, typer.Option(help="Plain server, no auth: http://127.0.0.1:<port>")],
    secure: Annotated[str, typer.Option(help="The P7 path, e.g. https://<host>.<tailnet>.ts.net")],
    token: Annotated[
        str | None, typer.Option(envvar="MEM_TOKEN", help="Bearer token for --secure.")
    ] = None,
    base_token: Annotated[str | None, typer.Option(help="Bearer token for --base, if any.")] = None,
    n: Annotated[int, typer.Option(help="Measured recall calls per leg.")] = 300,
    store: Annotated[str, typer.Option(help="personal or work")] = "personal",
    out: Annotated[Path | None, typer.Option(help="Write the result JSON here.")] = None,
) -> None:
    """B-H1: the B-C1 p95 increase from auth/TLS (client-side recall, --secure minus --base) must
    be ≤ 50 ms. Exit 1 if it is not. Both servers must hold the same stores and models."""
    from bench import auth_overhead

    res = auth_overhead.measure(base, secure, token=token, base_token=base_token, n=n,
                                store=store)  # fmt: skip
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = "PASS" if res["pass"] else "FAIL"
    typer.echo(f"{verdict} auth-overhead: base p95 {res['base']['p95_ms']} ms, secure p95 "
               f"{res['secure']['p95_ms']} ms, delta {res['delta_p95_ms']} ms "
               f"(budget {res['budget_ms']} ms, n={res['n']})")  # fmt: skip
    if not res["pass"]:
        raise typer.Exit(1)


@app.command("export-tune")
def export_tune(
    lme_file: Annotated[Path, typer.Argument(help="longmemeval_s_cleaned.json")],
    out: Annotated[Path, typer.Option(help="Labels file for `mem calibrate`.")] = Path(
        "data/lme/tune_labels.jsonl"
    ),
) -> None:
    """Export the TUNE split (answerables + negatives, with chunk labels) for `mem calibrate`."""
    from bench.datasets import calibration

    qs = lme.load_questions(lme_file)
    lme.verify_manifest(qs, MANIFEST)
    rows = calibration.tune_label_rows({q["question_id"]: q for q in qs}, lme.read_manifest(MANIFEST),
                                       negatives.read_negatives(NEGATIVES))  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_tune_labels(rows, out)
    typer.echo(f"wrote {out}: {len(rows)} tune queries "
               f"({sum(r['kind'] == 'answerable' for r in rows)} answerable)")  # fmt: skip


@app.command("delivered-dump")
def delivered_dump_cmd(
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json")],
    out: Annotated[
        Path,
        typer.Option(
            help="JSONL, one line per question; the summary goes next to it as <out>.summary.json"
        ),
    ],
    split: Annotated[
        str, typer.Option(help="tune (the only split: gold evidence is read)")
    ] = "tune",
    config: Annotated[
        Path | None, typer.Option(help="Production config.yaml (the switches).")
    ] = None,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """Which rows recall delivers for every LME tune question, against the gold evidence
    (report §7). recall-any/-all@k (k = 5, 8, 10) and the gold-session fraction, overall and per
    question type. No API spend; the config's switches (F1-F5, A, B, D) apply as for any run."""
    from bench import delivered_dump as dd
    from bench.adapters.lme_ours import OursLmeBackend

    if split != "tune":
        raise typer.BadParameter("--split must be tune")
    manifest = lme.read_manifest(MANIFEST)
    questions = lme.load_questions(lme_file)
    lme.verify_manifest(questions, MANIFEST)
    by_id = {q["question_id"]: q for q in questions}
    try:
        backend = OursLmeBackend(config_path=config, gate_path=gate_file, models=embedder,
                                 device=device, serve_port=serve_port, follow_pages=True)  # fmt: skip
    except (RuntimeError, ValueError, OSError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    recs: list[dict] = []
    try:
        lme_eval.run_lme(backend, by_id, manifest, split, include_abs=False, want_legs=False,
                         drop_negatives=frozenset(m.question_id for m in lme.select(manifest, split, "br1")),
                         on_item=lambda it: recs.append(dd.question_record(it)))  # fmt: skip
        gate_meta = gate_record(backend.gate, gate_file) if gate_file else None
        cfg_fp = records.config_fingerprint(backend.cfg)
    finally:
        backend.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8")
    summary = dd.summarize(recs) | {
        "split": split, "follow_pages": True, "config": cfg_fp, "gate": gate_meta, "git_sha": records.git_sha(),
        "dataset": {"file": lme_file.name, "sha256": lme.sha256_file(lme_file),
                    "manifest_sha256": lme.sha256_file(MANIFEST)}}  # fmt: skip
    sum_path = out.with_suffix(".summary.json")
    sum_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out} ({len(recs)} questions) and {sum_path}")
    for k in dd.KS:
        o = summary["overall"].get(str(k), {})
        if o:
            typer.echo(
                f"  @{k}: recall-any {o['recall_any']:.3f}  recall-all@{k} "
                f"{o['recall_all']:.3f}  gold sessions {o['gold_session_frac']:.3f}"
            )
    if o := summary["overall"].get(dd.WHOLE):
        typer.echo(
            f"  whole delivery ({summary['paged_questions']} questions paged): recall-any "
            f"{o['recall_any']:.3f}  recall-all {o['recall_all']:.3f}  gold rounds "
            f"{o['gold_round_frac']:.3f}  gold sessions {o['gold_session_frac']:.3f}"
        )
    for t, v in summary["by_type"].items():
        typer.echo(
            f"  {t:<28} n={v['10']['n']}  "
            + "  ".join(
                f"@{k} any {v[str(k)]['recall_any']:.2f} all {v[str(k)]['recall_all']:.2f}"
                for k in dd.KS
            )
            + f"  whole any {v[dd.WHOLE]['recall_any']:.2f} all {v[dd.WHOLE]['recall_all']:.2f}"
            f" rounds {v[dd.WHOLE]['gold_round_frac']:.2f}"
        )


@app.command("asof-eligibility")
def asof_eligibility_cmd(
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json")],
    out: Annotated[Path | None, typer.Option(help="Write the JSON (summary + rows) here.")] = None,
) -> None:
    """MEM2-265 / ADR 0028: the LME tune qids whose eligible gold sessions change when `as_of` is
    the question's day at 23:59:59Z instead of its time of day, with before/after counts. Reads the
    dataset's dates and gold ids and the frozen manifest's tune rows only; no model, no API."""
    from bench import asof_report

    questions = lme.load_questions(lme_file)
    lme.verify_manifest(questions, MANIFEST)
    s = asof_report.report({q["question_id"]: q for q in questions}, lme.read_manifest(MANIFEST))
    s["dataset"] = {"file": lme_file.name, "sha256": lme.sha256_file(lme_file),
                    "manifest_sha256": lme.sha256_file(MANIFEST)}  # fmt: skip
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for line in asof_report.summary_lines(s):
        typer.echo(line)
    typer.echo(asof_report.table(s))


@app.command("delivered-pair")
def delivered_pair_cmd(
    base: Annotated[Path, typer.Argument(help="delivered.jsonl of the baseline (all off)")],
    feature: Annotated[Path, typer.Argument(help="delivered.jsonl of the feature run")],
) -> None:
    """Feature minus baseline, paired by question: recall-any/-all@k and the gold-session fraction
    with paired-bootstrap 95% CIs, overall and per question type (Round 9), and the same over the
    whole delivery (every delivered row, every page: recall-any, recall-all, gold-round fraction).
    No model, no API."""
    from bench import delivered_dump as dd

    def rows(p: Path) -> list[dict]:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    out = dd.pair(rows(base), rows(feature))
    names = {"recall_all": "recall-all", "recall_any": "recall-any"}

    whole = {
        "recall_any": "recall-any",
        "recall_all": "recall-all",
        "gold_round_frac": "gold-rounds",
    }
    ci = lambda c: f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"

    def line(label: str, d: dict) -> str:
        parts = []
        for k in dd.KS:
            for m, name in names.items():
                parts.append(f"{name}@{k} {ci(d[str(k)][m])}")
        w = d.get(dd.WHOLE, {})
        if all(m in w for m in whole):
            parts.append("| whole: " + "  ".join(f"{name} {ci(w[m])}" for m, name in whole.items()))
        return f"  {label}: " + "  ".join(parts)

    typer.echo(f"paired n={out['n']} (feature minus baseline; 95% paired-bootstrap CI)")
    if out["overall"]:
        typer.echo(line("overall", out["overall"]))
    for t, d in out["by_type"].items():
        typer.echo(line(f"{t} (n={d['8']['recall_all']['n']})", d))


@app.command("pool-recall")
def pool_recall_cmd(
    records_file: Annotated[
        Path,
        typer.Option("--records", help="Scored tune pools from `mem calibrate --records-out`."),
    ],
    size: Annotated[
        list[int] | None,
        typer.Option(help="Pool sizes to report (first N items of each pool); default: as scored."),
    ] = None,
) -> None:
    """Pool recall on scored tune pools: the share of answerable questions whose rerank pool holds
    a gold round (the ceiling of coverage_any), overall and per category, with a 95% interval."""
    from mem import calibrate

    _fp, pools = calibrate.read_records(records_file)
    for n in size or [max((len(q.items) for q in pools), default=0)]:
        r = calibrate.pool_recall(pools, n)
        if r["recall"] is None:
            typer.echo(f"pool {n}: no answerable queries")
            continue
        typer.echo(f"pool {n}: recall {r['recall']:.3f} ({r['hits']}/{r['n']}, "
                   f"95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f})")  # fmt: skip
        for cat in sorted({q.category for q in pools if q.answerable}):
            c = calibrate.pool_recall([q for q in pools if q.category == cat], n)
            typer.echo(f"  {cat or '-':<28} {c['recall']:.3f} ({c['hits']}/{c['n']})")


# ---- P3: SYN, B-W5, B-C3, calibration data ------------------------------------------------------


@app.command("syn-generate")
def syn_generate(
    seed: Annotated[int, typer.Option()] = 1,
    size: Annotated[str, typer.Option(help="S, M or L")] = "S",
    out: Annotated[
        Path, typer.Option(help="Directory for ops.jsonl, expected_state.json, queries.jsonl")
    ] = Path("data/syn"),
) -> None:
    """Write one SYN world (benchmarks-v2 §3.2) and print the sha256 of its three files."""
    from bench.synth.generator import generate

    syn = generate(seed, size)
    syn.write(out / f"{size}{seed}")
    typer.echo(json.dumps({"counts": syn.expected["counts"], "sha256": syn.sha256()}, indent=1))


@app.command("w5-run")
def w5_run(
    out: Annotated[Path, typer.Option(help="The run file `bench gate P3 --w5-run` scores.")],
    split: Annotated[
        str, typer.Option(help="tune = dev seed 1 (iteration); test = seed 101")
    ] = "tune",
    budget_usd: Annotated[float, typer.Option("--budget-usd", help="Hard cap (B-C5).")] = 10.0,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm that this spends real money.")
    ] = False,
    model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    effort: Annotated[str, typer.Option()] = "low",
    ledger: Annotated[str | None, typer.Option(help="Why (ledger line).")] = None,
    who: Annotated[str, typer.Option()] = "bench",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """B-W5 T2 run: the writer model + the production skill over 40 SYN dialogues (≈ $2-3)."""
    import tempfile

    from bench import agent_replay
    from bench.adapters.p3_ours import P3Backend
    from bench.llm import AnthropicLLM
    from bench.suites import b_w5
    from bench.synth import data as syn_data

    if not yes:
        typer.echo(f"B-W5 spends real money (≈ $2-3, hard cap ${budget_usd:.2f}); add --yes")
        raise typer.Exit(2)
    if split not in ("tune", "test"):
        raise typer.BadParameter("--split must be tune or test")
    seed = 101 if split == "test" else 1
    syn, dialogues = b_w5.dialogues_for(seed, "M")
    try:
        llm = AnthropicLLM(model, effort)
        backend = P3Backend(config_path=config, gate_path=gate_file, models=embedder,
                            device=device, serve_port=serve_port)  # fmt: skip
    except (RuntimeError, ValueError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    with tempfile.TemporaryDirectory(prefix="bench-w5-") as tmp:
        try:
            rec = agent_replay.run_all(dialogues, syn, backend.factory, Path(tmp) / "w5.db", llm,
                                       budget_usd, syn_data.TZ)  # fmt: skip
        finally:
            backend.close()
    rec |= {"seed": seed, "size": "M", "split": split, "config": records.config_fingerprint(backend.cfg),
            "bench_version": records.BENCH_VERSION, "git_sha": records.git_sha()}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    records.append_ledger({"who": who, "why": ledger or f"B-W5 run ({split})", "phase": "P3",
                           "split": f"{split}-data", "kind": "w5-run", "bench_id": "B-W5",
                           "cost": rec["cost"], "config_hash": rec["config"]["fp_all"]})  # fmt: skip
    typer.echo(f"wrote {out}: {rec['cost']} complete={rec['complete']}")
    if not rec["complete"]:
        raise typer.Exit(1)


def _e1_run_m(lme_file, out, file_sha, backend, cfg, answerer, judge, gate_file, effort,
              budget_usd, limit, ledger, who) -> None:  # fmt: skip
    """`e1-run --lme-size M`: the streaming, resumable mem arm (`bench.e1_m`), one ledger line per
    session that put a test question in front of the answerer (a crash or a stop included)."""
    from bench import e1, e1_m

    jsonl, status = e1_m.paths(out)
    if backend.gate is None:
        typer.echo("UNAVAILABLE: no gate: pass --gate gate.json (from mem calibrate --out)")
        raise typer.Exit(2)
    meta = {"answerer": answerer.model, "judge": judge.model, "effort": effort,
            "gate": gate_record(backend.gate, gate_file) if gate_file else None,
            "config": records.config_fingerprint(cfg), "git_sha": records.git_sha(),
            "bench_version": records.BENCH_VERSION, "size": "M", "split": "full", "full": True,
            "spends_test_look": True, "mem_context": "expanded", "answerer_effort": None,
            "answerer_thinking": "between_tools",
            "dataset": {"name": "LongMemEval_M cleaned", "file": lme_file.name, "sha256": file_sha,
                        "manifest_sha256": None, "split": "full"}}  # fmt: skip
    kind = "e1-run-m" if limit is None else "e1-run-m-estimate"

    def look(cost: float | None, n: int, **extra: str) -> None:
        records.append_ledger({"who": who, "why": ledger, "phase": "P3", "split": "test",
                               "kind": kind, "bench_id": "B-E1", "cost": cost, "n_questions": n,
                               "config_hash": meta["config"]["fp_all"], **extra},
                              records.LEDGER)  # fmt: skip

    def retrieve(q: dict):
        return backend.retrieve(q, lme.haystack_of(q), want_legs=False)

    try:
        rec = e1_m.run_m(lme_file, retrieve, answerer, judge, jsonl=jsonl, status=status,
                         meta=meta, budget_usd=budget_usd, limit=limit)  # fmt: skip
    except e1_m.ResumeMismatch as e:
        typer.echo(f"REFUSED {e}")
        raise typer.Exit(2) from e
    except BaseException as e:  # a crash or a kill after a test question still spent the look
        look(None, len(e1_m.read_rows(jsonl)), crashed=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        backend.close()
    look(rec["cost"]["usd"], rec["n"])
    if limit is not None:
        rec["estimate"] = e1_m.estimate(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(
        f"wrote {out}: n={rec['n']} complete={rec['complete']} cost=${rec['cost']['usd']:.4f}"
    )
    for k in ("accuracy", "accuracy_ci_lo", "accuracy_ci_hi", "abstention_accuracy", "recall_any5"):
        if rec.get(k) is not None:
            typer.echo(f"  {k} = {rec[k]:.4f}")
    typer.echo(
        f"  judge_flagged = {rec['judge_flagged']}"
        + (f" ({e1.UNPARSED})" if rec["judge_flagged"] else "")
    )
    for t, v in rec["per_type"].items():
        typer.echo(f"  {t}: {v['accuracy']:.3f} (n={v['n']})")
    if "estimate" in rec:
        e = rec["estimate"]
        typer.echo(
            f"ESTIMATE for {e['questions']} questions (from {e['basis_n']}): "
            f"${e['usd']:.2f} API, {e['host_hours']:.1f} host hours"
        )
    if rec["stopped"]:
        typer.echo(f"STOPPED at the budget: {rec['stopped']}")
    if not rec["complete"] and limit is None:
        raise typer.Exit(1)


@app.command("e1-run")
def e1_run_cmd(
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json")],
    out: Annotated[Path, typer.Option(help="The run file `bench gate P3 --e1-run` scores.")],
    split: Annotated[
        str | None, typer.Option(help="tune (informational) or test (a look, once, at the freeze)")
    ] = None,
    condition: Annotated[
        list[str] | None, typer.Option(help="mem, none, oracle (repeat); default all three.")
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="All 500 questions of LongMemEval_S (not a split). Spends a "
            "test look: needs --spend-look and --ledger.",
        ),
    ] = False,
    spend_look: Annotated[
        bool, typer.Option("--spend-look", help="Confirm that --full spends one P3 test look.")
    ] = False,
    lme_size: Annotated[
        str,
        typer.Option(
            help="S (default) or M: LongMemEval_M, the same 500 questions with about 500 sessions "
            "each. M takes --full, runs the mem arm only, streams the file, one store per "
            "question, resumable (`<out>.jsonl`, `<out>.STATUS`); --limit 5 prints a cost and "
            "host-hours estimate. Spends a test look as --full does."
        ),
    ] = "S",
    answerer_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    judge_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    effort: Annotated[
        str,
        typer.Option(help="The JUDGE's effort; answerer: thinking off (between_tools), no effort."),
    ] = "low",
    budget_usd: Annotated[
        float | None,
        typer.Option(
            "--budget-usd",
            help="Hard cap: the run stops before going over (B-C5); default $10 ($30 with --full).",
        ),
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N questions (smoke run).")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm that this spends real money.")
    ] = False,
    ledger: Annotated[
        str | None, typer.Option(help="Why (ledger line; required on test and --full).")
    ] = None,
    oracle_pool: Annotated[
        list[Path] | None,
        typer.Option(
            "--oracle-pool",
            help="B-E1 validity: an earlier `bench e1-run` file of the same harness whose oracle "
            "answers pool with this run's (repeat; the current run is always in the pool). VALID "
            "needs the pooled oracle >= 0.92 and this run's >= 0.90 (the owner, 2026-10-27, re-based on "
            "the strict judge).",
        ),
    ] = None,
    context: Annotated[
        str,
        typer.Option(
            help="What the mem arm's answerer reads: `expanded` (default: recall's text "
            "plus the full text of every delivered row) or `recall` (recall's text alone; "
            "a diagnostic, not a B-E1 result)."
        ),
    ] = "expanded",
    who: Annotated[str, typer.Option()] = "bench",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """B-E1 T2 run: answerer + judge over the LongMemEval split (br1 + abs), conditions mem/none/oracle."""
    from bench import e1
    from bench.llm import PRICES, AnthropicLLM

    if lme_size not in ("S", "M"):
        raise typer.BadParameter("--lme-size must be S or M")
    is_m = lme_size == "M"
    if is_m and not full:
        raise typer.BadParameter("--lme-size M is a --full run (all 500 questions, a test look)")
    if is_m and any(c != "mem" for c in condition or ()):
        raise typer.BadParameter("--lme-size M runs the mem arm only (none and oracle reuse S's)")
    conditions = ("mem",) if is_m else tuple(condition or e1.CONDITIONS)
    if context not in ("recall", "expanded"):
        raise typer.BadParameter("--context must be recall or expanded")
    if full and split is not None:
        raise typer.BadParameter("--full runs every question: it takes no --split")
    if full and limit is not None and not is_m:
        raise typer.BadParameter("--full runs every question: it takes no --limit")
    if full and context != "expanded":  # a P3 look is not spent on a diagnostic
        raise typer.BadParameter("--full is the B-E1 result: it takes no --context recall")
    if spend_look and not full:
        raise typer.BadParameter("--spend-look belongs to --full")
    split = "test" if full else (split or "tune")
    if split not in ("tune", "test"):
        raise typer.BadParameter("--split must be tune or test")
    budget_usd = budget_usd if budget_usd is not None else (
        (5.0 if limit is not None else 15.0) if is_m else 30.0 if full else 10.0)  # fmt: skip
    bad = [c for c in conditions if c not in e1.CONDITIONS]
    if bad:
        raise typer.BadParameter(f"--condition must be one of {e1.CONDITIONS}, got {bad}")
    for m in (answerer_model, judge_model):
        if m not in PRICES:
            raise typer.BadParameter(f"no list price for {m!r}; known: {sorted(PRICES)}")
    if is_m and oracle_pool:
        raise typer.BadParameter("--lme-size M has no oracle arm: it takes no --oracle-pool")
    harness = {"prompt_version": e1.PROMPT_VERSION, "prompts_sha256": e1.prompts_sha256(),
               "answerer": answerer_model, "answerer_thinking": e1.ANSWERER_THINKING,
               "judge": judge_model, "split": "full" if full else split,
               "judge_version": e1.JUDGE_VERSION}  # fmt: skip
    if any(Path(p).resolve() == out.resolve() for p in oracle_pool or ()):
        typer.echo(f"REFUSED --oracle-pool: {out} is the --out file (the current run is always "
                   "in the pool, once)")  # fmt: skip
        raise typer.Exit(2)
    try:  # before any API call: a refused pool spends nothing
        pool = e1.load_pool(oracle_pool or [], harness)
    except (e1.PoolRefused, OSError, ValueError, KeyError) as e:
        typer.echo(f"REFUSED --oracle-pool: {e}")
        raise typer.Exit(2) from e
    if not yes:
        typer.echo(f"B-E1 spends real money (hard cap ${budget_usd:.2f}); add --yes")
        raise typer.Exit(2)
    if full and not spend_look:
        typer.echo(f"--full is all 500 questions of LongMemEval_{lme_size}: it spends a P3 test "
                   "look (benchmarks-v2 §5.4). Add --spend-look to say so")  # fmt: skip
        raise typer.Exit(2)
    if split == "test":
        if not ledger:
            typer.echo('a test look needs --ledger "<why>" (benchmarks-v2 §5.4)')
            raise typer.Exit(2)
        try:
            freeze_mod.check_test_look_allowed("P3")
        except freeze_mod.Refused as e:
            typer.echo(f"REFUSED {e}")
            raise typer.Exit(2) from e
    if is_m:
        qs = []  # streamed by `e1_m.run_m`, never loaded
        file_sha = fetch.verify(lme_file.name, lme_file)  # one pass over 2.7 GB, 1 MB at a time
    elif full:
        fetch.verify(lme_file.name, lme_file)  # the pinned file: all 500, not a manifest slice
        qs = lme.load_questions(lme_file)
    else:
        manifest = lme.read_manifest(MANIFEST)
        questions = lme.load_questions(lme_file)
        lme.verify_manifest(questions, MANIFEST)
        by_id = {q["question_id"]: q for q in questions}
        rows = lme.select(manifest, split, "br1") + lme.select(manifest, split, "abs")
        qs = [by_id[r.question_id] for r in rows][:limit]
    backend = None
    cfg = None
    try:
        answerer, judge = (
            AnthropicLLM(answerer_model, None),
            AnthropicLLM(judge_model, effort),
        )  # answerer: thinking off (between_tools), no effort, as e1_qa.py ask() (MEM2-227 F5)
        from bench.adapters.lme_ours import OursLmeBackend, load_config

        cfg = load_config(config, embedder)  # recorded whichever arms run
        if "mem" in conditions:
            backend = OursLmeBackend(cfg=cfg, config_path=config, gate_path=gate_file,
                                     models=embedder, device=device, serve_port=serve_port,
                                     expand=context == "expanded")  # fmt: skip
    except (RuntimeError, ValueError, OSError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e

    if is_m:
        _e1_run_m(lme_file, out, file_sha, backend, cfg, answerer, judge, gate_file, effort,
                  budget_usd, limit, ledger, who)  # fmt: skip
        return

    truncated: dict[str, list[str]] = {}

    def retrieve(q: dict) -> str:
        r = backend.retrieve(q, lme.haystack_of(q), want_legs=False)  # type: ignore[union-attr]
        if r.truncated:
            truncated[q["question_id"]] = r.truncated
        return e1.mem_context(r)

    def look(cost: float | None, **extra: str) -> None:
        """The ledger line of a test look: spent once the answerer has seen a test question."""
        if split == "test":
            records.append_ledger({"who": who, "why": ledger, "phase": "P3", "split": "test",
                                   "kind": "e1-run-full" if full else "e1-run",
                                   "bench_id": "B-E1", "cost": cost,
                                   "config_hash": records.config_fingerprint(cfg)["fp_all"],
                                   **({"n_questions": len(qs)} if full else {}), **extra},
                                  records.LEDGER)  # fmt: skip

    try:
        rec = e1.run_e1(qs, retrieve, answerer, judge, conditions=conditions,
                        budget_usd=budget_usd)  # fmt: skip
    except BaseException as e:  # a crash or a kill after a test question still spent the look
        look(None, crashed=f"{type(e).__name__}: {e}"[:300])
        raise
    finally:
        if backend:
            backend.close()
    look(rec["cost"]["usd"])  # before anything else can fail
    rec |= {"split": "full" if full else split, "limit": limit,
            "full": full, "spends_test_look": split == "test",
            "config": records.config_fingerprint(cfg),
            "gate": gate_record(backend.gate, gate_file) if backend and gate_file else None,
            "dataset": {"name": "LongMemEval_S cleaned", "file": lme_file.name,
                        "sha256": lme.sha256_file(lme_file),
                        "manifest_sha256": None if full else lme.sha256_file(MANIFEST),
                        "split": "full" if full else split},
            "bench_version": records.BENCH_VERSION, "git_sha": records.git_sha(),
            "effort": effort, "answerer_effort": None, "answerer_thinking": e1.ANSWERER_THINKING, "mem_context": context, "truncated_rows": truncated}  # fmt: skip
    refused = None
    try:
        vp = e1.oracle_pool(rec, pool)
    except e1.PoolRefused as e:  # a pooled file is this very run: kept alone, and exit 2 below
        refused, pool, vp = str(e), [], e1.oracle_pool(rec)
    rec |= {k: vp[k] for k in ("oracle_run", "oracle_pooled", "oracle_pool_n", "oracle_pool")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    metrics, _ = e1.score_run(rec, pool=pool)
    typer.echo(
        f"wrote {out}: n={len(rec['items'])}/{rec['planned']} complete={rec['complete']} "
        f"cost=${rec['cost']['usd']:.4f}"
    )
    for k in (
        "accuracy",
        "accuracy_ci_lo",
        "accuracy_ci_hi",
        "accuracy_none",
        "accuracy_oracle",
        "abstention_accuracy",
    ):
        if k in metrics:
            typer.echo(f"  {k} = {metrics[k]:.4f}")
    typer.echo(
        f"  judge_flagged = {rec['judge_flagged']}"
        + (f" ({e1.UNPARSED})" if rec["judge_flagged"] else "")
    )
    _echo_e1_validity(vp)
    if refused:
        typer.echo(f"REFUSED --oracle-pool: {refused}")
        raise typer.Exit(2)
    if rec["stopped"]:
        typer.echo(f"STOPPED at the budget: {rec['stopped']}")
    if not rec["complete"]:
        raise typer.Exit(1)


@app.command("e1-pair")
def e1_pair_cmd(
    arm: Annotated[Path, typer.Argument(help="e1-run file of the arm (its mem arm is read)")],
    base: Annotated[Path, typer.Argument(help="e1-run file of the base arm (its mem arm is read)")],
    b: Annotated[int, typer.Option(help="Bootstrap resamples.")] = 2000,
) -> None:
    """Round 10 (MEM2-270): an arm's B-E1 run against the base arm's, paired by question on the
    `mem` arm of both: accuracy, delta with a 95% paired-bootstrap CI (resampled within type), and
    the question count (right in the arm minus right in the base), overall and per type. Names every
    type down by more than 1 question (the Round 10 pass line). No model, no API."""
    from bench import e1_agentic

    try:
        cmp = e1_agentic.compare(json.loads(arm.read_text(encoding="utf-8")),
                                 json.loads(base.read_text(encoding="utf-8")),
                                 new_condition="mem", base_condition="mem", b=b)  # fmt: skip
    except ValueError as err:
        typer.echo(f"REFUSED: {err}")
        raise typer.Exit(2) from err

    def line(label: str, r: dict) -> str:
        lo, hi = r["delta_ci95"]
        return (f"  {label}: n={r['n']} arm {r['new']:.3f} base {r['baseline']:.3f}"
                f" delta {r['delta']:+.4f} CI [{lo:+.4f}, {hi:+.4f}] questions {r['questions']:+d}")  # fmt: skip

    typer.echo(f"paired n={cmp['n_paired']} (arm minus base, mem arms; 95% paired-bootstrap CI)")
    typer.echo(line("overall", cmp["overall"]))
    for t, r in cmp["per_type"].items():
        typer.echo(line(t, r))
    d = cmp["discordant"]
    typer.echo(
        f"  discordant: arm only right {d['new_only_right']}, base only right {d['baseline_only_right']}"
    )
    if cmp["new_only"] or cmp["baseline_only"]:
        typer.echo(
            f"  unpaired: arm only {len(cmp['new_only'])}, base only {len(cmp['baseline_only'])}"
        )
    down = [t for t, r in cmp["per_type"].items() if r["questions"] < -1]
    if down:
        typer.echo("  DOWN by more than 1 question: " + ", ".join(down))
    else:
        typer.echo("  no type down by more than 1 question")


@app.command("e1-agentic")
def e1_agentic_cmd(
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json")],
    out: Annotated[Path, typer.Option(help="The run file.")],
    baseline: Annotated[
        Path,
        typer.Option(
            help="A run file to pair against by question: `bench e1-run` (its mem arm) or an earlier e1-agentic run (the skill to beat)."
        ),
    ],
    split: Annotated[str, typer.Option(help="tune (the adoption run) or test.")] = "tune",
    skill: Annotated[
        Path, typer.Option(help="The skill the answerer gets as its system prompt.")
    ] = Path("skills/mem/SKILL.md"),
    answerer_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    judge_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    effort: Annotated[str, typer.Option(help="The JUDGE's effort.")] = "low",
    budget_usd: Annotated[
        float, typer.Option("--budget-usd", help="Hard cap: the run stops before going over.")
    ] = 10.0,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm that this spends real money.")
    ] = False,
    ledger: Annotated[str | None, typer.Option(help="Why (required on test).")] = None,
    who: Annotated[str, typer.Option()] = "bench",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """B-E1 agentic arm: the answerer uses recall/get itself (the skill as system prompt), paired with a baseline run."""
    from bench import e1, e1_agentic
    from bench.llm import PRICES, AnthropicLLM

    if split not in ("tune", "test"):
        raise typer.BadParameter("--split must be tune or test")
    for m in (answerer_model, judge_model):
        if m not in PRICES:
            raise typer.BadParameter(f"no list price for {m!r}; known: {sorted(PRICES)}")
    if not yes:
        typer.echo(f"the agentic arm spends real money (hard cap ${budget_usd:.2f}); add --yes")
        raise typer.Exit(2)
    if split == "test":
        if not ledger:
            typer.echo('a test look needs --ledger "<why>" (benchmarks-v2 §5.4)')
            raise typer.Exit(2)
        try:
            freeze_mod.check_test_look_allowed("P3")
        except freeze_mod.Refused as e:
            typer.echo(f"REFUSED {e}")
            raise typer.Exit(2) from e
    base = json.loads(baseline.read_text(encoding="utf-8"))
    if base.get("split") not in (None, split):
        typer.echo(f"REFUSED the baseline is on split {base['split']!r}, this run on {split!r}")
        raise typer.Exit(2)
    manifest = lme.read_manifest(MANIFEST)
    questions = lme.load_questions(lme_file)
    lme.verify_manifest(questions, MANIFEST)
    by_id = {q["question_id"]: q for q in questions}
    rows = lme.select(manifest, split, "br1") + lme.select(manifest, split, "abs")
    qs = [by_id[r.question_id] for r in rows][:limit]
    skill_text = skill.read_text(encoding="utf-8")
    backend = None
    try:
        answerer, judge = AnthropicLLM(answerer_model, None), AnthropicLLM(judge_model, effort)
        from bench.adapters.lme_ours import OursLmeBackend, load_config

        cfg = load_config(config, embedder)
        backend = OursLmeBackend(cfg=cfg, config_path=config, gate_path=gate_file,
                                 models=embedder, device=device, serve_port=serve_port)  # fmt: skip
    except (RuntimeError, ValueError, OSError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    try:
        rec = e1_agentic.run_agentic(
            qs, lambda q: backend.open_tools(q, lme.haystack_of(q)), answerer, judge,
            skill_text=skill_text, budget_usd=budget_usd)  # fmt: skip
    finally:
        backend.close()
    rec |= {"split": split, "limit": limit, "skill_path": str(skill), "baseline": str(baseline),
            "config": records.config_fingerprint(cfg),
            "gate": gate_record(backend.gate, gate_file) if gate_file else None,
            "dataset": {"name": "LongMemEval_S cleaned", "file": lme_file.name,
                        "sha256": lme.sha256_file(lme_file),
                        "manifest_sha256": lme.sha256_file(MANIFEST), "split": split},
            "bench_version": records.BENCH_VERSION, "git_sha": records.git_sha(),
            "effort": effort, "answerer_effort": None, "answerer_thinking": "between_tools"}  # fmt: skip
    cmp = e1_agentic.compare(rec, base, base_condition=e1_agentic.baseline_condition(base))
    rec["compare"], rec["adoption"] = cmp, e1_agentic.adoption(cmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if split == "test":
        records.append_ledger({"who": who, "why": ledger, "phase": "P3", "split": "test",
                               "kind": "e1-agentic", "bench_id": "B-E1", "cost": rec["cost"]["usd"],
                               "config_hash": rec["config"]["fp_all"]}, records.LEDGER)  # fmt: skip
    metrics, per = e1_agentic.score(rec)
    typer.echo(
        f"wrote {out}: n={len(rec['items'])}/{rec['planned']} complete={rec['complete']} "
        f"cost=${rec['cost']['usd']:.4f} skill={rec['skill_sha256'][:12]}"
    )
    lo, hi = per["ci95"]
    typer.echo(f"  accuracy = {metrics['accuracy']:.4f} [{lo:.4f}, {hi:.4f}]")
    typer.echo(f"  judge_flagged = {rec['judge_flagged']}"
               + (f" ({e1.UNPARSED})" if rec["judge_flagged"] else ""))  # fmt: skip
    for t, row in per["per_type"].items():
        c = cmp["per_type"].get(t)
        typer.echo(
            f"  {t}: n={row['n']} acc={row['accuracy']:.3f}"
            + (f" vs baseline {c['baseline']:+.3f} delta {c['delta']:+.3f}" if c else "")
        )
    o = cmp["overall"]
    typer.echo(f"  paired n={cmp['n_paired']}: delta {o['delta']:+.4f} CI {o['delta_ci95']}")
    verdict = "ADOPT" if rec["adoption"]["adopt"] else "KEEP CURRENT"
    if not rec["complete"]:
        verdict = "INCOMPLETE (not adoptable)"
    typer.echo(
        f"  adoption: {verdict} "
        + "; ".join(rec["adoption"]["reasons"])
        + " (B-W5 unchanged: check its own run)"
    )
    if rec["stopped"]:
        typer.echo(f"STOPPED at the budget: {rec['stopped']}")
    if not rec["complete"]:
        raise typer.Exit(1)


@app.command("locomo-fetch")
def locomo_fetch_cmd(
    dest: Annotated[Path, typer.Option(help="Where to put the file.")] = Path("data/locomo"),
) -> None:
    """Download the official locomo10.json (CC BY-NC 4.0) and check it against the pinned sha256 and size."""
    typer.echo(f"{locomo_data.fetch(dest)}  {locomo_data.FILE}  ({locomo_data.URL})")


@app.command("locomo-run")
def locomo_run_cmd(
    locomo_file: Annotated[Path, typer.Option(help="locomo10.json (`bench locomo-fetch`).")],
    out: Annotated[Path, typer.Option(help="The run file.")],
    answerer_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    judge_model: Annotated[str, typer.Option()] = "claude-sonnet-5-5",
    effort: Annotated[str, typer.Option()] = "low",
    budget_usd: Annotated[
        float,
        typer.Option("--budget-usd", help="Hard cap: the run stops before going over (B-C5)."),
    ] = 30.0,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N questions (smoke run; no ledger line).")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm that this spends real money.")
    ] = False,
    ledger: Annotated[str | None, typer.Option(help="Why (ledger line of a full run).")] = None,
    who: Annotated[str, typer.Option()] = "bench",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """LoCoMo T2 run: every conversation through ingest_session, answerer + judge over all 1,986 questions."""
    from bench import locomo
    from bench.llm import PRICES, AnthropicLLM

    for m in (answerer_model, judge_model):
        if m not in PRICES:
            raise typer.BadParameter(f"no list price for {m!r}; known: {sorted(PRICES)}")
    if not yes:
        typer.echo(f"LoCoMo spends real money (hard cap ${budget_usd:.2f}); add --yes")
        raise typer.Exit(2)
    digest = locomo_data.verify(locomo_file)
    convs = locomo_data.load(locomo_file)
    try:
        answerer, judge = AnthropicLLM(answerer_model, None), AnthropicLLM(judge_model, effort)
        from bench.adapters.lme_ours import load_config
        from bench.adapters.locomo_ours import OursLocomoBackend

        cfg = load_config(config, embedder)
        backend = OursLocomoBackend(cfg=cfg, config_path=config, gate_path=gate_file,
                                    models=embedder, device=device, serve_port=serve_port)  # fmt: skip
        if backend.gate is None:
            raise BackendUnavailable("no gate: pass --gate gate.json (from mem calibrate --out)")
    except (RuntimeError, ValueError, OSError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    try:
        rec = locomo.run_locomo(convs, backend.open_conversation, answerer, judge,
                                budget_usd=budget_usd, limit=limit)  # fmt: skip
    finally:
        backend.close()
    rec |= {"limit": limit, "config": records.config_fingerprint(cfg),
            "gate": gate_record(backend.gate, gate_file) if gate_file else None,
            "dataset": {"name": locomo_data.NAME, "file": locomo_data.FILE, "sha256": digest,
                        "commit": locomo_data.COMMIT, "licence": locomo_data.LICENSE},
            "bench_version": records.BENCH_VERSION, "git_sha": records.git_sha(),
            "effort": effort, "answerer_effort": None, "answerer_thinking": "between_tools",
            "mem_context": "expanded",
            "truncated_rows": backend.truncated_rows}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    metrics, extra = locomo.score_run(rec)
    if limit is None:
        records.append_ledger({"who": who, "why": ledger or "LoCoMo run", "phase": "final",
                               "split": "locomo", "kind": "locomo-run", "bench_id": "B-LoCoMo",
                               "cost": rec["cost"]["usd"], "config_hash": rec["config"]["fp_all"]},
                              records.LEDGER)  # fmt: skip
    typer.echo(
        f"wrote {out}: n={len(rec['items'])}/{rec['planned']} complete={rec['complete']} "
        f"cost=${rec['cost']['usd']:.4f}"
    )
    for k in (
        "judge_accuracy",
        "judge_accuracy_ci_lo",
        "judge_accuracy_ci_hi",
        "f1",
        "adversarial_accuracy",
    ):
        typer.echo(f"  {k} = {metrics[k]:.4f}")
    typer.echo(f"  judge_flagged = {int(metrics['judge_flagged'])} (replies without a CORRECT/WRONG "
               "label, or with both; not credited)")  # fmt: skip
    for name, row in extra["per_category"].items():
        typer.echo(
            "  "
            + name
            + ": "
            + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
    if rec["stopped"]:
        typer.echo(f"STOPPED at the budget: {rec['stopped']}")
    if not rec["complete"]:
        raise typer.Exit(1)


@app.command("w5-audit-sheet")
def w5_audit_sheet(
    run_file: Annotated[Path, typer.Argument(help="A `bench w5-run` file.")],
    out: Annotated[Path, typer.Option()] = Path("w5_audit.tsv"),
) -> None:
    """Write the 50-memory audit sheet (30 aligned + 20 unaligned, sampled with seed 7)."""
    from bench.suites import b_w5

    rec = json.loads(run_file.read_text(encoding="utf-8"))
    n = b_w5.write_audit_sheet(rec, out, rec["seed"], rec["size"])
    typer.echo(f"wrote {out}: {n}. Fill human_check (y/n) on every row; see p3-gate.md §4")


@app.command("w5-failures")
def w5_failures(
    run_file: Annotated[Path, typer.Argument(help="A `bench w5-run` file (dev split).")],
    out: Annotated[Path | None, typer.Option(help="Also write the full report as JSON.")] = None,
) -> None:
    """Group B-W5 `about_fill` and `conflict_action` misses (offline, no network, no dialogue text)."""
    from bench import w5_failures as wf

    rec = json.loads(run_file.read_text(encoding="utf-8"))
    rep = wf.analyse(rec, rec["seed"], rec["size"])
    if out:
        out.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    brief = {k: v for k, v in rep.items() if k not in ("about_misses", "conflict_failures")}
    typer.echo(json.dumps(brief, indent=1, sort_keys=True))


@app.command("c3-run")
def c3_run_cmd(
    lme_m: Annotated[Path, typer.Option(help="longmemeval_m_cleaned.json (corpus source)")],
    out: Annotated[Path, typer.Option(help="Output JSON for this process run.")],
    corpus: Annotated[str, typer.Option(help="L5k or L50k")] = "L5k",
    seed: Annotated[int, typer.Option(help="SYN-M seed whose remember ops are replayed")] = 101,
    gate_file: Annotated[Path | None, typer.Option("--gate", help="Gate JSON.")] = None,
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
) -> None:
    """One fresh-process B-C3 run: 1 cold + 20 warm-up + 300 `remember` (1 client) + 300 (4 clients)."""
    from bench import latency
    from bench.adapters.p3_ours import OursWriteServer, P3Backend
    from bench.suites.base import BackendUnavailable
    from bench.synth.generator import generate

    plan = corpora.plan_corpus_file(lme_m, corpora.SIZES[corpus])  # streamed, never loaded whole
    syn = generate(seed, "M")
    try:
        backend = P3Backend(config_path=config, gate_path=gate_file, models=embedder,
                            device=device, serve_port=serve_port)  # fmt: skip
        server = OursWriteServer(plan, syn, backend)
    except (BackendUnavailable, OSError, ValueError) as e:
        typer.echo(f"UNAVAILABLE c3-run: {e}")
        raise typer.Exit(2) from e
    try:
        need = 1 + latency.WARMUP + 2 * latency.N_QUERIES
        if server.n_ops() < need:
            typer.echo(f"SYN M seed {seed} has {server.n_ops()} remember ops, need {need}")
            raise typer.Exit(2)
        res = latency.run_write_process(server.timed_remember, [str(i) for i in range(need)])
    finally:
        server.close()
        backend.close()
    res |= {"corpus": corpus, "corpus_fingerprint": corpora.corpus_fingerprint(plan),
            "schedule": {"syn_seed": seed, "ops": need, "rule": "remember ops of SYN-M in order"},
            "config": records.config_fingerprint(backend.cfg), "hardware": records.hardware(),
            "git_sha": records.git_sha(), "bench_version": records.BENCH_VERSION}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}: cold {res['cold_ms']:.0f} ms, "
               f"warm p95 {res['one_client']['total']['p95']:.0f} ms")  # fmt: skip


@app.command("syn-score-dev")
def syn_score_dev(
    out: Annotated[Path, typer.Option(help="Scored pools (mem calibrate --records format).")],
    conflicts_out: Annotated[Path, typer.Option(help="Conflict-pair records for syn-fit-theta.")],
    seeds: Annotated[str, typer.Option(help="Dev seeds only (1-5).")] = "1,2,3,4,5",
    size: Annotated[str, typer.Option()] = "M",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
    fit: Annotated[Path | None, typer.Option(
        help="`syn-fit-theta --out` JSON: replay under its list cut (gated store states).")] = None,
) -> None:  # fmt: skip
    """Replay the SYN DEV seeds on the real stack; write their scored read pools and conflict lists.

    Runs with an in-memory gate whose theta_conf is -inf, so the candidate lists are not
    pre-filtered (§3.3 step 6). theta_conf = 0 would not do: it is compared to raw rerank logits.
    With --fit the oracle acts only on what that fit's list cut would have returned (the same
    `list_cut`, applied to the recorded features), so the store states are the gate's: a missed
    conflict leaves its stale memory active for every later op. Each record still carries the whole
    pool; `gated_by` names the fit (MEM2-156)."""
    import tempfile

    from bench.adapters.p3_ours import P3Backend
    from bench.suites.base import Context
    from bench.synth import calibration
    from bench.synth import data as syn_data
    from bench.synth.run import SynSetup, replay_seed, require_p3
    from mem import calibrate

    ids = tuple(int(x) for x in seeds.split(","))
    if bad := [s for s in ids if s not in syn_data.DEV_SEEDS]:
        typer.echo(f"REFUSED: {bad} are not dev seeds; only {syn_data.DEV_SEEDS} feed calibration")
        raise typer.Exit(2)
    cut = None
    if fit is not None:
        try:
            cut = calibration.fit_cut(json.loads(fit.read_text(encoding="utf-8")))
        except (ValueError, KeyError) as e:
            typer.echo(f"REFUSED: --fit {fit}: {e}")
            raise typer.Exit(2) from e
    try:
        backend = P3Backend(config_path=config, models=embedder, device=device,
                            serve_port=serve_port, unfiltered_conflicts=True)  # fmt: skip
    except (RuntimeError, ValueError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    pools, conflicts = [], []
    with tempfile.TemporaryDirectory(prefix="bench-syn-dev-") as tmp:
        ctx = Context(Path(tmp), backend.factory, None,
                      {"syn": SynSetup(ids, size, "tune", workdir=Path(tmp), cut=cut)})  # fmt: skip
        try:
            require_p3(ctx)
            for seed in ids:
                run = replay_seed(ctx, seed, size)
                conflicts += calibration.conflict_records(run)
                server, _ = backend.open_server(run.db, tz=syn_data.TZ,
                                                now=run.syn.expected["end_now"])  # fmt: skip
                try:
                    pools += calibration.dev_pools([(run, server)])
                finally:
                    server.close()
        except BackendUnavailable as e:
            typer.echo(f"UNAVAILABLE: {e}")
            raise typer.Exit(2) from e
        finally:
            backend.close()
    from mem import calibrate as cal

    cal.write_records(out, pools, fp_all=backend.cfg.fp_all())
    conflicts_out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in conflicts),
                             encoding="utf-8")  # fmt: skip
    del calibrate
    typer.echo(f"wrote {out} ({len(pools)} pools) and {conflicts_out} ({len(conflicts)} ops)"
               + (f"; gated by {cut.gated_by['fingerprint'][:12]} theta_conf "
                  f"{cut.gated_by['theta_conf']:g}" if cut else ""))  # fmt: skip


def _conflict_records(conflicts: Path) -> list[dict]:
    recs = [
        json.loads(ln) for ln in conflicts.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    if any(c.get("features") is None for r in recs for c in r["candidates"]):
        typer.echo("REFUSED: records without features; re-run `bench syn-score-dev`")
        raise typer.Exit(2)
    return recs


def _floats(text: str | None, default: tuple) -> tuple:
    return default if text is None else tuple(float(x) for x in text.split(",") if x.strip())


@app.command("syn-fit-theta")
def syn_fit_theta(
    conflicts: Annotated[Path, typer.Argument(help="`syn-score-dev --conflicts-out` file.")],
    out: Annotated[Path | None, typer.Option(help="Write the fit as JSON.")] = None,
    report: Annotated[Path | None, typer.Option(help="Write the CV report (default: next to --out).")] = None,
    floor: Annotated[float, typer.Option(help="Train precision floor for θ_conf.")] = 0.52,
    select_floor: Annotated[float | None, typer.Option(
        help="Worst-inner-fold precision a config needs to be eligible (default: --floor).")] = None,
    top_ks: Annotated[str | None, typer.Option(help="top_k grid, e.g. '3,5' (default).")] = None,
    deltas: Annotated[str | None, typer.Option(help="delta grid, e.g. '2,3,5,10' (default).")] = None,
) -> None:  # fmt: skip
    """Conflict scorer v2: fit the logistic model, the adaptive list cut and θ_conf on dev records.

    NESTED leave-one-seed-out CV: for each outer held-out dev seed, (C, label, top_k, delta,
    θ_conf) are chosen on the other four seeds only (inner leave-one-seed-out, train precision
    floor 0.52, a config is eligible when its worst inner fold's precision reaches the same floor,
    list cap <= 5); the report gives every outer fold, the worst fold, per subtype and the
    paraphrase rows, then the final fit on all five seeds. Printed and written (JSON + table).
    Target: worst outer fold surfacing >= 0.95 at precision >= 0.50 (exit 1 when missed: fix the
    scorer on dev, never on held-out data; the target is not relaxed). The options change the
    fit's rules (floors, grid), never the target; the fit records them."""
    from bench.synth import calibration

    recs = _conflict_records(conflicts)
    try:
        grid = calibration.make_grid(top_ks=[int(k) for k in _floats(top_ks, calibration.TOP_KS)],
                                     deltas=_floats(deltas, calibration.DELTAS))  # fmt: skip
        fit = calibration.fit_conf_model(recs, floor=floor, grid=grid, select_floor=select_floor)
    except ValueError as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None
    text = calibration.format_cv(fit)
    if out:
        out.write_text(json.dumps(fit, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        (report or out.with_suffix(".cv.txt")).write_text(text + "\n", encoding="utf-8")
    elif report:
        report.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    if not fit["feasible"]:
        n = fit["nested"]
        typer.echo(f"INFEASIBLE: nested CV worst outer fold surfacing {n['worst_surfacing']:.3f} /"
                   f" precision {n['worst_precision']:.3f} misses >= 0.95 / >= 0.50; fix the scorer"
                   " on dev first (no target relaxed)")  # fmt: skip
        raise typer.Exit(1)


@app.command("syn-fit-sweep")
def syn_fit_sweep(
    conflicts: Annotated[Path, typer.Argument(help="`syn-score-dev --conflicts-out` file.")],
    out_dir: Annotated[Path, typer.Option(help="theta_<variant>.json, .cv.txt and sweep.txt.")],
    variants: Annotated[str | None, typer.Option(
        help="Comma-separated variant names (default: all, in the pre-registered order).")] = None,
) -> None:  # fmt: skip
    """`syn-fit-theta` under each named variant of the fit rules, one table (MEM2-156 round 3).

    Every variant runs the same nested CV against the same target; the first feasible row in the
    pre-registered order (`base` first) is the one to gate with. A later row chosen because its
    outer folds pass is a choice made on the outer folds (optimistic): say so and confirm it on the
    live tune gate. Variants: base, select50, floor55, k3, d34, k3d34."""
    from bench.synth import calibration

    recs = _conflict_records(conflicts)
    names = [v.strip() for v in variants.split(",") if v.strip()] if variants else None
    try:
        rows = calibration.sweep(recs, names)
    except ValueError as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (out_dir / f"theta_{r['name']}.json").write_text(
            json.dumps(r["fit"], indent=1, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
        (out_dir / f"theta_{r['name']}.cv.txt").write_text(
            calibration.format_cv(r["fit"]) + "\n", encoding="utf-8")  # fmt: skip
    text = calibration.format_sweep(rows)
    (out_dir / "sweep.txt").write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    first = next((r["name"] for r in rows if r["fit"]["feasible"]), None)
    typer.echo(f"gate with: {out_dir / f'theta_{first}.json'}" if first
               else "no variant is feasible: the scorer needs a change, not a rule")  # fmt: skip


@app.command("syn-w3-diagnose")
def syn_w3_diagnose(
    conflicts: Annotated[Path | None, typer.Option(help="`syn-score-dev --conflicts-out`.")] = None,
    fit: Annotated[Path | None, typer.Option(help="`syn-fit-theta --out` JSON.")] = None,
    w3: Annotated[Path | None, typer.Option(help="B-W3 result JSON of the tune gate run.")] = None,
    r5: Annotated[Path | None, typer.Option(help="B-R5 result JSON of the same run.")] = None,
    size: Annotated[str, typer.Option()] = "M",
    out: Annotated[Path | None, typer.Option(help="Write the report as JSON.")] = None,
) -> None:
    """Read-only: why B-W3 misses (per subtype and cause), chain errors, B-R5 leakage joined.

    Dev data only: refuses seeds outside 1-5. Give what you have; a missing input drops its section."""
    from bench.synth import data as syn_data
    from bench.synth import w3_diagnose
    from bench.synth.generator import generate

    def load(p: Path | None) -> dict | None:
        return json.loads(p.read_text(encoding="utf-8")) if p else None

    def extra(rec: dict | None) -> dict | None:
        return None if rec is None else rec.get("extra", rec)

    recs = None
    if conflicts:
        recs = [json.loads(ln) for ln in conflicts.read_text(encoding="utf-8").splitlines()
                if ln.strip()]  # fmt: skip
    w3x, r5x = extra(load(w3)), extra(load(r5))
    seeds = {int(s) for x in (w3x, r5x) if x for s in (x.get("diagnosis") or x.get("leaked_queries")
                                                         or {})}  # fmt: skip
    if bad := sorted(
        s for s in seeds | {r["seed"] for r in recs or []} if s not in syn_data.DEV_SEEDS
    ):
        typer.echo(f"REFUSED: seeds {bad} are not dev seeds")
        raise typer.Exit(2)
    slots = {s: {int(k): v["slot"] for k, v in generate(s, size).expected["memories"].items()}
             for s in seeds}  # fmt: skip
    rep = w3_diagnose.report(recs, load(fit), w3x, r5x, slots)
    typer.echo(w3_diagnose.format_report(rep))
    if out:
        out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=dict) + "\n",
                       encoding="utf-8")  # fmt: skip


@app.command("merge-records")
def merge_records_cmd(
    paths: Annotated[list[Path], typer.Argument(help="Scored-record files, one fp_all.")],
    out: Annotated[Path, typer.Option()],
) -> None:
    """LME tune pools + SYN dev pools -> one file for `mem calibrate --records`."""
    from bench.synth import calibration

    typer.echo(f"wrote {out}: {calibration.merge_records(paths, out)} pools")


@app.command("read-gate-diagnose")
def read_gate_diagnose(
    records: Annotated[Path, typer.Argument(help="Scored pools (merge-records / syn-score-dev).")],
    gate: Annotated[Path | None, typer.Option(help="Gate JSON to judge (Platt or read model).")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the diagnosis as JSON.")] = None,
    examples: Annotated[int, typer.Option(help="Examples per cause.")] = 5,
) -> None:  # fmt: skip
    """Why current_recall_at_1 / as_of_accuracy miss (MEM2-155): per SYN dev current / as-of query,
    gold not in the pool (eligibility / retrieval), gold ranked below first (ranking), gold first
    but the gate abstains (the gate), or delivered first; with examples. Plus the AUROC of the top
    logit, cosine and p for answerable vs missing-slot. Reads records only: no model, dev data."""
    from mem import calibrate, readfit
    from mem.gate import Gate

    _, pools = calibrate.read_records(records)
    g = Gate(**json.loads(gate.read_text(encoding="utf-8"))) if gate else None
    d = readfit.diagnose(pools, g, examples)
    typer.echo(readfit.format_diagnosis(d))
    if out:
        out.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@app.command("package-gate")
def package_gate(
    gate_file: Annotated[Path, typer.Argument(help="The final gate JSON (with theta_conf).")],
    records: Annotated[
        Path, typer.Option(help="The LME tune records it was fitted on (mem calibrate --records-out).")
    ],
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json (sha256 is recorded).")],
    manifest: Annotated[
        Path, typer.Option(help="The frozen LME split; absent, it is derived from --lme-file.")
    ] = MANIFEST,
    out: Annotated[
        Path | None, typer.Option(help="Default: the file the wheel carries (src/mem/shipped/).")
    ] = None,
    syn_seeds: Annotated[str, typer.Option(help="SYN dev seeds θ_conf was fitted on.")] = "1,2,3,4,5",
    lme_version: Annotated[str, typer.Option(help="LongMemEval release name.")] = "cleaned",
) -> None:  # fmt: skip
    """Package a host-fitted gate as the serving gate that ships in the wheel, with its metadata.

    Refused (exit 1, nothing written) when the gate's fit saw any LME test question, when the
    records are not the gate's fit, or when a SYN seed is not a dev seed. The tune/test split is
    the frozen manifest when it is there (generated, untracked), else derived from --lme-file."""
    from bench import package_gate as pg
    from mem import shipped
    from mem.common import MemError

    try:
        seeds = [int(s) for s in syn_seeds.split(",")]
        meta = pg.package(gate_file, records=records, lme_file=lme_file, manifest=manifest,
                          dest=out or shipped.GATE_PATH, syn_seeds=seeds,
                          lme_version=lme_version)  # fmt: skip
    except (MemError, ValueError, OSError, TypeError) as err:
        typer.echo(f"REFUSED: {err}")
        raise typer.Exit(1) from err
    typer.echo(
        f"wrote {out or shipped.GATE_PATH}: fingerprint {meta['fingerprint'][:8]}, "
        f"{meta['tune_queries']} LME tune queries, SYN seeds {seeds}\n"
        f"split: {meta['split_source']}"
    )


@app.command("gate-add-theta")
def gate_add_theta(
    gate_json_path: Annotated[Path, typer.Argument(help="Gate JSON from `mem calibrate --out`.")],
    theta: Annotated[Path, typer.Argument(help="`bench syn-fit-theta --out` file.")],
    out: Annotated[Path, typer.Option(help="The gate with theta_conf set.")],
) -> None:
    """Put the fitted θ_conf and conf_model (coefficients, top_k, delta) into a gate file (every
    other field, incl. fp_all, unchanged). Refused when the nested CV fit is infeasible."""
    gate = json.loads(gate_json_path.read_text(encoding="utf-8"))
    fit = json.loads(theta.read_text(encoding="utf-8"))
    if not fit.get("feasible"):
        typer.echo("REFUSED: the θ_conf fit is infeasible (nested CV worst fold misses the target)")
        raise typer.Exit(2)
    if not fit.get("conf_model"):
        typer.echo("REFUSED: the fit has no conf_model coefficients (re-run bench syn-fit-theta)")
        raise typer.Exit(2)
    from mem.confmodel import ConfModel

    ConfModel.from_dict(fit["conf_model"])  # fingerprint check
    gate["theta_conf"], gate["conf_model"] = fit["theta_conf"], fit["conf_model"]
    out.write_text(json.dumps(gate, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}: theta_conf={fit['theta_conf']:g} "
               f"(surfacing recall {fit['surfacing_recall']:.3f}, "
               f"list precision {fit['list_precision']:.3f} on dev; conf_model "
               f"{fit['conf_model']['fingerprint'][:12]})")  # fmt: skip


@app.command("syn-fit-read-guard")
def syn_fit_read_guard(
    out: Annotated[Path, typer.Option(help="Write the fit as JSON (report next to it, .txt).")],
    gate: Annotated[Path | None, typer.Option(
        help="The P3 gate to complete: theta_conf + conf_model + read_model (gate-add-theta).")] = None,
    records_out: Annotated[Path | None, typer.Option(
        help="Write the per-probe records (delivered sets, scored pairs, gold validity).")] = None,
    from_records: Annotated[Path | None, typer.Option(
        help="Refit from a --records-out file (no replay; --gate still names theta_conf).")] = None,
    seeds: Annotated[str, typer.Option(help="Dev seeds only (1-5).")] = "1,2,3,4,5",
    size: Annotated[str, typer.Option()] = "M",
    config: Annotated[Path | None, typer.Option(help="Production config.yaml.")] = None,
    embedder: EmbedderOpt = "st",
    device: DeviceOpt = None,
    serve_port: ServePortOpt = None,
    margins: Annotated[str | None, typer.Option(help="Margin grid, e.g. '0,0.25,0.5,1'.")] = None,
) -> None:  # fmt: skip
    """Fit θ_read of the read-time conflict guard (ADR 0018) on the SYN DEV seeds, nested CV.

    Replays each dev seed under the gate's own conflict cut (missed updates leave stale members
    open, as in B-R5), reads every probe with the guard OFF, records the delivered set with its
    gold validity and every same-subject pair scored by the gate's conflict model, then fits
    θ_read: the smallest cut hiding no co-valid training pair (+ a margin chosen by inner CV), never
    below θ_conf. Feasible = every outer fold hides zero co-valid rows and reaches leakage 0; exit 1
    when missed (fix on dev; nothing is relaxed). Refuses held-out seeds."""
    import tempfile

    from bench.adapters.p3_ours import P3Backend
    from bench.suites.base import Context
    from bench.synth import data as syn_data
    from bench.synth import readguard_fit
    from bench.synth.run import SynSetup, replay_seed, require_p3

    if gate is None:
        typer.echo("REFUSED: --gate is required (the fit completes a gate with theta_conf)")
        raise typer.Exit(2)
    ids = tuple(int(x) for x in seeds.split(","))
    if bad := [s for s in ids if s not in syn_data.DEV_SEEDS]:
        typer.echo(f"REFUSED: {bad} are not dev seeds; only {syn_data.DEV_SEEDS} feed calibration")
        raise typer.Exit(2)
    grid = _floats(margins, readguard_fit.MARGINS)
    try:
        backend = P3Backend(config_path=config, gate_path=gate, models=embedder, device=device,
                            serve_port=serve_port, conflict_guard=False)  # fmt: skip
    except (RuntimeError, ValueError) as e:
        typer.echo(f"UNAVAILABLE: {e}")
        raise typer.Exit(2) from e
    g = backend.gate
    if g is None or g.theta_conf is None or g.model is None or g.read_model is None:
        backend.close()
        typer.echo("REFUSED: the gate needs theta_conf + conf_model (bench gate-add-theta) and a"
                   " read_model (mem calibrate --objective p3) before theta_read is fitted")  # fmt: skip
        raise typer.Exit(2)
    records: list[dict] = []
    if from_records is not None:
        backend.close()
        records = [json.loads(ln) for ln in from_records.read_text(encoding="utf-8").splitlines()
                   if ln.strip()]  # fmt: skip
    else:
        with tempfile.TemporaryDirectory(prefix="bench-syn-guard-") as tmp:
            ctx = Context(Path(tmp), backend.factory, None,
                          {"syn": SynSetup(ids, size, "tune", workdir=Path(tmp))})  # fmt: skip
            try:
                require_p3(ctx)
                for seed in ids:
                    run = replay_seed(ctx, seed, size)
                    client = backend.factory(run.db, scope="personal", tz=syn_data.TZ,
                                             now=run.syn.expected["end_now"])  # fmt: skip
                    try:
                        records += readguard_fit.probe_records(run, client)
                    finally:
                        client.close()
            except BackendUnavailable as e:
                typer.echo(f"UNAVAILABLE: {e}")
                raise typer.Exit(2) from e
            finally:
                backend.close()
        if records_out is not None:
            records_out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
                                   encoding="utf-8")  # fmt: skip
    try:
        fit = readguard_fit.fit_read_guard(records, g.theta_conf, g.model.fingerprint(), grid)
    except ValueError as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None
    fit["gate"] = {"fingerprint_hash": g.fingerprint_hash, "fit_id": g.fit_id,
                   "conf_model": g.model.fingerprint(), "size": size, "seeds": list(ids)}  # fmt: skip
    text = readguard_fit.format_report(fit)
    out.write_text(json.dumps(fit, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    if not fit["feasible"]:
        w = fit["nested"]["worst"]
        typer.echo(f"INFEASIBLE: nested CV worst outer fold hides {w['wrongly_hidden']} co-valid"
                   f" row(s) / leakage after {w['leakage_after']:.4f}; the read cut cannot reach"
                   " leakage 0 without hiding a co-valid memory on dev (no target relaxed)")  # fmt: skip
        raise typer.Exit(1)


@app.command("gate-add-read-guard")
def gate_add_read_guard(
    gate_json_path: Annotated[Path, typer.Argument(help="Gate JSON with theta_conf + conf_model.")],
    fit: Annotated[Path, typer.Argument(help="`bench syn-fit-read-guard --out` file.")],
    out: Annotated[Path, typer.Option(help="The gate with read_guard set.")],
) -> None:
    """Put the fitted θ_read (`read_guard`, fingerprinted against the gate's conf_model) into a gate
    file; every other field unchanged. Refused when the fit is infeasible, when the gate has no
    conf_model, or when the fit was made against another conf_model."""
    from mem.confmodel import ConfModel
    from mem.gate import Gate

    gate = json.loads(gate_json_path.read_text(encoding="utf-8"))
    f = json.loads(fit.read_text(encoding="utf-8"))
    if not f.get("feasible"):
        typer.echo("REFUSED: the theta_read fit is infeasible (a nested fold hides a co-valid"
                   " memory or leaks); fix on dev, never relax")  # fmt: skip
        raise typer.Exit(2)
    if not gate.get("conf_model"):
        typer.echo("REFUSED: the gate has no conf_model (bench gate-add-theta first)")
        raise typer.Exit(2)
    rg = f.get("read_guard") or {}
    if rg.get("conf_model_fingerprint") != ConfModel.from_dict(gate["conf_model"]).fingerprint():
        typer.echo("REFUSED: the fit was made against another conf_model; refit theta_read on this"
                   " gate (bench syn-fit-read-guard --gate)")  # fmt: skip
        raise typer.Exit(2)
    gate["read_guard"] = rg
    try:
        Gate(**{k: v for k, v in gate.items() if k in Gate.__dataclass_fields__})
    except ValueError as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None
    out.write_text(json.dumps(gate, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}: theta_read={rg['theta_read']:g} (nested worst fold: wrongly hidden"
               f" {f['nested']['worst']['wrongly_hidden']}, leakage after"
               f" {f['nested']['worst']['leakage_after']:.4f}; read_guard {rg['fingerprint'][:12]})")  # fmt: skip


@app.command("fit-tau-list")
def fit_tau_list(
    records_file: Annotated[Path, typer.Argument(help="LME tune pool records (`mem calibrate`).")],
    lme_file: Annotated[Path, typer.Option(help="longmemeval_s_cleaned.json (the query texts).")],
    gate: Annotated[Path, typer.Option(help="The gate whose Platt tau_list is a cut of.")],
    out: Annotated[Path, typer.Option(help="Write the fit as JSON (report next to it, .txt).")],
    cap: Annotated[int, typer.Option(help="List-mode token cap (ADR 0023).")] = 1000,
) -> None:
    """Fit tau_list of the list mode (ADR 0023) on LME TUNE by nested CV (MEM2-193).

    Reads only records, the gate's Platt and the question texts: no model. Refuses a pool whose
    question (its qid without the `:<kind>` suffix) is not in the manifest's tune split, and records whose fp_all is not the gate's
    fingerprint_hash. Prints the outer folds; exit 1 when an outer
    fold's training fit misses its floor (the fit file is written either way, `gate-add-tau-list`
    refuses an infeasible one)."""
    from mem import calibrate, listfit

    fp_all, pools = calibrate.read_records(records_file)
    g = json.loads(gate.read_text(encoding="utf-8"))
    if g.get("fingerprint_hash") != fp_all:
        typer.echo(f"REFUSED: the records' fp_all {fp_all[:12]} is not the gate's fingerprint_hash"
                   f" {str(g.get('fingerprint_hash'))[:12]}: tau_list, its cap and the Platt it"
                   " cuts go under one fingerprint (ADR 0023)")  # fmt: skip
        raise typer.Exit(2)
    questions = {q["question_id"]: q["question"] for q in lme.load_questions(lme_file)}
    split = {r.question_id: r.split for r in lme.read_manifest(MANIFEST)}
    bad = [p.qid for p in pools if split.get(listfit.question_of(p.qid)) != "tune"]
    if bad:
        typer.echo(f"REFUSED: {len(bad)} pool(s) are not LME tune questions (first: {bad[0]}):"
                   " tau_list is fitted on tune only")  # fmt: skip
        raise typer.Exit(2)
    lm, rep = listfit.fit_list_model(pools, questions, (g["platt_a"], g["platt_b"]), cap=cap)
    nested = lm["nested"]
    head = (
        f"tau_list {lm['tau_list']:.6f} on {rep['n']} aggregation questions"
        f" (floor {lm['floor']}, cap {cap} tokens; final fit is flat CV: optimistic)"
    )
    lines = [head,
             "outer fold  n_train  floor  tau_list   heldout coverage  tokens p95  truncated"]  # fmt: skip
    lines += [f"{f['fold']:>10}  {f['n_train']:>7}  {f['floor']:.2f}  {f['tau_list']:.6f}  "
              f"{f['heldout']['coverage']:>16.3f}  {f['heldout']['tokens_p95']:>10.0f}  "
              f"{f['heldout']['truncated']:.2f}" for f in nested["outer_folds"]]  # fmt: skip
    lines.append(f"mean held-out coverage {nested['mean_heldout_coverage']:.3f}, worst"
                 f" {nested['worst_heldout_coverage']:.3f}; feasible={nested['feasible']}")  # fmt: skip
    typer.echo("\n".join(lines))
    out.write_text(json.dumps({"feasible": nested["feasible"], "list_model": lm, "report": rep},
                              sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")  # fmt: skip
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not nested["feasible"]:
        raise typer.Exit(1)


@app.command("gate-add-tau-list")
def gate_add_tau_list(
    gate_json_path: Annotated[Path, typer.Argument(help="Gate JSON.")],
    fit: Annotated[Path, typer.Argument(help="`bench fit-tau-list --out` file.")],
    out: Annotated[Path, typer.Option(help="The gate with list_model set.")],
) -> None:
    """Put the fitted tau_list (`list_model`: cut, cap, floor, Platt, the nested outer folds, all
    under one fingerprint) into a gate file; every other field unchanged. Refused when the fit is
    infeasible or was made against another Platt."""
    from mem.gate import Gate

    gate = json.loads(gate_json_path.read_text(encoding="utf-8"))
    f = json.loads(fit.read_text(encoding="utf-8"))
    if not f.get("feasible"):
        typer.echo("REFUSED: the tau_list fit is infeasible (an outer fold misses its floor);"
                   " fix on tune, never relax")  # fmt: skip
        raise typer.Exit(2)
    gate["list_model"] = f["list_model"]
    try:
        Gate(**{k: v for k, v in gate.items() if k in Gate.__dataclass_fields__})
    except ValueError as e:
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from None
    out.write_text(json.dumps(gate, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    lm = f["list_model"]
    typer.echo(f"wrote {out}: tau_list={lm['tau_list']:.6f} (cap {lm['cap_tokens']} tokens, nested"
               f" mean held-out coverage {lm['nested']['mean_heldout_coverage']:.3f}; list_model"
               f" {lm['fingerprint'][:12]})")  # fmt: skip


@app.command("dump")
def dump_cmd(
    db: Annotated[Path, typer.Argument(help="Store file (opened read-only).", exists=True)],
    out: Annotated[Path, typer.Option(help="Where to write the canonical JSONL dump.")],
) -> None:
    """B-I1 canonical dump of a store, plus its dump and vector hashes (restore drill: `diff`)."""
    from bench.dump import canonical_dump, vectors_sha256

    data = canonical_dump(db)
    out.write_bytes(data)
    rows = len(data.splitlines())
    typer.echo(f"wrote {out}: {rows} rows  dump {hashlib.sha256(data).hexdigest()}  "
               f"vectors {vectors_sha256(db)}")  # fmt: skip
