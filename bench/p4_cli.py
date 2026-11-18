"""P4 commands of `bench` (benchmarks-v2 §4 P4 row); registered on the main app by `bench.cli`.

snapshot-make   tar the vault, print the sha256 to record
vault-import    verify + extract the snapshot, import it into the bench's own stores
pg-score-tune   score the PG tune pools for `mem calibrate` (re-calibration with PG tune)
i5-sample / i5-label   B-I5 (100 edges, ~30 min)
c4-run          B-C4: cold and warm import on this host
qmd-run / rg-run   the descriptive baselines of B-R3
vault-latency-run / vault-c3-run   B-C1 / B-C3 at V5k and V50k
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from bench import baselines_p4, i5, records
from bench import p4_setup as setup_mod
from bench.datasets import vault as vault_ds
from bench.gold import pg

EmbedderOpt = Annotated[str, typer.Option(help="st = pinned models; hash = test stand-ins.")]
DeviceOpt = Annotated[str | None, typer.Option(help="auto, cpu or cuda (default: config).")]
PortOpt = Annotated[int | None, typer.Option(help="Port of the bench's own mem serve.")]
ConfigOpt = Annotated[Path | None, typer.Option(help="Production config.yaml (vault.routes).")]


def register(app: typer.Typer) -> None:
    @app.command("snapshot-make")
    def snapshot_make(
        vault: Annotated[Path, typer.Option(help="Your vault folder, e.g. ~/.obsidian")],
        out: Annotated[Path, typer.Option(help="vault-YYYY-MM-DD.tar.zst")],
        pg_only: Annotated[
            bool,
            typer.Option(
                "--pg",
                help="The PG snapshot: personal/ and projects/ only, "
                "minus every folder routed to work (no work notes).",
            ),
        ] = False,
        config: ConfigOpt = None,
    ) -> None:
        """Tar the vault (sorted members, no .git/.trash) and print the sha256 to record. With
        `--pg`, tar only the files PG-v1 may use (name it `pgvault-YYYY-MM-DD.tar.zst`)."""
        if pg_only:
            if config is None:
                raise typer.BadParameter("--pg needs --config (your vault.routes): the work-routed "
                                         "folders under personal/ and projects/ are left out")  # fmt: skip
            sha, kept, dropped = vault_ds.make_pg_snapshot(
                vault, out, setup_mod.routes_from_config(config)
            )
            typer.echo(f"{out}\nPG snapshot: {kept} files from personal/ and projects/; "
                       f"{dropped} work-routed file(s) left out\nsha256 {sha}\nrecord it: echo "
                       f"'{sha}  {out.name}' > bench/fixtures/{out.name.split('.')[0]}.sha256")  # fmt: skip
            return
        sha = vault_ds.make_snapshot(vault, out)
        typer.echo(f"{out}\nsha256 {sha}\nrecord it: echo '{sha}  {out.name}' > "
                   f"bench/fixtures/{out.name.split('.')[0]}.sha256")  # fmt: skip

    @app.command("vault-import")
    def vault_import(
        snapshot: Annotated[Path, typer.Option(help="vault-YYYY-MM-DD.tar.zst")],
        sha256: Annotated[str, typer.Option(help="The recorded snapshot sha256.")],
        dest: Annotated[Path, typer.Option(help="Gets snapshot/ and stores/.")] = Path(
            "data/vault"
        ),
        snapshot_date: Annotated[str, typer.Option(help="YYYY-MM-DD of the snapshot.")] = "",
        config: ConfigOpt = None,
        embedder: EmbedderOpt = "st",
        device: DeviceOpt = None,
        serve_port: PortOpt = None,
    ) -> None:
        """Verify, extract, and import the snapshot into the bench's own personal.db / work.db."""
        from bench.adapters.lme_ours import load_config
        from bench.adapters.p4_ours import OursVaultBackend

        vault_ds.verify(snapshot, sha256)
        root = vault_ds.extract(snapshot, dest / "snapshot")
        cfg = load_config(config, embedder)
        be = OursVaultBackend(dest / "stores", snapshot_date or time.strftime("%Y-%m-%d"), cfg=cfg,
                              config_path=config, models=embedder, device=device,
                              serve_port=serve_port)  # fmt: skip
        try:
            t0 = time.perf_counter()
            rep = be.import_vault(root, routes=setup_mod.routes_from_config(config))
            typer.echo(f"imported {root} -> {dest / 'stores'} in {time.perf_counter() - t0:.0f} s: "
                       f"{json.dumps(rep)}; chunks {len(be.index)}")  # fmt: skip
        finally:
            be.close()

    @app.command("pg-score-tune")
    def pg_score_tune(
        out: Annotated[Path, typer.Option(help="Scored pools (mem calibrate --records format).")],
        gold_dir: Annotated[Path, typer.Option()] = setup_mod.GOLD_DIR,
        stores_dir: Annotated[Path, typer.Option()] = Path("data/pg/stores"),
        config: ConfigOpt = None,
        embedder: EmbedderOpt = "st",
        device: DeviceOpt = None,
        serve_port: PortOpt = None,
    ) -> None:
        """PG-v1 TUNE questions -> rerank pools with raw logits and rel labels (A4: re-calibrate
        with PG tune). Test questions are never scored here."""
        from bench.adapters.lme_ours import load_config
        from bench.adapters.p4_ours import OursVaultBackend
        from mem import calibrate

        gold = pg.load(gold_dir)
        cfg = load_config(config, embedder)
        be = OursVaultBackend(stores_dir, gold.snapshot["snapshot_date"], cfg=cfg, config_path=config,
                              models=embedder, device=device, serve_port=serve_port)  # fmt: skip
        try:
            stale = pg.stale_gold(gold, be.index)
            if stale:
                typer.echo(f"STALE GOLD ({len(stale)}): {stale[0]}")
                raise typer.Exit(1)
            pools = [calibrate.QueryPool(
                q.qid, q.answerable,
                [calibrate.PoolItem(g, z, q.gold.get(g, 0)) for g, z in be.calibration_pool(q)],
                q.category) for q in sorted(gold.of_split("tune"), key=lambda q: q.qid)]  # fmt: skip
        finally:
            be.close()
        calibrate.write_records(out, pools, fp_all=cfg.fp_all())
        typer.echo(f"wrote {out}: {len(pools)} tune pools "
                   f"({sum(p.answerable for p in pools)} answerable)")  # fmt: skip

    @app.command("i5-sample")
    def i5_sample(
        out: Annotated[Path, typer.Option(help="The B-I5 sheet (TSV).")],
        stores_dir: Annotated[Path, typer.Option()] = Path("data/vault/stores"),
    ) -> None:
        """Sample 100 live import edges, stratified by rel, seed 7."""
        from bench.gold.cli import open_conns

        rows = i5.sample(open_conns(stores_dir))
        i5.write_sheet(rows, out)
        by_rel: dict[str, int] = {}
        for r in rows:
            by_rel[r["rel"]] = by_rel.get(r["rel"], 0) + 1
        typer.echo(f"wrote {out}: {len(rows)} edges {json.dumps(by_rel)}")

    @app.command("i5-label")
    def i5_label(
        sheet: Annotated[Path, typer.Argument(help="Sheet from i5-sample.")],
        minutes: Annotated[float, typer.Option(help="Stop after this long.")] = 35,
    ) -> None:
        """Label edges y / n (does it mean what its rel says?). Resumable."""
        done, left = i5.label_sitting(sheet, minutes=minutes)
        typer.echo(f"labelled {done} now, {left} left")

    @app.command("c4-run")
    def c4_run(
        snapshot: Annotated[Path, typer.Option()],
        sha256: Annotated[str, typer.Option()],
        out: Annotated[Path, typer.Option(help="Record JSON for B-C4.")],
        config: ConfigOpt = None,
        embedder: EmbedderOpt = "st",
        device: DeviceOpt = None,
        serve_port: PortOpt = None,
    ) -> None:
        """B-C4: cold full import with embeddings, then the warm no-op re-import, into fresh stores."""
        import tempfile

        from bench.adapters.lme_ours import load_config
        from bench.adapters.p4_ours import OursVaultBackend
        from bench.suites import b_p4

        vault_ds.verify(snapshot, sha256)
        cfg = load_config(config, embedder)
        with tempfile.TemporaryDirectory(prefix="bench-c4-") as tmp:
            root = vault_ds.extract(snapshot, Path(tmp) / "snapshot")
            be = OursVaultBackend(Path(tmp) / "stores", time.strftime("%Y-%m-%d"), cfg=cfg,
                                  config_path=config, models=embedder, device=device,
                                  serve_port=serve_port)  # fmt: skip
            try:
                res = b_p4.c4_measure(
                    be, root, time.perf_counter, setup_mod.routes_from_config(config)
                )
            finally:
                be.close()
        res |= {"device": device or cfg.device, "config": records.config_fingerprint(cfg),
                "hardware": records.hardware(), "git_sha": records.git_sha(),
                "snapshot_sha256": sha256, "bench_version": records.BENCH_VERSION}  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}: cold {res['cold_s']:.0f} s, warm {res['warm_s']:.1f} s, "
                   f"warm all unchanged: {res['warm_all_unchanged']}")  # fmt: skip

    @app.command("qmd-run")
    def qmd_run(
        vault_dir: Annotated[
            Path, typer.Option(help="The extracted snapshot (data/vault/snapshot).")
        ],
        out: Annotated[Path, typer.Option()],
        gold_dir: Annotated[Path, typer.Option()] = setup_mod.GOLD_DIR,
    ) -> None:
        """qmd (hybrid query) over the snapshot, top 10 files per PG question (descriptive)."""
        gold = pg.load(gold_dir)
        rec = baselines_p4.qmd_run(gold.questions, vault_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}: {len(rec['runs'])} questions, qmd {rec['version']}")

    @app.command("rg-run")
    def rg_run(
        vault_dir: Annotated[Path, typer.Option()],
        out: Annotated[Path, typer.Option()],
        gold_dir: Annotated[Path, typer.Option()] = setup_mod.GOLD_DIR,
    ) -> None:
        """`rg -il` file ranking floor, top 10 files per PG question (descriptive)."""
        gold = pg.load(gold_dir)
        rec = baselines_p4.rg_run(gold.questions, vault_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}: {len(rec['runs'])} questions, {rec['version']}")

    @app.command("vault-latency-run")
    def vault_latency_run(
        snapshot: Annotated[Path, typer.Option()],
        sha256: Annotated[str, typer.Option()],
        out: Annotated[Path, typer.Option(help="Output JSON for this process run.")],
        corpus: Annotated[str, typer.Option(help="V5k or V50k")] = "V5k",
        lme_m: Annotated[Path | None, typer.Option(help="V50k: longmemeval_m_cleaned.json")] = None,
        gold_dir: Annotated[Path, typer.Option()] = setup_mod.GOLD_DIR,
        gate_file: Annotated[Path | None, typer.Option("--gate")] = None,
        config: ConfigOpt = None,
        embedder: EmbedderOpt = "st",
        device: DeviceOpt = None,
        serve_port: PortOpt = None,
    ) -> None:
        """One fresh-process B-C1 run at V5k / V50k: PG questions (sha1 order, cycled), 1 cold + 20
        warm-up + 300 (1 client) + 300 (4 clients). The whole vault is ONE store (decision 0012)."""
        from bench import latency
        from bench.adapters.lme_ours import load_config
        from bench.adapters.p4_load import VaultLoadServer

        if corpus not in ("V5k", "V50k"):
            raise typer.BadParameter("corpus must be V5k or V50k")
        if corpus == "V50k" and lme_m is None:
            raise typer.BadParameter("V50k needs --lme-m")
        vault_ds.verify(snapshot, sha256)
        gold = pg.load(gold_dir)
        qs = sorted(gold.questions, key=lambda q: pg.sha1_key(q.qid))
        queries = latency.schedule([q.question for q in qs])
        cfg = load_config(config, embedder)
        server = VaultLoadServer(snapshot, gold, qs, corpus, lme_m, cfg=cfg, config_path=config,
                                 gate_path=gate_file, models=embedder, device=device,
                                 serve_port=serve_port, routes=setup_mod.routes_from_config(config))  # fmt: skip
        try:
            res = latency.run_process(server.timed_recall, queries)
            fingerprint = server.fingerprint
        finally:
            server.close()
        res |= {"corpus": corpus, "corpus_fingerprint": fingerprint,
                "schedule": {"questions": len(qs), "calls": len(queries),
                             "rule": "PG-v1 questions in sha1 order, call i = question i mod n"},
                "config": records.config_fingerprint(cfg), "hardware": records.hardware(),
                "git_sha": records.git_sha(), "bench_version": records.BENCH_VERSION}  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}: cold {res['cold_ms']:.0f} ms, "
                   f"warm p95 {res['one_client']['total']['p95']:.0f} ms")  # fmt: skip

    @app.command("vault-c3-run")
    def vault_c3_run(
        snapshot: Annotated[Path, typer.Option()],
        sha256: Annotated[str, typer.Option()],
        out: Annotated[Path, typer.Option()],
        corpus: Annotated[str, typer.Option()] = "V5k",
        lme_m: Annotated[Path | None, typer.Option()] = None,
        seed: Annotated[int, typer.Option(help="SYN-M seed whose remember ops are replayed")] = 101,
        gold_dir: Annotated[Path, typer.Option()] = setup_mod.GOLD_DIR,
        gate_file: Annotated[Path | None, typer.Option("--gate")] = None,
        config: ConfigOpt = None,
        embedder: EmbedderOpt = "st",
        device: DeviceOpt = None,
        serve_port: PortOpt = None,
    ) -> None:
        """One fresh-process B-C3 run at V5k / V50k: 1 cold + 20 warm-up + 300 + 300 `remember`."""
        from bench import latency
        from bench.adapters.p3_ours import OursWriteServer, P3Backend
        from bench.adapters.p4_load import vault_write_server_build
        from bench.synth.generator import generate

        if corpus not in ("V5k", "V50k") or (corpus == "V50k" and lme_m is None):
            raise typer.BadParameter("corpus must be V5k, or V50k with --lme-m")
        vault_ds.verify(snapshot, sha256)
        gold = pg.load(gold_dir)
        syn = generate(seed, "M")
        backend = P3Backend(config_path=config, gate_path=gate_file, models=embedder, device=device,
                            serve_port=serve_port)  # fmt: skip
        build, info = vault_write_server_build(backend, snapshot, gold, corpus, lme_m,
                                               setup_mod.routes_from_config(config))  # fmt: skip
        server = OursWriteServer(None, syn, backend, build=build)
        try:
            need = 1 + latency.WARMUP + 2 * latency.N_QUERIES
            if server.n_ops() < need:
                typer.echo(f"SYN M seed {seed} has {server.n_ops()} remember ops, need {need}")
                raise typer.Exit(2)
            res = latency.run_write_process(server.timed_remember, [str(i) for i in range(need)])
        finally:
            server.close()
            backend.close()
        res |= {"corpus": corpus, "corpus_fingerprint": info["fingerprint"],
                "schedule": {"syn_seed": seed, "ops": need}, "config": records.config_fingerprint(
                    backend.cfg), "hardware": records.hardware(), "git_sha": records.git_sha(),
                "bench_version": records.BENCH_VERSION}  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}: cold {res['cold_ms']:.0f} ms, "
                   f"warm p95 {res['one_client']['total']['p95']:.0f} ms")  # fmt: skip
