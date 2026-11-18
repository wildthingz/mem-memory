"""`bench gate P4` rows, setup, freeze and looks, commands, the V5k load server (stub models)."""

import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import freeze, gates, i5, p4_setup
from bench.adapters import p4_load
from bench.cli import app
from bench.datasets import vault as vault_ds
from bench.gold import pg
from bench.suites import b_p4
from mem.testing import HashEmbedder, OverlapReranker
from tests.bench_t0 import fake_vault, pg_world, vault_world
from tests.bench_t0.test_p4_suites import make_renderer

SPEC_P4 = {  # benchmarks-v2 §4 P4 row, id -> metric -> (op, threshold); nothing else may gate
    ("B-I1", "diff_lines_fixture"): ("==", 0), ("B-I1", "diff_lines_snapshot"): ("==", 0),
    ("B-I2", "case_failures"): ("==", 0),
    ("B-I3", "mismatches_fixture"): ("==", 0), ("B-I3", "mismatches_snapshot"): ("==", 0),
    ("B-I4", "source_fidelity_failures"): ("==", 0), ("B-I4", "render_nondeterminism"): ("==", 0),
    ("B-I4", "render_unconfined_files"): ("==", 0),
    ("B-I5", "precision"): (">=", 0.98), ("B-I5", "rel_precision_min"): (">=", 0.95),
    ("B-I5", "unlabelled"): ("==", 0), ("B-I5", "sheet_matches_store"): (">=", 1),
    ("B-C4", "cold_s"): ("<=", 1200), ("B-C4", "warm_s"): ("<=", 10),
    ("B-C4", "warm_all_unchanged"): (">=", 1),
    ("B-R3", "stale_gold"): ("==", 0), ("B-R3", "coverage_any"): (">=", 0.85),
    ("B-R3", "far"): ("<=", 0.10), ("B-R3", "delivered_precision"): (">=", 0.70),
    ("B-R3", "ira"): ("<=", 0.5), ("B-R3", "abstention_recall"): (">=", 0.70),
    ("B-R3", "ndcg_std10_R"): (">=", 0.70),
    ("B-C1", "warm_p95_ms_median"): ("<=", 1500), ("B-C3", "warm_p95_ms_median"): ("<=", 1500),
    ("B-R1", "recall_any5_R"): (">=", 0.93), ("B-R1", "ndcg_std_superior"): (">=", 1),
    ("B-R1", "coverage_any"): (">=", 0.85), ("B-R1", "far"): ("<=", 0.05),
    ("B-R1", "delivered_precision"): (">=", 0.70), ("B-R1", "ira"): ("<=", 0.5),
    ("B-R2", "abstention_recall"): (">=", 0.85), ("B-R2", "far"): ("<=", 0.05),
    ("B-R2", "far_drift"): ("<=", 0.03),
}  # fmt: skip
RECORDED = {("B-R3", m) for m in ("delivered_f05", "ndcg_std10_path_qmd", "ndcg_std10_path_R",
                                  "dr4_reopen")}  # fmt: skip


def test_p4_rows_are_exactly_the_spec_thresholds_and_the_recorded_numbers():
    rows = {(g.id, g.metric): g for g in gates.load_gates()["P4"]}
    assert set(rows) == set(SPEC_P4) | RECORDED
    for key, (op, thr) in SPEC_P4.items():
        assert (rows[key].op, rows[key].threshold) == (op, thr), key
    for key in RECORDED:
        assert rows[key].op == "record" and rows[key].threshold is None
    assert all(g.frozen_at is None for g in rows.values())  # frozen only by `bench freeze P4`


def test_with_nothing_supplied_no_gate_passes(tmp_path):
    out = gates.run_phase("P4", lambda *a, **k: None, results_dir=tmp_path, options={})
    assert {o.status for o in out} <= {"UNAVAILABLE"} and len(out) == 11
    assert all(not o.passed for o in out)


# ---- a full setup with the stand-ins -------------------------------------------------------------


@pytest.fixture
def run(tmp_path):
    world = vault_world.make_world(tmp_path / "world")
    tar = tmp_path / "vault.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(world, arcname="vault")
    sha = vault_ds.sha256_file(tar)
    pg_tar, pg_sha, _ = pg_world.pg_snapshot(world, tmp_path)
    be = pg_world.backend(tmp_path / "gb", stores=("personal", "work"))
    be.import_vault(world, routes=vault_world.ROUTES)
    gold_dir = tmp_path / "gold"  # PG-v1 is built on the PG snapshot: no work notes
    pg.freeze(pg_world.build_questions(be), be.index, gold_dir, snapshot_sha256=pg_sha,
              snapshot_name=pg_tar.name, snapshot_date=pg_world.DATE,
              routes=vault_world.ROUTES)  # fmt: skip
    be.close()
    mini = vault_world.make_mini(tmp_path / "mini")
    life = tmp_path / "life.yaml"
    life.write_text(vault_world.LIFECYCLE)
    cfg = pg_world.backend(tmp_path / "cfgprobe").cfg
    r = p4_setup.build(
        snapshot=tar, snapshot_sha256=sha, pg_snapshot=pg_tar, pg_snapshot_sha256=pg_sha,
        gold_dir=gold_dir, split="tune", mini_vault=mini,
        mini_config=None, lifecycle=life, config_path=None, gate_path=None, embedder="hash",
        device=None, serve_port=None, render=make_renderer(), importer=fake_vault.run_import,
        loaded=(HashEmbedder(256), OverlapReranker()), cfg=cfg, gate=pg_world.open_gate(cfg),
        bootstrap_b=100)  # fmt: skip
    r.setup.routes = vault_world.ROUTES  # the mini-vault's override (client-x -> work)
    yield r, tmp_path, pg_tar, pg_sha
    r.close()


def test_build_checks_the_snapshot_sha_against_the_gold(tmp_path, run):
    r, tmp, tar, _ = run
    with pytest.raises(ValueError, match="sha256"):
        p4_setup.build(snapshot=None, snapshot_sha256=None, pg_snapshot=tar,
                       pg_snapshot_sha256="00" * 32, gold_dir=tmp / "gold", split="tune",
                       mini_vault=None, mini_config=None, lifecycle=None, config_path=None,
                       gate_path=None, embedder="hash", device=None, serve_port=None,
                       loaded=(HashEmbedder(256), OverlapReranker()), cfg=r.cfg)  # fmt: skip
    other = tmp / "other.tar"
    with tarfile.open(other, "w") as tf:
        tf.add(tmp / "mini", arcname="vault")
    with pytest.raises(ValueError, match="frozen PG-v1 was built on snapshot"):
        p4_setup.build(snapshot=None, snapshot_sha256=None, pg_snapshot=other,
                       pg_snapshot_sha256=None, gold_dir=tmp / "gold", split="tune",
                       mini_vault=None, mini_config=None, lifecycle=None, config_path=None,
                       gate_path=None, embedder="hash", device=None, serve_port=None,
                       loaded=(HashEmbedder(256), OverlapReranker()), cfg=r.cfg)  # fmt: skip


def test_the_import_gates_pass_on_the_standin_through_run_phase(tmp_path, run):
    r, tmp, _, _ = run
    out = gates.run_phase("P4", lambda *a, **k: None, results_dir=tmp / "res", split="tune",
                          options={b_p4.OPT: r.setup}, cfg=r.cfg,
                          only={"B-I1", "B-I2", "B-I3", "B-I4"})  # fmt: skip
    got = {o.id: (o.status, o.detail) for o in out}
    assert {k: v[0] for k, v in got.items()} == {"B-I1": "PASS", "B-I2": "PASS", "B-I3": "PASS",
                                                 "B-I4": "PASS"}, got  # fmt: skip
    rec = json.loads(next((tmp / "res").glob("*B-I3.json")).read_text())
    assert rec["metrics"]["mismatches_snapshot"] == 0 and rec["config"]["fp_all"] == r.cfg.fp_all()
    assert (tmp / "res" / (rec["run_id"] + ".items.jsonl")).exists()  # per-item results kept


def test_b_r3_row_set_reports_through_the_gate_with_the_record_rows(tmp_path, run):
    r, tmp, _, _ = run
    out = gates.run_phase("P4", lambda *a, **k: None, results_dir=tmp / "res", split="tune",
                          options={b_p4.OPT: r.setup}, cfg=r.cfg, only={"B-R3"})  # fmt: skip
    (o,) = out
    assert o.status in ("PASS", "FAIL") and "stale_gold=0" in o.detail
    # qmd was not run: its record row is "not produced", so the gate cannot pass without it
    assert o.status == "FAIL" and "ndcg_std10_path_qmd" in o.detail and "not produced" in o.detail
    rec = json.loads(next((tmp / "res").glob("*B-R3.json")).read_text())
    assert rec["dataset"]["name"] == "PG-v1" and rec["dataset"]["split"] == "tune"
    assert rec["dataset"]["pg_snapshot_sha256"] and rec["n_items"] == 40


# ---- freeze, looks, CLI ----------------------------------------------------------------------------


def test_freeze_p4_stamps_only_test_rows_and_a_test_look_is_logged(tmp_path):
    g = tmp_path / "gates.yaml"
    g.write_text(Path(gates.GATES_YAML).read_text())
    led = tmp_path / "ledger.jsonl"
    with pytest.raises(freeze.Refused, match="not frozen"):
        freeze.check_test_look_allowed("P4", g, led)
    freeze.freeze(
        "P4", who="t", reason="unchanged on tune", config_hash="x", gates_path=g, ledger=led
    )
    rows = {(s.id, s.metric): s for s in gates.load_gates(g)["P4"]}
    assert rows[("B-R3", "coverage_any")].frozen_at and rows[("B-C4", "cold_s")].frozen_at
    assert (
        rows[("B-I1", "diff_lines_fixture")].frozen_at is None
    )  # fixture rows are exact, never frozen
    assert rows[("B-R3", "coverage_any")].threshold == 0.85  # unchanged
    freeze.check_test_look_allowed("P4", g, led)


def test_gate_p4_test_look_needs_a_ledger_reason_and_frozen_thresholds():
    res = CliRunner().invoke(app, ["gate", "P4", "--split", "test"])
    assert res.exit_code == 2 and "needs --ledger" in res.output
    res = CliRunner().invoke(app, ["gate", "P4", "--split", "test", "--ledger", "why"])
    assert res.exit_code == 2 and "REFUSED" in res.output


# ---- commands ------------------------------------------------------------------------------------


def test_snapshot_is_verified_extracted_safely_and_round_trips(tmp_path):
    world = vault_world.make_world(tmp_path / "w")
    tar = tmp_path / "v.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(world, arcname="w")
    sha = vault_ds.sha256_file(tar)
    assert vault_ds.verify(tar, sha) == sha
    with pytest.raises(ValueError, match="sha256"):
        vault_ds.verify(tar, "0" * 64)
    root = vault_ds.extract(tar, tmp_path / "out")
    assert root.name == "w" and len(list(root.rglob("*.md"))) == sum(vault_world.FOLDERS.values())
    with pytest.raises(FileExistsError):
        vault_ds.extract(tar, tmp_path / "out")
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        tf.addfile(info)
    with pytest.raises(ValueError, match="unsafe"):
        vault_ds.extract(evil, tmp_path / "out2")


def test_gold_status_and_freeze_commands_on_a_worked_set(tmp_path, run):
    _, tmp, tar, _ = run
    # copy the work state of a finished PG build into a work dir: the frozen questions as accepted
    gold = pg.load(tmp / "gold")
    work = tmp / "work"
    work.mkdir()
    drafts, review, labels = [], [], []
    for i, q in enumerate(gold.questions):
        did = f"d{i:03d}"
        drafts.append({"draft_id": did, "unit": "u", "slot": 0, "category": q.category,
                       "question": q.question, "answer": q.answer, "gold": q.gold, "filters": {},
                       "key_terms": [], "source": q.source, "store": q.store})  # fmt: skip
        review.append({"draft_id": did, "action": "accept", "qid": q.qid, "question": q.question,
                       "answer": q.answer, "reworded": q.reworded})  # fmt: skip
        labels += [
            {"qid": q.qid, "gold_id": g, "rel": rel, "source": "draft"} for g, rel in q.gold.items()
        ]
    for name, rows in (("drafts", drafts), ("review", review), ("labels", labels)):
        (work / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    res = CliRunner().invoke(app, ["gold", "status", "--work", str(work)])
    assert res.exit_code == 0 and "accepted 120/120" in res.output
    stores = tmp / "gb"  # the bench-owned import the tool works on
    be = pg_world.backend(stores)
    be.import_vault(tmp / "world", routes={})
    be.close()
    out = tmp / "frozen2"
    res = CliRunner().invoke(app, ["gold", "freeze", "--snapshot", str(tar), "--snapshot-date",
                                   "2026-10-24", "--out", str(out), "--stores-dir", str(stores),
                                   "--work", str(work)])  # fmt: skip
    assert res.exit_code == 0, res.output
    assert f"qrels.tsv sha256 {vault_ds.sha256_file(out / 'qrels.tsv')}" in res.output
    again = CliRunner().invoke(app, ["gold", "freeze", "--snapshot", str(tar), "--snapshot-date",
                                     "2026-10-24", "--out", str(out), "--stores-dir", str(stores),
                                     "--work", str(work)])  # fmt: skip
    assert again.exit_code == 1 and "frozen" in again.output
    ver = CliRunner().invoke(
        app, ["gold", "verify", "--gold-dir", str(out), "--stores-dir", str(stores)]
    )
    assert ver.exit_code == 0 and "stale gold: 0" in ver.output


def test_i5_commands_sample_and_label(tmp_path, run):
    _, tmp, *_ = run
    be = pg_world.backend(tmp / "gb2")
    be.import_vault(tmp / "world", routes={})
    be.close()
    sheet = tmp / "i5.tsv"
    res = CliRunner().invoke(
        app, ["i5-sample", "--out", str(sheet), "--stores-dir", str(tmp / "gb2")]
    )
    assert (
        res.exit_code == 0
        and "edges" in res.output
        and len(i5.read_sheet(sheet)) == sum(vault_world.FOLDERS.values())
    )  # fewer than 100 live edges: all of them
    res = CliRunner().invoke(app, ["i5-label", str(sheet), "--minutes", "1"], input="y\ny\nn\nq\n")
    assert (
        res.exit_code == 0
        and f"labelled 3 now, {sum(vault_world.FOLDERS.values()) - 3} left" in res.output
    )


# ---- V5k / V50k load server ------------------------------------------------------------------------


def test_vault_load_server_builds_one_store_and_times_every_stage(tmp_path, run):
    r, tmp, tar, _ = run
    gold = pg.load(tmp / "gold")
    qs = sorted(gold.questions, key=lambda q: pg.sha1_key(q.qid))
    server = p4_load.VaultLoadServer(tar, gold, qs, "V5k", None, cfg=r.cfg, routes={},
                                     importer=fake_vault.run_import,
                                     loaded=(HashEmbedder(256), OverlapReranker()),
                                     gate=pg_world.open_gate(r.cfg))  # fmt: skip
    try:

        def count(name):  # sqlite connections are thread-bound: ask on the server's worker thread
            return server._worker.submit(
                lambda: (
                    server.backend.store(name)
                    .conn.execute("SELECT count(*) FROM chunks")
                    .fetchone()[0]
                )
            ).result()

        n_personal, n_work = count("personal"), count("work")
        assert n_work == 0 and n_personal > 100  # the whole vault in ONE store (decision 0012)
        stages = server.timed_recall(qs[0].question)
        assert {"eligible", "fts", "vec", "fusion", "rerank", "total"} <= set(stages)
        assert len(server.fingerprint) == 64
    finally:
        server.close()


def test_v50k_filler_fills_to_50k_minus_the_vault_and_is_never_gold(tmp_path):
    from bench.datasets import corpora

    with pytest.raises(ValueError, match="already has"):
        vault_ds.filler_plan(Path("unused"), 50_000)
    assert vault_ds.V50K_CHUNKS == 50_000 and corpora.SIZES["L50k"] == 50_000
    from bench.gold.chunks import FILLER_PREFIX

    assert FILLER_PREFIX == "lme:"  # `lme:load:<sid>` source refs, excluded from the gold index


def test_b_r1_and_b_r2_run_in_the_p4_gate_on_the_lme_setup(tmp_path):
    """ "B-R1/B-R2 no regression": the P2 forms run under `bench gate P4` with the P2 thresholds."""
    from tests.bench_t0.test_lme_gate import setup as lme_setup

    s = lme_setup("tune")
    out = gates.run_phase("P4", lambda *a, **k: None, results_dir=tmp_path, split="tune",
                          options={"lme": s}, only={"B-R1", "B-R2"})  # fmt: skip
    assert {o.id for o in out} == {"B-R1", "B-R2"} and all(o.status != "UNAVAILABLE" for o in out)
    assert "recall_any5_R" in out[0].detail and "abstention_recall" in out[1].detail


# ---- B-C3 at V5k: the `build=` override and `vault_write_server_build` -----------------------------


def test_write_server_build_override_replaces_the_plan_corpus_and_runs_on_the_worker():
    import threading
    from types import SimpleNamespace

    from bench.adapters.p3_ours import OursWriteServer

    seen = {}

    class Srv:
        def close(self):
            seen["closed"] = True

    srv = Srv()

    def build():
        seen["thread"] = threading.current_thread().name
        return srv

    syn = SimpleNamespace(ops=[{"tool": "remember"}, {"tool": "recall"}], expected={"entities": []})
    ws = OursWriteServer(None, syn, None, build=build)  # type: ignore[arg-type]  # no plan, no backend
    try:
        assert ws.server is srv and ws.n_ops() == 1
        assert seen["thread"].startswith("mem-serve")  # sqlite connections are worker-thread bound
    finally:
        ws.close()
    assert seen["closed"] is True

    def boom():
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):  # the temp db and worker are released
        OursWriteServer(None, syn, None, build=boom)  # type: ignore[arg-type]


def test_vault_write_server_build_fills_one_personal_store_from_the_snapshot(run):
    from concurrent.futures import ThreadPoolExecutor

    from bench.adapters.p3_ours import P3Backend

    r, tmp, tar, _ = run
    gold = pg.load(tmp / "gold")
    from tests.bench_t0.test_p3_real import stub_gate

    backend = P3Backend(models="hash", cfg=r.cfg, loaded=(HashEmbedder(256), OverlapReranker()))
    backend.gate = stub_gate(r.cfg)  # theta_conf + conf_model + read_model + read_guard (ADR 0018)
    build, info = p4_load.vault_write_server_build(backend, tar, gold, "V5k", None, {},
                                                   importer=fake_vault.run_import)  # fmt: skip
    assert info["fingerprint"] == ""  # set by the build, which runs lazily on the worker
    with ThreadPoolExecutor(1) as pool:

        def go():
            server = build()
            try:
                n = server.store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
                scope = server.store.scope
            finally:
                server.close()
            return n, scope

        n, scope = pool.submit(go).result()
    assert scope == "personal" and n > 100  # the whole vault in the one personal store
    assert len(info["fingerprint"]) == 64
    import hashlib

    assert info["fingerprint"] == hashlib.sha256(f"V5k:{n}".encode()).hexdigest()
