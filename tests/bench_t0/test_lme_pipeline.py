"""The real LME adapter on the real Store/write/read code, fake models (no network, no GPU).

The reviewer's standard (P2 review): labels -> scored tune records -> `mem calibrate` -> gate ->
B-R1 run through the production read path -> every real P2 response through B-R6. Plus a CLI
smoke test for each command of deliverables/runbooks/p2-gate.md that can run on fixtures.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import cli as bench_cli
from bench import latency, lme_eval
from bench.adapters.lme_ours import OursLmeBackend, OursLoadServer, ingest_haystack, load_config
from bench.datasets import corpora, lme, negatives
from bench.suites.b_p2_cost import response_violations
from mem import calibrate as calibrate_mod
from mem import read, write
from mem.cli import app as mem_app
from mem.common import MemError
from mem.config import stub_config
from mem.gate import Gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
S_FILE, M_FILE = FIX / "lme_s_mini.json", FIX / "lme_m_mini.json"
STUB = stub_config()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """The bench CLI's frozen manifest and negatives, built from the mini fixture."""
    qs = lme.load_questions(S_FILE)
    rows = lme.build_manifest(qs)
    for name in ("MANIFEST", "NEGATIVES", "AUDIT"):
        monkeypatch.setattr(bench_cli, name, tmp_path / getattr(bench_cli, name).name)
    lme.write_manifest(rows, bench_cli.MANIFEST)
    negatives.write_negatives(negatives.build_negatives(qs, rows), bench_cli.NEGATIVES)
    return qs, rows


@pytest.fixture
def calibrated(tmp_path, frozen, models_serve):
    """Runbook §4 on fixtures: `bench export-tune`, then `mem calibrate --labels --lme` scoring
    through a stub `mem serve --http --models-only` (D5)."""
    labels, gate, recs = (
        tmp_path / "tune_labels.jsonl",
        tmp_path / "gate.json",
        tmp_path / "rec.jsonl",
    )
    r = CliRunner().invoke(bench_cli.app, ["export-tune", str(S_FILE), "--out", str(labels)])
    assert r.exit_code == 0, r.output
    assert "8 tune queries (4 answerable)" in r.output
    r = CliRunner().invoke(mem_app, ["calibrate", "--labels", str(labels), "--lme", str(S_FILE),
                                     "--records-out", str(recs), "--out", str(gate),
                                     "--embedder", "hash", "--fit-id", "lme-mini-tune",
                                     "--port", str(models_serve)])  # fmt: skip
    assert r.exit_code == 0, r.output
    return labels, gate, recs


# ---- the mem helpers the adapter uses ----------------------------------------------------------


def test_ingest_session_writes_an_imported_chat_note_one_section_per_round(tmp_path):
    s = Store.open(tmp_path / "p.db", HashEmbedder(), tz="UTC")
    nid = write.ingest_session(s, source_ref="lme:q1:s1", date="2023-05-20T02:21:00Z",
                               rounds=[("hi there", "hello"), ("what about soccer", "")])  # fmt: skip
    node = s.conn.execute("SELECT kind, type, origin, valid_from, date, source_ref FROM nodes"
                          " WHERE id = ?", (nid,)).fetchone()  # fmt: skip
    assert node == ("note", "chat_session", "import", "2023-05-20T02:21:00Z", "2023-05-20",
                    "lme:q1:s1")  # fmt: skip
    chunks = s.conn.execute("SELECT chunk_key, key_kind, text FROM chunks WHERE node_id = ?"
                            " ORDER BY ord", (nid,)).fetchall()  # fmt: skip
    assert chunks == [("round/0", "section", "hi there\nhello"),
                      ("round/1", "section", "what about soccer")]  # fmt: skip
    (n_vec,) = s.conn.execute(f"SELECT count(*) FROM {s.vec_table}").fetchone()
    assert n_vec == 2
    with pytest.raises(MemError, match="already imported"):
        write.ingest_session(s, source_ref="lme:q1:s1", date="2023-05-20", rounds=[("a", "b")])
    s.close()


def test_recall_reports_stage_timings_and_a_size_override_extends_the_same_list(tmp_path):
    s = Store.open(tmp_path / "p.db", HashEmbedder(), tz="UTC", reranker=OverlapReranker())
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(70):
        write.ingest_session(s, source_ref=f"lme:x:{i}", date="2023-01-01",
                             rounds=[(f"soccer match {i}", "ok"), (f"soccer team {i}", "fine")])  # fmt: skip
    stages: dict[str, float] = {}
    read.recall(s, query="soccer", timings=stages)
    assert set(stages) == {"eligible", "fts", "vec", "fusion", "rerank", "total"}
    assert all(v >= 0 for v in stages.values()) and stages["total"] >= stages["rerank"]
    pool, _ = read.candidate_pool(s, "soccer", t=None)
    longer, _ = read.candidate_pool(s, "soccer", t=None, size=100)
    assert len(pool) == 30 and longer[:30] == pool and len(longer) > 30
    s.close()


def test_latency_schedule_cycles_the_questions_deterministically():
    calls = latency.schedule([f"q{i}" for i in range(269)])
    assert len(calls) == 1 + latency.WARMUP + latency.N_QUERIES == 321
    assert calls[:269] == [f"q{i}" for i in range(269)] and calls[269:] == [
        f"q{i}" for i in range(52)
    ]
    assert latency.schedule(["a"]) == ["a"] * 321
    with pytest.raises(ValueError):
        latency.schedule([])


# ---- the integration test -----------------------------------------------------------------------


def test_real_adapter_labels_to_gate_to_br1_responses_pass_br6(tmp_path, frozen, calibrated):
    qs, rows = frozen
    _labels, gate_file, recs = calibrated
    # the scored records: fp_all header, one pool per tune query with raw logits and labels
    fp_all, pools = calibrate_mod.read_records(recs)
    assert fp_all == STUB.fp_all() and len(pools) == 8
    assert sum(p.answerable for p in pools) == 4
    assert all(0 < len(p.items) <= 50 for p in pools)
    assert any(i.rel for p in pools if p.answerable for i in p.items)
    assert not any(i.rel for p in pools if not p.answerable for i in p.items)
    gate = Gate(**json.loads(gate_file.read_text()))
    assert gate.fingerprint_hash == STUB.fp_all() and gate.fit_id == "lme-mini-tune"
    assert gate.tune["queries"] == 8
    # refitting from the saved records reproduces the gate (fitting loads no model)
    r = CliRunner().invoke(mem_app, ["calibrate", "--records", str(recs), "--embedder", "hash",
                                     "--fit-id", "lme-mini-tune"])  # fmt: skip
    assert r.exit_code == 0, r.output
    refit = json.loads(r.output.strip().splitlines()[-1])  # the gate is the last line
    assert {k: v for k, v in refit.items() if k != "fitted_at"} == {
        k: v for k, v in json.loads(gate_file.read_text()).items() if k != "fitted_at"
    }

    backend = OursLmeBackend(tmp_path / "work", models="hash", gate_path=gate_file)
    items = lme_eval.run_lme(backend, {q["question_id"]: q for q in qs}, rows, "test")
    serve = backend.serve  # D5: every write and read went to the bench's own mem serve
    assert serve is not None and serve.running
    assert {i.kind for i in items} == {"answerable", "negative", "abs"}
    for it in items:
        assert it.r.text, it.qid
        assert response_violations(it.r.text, it.abstained) == [], (it.kind, it.qid, it.r.text)
        assert set(it.r.S) <= set(it.r.R[:30]) and len(it.r.F) <= 100
        assert sorted(it.r.R) == sorted(it.r.F)  # R_q = the pool reranked + the rest of F_q
        assert it.r.p_scores == sorted(it.r.p_scores, reverse=True)
    texts = [i.r.text for i in items]
    assert any(t.startswith("no relevant memories") for t in texts)  # real P2 abstention line
    assert any(" results (of " in t for t in texts)  # and real P2 rows
    ans = [i for i in items if i.kind == "answerable"]
    assert all(set(i.r.legs) == {"bm25", "dense", "rrf"} and i.r.legs["rrf"] == i.r.F for i in ans)
    sc = lme_eval.score(items, b=200)
    assert sc.n["answerable"] == 8 and 0 <= sc.metrics["recall_any5_R"] <= 1
    assert not list((tmp_path / "work").glob("*.db"))  # a fresh store per question, removed
    backend.close()
    assert not serve.running and backend.serve is None


def test_pools_scored_through_serve_equal_in_process_scoring(tmp_path, calibrated):
    """D5 moves the models, not the numbers: the served records are the in-process ones (the
    test hands the adapter in-process stand-ins; the adapter itself never constructs one)."""
    from bench.datasets import calibration

    labels, _, recs = calibrated
    _, served = calibrate_mod.read_records(recs)
    backend = OursLmeBackend(tmp_path / "local", models="hash",
                             loaded=(HashEmbedder(256), OverlapReranker()))  # fmt: skip
    rows = calibration.read_tune_labels(labels)
    local = calibration.score_tune(rows, {q["question_id"]: q for q in lme.load_questions(S_FILE)},
                                   backend)  # fmt: skip
    backend.close()
    assert served == local


def test_adapter_needs_a_gate_for_this_config_and_scores_without_one(tmp_path, calibrated):
    from bench.suites.base import BackendUnavailable

    _, gate_file, _ = calibrated
    q = lme.load_questions(S_FILE)[0]
    hay = lme.haystack_of(q)
    backend = OursLmeBackend(tmp_path, models="hash")
    with pytest.raises(BackendUnavailable, match="no gate"):
        backend.retrieve(q, hay, want_legs=False)
    scored = backend.score(q, hay)
    gold = lme.gold_rounds(q)
    assert scored and {r for r, *_ in scored} <= {lme.ref(s, rd.index) for s, _, rs in hay.sessions
                                                for rd in rs}  # fmt: skip
    assert gold & {r for r, *_ in scored}
    with pytest.raises(BackendUnavailable, match="fp_all"):
        OursLmeBackend(tmp_path, gate_path=gate_file)  # pinned config, stub-fitted gate


def test_ingest_haystack_maps_chunks_to_session_round_refs(tmp_path):
    q = lme.load_questions(S_FILE)[0]
    hay = lme.haystack_of(q)
    s = Store.open(tmp_path / "p.db", HashEmbedder(), tz="UTC")
    refs = ingest_haystack(s, q["question_id"], hay)
    assert sorted(refs.values()) == sorted(lme.ref(sid, r.index) for sid, _, rs in hay.sessions
                                           for r in rs)  # fmt: skip
    s.close()


def test_calibrate_refuses_bad_label_inputs(tmp_path, calibrated):
    labels, _, _ = calibrated
    run = lambda *a: CliRunner().invoke(mem_app, ["calibrate", "--embedder", "hash", *a])
    r = run("--labels", str(labels))
    assert r.exit_code != 0 and "--labels with --lme" in r.output
    other = tmp_path / "other.json"
    other.write_text(json.dumps([]))
    r = run("--labels", str(labels), "--lme", str(other))
    assert r.exit_code == 1 and "no question" in r.output
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    r = run("--labels", str(empty), "--lme", str(S_FILE))
    assert r.exit_code == 1 and "no tune queries" in r.output


# ---- CLI smoke: deliverables/runbooks/p2-gate.md, command by command ----------------------------


def test_runbook_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_cli.fetch, "fetch", lambda which, dest: {
        f"longmemeval_{w}_cleaned.json": "d" * 64 for w in which})  # fmt: skip
    r = CliRunner().invoke(bench_cli.app, ["fetch", "--which", "s", "--which", "m", "--dest",
                                           str(tmp_path)])  # fmt: skip
    assert r.exit_code == 0 and "longmemeval_m_cleaned.json" in r.output


def test_runbook_audit_apply(tmp_path, frozen):
    qs, _rows = frozen
    neg = negatives.read_negatives(bench_cli.NEGATIVES)
    negatives.write_audit_sheet(negatives.audit_sample(neg), qs, bench_cli.AUDIT)
    lines = bench_cli.AUDIT.read_text().splitlines()
    head = lines[0].split("\t")
    col = head.index("still_contains_answer")
    filled = [lines[0]] + ["\t".join(c if j != col else "n" for j, c in enumerate(ln.split("\t")))
                           for ln in lines[1:]]  # fmt: skip
    bench_cli.AUDIT.write_text("\n".join(filled) + "\n")
    r = CliRunner().invoke(bench_cli.app, ["audit-apply", str(bench_cli.AUDIT)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["dropped"] == 0


def test_runbook_build_corpus(monkeypatch):
    monkeypatch.setitem(corpora.SIZES, "L5k", 30)
    r = CliRunner().invoke(bench_cli.app, ["build-corpus", str(M_FILE), "--size", "L5k"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["chunks"] == 30


def _latency_runs(tmp_path, gate_file, n=3):
    outs = []
    for i in range(1, n + 1):
        out = tmp_path / "lat" / f"L5k-{i}.json"
        r = CliRunner().invoke(bench_cli.app, [
            "latency-run", "--lme-file", str(S_FILE), "--lme-m", str(M_FILE), "--corpus", "L5k",
            "--gate", str(gate_file), "--embedder", "hash", "--out", str(out)])  # fmt: skip
        assert r.exit_code == 0, r.output
        outs.append(out)
    return outs


def test_runbook_latency_run_real_server(tmp_path, monkeypatch, calibrated):
    monkeypatch.setitem(corpora.SIZES, "L5k", 30)
    _, gate_file, _ = calibrated
    (out,) = _latency_runs(tmp_path, gate_file, n=1)
    res = json.loads(out.read_text())
    assert res["n"] == 300 and res["schedule"]["calls"] == 321
    assert res["schedule"]["questions"] == len(lme.select(lme.read_manifest(bench_cli.MANIFEST),
                                                          "test", "br1"))  # fmt: skip
    # `guard` (ADR 0018) is recorded only when the read-time conflict guard runs: never on an LME
    # store (no memory pairs, a P2 gate without read_guard)
    assert set(res["one_client"]) == set(latency.STAGES) - {"guard"}
    assert res["config"]["fp_all"] == STUB.fp_all()
    r = CliRunner().invoke(bench_cli.app, [
        "latency-run", "--lme-file", str(S_FILE), "--lme-m", str(M_FILE), "--embedder", "hash",
        "--out", str(tmp_path / "x.json")])  # fmt: skip
    assert r.exit_code == 2 and "no gate" in r.output


def test_runbook_gate_tune_runs_every_p2_gate_on_the_real_adapter(
    tmp_path, monkeypatch, calibrated
):
    monkeypatch.setitem(corpora.SIZES, "L5k", 30)
    _, gate_file, _ = calibrated
    lat = _latency_runs(tmp_path, gate_file)
    results = tmp_path / "results"
    r = CliRunner().invoke(bench_cli.app, [
        "gate", "P2", "--split", "tune", "--lme-file", str(S_FILE), "--gate", str(gate_file),
        "--embedder", "hash", "--tokens", "est", "--results-dir", str(results),
        *[a for p in lat for a in ("--latency-run", str(p))]])  # fmt: skip
    assert "UNAVAILABLE" not in r.output, r.output
    assert "gates pass for P2 [P2 tune: informational, not a look]" in r.output
    for gid in ("B-R1", "B-R2", "B-R5", "B-R6", "B-C1", "B-C2"):
        assert any(ln.split()[1:2] == [gid] for ln in r.output.splitlines()), (gid, r.output)
    assert any(ln.startswith("PASS") and " B-R6 " in ln for ln in r.output.splitlines()), r.output
    rec = json.loads(next(results.glob("*-B-R1.json")).read_text())
    assert rec["config"]["fp_all"] == STUB.fp_all()
    assert rec["dataset"]["gate"]["fit_id"] == "lme-mini-tune"


def test_runbook_freeze_and_test_look_commands(tmp_path, monkeypatch):
    """§7 and §8 on a sandboxed gates.yaml and ledger (the real ones must not change)."""
    import functools
    import shutil

    from bench import freeze, gates, records

    real = {p: p.read_bytes() for p in (freeze.GATES_YAML, records.LEDGER)}
    y, led = tmp_path / "gates.yaml", tmp_path / "ledger.jsonl"
    shutil.copy(freeze.GATES_YAML, y)
    for mod, name, kw in ((freeze, "freeze", {"gates_path": y, "ledger": led}),
                          (freeze, "check_test_look_allowed", {"gates_path": y, "ledger": led}),
                          (gates, "run_phase", {"gates_path": y})):  # fmt: skip
        monkeypatch.setattr(mod, name, functools.partial(getattr(mod, name), **kw))
    append = records.append_ledger
    monkeypatch.setattr(records, "append_ledger", lambda entry, *_: append(entry, led))
    r = CliRunner().invoke(bench_cli.app, ["freeze", "P2", "--who", "t", "--reason", "unchanged"])
    assert r.exit_code == 0 and "frozen P2" in r.output, r.output
    r = CliRunner().invoke(bench_cli.app, ["gate", "P2", "--split", "test", "--who", "t",
                                           "--ledger", "look 1", "--only", "B-R5",
                                           "--results-dir", str(tmp_path / "res")])  # fmt: skip
    assert r.exit_code == 0 and "PASS" in r.output and "B-R5" in r.output, r.output
    assert [json.loads(ln).get("why") for ln in led.read_text().splitlines()][-1] == "look 1"
    assert all(p.read_bytes() == b for p, b in real.items())


def test_load_server_serialises_concurrent_clients(monkeypatch):
    plan = corpora.plan_corpus(lme.load_questions(M_FILE), 30)
    server = OursLoadServer(plan, cfg=STUB, models="hash", gate=Gate(1.0, 0.0, 0.0, 0.0, 0.0))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:
        out = list(pool.map(server.timed_recall, ["laptop"] * 12))
    assert all("total" in o for o in out)
    server.close()


def test_load_config_stub_and_pinned():
    assert load_config(None, "hash").fp_all() == STUB.fp_all()
    assert load_config(None, "st").fp_all() != STUB.fp_all()
    with pytest.raises(ValueError):
        load_config(None, "gpu")


# ---- MEM2-83 (F-P2b): what reaches the fit ------------------------------------------------------


def test_tune_records_label_has_answer_rounds_only_with_raw_logits_and_category(
    tmp_path, calibrated
):
    """Checks 1-2 of F-P2b on the real pipeline: `rel` is chunk-level (only the `has_answer`
    rounds; the evidence session's other rounds are 0), `z` is the reranker's raw logit for
    (question, ctx + "\n" + text), neither a sigmoid nor a rank, and every record carries its
    question type."""
    from bench.datasets import calibration

    labels, _, recs = calibrated
    _, pools = calibrate_mod.read_records(recs)
    qs = {q["question_id"]: q for q in lme.load_questions(S_FILE)}
    rows = {(r["qid"], r["kind"]): r for r in calibration.read_tune_labels(labels)}
    siblings_seen = 0
    for pool in pools:
        qid, kind = pool.qid.rsplit(":", 1)
        q, row = qs[qid], rows[(qid, kind)]
        assert pool.category == q["question_type"] == row["category"]
        keys = {i.key for i in pool.items}
        if kind == "negative":
            assert not pool.answerable and all(i.rel == 0 for i in pool.items)
            assert not {lme.session_of(k) for k in keys} & set(q["answer_session_ids"])
        else:
            gold = lme.gold_rounds(q)
            assert pool.answerable and {i.key for i in pool.items if i.rel >= 1} == gold & keys
            siblings = {k for k in keys if lme.session_of(k) in lme.gold_sessions(q)} - gold
            siblings_seen += len(siblings)
            assert all(i.rel == 0 for i in pool.items if i.key in siblings)
        # raw logits: the stub reranker's own score of (question, ctx + "\n" + text), per chunk
        s = Store.open(tmp_path / f"{pool.qid}.db", HashEmbedder(256), tz="UTC")
        refs = ingest_haystack(s, qid, calibration.label_haystack(q, row))
        by_ref = {ref: cid for cid, ref in refs.items()}
        docs = ["\n".join(s.conn.execute("SELECT coalesce(ctx, ''), text FROM chunks WHERE id = ?",
                                         (by_ref[i.key],)).fetchone()) for i in pool.items]  # fmt: skip
        s.close()
        assert [i.z for i in pool.items] == pytest.approx(
            OverlapReranker().score(q["question"], docs))  # fmt: skip
        assert any(i.z < 0 for i in pool.items) and all(-3 <= i.z <= 3 for i in pool.items)
    assert siblings_seen > 0  # the evidence sessions have non-answer rounds in the pool


def test_calibrate_cli_writes_the_tune_report(tmp_path, calibrated):
    _, gate_file, recs = calibrated
    report = tmp_path / "report.json"
    r = CliRunner().invoke(mem_app, ["calibrate", "--records", str(recs), "--embedder", "hash",
                                     "--fit-id", "lme-mini-tune", "--report", str(report)])  # fmt: skip
    assert r.exit_code == 0, r.output
    lines = r.output.strip().splitlines()
    gate = json.loads(lines[-1])
    assert gate["tau_keep"] == json.loads(gate_file.read_text())["tau_keep"]
    rep = json.loads(report.read_text())
    assert rep["queries"] == 8 and rep["choice"]["feasible"] in (True, False)
    assert rep["choice"]["feasible"] == gate["tune"]["feasible"]
    assert set(rep["choice"]["targets"]) == {"delivered_precision", "ira", "coverage_any",
                                             "abstention_recall", "far"}  # fmt: skip
    assert ("front" in gate["tune"]) == (not gate["tune"]["feasible"])
    cats = rep["per_category"]
    assert set(cats) == {
        "knowledge-update",
        "multi-session",
        "single-session-user",
        "temporal-reasoning",
    }  # the mini tune split; _abs never enters calibration
    assert all(c["n_answerable"] == 1 and c["n_negative"] == 1 for c in cats.values())
    assert rep["frontier"][0]["tau_keep"] == 0.0 and rep["inputs"]["n_items"] > 0
    # the human summary names the rule that chose the setting and lists every category
    text = "\n".join(lines[:-1])
    assert "tau_keep" in text and "multi-session" in text and "feasible:" in text
    # an infeasible fit says so on the line just before the gate line, and only there
    infeasible = not rep["choice"]["feasible"]
    assert infeasible == lines[-2].startswith("INFEASIBLE")
    assert sum("INFEASIBLE" in ln for ln in lines) == infeasible
    assert infeasible == any(ln.startswith("Pareto front") for ln in lines)
    assert gate["tune"]["per_category"] == cats


def test_pool_recall_and_br1_run_with_the_session_leg_on(tmp_path, frozen):
    """ADR 0022: with `retrieval.session_leg` on the same tooling runs end to end: scored tune
    pools -> `bench pool-recall` -> gate -> B-R1 with the session list reported next to the legs."""
    from bench.datasets import calibration
    from mem.calibrate import fit_gate, write_records

    qs, rows = frozen
    cfg_file = tmp_path / "leg.yaml"
    cfg_file.write_text("retrieval:\n  session_leg: true\n")
    loaded = (HashEmbedder(256), OverlapReranker())
    backend = OursLmeBackend(tmp_path / "w1", models="hash", loaded=loaded, config_path=cfg_file)
    assert backend.cfg.session_leg and backend.cfg.fp_all() != STUB.fp_all()
    labels = tmp_path / "labels.jsonl"
    r = CliRunner().invoke(bench_cli.app, ["export-tune", str(S_FILE), "--out", str(labels)])
    assert r.exit_code == 0, r.output
    pools = calibration.score_tune(calibration.read_tune_labels(labels),
                                   {q["question_id"]: q for q in qs}, backend)  # fmt: skip
    recs = tmp_path / "rec.jsonl"
    write_records(recs, pools, fp_all=backend.cfg.fp_all())
    r = CliRunner().invoke(bench_cli.app, ["pool-recall", "--records", str(recs)])
    assert r.exit_code == 0 and "recall" in r.output, r.output
    gate = fit_gate(pools, fit_id="leg", fingerprint_hash=backend.cfg.fp_all(),
                    limit=backend.cfg.retrieval["limit"], fitted_at="2026-10-26T00:00:00Z")  # fmt: skip
    backend.close()
    on = OursLmeBackend(tmp_path / "w2", models="hash", loaded=loaded, config_path=cfg_file,
                        gate=gate)  # fmt: skip
    items = lme_eval.run_lme(on, {q["question_id"]: q for q in qs}, rows, "test")
    ans = [i for i in items if i.kind == "answerable"]
    assert ans and all(set(i.r.legs) == {"bm25", "dense", "rrf", "session"} for i in ans)
    assert 0 <= lme_eval.score(items, b=200).metrics["recall_any5_R"] <= 1
    on.close()
