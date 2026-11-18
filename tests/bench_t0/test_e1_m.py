"""B-E1 on LongMemEval_M (`bench e1-run --full --lme-size M`, MEM2-220): a streaming, resumable mem-arm
runner. Tiny M-shaped fixture (about 40 sessions per question, a few hundred KB), hash models, fake
answerer and judge: no network, no real model."""

import json
import tracemalloc
from dataclasses import asdict

import pytest
from typer.testing import CliRunner

from bench import e1, e1_m, records
from bench import freeze as freeze_mod
from bench.adapters.lme_ours import OursLmeBackend, load_config
from bench.cli import app
from bench.datasets import fetch, lme
from bench.lme_eval import Retrieved
from mem.gate import Gate
from tests.bench_t0.test_e1 import TYPES, FakeLLM, make_q

OPEN_GATE = Gate(1.0, 0.0, 0.0, 0.0, 0.0)
N_Q, N_SESS, PAD = 8, 40, 4000  # questions, sessions per question, filler bytes per session


def m_question(i: int, qtype: str, qid: str | None = None) -> dict:
    q = make_q(qid or f"q{i}", qtype)
    q["haystack_session_ids"] = [f"q{i}-s{k}" for k in range(N_SESS)] + [f"q{i}-gold"]
    q["haystack_dates"] = [f"2023/01/{1 + k % 28:02d} (Sun) 09:00" for k in range(N_SESS + 1)]
    q["answer_session_ids"] = [f"q{i}-gold"]
    q["haystack_sessions"] = [
        [{"role": "user", "content": f"filler {k} of q{i} " + "lorem ipsum " * (PAD // 12)},
         {"role": "assistant", "content": "ok"}]
        for k in range(N_SESS)
    ] + [[{"role": "user", "content": f"GOLD-q{i} the answer is 42 apples", "has_answer": True},
          {"role": "assistant", "content": "noted"}]]  # fmt: skip
    return q


QS_M = [m_question(i, TYPES[i % len(TYPES)], f"q{i}_abs" if i == N_Q - 1 else None)
        for i in range(N_Q)]  # fmt: skip


@pytest.fixture
def m_file(tmp_path):
    path = tmp_path / "longmemeval_m_cleaned.json"
    path.write_text(json.dumps(QS_M), encoding="utf-8")
    return path


@pytest.fixture
def backend(tmp_path):
    b = OursLmeBackend(tmp_path / "w", models="hash", gate=OPEN_GATE, expand=True)
    yield b
    b.close()


META = {"answerer": "claude-sonnet-5-5", "judge": "claude-sonnet-5-5", "gate": {"sha256": "g" * 64},
        "config": {"fp_all": "f" * 64}}  # fmt: skip


def run_m(path, backend, tmp_path, *, answer=lambda p: "42 apples", judge=lambda p: "yes",
          meta=META, wrap=None, **kw):  # fmt: skip
    a, j = FakeLLM(answer), FakeLLM(judge, usage={"input_tokens": 300, "output_tokens": 3})

    def retrieve(q):
        return backend.retrieve(q, lme.haystack_of(q), want_legs=False)

    kw.setdefault("budget_usd", 5.0)
    rec = e1_m.run_m(path, wrap(retrieve) if wrap else retrieve, a, j, jsonl=tmp_path / "r.jsonl",
                     status=tmp_path / "r.STATUS", meta=meta, **kw)  # fmt: skip
    return rec, a, j


def lines(tmp_path):
    return [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]


# ---- streaming: the file is never held ----------------------------------------------------------


def test_the_file_is_streamed_never_loaded_whole(m_file, backend, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the whole M file was loaded")

    monkeypatch.setattr(lme, "load_questions", boom)
    real_read_text = type(m_file).read_text
    monkeypatch.setattr(type(m_file), "read_text", lambda self, *a, **k: (
        boom() if self == m_file else real_read_text(self, *a, **k)))  # fmt: skip
    monkeypatch.setattr(json, "load", boom)
    events: list[str] = []
    real = lme.iter_questions

    def spy(path):
        for q in real(path):
            events.append(f"read {q['question_id']}")
            yield q

    monkeypatch.setattr(lme, "iter_questions", spy)
    run_m(m_file, backend, tmp_path,
          wrap=lambda f: lambda q: (events.append(f"use {q['question_id']}"), f(q))[1])  # fmt: skip
    # one question is read, then used, before the next is read: interleaved, never read ahead
    ids = [q["question_id"] for q in QS_M]
    assert events == [e for i in ids for e in (f"read {i}", f"use {i}")]


def test_python_memory_stays_far_below_the_file_size(tmp_path):
    """The reader and the runner (no store here): about one question at a time, not the file."""
    m_file = tmp_path / "big.json"
    m_file.write_text(
        json.dumps([m_question(i, TYPES[i % 4]) for i in range(40)]), encoding="utf-8"
    )
    size = m_file.stat().st_size
    assert size > 1_000_000

    def retrieve(q):
        return Retrieved([], [], [], "no relevant memories")

    tracemalloc.start()
    try:
        a, j = FakeLLM(lambda p: "x"), FakeLLM(lambda p: "yes")
        e1_m.run_m(m_file, retrieve, a, j, jsonl=tmp_path / "r.jsonl", status=tmp_path / "s",
                   meta=META, budget_usd=5.0)  # fmt: skip
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < size / 3, (peak, size)  # one question is 1/40 of the file


# ---- resume: finished questions are skipped, the jsonl is appended to ----------------------------


class Crash(BaseException):
    pass


class Boom(Exception):
    pass


def crash_after(n):
    seen = []

    def wrap(f):
        def retrieve(q):
            if len(seen) == n:
                raise Crash
            seen.append(q["question_id"])
            return f(q)

        return retrieve

    return wrap, seen


def test_a_resumed_run_skips_finished_questions_and_appends(m_file, backend, tmp_path):
    wrap, _ = crash_after(3)
    with pytest.raises(Crash):
        run_m(m_file, backend, tmp_path, wrap=wrap)
    assert [x["qid"] for x in lines(tmp_path)] == ["q0", "q1", "q2"]
    st = json.loads((tmp_path / "r.STATUS").read_text())
    assert st["state"] == "crashed" and st["done"] == 3 and st["last_qid"] == "q2"

    redone: list[str] = []
    rec, a, _ = run_m(m_file, backend, tmp_path,
                      wrap=lambda f: lambda q: (redone.append(q["question_id"]), f(q))[1])  # fmt: skip
    assert redone == [q["question_id"] for q in QS_M[3:]]  # q0-q2 not retrieved again
    assert len(a.prompts) == N_Q - 3  # nor asked again, and not paid for again
    assert [x["qid"] for x in lines(tmp_path)] == [q["question_id"] for q in QS_M]
    assert rec["complete"] and rec["n"] == N_Q
    assert json.loads((tmp_path / "r.STATUS").read_text())["state"] == "complete"


def test_a_torn_last_line_is_redone_not_trusted(m_file, backend, tmp_path):
    run_m(m_file, backend, tmp_path, limit=2)
    with open(tmp_path / "r.jsonl", "a", encoding="utf-8") as f:
        f.write('{"qid": "q2", "corr')  # killed mid-write
    rec, a, _ = run_m(m_file, backend, tmp_path)
    assert [x["qid"] for x in lines(tmp_path)] == [q["question_id"] for q in QS_M]
    assert len(a.prompts) == N_Q - 2 and rec["complete"]


def test_resume_refuses_a_different_answerer_gate_or_fingerprint(m_file, backend, tmp_path):
    run_m(m_file, backend, tmp_path, limit=2)
    for key, val in (("answerer", "claude-haiku-4-5"), ("gate", {"sha256": "x" * 64}),
                     ("config", {"fp_all": "y" * 64})):  # fmt: skip
        with pytest.raises(e1_m.ResumeMismatch, match=key):
            run_m(m_file, backend, tmp_path, meta=META | {key: val})
    assert len(lines(tmp_path)) == 2  # nothing was appended


def test_the_budget_counts_what_earlier_sessions_spent(m_file, backend, tmp_path):
    first, _, _ = run_m(m_file, backend, tmp_path, limit=2)
    per_q = first["cost"]["usd"] / 2
    rec, _, _ = run_m(m_file, backend, tmp_path, budget_usd=per_q * 3.5)  # 1.5 questions left
    assert not rec["complete"] and rec["stopped"]
    assert rec["cost"]["usd"] <= per_q * 3.5
    assert json.loads((tmp_path / "r.STATUS").read_text())["state"] == "stopped"


# ---- the store is deleted per question ------------------------------------------------------------


def test_the_store_is_deleted_after_every_question(m_file, backend, tmp_path):
    left: list[list[str]] = []

    def wrap(f):
        def retrieve(q):
            out = f(q)
            left.append(sorted(p.name for p in backend._workdir.iterdir()))
            return out

        return retrieve

    run_m(m_file, backend, tmp_path, wrap=wrap)
    assert left == [[]] * N_Q  # no .db, -wal or -shm of a finished question remains
    assert list(backend._workdir.iterdir()) == []


# ---- the record -----------------------------------------------------------------------------------


def test_each_line_and_the_summary_carry_the_required_fields(m_file, backend, tmp_path):
    rec, _, _ = run_m(m_file, backend, tmp_path)
    rows = lines(tmp_path)
    assert len(rows) == N_Q
    for x in rows:
        assert x["condition"] == "mem" and x["answerer"] == "claude-sonnet-5-5"
        assert x["judge"] == "claude-sonnet-5-5" and x["prompt_version"] == 2
        assert x["prompts_sha256"] == e1.prompts_sha256()
        for k in ("qid", "type", "judge_kind", "correct", "judge_raw", "answer", "usd",
                  "delivered_tokens", "context_chars", "ingest_ms", "query_ms", "wall_s",
                  "n_sessions"):  # fmt: skip
            assert k in x, k
        assert x["ingest_ms"] > 0 and x["query_ms"] > 0 and x["n_sessions"] == N_SESS + 1
        assert x["delivered_tokens"] > 0
    assert rows[-1]["recall_any5"] is None  # the _abs question has nothing to find
    assert all(x["recall_any5"] in (0.0, 1.0) for x in rows[:-1])
    assert rows[0]["recall_any5"] == 1.0  # the open gate delivers the gold session
    assert rec["answerer"] == rows[0]["answerer"] and rec["prompt_version"] == 2
    assert rec["gate"] == META["gate"] and rec["config"] == META["config"]
    assert rec["prompts_sha256"] == e1.prompts_sha256()
    for k in ("accuracy", "accuracy_ci_lo", "accuracy_ci_hi", "per_type", "recall_any5",
              "delivered_tokens", "ingest_ms", "query_ms", "cost"):  # fmt: skip
        assert k in rec, k
    assert {*TYPES[:1], "abstention"} <= set(rec["per_type"])
    assert rec["accuracy"] == 1.0 and rec["n"] == N_Q and rec["conditions"] == ["mem"]
    assert rec["ingest_ms"]["mean"] > 0 and rec["query_ms"]["p95"] > 0


def test_the_summary_is_recomputable_from_the_jsonl_alone(m_file, backend, tmp_path):
    rec, _, _ = run_m(m_file, backend, tmp_path)
    again = e1_m.summarize(tmp_path / "r.jsonl", meta=META, planned=N_Q)
    assert again["accuracy"] == rec["accuracy"] and again["per_type"] == rec["per_type"]


# ---- the CLI: a look like S's full-500 mode ---------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch, tmp_path, m_file):
    monkeypatch.setattr(
        "bench.llm.AnthropicLLM",
        lambda model, effort: FakeLLM(
            lambda p: "yes" if "Is the model response correct" in p else "42 apples"
        ),
    )
    monkeypatch.setattr(freeze_mod, "check_test_look_allowed", lambda phase: None)
    monkeypatch.setattr(fetch, "verify", lambda name, path: "0" * 64)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(records, "LEDGER", led)
    cfg = load_config(None, "hash")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(asdict(Gate(1.0, 0.0, 0.0, 0.0, 0.0, fingerprint_hash=cfg.fp_all()))),
        encoding="utf-8",
    )
    return led, gate


def invoke_m(m_file, gate, out, *args):
    return CliRunner().invoke(app, [
        "e1-run", "--lme-file", str(m_file), "--lme-size", "M", "--full", "--out", str(out),
        "--embedder", "hash", "--gate", str(gate), *args])  # fmt: skip


def test_cli_refuses_without_spend_look_and_never_reaches_the_api(cli_env, m_file, tmp_path,
                                                                 monkeypatch):  # fmt: skip
    led, gate = cli_env
    monkeypatch.setattr("bench.llm.AnthropicLLM", lambda *a: pytest.fail("reached the API"))
    out = tmp_path / "m.json"
    r = invoke_m(m_file, gate, out, "--yes", "--ledger", "final M")
    assert r.exit_code == 2 and "--spend-look" in r.output and "test look" in r.output
    r = invoke_m(m_file, gate, out, "--spend-look", "--ledger", "final M")
    assert r.exit_code == 2 and "--yes" in r.output
    r = invoke_m(m_file, gate, out, "--yes", "--spend-look")
    assert r.exit_code == 2 and "--ledger" in r.output
    assert not led.exists() and not out.exists() and not out.with_suffix(".jsonl").exists()


def test_cli_m_is_the_mem_arm_of_the_whole_file_only(cli_env, m_file, tmp_path):
    _, gate = cli_env
    base = ("--yes", "--spend-look", "--ledger", "why")
    out = tmp_path / "m.json"
    for extra in (("--condition", "none"), ("--condition", "oracle"), ("--split", "tune"),
                  ("--context", "recall")):  # fmt: skip
        r = invoke_m(m_file, gate, out, *base, *extra)
        assert r.exit_code != 0 and not out.exists(), extra
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", str(m_file), "--lme-size", "M", "--out",
                                 str(out), *base])  # fmt: skip
    assert r.exit_code != 0 and "--full" in r.output


def test_cli_full_m_run_writes_jsonl_status_summary_and_a_ledger_line(cli_env, m_file, tmp_path):
    led, gate = cli_env
    out = tmp_path / "m.json"
    r = invoke_m(m_file, gate, out, "--yes", "--spend-look", "--ledger", "final M", "--who", "t",
                 "--budget-usd", "5")  # fmt: skip
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["n"] == N_Q and rec["complete"] and rec["split"] == "full" and rec["size"] == "M"
    assert rec["gate"]["sha256"] and rec["config"]["fp_all"] and rec["prompt_version"] == 2
    assert (
        rec["dataset"]["file"] == m_file.name and rec["dataset"]["name"] == "LongMemEval_M cleaned"
    )
    assert "S's questions" in rec["note"] and "25 Oct" in rec["note"]
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == N_Q
    assert json.loads((tmp_path / "m.STATUS").read_text())["state"] == "complete"
    (line,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert (line["phase"], line["split"], line["kind"], line["who"]) == (
        "P3",
        "test",
        "e1-run-m",
        "t",
    )
    assert line["why"] == "final M" and line["n_questions"] == N_Q
    assert "accuracy" in r.output


def test_cli_resumes_from_the_jsonl_after_a_kill(cli_env, m_file, tmp_path, monkeypatch):
    led, gate = cli_env
    out = tmp_path / "m.json"
    args = ("--yes", "--spend-look", "--ledger", "final M", "--budget-usd", "5")
    real = e1.run_e1
    n = []

    def killing(*a, **k):
        if len(n) == 2:
            raise Boom
        n.append(1)
        return real(*a, **k)

    monkeypatch.setattr(e1, "run_e1", killing)
    r = invoke_m(m_file, gate, out, *args)
    assert r.exit_code != 0 and not out.exists()
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 2
    monkeypatch.setattr(e1, "run_e1", real)
    r = invoke_m(m_file, gate, out, *args)
    assert r.exit_code == 0, r.output
    assert len((tmp_path / "m.jsonl").read_text().splitlines()) == N_Q
    assert json.loads(out.read_text())["n"] == N_Q
    kinds = [json.loads(x)["kind"] for x in led.read_text().splitlines()]
    assert kinds == ["e1-run-m", "e1-run-m"]  # each session that saw a test question is logged


def test_cli_limit_5_estimates_cost_and_host_hours(cli_env, m_file, tmp_path):
    led, gate = cli_env
    out = tmp_path / "est.json"
    r = invoke_m(m_file, gate, out, "--yes", "--limit", "5", "--ledger", "estimate")
    assert (
        r.exit_code == 2 and "--spend-look" in r.output
    )  # an estimate shows the answerer questions
    r = invoke_m(m_file, gate, out, "--yes", "--spend-look", "--limit", "5", "--ledger", "estimate",
                 "--who", "t")  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "ESTIMATE" in r.output and "500 questions" in r.output
    assert "$" in r.output and "host hours" in r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["n"] == 5 and rec["estimate"]["questions"] == 500
    assert rec["estimate"]["usd"] > 0 and rec["estimate"]["host_hours"] > 0
    (line,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert line["kind"] == "e1-run-m-estimate" and line["n_questions"] == 5


def test_the_runbook_lists_the_estimate_first_then_the_full_m_run():
    from pathlib import Path

    runbook = Path(__file__).resolve().parents[2] / "deliverables" / "runbooks" / "p3-gate.md"
    if not runbook.exists():
        pytest.skip("deliverables/ is not part of this tree")
    text = runbook.read_text()
    est = text.index("--lme-size M --full --spend-look --yes --limit 5")
    full = text.index("--lme-size M --full --spend-look --yes \\\n")
    assert est < full
    section = text[est - 2500 : full + 1500]
    for rule in ("MemAvailable", "MemoryMax=14G", "one model", "S's questions", "25 Oct"):
        assert rule in section, rule
