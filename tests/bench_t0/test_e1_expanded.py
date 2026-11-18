"""B-E1 mem arm (MEM2-200): the answerer reads recall's text and the full text of every delivered
row, not recall's index alone. Hash models, fake answerer and judge: no network, no real model."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import e1
from bench.adapters.lme_ours import OursLmeBackend
from bench.cli import app
from bench.datasets import lme
from bench.lme_eval import Retrieved
from mem.gate import Gate
from tests.bench_t0.test_e1 import QS, FakeLLM

S_FILE = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme" / "lme_s_mini.json"
OPEN_GATE = Gate(1.0, 0.0, 0.0, 0.0, 0.0)


def rounds_of(hay):
    return {
        (sid, r.index): "\n".join(t for t in (r.user, r.assistant) if t)
        for sid, _, rounds in hay.sessions
        for r in rounds
    }


def test_mem_context_is_recall_text_then_every_body():
    bodies = ["[session a, round 0, d]\nAAA", "[session b]\nBBB"]
    ctx = e1.mem_context(Retrieved([], [], [], "RECALL-INDEX", bodies=bodies))
    assert ctx.startswith("## mem recall\nRECALL-INDEX")
    assert "\n\n## full text of the rows shown\n" + "\n\n".join(bodies) in ctx


def test_an_empty_delivery_is_recall_text_only():
    r = Retrieved([], [], [], "no relevant memories (best .12) · personal")
    assert e1.mem_context(r) == r.text


def test_the_backend_delivers_the_text_of_exactly_the_delivered_rows(tmp_path):
    qs = lme.load_questions(S_FILE)
    backend = OursLmeBackend(tmp_path / "w", models="hash", gate=OPEN_GATE, expand=True)
    try:
        seen = 0
        for q in qs[:4]:
            hay = lme.haystack_of(q)
            dates = {sid: date for sid, date, _ in hay.sessions}
            r = backend.retrieve(q, hay, want_legs=False)
            texts = rounds_of(hay)
            assert len(r.bodies) == len(r.S) and r.truncated == []
            keys = [(x.partition("#")[0], int(x.partition("#")[2])) for x in r.S]
            for (sid, rnd), body in zip(keys, r.bodies, strict=True):
                assert body.startswith(f"[session {sid}, round {rnd}, {dates[sid]}]\n")
                assert texts[(sid, rnd)] in body  # none cut
            ctx = e1.mem_context(r)
            assert ctx.startswith("## mem recall\n" + r.text)
            for key in set(texts) - set(keys):
                assert texts[key] not in ctx
            seen += len(keys)
        assert seen  # the open gate delivered rows
        plain = OursLmeBackend(
            tmp_path / "p", models="hash", gate=OPEN_GATE, loaded=backend._loaded
        )
        assert plain.retrieve(qs[0], lme.haystack_of(qs[0]), want_legs=False).bodies == []
    finally:
        backend.close()


def test_none_and_oracle_prompts_do_not_change():
    q = QS[0]
    seen = []
    for cond in ("none", "oracle"):
        a = FakeLLM(lambda p: "x")
        j = FakeLLM(lambda p: "yes", usage={"input_tokens": 300, "output_tokens": 3})
        e1.run_e1([q], lambda q: "MEMCTX", a, j, conditions=(cond,), budget_usd=5.0)
        seen.append(a.prompts[0]["prompt"])
    assert seen[0] == e1.answer_prompt("none", q["question"], "", q["question_date"])
    assert seen[1] == e1.answer_prompt(
        "oracle", q["question"], e1.oracle_text(q), q["question_date"]
    )
    assert "MEMCTX" not in "".join(seen)


# ---- MEM2-202: the same holds through `bench e1-run`, split mode and `--full` --------------------


def expected_contexts(qs):
    """The mem context of each question, from a backend of its own (same hash models, open gate)."""
    from mem.config import stub_config
    from mem.testing import HashEmbedder, OverlapReranker

    b = OursLmeBackend(cfg=stub_config(), models="hash", gate=OPEN_GATE, expand=True,
                       loaded=(HashEmbedder(), OverlapReranker()))  # fmt: skip
    try:
        rs = {q["question_id"]: b.retrieve(q, lme.haystack_of(q), want_legs=False) for q in qs}
    finally:
        b.close()
    return rs


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    from dataclasses import asdict

    from bench import freeze as freeze_mod
    from bench import records
    from bench.datasets import fetch
    from mem.config import stub_config
    from mem.testing import HashEmbedder, OverlapReranker

    qs = lme.load_questions(S_FILE)[:3]
    llms = []

    def fake_llm(model, effort):
        llms.append(FakeLLM(lambda p: "yes"))
        return llms[-1]

    monkeypatch.setattr("bench.llm.AnthropicLLM", fake_llm)
    monkeypatch.setattr(lme, "load_questions", lambda f: qs)
    monkeypatch.setattr(lme, "read_manifest", lambda p: object())
    monkeypatch.setattr(lme, "verify_manifest", lambda q, m: None)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    rows = [type("R", (), {"question_id": q["question_id"]}) for q in qs]
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    monkeypatch.setattr(freeze_mod, "check_test_look_allowed", lambda phase: None)
    monkeypatch.setattr(fetch, "verify", lambda name, path: "0" * 64)
    monkeypatch.setattr(records, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr("bench.adapters.lme_ours.start_models",
                        lambda *a, **k: type("S", (), {"models": (HashEmbedder(), OverlapReranker()),
                                                      "stop": lambda self: None})())  # fmt: skip
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(asdict(Gate(1.0, 0.0, 0.0, 0.0, 0.0,
                                           fingerprint_hash=stub_config().fp_all()))))  # fmt: skip
    return qs, llms, gate, tmp_path / "run.json"


MODES = {"split": ["--split", "tune"],
         "full": ["--full", "--spend-look", "--ledger", "final run"]}  # fmt: skip


@pytest.mark.parametrize("mode", MODES)
def test_e1_run_gives_the_answerer_every_delivered_rows_text_by_default(cli_env, mode):
    qs, llms, gate, out = cli_env
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(out), "--yes",
                                 "--embedder", "hash", "--gate", str(gate), "--condition", "mem",
                                 "--budget-usd", "5", *MODES[mode]])  # fmt: skip
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["mem_context"] == "expanded" and rec["truncated_rows"] == {}
    answerer = llms[0]
    assert len(answerer.prompts) == len(qs)
    want = expected_contexts(qs)
    delivered = 0
    for q, p in zip(qs, answerer.prompts, strict=True):
        got = want[q["question_id"]]
        assert p["prompt"] == e1.answer_prompt(
            "mem", q["question"], e1.mem_context(got), q["question_date"]
        )
        texts = rounds_of(lme.haystack_of(q))
        for x in got.S:  # every delivered row's full text is in the answerer's prompt
            sid, _, rnd = x.partition("#")
            assert texts[(sid, int(rnd))] in p["prompt"]
        delivered += len(got.S)
    assert delivered


def test_e1_run_context_recall_is_the_index_alone(cli_env):
    _, llms, gate, out = cli_env
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(out), "--yes",
                                 "--embedder", "hash", "--gate", str(gate), "--condition", "mem",
                                 "--context", "recall"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text(encoding="utf-8"))["mem_context"] == "recall"
    assert not any(e1.EXPANDED_HEADING in p["prompt"] for p in llms[0].prompts)


def test_a_row_get_cuts_is_recorded_never_silent(tmp_path, monkeypatch):
    from bench.adapters import lme_ours

    monkeypatch.setattr(lme_ours, "EXPAND_TOKENS", 60)  # too small for the whole delivery
    qs = lme.load_questions(S_FILE)
    backend = OursLmeBackend(tmp_path / "w", models="hash", gate=OPEN_GATE, expand=True)
    try:
        r = next(
            r
            for q in qs
            if len((r := backend.retrieve(q, lme.haystack_of(q), want_legs=False)).S) > 1
        )
    finally:
        backend.close()
    assert r.truncated and all(c.startswith("c") for c in r.truncated)
