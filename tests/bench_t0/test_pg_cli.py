"""The `bench gold` commands the runbook names, through the CLI (scripted LLM, hash stand-ins)."""

import json

import pytest
from typer.testing import CliRunner

from bench import llm
from bench.cli import app
from bench.gold import kappa, pg, session
from tests.bench_t0 import pg_world
from tests.bench_t0.test_pg_tool import ScriptedLLM


@pytest.fixture
def paths(tmp_path):
    be, _ = pg_world.imported_world(tmp_path)
    index = be.index
    be.close()
    return tmp_path / "stores", tmp_path / "work", index


def run(*args, input=None):
    return CliRunner().invoke(app, ["gold", *args], input=input)


def test_draft_check_review_pool_label_status_commands(paths, monkeypatch):
    stores, work, index = paths
    monkeypatch.setattr(llm, "AnthropicLLM", lambda model, effort: ScriptedLLM(index))
    r = run(
        "draft", "--budget-usd", "50", "--yes", "--stores-dir", str(stores), "--work", str(work)
    )
    assert r.exit_code == 0, r.output
    assert "50 units, 200 drafts" in r.output and "200 drafts (0 unit(s) unparsable" in r.output
    assert "single-fact 50" in r.output and "no-answer-near-miss 35" in r.output
    r = run("check", "--stores-dir", str(stores), "--work", str(work))
    assert r.exit_code == 0 and "200 drafts checked" in r.output
    r = run("review", "--minutes", "5", "--snapshot-date", "2026-10-24", "--stores-dir", str(stores),
            "--work", str(work), input="a\n" * 400)  # fmt: skip
    assert r.exit_code == 0, r.output
    w = session.Work(work)
    assert sum(session.counts(w).values()) == 120  # every quota filled, no category over
    assert all(session.counts(w)[c] == pg.TOTAL[c] for c in pg.QUOTAS)
    r = run("pool", "--snapshot-date", "2026-10-24", "--embedder", "hash", "--stores-dir", str(stores),
            "--work", str(work))  # fmt: skip
    assert r.exit_code == 0 and "120 questions" in r.output and "OVER BUDGET" not in r.output
    monkeypatch.setattr(llm, "AnthropicLLM", lambda model, effort: pg_world.ScriptedJudge())
    r = run(
        "judge", "--budget-usd", "25", "--yes", "--stores-dir", str(stores), "--work", str(work)
    )
    assert r.exit_code == 0, r.output
    assert "still without a grade" in r.output and " 0 still without" in r.output
    assert w.judgments("claude") and not w.judgments("human", "pool")
    sample = w.kappa_sample(n=50, seed=kappa.SAMPLE_SEED)
    claude = w.judgments("claude", "pool")
    r = run("kappa-sample", "--stores-dir", str(stores), "--work", str(work),
            input="".join(f"{claude[k]}\n" for k in sample))  # fmt: skip
    assert r.exit_code == 0 and "n = 50" in r.output and "Claude's labels stand" in r.output
    r = run("label", "--stores-dir", str(stores), "--work", str(work))
    assert r.exit_code == 0 and "nothing for you to label" in r.output
    r = run("status", "--work", str(work))
    assert "accepted 120/120" in r.output and "pooled pairs" in r.output
    assert "labels by judge: claude" in r.output and "kappa (linear weights" in r.output


def test_draft_asks_before_sending_note_text_and_stops_at_the_budget(paths, monkeypatch):
    stores, work, index = paths
    monkeypatch.setattr(llm, "AnthropicLLM", lambda model, effort: ScriptedLLM(index))
    r = run(
        "draft", "--budget-usd", "50", "--stores-dir", str(stores), "--work", str(work), input="n\n"
    )
    assert r.exit_code == 1 and "sends the text of those notes" in r.output
    assert not (work / "drafts.jsonl").exists()
    r = run(
        "draft", "--budget-usd", "0.05", "--yes", "--stores-dir", str(stores), "--work", str(work)
    )
    assert r.exit_code == 1 and "STOPPED by the budget cap" in r.output


def test_draft_more_deals_other_notes_and_does_not_collide(paths, monkeypatch):
    stores, work, index = paths
    monkeypatch.setattr(llm, "AnthropicLLM", lambda model, effort: ScriptedLLM(index))
    for extra in ([], ["--more", "1"]):
        r = run("draft", "--budget-usd", "50", "--yes", "--stores-dir", str(stores), "--work",
                str(work), *extra)  # fmt: skip
        assert r.exit_code == 0, r.output
    ids = [json.loads(x)["draft_id"] for x in (work / "drafts.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 400 and any(".r1-" in i for i in ids)


def test_freeze_without_enough_questions_says_why_and_exits_1(paths, tmp_path):
    stores, work, _ = paths
    snap = tmp_path / "v.tar"
    snap.write_bytes(b"x")
    r = run("freeze", "--snapshot", str(snap), "--snapshot-date", "2026-10-24", "--out",
            str(tmp_path / "out"), "--stores-dir", str(stores), "--work", str(work))  # fmt: skip
    assert r.exit_code == 1 and "NOT FROZEN" in r.output and "quota needs exactly" in r.output
    assert not (tmp_path / "out" / "SNAPSHOT").exists()
