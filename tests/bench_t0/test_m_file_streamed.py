"""Every bench command that reads LongMemEval_M streams it: the whole file is never loaded
(c3-run was OOM-killed on the host at 12 GB when it was)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import cli as bench_cli
from bench.datasets import corpora, lme, vault
from bench.suites.base import BackendUnavailable

M_MINI = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme" / "lme_m_mini.json"


@pytest.fixture
def no_whole_m(monkeypatch):
    real = lme.load_questions

    def guard(path):
        if Path(path) == M_MINI:
            raise AssertionError("the M file was loaded whole")
        return real(path)

    monkeypatch.setattr(lme, "load_questions", guard)
    monkeypatch.setattr(
        corpora, "plan_corpus", lambda *a, **k: pytest.fail("plan_corpus needs the whole file")
    )


def test_c3_run_streams_m_and_plans_the_same_corpus(no_whole_m, monkeypatch, tmp_path):
    from bench.adapters import p3_ours

    got = {}

    class Stop:
        def __init__(self, plan, *a, **k):
            got["plan"] = plan
            raise BackendUnavailable("stop after planning")

    monkeypatch.setattr(p3_ours, "OursWriteServer", Stop)
    monkeypatch.setattr(p3_ours, "P3Backend", lambda **k: object())
    monkeypatch.setitem(corpora.SIZES, "L5k", 20)
    res = CliRunner().invoke(
        bench_cli.app, ["c3-run", "--lme-m", str(M_MINI), "--out", str(tmp_path / "o.json")]
    )
    assert res.exit_code == 2, res.output
    monkeypatch.undo()
    expected = corpora.plan_corpus(json.loads(M_MINI.read_text()), 20)
    assert got["plan"] == expected
    assert corpora.corpus_fingerprint(got["plan"]) == corpora.corpus_fingerprint(expected)


def test_build_corpus_streams_m(no_whole_m, monkeypatch):
    monkeypatch.setitem(corpora.SIZES, "L5k", 20)
    res = CliRunner().invoke(bench_cli.app, ["build-corpus", str(M_MINI), "--size", "L5k"])
    assert res.exit_code == 0, repr(res.exception) + res.output


def test_latency_run_streams_m(no_whole_m):
    import inspect

    assert "plan_corpus_file(lme_m" in inspect.getsource(bench_cli.latency_run)


def test_vault_commands_stream_m(no_whole_m):
    # vault-latency-run and vault-c3-run both fill V50k via vault.filler_plan
    plan = vault.filler_plan(M_MINI, 5, total=25)
    assert plan == corpora.plan_corpus_file(M_MINI, 20)
