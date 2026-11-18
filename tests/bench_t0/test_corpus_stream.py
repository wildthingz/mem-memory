"""H2 (MEM2-75): the LongMemEval_M corpus plan is built by streaming the file, never loading it
whole, and equals the in-memory plan (sessions, rounds, truncation, fingerprint) on the same input."""

from __future__ import annotations

import json
import random
import tracemalloc
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import cli as bench_cli
from bench.datasets import corpora, lme

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
M_MINI = FIX / "lme_m_mini.json"


def _total_rounds(qs) -> int:
    return sum(len(lme.rounds_of(t)) for _, t in corpora.collect_sessions(qs).values())


def test_streamed_plan_equals_in_memory_plan_for_every_size():
    qs = lme.load_questions(M_MINI)
    total = _total_rounds(qs)
    assert total > 20
    truncated_seen = False
    for n in range(1, total + 1):
        old = corpora.plan_corpus(qs, n)
        new = corpora.plan_corpus_file(M_MINI, n)
        assert new == old, n  # session ids, dates, rounds and the truncated flag
        assert corpora.corpus_fingerprint(new) == corpora.corpus_fingerprint(old)
        truncated_seen |= new[-1].truncated
    assert truncated_seen
    with pytest.raises(ValueError, match="only"):
        corpora.plan_corpus_file(M_MINI, total + 1)


def test_streamed_plan_refuses_a_conflicting_duplicate_session(tmp_path):
    m = json.loads(M_MINI.read_text())
    seen: set[str] = set()
    for q in m:
        for si, sid in enumerate(q["haystack_session_ids"]):
            if sid in seen:
                q["haystack_sessions"][si] = [{"role": "user", "content": "different"}]
                f = tmp_path / "m.json"
                f.write_text(json.dumps(m))
                with pytest.raises(ValueError, match="different"):
                    corpora.plan_corpus_file(f, 5)
                return
            seen.add(sid)
    pytest.fail("fixture has no session shared by two haystacks")


def test_streamed_plan_refuses_duplicate_question_ids(tmp_path):
    m = json.loads(M_MINI.read_text())
    f = tmp_path / "m.json"
    f.write_text(json.dumps([*m, m[0]]))
    with pytest.raises(ValueError, match="duplicate question"):
        corpora.plan_corpus_file(f, 5)


@pytest.fixture
def no_whole_file_load(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("the M file was loaded whole")

    monkeypatch.setattr(lme, "load_questions", boom)
    monkeypatch.setattr(json, "loads", boom)
    monkeypatch.setattr(json, "load", boom)
    monkeypatch.setattr(Path, "read_text", boom)
    monkeypatch.setattr(Path, "read_bytes", boom)


def test_m_path_never_loads_the_whole_file(no_whole_file_load):
    plan = corpora.plan_corpus_file(M_MINI, 20)
    assert corpora.n_chunks(plan) == 20


def _build_corpus_cli(monkeypatch) -> dict:
    monkeypatch.setitem(corpora.SIZES, "L5k", 20)  # the mini file has < 5k rounds
    r = CliRunner().invoke(bench_cli.app, ["build-corpus", str(M_MINI), "--size", "L5k"])
    assert r.exit_code == 0, (r.output, r.exception)
    return json.JSONDecoder().decode(r.output)


def test_build_corpus_cli_never_loads_the_whole_file(monkeypatch, no_whole_file_load):
    assert _build_corpus_cli(monkeypatch)["chunks"] == 20


def test_build_corpus_cli_output_equals_the_in_memory_plan(monkeypatch):
    old = corpora.plan_corpus(lme.load_questions(M_MINI), 20)
    assert _build_corpus_cli(monkeypatch) == {
        "size": "L5k", "chunks": 20, "sessions": len(old), "truncated_last": old[-1].truncated,
        "fingerprint": corpora.corpus_fingerprint(old),
    }  # fmt: skip


def _write_m_layout(path: Path, n_questions: int, n_sessions: int, pool: int) -> None:
    """A multi-MB file in the official M layout, written incrementally (never held whole)."""
    rng = random.Random(3)
    words = [f"w{i}" for i in range(500)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("[")
        for q in range(n_questions):
            ids = sorted(rng.sample(range(pool), n_sessions))
            sessions = [
                [{"role": r, "content": " ".join(random.Random(i * 7 + k).choices(words, k=60))}
                 for k in range(6) for r in ("user", "assistant")]
                for i in ids
            ]  # fmt: skip
            f.write(("," if q else "") + json.dumps({
                "question_id": f"q{q}", "question_type": "multi-session", "question": "q",
                "answer": "a", "question_date": "2023/06/01 (Thu) 10:00",
                "haystack_session_ids": [f"s{i:05d}" for i in ids],
                "haystack_dates": [f"2023/04/{1 + i % 28:02d} (Sat) 08:00" for i in ids],
                "haystack_sessions": sessions, "answer_session_ids": [f"s{ids[0]:05d}"],
            }))  # fmt: skip
        f.write("]")


def test_streamed_plan_peak_memory_is_far_below_the_file_size(tmp_path):
    f = tmp_path / "m_big.json"
    _write_m_layout(f, n_questions=120, n_sessions=40, pool=1500)
    size = f.stat().st_size
    assert size > 8_000_000
    tracemalloc.start()
    try:
        plan = corpora.plan_corpus_file(f, 300)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert corpora.n_chunks(plan) == 300
    # One question (~0.1 MB here) plus per-session digests and the plan; a whole-file load
    # would be several times the file size.
    assert peak < size / 4, (peak, size)


def test_latency_run_streams_the_m_file(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from bench.adapters import lme_ours

    manifest = [lme.ManifestRow(f"q{i}", "multi-session", False, "test", i) for i in range(5)]
    s_questions = [{"question_id": r.question_id, "question": "memory"} for r in manifest]
    s_file = tmp_path / "s.json"

    def load_questions(path):
        assert Path(path) == s_file, f"the M file {path} was loaded whole"
        return s_questions

    seen = {}

    def server(plan, **_kw):
        seen["plan"] = plan
        return SimpleNamespace(timed_recall=lambda _: {"total": 1.0})

    monkeypatch.setattr(lme, "read_manifest", lambda _: manifest)
    monkeypatch.setattr(lme, "load_questions", load_questions)
    monkeypatch.setitem(corpora.SIZES, "L5k", 20)
    monkeypatch.setattr(lme_ours, "load_config", lambda *_a: {})
    monkeypatch.setattr(lme_ours, "OursLoadServer", server)
    r = CliRunner().invoke(bench_cli.app, [
        "latency-run", "--lme-file", str(s_file), "--lme-m", str(M_MINI),
        "--out", str(tmp_path / "out.json"), "--embedder", "hash",
    ])  # fmt: skip
    assert r.exit_code == 0, repr(r.exception) + r.output
    old = corpora.plan_corpus(json.loads(M_MINI.read_text()), 20)
    assert seen["plan"] == old
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["corpus_fingerprint"] == corpora.corpus_fingerprint(old)
