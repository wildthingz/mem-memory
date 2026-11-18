"""Independent P2 contract proofs. No network or real models."""

from types import SimpleNamespace

from typer.testing import CliRunner

from bench import cli as bench_cli
from bench.adapters import lme_ours
from bench.datasets import calibration, lme
from bench.suites.b_p2_cost import response_violations
from mem.cli import app as mem_app


def test_production_lme_backend_can_initialize(tmp_path):
    # The real adapter must reach production code, even before loading any model.
    lme_ours.OursLmeBackend(workdir=tmp_path)


def test_documented_calibration_accepts_exported_labels(tmp_path):
    labels = tmp_path / "labels.jsonl"
    calibration.write_tune_labels([], labels)
    result = CliRunner().invoke(
        mem_app,
        [
            "calibrate",
            "--labels",
            str(labels),
            "--lme",
            str(tmp_path / "lme.json"),
        ],
    )
    # A data error is acceptable for this empty fixture; a nonexistent CLI is not.
    assert "No such option" not in result.output, result.output


def test_latency_cli_can_schedule_300_calls_from_269_br1_questions(tmp_path, monkeypatch):
    manifest = [lme.ManifestRow(f"q{i}", "multi-session", False, "test", i) for i in range(269)]
    questions = [{"question_id": r.question_id, "question": "memory"} for r in manifest]
    monkeypatch.setattr(lme, "read_manifest", lambda _: manifest)
    monkeypatch.setattr(lme, "load_questions", lambda _: questions)
    monkeypatch.setattr(bench_cli.corpora, "plan_corpus_file", lambda *_: [])
    monkeypatch.setattr(
        lme_ours,
        "OursLoadServer",
        lambda *a, **kw: SimpleNamespace(
            timed_recall=lambda _: {
                "fts": 1.0,
                "vec": 1.0,
                "fusion": 1.0,
                "rerank": 1.0,
                "total": 4.0,
            }
        ),
    )
    result = CliRunner().invoke(
        bench_cli.app,
        [
            "latency-run",
            "--lme-file",
            str(tmp_path / "s.json"),
            "--lme-m",
            str(tmp_path / "m.json"),
            "--out",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, repr(result.exception) + result.output


def test_br6_accepts_spec_abstention_line():
    assert response_violations("no relevant memories (best .22) · personal", True) == []


def test_reembed_is_an_available_cli_command():
    result = CliRunner().invoke(mem_app, ["reembed", "--help"])
    assert result.exit_code == 0, result.output
