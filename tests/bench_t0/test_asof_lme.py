"""MEM2-265: every LME path reads at the END of the question's day (ADR 0028).

LongMemEval stamps a question with a time of day, and some gold sessions of the same day are
stamped later (44/500 LME_S questions; no gold is on a later day). The harness passes
`as_of` = the question's day at 23:59:59Z, so a same-day-later session is eligible and a
next-day one is not. One synthetic question, every path that builds a store for it.
"""

import json
from pathlib import Path

import pytest

from bench.adapters import lme_ours
from bench.adapters.lme_ours import OursLmeBackend
from bench.datasets import calibration, lme
from mem import calibrate, models_http
from mem.config import stub_config
from mem.gate import Gate
from mem.testing import HashEmbedder, OverlapReranker
from tests.bench_t0.test_prf_check import stub_tagger

QDATE = "2023/05/20 (Sat) 02:21"
END_OF_DAY = "2023-05-20T23:59:59Z"
OPEN_GATE = Gate(1.0, 0.0, 0.0, 0.0, 0.0)


def _session(text, gold):
    return [{"role": "user", "content": text, "has_answer": gold},
            {"role": "assistant", "content": "Noted.", "has_answer": False}]  # fmt: skip


QUESTION = {
    "question_id": "asof_q1", "question_type": "temporal-reasoning",
    "question": "Which marathon did I run first, Berlin or Paris?", "answer": "Berlin",
    "question_date": QDATE,
    "haystack_session_ids": ["s_before", "s_same_day_later", "s_next_day", "s_filler"],
    "haystack_dates": ["2023/05/19 (Fri) 10:00", "2023/05/20 (Sat) 15:00",
                       "2023/05/21 (Sun) 01:00", "2023/05/18 (Thu) 09:00"],
    "haystack_sessions": [_session("I ran the Berlin marathon today.", True),
                          _session("I ran the Paris marathon today.", True),
                          _session("I ran the Rome marathon today.", False),
                          _session("I bought a new kettle for the kitchen.", False)],
    "answer_session_ids": ["s_before", "s_same_day_later"],
}  # fmt: skip


def test_lme_question_as_of_is_the_last_second_of_the_question_day():
    assert lme.question_as_of(QDATE) == END_OF_DAY
    assert lme.question_as_of("2023/05/20 (Sat) 23:59") == END_OF_DAY
    assert lme.question_as_of("2023/05/20 (Sat) 00:00") == END_OF_DAY


def test_session_dates_keep_their_time_of_day():
    assert lme.parse_lme_date("2023/05/20 (Sat) 15:00") == "2023-05-20T15:00:00Z"


def _backend(tmp_path, **kw):
    return OursLmeBackend(tmp_path / "w", cfg=stub_config(), models="hash", gate=OPEN_GATE,
                          loaded=(HashEmbedder(256), OverlapReranker()), **kw)  # fmt: skip


@pytest.fixture
def opened(monkeypatch):
    """Every `as_of` an LME path opens its fresh store with."""
    seen: list[str] = []
    real = OursLmeBackend._open

    def spy(self, name, as_of):
        seen.append(as_of)
        return real(self, name, as_of)

    monkeypatch.setattr(OursLmeBackend, "_open", spy)
    return seen


def _sessions(refs):
    return {lme.session_of(r) for r in refs}


def _score(tmp_path):  # `mem calibrate --labels` (tune scoring, `score_tune`)
    b = _backend(tmp_path)
    try:
        return _sessions(r for r, *_ in b.score(QUESTION, lme.haystack_of(QUESTION)))
    finally:
        b.close()


def _retrieve(tmp_path):  # B-R1, pool-recall, e1-run, --full, LME_M, delivered-dump
    b = _backend(tmp_path, follow_pages=True)
    try:
        return _sessions(b.retrieve(QUESTION, lme.haystack_of(QUESTION), want_legs=True).F)
    finally:
        b.close()


def _agentic(tmp_path):  # e1-agentic: the read tools on the store, no as_of in the call
    b = _backend(tmp_path)
    box = b.open_tools(QUESTION, lme.haystack_of(QUESTION))
    try:
        assert box.server.store.now_ts() == END_OF_DAY
        pool = calibrate.score_pool(box.server.store, QUESTION["question"])  # t = the store's now
        refs = lme_ours.chunk_refs(box.server.store)
        return _sessions(refs[int(i.key[1:])] for i in pool)
    finally:
        box.close()
        b.close()


def _prf(tmp_path, monkeypatch):  # bench prf-check
    cfg = stub_config()
    manifest = [lme.ManifestRow("asof_q1", "temporal-reasoning", False, "tune", 0)]
    rows = [r for r in calibration.tune_label_rows({"asof_q1": QUESTION}, manifest, [])
            if r["kind"] == "answerable"]  # fmt: skip
    labels, recs, lme_file = tmp_path / "l.jsonl", tmp_path / "r.jsonl", tmp_path / "lme.json"
    calibration.write_tune_labels(rows, labels)
    calibrate.write_records(recs, [calibrate.QueryPool("asof_q1:answerable", True, [])],
                            fp_all=cfg.fp_all())  # fmt: skip
    lme_file.write_text(json.dumps([QUESTION]), encoding="utf-8")
    monkeypatch.setattr(models_http, "connect", lambda port, c: (HashEmbedder(256),
                                                                 OverlapReranker()))  # fmt: skip
    from bench import prf_check

    out = prf_check.run(labels, lme_file, recs, cfg, 0, stub_tagger, min_chunks=1, size=8)
    return _sessions(out["rows"][0]["gained_plain"]) | _sessions(out["rows"][0]["gained"])


@pytest.mark.parametrize("path", ["score", "retrieve", "agentic", "prf"])
def test_every_lme_path_reads_at_the_end_of_the_question_day(tmp_path, monkeypatch, opened,
                                                             path):  # fmt: skip
    if path == "prf":
        got = _prf(tmp_path, monkeypatch)
    else:
        got = {"score": _score, "retrieve": _retrieve, "agentic": _agentic}[path](tmp_path)
    assert opened and set(opened) == {END_OF_DAY}, opened
    assert "s_same_day_later" in got  # gold stamped later on the question's day: eligible
    assert "s_next_day" not in got  # the next day: still not eligible


def test_tune_label_export_carries_the_end_of_day_as_of():
    manifest = [lme.ManifestRow("asof_q1", "temporal-reasoning", False, "tune", 0)]
    rows = calibration.tune_label_rows({"asof_q1": QUESTION}, manifest, [])
    assert {r["as_of"] for r in rows} == {END_OF_DAY}


def test_p4_direct_read_path_t_matches_recall_with_the_date_only_snapshot(tmp_path):
    """The P4 adapter passes its date-only snapshot to `recall(as_of=)` and uses `t` for the direct
    read-path calls (F_q, legs, calibration pools): both must be the same instant."""
    from bench.adapters.p4_ours import OursVaultBackend
    from mem.common import normalize_as_of

    b = OursVaultBackend(tmp_path, "2026-10-01", cfg=stub_config(), models="hash")
    assert b.t == normalize_as_of(b.snapshot_date, "UTC") == "2026-10-01T23:59:59Z"


def test_asof_report_counts_eligible_gold_before_and_after_per_tune_qid():
    from bench import asof_report

    other = dict(QUESTION, question_id="asof_q2", question_date="2023/05/22 (Mon) 09:00")
    manifest = [lme.ManifestRow("asof_q1", "temporal-reasoning", False, "tune", 0),
                lme.ManifestRow("asof_q2", "temporal-reasoning", False, "tune", 1),
                lme.ManifestRow("asof_q3", "temporal-reasoning", False, "test", 0)]  # fmt: skip
    s = asof_report.report({"asof_q1": QUESTION, "asof_q2": other}, manifest)  # q3 never read
    assert s["n"] == 2 and s["gold_changed"] == 1 and s["gold_later_day"] == 0
    (r,) = s["rows"]
    assert (r["qid"], r["gold"], r["gold_before"], r["gold_after"]) == ("asof_q1", 2, 1, 2)
    assert (r["hay_before"], r["hay_after"]) == (2, 3)  # s_next_day stays out
    assert "| asof_q1 | temporal-reasoning | 2 | 1 | 2 |" in asof_report.table(s)


def test_asof_eligibility_cli_on_the_mini_fixture(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from bench import cli as bench_cli

    s_file = Path(__file__).resolve().parents[2] / "bench/fixtures/lme/lme_s_mini.json"
    monkeypatch.setattr(bench_cli, "MANIFEST", tmp_path / "lme_split.tsv")
    lme.write_manifest(lme.build_manifest(lme.load_questions(s_file)), bench_cli.MANIFEST)
    out = tmp_path / "asof.json"
    r = CliRunner().invoke(bench_cli.app, ["asof-eligibility", "--lme-file", str(s_file),
                                           "--out", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "tune questions:" in r.output and "| qid |" in r.output
    assert json.loads(out.read_text())["split"] == "tune"
