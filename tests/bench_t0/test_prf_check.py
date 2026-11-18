"""`bench prf-check` (MEM2-216): per-type share of tune answerables whose gold enters the 30 -> 40 pool."""

import json
from pathlib import Path

from typer.testing import CliRunner

from bench import cli as bench_cli
from bench import prf_check
from bench.adapters.lme_ours import OursLmeBackend
from bench.datasets import calibration, lme, negatives
from mem import calibrate as calibrate_mod
from mem import prf
from mem.config import stub_config

FIX = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme"
S_FILE = FIX / "lme_s_mini.json"


def stub_tagger(text: str) -> prf.Tagged:
    words = [w.strip(".,?") for w in text.split()]
    return prf.Tagged([w.lower() for w in words if w.isalpha() and len(w) >= 4], [])


def row(qid, typ, gained):
    return {"qid": qid, "category": typ, "question": f"q {qid}", "gained": gained, "terms": ["t"]}


def test_summary_counts_the_share_per_type_and_overall_with_the_verdict():
    rows = [row("a", "x", ["s#0"]), row("b", "x", []), row("c", "y", []), row("d", "y", [])]
    s = prf_check.summarise(rows, threshold=0.25)
    assert (s["n"], s["gained"], s["share"]) == (4, 1, 0.25)
    assert s["by_type"] == {"x": {"n": 2, "gained": 1, "share": 0.5},
                            "y": {"n": 2, "gained": 0, "share": 0.0}}  # fmt: skip
    assert s["dropped"] is False  # 0.25 is not below the 0.25 threshold
    assert prf_check.summarise(rows, threshold=0.3)["dropped"] is True


def test_summary_examples_are_the_first_gaining_queries_in_qid_order():
    rows = [row(q, "x", ["s#0"]) for q in "dcba"]
    ex = prf_check.summarise(rows, threshold=0.05, examples=2)["examples"]
    assert [e["qid"] for e in ex] == ["a", "b"]


def test_verdict_line_names_c_dropped_below_the_threshold():
    below = prf_check.summarise([row("a", "x", []), row("b", "x", [])], threshold=0.05)
    assert prf_check.verdict(below).startswith("C is DROPPED: 0/2")
    above = prf_check.summarise([row("a", "x", ["s#0"])], threshold=0.05)
    assert prf_check.verdict(above).startswith("C is NOT dropped: 1/1")


def _backend():
    return OursLmeBackend(cfg=stub_config(), models="hash")


def _question_and_hay():
    qs = lme.load_questions(S_FILE)
    q = next(q for q in qs if q["question_type"] == "knowledge-update")
    return q, lme.haystack_of(q)


def test_probe_gains_a_gold_chunk_the_first_pass_pool_left_out():
    q, hay = _question_and_hay()
    b = _backend()
    try:
        db, store = b._open("probe", lme.parse_lme_date(q["question_date"]))
        refs = prf_check.ingest(store, q, hay)
        some = sorted(refs.values())
        # a before-pool of one chunk, with every other chunk "gold": the 2nd pass must add some
        out = prf_check.probe(store, refs, q["question"], lme.parse_lme_date(q["question_date"]),
                              before=some[:1], gold=set(some[1:]), tagger=stub_tagger,
                              top=1, min_chunks=1, size=4)  # fmt: skip
        assert out["gained"] and set(out["gained"]) <= set(some[1:])
        assert set(out["after"]) >= set(some[:1]) or len(out["after"]) == 4
        assert out["terms"]
    finally:
        store.close()
        db.unlink(missing_ok=True)


def test_probe_gain_excludes_gold_already_in_the_before_pool():
    q, hay = _question_and_hay()
    b = _backend()
    try:
        db, store = b._open("probe2", lme.parse_lme_date(q["question_date"]))
        refs = prf_check.ingest(store, q, hay)
        everything = sorted(refs.values())
        out = prf_check.probe(store, refs, q["question"], lme.parse_lme_date(q["question_date"]),
                              before=everything, gold=set(everything), tagger=stub_tagger,
                              top=3, min_chunks=1, size=len(everything))  # fmt: skip
        assert out["gained"] == []
    finally:
        store.close()
        db.unlink(missing_ok=True)


def test_cli_end_to_end_on_the_mini_fixture(tmp_path, monkeypatch, models_serve):
    qs = lme.load_questions(S_FILE)
    rows_m = lme.build_manifest(qs)
    for name in ("MANIFEST", "NEGATIVES", "AUDIT"):
        monkeypatch.setattr(bench_cli, name, tmp_path / getattr(bench_cli, name).name)
    lme.write_manifest(rows_m, bench_cli.MANIFEST)
    negatives.write_negatives(negatives.build_negatives(qs, rows_m), bench_cli.NEGATIVES)
    labels, recs, out = tmp_path / "labels.jsonl", tmp_path / "rec.jsonl", tmp_path / "prf.json"
    r = CliRunner().invoke(bench_cli.app, ["export-tune", str(S_FILE), "--out", str(labels)])
    assert r.exit_code == 0, r.output
    cfg = stub_config()
    b = OursLmeBackend(cfg=cfg, models="hash")
    pools = calibration.score_tune(calibration.read_tune_labels(labels),
                                   {q["question_id"]: q for q in qs}, b)  # fmt: skip
    calibrate_mod.write_records(recs, pools, fp_all=cfg.fp_all())
    monkeypatch.setattr(prf, "spacy_tagger", lambda: stub_tagger)
    r = CliRunner().invoke(bench_cli.app, ["prf-check", "--labels", str(labels), "--lme",
                                           str(S_FILE), "--records", str(recs), "--out", str(out),
                                           "--embedder", "hash", "--port", str(models_serve)])  # fmt: skip
    assert r.exit_code == 0, r.output
    res = json.loads(out.read_text())
    assert res["n"] == 4  # the mini fixture's 4 tune answerables
    assert "verdict" in res and ("C is DROPPED" in r.output or "C is NOT dropped" in r.output)
    assert sum(t["n"] for t in res["by_type"].values()) == 4
    assert res["first_pass_mismatches"] == 0  # recomputed first pass = the recorded pools
