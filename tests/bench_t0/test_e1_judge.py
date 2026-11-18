"""B-E1 judge reply guard (Round 10 task 4, MEM2-269; ADR 0029). The judge's cap was upstream's
`n_tokens=10`; Sonnet is sometimes verbose and a correct item (6c49646a, "3,000 miles") had its reply
("The response's final total is 3,000") cut before any yes, and scored wrong. Now: 64 tokens, the
verdict is the FIRST word-bounded yes/no token, a reply with neither is flagged (per item and counted
in the run record), and `judge_version` is a harness field so old and new runs never pool. The same
code judges every arm (mem, none, oracle), `e1-agentic`, `--full` and LME_M. Fake judge: no network."""

import json

import pytest
from typer.testing import CliRunner

from bench import e1, e1_agentic, e1_m, gates
from bench.cli import app
from bench.suites import b_e1

from .test_e1 import QS, make_q, run
from .test_e1_agentic import Agent, go
from .test_e1_m import META, backend, lines, m_file, run_m  # noqa: F401 - fixtures
from .test_e1_pool import HARNESS, e1_env, e1_run, record, write  # noqa: F401 - fixtures

VERBOSE_YES = "The response's final total is 3,000. Yes."
CUT = "The response's final total is 3,000"  # what a 10-token cap left of 6c49646a's reply


# ---- the parser -----------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "verdict"), [
    ("Yes, the answer matches the correct answer.", "yes"),
    ("No.", "no"),
    (VERBOSE_YES, "yes"),
    ("yes", "yes"), ("YES", "yes"), ("**Yes**", "yes"), ("Yes.", "yes"), ("no", "no"), ("NO", "no"),
    ("Answer: no", "no"),
    ("No, it is not yes", "no"),  # the first token, not upstream's "yes anywhere"
    ("The response says no, which is wrong. Yes.", "no"),  # the first token: a known limit (ADR 0029)
    (CUT, None),
    ("", None),
    ("Nothing in the response matches.", None),  # "no" inside a word is not a token
    ("The response is not correct.", None),  # neither is "not"
    ("Yesterday's total is right.", None),
    ("maybe", None),
])  # fmt: skip
def test_the_verdict_is_the_first_word_bounded_yes_or_no_token(text, verdict):
    assert e1.judge_verdict(text) == verdict
    assert e1.parse_verdict(text) is (verdict == "yes")  # flagged counts as not credited


def test_the_judge_prompt_text_is_upstreams_unchanged():
    # the fix is in the cap and the parsing only; the prompts (and so prompts_sha256) do not move
    for kind, t in e1.JUDGE_TEMPLATES.items():
        assert t.rstrip().endswith("Answer yes or no only."), kind
    assert e1.PROMPT_VERSION == 2


# ---- the cap reaches every judge call -----------------------------------------------------------


def test_the_cap_is_64_and_reaches_the_judge_call_on_every_arm():
    assert e1.JUDGE_MAX_TOKENS == 64
    _, _, j = run(conditions=("mem", "none", "oracle"))
    assert j.prompts and {p["max_tokens"] for p in j.prompts} == {64}


def test_the_cap_reaches_the_agentic_judge_too():
    _, _, j, _ = go(qs=QS[:1], agent=Agent(follow=False))
    assert {p["max_tokens"] for p in j.prompts} == {64}


def test_the_cap_reaches_the_m_runners_judge(m_file, backend, tmp_path):  # noqa: F811
    _, _, j = run_m(m_file, backend, tmp_path, limit=2)
    assert j.prompts and {p["max_tokens"] for p in j.prompts} == {64}


# ---- every arm is parsed the same way, a reply without yes/no is flagged ----------------------------


def test_a_verbose_yes_is_credited_on_every_arm():
    rec, _, _ = run(judge=lambda p: VERBOSE_YES, conditions=("mem", "none", "oracle"))
    assert len(rec["items"]) == 3 * len(QS)
    assert all(i["correct"] is True and i["judge_verdict"] == "yes" and not i["judge_flagged"]
               for i in rec["items"])  # fmt: skip
    assert rec["judge_flagged"] == 0
    metrics, _ = e1.score_run(rec)
    assert metrics["accuracy"] == metrics["accuracy_none"] == metrics["accuracy_oracle"] == 1.0
    assert metrics["judge_flagged"] == 0.0


def test_a_reply_without_yes_or_no_is_flagged_on_every_arm_not_silently_scored():
    flag = {"q1", "q2"}  # the judge is cut off on these questions, on every arm

    def judge(p):
        return CUT if any(f"Question: How many {q}?\n" in p for q in flag) else "yes"

    rec, _, _ = run(judge=judge, conditions=("mem", "none", "oracle"))
    flagged = [i for i in rec["items"] if i["judge_flagged"]]
    assert {i["qid"] for i in flagged} == flag and len(flagged) == 6  # 2 questions x 3 arms
    assert all(i["judge_verdict"] is None and i["correct"] is False and i["judge_raw"] == CUT
               for i in flagged)  # fmt: skip
    assert all(i["judge_verdict"] == "yes" for i in rec["items"] if not i["judge_flagged"])
    assert rec["judge_flagged"] == 6
    metrics, detail = e1.score_run(rec)
    assert metrics["judge_flagged"] == 6.0
    assert {c: v["judge_flagged"] for c, v in detail["per_condition"].items()} == {
        "mem": 2, "none": 2, "oracle": 2}  # fmt: skip
    assert metrics["accuracy"] == pytest.approx(3 / 5)  # flagged is not credited


def test_the_agentic_arm_uses_the_same_parser_and_flag():
    rec, _, _, _ = go(qs=QS[:2], agent=Agent(follow=False),
                      judge=lambda p: VERBOSE_YES if "q0" in p else CUT)  # fmt: skip
    a, b = rec["items"]
    assert a["correct"] is True and a["judge_verdict"] == "yes" and not a["judge_flagged"]
    assert b["correct"] is False and b["judge_verdict"] is None and b["judge_flagged"]
    assert rec["judge_flagged"] == 1 and rec["judge_version"] == e1.JUDGE_VERSION
    assert e1_agentic.score(rec)[0]["accuracy"] == 0.5
    assert all(i["correct"] is e1.parse_verdict(i["judge_raw"]) for i in rec["items"])


def test_the_m_runner_flags_and_counts_too(m_file, backend, tmp_path):  # noqa: F811
    rec, _, _ = run_m(m_file, backend, tmp_path, judge=lambda p: CUT, limit=2)
    rows = lines(tmp_path)
    assert len(rows) == 2 and all(r["judge_flagged"] and r["judge_verdict"] is None for r in rows)
    assert all(r["judge_version"] == e1.JUDGE_VERSION for r in rows)
    assert rec["judge_flagged"] == 2 and rec["judge_version"] == e1.JUDGE_VERSION
    assert rec["accuracy"] == 0.0


def test_scoring_an_old_run_file_flags_the_cut_replies_instead_of_scoring_them_no():
    # a run made under the 10-token cap: its replies are re-parsed, the cut ones surface as flagged
    rec, _, _ = run(conditions=("mem",))
    for i in rec["items"]:
        i.pop("judge_verdict"), i.pop("judge_flagged")  # an old file has neither field
    rec["items"][0]["judge_raw"] = CUT
    metrics, detail = e1.score_run(rec)
    assert metrics["judge_flagged"] == 1.0 and detail["per_condition"]["mem"]["judge_flagged"] == 1


# ---- judge_version: a harness field, so runs before and after never pool ---------------------------


def test_judge_version_is_2_and_a_harness_field_in_every_run_record(m_file, backend, tmp_path):  # noqa: F811
    assert e1.JUDGE_VERSION == 2 and "judge_version" in e1.HARNESS_FIELDS
    rec, _, _ = run(conditions=("none",))
    assert rec["judge_version"] == 2
    arec, _, _, _ = go(qs=QS[:1], agent=Agent(follow=False))
    assert arec["judge_version"] == 2
    mrec, _, _ = run_m(m_file, backend, tmp_path, limit=1)
    assert mrec["judge_version"] == 2 and lines(tmp_path)[0]["judge_version"] == 2


def test_the_pool_refuses_a_run_of_the_old_judge_naming_the_field(tmp_path):
    assert HARNESS["judge_version"] == 2
    old = record(119, tag="old", judge_version=1)
    with pytest.raises(
        e1.PoolRefused, match="harness field judge_version is 1, the current run's is 2"
    ):
        e1.load_pool([write(tmp_path, "old.json", old)], HARNESS)
    older = record(119, tag="older")
    del older["judge_version"]  # a run file from before the field existed
    with pytest.raises(e1.PoolRefused, match="harness field judge_version is None"):
        e1.load_pool([write(tmp_path, "older.json", older)], HARNESS)
    # and the other way round: a new-judge file cannot pool with an old-judge run's harness
    new = write(tmp_path, "new.json", record(119, tag="new"))
    with pytest.raises(
        e1.PoolRefused, match="harness field judge_version is 2, the current run's is 1"
    ):
        e1.load_pool([new], dict(HARNESS, judge_version=1))
    # the gate and e1-run refuse it before anything else (e1-run: no --yes, no key)
    run_path = write(tmp_path, "run.json", record(117))
    old_path = write(tmp_path, "old.json", old)
    r = CliRunner().invoke(app, ["gate", "P3", "--split", "tune", "--e1-run", str(run_path),
                                 "--oracle-pool", str(old_path)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and "harness field judge_version" in r.output
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(tmp_path / "o.json"),
                                 "--oracle-pool", str(old_path)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and "harness field judge_version" in r.output


def test_the_pooled_rule_still_holds_across_new_judge_runs(tmp_path):
    base = write(tmp_path, "base.json", record(119, tag="r10"))
    rec = record(114, tag="f")
    pool = e1.load_pool([base], rec)
    p = e1.oracle_pool(rec, pool)
    assert p["oracle_pooled"] == pytest.approx(0.932) and e1.pool_invalid_reason(p) is None
    finals = write(tmp_path, "finals.json", {**record(460, n=500, tag="full"), "split": "full"})
    assert e1.load_pool([base], json.loads(finals.read_text()))[0]["n"] == 125


def test_an_m_resume_refuses_another_judge_version(m_file, backend, tmp_path):  # noqa: F811
    assert "judge_version" in e1_m.CHECKED
    run_m(m_file, backend, tmp_path, limit=1)
    status_path = tmp_path / "r.STATUS"
    status = json.loads(status_path.read_text())
    assert status["meta"]["judge_version"] == e1.JUDGE_VERSION
    status["meta"]["judge_version"] = 1  # a STATUS file left by a run under the old judge
    status_path.write_text(json.dumps(status))
    with pytest.raises(e1_m.ResumeMismatch, match="judge_version"):
        run_m(m_file, backend, tmp_path)
    assert len(lines(tmp_path)) == 1  # nothing was appended


# ---- the gate and e1-run report the flagged count ---------------------------------------------------


def write_run(tmp_path, flagged_qids=(), n=125):
    qs = [make_q(f"s{i}", "multi-session") for i in range(n)]

    def judge(p):
        return CUT if any(f"Question: How many {q}?\n" in p for q in flagged_qids) else "yes"

    rec, _, _ = run(qs=qs, judge=judge, conditions=("mem", "oracle"), budget_usd=50.0)
    rec |= {"split": "tune", "complete": True, "config": {"fp_all": "x"},
            "dataset": {"name": "LongMemEval_S cleaned", "split": "tune"}}  # fmt: skip
    path = tmp_path / "e1.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_the_gate_counts_the_flagged_replies_and_says_so(tmp_path):
    out = gates.run_phase("P3", None, results_dir=tmp_path / "r", only={"B-E1"}, split="tune",
                          options={b_e1.OPT: b_e1.E1Setup(write_run(tmp_path, {"s3"}), "tune")})[0]  # fmt: skip
    rr = json.loads(out.run_file.read_text(encoding="utf-8"))
    assert rr["metrics"]["judge_flagged"] == 2.0  # s3 on mem and on oracle
    assert any("2 judge replies" in n and "yes/no" in n for n in rr["notes"]), rr["notes"]
    assert rr["extra"]["judge_version"] == 2
    clean = gates.run_phase("P3", None, results_dir=tmp_path / "c", only={"B-E1"}, split="tune",
                            options={b_e1.OPT: b_e1.E1Setup(write_run(tmp_path), "tune")})[0]  # fmt: skip
    rc = json.loads(clean.run_file.read_text(encoding="utf-8"))
    assert rc["metrics"]["judge_flagged"] == 0.0 and not any(
        "judge replies" in n for n in rc["notes"]
    )


def test_the_gate_notes_a_run_of_the_old_judge(tmp_path):
    path = write_run(tmp_path)
    rec = json.loads(path.read_text())
    rec["judge_version"] = 1
    path.write_text(json.dumps(rec), encoding="utf-8")
    out = gates.run_phase("P3", None, results_dir=tmp_path / "r", only={"B-E1"}, split="tune",
                          options={b_e1.OPT: b_e1.E1Setup(path, "tune")})[0]  # fmt: skip
    rr = json.loads(out.run_file.read_text(encoding="utf-8"))
    assert any("judge_version 1" in n and "ADR 0029" in n for n in rr["notes"]), rr["notes"]


def test_e1_run_records_and_prints_the_flagged_count(e1_env, monkeypatch):  # noqa: F811
    out = e1_env / "r10.json"
    r = e1_run(out)
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["judge_version"] == 2 and rec["judge_flagged"] == 0
    assert "judge_flagged = 0" in r.output
