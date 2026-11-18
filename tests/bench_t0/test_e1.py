"""B-E1 (`bench e1-run`, MEM2-184): budget cap, judge prompt per type, scoring and per-type rollups, the
gate row, and which context each condition gives the answerer. Fake answerer and judge: no network, no
API key, no models."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from bench import e1, gates
from bench.cli import app
from bench.llm import Reply, call_cost
from bench.suites import PHASE_REGISTRY, b_e1
from bench.suites.base import BackendUnavailable, Context

TYPES = ("single-session-user", "knowledge-update", "multi-session", "temporal-reasoning")


def make_q(qid, qtype, answer="42 apples"):
    return {
        "question_id": qid, "question_type": qtype, "question": f"How many {qid}?",
        "answer": answer, "question_date": "2023/05/30 (Tue) 10:00",
        "answer_session_ids": [f"{qid}-gold"],
        "haystack_session_ids": [f"{qid}-noise", f"{qid}-gold"],
        "haystack_dates": ["2023/01/01 (Sun) 09:00", "2023/02/02 (Thu) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": f"NOISE-{qid}"}, {"role": "assistant", "content": "ok"}],
            [{"role": "user", "content": f"GOLD-{qid}"}, {"role": "assistant", "content": "ok"}],
        ],
    }  # fmt: skip


QS = [make_q(f"q{i}", t) for i, t in enumerate(TYPES)] + [make_q("q9_abs", "multi-session", "no")]


class FakeLLM:
    """Records every prompt; `reply(prompt)` decides the text. Usage is fixed per call."""

    def __init__(self, reply, model="claude-sonnet-5-5", usage=None):
        self.model, self.reply_fn = model, reply
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 100}
        self.prompts = []

    def create(self, *, system, messages, tools, max_tokens):
        prompt = messages[0]["content"]
        self.prompts.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        return Reply([{"type": "text", "text": self.reply_fn(prompt)}], "end_turn", self.usage)


def run(qs=QS, answer=lambda p: "42 apples", judge=lambda p: "yes", retrieve=None, **kw):
    a, j = FakeLLM(answer), FakeLLM(judge, usage={"input_tokens": 300, "output_tokens": 3})
    kw.setdefault("budget_usd", 5.0)
    rec = e1.run_e1(qs, retrieve or (lambda q: f"MEM[{q['question_id']}]"), a, j, **kw)
    return rec, a, j


# ---- the budget cap ---------------------------------------------------------------------------


def test_the_budget_cap_stops_the_run_before_going_over():
    full, _, _ = run()
    per_item = full["cost"]["usd"] / len(full["items"])
    budget = (
        per_item * 5.5
    )  # room for 5 item-pairs of actual spend, but the worst-case guard bites first
    rec, a, j = run(budget_usd=budget)
    assert not rec["complete"] and rec["stopped"] and "budget" in rec["stopped"]
    assert 0 < len(rec["items"]) < rec["planned"]
    assert rec["cost"]["usd"] <= budget  # never over, not even by one call
    assert len(a.prompts) == len(rec["items"]) + (1 if len(a.prompts) > len(j.prompts) else 0)


def test_a_budget_smaller_than_one_call_makes_no_call_at_all():
    rec, a, j = run(budget_usd=0.001)
    assert rec["items"] == [] and not rec["complete"] and a.prompts == [] and j.prompts == []


def test_the_guard_counts_the_worst_case_output_not_just_what_was_spent():
    # one call's actual cost is ~$0.003, its worst case (1024 max tokens of output) is ~$0.012
    rec, a, _ = run(qs=QS[:1], conditions=("none",), budget_usd=0.005)
    assert a.prompts == [] and not rec["complete"]


def test_a_roomy_budget_completes_and_records_the_spend():
    rec, a, j = run()
    assert rec["complete"] and rec["stopped"] is None
    assert len(rec["items"]) == rec["planned"] == len(QS) * 3
    want = 3 * len(QS) * (call_cost("claude-sonnet-5-5", a.usage)
                          + call_cost("claude-sonnet-5-5", j.usage))  # fmt: skip
    assert rec["cost"]["usd"] == pytest.approx(want, abs=1e-3)
    assert rec["cost"]["answer"]["calls"] == rec["cost"]["judge"]["calls"] == len(rec["items"])


def test_budget_is_shared_when_answerer_and_judge_are_different_models():
    a = FakeLLM(
        lambda p: "x", model="claude-opus-5-5", usage={"input_tokens": 1000, "output_tokens": 100}
    )
    j = FakeLLM(lambda p: "yes", usage={"input_tokens": 100, "output_tokens": 3})
    rec = e1.run_e1(QS, lambda q: "m", a, j, conditions=("none",), budget_usd=0.02)
    assert rec["cost"]["usd"] <= 0.02 and not rec["complete"]
    assert rec["answerer"] == "claude-opus-5-5" and rec["judge"] == "claude-sonnet-5-5"


# ---- the judge prompt per type ----------------------------------------------------------------

DEFAULT_ONLY = "contains all the intermediate steps"
OFF_BY_ONE = "do not penalize off-by-one errors"
UPDATED = "as long as the updated answer is the required answer"
RUBRIC = "rubric for desired personalized response"
UNANSWERABLE = "correctly identifies the question as unanswerable"


@pytest.mark.parametrize(
    ("kind", "has", "lacks"),
    [
        ("single-session-user", [DEFAULT_ONLY], [OFF_BY_ONE, UPDATED, RUBRIC, UNANSWERABLE]),
        ("single-session-assistant", [DEFAULT_ONLY], [OFF_BY_ONE, UPDATED, RUBRIC, UNANSWERABLE]),
        ("multi-session", [DEFAULT_ONLY], [OFF_BY_ONE, UPDATED, RUBRIC, UNANSWERABLE]),
        ("temporal-reasoning", [DEFAULT_ONLY, OFF_BY_ONE], [UPDATED, RUBRIC, UNANSWERABLE]),
        ("knowledge-update", [DEFAULT_ONLY, UPDATED], [OFF_BY_ONE, RUBRIC, UNANSWERABLE]),
        ("single-session-preference", [RUBRIC, "Rubric: R"], [DEFAULT_ONLY, OFF_BY_ONE, UPDATED]),
        ("abstention", [UNANSWERABLE, "Explanation: R"], [DEFAULT_ONLY, OFF_BY_ONE, UPDATED, RUBRIC]),
    ],
)  # fmt: skip
def test_each_type_gets_its_own_judge_prompt(kind, has, lacks):
    p = e1.judge_prompt(kind, "Q?", "R", "RESP")
    assert "Question: Q?" in p and "Model Response: RESP" in p and p.rstrip().endswith("only.")
    assert all(h in p for h in has) and not any(x in p for x in lacks)


def test_the_run_sends_each_question_the_prompt_of_its_type():
    rec, _, j = run(conditions=("mem",))
    by_q = {i["qid"]: i["judge_kind"] for i in rec["items"]}
    assert by_q == {"q0": "single-session-user", "q1": "knowledge-update", "q2": "multi-session",
                    "q3": "temporal-reasoning", "q9_abs": "abstention"}  # fmt: skip
    sent = {k: p["prompt"] for k, p in zip(by_q, j.prompts, strict=True)}
    assert OFF_BY_ONE in sent["q3"] and UPDATED in sent["q1"] and UNANSWERABLE in sent["q9_abs"]
    assert OFF_BY_ONE not in sent["q0"] and "Correct Answer: 42 apples" in sent["q0"]
    assert (
        "Explanation: no" in sent["q9_abs"]
    )  # an _abs question is judged as abstention, not as multi-session


def test_the_judge_sees_the_answerers_response():
    rec, _, j = run(answer=lambda p: "I counted 17.", conditions=("none",))
    assert all("Model Response: I counted 17." in p["prompt"] for p in j.prompts)
    assert rec["items"][0]["answer"] == "I counted 17."


@pytest.mark.parametrize(("text", "ok"), [("yes", True), ("Yes.", True), ("**Yes**", True),
                                          ("no", False), ("No, it is not yes", False),
                                          ("", False), ("maybe yes", True), ("NO", False)])  # fmt: skip
def test_verdict_is_the_first_yes_or_no_token(text, ok):
    # ADR 0029 (MEM2-269): the first word-bounded yes/no token, not upstream's "yes anywhere"
    assert e1.parse_verdict(text) is ok


def test_the_judge_is_capped_at_64_tokens_not_upstreams_10():
    _, _, j = run()
    assert j.prompts and {p["max_tokens"] for p in j.prompts} == {e1.JUDGE_MAX_TOKENS} == {64}


# ---- conditions select the context ------------------------------------------------------------


def test_the_three_conditions_give_the_answerer_three_different_contexts():
    rec, a, _ = run(qs=QS[:1], conditions=("mem", "none", "oracle"))
    mem, none, oracle = (p["prompt"] for p in a.prompts)
    assert "MEM[q0]" in mem and "GOLD-q0" not in mem and "NOISE-q0" not in mem
    assert "MEM[" not in none and "GOLD" not in none and "NOISE" not in none
    assert "History Chats" not in none and "How many q0?" in none
    assert "GOLD-q0" in oracle and "NOISE-q0" not in oracle and "MEM[" not in oracle
    assert "[session q0-gold, 2023/02/02" in oracle  # evidence sessions come whole, with their date
    assert [i["condition"] for i in rec["items"]] == ["mem", "none", "oracle"]
    assert all(p["system"] == "" for p in a.prompts)


def test_retrieval_runs_once_per_question_and_only_for_mem():
    calls = []

    def retrieve(q):
        calls.append(q["question_id"])
        return "R"

    run(qs=QS[:2], retrieve=retrieve, conditions=("none", "oracle"))
    assert calls == []
    run(qs=QS[:2], retrieve=retrieve, conditions=("mem", "mem".replace("mem", "none"), "oracle"))
    assert calls == ["q0", "q1"]


def test_unknown_condition_and_missing_retrieve_are_refused():
    with pytest.raises(ValueError, match="unknown conditions"):
        run(conditions=("full",))
    with pytest.raises(ValueError, match="needs a retrieve"):
        e1.run_e1(QS, None, FakeLLM(lambda p: ""), FakeLLM(lambda p: ""), budget_usd=1.0)


# ---- scoring and rollups ----------------------------------------------------------------------


def test_scoring_and_per_type_rollups_are_exact():
    # 5 questions, mem only: q0 ok, q1 no, q2 ok, q3 no, q9_abs ok  -> 3/5
    verdicts = iter(["yes", "no", "yes", "no", "yes"])
    rec, _, _ = run(judge=lambda p: next(verdicts), conditions=("mem",))
    m, d = e1.score_run(rec)
    assert m["accuracy"] == pytest.approx(0.6) and m["n"] == 5 and m["run_complete"] == 1
    assert m["abstention_accuracy"] == 1.0  # the _abs question is its own bucket, not multi-session
    pt = d["per_condition"]["mem"]["per_type"]
    assert (
        pt["single-session-user"]["accuracy"] == 1.0 and pt["knowledge-update"]["accuracy"] == 0.0
    )
    assert pt["multi-session"] == {**pt["multi-session"], "n": 1, "accuracy": 1.0}
    assert pt["temporal-reasoning"]["accuracy"] == 0.0 and pt["abstention"]["n"] == 1
    assert m["accuracy_macro"] == pytest.approx(3 / 5)  # five buckets of one: macro == micro here
    assert m["accuracy_ci_lo"] <= m["accuracy"] <= m["accuracy_ci_hi"]
    assert m["cost_usd"] == pytest.approx(rec["cost"]["usd"])


def test_micro_and_macro_differ_when_buckets_differ_in_size():
    qs = [make_q(f"a{i}", "multi-session") for i in range(8)] + [make_q("k0", "knowledge-update")]
    verdicts = iter(["yes"] * 8 + ["no"])
    rec, _, _ = run(qs=qs, judge=lambda p: next(verdicts), conditions=("mem",))
    m, _ = e1.score_run(rec)
    assert m["accuracy"] == pytest.approx(8 / 9) and m["accuracy_macro"] == pytest.approx(0.5)


def test_other_conditions_are_reported_beside_the_gated_mem_arm():
    # the loop is question-major: per question the verdicts are mem, none, oracle
    seq = iter([("yes" if c != "none" else "no") for _q in QS for c in ("mem", "none", "oracle")])
    rec, _, _ = run(judge=lambda p: next(seq))
    m, _ = e1.score_run(rec)
    assert (m["accuracy"], m["accuracy_none"], m["accuracy_oracle"]) == (1.0, 0.0, 1.0)


def test_scoring_reparses_the_judge_reply_and_does_not_trust_the_stored_flag():
    rec, _, _ = run(judge=lambda p: "no", conditions=("mem",))
    for i in rec["items"]:
        i["correct"] = True  # a hand-edited flag must not change the score
    assert e1.score_run(rec)[0]["accuracy"] == 0.0


def test_an_incomplete_run_scores_what_finished_and_says_it_is_incomplete():
    rec, _, _ = run(budget_usd=0.02, conditions=("mem",))
    m, _ = e1.score_run(rec)
    assert m["run_complete"] == 0 and 0 < m["n"] < len(QS)


def test_a_run_without_mem_has_no_gated_accuracy():
    rec, _, _ = run(conditions=("none",))
    m, _ = e1.score_run(rec)
    assert math.isnan(m["accuracy"]) and m["accuracy_none"] == 1.0


# ---- the gate row -----------------------------------------------------------------------------


def row(metric):
    return next(r for r in gates.load_gates()["P3"] if r.id == "B-E1" and r.metric == metric)


def test_the_gate_row_reads_the_threshold_from_gates_yaml():
    r = row("accuracy")
    assert (r.op, r.threshold, r.split, r.descriptive, r.known_miss) == (
        ">=",
        0.75,
        "test",
        False,
        False,
    )
    assert (row("run_complete").op, row("run_complete").threshold) == (">=", 1)
    assert PHASE_REGISTRY["P3"]["B-E1"] is b_e1.run


def write_run(tmp_path, n_yes, n=20, split="tune", complete=True):
    qs = [make_q(f"s{i}", "multi-session") for i in range(n)]
    mem = iter(["yes"] * n_yes + ["no"] * (n - n_yes))
    turn = iter(range(10**6))  # arms alternate per question: mem, then oracle (always right)
    rec, _, _ = run(
        qs=qs,
        judge=lambda p: next(mem) if next(turn) % 2 == 0 else "yes",
        conditions=("mem", "oracle"),
    )
    rec |= {"split": split, "complete": complete, "config": {"fp_all": "x"},
            "dataset": {"name": "LongMemEval_S cleaned", "split": split}}  # fmt: skip
    path = tmp_path / "e1.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def gate(path, split="tune"):
    opts = {b_e1.OPT: b_e1.E1Setup(path, split)} if path else {}
    return gates.run_phase("P3", None, results_dir=None, only={"B-E1"}, options=opts, split=split)[
        0
    ]


@pytest.mark.parametrize(("yes", "status"), [(15, "PASS"), (14, "FAIL"), (20, "PASS"), (0, "FAIL")])
def test_gate_passes_at_075_and_fails_below(tmp_path, yes, status):
    o = gate(write_run(tmp_path, yes))
    assert o.status == status, o.detail
    assert "accuracy=" in o.detail and "need >= 0.75" in o.detail


def test_gate_is_unavailable_without_a_run_never_a_pass():
    o = gate(None)
    assert o.status == "UNAVAILABLE" and "bench e1-run" in o.detail


def test_gate_fails_an_incomplete_run_even_with_a_high_accuracy(tmp_path):
    o = gate(write_run(tmp_path, 20, complete=False))
    assert o.status == "FAIL" and "FAILED run_complete=0" in o.detail


def test_gate_refuses_a_tune_run_on_the_test_split(tmp_path):
    o = gate(write_run(tmp_path, 20, split="tune"), split="test")
    assert o.status == "FAIL" and "split" in o.detail


def test_suite_result_carries_ci_cost_and_the_other_arms(tmp_path):
    ctx = Context(tmp_path, None, None, {b_e1.OPT: b_e1.E1Setup(write_run(tmp_path, 16), "tune")})
    res = b_e1.run(ctx)
    assert res.tier == "t2" and res.metrics["accuracy"] == 0.8
    assert (
        res.extra["per_condition"]["mem"]["ci95"][0]
        < 0.8
        < res.extra["per_condition"]["mem"]["ci95"][1]
    )
    assert res.extra["dataset"]["n"] == 20 and res.extra["cost"]["usd"] > 0 and len(res.items) == 40
    with pytest.raises(BackendUnavailable):
        b_e1.run(Context(tmp_path, None, None, {}))


def test_the_run_record_keeps_per_item_results_and_the_prompt_fingerprint(tmp_path):
    out = gates.run_phase(
        "P3",
        None,
        results_dir=tmp_path / "r",
        only={"B-E1"},
        split="tune",
        options={b_e1.OPT: b_e1.E1Setup(write_run(tmp_path, 16), "tune")},
    )[0]
    rec = json.loads(out.run_file.read_text(encoding="utf-8"))
    assert rec["bench_id"] == "B-E1" and rec["tier"] == "t2" and rec["n_items"] == 40
    assert rec["extra"]["prompts_sha256"] == e1.prompts_sha256() and rec["extra"]["answerer"]
    assert rec["extra"]["cost"]["usd"] > 0 and rec["config"]["fp_all"]
    items = out.run_file.with_suffix(".items.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(items) == 40 and {"qid", "answer", "judge_raw", "correct"} <= set(
        json.loads(items[0])
    )


# ---- the CLI ----------------------------------------------------------------------------------


def invoke(*args):
    return CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", "o.json", *args])


def test_cli_refuses_without_yes_and_never_reaches_the_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = invoke()
    assert r.exit_code == 2 and "--yes" in r.output and "hard cap $10.00" in r.output
    assert "$3.50" in invoke("--budget-usd", "3.5").output


def test_cli_test_split_needs_a_reason_and_a_freeze():
    r = invoke("--yes", "--split", "test")
    assert r.exit_code == 2 and "--ledger" in r.output
    r = invoke("--yes", "--split", "test", "--ledger", "why")
    assert r.exit_code == 2 and "REFUSED" in r.output  # no frozen P3 thresholds: no test look


def test_cli_rejects_unknown_conditions_and_models():
    assert invoke("--condition", "full").exit_code != 0
    assert invoke("--answerer-model", "gpt-9").exit_code != 0
    assert invoke("--split", "dev").exit_code != 0


def test_cli_records_the_config_files_fingerprint_without_the_mem_arm(monkeypatch, tmp_path):
    from bench import records
    from bench.datasets import lme
    from bench.llm import AnthropicLLM  # noqa: F401  (patched below)

    monkeypatch.setattr("bench.llm.AnthropicLLM", lambda model, effort: FakeLLM(lambda p: "yes"))
    monkeypatch.setattr(lme, "read_manifest", lambda p: object())
    monkeypatch.setattr(lme, "load_questions", lambda f: QS)
    monkeypatch.setattr(lme, "verify_manifest", lambda qs, m: None)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    rows = [type("R", (), {"question_id": q["question_id"]}) for q in QS]
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    cfg = tmp_path / "config.yaml"
    cfg.write_text("retrieval:\n  limit: 7\n", encoding="utf-8")
    out = tmp_path / "run.json"

    def go(*extra):
        r = CliRunner().invoke(
            app, ["e1-run", "--lme-file", "x.json", "--out", str(out), "--yes",
                  "--condition", "none", *extra])  # fmt: skip
        assert r.exit_code == 0, r.output
        return json.loads(out.read_text(encoding="utf-8"))["config"]["fp_all"]

    default_fp = go()
    assert default_fp == records.config_fingerprint()["fp_all"]
    assert go("--config", str(cfg)) != default_fp


# ---- the full-500 mode (MEM2-196): one test look, over every question of the file ---------------------


def invoke_full(*args, out="o.json"):
    return CliRunner().invoke(
        app, ["e1-run", "--lme-file", "x.json", "--out", out, "--full", *args]
    )


def test_full_mode_refuses_without_spend_look_and_never_reaches_the_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = invoke_full("--yes", "--ledger", "final run")
    assert r.exit_code == 2 and "--spend-look" in r.output and "test look" in r.output
    r = invoke_full("--spend-look", "--ledger", "final run")  # spends money: --yes as ever
    assert r.exit_code == 2 and "--yes" in r.output


def test_full_mode_needs_a_reason_the_freeze_and_all_questions():
    r = invoke_full("--yes", "--spend-look")
    assert r.exit_code == 2 and "--ledger" in r.output
    r = invoke_full("--yes", "--spend-look", "--ledger", "why")
    assert r.exit_code == 2 and "REFUSED" in r.output  # unfrozen P3 thresholds: no test look
    assert invoke_full("--yes", "--spend-look", "--ledger", "why", "--limit", "5").exit_code != 0
    assert invoke_full("--yes", "--spend-look", "--ledger", "why", "--split", "tune").exit_code != 0


def test_full_mode_refuses_the_recall_context_diagnostic(full_env):
    led, out = full_env
    r = invoke_full("--yes", "--spend-look", "--ledger", "why", "--context", "recall",
                    out=str(out))  # fmt: skip
    assert r.exit_code == 2 and "--context recall" in r.output
    assert not led.exists() and not out.exists()  # no look spent


@pytest.fixture
def full_env(monkeypatch, tmp_path):
    from bench import freeze as freeze_mod
    from bench import records
    from bench.datasets import fetch, lme

    monkeypatch.setattr("bench.llm.AnthropicLLM", lambda model, effort: FakeLLM(lambda p: "yes"))
    monkeypatch.setattr(freeze_mod, "check_test_look_allowed", lambda phase: None)
    monkeypatch.setattr(lme, "load_questions", lambda f: QS)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    monkeypatch.setattr(fetch, "verify", lambda name, path: "0" * 64)
    monkeypatch.setattr(lme, "select", lambda *a: pytest.fail("the manifest picks no questions"))
    monkeypatch.setattr(lme, "read_manifest", lambda p: pytest.fail("the manifest is not read"))
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(records, "LEDGER", led)
    return led, tmp_path / "run.json"


def test_full_mode_runs_every_question_marks_the_look_and_writes_the_ledger(full_env):
    led, out = full_env
    r = invoke_full("--yes", "--spend-look", "--ledger", "final run", "--who", "tester",
                    "--condition", "none", "--budget-usd", "5", out=str(out))  # fmt: skip
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert len(rec["items"]) == len(QS) == rec["planned"] and rec["complete"]
    assert rec["full"] is True and rec["spends_test_look"] is True and rec["split"] == "full"
    (line,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert (line["phase"], line["split"], line["kind"], line["who"]) == ("P3", "test",
                                                                          "e1-run-full", "tester")  # fmt: skip
    assert line["why"] == "final run" and line["cost"] == rec["cost"]["usd"]
    assert line["n_questions"] == len(QS)


def test_full_mode_logs_the_look_even_when_the_budget_stops_it(full_env):
    led, out = full_env
    r = invoke_full("--yes", "--spend-look", "--ledger", "final run", "--condition", "none",
                    "--budget-usd", "0.005", out=str(out))  # fmt: skip
    assert r.exit_code == 1 and "STOPPED" in r.output
    assert len(led.read_text().splitlines()) == 1  # the look is spent, complete or not


# ---- MEM2-207/MEM2-227: the owner's e1_qa.py answer prompts and request shape, oracle validity ----

# tests/fixtures/e1_qa_prompts.py holds `ANSWER` and `ANSWER_NONE` copied verbatim from the owner's
# `data/host-pass7/e1_qa.py` (its header records that file's sha256). The fixture's own sha256 is
# pinned here so neither side can drift silently; it is exec'd, never imported from data/.
OWNER_PROMPTS = Path(__file__).resolve().parents[1] / "fixtures" / "e1_qa_prompts.py"
OWNER_PROMPTS_SHA256 = "980adae8cb8e0b8d455d10526fe5acccc5968cf243faedd543b864c5d8f3cb06"


def owner_prompts() -> tuple[str, str]:
    raw = OWNER_PROMPTS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OWNER_PROMPTS_SHA256, (
        "fixture changed: re-pin on purpose"
    )
    ns: dict[str, Any] = {}
    exec(compile(raw.decode("utf-8"), str(OWNER_PROMPTS), "exec"), ns)  # noqa: S102
    return ns["ANSWER"], ns["ANSWER_NONE"]


def test_the_answer_prompts_are_the_owners_e1_qa_prompts_byte_for_byte():
    answer, answer_none = owner_prompts()
    assert e1.ANSWER == answer
    assert e1.ANSWER_NONE == answer_none
    assert "TODO" not in answer_none and answer_none.endswith(
        "(say you don't know if you can't know):"
    )


def test_mem_and_oracle_prompts_are_the_owners_template_byte_for_byte():
    answer, _ = owner_prompts()
    q = QS[0]
    _, a, _ = run(qs=[q], conditions=("mem", "oracle"))
    mem, oracle = (p["prompt"] for p in a.prompts)
    d, qq = q["question_date"], q["question"]
    assert mem == answer.format(ctx="MEM[q0]", date=d, q=qq)
    assert oracle == answer.format(ctx=e1.oracle_text(q), date=d, q=qq)


def test_none_uses_answer_none_and_no_system_prompt_is_sent():
    q = QS[0]
    _, a, _ = run(qs=[q], conditions=("mem", "none", "oracle"))
    assert a.prompts[1]["prompt"] == e1.ANSWER_NONE.format(date=q["question_date"], q=q["question"])
    assert {p["system"] for p in a.prompts} == {""}


def test_the_judge_request_is_the_upstream_prompt_under_the_64_token_cap():
    _, _, j = run(qs=[QS[0]], conditions=("mem",), answer=lambda p: "R")
    assert j.prompts[0] == {"system": "", "max_tokens": 64,
                            "prompt": e1.judge_prompt("single-session-user", QS[0]["question"], "42 apples", "R")}  # fmt: skip


class _Capture:
    def __init__(self):
        self.kw = None

    def create(self, **kw):
        self.kw = kw
        u = SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return SimpleNamespace(content=[], stop_reason="end_turn", usage=u)


def _llm(effort):
    from bench.llm import AnthropicLLM

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model, llm.effort = "claude-sonnet-5-5", effort
    llm._client = type("C", (), {"messages": _Capture()})
    return llm


def test_the_answerer_sends_what_e1_qa_py_ask_sends_and_the_judge_keeps_its_effort():
    # MEM2-227 F5: the owner's e1_qa.py ask() sends thinking between_tools, no system prompt, no
    # sampling parameters, no effort. (`thinking: disabled` is a 400 on claude-sonnet-5-5.)
    ans, jdg = _llm(None), _llm("low")
    msgs = [{"role": "user", "content": "Q"}]
    ans.create(system="", messages=msgs, tools=[], max_tokens=5)
    jdg.create(system="", messages=msgs, tools=[], max_tokens=5)
    a, j = ans._client.messages.kw, jdg._client.messages.kw
    assert a == {"model": "claude-sonnet-5-5", "max_tokens": 5, "messages": msgs,
                 "cache_control": {"type": "ephemeral"}, "thinking": {"type": "between_tools"}}  # fmt: skip
    assert not {"system", "output_config", "temperature", "top_p", "top_k", "tools"} & a.keys()
    assert j["output_config"] == {"effort": "low"} and "thinking" not in j


def write_two_arm(tmp_path, oracle_yes, arms=("mem", "oracle"), n=100):
    qs = [make_q(f"s{i}", "multi-session") for i in range(n)]
    verdicts = iter(["yes"] * n + ["yes"] * oracle_yes + ["no"] * (n - oracle_yes))
    rec, _, _ = run(qs=qs, judge=lambda p: next(verdicts), conditions=arms, budget_usd=50.0)
    # arms run question by question; reorder the verdicts per arm instead of relying on iteration order
    for cond, k in (("mem", n), ("oracle", oracle_yes)):
        its = [i for i in rec["items"] if i["condition"] == cond]
        for idx, it in enumerate(its):
            it["judge_raw"] = "yes" if idx < k else "no"
    rec |= {"split": "tune", "complete": True, "config": {"fp_all": "x"},
            "dataset": {"name": "LongMemEval_S cleaned", "split": "tune"}}  # fmt: skip
    path = tmp_path / "e1.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


@pytest.mark.parametrize(("oracle_yes", "status"), [(91, "INVALID"), (92, "PASS")])
def test_oracle_validity_floor_is_092(tmp_path, oracle_yes, status):
    # re-based on the strict judge (the owner, 2026-10-27, MEM2-260): the alone-run pool needs >= 0.92
    o = gate(write_two_arm(tmp_path, oracle_yes))
    assert o.status == status, o.detail
    if status == "INVALID":
        assert "oracle accuracy 0.910" in o.detail


def test_a_run_without_the_oracle_arm_is_invalid_even_at_full_accuracy(tmp_path):
    o = gate(write_two_arm(tmp_path, 0, arms=("mem",)))
    assert o.status == "INVALID" and "no oracle arm" in o.detail


def test_an_invalid_run_is_not_a_pass_or_fail_in_the_record(tmp_path):
    out = gates.run_phase(
        "P3",
        None,
        results_dir=tmp_path / "r",
        only={"B-E1"},
        split="tune",
        options={b_e1.OPT: b_e1.E1Setup(write_two_arm(tmp_path, 91), "tune")},
    )[0]
    rec = json.loads(out.run_file.read_text(encoding="utf-8"))
    assert not out.passed and rec["extra"]["invalid"] and rec["metrics"]["accuracy_oracle"] == 0.91
    assert all(g["passed"] is None for g in rec["gates"])


def test_prompt_version_2_and_the_sha_are_recorded(tmp_path):
    rec, _, _ = run(conditions=("none",))
    assert e1.PROMPT_VERSION == 2 and rec["prompt_version"] == 2
    assert rec["prompts_sha256"] == e1.prompts_sha256()
    old = e1.ANSWER
    try:
        e1.ANSWER = old + " "
        assert e1.prompts_sha256() != rec["prompts_sha256"]
    finally:
        e1.ANSWER = old
    res = b_e1.run(
        Context(tmp_path, None, None, {b_e1.OPT: b_e1.E1Setup(write_two_arm(tmp_path, 94), "tune")})
    )
    assert res.extra["prompt_version"] == 2
