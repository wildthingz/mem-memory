"""B-E1 agentic arm (`bench e1-agentic`, MEM2-214): the answerer drives the mem tools itself.
Fake tool-calling answerer, fake judge, fake tool box: no network, no models."""

import itertools
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import e1, e1_agentic
from bench.adapters.lme_ours import OursLmeBackend
from bench.cli import app
from bench.datasets import lme
from bench.llm import Reply
from mem.gate import Gate
from tests.bench_t0.test_e1 import QS, FakeLLM

_ids = itertools.count(1)
USAGE = {"input_tokens": 1000, "output_tokens": 100}
S_FILE = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme" / "lme_s_mini.json"


def use(name, **inp):
    return {"type": "tool_use", "id": f"tu_{next(_ids)}", "name": name, "input": inp}


class Agent:
    """Calls `recall`; follows `page=N` offers in the last tool result; then answers."""

    model = "claude-sonnet-5-5"

    def __init__(self, final="42 apples", follow=True, usage=USAGE, forever=False):
        self.final, self.follow, self.usage, self.forever = final, follow, usage, forever
        self.seen = []  # (system, tool names) of every call

    def create(self, *, system, messages, tools, max_tokens):
        self.seen.append((system, [t["name"] for t in tools], max_tokens))
        last = messages[-1]["content"]
        if isinstance(last, str):  # the question
            self.first = last
            return Reply([use("recall", query=last[-30:])], "tool_use", self.usage)
        text = last[-1]["content"]
        m = re.search(r"page=(\d+)", text)
        if self.forever or (self.follow and m):
            n = m.group(1) if m else "9"
            return Reply([use("recall", query="q", page=int(n))], "tool_use", self.usage)
        return Reply([{"type": "text", "text": self.final}], "end_turn", self.usage)


class Box:
    """A fake tool box: recall's first page offers page=2, the second does not."""

    schemas = ({"name": "recall"}, {"name": "get"})

    def __init__(self, qid, pages=True, boom=False):
        self.qid, self.pages, self.boom, self.calls = qid, pages, boom, []

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        if self.boom:
            raise RuntimeError("tool exploded")
        if args.get("page") == 2:
            return f"m2 · fact · row on page 2 of {self.qid}"
        return f"m1 · fact · row {self.qid}\n… cut 3 rows · page=2" if self.pages else "m1 · fact"

    def close(self):
        self.closed = True


def go(qs=QS, agent=None, judge=lambda p: "yes", boxes=None, skill="SKILL-A", **kw):
    agent = agent or Agent()
    boxes = {} if boxes is None else boxes
    j = FakeLLM(judge, usage={"input_tokens": 300, "output_tokens": 3})

    def open_tools(q):
        boxes[q["question_id"]] = Box(q["question_id"])
        return boxes[q["question_id"]]

    kw.setdefault("budget_usd", 5.0)
    rec = e1_agentic.run_agentic(qs, open_tools, agent, j, skill_text=skill, **kw)
    return rec, agent, j, boxes


def test_the_tool_loop_feeds_results_back_and_judges_the_final_answer():
    rec, _, judge, boxes = go(qs=QS[:1], agent=Agent(follow=False))
    (it,) = rec["items"]
    assert it["condition"] == "agentic" and it["answer"] == "42 apples" and it["correct"] is True
    assert [c["tool"] for c in it["calls"]] == ["recall"]
    assert it["calls"][0]["text"].startswith("m1 · fact · row q0")
    assert it["turns"] == 2 and it["stop"] == "end_turn"
    assert "42 apples" in judge.prompts[0]["prompt"]
    assert boxes["q0"].closed


def test_the_answerer_gets_the_skill_as_system_and_only_the_read_tools():
    _, agent, _, _ = go(qs=QS[:1], agent=Agent(follow=False))
    system, names, _ = agent.seen[0]
    assert system == "SKILL-A" and names == ["recall", "get"]


def test_paging_is_followed_when_the_header_offers_it_and_the_arg_passes_through():
    rec, _, _, boxes = go(qs=QS[:1])
    assert boxes["q0"].calls[0][1].get("page") is None
    assert boxes["q0"].calls[1] == ("recall", {"query": "q", "page": 2})
    assert [c["tool"] for c in rec["items"][0]["calls"]] == ["recall", "recall"]


def test_a_tool_error_goes_back_to_the_answerer_not_out_of_the_run():
    j = FakeLLM(lambda p: "yes")
    rec = e1_agentic.run_agentic(QS[:1], lambda q: Box("q0", boom=True), Agent(follow=False), j,
                                 skill_text="S", budget_usd=5.0)  # fmt: skip
    call = rec["items"][0]["calls"][0]
    assert call["error"] is True and "tool exploded" in call["text"]
    assert rec["items"][0]["answer"] == "42 apples"


def test_an_answerer_that_never_stops_is_cut_at_max_turns():
    rec, agent, _, _ = go(qs=QS[:1], agent=Agent(forever=True), max_turns=4)
    it = rec["items"][0]
    assert it["stop"] == "max_turns" and len(agent.seen) == 4 and len(it["calls"]) == 4


def test_the_budget_cap_stops_the_run_and_drops_the_unfinished_question():
    cheap = {"input_tokens": 10, "output_tokens": 10}
    cap = 0.003  # one question costs $0.0009 and the worst case of the next call is ~$0.0012
    rec, _, judge, _ = go(budget_usd=cap, agent=Agent(follow=False, usage=cheap),
                          answer_max_tokens=100)  # fmt: skip
    assert rec["complete"] is False and rec["stopped"] and rec["planned"] == len(QS)
    assert len(rec["items"]) == 2
    assert len(judge.prompts) == len(rec["items"])  # a question cut mid-way is not scored
    assert rec["cost"]["usd"] <= cap


def test_the_record_carries_prompt_and_skill_sha_and_the_skill_changes_it():
    a, *_ = go(qs=QS[:1], skill="SKILL-A")
    b, *_ = go(qs=QS[:1], skill="SKILL-A")
    c, *_ = go(qs=QS[:1], skill="SKILL-B")
    assert a["skill_sha256"] == b["skill_sha256"] != c["skill_sha256"]
    import hashlib

    assert a["skill_sha256"] == hashlib.sha256(b"SKILL-A").hexdigest()
    assert a["prompts_sha256"] == c["prompts_sha256"] == e1_agentic.prompts_sha256()
    assert a["prompts_sha256"] != e1.prompts_sha256()
    assert a["prompt_version"] == e1.PROMPT_VERSION and a["conditions"] == ["agentic"]


def test_the_question_prompt_is_the_v2_answer_wording_with_the_date():
    _, agent, _, _ = go(qs=QS[:1], agent=Agent(follow=False))
    p = agent.first
    assert p == e1_agentic.agent_prompt(QS[0]["question"], QS[0]["question_date"])
    assert "Current Date: 2023/05/30 (Tue) 10:00" in p and "How many q0?" in p
    assert "don't know" in p


# ---- rollups and pairing -------------------------------------------------------------------------


def run_file(correct_by_qid, cond, types=None):
    types = types or {}
    return {"conditions": [cond], "complete": True, "cost": {"usd": 1.0}, "items": [
        {"qid": q, "type": types.get(q, "multi-session"), "judge_kind": types.get(q, "multi-session"),
         "condition": cond, "judge_raw": "yes" if ok else "no", "correct": ok}
        for q, ok in correct_by_qid.items()]}  # fmt: skip


def test_per_type_rollup_and_pairing_by_question_against_a_baseline():
    types = {"a": "knowledge-update", "b": "knowledge-update", "c": "multi-session",
             "d": "multi-session", "e": "multi-session"}  # fmt: skip
    new = run_file({"a": True, "b": True, "c": False, "d": True, "e": True}, "agentic", types)
    base = run_file({"a": True, "b": False, "c": True, "d": True, "z": True}, "mem", types)
    s = e1_agentic.compare(new, base, b=200)
    assert s["n_paired"] == 4  # e is new only, z baseline only: not paired
    assert s["overall"]["new"] == 0.75 and s["overall"]["baseline"] == 0.75
    assert s["overall"]["delta"] == 0.0
    assert (
        s["per_type"]["knowledge-update"]["delta"] == 0.5
        and s["per_type"]["knowledge-update"]["n"] == 1 + 1
    )
    assert s["per_type"]["multi-session"]["delta"] == -0.5
    lo, hi = s["overall"]["delta_ci95"]
    assert lo <= 0.0 <= hi
    assert s["new_only"] == ["e"] and s["baseline_only"] == ["z"]


def test_adoption_needs_overall_not_below_and_no_type_down_more_than_5_points():
    def cmp(dov, **types):
        return {"overall": {"delta": dov}, "per_type": {t: {"delta": d} for t, d in types.items()}}

    assert e1_agentic.adoption(cmp(0.0, a=-0.05, b=0.1))["adopt"] is True
    assert e1_agentic.adoption(cmp(-0.01, a=0.0))["adopt"] is False
    v = e1_agentic.adoption(cmp(0.02, a=-0.06, b=0.1))
    assert v["adopt"] is False and "a" in v["reasons"][0]


def test_score_gives_per_type_accuracy_with_cis():
    rec, *_ = go(
        judge=lambda p: "yes" if "q0" in p or "q1" in p else "no", agent=Agent(follow=False)
    )
    m, per = e1_agentic.score(rec, b=100)
    assert m["accuracy"] == 0.4 and m["n"] == 5
    assert per["per_type"]["single-session-user"]["accuracy"] == 1.0
    assert len(per["per_type"]["single-session-user"]["ci95"]) == 2
    assert per["ci95"][0] <= 0.4 <= per["ci95"][1]


# ---- the real tools --------------------------------------------------------------------------


def test_the_backend_opens_the_real_read_tools_on_a_haystack(tmp_path):
    qs = lme.load_questions(S_FILE)
    backend = OursLmeBackend(tmp_path / "w", models="hash", gate=Gate(1.0, 0.0, 0.0, 0.0, 0.0))
    try:
        q = qs[0]
        box = backend.open_tools(q, lme.haystack_of(q))
        try:
            assert [s["name"] for s in box.schemas] == ["recall", "get"]
            assert "input_schema" in box.schemas[0]
            text = box.call("recall", {"query": q["question"]})
            cid = re.search(r"^(c\d+) ·", text, re.MULTILINE).group(1)
            assert box.call("get", {"ids": [cid]}).startswith(cid + " ·")
            with pytest.raises(Exception, match="bad arguments"):
                box.call("recall", {"query": "x", "nonsense": 1})
        finally:
            box.close()
    finally:
        backend.close()


def test_cli_asks_for_yes_before_spending(tmp_path):
    base = tmp_path / "b.json"
    base.write_text(json.dumps(run_file({"a": True}, "mem")))
    r = CliRunner().invoke(app, ["e1-agentic", "--lme-file", str(S_FILE), "--out",
                                 str(tmp_path / "o.json"), "--baseline", str(base)])  # fmt: skip
    assert r.exit_code == 2 and "--yes" in r.output


def test_a_baseline_that_is_itself_an_agentic_run_pairs_on_its_agentic_arm():
    assert e1_agentic.baseline_condition(run_file({"a": True}, "agentic")) == "agentic"
    assert e1_agentic.baseline_condition(run_file({"a": True}, "mem")) == "mem"
    both = {"conditions": ["mem", "none", "oracle"], "items": []}
    assert e1_agentic.baseline_condition(both) == "mem"


def test_two_e1_run_files_pair_mem_against_mem_with_question_count_deltas_per_type():
    """Round 10 (MEM2-270): an arm's `e1-run` against the base arm's, both `mem` arms, paired by question."""
    types = {"a": "knowledge-update", "b": "knowledge-update", "c": "temporal-reasoning",
             "d": "temporal-reasoning", "e": "temporal-reasoning"}  # fmt: skip
    arm = run_file({"a": True, "b": False, "c": True, "d": True, "e": False}, "mem", types)
    base = run_file({"a": True, "b": True, "c": False, "d": True, "e": False}, "mem", types)
    s = e1_agentic.compare(arm, base, new_condition="mem", b=200)
    assert s["n_paired"] == 5 and s["overall"]["delta"] == 0.0
    assert s["per_type"]["knowledge-update"]["questions"] == -1  # b lost
    assert s["per_type"]["temporal-reasoning"]["questions"] == 1  # c gained
    assert s["overall"]["questions"] == 0
    assert s["discordant"] == {"new_only_right": 1, "baseline_only_right": 1}
    # the default pairs an agentic run against a mem baseline, as before
    assert e1_agentic.compare(run_file({"a": True}, "agentic", types), base, b=50)["n_paired"] == 1


def test_e1_pair_command_prints_the_paired_lines_and_names_a_type_down_more_than_one(tmp_path):
    from typer.testing import CliRunner

    from bench.cli import app

    types = {q: "temporal-reasoning" for q in "abcdef"}
    arm = run_file(dict.fromkeys("abcdef", False), "mem", types)
    base = run_file(dict.fromkeys("abcdef", True), "mem", types)
    (tmp_path / "arm.json").write_text(json.dumps(arm))
    (tmp_path / "base.json").write_text(json.dumps(base))
    r = CliRunner().invoke(
        app, ["e1-pair", str(tmp_path / "arm.json"), str(tmp_path / "base.json"), "--b", "50"]
    )
    assert r.exit_code == 0, r.output
    assert (
        "paired n=6" in r.output and "temporal-reasoning" in r.output and "questions -6" in r.output
    )
    assert "DOWN by more than 1 question: temporal-reasoning" in r.output
    assert "overall" in r.output and "CI [" in r.output
    base["split"] = "test"
    (tmp_path / "base.json").write_text(json.dumps(base))
    arm["split"] = "tune"
    (tmp_path / "arm.json").write_text(json.dumps(arm))
    r = CliRunner().invoke(
        app, ["e1-pair", str(tmp_path / "arm.json"), str(tmp_path / "base.json")]
    )
    assert r.exit_code == 2 and "not comparable" in r.output
