"""B-W5 harness (benchmarks-v2 B-W5): the dialogues, the writer loop, the ledger, the scoring and the
audit sheet, with scripted writers (no network, no money)."""

import csv
import json
import math

import pytest

from bench import agent_replay
from bench.llm import BudgetExceeded, Ledger, Reply, call_cost
from bench.suites import b_w5
from bench.suites.base import BackendUnavailable, Context
from bench.synth import data as D
from bench.synth import dialogue as dlg
from bench.synth.generator import generate
from tests.bench_t0.fake_llm import Writer
from tests.bench_t0.fake_p3 import factory_p3
from tests.bench_t0.fake_serve import list_tools

SYN = generate(101, "M")
DLS = dlg.build_dialogues(SYN)


def run_mode(tmp_path, mode, budget=10.0, dialogues=DLS):
    return agent_replay.run_all(dialogues, SYN, factory_p3(), tmp_path / f"{mode}.db",
                                Writer(dialogues, SYN, mode), budget, D.TZ)  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("w5")
    return {m: run_mode(d, m) for m in ("perfect", "lazy", "noisy", "no_link", "wrong_rel",
                                         "paraphrase_lossy")}  # fmt: skip


def metrics(run):
    return b_w5.score_run(run)[0]


# ---- the dialogues -------------------------------------------------------------------------------


def test_forty_dialogues_of_3_to_12_facts_with_small_talk():
    assert len(DLS) == 40
    assert all(3 <= len(d.facts) <= 12 for d in DLS)
    assert all(1 <= len(d.ignores) <= 2 for d in DLS)
    for d in DLS:
        for m in d.ignores:
            assert m in d.transcript


def test_no_value_returns_and_every_value_is_one_gold_fact():
    ops = {op["i"]: op for op in SYN.ops}
    values = []
    for d in DLS:
        for f in d.facts:
            op = ops[f["op"]]
            assert op["kind"] in ("new_fact", "compatible_add", "update")
            assert op["kind"] != "update" or op["gold"]["subtype"] == "normal"
            values.append((f["subject"], f["relation"], f["value"]))
    assert len(values) == len(set(values))
    assert len({v[2] for v in values}) == len(values)  # world-unique values (A17)


def test_dialogues_are_deterministic_and_chronological():
    again = dlg.build_dialogues(generate(101, "M"))
    assert [d.transcript for d in again] == [d.transcript for d in DLS]
    assert [d.start for d in DLS] == sorted(d.start for d in DLS)
    assert len({d.start for d in DLS}) == len(DLS)


def test_conflict_facts_have_a_predecessor_fact_in_an_earlier_or_same_dialogue():
    n = 0
    for d in DLS:
        for f in d.facts:
            if f["conflict_with"]:
                n += 1
                d0, i0 = f["conflict_with"]
                assert d0 <= d.index
                old = DLS[d0].facts[i0]
                assert (old["subject"], old["relation"]) == (f["subject"], f["relation"])
                assert old["value"] != f["value"]
    assert n >= 20


def test_gold_current_is_the_last_value_of_single_valued_slots():
    cur = dlg.gold_current(DLS)
    last: dict[tuple[str, str], tuple[int, int]] = {}
    for d in DLS:
        for i, f in enumerate(d.facts):
            if not D.RELATIONS[f["relation"]].multi:
                last[(f["subject"], f["relation"])] = (d.index, i)
    assert set(last.values()) <= cur


def test_gold_answer_names_the_entities_the_question_mentions():
    d = DLS[0]
    ent = next(e for e in SYN.expected["entities"] if e["key"] == d.facts[0]["subject"])
    assert ent["name"] in dlg.gold_answer(f"Which {ent['name']} do you mean?", d, SYN)
    assert "best judgement" in dlg.gold_answer("Anything else?", d, SYN)


# ---- the ledger (B-C5) ---------------------------------------------------------------------------


def test_call_cost_uses_list_prices_and_cache_multipliers():
    usage = {"input_tokens": 3000, "cache_read_input_tokens": 6000, "output_tokens": 500,
             "cache_creation_input_tokens": 0}  # fmt: skip
    assert call_cost("claude-sonnet-5-5", usage) == pytest.approx(
        (3000 * 2 + 6000 * 0.2 + 500 * 10) / 1e6
    )  # the spec's ~$0.012 per agent turn
    assert call_cost("claude-sonnet-5-5", {"cache_creation_input_tokens": 1_000_000}) == 2.5


def test_budget_guard_stops_before_the_call_that_could_cross(tmp_path):
    ledger = Ledger("claude-sonnet-5-5", 0.05)
    ledger.add({"input_tokens": 1_000_000}, "x")  # $2 spent: over budget already
    with pytest.raises(BudgetExceeded):
        ledger.check(4096, 1000)
    run = run_mode(tmp_path, "perfect", budget=0.30)
    assert run["complete"] is False
    assert run["cost"]["usd"] <= 0.30
    assert run["dialogues"][-1]["stop"] == "budget"


def test_run_records_every_call_with_usage_and_cost(runs):
    r = runs["perfect"]
    assert r["cost"]["calls"] == len(r["ledger"]) > 40
    assert r["cost"]["usd"] == pytest.approx(sum(c["usd"] for c in r["ledger"]), abs=1e-3)
    assert r["skill_sha256"] and r["tools"] == ["recall", "remember", "link", "subjects"]
    assert all(d["calls"] for d in r["dialogues"])


def test_the_writer_is_given_the_production_skill_and_the_wrapper(tmp_path):
    seen = {}

    class Spy(Writer):
        def create(self, *, system, messages, tools, max_tokens):
            seen.setdefault("system", system)
            seen.setdefault("first", messages[0]["content"])
            seen["tools"] = [t["name"] for t in tools]
            return super().create(system=system, messages=messages, tools=tools,
                                  max_tokens=max_tokens)  # fmt: skip

    agent_replay.run_all(DLS[:2], SYN, factory_p3(), tmp_path / "s.db", Spy(DLS, SYN), 5.0, D.TZ)
    assert seen["system"] == agent_replay.SKILL.read_text(encoding="utf-8")
    assert seen["first"].endswith(f"This conversation just ended on {DLS[0].date}. "
                                  "Follow your memory skill.")  # fmt: skip
    assert DLS[0].facts[0]["user_text"] in seen["first"]
    assert seen["tools"] == ["recall", "remember", "link", "subjects"]


def test_a_writer_question_is_answered_from_gold(tmp_path):
    class Asker(Writer):
        def create(self, *, system, messages, tools, max_tokens):
            if len(messages) == 1:
                ent = SYN.expected["entities"][0]
                return Reply(
                    [{"type": "text", "text": f"Which {ent['name']} do you mean?"}], "end_turn",
                    {"input_tokens": 1, "output_tokens": 1})  # fmt: skip
            self.answer = messages[-1]["content"]
            return Reply([{"type": "text", "text": "ok"}], "end_turn",
                                      {"input_tokens": 1, "output_tokens": 1})  # fmt: skip

    w = Asker(DLS, SYN)
    rec = agent_replay.run_all(DLS[:1], SYN, factory_p3(), tmp_path / "q.db", w, 5.0, D.TZ)
    assert rec["dialogues"][0]["questions"] == 1 and hasattr(w, "answer")


# ---- scoring: a correct writer scores perfectly, every broken one fails its metric ---------------


def test_scripted_perfect_writer_meets_every_b_w5_threshold(runs):
    m = metrics(runs["perfect"])
    assert m["capture"] == 1 and m["noise_rate"] == 0 and m["about_fill"] == 1
    assert m["conflict_action"] == 1 and m["conflicts_surfaced"] >= 20
    assert m["wrong_invalidation"] == 0 and m["slot_accuracy"] == 1 and m["atomicity"] == 1
    assert m["run_complete"] == 1
    assert m["capture_ci_lo"] <= m["capture"] <= m["capture_ci_hi"]


@pytest.mark.parametrize(
    ("mode", "metric", "bad"),
    [
        ("lazy", "capture", lambda v: v < 0.80),
        ("paraphrase_lossy", "capture", lambda v: v < 0.80),  # alignment needs the relation cue
        ("noisy", "noise_rate", lambda v: v > 0.10),
        ("no_link", "conflict_action", lambda v: v < 0.85),
        ("wrong_rel", "conflict_action", lambda v: v < 0.85),
        ("no_link", "slot_accuracy", lambda v: v < 1),
    ],
)
def test_broken_writers_fail_the_right_metric(runs, mode, metric, bad):
    m = metrics(runs[mode])
    assert bad(m[metric]), (mode, metric, m[metric])


def test_noise_counts_ignore_markers_and_unaligned_memories(runs):
    assert metrics(runs["noisy"])["noise_rate"] > 0.10
    assert metrics(runs["paraphrase_lossy"])["noise_rate"] == 1.0


def test_score_run_is_deterministic(runs):
    assert b_w5.score_run(runs["noisy"]) == b_w5.score_run(runs["noisy"])


def test_incomplete_run_cannot_pass(tmp_path):
    run = run_mode(tmp_path, "perfect", budget=0.30)
    assert metrics(run)["run_complete"] == 0


# ---- the audit sheet -----------------------------------------------------------------------------


def test_audit_sheet_has_30_aligned_and_20_unaligned_sampled_with_seed_7(runs, tmp_path):
    run = runs["noisy"]
    out = tmp_path / "audit.tsv"
    n = b_w5.write_audit_sheet(run, out)
    assert n == {"aligned": 30, "unaligned": 20}
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    assert [r["kind"] for r in rows].count("aligned") == 30
    out2 = tmp_path / "audit2.tsv"
    b_w5.write_audit_sheet(run, out2)
    assert out.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")  # seed 7
    _, details = b_w5.score_run(run)
    aligned = {r["memory"] for r in rows if r["kind"] == "aligned"}
    assert all(int(a[1:]) not in details["unaligned_memories"] for a in aligned)


def test_apply_audit_counts_wrong_alignments_and_misses(runs, tmp_path):
    out = tmp_path / "audit.tsv"
    b_w5.write_audit_sheet(runs["noisy"], out)
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    with pytest.raises(ValueError, match="fill human_check"):
        b_w5.apply_audit(out)
    for r in rows:
        r["human_check"] = "y" if r["kind"] == "aligned" else "n"
    rows[0]["human_check"] = "n"  # one aligned memory judged wrong
    rows[-1]["human_check"] = "y"  # one unaligned memory that should have aligned
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    got = b_w5.apply_audit(out)
    assert got["alignment_audit_wrong"] == 1 and got["alignment_audit_aligned_n"] == 30
    assert got["alignment_audit_miss_rate"] == pytest.approx(1 / 20)


# ---- the suite and its gate rows -----------------------------------------------------------------


def test_suite_reads_a_run_file_and_the_audit(runs, tmp_path):
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({**runs["perfect"], "seed": 101}), encoding="utf-8")
    sheet = tmp_path / "audit.tsv"
    b_w5.write_audit_sheet(runs["perfect"], sheet)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8"), delimiter="\t"))
    with sheet.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows({**r, "human_check": "y" if r["kind"] == "aligned" else "n"} for r in rows)
    ctx = Context(tmp_path, factory_p3(), list_tools,
                  {"w5": b_w5.W5Setup(run_file, sheet, "test")})  # fmt: skip
    res = b_w5.run(ctx)
    assert res.metrics["alignment_audit_wrong"] == 0 and res.metrics["capture"] == 1
    assert res.extra["cost"]["usd"] > 0 and res.tier == "t2"
    # without a filled sheet the audit metric is NaN, so its gate row fails
    res2 = b_w5.run(Context(tmp_path, factory_p3(), list_tools,
                            {"w5": b_w5.W5Setup(run_file, None, "test")}))  # fmt: skip
    assert math.isnan(res2.metrics["alignment_audit_wrong"])


def test_suite_is_unavailable_without_a_recorded_run(tmp_path):
    with pytest.raises(BackendUnavailable, match="w5-run"):
        b_w5.run(Context(tmp_path, factory_p3(), list_tools, {}))


def test_anthropic_llm_request_shape_and_usage_mapping(monkeypatch):
    """No network: a fake `anthropic` module records the request the writer model would get."""
    import sys
    import types

    from bench.llm import AnthropicLLM

    seen = {}

    class Block:
        def model_dump(self, exclude_none=True):
            return {"type": "text", "text": "ok"}

    class Usage:
        input_tokens, output_tokens = 100, 20
        cache_creation_input_tokens, cache_read_input_tokens = None, 5000

    class Messages:
        def create(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(content=[Block()], stop_reason="end_turn", usage=Usage())

    mod = types.ModuleType("anthropic")
    mod.Anthropic = lambda: types.SimpleNamespace(messages=Messages())
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    llm = AnthropicLLM()
    reply = llm.create(system="skill", messages=[{"role": "user", "content": "hi"}],
                       tools=[{"name": "recall"}], max_tokens=4096)  # fmt: skip
    assert seen["model"] == "claude-sonnet-5-5" and seen["output_config"] == {"effort": "low"}
    assert seen["cache_control"] == {"type": "ephemeral"} and seen["max_tokens"] == 4096
    assert "temperature" not in seen and "top_p" not in seen  # rejected by the model
    assert reply.usage == {"input_tokens": 100, "output_tokens": 20,
                           "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5000}  # fmt: skip
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


# ---- failure analysis (bench w5-failures) ----------------------------------------------------------


def test_failures_clean_on_perfect_writer(runs):
    from bench import w5_failures

    rep = w5_failures.analyse(runs["perfect"], 101, "M")
    assert rep["about_miss_n"] == 0 and rep["conflict_fail_n"] == 0


def test_failures_group_conflict_misses_by_action(runs):
    from bench import w5_failures

    nolink = w5_failures.analyse(runs["no_link"], 101, "M")
    assert nolink["conflict_fail_n"] > 0
    assert set(nolink["conflict_fail_by_action"]) == {"no_call"}
    wrong = w5_failures.analyse(runs["wrong_rel"], 101, "M")
    assert set(wrong["conflict_fail_by_action"]) == {"relates_to"}
    m = metrics(runs["no_link"])
    assert nolink["conflict_fail_n"] == round(m["conflicts_surfaced"] * (1 - m["conflict_action"]))


def test_failures_about_misses_are_counted_like_the_scorer(runs):
    import copy

    from bench import w5_failures

    run = copy.deepcopy(runs["perfect"])
    for mem in run["snapshot"]["memories"][:3]:
        mem["about"] = []
    rep = w5_failures.analyse(run, 101, "M")
    base = metrics(runs["perfect"])
    assert rep["about_miss_n"] == 3
    assert rep["about_miss_by_cause"] == {"about_empty": 3}
    assert rep["about_fill"] < base["about_fill"]
    assert "text" not in rep["about_misses"][0]
