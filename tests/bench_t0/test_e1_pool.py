"""B-E1 run validity pooled over the runs of one harness (`--oracle-pool`; the owner, 2026-10-27, MEM2-253).

VALID needs the pooled oracle (this run plus every pooled run of the same harness) >= 0.92; a single run
under 0.90 is a harness failure, INVALID whatever the pool (re-based on the strict judge: the owner,
2026-10-27, MEM2-260; was 0.94 / 0.92). A pooled file of another harness, or a run
counted twice, is refused with the cause named. Fake answerer and judge: no network, no API key, no models."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from bench import e1, gates, records
from bench.cli import app
from bench.suites import b_e1

from .test_e1 import FakeLLM, make_q

HARNESS = {"prompt_version": e1.PROMPT_VERSION, "prompts_sha256": e1.prompts_sha256(),
           "answerer": "claude-sonnet-5-5", "answerer_thinking": "between_tools",
           "judge": "claude-sonnet-5-5", "split": "tune", "judge_version": e1.JUDGE_VERSION}  # fmt: skip
OTHER = {"prompt_version": 1, "prompts_sha256": "0" * 64, "answerer": "claude-opus-5-5",
         "answerer_thinking": "off", "judge": "claude-haiku-4-5-20251001", "split": "test",
         "judge_version": 1}  # fmt: skip


def record(oracle_yes, n=125, tag="a", **harness):
    """A run record: mem all right, `oracle_yes` of `n` oracle answers right. `tag` makes the run's
    answers its own (two runs never share their items)."""
    items = []
    for i in range(n):
        for cond, ok in (("mem", True), ("oracle", i < oracle_yes)):
            items.append({"qid": f"q{i}", "type": "multi-session", "judge_kind": "multi-session",
                          "condition": cond, "context_chars": 10, "answer": f"{tag}-{cond}-{i}",
                          "judge_raw": "yes" if ok else "no", "correct": ok, "usd": 0.0})  # fmt: skip
    return {"items": items, "planned": len(items), "complete": True, "stopped": None,
            "cost": {"usd": 1.0}, "conditions": ["mem", "oracle"], "config": {"fp_all": "x"},
            "dataset": {"name": "LongMemEval_S cleaned", "split": "tune"},
            **HARNESS, **harness}  # fmt: skip


def write(tmp_path, name, rec):
    path = tmp_path / name
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def gate(run_path, pool=None, split="tune"):
    opts = {b_e1.OPT: b_e1.E1Setup(run_path, split, pool)}
    return gates.run_phase("P3", None, results_dir=None, only={"B-E1"}, options=opts, split=split)[
        0
    ]


def verdict(rec, pool_recs):
    pool = [{"file": f"p{i}", "sha256": "", "digest": e1.run_digest(r),
             "correct": e1.oracle_counts(r)[0], "n": e1.oracle_counts(r)[1]}
            for i, r in enumerate(pool_recs)]  # fmt: skip
    p = e1.oracle_pool(rec, pool)
    return p, e1.pool_invalid_reason(p)


# ---- the rule (re-based on the strict judge: the owner, 2026-10-27, MEM2-260) ----------------------


def test_the_thresholds():
    assert e1.ORACLE_POOL_MIN == 0.92 and e1.ORACLE_RUN_FLOOR == 0.90
    assert e1.ORACLE_MIN == e1.ORACLE_POOL_MIN


def test_a_pool_of_0921_with_every_run_over_090_is_valid():
    p, why = verdict(record(902, n=1000, tag="f1"), [record(940, n=1000, tag="r8c")])
    assert p["oracle_pooled"] == pytest.approx(0.921) and p["oracle_run"] == pytest.approx(0.902)
    assert why is None


def test_a_pool_of_0919_is_invalid():
    p, why = verdict(record(919, n=1000, tag="f1"), [])
    assert p["oracle_pooled"] == pytest.approx(0.919)
    assert why and "pooled oracle accuracy 0.919 < 0.92" in why
    p, why = verdict(record(902, n=1000, tag="f1"), [record(936, n=1000, tag="r8c")])
    assert p["oracle_pooled"] == pytest.approx(0.919) and why and "0.919 < 0.92" in why


def test_a_run_at_0896_is_invalid_even_when_the_pool_is_over_092():
    p, why = verdict(record(896, n=1000, tag="bad"), [record(1000, n=1000, tag="x")])
    assert p["oracle_pooled"] >= 0.92 and p["oracle_run"] == pytest.approx(0.896)
    assert why and "0.896 < 0.90" in why and "harness failure" in why


@pytest.mark.parametrize(
    "rec,pool",
    [
        (record(919, n=1000), []),  # the pool
        (record(896, n=1000, tag="bad"), [record(1000, n=1000, tag="x")]),  # the run floor
    ],
)
def test_the_invalid_message_prints_both_numbers(rec, pool):
    _, why = verdict(rec, pool)
    assert why and "pooled >= 0.92" in why and "every run >= 0.90" in why
    assert ">= 0.94" not in why and "< 0.94" not in why


def test_the_old_094_run_is_now_valid_alone():
    p, why = verdict(record(117, tag="f1"), [])  # 0.936: INVALID alone under the lenient-judge 0.94
    assert p["oracle_pooled"] == pytest.approx(0.936) and why is None


def test_the_floor_and_the_pool_line_are_at_their_boundaries():
    assert (
        verdict(record(90, n=100), [record(100, n=100, tag="b")])[1] is None
    )  # 0.900 is not < 0.90
    assert verdict(record(89, n=100), [record(100, n=100, tag="b")])[1]  # 0.890 < 0.90
    assert verdict(record(92, n=100), [])[1] is None  # alone: 0.92 is valid
    assert verdict(record(91, n=100), [])[1]


def test_the_printed_line():
    p, _ = verdict(record(117, tag="f1"), [record(119, tag="r8c")])
    assert e1.pool_line(p) == "oracle run=0.936, pooled=0.944 (n=250)"
    p, _ = verdict({**record(0), "items": [], "conditions": ["mem"]}, [])
    assert e1.pool_line(p) == "oracle run=n/a, pooled=n/a (n=0)"


# ---- refusals: another harness, a run counted twice --------------------------------------------


@pytest.mark.parametrize("field", e1.HARNESS_FIELDS)
def test_a_pooled_file_of_another_harness_is_refused_naming_the_field(tmp_path, field):
    assert set(e1.HARNESS_FIELDS) == set(HARNESS)
    other = write(tmp_path, "other.json", record(119, tag="o", **{field: OTHER[field]}))
    with pytest.raises(e1.PoolRefused, match=f"harness field {field} "):
        e1.load_pool([other], HARNESS)
    run = write(tmp_path, "run.json", record(117))
    r = CliRunner().invoke(app, ["gate", "P3", "--split", "tune", "--e1-run", str(run),
                                 "--oracle-pool", str(other)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and f"harness field {field} " in r.output
    # e1-run refuses it before any API call (no --yes, no key: the refusal comes first)
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(tmp_path / "o.json"),
                                 "--oracle-pool", str(other)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and f"harness field {field} " in r.output


def test_a_missing_harness_field_is_refused_too(tmp_path):
    rec = record(119, tag="old")
    del rec["answerer_thinking"]
    with pytest.raises(e1.PoolRefused, match="harness field answerer_thinking is None"):
        e1.load_pool([write(tmp_path, "old.json", rec)], HARNESS)


def test_a_finals_run_pools_with_the_tune_runs_of_its_harness(tmp_path):
    finals = dict(HARNESS, split="full")
    tune = write(tmp_path, "tune.json", record(119, tag="t"))
    assert e1.load_pool([tune], finals)[0]["n"] == 125
    other_full = write(tmp_path, "full.json", record(119, tag="f", split="full"))
    with pytest.raises(e1.PoolRefused, match="harness field split .*tune runs"):
        e1.load_pool([other_full], finals)
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(tmp_path / "o.json"),
                                 "--full", "--oracle-pool", str(other_full)])  # fmt: skip
    assert r.exit_code == 2 and "harness field split" in r.output


def test_a_run_counted_twice_is_refused(tmp_path):
    a = write(tmp_path, "a.json", record(119, tag="a"))
    copy = write(tmp_path, "copy.json", {**record(119, tag="a"), "note": "renamed"})
    for paths in ([a, a], [a, copy]):
        with pytest.raises(e1.PoolRefused, match="the same run as"):
            e1.load_pool(paths, HARNESS)
    r = CliRunner().invoke(
        app, ["gate", "P3", "--split", "tune", "--e1-run", str(a), "--oracle-pool", str(copy)]
    )  # the current run in its own pool
    assert r.exit_code == 2 and "REFUSED" in r.output and "the current run itself" in r.output
    r = CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(a),
                                 "--oracle-pool", str(a)])  # fmt: skip
    assert r.exit_code == 2 and "REFUSED" in r.output and "--out file" in r.output


def test_a_pooled_file_without_an_oracle_arm_is_refused(tmp_path):
    rec = record(0, tag="m")
    rec["items"] = [i for i in rec["items"] if i["condition"] == "mem"]
    with pytest.raises(e1.PoolRefused, match="no oracle arm"):
        e1.load_pool([write(tmp_path, "m.json", rec)], HARNESS)


# ---- the gate reads the pooled verdict --------------------------------------------------------


def test_gate_p3_reads_the_pooled_verdict(tmp_path):
    run = write(tmp_path, "run.json", record(114, tag="f1"))
    base = write(tmp_path, "tune-r8c.json", record(119, tag="r8c"))
    assert gate(run).status == "INVALID"  # alone: 0.912
    o = gate(run, [base])
    assert o.status == "PASS", o.detail
    o = gate(run, [base, write(tmp_path, "weak.json", record(100, tag="w"))])
    assert o.status == "INVALID" and "pooled oracle accuracy" in o.detail
    assert "pooled >= 0.92, every run >= 0.90" in o.detail and ">= 0.94" not in o.detail
    out = gates.run_phase("P3", None, results_dir=tmp_path / "r", only={"B-E1"}, split="tune",
                          options={b_e1.OPT: b_e1.E1Setup(run, "tune", [base])})[0]  # fmt: skip
    rr = json.loads(out.run_file.read_text(encoding="utf-8"))
    assert rr["metrics"]["oracle_pooled"] == pytest.approx(0.932)
    assert rr["metrics"]["oracle_pool_n"] == 250 and rr["metrics"]["oracle_valid"] == 1.0
    assert rr["extra"]["oracle"]["oracle_pool"][0]["file"] == str(base)


def test_gate_p3_uses_the_pool_the_run_recorded_and_checks_its_sha256(tmp_path):
    base = write(tmp_path, "tune-r8c.json", record(119, tag="r8c"))
    sha = hashlib.sha256(base.read_bytes()).hexdigest()
    run = write(tmp_path, "run.json", {**record(114, tag="f1"),
                                        "oracle_pool": [{"file": str(base), "sha256": sha}]})  # fmt: skip
    assert gate(run).status == "PASS"
    assert gate(run, []).status == "INVALID"  # an explicit empty pool: the run alone
    base.write_text(json.dumps(record(125, tag="r8c")), encoding="utf-8")  # the file changed
    o = gate(run)
    assert o.status == "FAIL" and "the run recorded" in o.detail
    r = CliRunner().invoke(app, ["gate", "P3", "--split", "tune", "--e1-run", str(run)])
    assert r.exit_code == 2 and "REFUSED" in r.output and "the file changed" in r.output


# ---- e1-run: the printed line and the record fields -------------------------------------------


@pytest.fixture
def e1_env(monkeypatch, tmp_path):
    """`bench e1-run --condition oracle` over 125 questions; the judge says yes to the first 114."""
    from bench.datasets import lme

    qs = [make_q(f"s{i}", "multi-session") for i in range(125)]
    right = {f"How many s{i}?" for i in range(114)}

    runs = []

    def fake(model, effort):
        if effort is None:  # the answerer; each run's answers are its own
            runs.append(model)
            n = len(runs)
            return FakeLLM(lambda p: f"42 apples (run {n})", model=model)
        return FakeLLM(lambda p: "yes" if any(f"Question: {q}\n" in p for q in right) else "no",
                       model=model)  # fmt: skip

    monkeypatch.setattr("bench.llm.AnthropicLLM", fake)
    monkeypatch.setattr(lme, "read_manifest", lambda p: object())
    monkeypatch.setattr(lme, "load_questions", lambda f: qs)
    monkeypatch.setattr(lme, "verify_manifest", lambda q, m: None)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    rows = [type("R", (), {"question_id": q["question_id"]}) for q in qs]
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    monkeypatch.setattr(records, "LEDGER", tmp_path / "ledger.jsonl")
    return tmp_path


def e1_run(out, *extra):
    return CliRunner().invoke(app, ["e1-run", "--lme-file", "x.json", "--out", str(out), "--yes",
                                    "--condition", "oracle", "--budget-usd", "50", *extra])  # fmt: skip


def test_e1_run_prints_both_oracles_and_records_the_pool(e1_env):
    base = write(e1_env, "tune-r8c.json", record(119, tag="r8c"))
    out = e1_env / "f1.json"
    r = e1_run(out, "--oracle-pool", str(base))
    assert r.exit_code == 0, r.output
    assert "oracle run=0.912, pooled=0.932 (n=250)\nVALID\n" in r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["oracle_run"] == pytest.approx(0.912) and rec["oracle_pooled"] == pytest.approx(
        0.932
    )
    assert rec["oracle_pool_n"] == 250
    assert rec["oracle_pool"] == [{"file": str(base), "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                                   "oracle": 119 / 125, "n": 125}]  # fmt: skip
    # the next run of Round 9 pools with both: the record it reads is the one just written
    r = e1_run(e1_env / "f2.json", "--oracle-pool", str(base), "--oracle-pool", str(out))
    assert r.exit_code == 0 and "oracle run=0.912, pooled=0.925 (n=375)\nVALID\n" in r.output


def test_e1_run_without_a_pool_is_the_run_alone(e1_env):
    out = e1_env / "alone.json"
    r = e1_run(out)
    assert r.exit_code == 0, r.output
    assert (
        "oracle run=0.912, pooled=0.912 (n=125)\nINVALID: pooled oracle accuracy 0.912 < 0.92"
        in r.output
    )
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["oracle_pool"] == [] and rec["oracle_pool_n"] == 125


def test_e1_run_help_states_the_strict_judge_numbers():
    r = CliRunner().invoke(app, ["e1-run", "--help"], env={"COLUMNS": "400", "NO_COLOR": "1"})
    text = " ".join(r.output.replace("│", " ").split())
    assert "pooled oracle >= 0.92 and this run's >= 0.90" in text and ">= 0.94" not in text
