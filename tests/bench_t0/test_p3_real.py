"""The P3 T0 suites through the gate adapter (`bench.adapters.p3_ours.P3Backend`) on the REAL
`mem.store.Store` / `mem.serve.MemServer`, with the `mem.testing` stand-ins handed in (CPU, no
network, no serve process). test_p3_suites.py runs the same suites on the stand-in server to prove
each check can fail; this file proves the production code passes the exact ones.

Only the exact T0 checks are asserted. The SYN metrics that depend on what the reranker surfaces
(B-W2 paraphrase_surfacing / final_duplicate_rate, B-W3, the B-R5 SYN probes) are model quality:
with the stub reranker they are far below the gate and are judged on the real models (T1,
deliverables/runbooks/p3-gate.md), not here.
"""

import re

import pytest

from bench import agent_replay
from bench.adapters.p3_ours import P3Backend
from bench.llm import Ledger, Reply
from bench.suites import b_r5, b_r5_p3, b_w6_p3, b_w123
from bench.suites.base import Context, SuiteResult
from bench.synth import data as D
from bench.synth import dialogue as dlg
from bench.synth.generator import generate
from bench.synth.replay import Replay
from bench.synth.run import SynSetup
from mem.confmodel import ConfModel
from mem.gate import Gate
from mem.readguard import ReadGuard
from mem.readmodel import ReadModel
from mem.testing import HashEmbedder, OverlapReranker, stub_conf_model
from tests.bench_t0.fake_serve import list_tools

STUB_THETA_READ = 2.5  # OverlapReranker logit 6·overlap − 3: a pair needs > 91% word overlap


def stub_gate(cfg, theta_read: float | None = STUB_THETA_READ) -> Gate:
    """The stub-model gate of tests/helpers.py (p = σ(z), keep p ≥ .5, no gap rule) with a θ_conf,
    the stub read model and (ADR 0018) a read_guard at `theta_read` (None: no guard)."""
    rm = ReadModel.build(["rerank"], [1.0], 0.0)  # p = σ(z): the same numbers as the Platt stub
    cm = stub_conf_model()
    rg = None if theta_read is None else ReadGuard.build(
        theta_read, ConfModel.from_dict(cm).fingerprint()).to_dict()  # fmt: skip
    return Gate(1.0, 0.0, 0.5, 0.5, 0.0, theta_conf=0.5, conf_model=cm, fit_id="t0-stub",
                fingerprint_hash=cfg.fp_all(), read_model=rm.to_dict(), read_guard=rg)  # fmt: skip


@pytest.fixture
def backend():
    b = P3Backend(models="hash", loaded=(HashEmbedder(256), OverlapReranker()))
    b.gate = stub_gate(b.cfg)
    yield b
    b.close()


def ctx(tmp_path, backend) -> Context:
    return Context(tmp_path, backend.factory, list_tools, {"syn": SynSetup((1,), "S", "fixture")})


def failed(res: SuiteResult) -> dict[str, str]:
    return {i["check"]: i.get("detail", "") for i in res.items if "check" in i and not i["ok"]}


def test_factory_defaults_to_the_fixture_zone(tmp_path, backend):
    """Suites that open a client without `tz` (B-R5 (e), the SYN probes) get Europe/Amsterdam, as
    from `ours.open_client` and the stand-in server; UTC here failed case (e) on the real store."""
    c = backend.factory(tmp_path / "z.db")
    try:
        assert c._server.store.tz == "Europe/Amsterdam"
    finally:
        c.close()


# ---- B-R5 T0 cases (a)-(f), one test each ------------------------------------------------------

R5_CASES = [
    pytest.param(b_r5_p3.case_a_link, "a_", id="a"),
    pytest.param(b_r5_p3.case_b, "b_", id="b"),
    pytest.param(b_r5_p3.case_c, "c_", id="c"),
    pytest.param(b_r5_p3.case_d, "d_", id="d"),
    pytest.param(b_r5.case_e, "e_", id="e"),
    pytest.param(b_r5_p3.case_f, "f_", id="f"),
]


@pytest.mark.parametrize(("case", "prefix"), R5_CASES)
def test_r5_t0_case_passes_on_the_real_server(tmp_path, backend, case, prefix):
    res = SuiteResult("B-R5")
    case(ctx(tmp_path, backend), res)
    checks = [i["check"] for i in res.items if "check" in i]
    assert checks and all(c.startswith(prefix) for c in checks), checks
    assert failed(res) == {}


# ---- B-W2 exact, B-W1 table cases, B-W6 -------------------------------------------------------


def test_w2_exact_checks_pass_on_the_real_server(tmp_path, backend):
    """Exact re-saves are `exists` (exists_rate = 1), `remember` never invalidates, and the
    concurrency and superseded-original (n2) cases pass."""
    res = b_w123.run_b_w2(ctx(tmp_path, backend))
    assert res.metrics["exists_rate"] == 1
    assert res.metrics["wrong_invalidation_compat"] == 0
    assert res.metrics["case_failures"] == 0, failed(res)
    names = {i["check"] for i in res.items if "check" in i}
    assert {"concurrent_same_text_one_row_other_exists",
            "n2_resave_superseded_text_is_saved_not_exists"} <= names  # fmt: skip


def test_w1_table_cases_pass_on_the_real_server(tmp_path, backend):
    res = b_w123.run_b_w1(ctx(tmp_path, backend))
    assert res.metrics["table_case_failures"] == 0, failed(res)
    assert res.metrics["op_errors"] == 0


def test_w6_all_items_pass_on_the_real_server(tmp_path, backend):
    res = b_w6_p3.run(ctx(tmp_path, backend))
    assert res.violations == [] and res.metrics["violations"] == 0, res.violations
    names = {i["check"] for i in res.items}
    for n in range(1, 12):
        assert any(x.startswith(f"{n}_") for x in names), f"item {n} not run"


# ---- ADR 0027: `new:` against an existing name of another type --------------------------------


def test_oracle_replay_keeps_cross_kind_pairs_apart_on_the_real_server(tmp_path, backend):
    """SYN worlds hold project/org pairs sharing one name. The oracle's `new:` for the second one
    now gets `needs_choice` (ADR 0027); it must answer with `distinct_new`, not fail."""
    syn = generate(1, "S")
    c = backend.factory(tmp_path / "r.db", tz=D.TZ, now=syn.ops[0]["now"])
    try:
        rep = Replay(c, syn)
        logs = rep.run()
        assert [lg.error for lg in logs if lg.error] == []
        conn = c._server.store.conn
        pairs = [e for e in syn.expected["entities"] if (e.get("group") or "").startswith("X")]
        assert pairs
        for e in pairs:
            assert e["key"] in rep.eid
            assert conn.execute("SELECT type, name FROM nodes WHERE id = ?",
                                (rep.eid[e["key"]],)).fetchone() == (e["type"], e["name"])  # fmt: skip
    finally:
        c.close()


def test_w5_writer_loop_resends_after_needs_choice(tmp_path, backend):
    """`bench w5-run`: the needs_choice text goes back to the writer, whose resend with `#e<id>`
    links the existing entity (as a real agent would)."""
    syn = generate(101, "M")
    dl = dlg.build_dialogues(syn)[0]
    c = backend.factory(tmp_path / "w.db", tz=D.TZ, now=dl.start)
    seen = []

    class Resender:
        model = "claude-sonnet-5-5"

        def create(self, *, system, messages, tools, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            turn = sum(m["role"] == "assistant" for m in messages)
            about = ["new:topic:Juniper"]
            if turn == 1:
                seen.append(messages[-1]["content"][0]["content"])
                ids = re.findall(r"^e(\d+) · place · Juniper$", seen[0], re.MULTILINE)
                about = [f"#e{ids[0]}"] if ids else about
            if turn == 2:
                seen.append(messages[-1]["content"][0]["content"])
                return Reply([{"type": "text", "text": "Saved."}], "end_turn", usage)
            use = {
                "type": "tool_use",
                "id": f"t{turn}",
                "name": "remember",
                "input": {"text": "The offsite is in Juniper.", "type": "fact", "about": about},
            }
            return Reply([use], "tool_use", usage)

    try:
        place = c.call("remember", {"text": "Juniper is a town.", "type": "fact",
                                    "about": ["new:place:Juniper"]}).structured  # fmt: skip
        pid = int(place["subjects"][0].split()[0][1:])
        rec = agent_replay.run_dialogue(dl, syn, c, Resender(), Ledger("claude-sonnet-5-5", 5.0),
                                        "", agent_replay.tool_schemas())  # fmt: skip
    finally:
        c.close()
    assert seen[0].startswith("needs_choice") and f"e{pid} · place · Juniper" in seen[0]
    assert seen[1].startswith("saved") and f"e{pid} Juniper (matched)" in seen[1]
    assert [x["structured"]["status"] for x in rec["calls"]] == ["needs_choice", "saved"]
