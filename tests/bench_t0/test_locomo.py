"""B-LoCoMo (`bench locomo-run`, MEM2-196): the pinned file, session dates through `ingest_session`,
the official per-category scoring (snap-research/locomo `task_eval/evaluation.py`), the Mem0-paper judge
over categories 1-4, the adversarial rule, the hard budget. Hash models and a fake answerer/judge: no
network, no API key, no models. Fixture: bench/fixtures/locomo/locomo_mini.json (LoCoMo-shaped)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bench import e1, locomo, records
from bench.adapters.locomo_ours import OursLocomoBackend
from bench.cli import app
from bench.datasets import locomo as data
from bench.llm import Reply
from mem.config import stub_config
from mem.gate import Gate
from mem.testing import HashEmbedder, OverlapReranker

FILE = Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "locomo" / "locomo_mini.json"
CONVS = data.load(FILE)
GATE = Gate(1.0, 0.0, 0.0, 0.0, 0.0)  # keep everything: this tests plumbing, not the gate


class FakeLLM:
    def __init__(self, reply, model="claude-sonnet-5-5", usage=None):
        self.model, self.reply_fn = model, reply
        self.usage = usage or {"input_tokens": 1000, "output_tokens": 50}
        self.prompts = []

    def create(self, *, system, messages, tools, max_tokens):
        prompt = messages[0]["content"]
        self.prompts.append({"prompt": prompt, "max_tokens": max_tokens})
        return Reply([{"type": "text", "text": self.reply_fn(prompt)}], "end_turn", self.usage)


class FakeOpened:
    def __init__(self, conv):
        self.conv, self.closed = conv, False

    def recall(self, query):
        return f"MEM[{self.conv.sample_id}] {query}"

    def close(self):
        self.closed = True


def run(convs=CONVS, answer=lambda p: "Lisbon", judge=lambda p: '{"label": "CORRECT"}', **kw):
    a, j = FakeLLM(answer), FakeLLM(judge, usage={"input_tokens": 400, "output_tokens": 30})
    kw.setdefault("budget_usd", 5.0)
    opened = []

    def open_conv(conv):
        opened.append(FakeOpened(conv))
        return opened[-1]

    return locomo.run_locomo(convs, open_conv, a, j, **kw), a, j, opened


# ---- the pinned file and its parsing ------------------------------------------------------------


def test_pin_is_the_official_file_by_sha256_and_size():
    assert data.SHA256 == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    assert data.BYTES == 2_805_274 and len(data.COMMIT) == 40


def test_verify_accepts_the_pinned_bytes_and_refuses_anything_else(tmp_path, monkeypatch):
    f = tmp_path / data.FILE
    f.write_bytes(b"[]")
    monkeypatch.setattr(data, "BYTES", 2)
    with pytest.raises(ValueError, match="sha256"):
        data.verify(f)
    monkeypatch.setattr(data, "SHA256", data.sha256_file(f))
    assert data.verify(f) == data.SHA256
    f.write_bytes(b"[1]")
    with pytest.raises(ValueError, match="bytes"):
        data.verify(f)


@pytest.mark.parametrize(
    ("raw", "iso"),
    [("1:56 pm on 8 May, 2023", "2023-05-08T13:56:00Z"),
     ("12:09 am on 13 September, 2023", "2023-09-13T00:09:00Z"),
     ("12:00 pm on 1 January, 2024", "2024-01-01T12:00:00Z")],
)  # fmt: skip
def test_session_dates_parse_to_utc_iso(raw, iso):
    assert data.parse_date(raw) == iso


def test_conversations_keep_sessions_in_numeric_order_with_their_dates_and_rounds():
    c = CONVS[0]
    assert [s.id for s in c.sessions] == ["D1", "D2", "D3"]
    assert [s.date for s in c.sessions] == ["2023-05-08T13:56:00Z", "2023-06-03T09:05:00Z",
                                            "2023-07-21T23:30:00Z"]  # fmt: skip
    # speaker_a opens a round; speaker_b's turns after it are its reply; a lone speaker_b turn first
    # is a round with an empty user side. Turns carry the speaker name; an image adds its caption.
    assert c.sessions[0].rounds[0] == ("Nora: I joined a pottery class yesterday.",
                                       "Owen: Nice! I adopted a dog named Biscuit.")  # fmt: skip
    assert c.sessions[0].rounds[1:] == [("Nora: Send a photo!", ""), ("Nora: Also I love it.", "")]
    assert c.sessions[1].rounds[0][0] == ""
    assert "[image: a photo of a dog on a trail]" in c.sessions[1].rounds[0][1]
    assert [q.category for q in c.qa] == [4, 2, 1, 3, 5]
    assert CONVS[1].qa[1].answer == "2024"  # a numeric gold is a string
    assert c.qa[4].answer is None  # adversarial: no gold, the lure is never shown


# ---- ingest: dates reach the store --------------------------------------------------------------


def test_ingest_session_gets_each_sessions_date_and_source(tmp_path):
    b = OursLocomoBackend(tmp_path, cfg=stub_config(), models="hash", gate=GATE,
                          loaded=(HashEmbedder(), OverlapReranker()))  # fmt: skip
    opened = b.open_conversation(CONVS[0])
    try:
        rows = opened.store.conn.execute(
            "SELECT source_ref, valid_from FROM nodes WHERE origin = 'import' ORDER BY valid_from"
        ).fetchall()
        assert rows == [("locomo:conv-t1:D1", "2023-05-08T13:56:00Z"),
                        ("locomo:conv-t1:D2", "2023-06-03T09:05:00Z"),
                        ("locomo:conv-t1:D3", "2023-07-21T23:30:00Z")]  # fmt: skip
        text = opened.recall("Where is Nora moving? Lisbon")
        assert "Lisbon" in text
        assert opened.as_of == "2023-07-22T23:30:00Z"  # the day after the last session
    finally:
        opened.close()
        b.close()


# ---- per-category scoring (official rules) -------------------------------------------------------


def test_f1_stems_drops_articles_and_commas_like_the_official_scorer():
    assert locomo.f1_score("The dogs running", "dog run") == 1.0
    assert locomo.f1_score("Lisbon", "Paris") == 0
    assert locomo.f1_score("7 May 2023", "May 7 2023") == 1.0
    assert 0 < locomo.f1_score("in May 2023 probably", "7 May 2023") < 1


def test_multi_answer_f1_for_category_1_splits_on_commas():
    assert locomo.multi_f1("dog, pottery", "dog, pottery") == 1.0
    assert locomo.multi_f1("a dog", "dog, pottery") == pytest.approx(0.5)


def test_category_rules():
    s = locomo.score_item
    assert s(1, "pottery, a dog", "dog, pottery") == 1.0
    assert s(2, "7 May 2023", "7 May 2023") == 1.0
    assert s(3, "Yes", "Yes; she is moving to Lisbon") == 1.0  # category 3: gold cut at ';'
    assert s(4, "Paris", "Lisbon") == 0
    assert s(5, "No information available", None) == 1
    assert s(5, "That is not mentioned in the memory.", None) == 1
    assert s(5, "blue", None) == 0


def test_the_run_scores_each_category_and_keeps_adversarial_out_of_the_judge():
    rec, _a, j, _ = run(answer=lambda p: "No information available" if "miles" in p else "Lisbon")
    assert rec["complete"] and len(rec["items"]) == rec["planned"] == 7
    assert len(j.prompts) == 5  # categories 1-4 only (2 adversarial items are never judged)
    m, extra = locomo.score_run(rec)
    per = extra["per_category"]
    assert {k: v["n"] for k, v in per.items()} == {"multi-hop": 1, "temporal": 2, "open-domain": 1,
                                                  "single-hop": 1, "adversarial": 2}  # fmt: skip
    assert m["judge_accuracy"] == 1.0 and m["n_judged"] == 5
    assert per["single-hop"]["f1"] == 1.0 and per["temporal"]["f1"] == 0
    assert per["adversarial"]["accuracy"] == 0.5  # "Lisbon" for one, abstained on the other
    assert m["adversarial_accuracy"] == 0.5 and m["n_adversarial"] == 2


def test_the_adversarial_lure_never_reaches_the_answerer_or_the_judge():
    rec, a, j, _ = run()
    seen = "\n".join(p["prompt"] for p in a.prompts + j.prompts)
    assert "adversarial_answer" not in seen and "blue" not in seen and "five" not in seen
    assert all(i["gold"] is None for i in rec["items"] if i["category"] == 5)


def test_judge_label_parsing():
    p = locomo.parse_label
    assert p('{"label": "CORRECT"}') is True
    assert p('Same date. {"label": "WRONG"}') is False
    assert p("The topic matches.\nCORRECT") is True
    assert p("both CORRECT and WRONG") is False  # ambiguous: not credited
    assert p("") is False


def test_a_reply_without_a_label_is_counted_as_flagged_not_silently_scored():
    # MEM2-269 (ADR 0029) for LoCoMo: its judge is Mem0's (200 tokens, a JSON label), not the LME
    # yes/no judge, so its cap and parser stay; a reply with neither label is counted, as in B-E1
    rec, *_ = run()
    core = [i for i in rec["items"] if i["category"] != 5]
    assert locomo.score_run(rec)[0]["judge_flagged"] == 0.0
    core[0]["judge_raw"] = "The answer touches on the same topic"  # cut before the label
    core[1]["judge_raw"] = "both CORRECT and WRONG"  # ambiguous
    metrics, _ = locomo.score_run(rec)
    assert metrics["judge_flagged"] == 2.0
    assert (
        locomo.judge_label(core[0]["judge_raw"]) is None and locomo.judge_label("CORRECT") is True
    )


def test_judge_prompt_is_the_mem0_template():
    assert "label an answer to a question as" in locomo.JUDGE_TEMPLATE
    assert "{question}" in locomo.JUDGE_TEMPLATE and "{gold_answer}" in locomo.JUDGE_TEMPLATE
    assert "{generated_answer}" in locomo.JUDGE_TEMPLATE


def test_the_judge_is_recomputed_from_the_run_file():
    rec, *_ = run()
    for i in rec["items"]:
        i["judge_label"] = True  # tampered; the raw reply is what counts
        if i["category"] != 5:
            i["judge_raw"] = '{"label": "WRONG"}'
    assert locomo.score_run(rec)[0]["judge_accuracy"] == 0.0


# ---- the budget cap -----------------------------------------------------------------------------


def test_the_budget_cap_stops_the_run_before_going_over_and_keeps_what_finished():
    full, *_ = run()
    per_item = full["cost"]["usd"] / len(full["items"])
    budget = per_item * 3.5
    rec, _a, _j, opened = run(budget_usd=budget)
    assert not rec["complete"] and rec["stopped"] and "budget" in rec["stopped"]
    assert 0 < len(rec["items"]) < rec["planned"]
    assert rec["cost"]["usd"] <= budget
    assert all(o.closed for o in opened)  # the open conversation is closed on the stop


def test_a_budget_smaller_than_one_call_makes_no_call():
    rec, a, j, _ = run(budget_usd=0.0005)
    assert rec["items"] == [] and a.prompts == [] and j.prompts == [] and not rec["complete"]


# ---- the CLI ------------------------------------------------------------------------------------


def invoke(*args):
    return CliRunner().invoke(app, ["locomo-run", "--locomo-file", str(FILE), "--out", "o.json",
                                    *args])  # fmt: skip


def test_cli_refuses_without_yes_and_states_the_cap():
    r = invoke("--budget-usd", "3.5")
    assert r.exit_code == 2 and "--yes" in r.output and "$3.50" in r.output


def test_cli_end_to_end_on_the_fixture_writes_the_run_file_and_a_ledger_line(monkeypatch, tmp_path):
    monkeypatch.setattr("bench.llm.AnthropicLLM",
                        lambda model, effort: FakeLLM(lambda p: "Lisbon"))  # fmt: skip
    monkeypatch.setattr(data, "verify", lambda p: "0" * 64)
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(records, "LEDGER", led)
    out = tmp_path / "run.json"
    r = CliRunner().invoke(app, ["locomo-run", "--locomo-file", str(FILE), "--out", str(out),
                                 "--yes", "--embedder", "hash", "--budget-usd", "5",
                                 "--ledger", "final run", "--who", "t"])  # fmt: skip
    assert r.exit_code == 2 and "UNAVAILABLE" in r.output and "gate" in r.output  # no gate given
    gate = tmp_path / "gate.json"
    from dataclasses import asdict

    cfg = stub_config()
    gate.write_text(json.dumps(asdict(Gate(1.0, 0.0, 0.0, 0.0, 0.0,
                                          fingerprint_hash=cfg.fp_all()))), encoding="utf-8")  # fmt: skip
    monkeypatch.setattr("bench.adapters.lme_ours.start_models",
                        lambda *a, **k: type("S", (), {"models": (HashEmbedder(), OverlapReranker()),
                                                      "stop": lambda self: None})())  # fmt: skip
    r = CliRunner().invoke(app, ["locomo-run", "--locomo-file", str(FILE), "--out", str(out),
                                 "--yes", "--embedder", "hash", "--budget-usd", "5",
                                 "--gate", str(gate), "--ledger", "final run", "--who", "t"])  # fmt: skip
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["complete"] and rec["dataset"]["name"] == "LoCoMo (locomo10)"
    assert rec["dataset"]["commit"] == data.COMMIT and len(rec["items"]) == 7
    assert "judge_accuracy" in r.output and "adversarial" in r.output
    (line,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert line["kind"] == "locomo-run" and line["split"] != "test"  # not a P3 look
    assert line["why"] == "final run"


# ---- what the answerer reads: the index alone has no chunk text ---------------------------------


def test_recall_alone_lists_titles_and_expanded_adds_the_chunk_text():
    from bench.adapters.lme_ours import OursLmeBackend
    from bench.datasets import lme

    qs = lme.load_questions(
        Path(__file__).resolve().parents[2] / "bench" / "fixtures" / "lme" / "lme_s_mini.json"
    )
    q = qs[0]
    gold = next(t["content"] for s in q["haystack_sessions"] for t in s if t.get("has_answer"))
    texts = {}
    for expand in (False, True):
        b = OursLmeBackend(cfg=stub_config(), models="hash", gate=GATE, expand=expand,
                           loaded=(HashEmbedder(), OverlapReranker()))  # fmt: skip
        texts[expand] = e1.mem_context(b.retrieve(q, lme.haystack_of(q), want_legs=False))
        b.close()
    assert gold not in texts[False] and gold in texts[True]
    assert texts[True].startswith("## mem recall\n" + texts[False])


def delivered(opened, query):
    """The rows recall delivers for `query`, as (session id, chunk text) from the store itself."""
    from mem import read

    res = read.recall(opened.store, query=query, as_of=opened.as_of)
    ids = [int(r["id"][1:]) for r in res.structured["rows"] if r["id"].startswith("c")]
    rows = opened.store.conn.execute(
        "SELECT c.id, n.source_ref, c.text FROM chunks c JOIN nodes n ON n.id = c.node_id"
    ).fetchall()
    by_id = {cid: (src.rsplit(":", 1)[1], text) for cid, src, text in rows}
    return [by_id[i] for i in ids], [t for i, (_, t) in by_id.items() if i not in ids]


def test_locomo_answerer_context_holds_every_delivered_rows_full_text(tmp_path):
    b = OursLocomoBackend(tmp_path, cfg=stub_config(), models="hash", gate=GATE,
                          loaded=(HashEmbedder(), OverlapReranker()))  # fmt: skip
    seen = 0
    try:
        for conv in CONVS:
            opened = b.open_conversation(conv)
            dates = {s.id: s.date for s in conv.sessions}
            try:
                for q in conv.qa:
                    ctx = opened.recall(q.question)
                    rows, others = delivered(opened, q.question)
                    if rows:
                        assert ctx.startswith("## mem recall\n")
                        assert f"\n\n{e1.EXPANDED_HEADING}\n" in ctx
                    for sid, text in rows:  # whole, headed by its session and date
                        assert text in ctx
                        assert f"[session {sid}, round " in ctx and dates[sid] in ctx
                    for text in others:  # and no row it did not deliver
                        assert text not in ctx
                    seen += len(rows)
            finally:
                opened.close()
    finally:
        b.close()
    assert seen and b.truncated_rows == {}


def test_locomo_run_records_expanded_context_and_no_cut_rows(monkeypatch, tmp_path):
    from dataclasses import asdict

    llms = []

    def fake_llm(model, effort):
        llms.append(FakeLLM(lambda p: "Lisbon"))
        return llms[-1]

    monkeypatch.setattr("bench.llm.AnthropicLLM", fake_llm)
    monkeypatch.setattr(data, "verify", lambda p: "0" * 64)
    monkeypatch.setattr(records, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr("bench.adapters.lme_ours.start_models",
                        lambda *a, **k: type("S", (), {"models": (HashEmbedder(), OverlapReranker()),
                                                      "stop": lambda self: None})())  # fmt: skip
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(asdict(Gate(1.0, 0.0, 0.0, 0.0, 0.0,
                                          fingerprint_hash=stub_config().fp_all()))))  # fmt: skip
    out = tmp_path / "run.json"
    r = CliRunner().invoke(app, ["locomo-run", "--locomo-file", str(FILE), "--out", str(out),
                                 "--yes", "--embedder", "hash", "--gate", str(gate),
                                 "--budget-usd", "5", "--limit", "3"])  # fmt: skip
    assert r.exit_code == 0, r.output
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["mem_context"] == "expanded" and rec["truncated_rows"] == {}
    assert any(e1.EXPANDED_HEADING in p["prompt"] for p in llms[0].prompts)


def test_answer_prompt_is_the_locomo_paper_wording_with_the_temporal_suffix():
    rec, a, _, _ = run()
    assert locomo.PROMPT_VERSION == 2 and rec["prompt_version"] == 2
    assert rec["prompts_sha256"] == locomo.prompts_sha256()
    by_cat = {i["question"]: i["category"] for i in rec["items"]}
    for p in (x["prompt"] for x in a.prompts):
        assert "\n\nBased on the above context, write an answer in the form of a short phrase" in p
        assert p.endswith(" Short answer:") and "No information available" not in p
    temporal = [
        x["prompt"]
        for x in a.prompts
        if by_cat[
            x["prompt"]
            .split("Question: ")[1]
            .rsplit(" Short answer:")[0]
            .removesuffix(locomo.TEMPORAL_SUFFIX)
        ]
        == locomo.TEMPORAL
    ]
    assert all(p.count(locomo.TEMPORAL_SUFFIX) == 1 for p in temporal)
    assert sum(locomo.TEMPORAL_SUFFIX in x["prompt"] for x in a.prompts) == len(temporal)


def test_locomo_run_cli_answerer_has_no_effort(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "bench.llm.AnthropicLLM",
        lambda model, effort: seen.append(effort) or FakeLLM(lambda p: "x"),
    )
    monkeypatch.setattr(data, "verify", lambda p: "0" * 64)
    CliRunner().invoke(
        app, ["locomo-run", "--locomo-file", str(FILE), "--out", str(tmp_path / "r.json"), "--yes"]
    )
    assert seen[:2] == [None, "low"]
