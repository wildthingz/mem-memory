"""Proof tests for the MEM2-222 review of bosmang/rv2-integrate (+ MEM2-248) at 43860d3 / 0135349.

Each test names the finding it proves. The tests marked "missing test" pass on the current code and
exist because a mutation probe showed nothing covered the behaviour; the others fail on the current code.
"""

from __future__ import annotations

import json
import sys
import time
import types

import pytest
from typer.testing import CliRunner

from bench import delivered_dump as dd
from bench import e1_agentic
from bench.cli import app
from mem import catalog, listmode, read
from mem.common import MemError
from mem.embedder import SentenceTransformerEmbedder
from mem.rerank import CrossEncoderReranker
from mem.splitq import split_endpoints
from tests.bench_t0.test_delivered_dump import item
from tests.bench_t0.test_e1 import QS, FakeLLM
from tests.bench_t0.test_e1_agentic import Agent, go, run_file
from tests.bench_t0.test_locomo import FakeLLM as LocomoFakeLLM
from tests.bench_t0.test_locomo import run as locomo_run
from tests.test_list_mode import Q as LIST_Q
from tests.test_list_paging import CAP, _paged, _walk
from tests.test_split_endpoints import Q as SPLIT_Q
from tests.test_split_endpoints import _mk
from tests.test_split_endpoints import _store as _split_store

# ---- F1 (B): the page token is not bound to the budget ------------------------------------------


def test_f1_token_refused_when_the_budget_changed(tmp_path):
    s = _paged(tmp_path)
    p1 = read.recall(s, query=LIST_Q, budget=CAP)
    with pytest.raises(MemError, match="stale"):
        read.recall(s, query=LIST_Q, budget=170, page=p1.structured["next_page"])
    s.close()


def test_f1_a_changed_budget_silently_skips_rows(tmp_path):
    s = _paged(tmp_path)
    p1 = read.recall(s, query=LIST_Q, budget=CAP)
    shown = {r["id"] for r in p1.structured["rows"]}
    page, budget = p1.structured["next_page"], 170
    while page is not None:
        try:
            out = read.recall(s, query=LIST_Q, budget=budget, page=page)
        except MemError:
            s.close()
            return  # refused: fine
        shown |= {r["id"] for r in out.structured["rows"]}
        page = out.structured.get("next_page")
    every = {r["id"] for p in _walk(s) for r in p.structured["rows"]}
    s.close()
    assert shown == every, "a page walk with a changed budget must show every row or be refused"


# ---- F2 (D): a natural ", or" question splits into an empty sub-query and the recall raises -----


@pytest.mark.parametrize("q", [
    "Did I set up the thermostat first, or the router?",
    "Did I replace the spark plugs first, or harvest the herbs?",
    "How many days between , and the day I harvested my herbs",
], ids=["first-or", "first-or-verb", "between-empty"])  # fmt: skip
def test_f2_a_split_never_yields_an_empty_sub_query(q):
    s = split_endpoints(q)
    assert s is None or all(p.strip() for p in s.parts), s


def test_f2_a_comma_or_question_is_answered_not_refused(tmp_path):
    on = _split_store(_mk(tmp_path, "on"), split=True)
    off = _split_store(_mk(tmp_path, "off"), split=False)
    try:
        q = "Did I replace the spark plugs first, or harvest the herbs?"
        assert read.recall(off, query=q).text  # the switch off answers it
        out = read.recall(on, query=q)  # the switch on must not raise MemError
        assert out.structured["rows"] or out.structured["abstained"]
    finally:
        on.close()
        off.close()


# ---- F3 (B): a page can exceed the cap (the middle-page tail is longer than the one priced) -----


def test_f3_every_page_is_under_the_cap_at_a_tight_budget(tmp_path):
    s = _paged(tmp_path)
    for p in _walk(s, budget=92):
        assert not p.text.startswith("budget")
        assert listmode.response_tokens(p.text) <= 92, p.text.splitlines()[0]
    s.close()


# ---- F4 (E): compare/adoption on a foreign baseline or an empty pairing says ADOPT --------------


def test_f4_compare_refuses_a_baseline_from_another_split():
    new, base = run_file({"a": True}, "agentic"), run_file({"a": True}, "mem")
    new["split"], base["split"] = "tune", "test"
    with pytest.raises(ValueError):
        e1_agentic.compare(new, base, b=50)


def test_f4_an_empty_pairing_is_not_adoptable():
    new = run_file({"a": True}, "agentic")
    base = run_file({"z": True}, "mem")
    cmp = e1_agentic.compare(new, base, b=50)
    assert cmp["n_paired"] == 0
    assert e1_agentic.adoption(cmp)["adopt"] is False


# ---- F5 (E): a run cut by the budget cap is still scored as adoptable ---------------------------


def test_f5_a_run_cut_by_the_budget_is_not_adoptable():
    cheap = {"input_tokens": 10, "output_tokens": 10}
    rec, *_ = go(budget_usd=0.003, agent=Agent(follow=False, usage=cheap), answer_max_tokens=100)
    assert rec["complete"] is False and len(rec["items"]) == 2
    base = run_file({i["qid"]: False for i in rec["items"]} | {"q9_abs": True}, "mem")
    cmp = e1_agentic.compare(rec, base, b=50)
    assert e1_agentic.adoption(cmp)["adopt"] is False


# ---- F6 (D): the clause pattern backtracks super-linearly (ReDoS on the serve path) -------------


@pytest.mark.parametrize("s", ["a" + " " * 2000 + "b", "word " * 4000], ids=["spaces", "words"])
def test_f6_the_splitter_is_linear_enough_on_a_long_query(s):
    t = time.perf_counter()
    split_endpoints(s)
    assert time.perf_counter() - t < 0.5, "the clause pattern backtracks super-linearly"


# ---- F7 (B): a malformed page is a ValueError, which escapes serve's `except MemError` -----------


@pytest.mark.parametrize("bad", ["²-abc", "9" * 5000 + "-abc"], ids=["superscript", "5000-digits"])
def test_f7_malformed_page_is_a_clean_error_not_a_crash(tmp_path, bad):
    s = _paged(tmp_path)
    with pytest.raises(MemError):
        read.recall(s, query=LIST_Q, budget=CAP, page=bad)
    s.close()


# ---- F8 (B): with zero gated rows any token is accepted instead of refused as stale -------------


def test_f8_token_refused_when_no_row_passes_the_gate(tmp_path):
    s = _paged(tmp_path, tau=0.999)
    with pytest.raises(MemError, match="stale|page"):
        read.recall(s, query=LIST_Q, budget=CAP, page="2-abc")
    s.close()


# ---- F9 (D): a page token on a split question is ignored, not refused ---------------------------


def test_f9_a_page_token_on_a_split_question_is_refused_like_the_plain_path(tmp_path):
    s = _split_store(tmp_path, split=True)
    try:
        with pytest.raises(MemError, match="page"):
            read.recall(s, query=SPLIT_Q, page="not-a-token")
    finally:
        s.close()


# ---- F10 (F): the real loaders' pin handling has no test (missing test; red under mutation) -----


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    calls = []

    class _Model:
        def __init__(self, path, **kw):
            calls.append((path, kw))
            self.max_seq_length = None

        def encode(self, texts, **kw):
            return [[0.0] * 4 for _ in texts]

        def predict(self, pairs, **kw):
            return [0.0] * len(pairs)

    torch = types.ModuleType("torch")
    torch.float32, torch.bfloat16, torch.float16 = "f32", "bf16", "f16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace()
    )
    torch.nn = types.SimpleNamespace(Identity=lambda: "identity")
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = st.CrossEncoder = _Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    return tmp_path, calls


def test_f10_embedder_loads_the_sections_pinned_snapshot_offline(fake_models):
    cache, calls = fake_models
    section = catalog.EMBEDDERS["qwen3-embedding-0.6b"].section()
    emb = SentenceTransformerEmbedder(section, "cpu")
    path, kw = calls[0]
    assert path == str(
        cache
        / "models--Qwen--Qwen3-Embedding-0.6B"
        / "snapshots"
        / "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )
    assert kw["local_files_only"] is True
    assert kw["model_kwargs"] == {"torch_dtype": "f32"}
    assert kw["tokenizer_kwargs"] == {"padding_side": "left"}
    assert emb._model.max_seq_length == 512 and emb.dim == 1024
    assert emb.prompts.query_prompt.startswith("Instruct: ")


def test_f10_reranker_loads_the_sections_pinned_snapshot_offline_at_its_dtype(fake_models):
    cache, calls = fake_models
    section = catalog.RERANKERS["qwen3-reranker-4b"].section()
    rr = CrossEncoderReranker(section, "cpu", 4)
    path, kw = calls[0]
    assert path == str(
        cache
        / "models--Qwen--Qwen3-Reranker-4B"
        / "snapshots"
        / "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    )
    assert kw["local_files_only"] is True and kw["model_kwargs"] == {"torch_dtype": "bf16"}
    assert kw["activation_fn"] == "identity" and kw["max_length"] == 512
    assert rr.batch_size == 4


def test_f10_loaders_refuse_a_section_whose_revision_is_not_the_pin(fake_models):
    _, calls = fake_models
    bad = {**catalog.EMBEDDERS["qwen3-embedding-0.6b"].section(), "revision": "0" * 40}
    with pytest.raises(ValueError, match="not a pinned catalog entry"):
        SentenceTransformerEmbedder(bad, "cpu")
    bad_rr = {**catalog.RERANKERS["mxbai-rerank-large-v2"].section(), "revision": "f" * 40}
    with pytest.raises(ValueError, match="not a pinned catalog entry"):
        CrossEncoderReranker(bad_rr, "cpu")
    assert calls == []


# ---- F11 (carry): three behaviours of the prompt-v2 carry with no test (missing tests) ----------


def test_f11_e1_run_cli_answerer_has_no_effort_and_the_judge_keeps_its(monkeypatch, tmp_path):
    from bench.datasets import lme

    seen = []
    monkeypatch.setattr(
        "bench.llm.AnthropicLLM",
        lambda model, effort: seen.append(effort) or FakeLLM(lambda p: "yes"),
    )
    monkeypatch.setattr(lme, "read_manifest", lambda p: object())
    monkeypatch.setattr(lme, "load_questions", lambda f: QS)
    monkeypatch.setattr(lme, "verify_manifest", lambda qs, m: None)
    monkeypatch.setattr(lme, "sha256_file", lambda p: "0" * 64)
    rows = [type("R", (), {"question_id": q["question_id"]}) for q in QS]
    monkeypatch.setattr(lme, "select", lambda m, split, kind: rows if kind == "br1" else [])
    out = tmp_path / "run.json"
    r = CliRunner().invoke(
        app, ["e1-run", "--lme-file", "x.json", "--out", str(out), "--yes", "--condition", "none"]
    )
    assert r.exit_code == 0, r.output
    assert seen == [None, "low"]
    assert json.loads(out.read_text())["answerer_effort"] is None


def test_f11_the_k_th_delivered_row_counts_and_the_k_plus_first_does_not():
    for k in dd.KS:
        at_k = dd.question_record(item("q", "t", ["a#0"], ["x#0"] * (k - 1) + ["a#0"]))
        after_k = dd.question_record(item("q", "t", ["a#0"], ["x#0"] * k + ["a#0"]))
        assert at_k["at"][str(k)]["recall_any"] == 1 and at_k["at"][str(k)]["recall_all"] == 1
        assert after_k["at"][str(k)]["recall_any"] == 0 and after_k["at"][str(k)]["recall_all"] == 0


def test_f11_locomo_answerer_and_judge_send_no_system_prompt(monkeypatch):
    seen = []
    orig = LocomoFakeLLM.create

    def create(self, *, system, messages, tools, max_tokens):
        seen.append((system, tools))
        return orig(self, system=system, messages=messages, tools=tools, max_tokens=max_tokens)

    monkeypatch.setattr(LocomoFakeLLM, "create", create)
    locomo_run()
    assert seen and all(s == "" and t == [] for s, t in seen)


# ---- F12 (B): the running count on middle pages has no test (missing test) ---------------------


def test_f12_middle_pages_carry_the_running_count(tmp_path):
    s = _paged(tmp_path)
    pages = _walk(s)
    assert len(pages) >= 3
    seen = 0
    for p in pages[:-1]:
        seen += len(p.structured["rows"])
        assert f" · {seen} of 6 shown · more:" in p.text.splitlines()[0]
    s.close()
