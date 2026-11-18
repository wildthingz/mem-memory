"""Coverage mode for counting questions (MEM2-212): list mode paged behind a continuation token.
T0, hash models. Pages cover every gated row once, each under the cap; the token is bound; the
switch off is today's output."""

from __future__ import annotations

import re

import pytest

from mem import config, listmode, read
from mem.common import MemError
from mem.contracts import mcp_tools
from tests.test_list_mode import DAYS, Q, _store, _sub

CAP = 150  # an explicit budget wins over the mode's cap: small, so six rows need several pages


def _paged(tmp_path, **kw):
    s = _store(tmp_path, list_mode=True, **kw)
    s.read_cfg = {**s.read_cfg, "list_paging": True}
    return s


def _token(text):
    m = re.search(r"more: recall\(…, page=(\S+)\)", text)
    return m.group(1) if m else None


def _walk(s, q=Q, budget=CAP):
    pages, page = [], None
    while True:
        out = read.recall(s, query=q, budget=budget, page=page)
        pages.append(out)
        page = out.structured.get("next_page")
        if page is None:
            return pages
        assert _token(out.text) == page


def _ids(pages):
    return [r["id"] for p in pages for r in p.structured["rows"]]


def test_pages_cover_every_gated_row_exactly_once_under_the_cap(tmp_path):
    s = _paged(tmp_path)
    pages = _walk(s)
    assert len(pages) > 1
    ids = _ids(pages)
    assert len(ids) == len(set(ids)) == 6  # no duplicates, nothing lost
    whole = read.recall(s, query=Q, budget=100_000)  # one page: everything the gate kept
    assert sorted(ids) == sorted(r["id"] for r in whole.structured["rows"])
    for p in pages:
        assert listmode.response_tokens(p.text) <= CAP
    s.close()


def test_header_carries_the_token_and_the_last_page_says_so(tmp_path):
    s = _paged(tmp_path)
    pages = _walk(s)
    first, last = pages[0], pages[-1]
    head = first.text.splitlines()[0]
    n = len(first.structured["rows"])
    assert f"{n} of 6 shown · more: recall(…, page=2-" in head
    assert first.structured["page"] == 1 and first.structured["pages"] == len(pages)
    assert "more:" not in last.text and "last page" in last.text.splitlines()[0]
    assert f"6 of 6 shown · page {len(pages)} of {len(pages)} · last page" in last.text
    assert all("more matches above tau_list not shown" not in p.text for p in pages)
    s.close()


def test_same_order_as_the_gate_across_pages(tmp_path):
    s = _paged(tmp_path)
    ids = _ids(_walk(s))
    ps = [r["p"] for p in _walk(s) for r in p.structured["rows"]]
    assert ps == sorted(ps, reverse=True)  # the order of the lowest-p-last list mode
    assert ids == _ids(_walk(s))  # deterministic: asking again gives the same pages
    s.close()


def test_each_page_is_grouped_by_session_with_the_date(tmp_path):
    s = _paged(tmp_path)
    for p in _walk(s):
        lines = p.text.splitlines()
        assert lines[1] == listmode.INSTRUCTION
        heads = [ln.split(" · ")[0] for ln in lines[2:] if not ln.startswith("  ")]
        assert heads == sorted(heads) and set(heads) <= set(DAYS)
        assert sum(ln.startswith("  ") for ln in lines) == len(p.structured["rows"])
    s.close()


def test_a_single_page_answer_is_plain_list_mode(tmp_path):
    on = _paged(_sub(tmp_path, "a"))
    off = _store(_sub(tmp_path, "b"), list_mode=True)
    a, b = read.recall(on, query=Q), read.recall(off, query=Q)
    assert a.text == b.text and "shown" not in a.text
    on.close()
    off.close()


def test_the_default_cap_holds_p95_on_a_page(tmp_path):
    s = _paged(tmp_path, nrounds=3, retrieval={**config.default_config().retrieval,
                                               "per_node_cap": 9})  # fmt: skip
    pages = _walk(s, budget=None)
    assert all(listmode.response_tokens(p.text) <= listmode.CAP_TOKENS for p in pages)
    assert len(set(_ids(pages))) == len(_ids(pages))
    s.close()


def test_token_refused_when_the_query_changed(tmp_path):
    s = _paged(tmp_path)
    tok = _walk(s)[0].structured["next_page"]
    with pytest.raises(MemError, match="stale"):
        read.recall(s, query="How many hiking trips did I take in total", budget=CAP, page=tok)
    s.close()


def test_token_refused_when_a_filter_changed(tmp_path):
    s = _paged(tmp_path)
    tok = _walk(s)[0].structured["next_page"]
    with pytest.raises(MemError, match="stale"):
        read.recall(s, query=Q, budget=CAP, page=tok, since="2024-03-05")
    s.close()


def test_token_refused_when_the_config_changed(tmp_path):
    s = _paged(tmp_path)
    tok = _walk(s)[0].structured["next_page"]
    s.retrieval = {**s.retrieval, "leg_k": s.retrieval["leg_k"] + 1}
    with pytest.raises(MemError, match="stale"):
        read.recall(s, query=Q, budget=CAP, page=tok)
    s.close()


def test_token_refused_when_the_store_changed_between_pages(tmp_path):
    from mem import write

    s = _paged(tmp_path)
    tok = _walk(s)[0].structured["next_page"]
    write.ingest_session(s, source_ref="new", date="2024-05-20",
                         rounds=[("I did take many hiking trips with friends",
                                  "Nice, how many hiking trips")])  # fmt: skip
    with pytest.raises(MemError, match="stale"):
        read.recall(s, query=Q, budget=CAP, page=tok)
    s.close()


@pytest.mark.parametrize("bad", ["2", "x-abc", "1-abc", "", "2-"])
def test_malformed_or_forged_tokens_are_refused(tmp_path, bad):
    s = _paged(tmp_path)
    with pytest.raises(MemError):
        read.recall(s, query=Q, budget=CAP, page=bad)
    s.close()


def test_page_past_the_end_is_refused(tmp_path):
    s = _paged(tmp_path)
    tok = _walk(s)[0].structured["next_page"]
    with pytest.raises(MemError, match="past the last page"):
        read.recall(s, query=Q, budget=CAP, page="99-" + tok.split("-")[1])
    s.close()


def test_page_needs_paging_and_an_aggregation_query(tmp_path):
    off = _store(_sub(tmp_path, "a"), list_mode=True)
    with pytest.raises(MemError, match="page"):
        read.recall(off, query=Q, page="2-abc")
    off.close()
    on = _paged(_sub(tmp_path, "b"))
    with pytest.raises(MemError, match="page"):
        read.recall(on, query="Which hiking trip was in March", page="2-abc")
    on.close()


def test_gate_invariant_nothing_added_after_the_gate(tmp_path):
    """The paged rows are the list-mode rows above tau_list, no more: a higher cut gives fewer."""
    lo = _paged(_sub(tmp_path, "a"), tau=0.5)
    hi = _paged(_sub(tmp_path, "b"), tau=0.999)
    assert set(_ids(_walk(lo))) == {
        r["id"] for r in read.recall(lo, query=Q, budget=100_000).structured["rows"]
    }
    out = read.recall(hi, query=Q, budget=CAP)
    assert out.structured["abstained"] is True and out.text.startswith("no relevant memories")
    lo.close()
    hi.close()


def test_switch_off_is_todays_output_and_normal_recall_is_unchanged(tmp_path):
    off = _store(_sub(tmp_path, "a"), list_mode=True)  # paging absent
    off2 = _store(_sub(tmp_path, "b"), list_mode=True)
    off2.read_cfg = {**off2.read_cfg, "list_paging": False}
    for q in (Q, "Where did I park the car", "Which hiking trip was in March"):
        a, b = read.recall(off, query=q, budget=CAP), read.recall(off2, query=q, budget=CAP)
        assert a.text == b.text and a.structured == b.structured
    assert "more matches above tau_list not shown" in read.recall(off, query=Q, budget=CAP).text
    # a non-aggregation query is the normal path with paging on, byte for byte
    on = _paged(_sub(tmp_path, "c"))
    q = "Which hiking trip was in March"
    assert read.recall(on, query=q).text == read.recall(off, query=q).text
    for s in (off, off2, on):
        s.close()


def test_config_switch(tmp_path):
    cfg = config.default_config()
    assert "list_paging" not in cfg.read and cfg.list_paging is False
    assert cfg.replace("read", list_paging=False).fp_all() == cfg.fp_all()
    assert cfg.replace("read", list_mode=True, list_paging=True).fp_all() != (
        cfg.replace("read", list_mode=True).fp_all())  # fmt: skip
    p = tmp_path / "c.yaml"
    p.write_text("read: {list_paging: true}\n")
    with pytest.raises(config.ConfigError, match="list_mode"):
        config.load(p)
    p.write_text("read: {list_mode: true, list_paging: 'yes'}\n")
    with pytest.raises(config.ConfigError, match="list_paging"):
        config.load(p)
    p.write_text("read: {list_mode: true, list_paging: true}\n")
    assert config.load(p).list_paging is True


def test_mcp_contract_gains_only_the_optional_page_param():
    t = next(t for t in mcp_tools()["tools"] if t["name"] == "recall")
    props = t["inputSchema"]["properties"]
    assert props["page"] == {"type": "string", "maxLength": 32}  # bounded (F23, MEM2-249)
    assert t["inputSchema"]["required"] == ["query"]
    assert sorted(props) == ["about", "as_of", "budget", "include_history", "kinds", "limit",
                             "page", "query", "since", "type", "until"]  # fmt: skip


def test_every_page_with_rows_is_under_the_cap_across_budgets(tmp_path):
    s = _paged(tmp_path)  # F3 (MEM2-249): middle pages carry the longer 'more:' tail; it is priced
    for budget in range(97, 260, 7):  # from 97 one row fits a page
        pages = _walk(s, budget=budget)
        assert sorted(_ids(pages)) == sorted(_ids(_walk(s, budget=100_000)))
        for p in pages:
            if p.structured["rows"]:
                assert listmode.response_tokens(p.text) <= budget, (budget, p.text)
    s.close()


def test_an_oversize_row_page_keeps_the_header_and_the_walk_goes_on(tmp_path):
    s = _paged(tmp_path)  # F23 (MEM2-249): one row alone over the cap
    pages = _walk(s, budget=92)
    assert len(pages) == 6 and not _ids(pages)
    first = pages[0].text.splitlines()
    assert "0 of 6 shown · more: recall(…, page=2-" in first[0]
    assert first[-1] == "c1 is over the budget 92: get(c1) to read it"
    assert " · 0 of 6 shown · page 6 of 6 · last page" in pages[-1].text.splitlines()[0]
    s.close()


def test_a_page_token_on_a_p1_store_is_refused(tmp_path):
    s = _paged(tmp_path)
    s.reranker = None  # the P1 build answers without list mode: a token has nothing to continue
    with pytest.raises(MemError, match="page"):
        read.recall(s, query=Q, page="2-abc")
    s.close()
