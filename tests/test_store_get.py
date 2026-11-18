import math

import pytest

from mem.common import MemError
from mem.read import estimate_tokens

TEXT = "".join(f"Paragraph {i} " + "word " * 40 + "\n\n" for i in range(30))  # ~7k chars


def _note(store, text=TEXT):
    cur = store.conn.execute(
        "INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin,source_ref,content_hash)"
        " VALUES ('note','note','Big note','2026-03-01T00:00:00Z','2026-03-01T00:00:00Z',"
        "'import',?,'h')",
        (f"big{store.conn.execute('SELECT count(*) FROM nodes').fetchone()[0]}.md",),
    )
    nid = cur.lastrowid
    cur = store.conn.execute(
        "INSERT INTO chunks(node_id,chunk_key,ord,key_kind,heading_path,ctx,text,tokens,content_hash)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (nid, "Intro#1/p1", 0, "section", "Intro", "Big note > Intro", text, 2000, "h"),
    )
    return nid, cur.lastrowid


def _get(store, ident, max_tokens=2000):
    item = store.get(ids=[ident], max_tokens=max_tokens).structured["items"][0]
    return item


def test_get_continuation_returns_tail(store):
    _, cid = _note(store)
    first = _get(store, f"c{cid}", max_tokens=300)
    assert first["found"] and 0 < len(first["text"]) < len(TEXT)
    assert first["continuation"] == f"c{cid}@{len(first['text'])}"
    assert first["text"].endswith("\n\n")  # cut at a paragraph boundary

    parts, item = [first["text"]], first
    while item["continuation"]:
        item = _get(store, item["continuation"], max_tokens=300)
        parts.append(item["text"])
    assert "".join(parts) == TEXT  # byte-identical across continuations
    tail = _get(store, first["continuation"], max_tokens=5000)
    assert tail["text"] == TEXT[len(first["text"]) :] and tail["continuation"] is None


def test_get_cuts_at_character_boundary_without_paragraphs(store):
    _, cid = _note(store, "x" * 6000)
    first = _get(store, f"c{cid}", max_tokens=200)
    assert 0 < len(first["text"]) < 6000 and first["continuation"] == f"c{cid}@{len(first['text'])}"


def test_get_max_tokens_is_total_for_the_response(store):
    _, c1 = _note(store)
    _, c2 = _note(store)
    res = store.get(ids=[f"c{c1}", f"c{c2}"], max_tokens=400)
    assert estimate_tokens(res.text) <= 400
    assert all(i["continuation"] for i in res.structured["items"])


def test_get_memory_header_and_verbatim_text(store):
    m = store.remember(text="Ana Silva  prefers tea.", type="preference", about=["new:person:Ana"],
                       as_of="2026-02-01", stated_by_user=True)  # fmt: skip
    item = _get(store, m.structured["id"])
    assert item["text"] == "Ana Silva  prefers tea."
    assert item["header"] == (
        f"{m.structured['id']} · preference · 2026-02-01T00:00:00Z– · active · user"
    )


def test_get_entity_lists_sections_and_edges_with_pagination(store):
    nid, cid = _note(store)
    for i in range(14):
        store.conn.execute(
            "INSERT INTO nodes(id,kind,type,name,valid_from,recorded_at,origin)"
            " VALUES (?, 'entity','topic',?,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','agent')",
            (1000 + i, f"Topic {i}"),
        )
        store.conn.execute(
            "INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES (?,?,'links_to','import','x')",
            (nid, 1000 + i),
        )
    item = _get(store, f"n{nid}")
    assert f"c{cid} · Intro · 2000" in item["text"]
    assert item["text"].count("links_to") == 10
    assert f"… 4 more: get n{nid}@edges:10" in item["text"]
    page2 = _get(store, f"n{nid}@edges:10")
    assert page2["text"].count("links_to") == 4 and "more:" not in page2["text"]


def test_get_missing_and_stale_ids(store):
    assert _get(store, "c9999")["found"] is False
    assert "c9999 no longer exists" in _get(store, "c9999")["text"]
    assert _get(store, "m77")["found"] is False
    assert _get(store, "garbage")["found"] is False


def test_estimate_tokens_is_ceil_bytes_over_three():
    assert estimate_tokens("abc") == 1 and estimate_tokens("abcd") == 2
    assert estimate_tokens("é") == math.ceil(2 / 3)


def _long_memory(store):
    """`remember` caps text at 1,200 chars; a long body arrives by other routes (import)."""
    return store.conn.execute(
        "INSERT INTO nodes(kind,type,name,body,date,valid_from,recorded_at,origin,content_hash)"
        " VALUES ('memory','procedure','Long','"
        + "word " * 2000
        + "','2026-10-22','2026-10-22T09:00:00Z','2026-10-22T09:00:00Z','agent','long')"
    ).lastrowid


def test_get_ten_ids_share_max_tokens_so_the_total_holds(store):
    ids = [
        store.remember(
            text=f"Fact number {i} " + "about the garden and the dog " * 3,
            type="fact",
            about=["new:topic:T"],
        ).structured["id"]
        for i in range(9)
    ]
    long_id = f"m{_long_memory(store)}"
    res = store.get(ids=[*ids, long_id], max_tokens=300)
    assert estimate_tokens(res.text) <= 300
    items = res.structured["items"]
    assert all(i["found"] for i in items)
    assert all(_progress(i) for i in items)


def _progress(item):
    return item["continuation"] is None or item["text"] or item["continuation"] == item["id"]


def _follow(store, ids, max_tokens):
    """Call get, then get again on every continuation token, until none is left."""
    pages, pending = [], list(ids)
    for _ in range(200):
        if not pending:
            return pages
        res = store.get(ids=pending[:10], max_tokens=max_tokens)
        assert estimate_tokens(res.text) <= max_tokens, res.text
        pages.append(res)
        conts = [i["continuation"] for i in res.structured["items"] if i.get("continuation")]
        assert conts != pending[:10], f"no progress: {conts}"
        pending = conts + pending[10:]
    raise AssertionError("continuation chain did not terminate")


def test_get_chain_over_body_and_edges_loses_nothing_and_stays_in_budget(store):
    nid, _ = _note(store)
    body = "".join(f"Body paragraph {i} " + "text " * 15 + "\n\n" for i in range(12)) + "END"
    store.conn.execute("UPDATE nodes SET body = ? WHERE id = ?", (body, nid))
    for i in range(14):
        store.conn.execute(
            "INSERT INTO nodes(id,kind,type,name,valid_from,recorded_at,origin)"
            " VALUES (?, 'entity','topic',?,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','agent')",
            (2000 + i, f"Topic {i}"),
        )
        store.conn.execute(
            "INSERT INTO edges(src,dst,rel,origin,recorded_at) VALUES (?,?,'links_to','import','x')",
            (nid, 2000 + i),
        )
    pages = _follow(store, [f"n{nid}"], max_tokens=120)
    text = "\n".join(p.text for p in pages)
    assert len(pages) > 2
    for paragraph in body.split("\n\n"):
        assert paragraph in text
    for i in range(14):
        assert f"Topic {i}\n" in text + "\n"
    assert text.count("links_to") == 14


def test_get_ten_ids_over_budget_defer_the_rest_and_the_chain_returns_all(store):
    texts = [f"Preference {i}: tea over coffee, " + "always " * i for i in range(10)]
    ids = [
        store.remember(text=t, type="preference", about=["new:person:Ana"]).structured["id"]
        for t in texts
    ]
    first = store.get(ids=ids, max_tokens=300)
    deferred = [i for i in first.structured["items"] if i["continuation"] == i["id"]]
    assert deferred and all(i["found"] and i["text"] == "" for i in deferred)
    assert "… more: get " + ", ".join(i["id"] for i in deferred) in first.text
    got = dict.fromkeys(ids, "")
    for page in _follow(store, ids, max_tokens=300):
        for item in page.structured["items"]:
            got[item["id"]] += item["text"]
    assert list(got.values()) == texts  # byte-identical, nothing lost across pages


def test_get_max_tokens_too_small_for_any_progress_is_an_error(store):
    mid = store.remember(text="Ana likes tea.", type="fact", about=["new:person:Ana"]).structured[
        "id"
    ]
    with pytest.raises(MemError, match="max_tokens"):
        store.get(ids=[mid], max_tokens=10)


def test_progress_compares_positions_not_text():
    from mem.read import _progress

    edge_page = {"continuation": "e5@edges:3", "text": "… 4 more: get e5@edges:3"}
    assert not _progress("e5@edges:3", edge_page)
    assert not _progress("m8", {"continuation": "m8@0", "text": ""})
    assert _progress("m8@40", {"continuation": "m8@69", "text": "x"})
    assert _progress("e5@900", {"continuation": "e5@edges:0", "text": "tail"})
    assert _progress("e5@edges:3", {"continuation": None, "text": ""})
