"""MEM2-254, ADR 0021 (amended): the recall row's `dates:` field is contract-legal, short and never
cut inside a date; `get` carries the dates; with the switch off recall and get are byte for byte as
before (Round 9 finding: P2 B-R6 failed with violations=148 with `ingest.resolve_dates` on)."""

import re
from datetime import UTC, datetime

import pytest

from bench.contract_check import structured_violations
from bench.suites import b_p2_cost, helpers
from mem import read, reldates, write
from mem.gate import Gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker

ON = {"resolve_dates": True}
NOW = datetime(2026, 10, 22, 9, tzinfo=UTC)
LAKE = [("I went hiking last week and again last Saturday by the lake.", "Sounds lovely.")]
# Ten resolvable expressions: the full field is far over the row's field cap.
MANY = [(("Yesterday I swam, two days ago I ran, three days ago I biked, last week I hiked, "
          "last month I sailed, last year I moved, four days ago I rested, five days ago I read, "
          "six days ago I cooked and a week ago I wrote, all by the lake."), "Busy.")]  # fmt: skip
ENTRY = re.compile(rf"^{reldates.DATE_VALUE}(?: \([a-z ]+\))?$", re.IGNORECASE)


def _store(tmp_path, name, **kw):
    s = Store.open(tmp_path / f"{name}.db", HashEmbedder(), scope="personal", tz="UTC",
                   reranker=OverlapReranker(), now=lambda: NOW, **kw)  # fmt: skip
    s.gate = Gate(1.0, 0.0, 0.0, 0.0, 0.0)
    return s


@pytest.fixture
def stores(tmp_path):
    opened = []

    def _make(name, rounds, **kw):
        s = _store(tmp_path, name, **kw)
        write.ingest_session(s, source_ref="s1", date="2023-05-30", rounds=rounds)
        opened.append(s)
        return s

    yield _make
    for s in opened:
        s.close()


def _field(line):
    m = re.search(r" · dates: ([^·]+) · ", line)
    return m and m.group(1)


# ---- rendering (mem.reldates.render) -------------------------------------------------------------


def test_render_puts_the_absolute_date_first_and_keeps_the_words_when_they_fit():
    ann = "yesterday [= 2023-03-09]; last month [= 2023-02]"
    assert reldates.render(ann) == "2023-03-09 (yesterday), 2023-02 (last month)"
    assert reldates.render("last week [= 2023-05-22..2023-05-28]") == (
        "2023-05-22..2023-05-28 (last week)")  # fmt: skip


def test_render_shows_only_resolved_entries():
    assert reldates.render("last Saturday; yesterday [= 2023-03-09]") == "2023-03-09 (yesterday)"
    assert reldates.render("last Saturday") is None
    assert reldates.render("last Saturday [= soon]") is None
    assert reldates.render(None) is None and reldates.render("") is None


def test_render_collapses_whitespace_so_the_field_stays_on_one_line():
    assert reldates.render("last\n  week [= 2023-05-22..2023-05-28]") == (
        "2023-05-22..2023-05-28 (last week)")  # fmt: skip


@pytest.mark.parametrize("limit", range(10, 120))
def test_render_under_a_limit_drops_words_then_whole_trailing_dates_never_a_partial_one(limit):
    ann = reldates.annotation(MANY[0][0], datetime(2023, 5, 30).date())
    full = reldates.render(ann)
    out = reldates.render(ann, limit)
    values = [v for _, v in reldates.entries(ann)]
    assert out is not None and out.split(", ")[0].split(" ")[0] == values[0]  # first date always
    assert all(ENTRY.match(e) for e in out.split(", ")), out
    if len(full) <= limit:
        assert out == full
    else:  # words gone first, then whole trailing dates: a prefix of the distinct values
        shown = out.split(", ")
        assert "(" not in out and shown == list(dict.fromkeys(values))[: len(shown)]
        assert len(out) <= max(limit, len(values[0]))


# ---- recall: B-R6 with the switch on ----------------------------------------------------------


def test_br6_passes_with_the_switch_on_over_relative_dates(stores):
    s = stores("on", LAKE + MANY + [("Nothing relative in this one.", "ok")], ingest=ON)
    for q in ("hiking lake", "swam ran biked lake", "relative"):
        r = s.recall(query=q)
        assert b_p2_cost.response_violations(r.text, r.structured["abstained"]) == [], r.text
        assert structured_violations("recall", r.structured) == [], r.structured
    r = s.recall(query="hiking lake")
    line = next(ln for ln in r.text.splitlines() if ln.startswith("c1 "))
    # with the words the field is over the cap: the dates alone, both whole
    assert _field(line) == "2023-05-22..2023-05-28, 2023-05-27"
    assert r.structured["rows"][0]["dates"] == _field(line)
    r = stores("one", [("We went hiking yesterday by the lake.", "ok")], ingest=ON).recall(
        query="hiking lake")  # fmt: skip
    assert _field(r.text.splitlines()[1]) == "2023-05-29 (yesterday)"


def test_the_p1_row_grammar_takes_the_dates_field_too():
    line = "c1 · note · 2023-05-30 · t · about: A · dates: 2023-05-27 (last Saturday) · .5"
    assert helpers.ROW.match(line) and b_p2_cost.ROW.match(line)
    for bad in ("dates: last Saturday [= ", "dates: 2023-05-2", "dates: last Saturday",
                "dates: 2023-05-27 (last Saturday), "):  # fmt: skip
        assert not b_p2_cost.ROW.match(f"c1 · note · 2023-05-30 · t · {bad} · .5"), bad


def test_a_long_dates_field_keeps_the_absolute_date_and_never_a_partial_one(stores):
    s = stores("many", MANY, ingest=ON)
    r = s.recall(query="swam ran biked lake")
    (line,) = r.text.splitlines()[1:]
    field = _field(line)
    assert field.startswith("2023-05-29") and len(field) <= reldates.ROW_CHARS
    assert all(ENTRY.match(e) for e in field.split(", ")), field
    assert b_p2_cost.response_violations(r.text, False) == []


def test_a_chunk_with_only_unresolved_expressions_gets_no_dates_field(stores):
    s = stores("unresolved", [("Maybe Saturday, the other day or soon it rains.", "ok")], ingest=ON)
    r = s.recall(query="rains")
    assert " · dates: " not in r.text and "dates" not in r.structured["rows"][0]


# ---- get --------------------------------------------------------------------------------------


def test_get_carries_the_resolved_dates_in_full_and_the_text_stays_verbatim(stores):
    s = stores("on", MANY, ingest=ON)
    item = read.get(s, ["c1"]).structured["items"][0]
    assert item["header"].endswith(" · dates: " + reldates.render(
        s.conn.execute("SELECT dates FROM chunks WHERE id = 1").fetchone()[0]))  # fmt: skip
    assert item["text"] == f"{MANY[0][0]}\n{MANY[0][1]}"
    assert structured_violations("get", read.get(s, ["c1"]).structured) == []


def test_get_of_a_memory_carries_its_dates(tmp_path):
    s = _store(tmp_path, "mem", ingest=ON)
    mid = s.remember(text="We moved the server yesterday.", type="event", about=["server"],
                     as_of="2023-03-10").structured["id"]  # fmt: skip
    head = read.get(s, [mid]).structured["items"][0]["header"]
    assert head.endswith(" · dates: 2023-03-09 (yesterday)")
    s.close()


# ---- switch off: byte for byte today's ------------------------------------------------------------


def test_switch_off_recall_and_get_are_byte_for_byte_unchanged(stores):
    s = stores("off", LAKE)
    r = s.recall(query="hiking lake")
    assert r.text == ("1 results (of 1 checked) · personal\n"
                      "c1 · chat_session · 2023-05-30 · chat session 2023-05-30 · .95")  # fmt: skip
    assert "dates" not in r.structured["rows"][0]
    assert read.get(s, ["c1"]).text == (
        "c1 · n1 · section · 14\nI went hiking last week and again last Saturday by the lake.\n"
        "Sounds lovely.")  # fmt: skip
