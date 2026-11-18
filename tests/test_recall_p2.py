"""P2 read path (architecture-v2 §3.2 steps 4-9): rerank, Platt, gate, abstention, flags, budget."""

import json
import math
import random

import pytest

from mem.common import estimate_tokens
from mem.gate import Gate, write_gate
from mem.read import candidate_pool
from mem.store import Store
from mem.testing import HashEmbedder

ANA = "new:person:Ana Silva"


class KeyedReranker:
    """Logit by keyword: the first key found in the doc decides; unknown docs get -8."""

    def __init__(self, logits):
        self.logits = logits
        self.seen = []

    def score(self, query, docs):
        self.seen.append(list(docs))
        return [next((z for k, z in self.logits.items() if k in d), -8.0) for d in docs]


def logit(p):
    return math.log(p / (1 - p))


GATE = Gate(platt_a=1.0, platt_b=0.0, tau_keep=0.5, tau_high=0.5, gap=0.0, fit_id="test",
            fingerprint_hash="0" * 64)  # fmt: skip


@pytest.fixture
def gated(tmp_path, clock):
    opened = []

    def _make(logits, gate=GATE, name="personal"):
        s = Store.open(tmp_path / f"{name}.db", HashEmbedder(), scope=name, now=clock)
        write_gate(s.conn, gate)
        s.close()
        rr = KeyedReranker(logits)
        s = Store.open(tmp_path / f"{name}.db", HashEmbedder(), scope=name, now=clock, reranker=rr)
        opened.append(s)
        return s

    yield _make
    for s in opened:
        s.close()


def _save(store, text, about=(ANA,), **kw):
    return store.remember(text=text, type="fact", about=list(about), **kw).structured["id"]


def _ids(res):
    return [r["id"] for r in res.structured["rows"]]


def test_gate_drops_below_tau(gated):
    s = gated({"alpha": logit(0.88), "beta": logit(0.27), "gamma": logit(0.62)})
    a = _save(s, "Garden note alpha: tomatoes need sun.")
    _save(s, "Garden note beta: basil wants shade.")
    c = _save(s, "Garden note gamma: water the garden at dawn.")
    res = s.recall(query="garden note")
    assert _ids(res) == [a, c]
    assert [r["p"] for r in res.structured["rows"]] == pytest.approx([0.88, 0.62])
    assert res.structured["abstained"] is False
    assert res.structured["checked"] == 3
    lines = res.text.splitlines()
    assert lines[0] == "2 results (of 3 checked) · personal"
    assert lines[1].startswith(f"{a} · fact · 2026-10-22 · Garden note alpha: tomatoes need sun.")
    assert lines[1].endswith(" · about: Ana Silva · .88")


@pytest.mark.parametrize(
    ("p_top", "p_second", "abstain"),
    [
        (0.62, 0.55, True),
        (0.62, 0.35, False),
        (0.25, 0.10, True),
        (0.85, 0.80, False),
        (0.62, 0.2, False),
    ],
)
def test_abstain_low_top_small_gap(gated, p_top, p_second, abstain):
    gate = Gate(1.0, 0.0, tau_keep=0.3, tau_high=0.8, gap=0.2, fingerprint_hash="0" * 64)
    s = gated({"alpha": logit(p_top), "beta": logit(p_second)}, gate=gate)
    a = _save(s, "Chess opening alpha: the Sicilian.")
    b = _save(s, "Chess opening beta: the French.")
    res = s.recall(query="chess opening")
    assert res.structured["abstained"] is abstain
    if abstain:
        assert res.structured["rows"] == []
        best = f"{p_top:.2f}".lstrip("0")
        assert res.text == f"no relevant memories (best {best}) · personal"
        assert estimate_tokens(res.text) <= 30
    else:
        assert _ids(res) == ([a, b] if p_second >= 0.3 else [a])


def test_abstain_on_empty_eligible_set(gated):
    s = gated({})
    res = s.recall(query="anything")
    assert res.structured["abstained"] is True
    assert res.text == "no relevant memories · personal"


def test_no_rows_added_after_gate(gated):
    rng = random.Random(3)
    logits = {f"item{i:02d}": rng.uniform(-4, 4) for i in range(40)}
    logits["item00"] = 3.0
    s = gated(logits)
    ids = {k: _save(s, f"Recipe {k} uses flour and salt.") for k in logits}
    # A superseded chain: the open successor is future-dated, so it is not eligible today; the
    # current member passes the gate and only gains a flag, the successor is never injected.
    old = _save(s, "Recipe item00 bakes at 200 degrees.")
    new = _save(s, "Recipe item00 bakes at 180 degrees.", as_of="2026-11-01T00:00:00Z")
    s.conn.execute("UPDATE nodes SET valid_to = '2026-11-01T00:00:00Z' WHERE id = ?",
                   (int(old[1:]),))  # fmt: skip
    s.conn.execute("INSERT INTO edges(src, dst, rel, origin, recorded_at) VALUES (?, ?,"
                   " 'supersedes', 'agent', '2026-10-22T09:00:00Z')",
                   (int(new[1:]), int(old[1:])))  # fmt: skip
    for limit in (3, 8, 40):
        res = s.recall(query="recipe flour salt bakes item00", limit=limit, budget=5000)
        rows = res.structured["rows"]
        assert all(r["p"] >= GATE.tau_keep for r in rows)
        assert new not in _ids(res)
        passed = {ids[k] for k, z in logits.items() if 1 / (1 + math.exp(-z)) >= 0.5}
        assert set(_ids(res)) <= passed | {old}
        assert len(rows) <= limit
    (flagged,) = [r for r in rows if r["id"] == old]
    assert flagged["flags"] == [f"until 2026-11-01 → {new}"]


def test_as_of_returns_old_value(gated):
    s = gated({"Mondays": logit(0.9), "Thursdays": logit(0.9)})
    old = _save(s, "Weekly 1:1s with Ana on Mondays.", as_of="2026-06-03")
    new = _save(s, "Weekly 1:1s with Ana move to Thursdays.", as_of="2026-11-01")
    s.conn.execute("UPDATE nodes SET valid_to = ? WHERE id = ?",
                   ("2026-11-01T00:00:00Z", int(old[1:])))  # fmt: skip
    s.conn.execute("INSERT INTO edges(src, dst, rel, origin, recorded_at) VALUES (?, ?,"
                   " 'supersedes', 'agent', '2026-10-22T09:00:00Z')",
                   (int(new[1:]), int(old[1:])))  # fmt: skip
    q = "weekly 1:1 with Ana"
    assert _ids(s.recall(query=q, as_of="2026-07-01")) == [old]
    assert _ids(s.recall(query=q, as_of="2026-12-01")) == [new]
    today = s.recall(query=q)  # 2026-10-22: the old value, flagged with its end and successor
    assert _ids(today) == [old]
    assert today.structured["rows"][0]["flags"] == [f"until 2026-11-01 → {new}"]
    assert f" · until 2026-11-01 → {new} · .90" in today.text
    hist = s.recall(query=q, include_history=True)
    assert set(_ids(hist)) == {old, new}
    flags = {r["id"]: r["flags"] for r in hist.structured["rows"]}
    assert flags == {old: ["[2026-06-03 – 2026-11-01]"], new: ["[2026-11-01 – ]"]}


def test_same_instant_predecessor_shows_in_history_only(gated):
    s = gated({"Lisbon": logit(0.9), "Porto": logit(0.9)})
    first = _save(s, "Ana lives in Lisbon.", as_of="2026-05-01")
    second = _save(s, "Ana lives in Porto.", as_of="2026-05-01")
    s.conn.execute("UPDATE nodes SET valid_to = valid_from WHERE id = ?", (int(first[1:]),))
    assert _ids(s.recall(query="where does Ana live")) == [second]
    hist = s.recall(query="where does Ana live", include_history=True)
    flags = {r["id"]: r["flags"] for r in hist.structured["rows"]}
    assert flags[first] == ["superseded same instant"]


def test_about_lookup_miss_writes_nothing(gated):
    s = gated({"vim": logit(0.9)})
    _save(s, "Ana Silva uses vim keybindings.")
    s.conn.executescript(
        """INSERT INTO nodes(kind,type,name,valid_from,recorded_at,origin) VALUES
             ('entity','person','Ana Lopez','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z','agent');
           INSERT INTO aliases(alias,alias_norm,node_id,origin) VALUES ('Ana','ana',2,'agent'),
             ('Ana','ana',3,'agent');"""
    )
    tables = ("nodes", "edges", "aliases", "chunks", "props")

    def counts():
        return {t: s.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}

    before = counts()
    unknown = s.recall(query="vim", about=["Nobody Known"])
    ambiguous = s.recall(query="vim", about=["Ana"])
    assert counts() == before
    assert unknown.structured["filters_dropped"] == ["about 'Nobody Known': unknown, not applied"]
    assert ambiguous.structured["filters_dropped"] == ["about 'Ana': ambiguous e2/e3, not applied"]
    assert len(_ids(unknown)) == 1  # the filter is dropped, the query still runs
    assert "about 'Nobody Known': unknown, not applied" in unknown.text


@pytest.mark.parametrize("budget", range(40, 400, 7))
def test_budget_counts_whole_response(gated, budget):
    s = gated({"Kyoto": logit(0.9)})
    for i in range(12):
        _save(s, f"Kyoto trip day {i}: temples, gardens and a long walk along the river.")
    res = s.recall(query="Kyoto trip", about=["Somebody Unknown With A Long Name"],
                   budget=budget, limit=20)  # fmt: skip
    assert estimate_tokens(res.text) <= budget
    rows = res.structured["rows"]
    if rows:  # the dropped-filter line is part of the response, so it is part of the count
        assert "about 'Somebody Unknown With A Long Name': unknown, not applied" in res.text
    if budget >= 150:
        assert len(rows) >= 2


def test_recall_with_reranker_but_no_gate_refuses(tmp_path, clock):
    s = Store.open(tmp_path / "p.db", HashEmbedder(), now=clock, reranker=KeyedReranker({}))
    _save(s, "Anything at all.")
    with pytest.raises(Exception, match="mem calibrate"):
        s.recall(query="anything")
    s.close()


def test_pool_keeps_up_to_four_chunks_per_node_and_thirty_pairs(gated):
    s = gated({"facts": logit(0.9)})
    for i in range(30):
        s.remember(text=f"Note {i} about facts and figures.", type="fact", about=[ANA],
                   facts=[f"fact {i} {w}" for w in ("one", "two", "three", "four", "five")])  # fmt: skip
    res = s.recall(query="facts figures fact one two three four five note")
    (docs,) = s.reranker.seen
    assert len(docs) == 30
    assert res.structured["checked"] == 30
    pool, _ = candidate_pool(s, "facts figures fact one two three four five note", t=None)
    per_node = [r[0] for r in s.conn.execute(
        f"SELECT COUNT(*) FROM chunks WHERE id IN ({','.join('?' * len(pool))}) GROUP BY node_id",
        [c for c, _ in pool])]  # fmt: skip
    assert max(per_node) == 4
    assert len(_ids(res)) == len(set(_ids(res)))  # a node's chunks give one row


def test_recall_event_logs_p_and_abstention(gated):
    s = gated({"alpha": logit(0.88)})
    a = _save(s, "Garden note alpha.")
    s.recall(query="garden note")
    (result,) = s.conn.execute(
        "SELECT result FROM events WHERE tool='recall' ORDER BY id DESC LIMIT 1").fetchone()  # fmt: skip
    got = json.loads(result)
    assert got["ids"] == [a] and got["abstained"] is False
    assert got["p"] == pytest.approx([0.88])
