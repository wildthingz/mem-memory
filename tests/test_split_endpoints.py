"""ADR 0026: `read.split_endpoints`. A question that names two events gets a pool, a rerank and a gate
per event, merged (deduplicated, per-node cap, round-robin to the limit), delivered in date order. T0,
on the hash models. The patterns are the dry run's (MEM2-209), frozen."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mem import config, read, write
from mem.gate import Gate, write_gate
from mem.splitq import Kept, merge_endpoints, split_endpoints
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker
from tests.test_config import GOLDEN_FP_ALL_OFF, GOLDEN_FP_ALL_SPLIT_ENDPOINTS, all_off

# ---- the patterns (frozen from MEM2-209) -----------------------------------------------------

POSITIVE = [
    ("How many days between the day I replaced my spark plugs and the day I harvested my herbs?",
     "between_and", ["the day I replaced my spark plugs", "the day I harvested my herbs"]),
    ("How long had I been taking guitar lessons when I bought the new guitar amp?",
     "how_long_when", ["taking guitar lessons", "I bought the new guitar amp"]),
    ("How many weeks had passed since I recovered from the flu when I went on my 10th jog?",
     "since_when", ["I recovered from the flu", "I went on my 10th jog"]),
    ("Which device did I set up first, the smart thermostat or the new router?",
     "a_or_b", ["the smart thermostat", "the new router"]),
    ("Who became a parent first, Rachel or Alex?", "a_or_b", ["Rachel", "Alex"]),
    ("What did I plant on the balcony, before the heatwave started in July?",
     "before_clause", ["What did I plant on the balcony", "the heatwave started in July"]),
]  # fmt: skip
NEAR_MISS = [
    "How many days ago did I buy the shoes?",
    "What is the order of the six museums I visited from earliest to latest?",
    "Which streaming service did I start using most recently?",
    "What time do I wake up on Tuesdays and Thursdays?",
    "How many days before my best friend's birthday party did I order her gift?",
    "Where did I park the car?",
    "What was my goal?",
    "",
]  # fmt: skip


@pytest.mark.parametrize(("q", "pattern", "parts"), POSITIVE)
def test_pattern_table_positives(q, pattern, parts):
    s = split_endpoints(q)
    assert s is not None and (s.pattern, s.parts) == (pattern, parts)


@pytest.mark.parametrize("q", NEAR_MISS)
def test_pattern_table_near_misses_do_not_split(q):
    assert split_endpoints(q) is None


# ---- the merge ---------------------------------------------------------------------------------


def _k(cid, p, node, date):
    return Kept(cid, p, node, date)


def test_merge_dedups_a_chunk_both_sub_queries_kept():
    a = [_k(1, 0.9, 1, "2024-01-02"), _k(2, 0.6, 2, "2024-01-03")]
    b = [_k(1, 0.8, 1, "2024-01-02"), _k(3, 0.7, 3, "2024-01-01")]
    assert [k.chunk_id for k in merge_endpoints([a, b], per_node_cap=4, limit=8)] == [3, 1, 2]


def test_merge_keeps_the_higher_p_of_a_duplicate():
    a, b = [_k(1, 0.6, 1, "2024-01-02")], [_k(1, 0.9, 1, "2024-01-02")]
    assert [k.p for k in merge_endpoints([a, b], per_node_cap=4, limit=8)] == [0.9]


def test_merge_caps_chunks_per_node_across_both_sub_queries():
    a = [_k(1, 0.9, 7, "2024-01-01"), _k(2, 0.8, 7, "2024-01-01")]
    b = [_k(3, 0.7, 7, "2024-01-01"), _k(4, 0.6, 8, "2024-02-01")]
    out = merge_endpoints([a, b], per_node_cap=2, limit=8)
    assert [k.chunk_id for k in out] == [1, 3, 4]  # round-robin: chunk 2 would be node 7's third


def test_merge_is_round_robin_to_the_limit_so_both_endpoints_survive():
    a = [_k(i, 0.9 - i / 100, i, f"2024-03-{i:02d}") for i in range(1, 6)]  # five late rows
    b = [_k(10, 0.5, 10, "2024-01-01")]  # one early row, lowest p
    out = merge_endpoints([a, b], per_node_cap=4, limit=3)
    assert [k.chunk_id for k in out] == [10, 1, 2]


def test_merge_orders_by_date_undated_last_ties_by_p():
    a = [_k(1, 0.5, 1, None), _k(2, 0.6, 2, "2024-05-01")]
    b = [_k(3, 0.9, 3, "2024-05-01"), _k(4, 0.4, 4, "2023-12-31")]
    assert [k.chunk_id for k in merge_endpoints([a, b], per_node_cap=4, limit=8)] == [4, 3, 2, 1]


# ---- the read path -----------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, tzinfo=UTC)
Q = (
    "How many days were there between the day I replaced my spark plugs and the day I "
    "harvested my first batch of fresh herbs"
)
SESSIONS = [
    ("plugs", "2024-03-01", ("I replaced my spark plugs on the car", "Nice")),
    ("herbs", "2024-04-05", ("I harvested my first batch of fresh herbs in the garden", "Great")),
    ("noise", "2024-04-10", ("Dinner was pasta with a lot of cheese", "Yum")),
]


def _store(tmp_path, *, split, tau_keep=0.5, tau_high=0.5, gate=True):
    read_cfg = dict(config.default_config().read)
    if split is not None:
        read_cfg["split_endpoints"] = split
    s = Store.open(tmp_path / "s.db", HashEmbedder(), reranker=OverlapReranker(), read=read_cfg,
                   now=lambda: NOW)  # fmt: skip
    for ref, date, rnd in SESSIONS:
        write.ingest_session(s, source_ref=ref, date=date, rounds=[rnd])
    g = Gate(1.0, 0.0, tau_keep, tau_high, 0.0)
    write_gate(s.conn, g)
    s.gate = g
    return s


def _dates(out):
    return [r["date"] for r in out.structured["rows"]]


def test_the_single_query_misses_an_endpoint_the_split_delivers_both(tmp_path):
    off = _store(_mk(tmp_path, "off"), split=False)
    on = _store(_mk(tmp_path, "on"), split=True)
    try:
        # one gold row only: the long query's overlap with the herbs round is just above tau_keep
        assert _dates(read.recall(off, query=Q)) != ["2024-03-01", "2024-04-05"]
        out = read.recall(on, query=Q)
        assert _dates(out) == ["2024-03-01", "2024-04-05"] and not out.structured["abstained"]
        assert out.structured["split"]["pattern"] == "between_and"
    finally:
        off.close()
        on.close()


def _mk(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def test_each_sub_query_is_gated_on_its_own(tmp_path):
    s = _store(tmp_path, split=True)
    try:
        # the second event matches nothing in the store: only the first endpoint's row comes back
        q = "How many days between the day I replaced my spark plugs and the day I flew to Mars"
        out = read.recall(s, query=q)
        assert _dates(out) == ["2024-03-01"]
        assert out.structured["split"]["kept"] == [1, 0]
    finally:
        s.close()


def test_both_sub_queries_abstaining_abstains_the_answer(tmp_path):
    s = _store(tmp_path, split=True)
    try:
        out = read.recall(
            s, query="How many days between the day I flew to Mars and the day I swam to Venus"
        )
        assert out.structured["abstained"] is True and out.structured["rows"] == []
        assert out.text.startswith("no relevant memories")
    finally:
        s.close()


def test_a_question_without_a_pattern_takes_the_normal_path_unchanged(tmp_path):
    on = _store(_mk(tmp_path, "a"), split=True)
    off = _store(_mk(tmp_path, "b"), split=None)
    try:
        for q in ("Where did I replace my spark plugs", "How many days ago did I harvest herbs"):
            x, y = read.recall(on, query=q), read.recall(off, query=q)
            assert x.text == y.text and x.structured == y.structured
            assert "split" not in x.structured
    finally:
        on.close()
        off.close()


def test_switch_off_is_todays_output_for_matching_questions(tmp_path):
    a = _store(_mk(tmp_path, "a"), split=None)  # key absent: the default
    b = _store(_mk(tmp_path, "b"), split=False)
    try:
        for q in (Q, "Which was first, the spark plugs or the herbs?"):
            x, y = read.recall(a, query=q), read.recall(b, query=q)
            assert x.text == y.text and x.structured == y.structured and "split" not in x.structured
    finally:
        a.close()
        b.close()


def test_extra_latency_is_recorded_when_split_and_absent_when_not(tmp_path):
    on = _store(_mk(tmp_path, "a"), split=True)
    off = _store(_mk(tmp_path, "b"), split=False)
    try:
        t_on, t_off = {}, {}
        read.recall(on, query=Q, timings=t_on)
        read.recall(off, query=Q, timings=t_off)
        assert t_on["split"] > 0 and "split" not in t_off
    finally:
        on.close()
        off.close()


def test_recall_log_names_the_split(tmp_path):
    import json

    s = _store(tmp_path, split=True)
    try:
        read.recall(s, query=Q)
        (row,) = s.conn.execute("SELECT result FROM events WHERE tool = 'recall'").fetchall()
        assert json.loads(row[0])["split"]["parts"][0] == "the day I replaced my spark plugs"
    finally:
        s.close()


# ---- the switch --------------------------------------------------------------------------------


def test_switch_defaults_off_and_leaves_the_fingerprint_alone():
    assert config.default_config().split_endpoints is False
    cfg = all_off()
    assert "split_endpoints" not in cfg.read and cfg.split_endpoints is False
    assert cfg.fp_all() == GOLDEN_FP_ALL_OFF
    assert cfg.replace("read", split_endpoints=True).fp_all() == GOLDEN_FP_ALL_SPLIT_ENDPOINTS


def test_switch_must_be_a_bool(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("read: {split_endpoints: 'yes'}\n")
    with pytest.raises(config.ConfigError, match="split_endpoints"):
        config.load(p)
    p.write_text("read: {split_endpoints: true}\n")
    assert config.load(p).split_endpoints is True
