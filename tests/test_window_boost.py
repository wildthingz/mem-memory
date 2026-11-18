"""ADR 0031 (MEM2-266): `read.window_boost`. A query with a relative date ("last Saturday") adds
`delta` to the reranker logit of pool members whose valid_from is in the window and delivers the
gated in-window rows first; nothing is filtered, no parse changes nothing, and the switch off is
today's output byte for byte. T0, on the hash models (OverlapReranker: z = 6 x overlap - 3)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from mem import config, listmode, read, window, write
from mem.gate import Gate, write_gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker
from tests.test_config import GOLDEN_FP_ALL, GOLDEN_FP_ALL_OFF, GOLDEN_FP_ALL_WINDOW_BOOST, all_off

NOW = datetime(2024, 6, 1, 12, tzinfo=UTC)  # a Saturday: last Saturday is 2024-05-25
Q = "who gave me the jewelry last saturday"  # 7 words
SESSIONS = [
    (
        "old",
        "2024-03-01",
        ("who gave me the jewelry box", "ok"),
    ),  # 5/7 (+ok): z = 1.29, out of window
    ("sat", "2024-05-25", ("my aunt gave me jewelry", "ok")),  # 3/7: z = -0.43, in the window
    ("near", "2024-05-24", ("the weather", "ok")),  # 1/7: z = -2.14, in the window
    ("noise", "2024-04-01", ("the pasta", "ok")),  # 1/7: z = -2.14, out of window
]
BOOST = {"delta": 1.0, "slack": 0.1}


def _store(path, *, boost=None, list_mode=False, tau=0.5):
    read_cfg = {"conflict_guard": False, "p3_model": False}
    if list_mode:
        read_cfg["list_mode"] = True
    if boost is not None:
        read_cfg["window_boost"] = boost
    path.mkdir(exist_ok=True)
    s = Store.open(path / "s.db", HashEmbedder(), reranker=OverlapReranker(), read=read_cfg,
                   now=lambda: NOW)  # fmt: skip
    for ref, date, rnd in SESSIONS:
        write.ingest_session(s, source_ref=ref, date=date, rounds=[rnd])
    lm = None
    if list_mode:
        lm = listmode.ListModel.build(tau, listmode.CAP_TOKENS, listmode.FLOOR, 1.0, 0.0,
                                      {"outer_folds": [{"fold": 0}]}).to_dict()  # fmt: skip
    g = Gate(1.0, 0.0, tau, tau, 0.0, list_model=lm)  # kept iff z >= 0
    write_gate(s.conn, g)
    s.gate = g
    return s


def _dates(out):
    return [r["date"] for r in out.structured["rows"]]


@pytest.fixture
def pair(tmp_path):
    off, on = _store(tmp_path / "off"), _store(tmp_path / "on", boost=BOOST)
    yield off, on
    off.close()
    on.close()


def test_an_in_window_row_is_boosted_over_the_gate_and_delivered_first(pair):
    off, on = pair
    assert _dates(read.recall(off, query=Q)) == ["2024-03-01"]
    out = read.recall(on, query=Q)
    # the out-of-window row stays (a boost, never a filter); the in-window row comes first
    assert _dates(out) == ["2024-05-25", "2024-03-01"]
    assert not out.structured["abstained"]


def test_the_gate_invariant_holds_every_row_passed_the_gate_on_its_shown_p(pair):
    _, on = pair
    out = read.recall(on, query=Q)
    tau = on.gate.tau_keep
    sat = next(r for r in out.structured["rows"] if r["date"] == "2024-05-25")
    z = 6 * 3 / 7 - 3
    assert sat["p"] == pytest.approx(1 / (1 + math.exp(-(z + BOOST["delta"]))), abs=1e-6)
    assert all(r["p"] >= tau for r in out.structured["rows"])
    # the near row is in the window but its boosted z is still under the cut: not delivered
    assert "2024-05-24" not in _dates(out)


def test_no_parse_is_no_change(pair):
    off, on = pair
    for q in ("who gave me the jewelry", "who gave me the jewelry today", "jewelry next saturday"):
        a, b = read.recall(off, query=q), read.recall(on, query=q)
        assert a.text == b.text and a.structured == b.structured


def test_switch_off_is_todays_output_byte_for_byte(tmp_path):
    absent = _store(tmp_path / "a")
    false = _store(tmp_path / "b", boost=False)
    try:
        for q in (Q, "who gave me the jewelry"):
            x, y = read.recall(absent, query=q), read.recall(false, query=q)
            assert x.text == y.text and x.structured == y.structured
        logs = [s.conn.execute("SELECT result FROM events WHERE tool = 'recall' ORDER BY id").fetchall()
                for s in (absent, false)]  # fmt: skip
        assert logs[0] == logs[1] and all("window" not in json.loads(r[0]) for r in logs[0])
    finally:
        absent.close()
        false.close()


def test_the_log_and_the_timings_name_the_window(pair):
    _, on = pair
    t: dict[str, float] = {}
    read.recall(on, query=Q, timings=t)
    (row,) = on.conn.execute("SELECT result FROM events WHERE tool = 'recall'").fetchall()
    logged = json.loads(row[0])
    assert logged["window"] == {
        "phrases": [["last saturday", "2024-05-24", "2024-05-26"]],
        "in_window": 2,
    }
    assert "window" in t
    # the log keeps the raw reranker logit (calibration input) beside the boosted p
    sat = logged["feat"][1]
    assert sat["z"] == pytest.approx(6 * 3 / 7 - 3, abs=1e-4)
    assert sat["p"] == pytest.approx(1 / (1 + math.exp(-(sat["z"] + BOOST["delta"]))), abs=1e-4)


def test_list_mode_boosts_and_orders_in_window_rows_first(tmp_path):
    tau = 1 / (1 + math.exp(0.5))  # z >= -0.5
    off = _store(tmp_path / "off", list_mode=True, tau=tau)
    on = _store(tmp_path / "on", list_mode=True, boost=BOOST, tau=tau)
    try:
        # 9 words: the old session 4/9 (z = -0.33, kept), the Saturday one 3/9 (z = -1.0; -0.0 boosted)
        q = "how many times who gave me jewelry last saturday"
        a, b = read.recall(off, query=q), read.recall(on, query=q)
        assert b.structured.get("list_mode") and "2024-05-25" in _dates(b)
        assert "2024-05-25" not in _dates(a)
        assert set(_dates(a)) <= set(_dates(b))  # nothing the gate kept is dropped
    finally:
        off.close()
        on.close()


def test_as_of_is_the_reference_day(pair):
    _, on = pair
    # a week later, "last Saturday" is 2024-06-01: no session there, nothing boosted
    out = read.recall(on, query=Q, as_of="2024-06-08T12:00:00Z")
    assert _dates(out) == ["2024-03-01"]


# ---- the switch --------------------------------------------------------------------------------


@pytest.mark.release_defaults
def test_switch_is_on_at_delta_3_in_the_release_default_and_off_drops_it():
    """MEM2-274: the release default is Round 10 r10-all, window boost {delta 3.0, slack 0.1} (ADR 0031
    adopted at the host's tuned cell). Off drops the key and so changes the fingerprint."""
    cfg = config.default_config()
    assert cfg.read["window_boost"] == {"delta": 3.0, "slack": 0.1} == cfg.window_boost
    assert cfg.fp_all() == GOLDEN_FP_ALL
    off = cfg.replace("read", window_boost=False)
    assert "window_boost" not in off.read and off.fp_all() != GOLDEN_FP_ALL
    assert all_off().fp_all() == GOLDEN_FP_ALL_OFF
    on = all_off().replace("read", window_boost=window.settings(True))
    assert on.fp_all() == GOLDEN_FP_ALL_WINDOW_BOOST


@pytest.mark.release_defaults
def test_switch_in_yaml_true_is_the_tuned_values_a_mapping_is_checked(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("read: {window_boost: true}\n")
    assert config.load(p).window_boost == {"delta": window.DELTA, "slack": window.SLACK}
    assert config.load(p).read["window_boost"] == {"delta": window.DELTA, "slack": window.SLACK}
    p.write_text("read: {window_boost: {delta: 2, slack: 0.3}}\n")
    assert config.load(p).window_boost == {"delta": 2.0, "slack": 0.3}
    p.write_text("read: {window_boost: {delta: 3.0, slack: 0.1}}\n")  # the default written out
    assert config.load(p).fp_all() == GOLDEN_FP_ALL
    p.write_text("read: {window_boost: false}\n")
    assert config.load(p).window_boost is None and config.load(p).fp_all() != GOLDEN_FP_ALL
    for bad in ("'yes'", "{delta: 0, slack: 0.1}", "{delta: 1}"):
        p.write_text(f"read: {{window_boost: {bad}}}\n")
        with pytest.raises(config.ConfigError, match="window_boost"):
            config.load(p)


def test_the_tuned_values_are_the_dry_runs():
    assert (window.DELTA, window.SLACK) == (1.0, 0.1)


def test_list_mode_with_the_list_fill_on_too_keeps_the_matches_first_then_fills_in_window_first(
    tmp_path,
):
    """Round 10 merge (MEM2-270): `read.window_boost` (ADR 0031) and `read.list_fill` (ADR 0030) both on.
    The matches (above tau_list) go first, the in-window ones ahead; the fill never displaces a match,
    is chosen by p, and within it the in-window rows go first too."""
    tau = 1 / (1 + math.exp(0.5))  # z >= -0.5
    s = _store(tmp_path / "on", list_mode=True, boost=BOOST, tau=tau)
    # 3/9 words, out of the window: z = -1.0, a fill row ranked above the in-window `near` (-2.33 + 1)
    write.ingest_session(
        s, source_ref="mid", date="2024-04-15", rounds=[("who gave me a book", "ok")]
    )
    s.read_cfg = {**s.read_cfg, "list_fill": 4}
    try:
        q = "how many times who gave me jewelry last saturday"
        out = read.recall(s, query=q)
        assert out.structured["list_mode"] and out.structured["matches"] == 2
        assert out.structured["filled"] == 2 and len(out.structured["rows"]) == 4
        # delivery order: the in-window match, the other match, then the fill with its in-window row first
        assert _dates(out) == ["2024-05-25", "2024-03-01", "2024-05-24", "2024-04-15"]
        ps = [r["p"] for r in out.structured["rows"]]
        assert all(p >= tau for p in ps[:2]) and all(p < tau for p in ps[2:])
        assert ps[2] < ps[3]  # the fill is by p, but the in-window fill row is delivered ahead
        assert "4 results (of " in out.text and "4 sessions" in out.text.splitlines()[0]
    finally:
        s.close()
