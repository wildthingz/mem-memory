"""Scope note §0, the everything-local test (MEM2-221): with every Round 9 switch on (F1 dates, F2+F3
list mode, B paging, F5 session leg, D split endpoints) and the F model entries chosen in `models:`,
`remember`, `recall` (plain, list, paged, split) and `get` make no network call. Stub models only;
any non-Unix-socket connection or name lookup fails the test."""

from __future__ import annotations

import socket
from datetime import UTC, datetime

import pytest

from mem import catalog, config, read, write
from mem.gate import Gate, write_gate
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker
from tests.test_list_mode import _lm

ALL_ON = """\
models: {embedder: qwen3-embedding-0.6b, reranker: mxbai-rerank-large-v2}
ingest: {resolve_dates: true}
retrieval: {session_leg: true}
read: {list_mode: true, list_paging: true, split_endpoints: true, window_boost: true, list_fill: 16}
"""
NOW = datetime(2024, 6, 1, tzinfo=UTC)
SESSIONS = [
    ("plugs", "2024-03-01", ("Yesterday I replaced my spark plugs on the car", "Nice")),
    ("herbs", "2024-04-05", ("I harvested my first batch of fresh herbs last week", "Great")),
    ("noise", "2024-04-10", ("Dinner was pasta with a lot of cheese", "Yum")),
]
QUESTIONS = [
    "Where did I replace my spark plugs",  # plain
    "How many times did I replace or harvest things, list them all",  # list mode
    "How many days between the day I replaced my spark plugs and the day I harvested my herbs",  # D
    "What did I harvest last month",  # the window boost
]


@pytest.fixture
def no_network(monkeypatch):
    calls: list[str] = []

    def refuse(name):
        def _f(*a, **k):
            calls.append(f"{name}{a[1:2]}")
            raise AssertionError(f"network call: {name}")

        return _f

    real_connect = socket.socket.connect

    def connect(self, address):
        if self.family != socket.AF_UNIX:  # local files and Unix sockets are not the network
            refuse("connect")(self, address)
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("connect_ex"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    monkeypatch.setattr(socket, "gethostbyname", refuse("gethostbyname"))
    return calls


def test_every_switch_on_and_the_f_models_chosen_make_no_network_call(tmp_path, no_network):
    p = tmp_path / "c.yaml"
    p.write_text(ALL_ON)
    cfg = config.load(p)
    assert cfg.fp_all() != config.default_config().fp_all()  # the switches really are on
    assert (
        cfg.list_mode
        and cfg.list_paging
        and cfg.split_endpoints
        and cfg.retrieval["session_leg"]
        and cfg.window_boost
        and cfg.list_fill == 16
    )
    assert catalog.vram_gib(cfg) > 0  # the F entries resolve from the pinned catalog, no download

    s = Store.open(tmp_path / "s.db", HashEmbedder(), reranker=OverlapReranker(), read=cfg.read,
                   retrieval=cfg.retrieval, ingest=cfg.ingest, now=lambda: NOW)  # fmt: skip
    try:
        for ref, date, rnd in SESSIONS:
            write.ingest_session(s, source_ref=ref, date=date, rounds=[rnd])
        write.remember(s, text="I like green tea", type="preference", about=["me"])
        g = Gate(1.0, 0.0, 0.5, 0.5, 0.0, list_model=_lm(0.5, (1.0, 0.0)))
        write_gate(s.conn, g)
        s.gate = g
        ids: list[str] = []
        for q in QUESTIONS:
            out = read.recall(s, query=q)
            ids += [r["id"] for r in out.structured["rows"]]
            assert out.text
        assert ids
        assert read.get(s, ids[:2]).text
    finally:
        s.close()
    assert no_network == []


def test_the_guard_catches_a_network_connection(no_network):
    with pytest.raises(AssertionError, match="network call"):
        socket.create_connection(("example.com", 80))
    with socket.socket(socket.AF_INET) as sock, pytest.raises(AssertionError, match="network call"):
        sock.connect(("127.0.0.1", 9))
