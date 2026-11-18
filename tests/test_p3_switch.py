"""`read.p3_model` (MEM2-183, ADR 0020): the P3 read model is OFF by default; the serving gate is the
Platt gate on the rerank logit. A gate without a read_model is fine when the switch is off and
refused when it is on."""

import pytest

from mem import config, read
from mem.gate import Gate, write_gate
from mem.readmodel import ReadModel
from mem.store import Store
from mem.testing import HashEmbedder, OverlapReranker


def _store(tmp_path, gate, **read_cfg):
    cfg = {**config.default_config().read, **read_cfg}
    s = Store.open(tmp_path / "p.db", HashEmbedder(), reranker=OverlapReranker(), read=cfg)
    s.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    write_gate(s.conn, gate)
    s.gate = gate
    return s


# a read model whose p is ~0 for every row: if it scored, the answer would abstain
_NEVER = ReadModel.build(["rerank", "cosine"], [0.0, 0.0], -20.0).to_dict()


def test_p3_model_defaults_off_and_is_in_the_fingerprint():
    cfg = config.default_config()
    assert cfg.read["p3_model"] is False and cfg.p3_model is False
    assert cfg.replace("read", p3_model=True).fp_all() != cfg.fp_all()


def test_p3_model_must_be_a_bool(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("read: {p3_model: 'yes'}\n")
    with pytest.raises(config.ConfigError, match="p3_model"):
        config.load(p)
    p.write_text("read: {p3_model: true}\n")
    assert config.load(p).p3_model is True


def test_switch_off_serves_the_platt_gate_even_when_the_gate_carries_a_read_model(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(read, "pool_cosines", lambda *a, **k: pytest.fail("P3 is off: no cosine"))
    s = _store(tmp_path, Gate(1.0, 0.0, 0.5, 0.5, 0.0, read_model=_NEVER))
    out = read.recall(s, query="dog Bosco")
    assert [r["id"] for r in out.structured["rows"]] == ["m1"]
    s.close()


def test_switch_on_scores_with_the_read_model(tmp_path):
    s = _store(tmp_path, Gate(1.0, 0.0, 0.5, 0.5, 0.0, read_model=_NEVER), p3_model=True)
    out = read.recall(s, query="dog Bosco")
    assert out.structured["abstained"] is True
    s.close()


def test_switch_on_refuses_a_gate_without_a_read_model(tmp_path):
    s = _store(tmp_path, Gate(1.0, 0.0, 0.5, 0.5, 0.0), p3_model=True)
    with pytest.raises(ValueError, match="p3_model"):
        read.recall(s, query="dog Bosco")
    s.close()
