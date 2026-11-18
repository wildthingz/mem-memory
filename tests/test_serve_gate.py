"""`mem serve` refuses to start on a fingerprint or gate mismatch (architecture-v2 §3.3, A2, A13),
`mem calibrate` writes the gate, `reembed` + `switch` change the embedder atomically (A21)."""

import json
import sqlite3

import pytest
import yaml
from typer.testing import CliRunner

from mem import calibrate, reembed
from mem.cli import app
from mem.config import default_config
from mem.gate import Gate, gate_json, read_gate
from mem.serve import preflight
from mem.store import Store
from mem.testing import HashEmbedder
from tests.helpers import STUB, gate_stub_store

_gated_store = gate_stub_store


def _dump(path):
    return path.read_bytes()


def _serve(tmp_path, *extra):
    return CliRunner().invoke(app, ["serve", "--allow-cpu", "--embedder", "hash", "--store",
                                    f"personal={tmp_path}/p.db", *extra])  # fmt: skip


def test_preflight_passes_a_matching_store(tmp_path):
    _gated_store(tmp_path / "p.db")
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []


def test_serve_refuses_on_fingerprint_mismatch(tmp_path):
    _gated_store(tmp_path / "p.db")
    before = _dump(tmp_path / "p.db")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"retrieval": {"pool": 20}}))
    r = _serve(tmp_path, "--config", str(cfg_file))
    assert r.exit_code == 1
    assert "gate" in r.output and "fingerprint" in r.output
    assert "mem calibrate" in r.output
    assert _dump(tmp_path / "p.db") == before  # refused before touching the store


def test_serve_refuses_on_embedder_fingerprint_mismatch(tmp_path):
    _gated_store(tmp_path / "p.db")
    with sqlite3.connect(tmp_path / "p.db") as conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'fingerprint'",
                     (json.dumps({"embed": "9" * 64}),))  # fmt: skip
    problems = preflight({"personal": tmp_path / "p.db"}, STUB)
    assert len(problems) == 1 and "embedder fingerprint" in problems[0]
    assert "mem switch" in problems[0]
    assert (
        f"mem reembed --store personal={tmp_path / 'p.db'} --config <new config.yaml>"
        in problems[0]
    )
    assert _serve(tmp_path).exit_code == 1


def test_serve_refuses_without_a_gate(tmp_path):
    Store.open(tmp_path / "p.db", HashEmbedder(), fp=STUB.fp_embed()).close()
    r = _serve(tmp_path)
    assert r.exit_code == 1 and "no gate" in r.output and "mem calibrate" in r.output


def test_serve_refuses_a_store_that_does_not_exist(tmp_path):
    r = _serve(tmp_path)
    assert r.exit_code == 1 and "mem calibrate" in r.output
    assert not (tmp_path / "p.db").exists()


def test_serve_warns_but_starts_on_an_infeasible_gate(tmp_path):
    """MEM2-87: a gate whose fit met no setting of the five P2 tune targets is served (the person
    sees the numbers in use) with one WARNING naming the unmet targets; preflight does not block."""
    from mem.serve import gate_warnings

    tune = {"feasible": False, "rule": "fallback_coverage_ceiling",
            "targets": {"coverage_any": {"met": False}, "far": {"met": True}}}  # fmt: skip
    _gated_store(tmp_path / "p.db", tune=tune)
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []
    warnings = gate_warnings({"personal": tmp_path / "p.db"})
    assert len(warnings) == 1 and "INFEASIBLE" in warnings[0] and "coverage_any" in warnings[0]
    assert "far" not in warnings[0].split("unmet:")[1].split(")")[0]
    _gated_store(tmp_path / "q.db", tune={"feasible": True, "rule": "strictest"})
    assert gate_warnings({"personal": tmp_path / "q.db", "missing": tmp_path / "x.db"}) == []


def test_serve_refuses_when_config_gate_differs_from_the_store(tmp_path):
    _gated_store(tmp_path / "p.db", tau_keep=0.5)
    cfg = STUB.replace("gate", tau_keep=0.4)
    problems = preflight({"personal": tmp_path / "p.db"}, cfg)
    assert len(problems) == 1 and "tau_keep" in problems[0]


def test_preflight_refuses_a_gate_without_a_read_model_only_when_p3_model_is_on(tmp_path):
    cfg_on = STUB.replace("read", p3_model=True)
    _gated_store(tmp_path / "p.db", cfg=cfg_on)
    problems = preflight({"personal": tmp_path / "p.db"}, cfg_on)
    assert len(problems) == 1 and "p3_model" in problems[0] and "read_model" in problems[0]


def test_preflight_accepts_a_gate_without_a_read_model_when_p3_model_is_off(tmp_path):
    _gated_store(tmp_path / "p.db")
    assert STUB.p3_model is False
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []


# ---- mem calibrate --------------------------------------------------------------------------


def _records(tmp_path, fp_all):
    pools = [
        calibrate.QueryPool("q1", True, [calibrate.PoolItem("c1", 3.0, 1),
                                         calibrate.PoolItem("c2", -2.0, 0)]),
        calibrate.QueryPool("q2", True, [calibrate.PoolItem("c3", 2.0, 1),
                                         calibrate.PoolItem("c4", -1.0, 0)]),
        calibrate.QueryPool("q3", False, [calibrate.PoolItem("c5", -1.5, 0),
                                          calibrate.PoolItem("c6", -3.0, 0)]),
    ]  # fmt: skip
    path = tmp_path / "records.jsonl"
    calibrate.write_records(path, pools, fp_all=fp_all)
    return path


def test_calibrate_cli_writes_a_gate_that_serve_accepts(tmp_path):
    rec = _records(tmp_path, STUB.fp_all())
    r = CliRunner().invoke(app, ["calibrate", "--records", str(rec), "--embedder", "hash",
                                 "--fit-id", "lme-tune-test", "--store",
                                 f"personal={tmp_path}/p.db"])  # fmt: skip
    assert r.exit_code == 0, r.output
    conn = sqlite3.connect(tmp_path / "p.db")
    gate = json.loads(conn.execute("SELECT value FROM meta WHERE key='gate'").fetchone()[0])
    conn.close()
    assert gate["fit_id"] == "lme-tune-test"
    assert gate["fingerprint_hash"] == STUB.fp_all()
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []


def test_calibrate_cli_refuses_records_from_another_config(tmp_path):
    rec = _records(tmp_path, "1" * 64)
    r = CliRunner().invoke(app, ["calibrate", "--records", str(rec), "--embedder", "hash",
                                 "--store", f"personal={tmp_path}/p.db"])  # fmt: skip
    assert r.exit_code == 1 and "fp_all" in r.output
    assert not (tmp_path / "p.db").exists()


def test_score_pool_returns_raw_logits_without_a_gate(tmp_path):
    from mem.testing import OverlapReranker

    s = Store.open(tmp_path / "p.db", HashEmbedder(), reranker=OverlapReranker())
    s.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    s.remember(text="The cat sleeps.", type="fact", about=["new:thing:Cat"])
    pool = calibrate.score_pool(s, "dog Bosco")
    assert [(item.key, item.z) for item in pool] == [("c2", 3.0), ("c4", -3.0)]
    assert s.conn.execute("SELECT count(*) FROM events").fetchone() == (2,)  # remembers only
    s.close()


# ---- reembed + switch -----------------------------------------------------------------------


class Wide(HashEmbedder):
    def __init__(self):
        super().__init__(dim=64)


def test_reembed_then_switch_moves_vectors_fingerprint_and_gate_at_once(tmp_path):
    path = tmp_path / "p.db"
    _gated_store(path)
    s = Store.open(path, HashEmbedder(), fp=STUB.fp_embed())
    s.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    old_table = s.vec_table
    new_cfg = STUB.replace("embedder", dim=64)
    table = reembed.reembed(s.conn, Wide(), new_cfg.fp_embed())
    assert table == new_cfg.vec_table() != old_table
    assert s.conn.execute(f"SELECT count(*) FROM {table}").fetchone() == (2,)
    s.close()
    new_gate = Gate(1.0, 0.0, 0.4, 0.6, 0.1, fit_id="new", fingerprint_hash=new_cfg.fp_all())
    with pytest.raises(ValueError, match="fingerprint"):  # a gate for another config
        reembed.switch(path, new_cfg, Gate(1.0, 0.0, 0.4, 0.6, 0.1, fingerprint_hash="2" * 64))
    conn = sqlite3.connect(path)
    assert json.loads(conn.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()[0]) == {
        "embed": STUB.fp_embed()}  # unchanged after the refused switch  # fmt: skip
    conn.close()
    reembed.switch(path, new_cfg, new_gate)
    assert preflight({"personal": path}, new_cfg) == []
    s = Store.open(path, Wide(), fp=new_cfg.fp_embed())
    assert read_gate(s.conn) == new_gate
    assert s.vec_table == table
    s.close()


def test_switch_refuses_before_reembed(tmp_path):
    path = tmp_path / "p.db"
    _gated_store(path)
    new_cfg = STUB.replace("embedder", dim=64)
    gate = Gate(1.0, 0.0, 0.4, 0.6, 0.1, fingerprint_hash=new_cfg.fp_all())
    with pytest.raises(ValueError, match="reembed"):
        reembed.switch(path, new_cfg, gate)


# ---- mem reembed CLI ------------------------------------------------------------------------


def test_reembed_cli_then_switch_moves_a_store_to_the_new_embedder(tmp_path, free_port):
    path = tmp_path / "p.db"
    old = Store.open(path, HashEmbedder(dim=768), fp=default_config().fp_embed())
    old.remember(text="Bosco is a dog.", type="fact", about=["new:thing:Bosco"])
    old.close()
    r = CliRunner().invoke(app, ["reembed", "--embedder", "hash", "--store", f"personal={path}",
                                 "--port", str(free_port)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert f"personal: 2 chunks in {STUB.vec_table()}" in r.output
    assert "mem switch" in r.output
    gate_file = tmp_path / "gate.json"
    gate_file.write_text(gate_json(Gate(1.0, 0.0, 0.5, 0.5, 0.0, fingerprint_hash=STUB.fp_all())))
    r = CliRunner().invoke(app, ["switch", "--embedder", "hash", "--gate", str(gate_file),
                                 "--store", f"personal={path}"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert preflight({"personal": path}, STUB) == []


def test_reembed_cli_never_creates_a_store(tmp_path):
    r = CliRunner().invoke(app, ["reembed", "--embedder", "hash", "--store",
                                 f"personal={tmp_path}/p.db"])  # fmt: skip
    assert r.exit_code == 1
    assert "no store" in r.output
    assert not (tmp_path / "p.db").exists()


def test_reembed_cli_refuses_an_embedder_the_pinned_model_is_not(tmp_path, free_port):
    _gated_store(tmp_path / "p.db")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump({"embedder": {"max_len": 256}}))
    r = CliRunner().invoke(app, ["reembed", "--config", str(cfg_file), "--store",
                                 f"personal={tmp_path}/p.db", "--port", str(free_port)])  # fmt: skip
    assert r.exit_code == 1  # refused before any model loads
    assert "pinned" in r.output


def test_preflight_refuses_a_gate_without_a_list_model_only_when_list_mode_is_on(tmp_path):
    cfg_on = STUB.replace("read", list_mode=True)
    _gated_store(tmp_path / "p.db", cfg=cfg_on)
    problems = preflight({"personal": tmp_path / "p.db"}, cfg_on)
    assert len(problems) == 1 and "list_mode" in problems[0] and "list_model" in problems[0]
    _gated_store(tmp_path / "q.db")
    assert STUB.list_mode is False
    assert preflight({"personal": tmp_path / "q.db"}, STUB) == []
