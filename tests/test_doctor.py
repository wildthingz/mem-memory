import hashlib
import json
import sqlite3

from typer.testing import CliRunner

from mem import doctor
from mem.cli import app
from mem.pins import EMBEDDER, RERANKER, canonical_json, fp_embed

runner = CliRunner()


def _snap(hub, pin):
    snap = hub / f"models--{pin.id.replace('/', '--')}" / "snapshots" / pin.revision
    snap.mkdir(parents=True, exist_ok=True)
    return snap


def _make_snapshot(hub, pin):
    snap = _snap(hub, pin)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"w")


def test_pinned_revisions_are_the_spec_ones():
    assert EMBEDDER.id == "google/embeddinggemma-300m"
    assert EMBEDDER.revision.startswith("57c266a7") and len(EMBEDDER.revision) == 40
    assert RERANKER.id == "BAAI/bge-reranker-v2-m3"
    assert RERANKER.revision.startswith("953dc6f6") and len(RERANKER.revision) == 40


def test_fp_embed_is_sha256_of_canonical_embedder_section():
    section = {"b": 1, "a": [1, 2]}
    assert canonical_json(section) == '{"a":[1,2],"b":1}'
    expected = hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    assert fp_embed(section) == expected


def test_models_missing_then_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    missing = doctor.check_models()
    assert [c.status for c in missing] == ["fail", "fail"]
    for pin in (EMBEDDER, RERANKER):
        _make_snapshot(tmp_path / "hub", pin)
    assert [c.status for c in doctor.check_models()] == ["ok", "ok"]


def test_fetch_calls_downloader_with_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    calls = []

    def fake(repo_id, revision):
        calls.append((repo_id, revision))
        _make_snapshot(tmp_path / "hub", EMBEDDER if repo_id == EMBEDDER.id else RERANKER)
        return str(tmp_path)

    results = doctor.fetch_models(downloader=fake)
    assert calls == [(EMBEDDER.id, EMBEDDER.revision), (RERANKER.id, RERANKER.revision)]
    assert [c.status for c in results] == ["ok", "ok"]


def test_fetch_failure_names_the_licence_hint(tmp_path):
    def fake(repo_id, revision):
        raise OSError("401 gated repo")

    results = doctor.fetch_models(downloader=fake)
    assert results[0].status == "fail"
    assert "hf auth login" in results[0].detail
    assert "Gemma" in results[0].detail


def test_sqlite_vec_loads_and_reports_version():
    c = doctor.check_sqlite_vec()
    assert c.status == "ok"
    assert "vec" in c.detail and "fts5" in c.detail


def test_schema_version_read_from_store(tmp_path):
    db = tmp_path / "personal.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    con.commit()
    con.close()
    c = doctor.check_schema(db)
    assert c.status == "ok" and "2" in c.detail


def test_schema_missing_store_is_a_warning_not_a_crash(tmp_path):
    assert doctor.check_schema(tmp_path / "none.db").status == "warn"


def test_cli_doctor_exit_code_one_when_a_check_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "google/embeddinggemma-300m" in result.output


def test_cli_doctor_fetch_downloads_then_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    def fake(repo_id, revision):
        pin = EMBEDDER if repo_id == EMBEDDER.id else RERANKER
        _make_snapshot(tmp_path / "hub", pin)
        return str(tmp_path)

    monkeypatch.setattr(doctor, "_download", fake)
    monkeypatch.setattr(doctor, "check_at_rest",
                        lambda p, **_: doctor.Check("at-rest", "ok", "encrypted"))  # fmt: skip
    _fake_torch(monkeypatch, free_gib=10)
    _ship_a_gate_for_the_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["doctor", "--fetch"])
    assert result.exit_code == 0, result.output


def _ship_a_gate_for_the_config(tmp_path, monkeypatch):
    """Plain `mem doctor` checks the gates of the install's stores (MEM2-242): ship one that fits
    the config in force, and keep the install's folders in `tmp_path`."""
    from mem import install, shipped
    from mem.cli import _config
    from mem.gate import Gate

    monkeypatch.setattr(install.Paths, "here", classmethod(lambda c: c.for_home(tmp_path, env={})))
    gate = Gate(platt_a=1.0, platt_b=0.0, tau_keep=0.5, tau_high=0.5, gap=0.0, fit_id="t",
                fitted_at="2026-10-26T00:00:00Z", fingerprint_hash=_config(None, "st").fp_all(),
                tune={"queries": 1})  # fmt: skip
    path = tmp_path / "pkg" / "serving_gate.json"
    monkeypatch.setattr(shipped, "GATE_PATH", path)
    shipped.write(path, gate, shipped.metadata(
        gate, lme={"name": "LME", "version": "cleaned", "sha256": "a" * 64, "split": "tune"},
        syn={"name": "SYN", "version": 1, "seeds": [1, 2, 3, 4, 5]}))  # fmt: skip


def _hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    return tmp_path / "hub"


def _embedder_check():
    return next(c for c in doctor.check_models() if c.name == EMBEDDER.id)


def test_snapshot_with_only_readme_fails(tmp_path, monkeypatch):
    snap = _snap(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    (snap / "README.md").write_text("gated")
    c = _embedder_check()
    assert c.status == "fail"
    assert "config.json" in c.detail and "model.safetensors" in c.detail
    assert "hf auth login" in c.detail and "mem doctor --fetch" in c.detail


def test_snapshot_with_config_but_no_weights_fails(tmp_path, monkeypatch):
    snap = _snap(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    (snap / "config.json").write_text("{}")
    c = _embedder_check()
    assert c.status == "fail"
    assert "model.safetensors" in c.detail and "config.json" not in c.detail


def test_snapshot_with_config_and_weights_is_ok(tmp_path, monkeypatch):
    _make_snapshot(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    assert _embedder_check().status == "ok"


def _sharded(snap, shards, present):
    (snap / "config.json").write_text("{}")
    weight_map = {f"layer{i}": name for i, name in enumerate(shards)}
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in present:
        (snap / name).write_bytes(b"w")


def test_sharded_snapshot_with_all_shards_is_ok(tmp_path, monkeypatch):
    snap = _snap(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _sharded(snap, shards, shards)
    assert _embedder_check().status == "ok"


def test_sharded_snapshot_with_a_missing_shard_fails(tmp_path, monkeypatch):
    snap = _snap(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    _sharded(snap, shards, shards[:1])
    c = _embedder_check()
    assert c.status == "fail"
    assert "model-00002-of-00002.safetensors" in c.detail
    assert "model-00001-of-00002.safetensors" not in c.detail


def test_dangling_weight_symlink_fails(tmp_path, monkeypatch):
    snap = _snap(_hf_home(tmp_path, monkeypatch), EMBEDDER)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").symlink_to(tmp_path / "blobs" / "missing")
    assert _embedder_check().status == "fail"


def test_cli_doctor_exit_code_one_when_snapshot_holds_only_readme(tmp_path, monkeypatch):
    hub = _hf_home(tmp_path, monkeypatch)
    _make_snapshot(hub, RERANKER)
    (_snap(hub, EMBEDDER) / "README.md").write_text("gated")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1, result.output
    assert f"[FAIL] {EMBEDDER.id}" in result.output
    assert f"[ok  ] {RERANKER.id}" in result.output


def test_fetch_that_returns_but_leaves_a_half_snapshot_fails(tmp_path, monkeypatch):
    hub = _hf_home(tmp_path, monkeypatch)

    def fake(repo_id, revision):
        pin = EMBEDDER if repo_id == EMBEDDER.id else RERANKER
        snap = _snap(hub, pin)
        (snap / "README.md").write_text("gated")
        return str(snap)

    results = doctor.fetch_models(downloader=fake)
    assert [c.status for c in results] == ["fail", "fail"]
    assert "model.safetensors" in results[0].detail
    assert "Gemma" in results[0].detail


def test_cli_doctor_fetch_exits_non_zero_when_download_fails(tmp_path, monkeypatch):
    hub = _hf_home(tmp_path, monkeypatch)

    def fake(repo_id, revision):
        pin = EMBEDDER if repo_id == EMBEDDER.id else RERANKER
        if pin is EMBEDDER:
            (_snap(hub, pin) / "README.md").write_text("gated")
            raise OSError("401 Client Error: gated repo")
        _make_snapshot(hub, pin)
        return str(hub)

    monkeypatch.setattr(doctor, "_download", fake)
    monkeypatch.setattr(doctor, "check_at_rest",
                        lambda p, **_: doctor.Check("at-rest", "ok", "encrypted"))  # fmt: skip
    result = runner.invoke(app, ["doctor", "--fetch"])
    assert result.exit_code == 1, result.output
    assert f"[FAIL] fetch {EMBEDDER.id}" in result.output
    assert f"[FAIL] {EMBEDDER.id}" in result.output


def test_index_without_weight_map_is_not_a_model(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    assert doctor.missing_model_files(tmp_path) == ["model.safetensors.index.json (unreadable)"]


# --- MEM2-132: mem needs an NVIDIA GPU (the owner, 2026-10-24) ---


def _fake_torch(monkeypatch, cuda=True, free_gib=10.0):
    import sys
    import types

    t = types.ModuleType("torch")
    t.__version__ = "0-fake"
    t.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=lambda i: types.SimpleNamespace(
            name="Fake RTX", total_memory=16 * 2**30
        ),
        mem_get_info=lambda i: (int(free_gib * 2**30), 16 * 2**30),
    )
    monkeypatch.setitem(sys.modules, "torch", t)


def test_doctor_device_fails_without_torch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    assert doctor.check_device("cuda").status == "fail"


def test_doctor_device_fails_without_cuda(monkeypatch):
    _fake_torch(monkeypatch, cuda=False)
    c = doctor.check_device("cuda")
    assert c.status == "fail" and "no CUDA" in c.detail


def test_doctor_device_fails_on_device_cpu(monkeypatch):
    _fake_torch(monkeypatch)
    assert doctor.check_device("cpu").status == "fail"


def test_doctor_device_fails_under_4_gib_free(monkeypatch):
    _fake_torch(monkeypatch, free_gib=3.9)
    c = doctor.check_device("cuda")
    assert c.status == "fail" and "free VRAM" in c.detail


def test_doctor_device_passes_on_cuda_with_vram(monkeypatch):
    _fake_torch(monkeypatch, free_gib=4.5)
    assert doctor.check_device("cuda").status == "ok"
    assert doctor.check_device("auto").status == "ok"


def test_doctor_cli_exits_nonzero_on_cpu(tmp_path, monkeypatch):
    _fake_torch(monkeypatch, cuda=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    r = runner.invoke(app, ["doctor", "--allow-unencrypted", "--opt-out-log", str(tmp_path / "l")])
    assert r.exit_code == 1 and "NVIDIA GPU" in r.output
