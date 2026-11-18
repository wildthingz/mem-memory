"""The serving gate fitted on public data ships as package data (MEM2-233, ADR 0017).

T0, on the hash models: the shipped file is whatever `bench package-gate` wrote; the real fit is a
host step (deliverables/runbooks/setup-gate.md)."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from bench.cli import app as bench_app
from bench.datasets import lme
from mem import calibrate, doctor, shipped
from mem.cli import app
from mem.common import MemError
from mem.gate import Gate, gate_json, read_gate
from mem.serve import _open_all, preflight
from mem.store import Store
from mem.testing import HashEmbedder
from tests.helpers import STUB, gate_stub_store

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_CONFIG = {"retrieval": {"pool": 20}}  # a different fp_all than STUB


def _gate(cfg=STUB, **kw) -> Gate:
    fields = {"platt_a": 1.0, "platt_b": 0.0, "tau_keep": 0.5, "tau_high": 0.5, "gap": 0.0,
              "fit_id": "shipped-test", "fitted_at": "2026-10-26T12:00:00Z",
              "fingerprint_hash": cfg.fp_all(), "tune": {"queries": 3}, **kw}  # fmt: skip
    return Gate(**fields)


def _meta(gate: Gate) -> dict:
    return shipped.metadata(
        gate,
        lme={"name": "LongMemEval-S cleaned", "version": "cleaned", "sha256": "a" * 64,
             "split": "tune"},
        syn={"name": "SYN", "version": 1, "seeds": [1, 2, 3, 4, 5]},
    )  # fmt: skip


@pytest.fixture
def ship(tmp_path, monkeypatch):
    """Write a shipped gate for STUB where the package looks for it."""
    path = tmp_path / "pkg" / "serving_gate.json"
    monkeypatch.setattr(shipped, "GATE_PATH", path)

    def put(gate: Gate | None = None) -> Gate:
        gate = gate or _gate()
        shipped.write(path, gate, _meta(gate))
        return gate

    return put


def _serve(tmp_path, *extra):
    return CliRunner().invoke(app, ["serve", "--allow-cpu", "--embedder", "hash", "--store",
                                    f"personal={tmp_path}/p.db", *extra])  # fmt: skip


# ---- the file in the wheel ---------------------------------------------------------------------


def test_the_wheel_contains_the_gate(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(ROOT / "src" / "mem", src / "src" / "mem",
                    ignore=shutil.ignore_patterns("__pycache__", "shipped"))  # fmt: skip
    shutil.copytree(ROOT / "bench", src / "bench", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", src / "pyproject.toml")
    shutil.copytree(ROOT / "deploy", src / "deploy")  # force-included as package data
    shutil.copytree(ROOT / "skills", src / "skills")
    for extra in ("README.md", "LICENSE"):
        if (ROOT / extra).exists():
            shutil.copy(ROOT / extra, src / extra)
    gate = _gate()
    shipped.write(src / "src" / "mem" / "shipped" / "serving_gate.json", gate, _meta(gate))
    out = tmp_path / "dist"
    cmd = ["uv", "build", "--wheel", "--offline", "--out-dir", str(out)]
    done = subprocess.run(cmd, check=False, cwd=src, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    names = zipfile.ZipFile(next(out.glob("*.whl"))).namelist()
    assert "mem/shipped/serving_gate.json" in names


def test_the_shipped_location_is_under_the_package():
    assert shipped.GATE_PATH == ROOT / "src" / "mem" / "shipped" / "serving_gate.json"


# ---- serving with no local gate ----------------------------------------------------------------


def test_serve_starts_with_no_local_gate_when_the_shipped_one_fits(tmp_path, ship):
    ship()
    Store.open(tmp_path / "p.db", HashEmbedder(), fp=STUB.fp_embed()).close()  # no gate in it
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []
    servers = _open_all({"personal": tmp_path / "p.db"}, HashEmbedder(), object(),  # type: ignore[arg-type]
                        STUB, "h", "UTC", "s")  # fmt: skip
    try:
        assert servers["personal"].store.gate.fit_id == "shipped-test"
    finally:
        servers["personal"].close()
    plain = Store.open(tmp_path / "p.db", HashEmbedder(), fp=STUB.fp_embed())
    try:
        assert read_gate(plain.conn) is None  # never written into the store
    finally:
        plain.close()


def test_a_store_that_does_not_exist_yet_may_start_on_the_shipped_gate(tmp_path, ship):
    ship()
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []


def test_without_a_shipped_gate_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "absent.json")
    Store.open(tmp_path / "p.db", HashEmbedder(), fp=STUB.fp_embed()).close()
    (problem,) = preflight({"personal": tmp_path / "p.db"}, STUB)
    assert "no gate" in problem and "mem calibrate" in problem
    r = _serve(tmp_path)
    assert r.exit_code == 1 and "no gate" in r.output


def test_a_local_gate_wins(tmp_path, ship):
    ship()
    gate_stub_store(tmp_path / "p.db", fit_id="local-fit")
    assert preflight({"personal": tmp_path / "p.db"}, STUB) == []
    servers = _open_all({"personal": tmp_path / "p.db"}, HashEmbedder(), object(),  # type: ignore[arg-type]
                        STUB, "h", "UTC", "s")  # fmt: skip
    try:
        assert servers["personal"].store.gate.fit_id == "local-fit"
    finally:
        servers["personal"].close()


def test_a_local_gate_for_another_config_is_not_rescued_by_the_shipped_one(tmp_path, ship):
    ship()
    other = STUB.replace("retrieval", pool=20)
    gate_stub_store(tmp_path / "p.db", cfg=STUB, fingerprint_hash=other.fp_all())
    (problem,) = preflight({"personal": tmp_path / "p.db"}, STUB)
    assert "gate fingerprint mismatch" in problem and "mem calibrate" in problem


# ---- mismatch: refused, the fix printed --------------------------------------------------------


def test_a_shipped_gate_for_another_config_is_refused_with_the_fix(tmp_path, ship):
    ship()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(PROBLEM_CONFIG))
    store = tmp_path / "p.db"
    r = _serve(tmp_path, "--config", str(cfg_file))
    assert r.exit_code == 1
    assert "shipped gate" in r.output and "fingerprint" in r.output
    assert f"mem calibrate --records <tune records> --config <config.yaml> --store personal={store}" \
        in r.output  # fmt: skip
    assert not store.exists()


def test_doctor_says_which_command_fixes_a_fingerprint_change(tmp_path, ship):
    ship()
    other = STUB.replace("retrieval", pool=20)
    store = tmp_path / "personal.db"
    Store.open(store, HashEmbedder(), fp=other.fp_embed()).close()
    (check,) = doctor.check_gate([store], other)
    assert check.status == "fail"
    assert "mem calibrate" in check.detail and f"personal={store}" in check.detail
    assert shipped.load().metadata["fingerprint"][:8] in check.detail


def test_doctor_reports_ok_for_the_shipped_and_for_a_local_gate(tmp_path, ship):
    ship()
    store = tmp_path / "personal.db"
    Store.open(store, HashEmbedder(), fp=STUB.fp_embed()).close()
    (check,) = doctor.check_gate([store], STUB)
    assert check.status == "ok" and "shipped" in check.detail
    gate_stub_store(tmp_path / "work.db", scope="work")
    (check,) = doctor.check_gate([tmp_path / "work.db"], STUB)
    assert check.status == "ok" and "local" in check.detail


def test_doctor_fails_with_no_gate_anywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "absent.json")
    store = tmp_path / "personal.db"
    Store.open(store, HashEmbedder(), fp=STUB.fp_embed()).close()
    (check,) = doctor.check_gate([store], STUB)
    assert check.status == "fail" and "mem calibrate" in check.detail


# ---- metadata ----------------------------------------------------------------------------------


def test_the_metadata_is_complete(tmp_path, ship):
    gate = ship()
    meta = shipped.load().metadata
    assert meta["fingerprint"] == gate.fingerprint_hash == STUB.fp_all()
    assert meta["fit_date"] == "2026-10-26T12:00:00Z"
    assert meta["mem_version"]
    assert meta["datasets"]["lme"] == {"name": "LongMemEval-S cleaned", "version": "cleaned",
                                       "sha256": "a" * 64, "split": "tune"}  # fmt: skip
    assert meta["datasets"]["syn"]["version"] == 1
    assert meta["datasets"]["syn"]["seeds"] == [1, 2, 3, 4, 5]
    (lic,) = meta["licences"]
    assert lic["name"] == "LongMemEval" and lic["spdx"] == "MIT"
    assert lic["copyright"] == "Copyright (c) 2024 Di Wu"


def test_an_incomplete_file_is_refused_when_loaded(tmp_path, ship):
    gate = ship()
    doc = json.loads(shipped.GATE_PATH.read_text())
    del doc["metadata"]["datasets"]["lme"]["sha256"]
    shipped.GATE_PATH.write_text(json.dumps(doc))
    with pytest.raises(MemError, match="sha256"):
        shipped.load()
    doc["metadata"]["datasets"]["lme"]["sha256"] = "a" * 64
    doc["metadata"]["fingerprint"] = "0" * 64
    shipped.GATE_PATH.write_text(json.dumps(doc))
    with pytest.raises(MemError, match="fingerprint"):
        shipped.load()
    assert gate  # the gate itself was fine; the metadata disagreed with it


def test_a_test_split_dataset_label_is_refused_in_metadata():
    gate = _gate()
    with pytest.raises(MemError, match="tune"):
        shipped.metadata(gate, lme={"name": "x", "version": "v", "sha256": "a" * 64,
                                    "split": "test"},
                         syn={"name": "SYN", "version": 1, "seeds": [1]})  # fmt: skip


# ---- the packager ------------------------------------------------------------------------------


def _lme_world(tmp_path, fit_ids, test_ids=("t1", "t2")):
    lme_file = tmp_path / "lme.json"
    lme_file.write_text("[]")
    rows = [lme.ManifestRow(i, "multi-session", False, "tune", k) for k, i in enumerate(
        ["a1", "a2", "a3"])]  # fmt: skip
    rows += [lme.ManifestRow(i, "multi-session", False, "test", 10 + k)
             for k, i in enumerate(test_ids)]  # fmt: skip
    manifest = tmp_path / "lme_split.tsv"
    lme.write_manifest(rows, manifest)
    pools = [calibrate.QueryPool(i, True, [calibrate.PoolItem("c1", 1.0, 1)]) for i in fit_ids]
    records = tmp_path / "tune_records.jsonl"
    calibrate.write_records(records, pools, fp_all=STUB.fp_all())
    gate = tmp_path / "gate.json"
    gate.write_text(gate_json(_gate(tune={"queries": len(fit_ids)})) + "\n")
    return lme_file, manifest, records, gate


def _package(tmp_path, fit_ids, dest=None, **over):
    lme_file, manifest, records, gate = _lme_world(tmp_path, fit_ids)
    dest = dest or tmp_path / "out" / "serving_gate.json"
    args = ["package-gate", str(gate), "--records", str(records), "--lme-file", str(lme_file),
            "--manifest", str(manifest), "--out", str(dest)]  # fmt: skip
    for k, v in over.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return CliRunner().invoke(bench_app, args), dest


def test_the_packager_writes_the_file_with_metadata(tmp_path):
    r, dest = _package(tmp_path, ["a1", "a2", "a3"])
    assert r.exit_code == 0, r.output
    loaded = shipped.load(dest)
    assert loaded.gate.fit_id == "shipped-test"
    lme_meta = loaded.metadata["datasets"]["lme"]
    assert lme_meta["split"] == "tune" and len(lme_meta["sha256"]) == 64
    assert loaded.metadata["datasets"]["syn"]["seeds"] == [1, 2, 3, 4, 5]
    assert loaded.metadata["licences"][0]["spdx"] == "MIT"


def test_the_packager_refuses_a_fit_on_a_test_question(tmp_path):
    r, dest = _package(tmp_path, ["a1", "a2", "t1"])
    assert r.exit_code == 1
    assert "t1" in r.output and "test" in r.output
    assert not dest.exists()


def test_the_packager_refuses_records_that_are_not_the_gates_fit(tmp_path):
    lme_file, manifest, records, gate = _lme_world(tmp_path, ["a1", "a2"])
    gate.write_text(gate_json(_gate(tune={"queries": 3})) + "\n")  # fitted on 3, records hold 2
    dest = tmp_path / "serving_gate.json"
    r = CliRunner().invoke(bench_app, ["package-gate", str(gate), "--records", str(records),
                                       "--lme-file", str(lme_file), "--manifest", str(manifest),
                                       "--out", str(dest)])  # fmt: skip
    assert r.exit_code == 1 and "records" in r.output and not dest.exists()


def test_the_packager_refuses_held_out_syn_seeds(tmp_path):
    r, dest = _package(tmp_path, ["a1", "a2", "a3"], syn_seeds="1,2,101")
    assert r.exit_code == 1 and "101" in r.output and not dest.exists()


def test_the_packager_refuses_records_for_another_config(tmp_path):
    lme_file, manifest, records, gate = _lme_world(tmp_path, ["a1", "a2", "a3"])
    calibrate.write_records(records, calibrate.read_records(records)[1], fp_all="0" * 64)
    dest = tmp_path / "serving_gate.json"
    r = CliRunner().invoke(bench_app, ["package-gate", str(gate), "--records", str(records),
                                       "--lme-file", str(lme_file), "--manifest", str(manifest),
                                       "--out", str(dest)])  # fmt: skip
    assert r.exit_code == 1 and "fp_all" in r.output and not dest.exists()


def test_the_default_destination_is_the_package_location(tmp_path, monkeypatch):
    monkeypatch.setattr(shipped, "GATE_PATH", tmp_path / "pkg" / "serving_gate.json")
    lme_file, manifest, records, gate = _lme_world(tmp_path, ["a1", "a2", "a3"])
    r = CliRunner().invoke(
        bench_app,
        [
            "package-gate",
            str(gate),
            "--records",
            str(records),
            "--lme-file",
            str(lme_file),
            "--manifest",
            str(manifest),
        ],
    )
    assert r.exit_code == 0, r.output
    assert shipped.load() is not None


@pytest.mark.release_defaults  # the real default config, not conftest's Round 8 one (MEM2-274)
def test_a_committed_shipped_gate_matches_the_default_config():
    """Release check (ADR 0017): once a gate is committed, its fingerprint is the default config's.
    Absent until the host step (setup-gate runbook) has run."""
    loaded = shipped.load(ROOT / "src" / "mem" / "shipped" / "serving_gate.json")
    if loaded is None:
        pytest.skip(
            "no gate committed yet: the host step is in deliverables/runbooks/setup-gate.md"
        )
    from mem.config import default_config

    assert loaded.gate.fingerprint_hash == default_config().fp_all()


# ---- the packager without the frozen manifest (MEM2-277) --------------------------------------
# `bench/datasets/lme_split.tsv` is generated (`bench build-lme`) and untracked: on the host the
# packager refused with "No such file". It now derives the split from the pinned file by the same rule.


def _lme_questions(tmp_path):
    """Ten questions in one stratum: the rule puts 3 in tune and 7 in test (sha1 order)."""
    qs = [{"question_id": f"q{i:02d}", "question_type": "multi-session"} for i in range(10)]
    lme_file = tmp_path / "lme.json"
    lme_file.write_text(json.dumps(qs))
    rows = lme.build_manifest(qs)
    tune = [r.question_id for r in rows if r.split == "tune"]
    test = [r.question_id for r in rows if r.split == "test"]
    assert len(tune) == 3 and len(test) == 7
    return lme_file, tune, test


def _package_without_manifest(tmp_path, fit_ids, lme_file):
    pools = [calibrate.QueryPool(i, True, [calibrate.PoolItem("c1", 1.0, 1)]) for i in fit_ids]
    records = tmp_path / "tune_records.jsonl"
    calibrate.write_records(records, pools, fp_all=STUB.fp_all())
    gate = tmp_path / "gate.json"
    gate.write_text(gate_json(_gate(tune={"queries": len(fit_ids)})) + "\n")
    dest = tmp_path / "out" / "serving_gate.json"
    absent = tmp_path / "absent" / "lme_split.tsv"
    args = ["package-gate", str(gate), "--records", str(records), "--lme-file", str(lme_file),
            "--manifest", str(absent), "--out", str(dest)]  # fmt: skip
    return CliRunner().invoke(bench_app, args), dest


def test_the_packager_derives_the_split_from_the_lme_file_when_the_manifest_is_missing(tmp_path):
    lme_file, tune, _test = _lme_questions(tmp_path)
    r, dest = _package_without_manifest(tmp_path, tune, lme_file)
    assert r.exit_code == 0, r.output
    assert "split: derived from" in r.output and "lme.json" in r.output
    assert shipped.load(dest).gate.tune["queries"] == 3


def test_the_derived_split_still_refuses_a_fit_on_a_test_question(tmp_path):
    lme_file, tune, test = _lme_questions(tmp_path)
    r, dest = _package_without_manifest(tmp_path, [*tune, test[0]], lme_file)
    assert r.exit_code == 1, r.output
    assert test[0] in r.output and "test" in r.output
    assert not dest.exists()


def test_the_packager_names_the_frozen_manifest_when_it_uses_it(tmp_path):
    r, _dest = _package(tmp_path, ["a1", "a2", "a3"])
    assert r.exit_code == 0, r.output
    assert "split: " in r.output and "lme_split.tsv" in r.output
