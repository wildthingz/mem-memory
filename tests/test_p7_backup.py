"""P7 (B-H1): encrypted nightly backups (`VACUUM INTO` + AES-256-GCM) and the restore drill."""

import json
import os
import sqlite3
import stat
import subprocess

import pytest
from typer.testing import CliRunner

from bench.dump import canonical_dump
from mem import backup, doctor
from mem.cli import app

ANA = "new:person:Ana Silva"
SECRET = "Ana Silva keeps the spare key under the blue flowerpot."


def _fill(store):
    store.remember(text=SECRET, type="fact", about=[ANA], title="Spare key")
    store.remember(text="Ana Silva prefers async standups.", type="preference", about=[ANA])


@pytest.fixture
def key(tmp_path):
    path = tmp_path / "backup.key"
    backup.new_key(path)
    return backup.load_key(path)


def _vec_rows(db):
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    (table,) = conn.execute("SELECT value FROM meta WHERE key = 'vec_table'").fetchone()
    rows = conn.execute(f"SELECT chunk_id, embedding FROM {table} ORDER BY chunk_id").fetchall()
    conn.close()
    return rows


def test_restore_gives_zero_diff_dump_while_store_is_open(store, tmp_path, key):
    _fill(store)  # still open: the last writes sit in the WAL, not yet checkpointed
    db = tmp_path / "personal.db"
    out = backup.backup_store(db, tmp_path / "bk", key, name="personal")
    restored = backup.restore(out, key, tmp_path / "restored.db")
    assert canonical_dump(restored) == canonical_dump(db)
    assert b"Spare key" in canonical_dump(restored)


def test_restore_keeps_the_vectors(store, tmp_path, key):
    _fill(store)
    db = tmp_path / "personal.db"
    out = backup.backup_store(db, tmp_path / "bk", key, name="personal")
    restored = backup.restore(out, key, tmp_path / "restored.db")
    assert _vec_rows(restored) == _vec_rows(db)
    assert len(_vec_rows(db)) == 3  # 2 memories + the Ana entity


def test_backup_file_holds_no_plaintext(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    data = out.read_bytes()
    assert b"flowerpot" not in data and b"SQLite format 3" not in data


def test_backup_file_is_unreadable_as_a_database(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(f"file:{out}?mode=ro", uri=True).execute("SELECT * FROM nodes")


def test_restore_with_wrong_key_fails_and_writes_nothing(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    other = tmp_path / "other.key"
    backup.new_key(other)
    dest = tmp_path / "restored.db"
    with pytest.raises(backup.BackupError, match="wrong key or corrupted"):
        backup.restore(out, backup.load_key(other), dest)
    assert list(tmp_path.glob("restored.db*")) == []


def test_tampered_backup_is_rejected(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    data = bytearray(out.read_bytes())
    data[-40] ^= 1
    out.write_bytes(bytes(data))
    with pytest.raises(backup.BackupError, match="wrong key or corrupted"):
        backup.restore(out, key, tmp_path / "restored.db")


def test_truncated_backup_is_rejected(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    out.write_bytes(out.read_bytes()[:-20])
    with pytest.raises(backup.BackupError):
        backup.restore(out, key, tmp_path / "restored.db")


def test_restore_refuses_an_existing_destination(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    with pytest.raises(backup.BackupError, match="exists"):
        backup.restore(out, key, tmp_path / "personal.db")


def test_backup_leaves_no_plaintext_snapshot_behind(store, tmp_path, key):
    _fill(store)
    bk = tmp_path / "bk"
    out = backup.backup_store(tmp_path / "personal.db", bk, key, name="personal")
    assert sorted(p.name for p in bk.iterdir()) == [out.name]


def test_prune_keeps_the_newest_n_per_store(tmp_path):
    bk = tmp_path / "bk"
    bk.mkdir()
    for day in ("01", "02", "03", "04"):
        (bk / f"personal-202610{day}T030000Z.mbk").write_bytes(b"x")
    (bk / "work-20261001T030000Z.mbk").write_bytes(b"x")
    removed = backup.prune(bk, "personal", keep=2)
    assert [p.name for p in removed] == ["personal-20261001T030000Z.mbk",
                                         "personal-20261002T030000Z.mbk"]  # fmt: skip
    assert sorted(p.name for p in bk.iterdir()) == ["personal-20261003T030000Z.mbk",
        "personal-20261004T030000Z.mbk", "work-20261001T030000Z.mbk"]  # fmt: skip


def test_new_key_is_owner_only(tmp_path):
    path = tmp_path / "backup.key"
    backup.new_key(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(backup.load_key(path)) == 32


def test_new_key_refuses_to_overwrite(tmp_path):
    path = tmp_path / "backup.key"
    backup.new_key(path)
    before = path.read_bytes()
    with pytest.raises(backup.BackupError, match="exists"):
        backup.new_key(path)
    assert path.read_bytes() == before


def test_load_key_rejects_a_group_readable_file(tmp_path):
    path = tmp_path / "backup.key"
    backup.new_key(path)
    os.chmod(path, 0o640)
    with pytest.raises(backup.BackupError, match="chmod 600"):
        backup.load_key(path)


def test_cli_backup_then_restore_round_trip(store, tmp_path):
    _fill(store)
    keyf, bk, dest = tmp_path / "k", tmp_path / "bk", tmp_path / "restored.db"
    r = CliRunner().invoke(app, ["backup-key", "--out", str(keyf)])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(app, ["backup", "--store", f"personal={tmp_path / 'personal.db'}",
                                 "--dir", str(bk), "--key-file", str(keyf)])  # fmt: skip
    assert r.exit_code == 0, r.output
    (out,) = bk.glob("personal-*.mbk")
    assert out.name in r.output
    r = CliRunner().invoke(app, ["restore", str(out), "--key-file", str(keyf), "--to", str(dest)])
    assert r.exit_code == 0, r.output
    assert "integrity ok" in r.output
    assert canonical_dump(dest) == canonical_dump(tmp_path / "personal.db")


def test_cli_restore_with_wrong_key_exits_1(store, tmp_path):
    _fill(store)
    k1, k2, bk = tmp_path / "k1", tmp_path / "k2", tmp_path / "bk"
    for k in (k1, k2):
        assert CliRunner().invoke(app, ["backup-key", "--out", str(k)]).exit_code == 0
    CliRunner().invoke(app, ["backup", "--store", f"personal={tmp_path / 'personal.db'}",
                             "--dir", str(bk), "--key-file", str(k1)])  # fmt: skip
    (out,) = bk.glob("personal-*.mbk")
    r = CliRunner().invoke(app, ["restore", str(out), "--key-file", str(k2),
                                 "--to", str(tmp_path / "r.db")])  # fmt: skip
    assert r.exit_code == 1
    assert "wrong key or corrupted" in r.output


def _runner(outputs):
    def run(cmd):
        return outputs[cmd[0]]

    return run


def test_at_rest_check_ok_on_dm_crypt(tmp_path):
    run = _runner({"findmnt": "/dev/mapper/home ext4\n", "lsblk": "crypt\npart\ndisk\n"})
    c = doctor.check_at_rest(tmp_path, run=run)
    assert (c.status, c.detail) == ("ok", f"{tmp_path} on dm-crypt (/dev/mapper/home)")


def test_at_rest_check_ok_on_gocryptfs(tmp_path):
    c = doctor.check_at_rest(tmp_path, run=_runner({"findmnt": "/mnt/h/.c fuse.gocryptfs\n"}))
    assert c.status == "ok" and "gocryptfs" in c.detail


def test_at_rest_check_warns_on_a_plain_disk_until_required(tmp_path):
    run = _runner({"findmnt": "/dev/nvme0n1p2 ext4\n", "lsblk": "part\ndisk\n"})
    c = doctor.check_at_rest(tmp_path, run=run)
    assert c.status == "warn" and "NOT encrypted" in c.detail
    assert "deferred (owner's backlog); see p7-backup.md" in c.detail


def test_at_rest_check_fails_on_a_plain_disk_when_required(tmp_path):
    run = _runner({"findmnt": "/dev/nvme0n1p2 ext4\n", "lsblk": "part\ndisk\n"})
    c = doctor.check_at_rest(tmp_path, run=run, required=True)
    assert c.status == "fail" and "NOT encrypted" in c.detail
    assert "--allow-unencrypted" in c.detail


def test_at_rest_check_fails_when_it_cannot_tell(tmp_path):
    def run(cmd):
        raise subprocess.CalledProcessError(1, cmd)

    c = doctor.check_at_rest(tmp_path, run=run, required=True)
    assert c.status == "fail" and "cannot tell" in c.detail
    assert doctor.check_at_rest(tmp_path, run=run).status == "warn"


def test_at_rest_opt_out_turns_the_fail_into_a_warn(tmp_path):
    run = _runner({"findmnt": "/dev/nvme0n1p2 ext4\n", "lsblk": "part\ndisk\n"})
    c = doctor.check_at_rest(tmp_path, run=run, allow_unencrypted=True, required=True)
    assert c.status == "warn"
    assert "NOT encrypted" in c.detail and "opted out (--allow-unencrypted)" in c.detail


def test_at_rest_opt_out_leaves_an_encrypted_disk_ok(tmp_path):
    run = _runner({"findmnt": "/dev/mapper/home ext4\n", "lsblk": "crypt\npart\ndisk\n"})
    assert doctor.check_at_rest(tmp_path, run=run, allow_unencrypted=True).status == "ok"


def test_at_rest_check_follows_a_btrfs_subvolume_to_its_luks_device(tmp_path):
    seen = []

    def run(cmd):
        seen.append(cmd)
        return {"findmnt": "/dev/mapper/luks-1a2b[/@home] btrfs\n", "lsblk": "crypt\npart\n"}[
            cmd[0]
        ]

    assert doctor.check_at_rest(tmp_path, run=run).status == "ok"
    assert seen[-1][-1] == "/dev/mapper/luks-1a2b"


def _doctor_with_at_rest(monkeypatch, status, detail="x"):
    """`mem doctor` with every other row ok and the disk probe answering `status`."""
    ok = doctor.Check("other", "ok", "fine")
    monkeypatch.setattr(doctor, "check_sqlite_vec", lambda: ok)
    monkeypatch.setattr(doctor, "check_models", lambda cfg=None: [ok])
    monkeypatch.setattr(doctor, "check_device", lambda device="cuda": ok)
    monkeypatch.setattr(doctor, "check_fingerprint", lambda cfg=None: ok)
    monkeypatch.setattr(doctor, "check_gate", lambda stores, cfg: [ok])
    seen = []

    def at_rest(path, allow_unencrypted=False, required=False):
        seen.append((path, allow_unencrypted))
        if status == "fail" and allow_unencrypted:
            return doctor.Check("at-rest", "warn", f"{detail}; opted out")
        return doctor.Check("at-rest", status, detail)

    monkeypatch.setattr(doctor, "check_at_rest", at_rest)
    return seen


def test_doctor_reports_at_rest_for_the_stores_folder(tmp_path, monkeypatch):
    seen = _doctor_with_at_rest(monkeypatch, "ok")
    db = tmp_path / "s" / "personal.db"
    r = CliRunner().invoke(app, ["doctor", "--store", str(db)])
    assert "[ok  ] at-rest: x" in r.output and r.exit_code == 0
    assert seen == [(db.parent, False)]


def test_cli_doctor_passes_security_at_rest_required_from_the_config(tmp_path, monkeypatch):
    seen = []

    def at_rest(path, allow_unencrypted=False, required=False):
        seen.append(required)
        return doctor.Check("at-rest", "ok", "x")

    _doctor_with_at_rest(monkeypatch, "ok")
    monkeypatch.setattr(doctor, "check_at_rest", at_rest)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("security:\n  at_rest_required: true\n")
    CliRunner().invoke(app, ["doctor", "--store", str(tmp_path / "p.db"), "--config", str(cfg)])
    CliRunner().invoke(app, ["doctor", "--store", str(tmp_path / "p.db")])
    assert seen == [True, False]


def test_cli_doctor_exits_1_when_the_stores_are_not_encrypted(tmp_path, monkeypatch):
    _doctor_with_at_rest(monkeypatch, "fail", "NOT encrypted")
    r = CliRunner().invoke(app, ["doctor", "--store", str(tmp_path / "personal.db")])
    assert r.exit_code == 1
    assert "[FAIL] at-rest: NOT encrypted" in r.output


def test_cli_doctor_opt_out_exits_0_and_is_logged(tmp_path, monkeypatch):
    _doctor_with_at_rest(monkeypatch, "fail", "NOT encrypted")
    log = tmp_path / "state" / "doctor.jsonl"
    db = tmp_path / "personal.db"
    args = ["doctor", "--store", str(db), "--allow-unencrypted", "--opt-out-log", str(log)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "[warn] at-rest: NOT encrypted; opted out" in r.output
    assert f"opt-out logged to {log}" in r.output
    (line,) = log.read_text().splitlines()
    rec = json.loads(line)
    assert rec["event"] == "at-rest-opt-out" and rec["path"] == str(tmp_path)
    assert rec["detail"] == "NOT encrypted; opted out" and rec["ts"]
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_cli_doctor_opt_out_on_an_encrypted_disk_logs_nothing(tmp_path, monkeypatch):
    _doctor_with_at_rest(monkeypatch, "ok")
    log = tmp_path / "doctor.jsonl"
    args = ["doctor", "--store", str(tmp_path / "p.db"), "--allow-unencrypted",
            "--opt-out-log", str(log)]  # fmt: skip
    assert CliRunner().invoke(app, args).exit_code == 0
    assert not log.exists()


def test_bench_dump_writes_the_canonical_dump_and_vector_hash(store, tmp_path):
    from bench.cli import app as bench_app
    from bench.dump import vectors_sha256

    _fill(store)
    db, out = tmp_path / "personal.db", tmp_path / "d.jsonl"
    r = CliRunner().invoke(bench_app, ["dump", str(db), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.read_bytes() == canonical_dump(db)
    assert f"vectors {vectors_sha256(db)}" in r.output
    assert f": {len(canonical_dump(db).splitlines())} rows" in r.output


def test_vectors_sha256_changes_when_a_vector_goes(store, tmp_path):
    from bench.dump import vectors_sha256

    _fill(store)
    db = tmp_path / "personal.db"
    before = vectors_sha256(db)
    assert vectors_sha256(db) == before
    store.conn.execute(f"DELETE FROM {store.vec_table} WHERE chunk_id = 1")
    assert vectors_sha256(db) != before


def test_backup_cut_at_a_chunk_boundary_is_rejected(store, tmp_path, key, monkeypatch):
    monkeypatch.setattr(backup, "CHUNK", 4096)
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    data = out.read_bytes()
    with out.open("rb") as f:
        backup.read_header(f)
        offsets = []
        while prefix := f.read(4):
            offsets.append(f.tell() - 4)
            f.seek(int.from_bytes(prefix, "big"), 1)
    assert len(offsets) > 2
    out.write_bytes(data[: offsets[-1]])  # drop the whole final chunk
    with pytest.raises(backup.BackupError, match="wrong key or corrupted"):
        backup.restore(out, key, tmp_path / "restored.db")


def test_backup_files_and_folder_are_owner_only(store, tmp_path, key):
    _fill(store)
    bk = tmp_path / "bk"
    out = backup.backup_store(tmp_path / "personal.db", bk, key, name="personal")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert stat.S_IMODE(bk.stat().st_mode) == 0o700


def test_restored_store_is_owner_only(store, tmp_path, key):
    _fill(store)
    out = backup.backup_store(tmp_path / "personal.db", tmp_path / "bk", key, name="personal")
    restored = backup.restore(out, key, tmp_path / "restored.db")
    assert stat.S_IMODE(restored.stat().st_mode) == 0o600


def test_cli_backup_still_backs_up_the_other_store_when_one_fails(store, tmp_path):
    _fill(store)
    keyf, bk = tmp_path / "k", tmp_path / "bk"
    CliRunner().invoke(app, ["backup-key", "--out", str(keyf)])
    r = CliRunner().invoke(app, ["backup", "--store", f"personal={tmp_path / 'gone.db'}",
                                 "--store", f"work={tmp_path / 'personal.db'}",
                                 "--dir", str(bk), "--key-file", str(keyf)])  # fmt: skip
    assert r.exit_code == 1
    assert "no store at" in r.output
    assert len(list(bk.glob("work-*.mbk"))) == 1
