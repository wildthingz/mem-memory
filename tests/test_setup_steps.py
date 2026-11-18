"""The setup interface (MEM2-232): byte-exact config backups, one runner."""

from __future__ import annotations

import pytest

from mem import install
from mem import setup_steps as ss
from mem.common import MemError


def test_restore_puts_back_the_exact_bytes(tmp_path):
    f = tmp_path / "settings.json"
    original = b'{"a":1}\r\n\xff\x00 odd bytes'
    f.write_bytes(original)
    f.chmod(0o640)
    bk = ss.Backups(tmp_path / "bk")
    bk.save(f)
    f.write_bytes(b"edited")
    f.chmod(0o600)
    bk.restore(f)
    assert f.read_bytes() == original
    assert f.stat().st_mode & 0o777 == 0o640


def test_restore_removes_a_file_that_did_not_exist_before(tmp_path):
    f = tmp_path / "new.yaml"
    bk = ss.Backups(tmp_path / "bk")
    bk.save(f)
    f.write_text("made by setup")
    bk.restore(f)
    assert not f.exists()


def test_first_backup_wins_so_a_second_apply_keeps_the_true_original(tmp_path):
    f = tmp_path / "c.json"
    f.write_text("original")
    bk = ss.Backups(tmp_path / "bk")
    bk.save(f)
    f.write_text("after setup")
    bk.save(f)  # a re-run must not replace the original with the edited file
    bk.restore(f)
    assert f.read_text() == "original"


def test_backups_survive_a_new_instance(tmp_path):
    f = tmp_path / "c.json"
    f.write_text("original")
    ss.Backups(tmp_path / "bk").save(f)
    f.write_text("changed")
    ss.Backups(tmp_path / "bk").restore(f)
    assert f.read_text() == "original"


def test_restore_without_a_backup_is_an_error(tmp_path):
    with pytest.raises(MemError, match="no backup"):
        ss.Backups(tmp_path / "bk").restore(tmp_path / "x")


def test_after_restore_the_backup_is_forgotten(tmp_path):
    f = tmp_path / "c"
    f.write_text("o")
    bk = ss.Backups(tmp_path / "bk")
    bk.save(f)
    bk.restore(f)
    assert not bk.has(f)


def test_the_runner_is_the_one_from_mem_install():
    assert ss.Runner is install.Runner
    assert ss.SystemRunner is install.SystemRunner
