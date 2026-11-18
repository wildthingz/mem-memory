"""MEM2-251: regression tests for the rv2-review (MEM2-222) findings F19, F22 and F23."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mem.cli import app

# ---- F19: a non-catalog model revision is a [FAIL] line, not a traceback ------------------------


@pytest.fixture
def off_pin_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text('embedder:\n  revision: "' + "0" * 40 + '"\n', encoding="utf-8")
    return p


@pytest.mark.parametrize(
    "argv",
    [["doctor"], ["serve", "--allow-cpu", "--store", "personal=STORE"]],
    ids=["doctor", "serve"],
)
def test_f19_an_off_pin_revision_is_a_fail_line(off_pin_config, tmp_path, argv):
    argv = [a.replace("STORE", str(tmp_path / "p.db")) for a in argv]
    r = CliRunner().invoke(app, [*argv, "--config", str(off_pin_config)])
    assert r.exit_code == 1, r.output
    assert not isinstance(r.exception, ValueError)
    assert "[FAIL]" in r.output and "not a pinned catalog entry" in r.output
