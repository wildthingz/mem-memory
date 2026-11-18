"""The systemd unit and the host runbooks: uv from PATH, the models extra on every uv call."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RUNBOOKS = sorted((ROOT / "deliverables" / "runbooks").glob("*.md"))


def test_unit_finds_uv_on_path():
    unit = (ROOT / "deploy" / "mem.service").read_text()
    (exec_start,) = re.findall(r"^ExecStart=(.*)$", unit, flags=re.MULTILINE)
    assert exec_start.startswith("/usr/bin/env uv run --extra models mem serve ")
    (path,) = re.findall(r"^Environment=PATH=(.*)$", unit, flags=re.MULTILINE)
    assert {"%h/.local/bin", "/usr/bin"} <= set(path.split(":"))


def test_runbooks_never_drop_the_models_extra():
    if not RUNBOOKS:
        pytest.skip("deliverables/ is not part of this tree")
    assert RUNBOOKS
    for book in RUNBOOKS:
        text = book.read_text()
        commands = "\n".join(
            re.findall(r"^```bash\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL)
        )
        assert not re.search(r"/uv (?:sync|run)\b", commands), f"{book.name}: uv by path"
        for call in re.findall(r"\buv (?:sync|run)\b.*", commands):
            assert "--extra models" in call, f"{book.name}: {call}"
