"""`mem serve` command line: store specs, refusal to start without models, stdio transport."""

import asyncio
import os
import re
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from typer.testing import CliRunner

from mem.cli import app
from mem.serve import parse_stores
from tests.helpers import gate_stub_store


def test_parse_stores_reads_name_equals_path(tmp_path):
    got = parse_stores([f"personal={tmp_path}/p.db", f"work={tmp_path}/w.db"])
    assert {k: str(v) for k, v in got.items()} == {
        "personal": f"{tmp_path}/p.db",
        "work": f"{tmp_path}/w.db",
    }


@pytest.mark.parametrize("bad", ["personal", "other=/x.db", "personal=/a.db,work=/b.db"])
def test_parse_stores_rejects_bad_specs(bad):
    with pytest.raises(ValueError, match="--store"):
        parse_stores([bad])


def test_parse_stores_rejects_duplicates():
    with pytest.raises(ValueError, match="twice"):
        parse_stores(["personal=/a.db", "personal=/b.db"])


def test_serve_refuses_to_start_when_pinned_models_are_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    r = CliRunner().invoke(
        app, ["serve", "--allow-cpu", "--http", "--store", f"personal={tmp_path}/p.db"]
    )
    assert r.exit_code == 1
    assert "mem doctor --fetch" in r.output
    assert not (tmp_path / "p.db").exists()  # nothing was created


def test_stdio_serves_one_store_with_a_per_process_session(tmp_path, round8_config):
    gate_stub_store(tmp_path / "p.db")  # P2: serve needs a gate that matches the config
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mem.cli", "serve", "--allow-cpu", "--embedder", "hash", "--host-label", "ci",
              "--store", f"personal={tmp_path}/p.db", "--config", str(round8_config)],
    )  # fmt: skip

    async def scenario():
        async with Client(params) as c:
            saved = await c.call_tool("remember", {"text": "Bosco is a dog.", "type": "fact",
                                                   "about": ["new:thing:Bosco"]})  # fmt: skip
            seen = await c.call_tool("recall", {"query": "dog Bosco"})
            return saved.structured_content["status"], [
                r["id"] for r in seen.structured_content["rows"]
            ]

    assert asyncio.run(scenario()) == ("saved", ["m1"])


def _runbook_stdio_args() -> list[str]:
    """The `mem serve` arguments of the stdio registration line in the P1 gate runbook."""
    runbook = Path(__file__).parents[1] / "deliverables" / "runbooks" / "p1-gate.md"
    if not runbook.exists():
        pytest.skip("deliverables/ is not part of this tree")
    (line,) = [ln.strip() for ln in runbook.read_text().splitlines()
               if ln.strip().startswith("claude mcp add mem-stdio -- ")]  # fmt: skip
    argv = shlex.split(line)
    i = argv.index("serve")
    assert argv[i - 1] == "mem"
    return argv[i + 1 :]


def test_runbook_stdio_command_starts(tmp_path, round8_config):
    args = _runbook_stdio_args()
    stores = [a for a in args if "=" in a and a.split("=")[0] in ("personal", "work")]
    assert len(stores) == 1 and args[args.index(stores[0]) - 1] == "--store"
    name = stores[0].split("=")[0]
    args[args.index(stores[0])] = f"{name}={tmp_path}/s.db"  # keep the shape, not the path
    gate_stub_store(tmp_path / "s.db", scope=name)
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "mem.cli",
            "serve",
            "--allow-cpu",
            *args,
            "--embedder",
            "hash",
            "--config",
            str(round8_config),
        ],
    )

    async def scenario():
        async with Client(params) as c:
            return sorted(t.name for t in (await c.list_tools()).tools)

    assert "recall" in asyncio.run(scenario())


def test_stdio_without_one_store_says_what_to_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("mem.cli.DATA_HOME", tmp_path)
    r = CliRunner().invoke(app, ["serve", "--allow-cpu", "--embedder", "hash"])
    assert r.exit_code == 2
    text = " ".join(re.sub("[│╭╮╰╯─]", " ", r.output).split())  # unwrap the rich error box
    assert "pass --store personal=<path>" in text
    assert not list(tmp_path.iterdir())  # refused before opening anything


def _stored_valid_from(tmp_path: Path, extra: list[str], env: dict[str, str], cfg: Path) -> str:
    """Run `mem serve` over stdio, save a memory with a date-only as_of, read back valid_from."""
    gate_stub_store(tmp_path / "p.db")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mem.cli", "serve", "--allow-cpu", "--embedder", "hash", "--store",
              f"personal={tmp_path}/p.db", "--config", str(cfg), *extra],
        env={**os.environ, **env},
    )  # fmt: skip

    async def scenario():
        async with Client(params) as c:
            await c.call_tool("remember", {"text": "Bosco is a dog.", "type": "fact",
                                           "about": ["new:thing:Bosco"], "as_of": "2026-03-01"})  # fmt: skip

    asyncio.run(scenario())
    with sqlite3.connect(tmp_path / "p.db") as conn:
        (got,) = conn.execute("SELECT valid_from FROM nodes WHERE kind = 'memory'").fetchone()
    return got


def test_date_only_inputs_default_to_the_system_zone(tmp_path, round8_config):
    # 2026-03-01 00:00 in Los Angeles is PST (UTC-8), before the March DST switch.
    got = _stored_valid_from(tmp_path, [], {"TZ": "America/Los_Angeles"}, round8_config)
    assert got == "2026-03-01T08:00:00Z"


def test_tz_option_wins_over_the_system_zone(tmp_path, round8_config):
    # 2026-03-01 00:00 in Amsterdam is CET (UTC+1).
    got = _stored_valid_from(tmp_path, ["--tz", "Europe/Amsterdam"], {"TZ": "America/Los_Angeles"},
                             round8_config)  # fmt: skip
    assert got == "2026-02-28T23:00:00Z"


def test_serve_refuses_cpu_without_allow_cpu(tmp_path, monkeypatch):
    gate_stub_store(tmp_path / "p.db")
    for dev in ("cpu", None):  # None = config default cuda; the stub test host has no GPU
        args = ["serve", "--http", "--embedder", "hash", "--store", f"personal={tmp_path}/p.db"]
        args += ["--device", dev] if dev else []
        if dev is None:
            monkeypatch.setattr("mem.doctor.cpu_reason", lambda device: "torch sees no CUDA")
        r = CliRunner().invoke(app, args)
        assert r.exit_code == 1 and "--allow-cpu" in r.output


def test_serve_starts_with_allow_cpu(tmp_path, monkeypatch):
    gate_stub_store(tmp_path / "p.db")
    seen = {}
    monkeypatch.setattr("mem.serve.run_http", lambda *a, **k: seen.setdefault("ok", True))
    r = CliRunner().invoke(
        app,
        [
            "serve",
            "--allow-cpu",
            "--http",
            "--embedder",
            "hash",
            "--device",
            "cpu",
            "--store",
            f"personal={tmp_path}/p.db",
        ],
    )
    assert r.exit_code == 0 and seen["ok"]
