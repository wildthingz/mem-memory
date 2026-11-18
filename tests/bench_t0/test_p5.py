"""B-E5 / B-C6 runners against a localhost `mem serve` (stand-in models), the P5 gate rows, the CLI."""

import asyncio
import copy
import json
import socket
import threading
import time
from datetime import UTC, datetime

import pytest
import uvicorn
from typer.testing import CliRunner

from bench import c6, e5, gates
from bench.cli import app
from bench.suites import b_p5
from mem.serve import MemServer, build_app, open_inprocess
from mem.testing import HashEmbedder

SPEC_P5 = {  # benchmarks-v2 §2 B-E5 gate, §4 P5 row: nothing else may gate
    ("B-E5", "parity_id_mismatches"): ("==", 0), ("B-E5", "parity_max_p_diff"): ("<=", 1e-6),
    ("B-E5", "cross_store_ids"): ("==", 0), ("B-E5", "added_p95_ms"): ("<=", 300),
    ("B-E5", "reconnect_failures"): ("==", 0), ("B-E5", "smoke_failures"): ("==", 0),
    ("B-C6", "overhead_ok"): (">=", 1),
}  # fmt: skip

QUERIES = [
    {"qid": f"q{i}", "query": q}
    for i, q in enumerate(
        ["espresso", "release day", "Atlas deadline", "coffee order", "unknown topic zzz"]
    )
]
OPS = [{"store": s, "tool": "remember",
        "args": {"text": f"Op {i} says the {s} store keeps item {i}.", "type": "fact",
                 "about": [f"new:topic:Item{i}"]}}
       for i, s in enumerate(["personal", "work"] * 3)]  # fmt: skip
SEED = [("personal", "Alice prefers espresso over filter coffee."),
        ("personal", "Atlas release day is the first Friday."),
        ("work", "Atlas deadline moved to November for the work team."),
        ("work", "The coffee order for the office is on Tuesday.")]  # fmt: skip


def _clock():
    return datetime(2026, 10, 22, 9, 0, tzinfo=UTC)


class Server:
    """uvicorn on a fixed port in its own thread; `restart()` stops and starts it again (new
    process-like state: MCP sessions are gone, stores persist on disk)."""

    def __init__(self, tmp, tag):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.tmp, self.tag = tmp, tag
        self.url = f"http://127.0.0.1:{self.port}"
        self.uv = self.thread = None

    def start(self):
        uv: list[uvicorn.Server] = []

        def serve():
            servers: dict[str, MemServer] = {}
            for name in e5.STORES:
                servers[name] = open_inprocess(self.tmp / f"{self.tag}-{name}.db", scope=name, tz="UTC",
                                               embedder=HashEmbedder(256), host="test", clock=_clock)  # fmt: skip
            cfg = uvicorn.Config(
                build_app(servers), host="127.0.0.1", port=self.port, log_level="error"
            )
            uv.append(uvicorn.Server(cfg))
            uv[0].run()
            for s in servers.values():
                s.close()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        for _ in range(100):
            if uv and uv[0].started:
                break
            time.sleep(0.05)
        assert uv and uv[0].started
        self.uv = uv[0]

    def stop(self):
        self.uv.should_exit = True
        self.thread.join(10)

    def restart(self):
        self.stop()
        self.start()


async def _seed(url):
    for store in e5.STORES:
        async with e5._mcp_client(url, store, "seed") as c:
            for s, text in SEED:
                if s == store:
                    r = await c.call_tool(
                        "remember",
                        {
                            "text": text,
                            "type": "fact",
                            "about": [f"new:topic:T{abs(hash(text)) % 999}"],
                        },
                    )
                    assert not r.is_error, r.content


@pytest.fixture
def servers(tmp_path):
    live, scratch = Server(tmp_path, "live"), Server(tmp_path, "scratch")
    live.start()
    scratch.start()
    asyncio.run(_seed(live.url))
    yield live, scratch
    live.stop()
    scratch.stop()


def _run(live, scratch, label, role, host="bench-home"):
    return e5.run_client(live.url, label=label, role=role, host_label=host, queries=QUERIES,
                         ops=OPS, scratch=scratch.url, runs=3, transport="http")  # fmt: skip


def _write(tmp, name, rec):
    p = tmp / name
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def test_p5_rows_are_exactly_the_spec_thresholds():
    rows = {(g.id, g.metric): g for g in gates.load_gates()["P5"]}
    assert set(rows) == set(SPEC_P5) | {("B-C6", "session_tokens")}
    for key, (op, thr) in SPEC_P5.items():
        assert (rows[key].op, rows[key].threshold) == (op, thr), key
    assert rows[("B-C6", "session_tokens")].op == "record"
    assert all(g.frozen_at is None and g.split == "fixture" for g in rows.values())


def test_with_nothing_supplied_no_gate_passes(tmp_path):
    out = gates.run_phase("P5", lambda *a, **k: None, results_dir=tmp_path, options={})
    assert [o.status for o in out] == ["UNAVAILABLE", "UNAVAILABLE"]


def test_client_record_against_localhost_server(servers):
    live, scratch = servers
    rec = _run(live, scratch, "home-local", "local")
    for s in e5.STORES:
        leg = rec["stores"][s]
        assert len(leg["latencies_ms"]) == 3 * len(QUERIES) and not leg["unstable_qids"]
        assert any(r["ids"] for r in leg["results"].values())  # the seeded memories come back
    assert len(rec["writes"]) == len(OPS) and all(
        w["structured"]["status"] == "saved" for w in rec["writes"]
    )
    assert rec["cross_store_ids"] == 0 and rec["smoke_failures"] == 0, rec


def test_parity_added_p95_and_gate_pass_then_each_break_fails(servers, tmp_path):
    live, scratch = servers
    local = _run(live, scratch, "home-local", "local")
    # a second client over the same server stands in for the work machine; the scratch server is reset
    scratch.stop()
    for f in tmp_path.glob("scratch-*"):
        f.unlink()
    scratch.start()
    remote = _run(live, scratch, "work-tailscale", "remote", host="bench-work")
    cmp = e5.compare_runs(local, remote)
    assert cmp["id_mismatches"] == 0 and cmp["max_p_diff"] <= 1e-6 and not cmp["write_diffs"], cmp
    recon = _write(tmp_path, "rc.json", {"kind": "e5-reconnect", "label": "w", "ok": True,
                                         "retries": 1, "detail": ""})  # fmt: skip
    c6p = _write(tmp_path, "c6.json", c6.measure(_fake_counter(1000)))
    paths = [_write(tmp_path, "a.json", local), _write(tmp_path, "b.json", remote)]

    def gate(paths, recons=(recon,), c6run=c6p):
        setup = b_p5.P5Setup(paths, list(recons), c6run)
        return {o.id: o for o in gates.run_phase("P5", lambda *a, **k: None, results_dir=None,
                                                 options={b_p5.OPT: setup})}  # fmt: skip

    out = gate(paths)
    assert out["B-E5"].status == "PASS" and out["B-C6"].status == "PASS", (
        out["B-E5"].detail,
        out["B-C6"].detail,
    )
    # red: one rank swapped on the remote machine
    bad = copy.deepcopy(remote)
    q = next(k for k, v in bad["stores"]["personal"]["results"].items() if len(v["ids"]) > 1)
    ids = bad["stores"]["personal"]["results"][q]["ids"]
    ids[0], ids[1] = ids[1], ids[0]
    assert gate([paths[0], _write(tmp_path, "c.json", bad)])["B-E5"].status == "FAIL"
    # red: p differs by 1e-5
    bad = copy.deepcopy(remote)
    p = bad["stores"]["personal"]["results"][q]["p"]
    p[0] = p[0] + 1e-5
    assert gate([paths[0], _write(tmp_path, "d.json", bad)])["B-E5"].status == "FAIL"
    # red: a cross-store id
    bad = copy.deepcopy(remote)
    bad["cross_store_ids"] = 1
    assert gate([paths[0], _write(tmp_path, "e.json", bad)])["B-E5"].status == "FAIL"
    # red: +400 ms
    bad = copy.deepcopy(remote)
    bad["stores"]["personal"]["latencies_ms"] = [
        v + 400 for v in bad["stores"]["personal"]["latencies_ms"]
    ]
    bad["stores"]["work"]["latencies_ms"] = [v + 400 for v in bad["stores"]["work"]["latencies_ms"]]
    assert gate([paths[0], _write(tmp_path, "f.json", bad)])["B-E5"].status == "FAIL"
    # red: write results differ; reconnect needed 2 retries; no reconnect record; no remote
    bad = copy.deepcopy(remote)
    bad["writes"][0]["structured"]["id"] = "m999"
    assert gate([paths[0], _write(tmp_path, "g.json", bad)])["B-E5"].status == "FAIL"
    slow = _write(tmp_path, "rc2.json", {"kind": "e5-reconnect", "label": "w", "ok": True,
                                         "retries": 2, "detail": ""})  # fmt: skip
    assert gate(paths, (slow,))["B-E5"].status == "FAIL"
    assert gate(paths, ())["B-E5"].status == "FAIL"  # reconnect_failures not produced
    assert gate(paths[:1])["B-E5"].status == "FAIL"


def _fake_counter(overhead):
    def count(*, messages, tools=None, system=None):
        return 10 + (overhead * 3 // 5 if tools else 0) + (overhead * 2 // 5 if system else 0)

    return count


def test_cross_store_leak_is_detected(servers, tmp_path, monkeypatch):
    """Red for the isolation check: make 'work' return the personal store's rows."""
    _live, scratch = servers
    real = e5._mcp_client

    def mount_work_on_personal(base, store, host_label):
        return real(base, "personal" if store == "work" else store, host_label)

    monkeypatch.setattr(e5, "_mcp_client", mount_work_on_personal)
    out = asyncio.run(e5._isolation_and_smoke(scratch.url, "x"))
    assert out["cross_store_ids"] > 0 or out["smoke_failures"] > 0, out


def test_reconnect_after_restart_needs_one_retry(servers):
    live, _ = servers
    rec = e5.run_reconnect(live.url, label="t", host_label="t", restart_cmd=None,
                           ask=lambda _msg: live.restart(), wait_s=30)  # fmt: skip
    assert rec["before_ok"] and rec["ok"] and rec["retries"] in (0, 1), rec


def test_reconnect_fails_when_server_stays_down(servers):
    live, _ = servers
    rec = e5.run_reconnect(live.url, label="t", host_label="t", restart_cmd=None,
                           ask=lambda _msg: live.stop(), wait_s=1)  # fmt: skip
    live.start()  # for the fixture teardown
    assert not rec["ok"] and "did not come back" in rec["detail"]


def _verdict(rec, tmp):
    p = tmp / "c6.json"
    p.write_text(json.dumps(rec))
    return b_p5.run_b_c6(_ctx(b_p5.P5Setup([], [], p))).metrics.get("overhead_ok")


def test_c6_budget_estimate_and_r5():
    ok = c6.measure(_fake_counter(2000))
    assert ok["session_tokens"] <= 2500 and ok["tools_n"] == 12 and ok["method"] == "count_tokens"
    big = c6.measure(_fake_counter(3000))
    assert big["session_tokens"] > 2500
    r5 = c6.measure(_fake_counter(3000), r5_applied=True)
    assert r5["session_tokens"] > 2500 and r5["r5_applied"]


def test_c6_gate_verdicts(tmp_path):
    ok, big = c6.measure(_fake_counter(2000)), c6.measure(_fake_counter(3000))
    r5 = c6.measure(_fake_counter(3000), tools=_r5_tools(), r5_applied=True)
    assert (
        _verdict(ok, tmp_path) == 1 and _verdict(big, tmp_path) == 0 and _verdict(r5, tmp_path) == 1
    )
    est = c6.measure(_fake_counter(100), method="est")
    assert _verdict(est, tmp_path) is None  # an estimate is not a count_tokens measurement


def _r5_tools():
    """The contract tools reshaped as R5 would advertise them: each takes a `store` param."""
    from mem import contracts

    tools = copy.deepcopy(contracts.mcp_tools()["tools"])
    for t in tools:
        t["inputSchema"].setdefault("properties", {})["store"] = {"enum": list(e5.STORES)}
    return tools


def test_c6_r5_is_measured_from_the_tool_shape(tmp_path):
    """MEM2-41 F2: with R5 tools the record counts one server and 6 tools; the flag alone counts
    what is really there (two servers, 12 tools) and fails; a forged record fails too."""
    r5 = c6.measure(_fake_counter(3000), tools=_r5_tools(), r5_applied=True)
    assert (r5["servers"], r5["tools_n"], r5["store_param"]) == (["mem"], 6, True)
    assert _verdict(r5, tmp_path) == 1
    flag_only = c6.measure(_fake_counter(3000), r5_applied=True)
    assert (len(flag_only["servers"]), flag_only["tools_n"], flag_only["store_param"]) == (
        2,
        12,
        False,
    )
    assert _verdict(flag_only, tmp_path) == 0
    forged = {**flag_only, "servers": ["mem"], "tools_n": 6}  # one server, but no `store` param
    assert _verdict(forged, tmp_path) == 0
    forged = {**r5, "servers": list(c6.SERVERS), "tools_n": 12}  # store param, two servers
    assert _verdict(forged, tmp_path) == 0
    # R5 tools without the flag are still counted as the two servers Claude Code has configured
    plain = c6.measure(_fake_counter(1000), tools=_r5_tools())
    assert len(plain["servers"]) == 2 and _verdict(plain, tmp_path) == 1


def test_c6_fetch_tools_from_a_live_endpoint(servers):
    live, _ = servers
    tools = c6.fetch_tools(f"{live.url}/personal/mcp")
    assert len(tools) == 6 and all("inputSchema" in t for t in tools)
    assert not c6.has_store_param(tools)  # today's per-store servers: no R5 shape
    assert c6.measure(_fake_counter(3000), tools=tools, r5_applied=True)["tools_n"] == 12


def _home_gate(tmp_path, recs, recons=(), home_only=True, c6run=None):
    paths = [_write(tmp_path, f"r{i}.json", r) for i, r in enumerate(recs)]
    c6run = c6run or _write(tmp_path, "c6.json", c6.measure(_fake_counter(1000)))
    setup = b_p5.P5Setup(paths, list(recons), c6run, home_only)
    return {o.id: o for o in gates.run_phase("P5", lambda *a, **k: None, results_dir=tmp_path / "res",
                                             options={b_p5.OPT: setup})}  # fmt: skip


def test_home_only_gate_defers_cross_machine_rows(servers, tmp_path):
    """MEM2-41 F9: without a work run, --home-only reports the home rows and DEFERS the others;
    a broken home record still FAILs; without --home-only a missing remote still FAILs."""
    live, scratch = servers
    local = _run(live, scratch, "home-local", "local", host="bench-home")
    out = _home_gate(tmp_path, [local])
    assert out["B-E5"].status == "DEFERRED" and not out["B-E5"].passed, out["B-E5"].detail
    assert out["B-C6"].status == "PASS"
    d = out["B-E5"].detail
    assert "cross_store_ids=0 (need == 0)" in d and "smoke_failures=0 (need == 0)" in d, d
    for m in (*b_p5.CROSS_MACHINE, "reconnect_failures"):
        assert f"DEFERRED {m}" in d and f"FAILED DEFERRED {m}" not in d, d
    rec = json.loads(out["B-E5"].run_file.read_text())
    rows = {g["metric"]: g for g in rec["gates"]}
    assert (
        rows["parity_id_mismatches"]["passed"] is None and rows["parity_id_mismatches"]["deferred"]
    )
    assert rows["cross_store_ids"]["passed"] is True
    # red: a home-side row fails -> FAIL, not DEFERRED
    bad = copy.deepcopy(local)
    bad["cross_store_ids"] = 1
    assert _home_gate(tmp_path, [bad])["B-E5"].status == "FAIL"
    bad = copy.deepcopy(local)
    bad["runs"] = 1  # a failed check is not hidden behind a deferral
    assert _home_gate(tmp_path, [bad])["B-E5"].status == "FAIL"
    assert _home_gate(tmp_path, [])["B-E5"].status == "FAIL"  # no local record at all
    assert _home_gate(tmp_path, [local], home_only=False)["B-E5"].status == "FAIL"
    # an optional remote record (the tailnet path from the desktop) is evaluated, not deferred
    remote = copy.deepcopy(local)
    remote.update(role="remote", label="home-tailscale")
    out = _home_gate(tmp_path, [local, remote])
    assert out["B-E5"].status == "DEFERRED" and "parity_id_mismatches=0" in out["B-E5"].detail
    assert "DEFERRED reconnect_failures" in out["B-E5"].detail
    bad = copy.deepcopy(remote)
    bad["cross_store_ids"] = 2
    assert _home_gate(tmp_path, [local, bad])["B-E5"].status == "FAIL"


def test_cli_home_only_gate_exit_code(servers, tmp_path):
    live, scratch = servers
    local = _write(
        tmp_path, "local.json", _run(live, scratch, "home-local", "local", host="bench-home")
    )
    c6p = _write(tmp_path, "c6.json", c6.measure(_fake_counter(1000)))
    args = ["gate", "P5", "--results-dir", str(tmp_path / "res"), "--e5-run", str(local),
            "--c6-run", str(c6p)]  # fmt: skip
    r = CliRunner().invoke(app, [*args, "--home-only"])
    assert r.exit_code == 0, r.output
    assert "DEFERRED    B-E5" in r.output and "PASS        B-C6" in r.output, r.output
    assert "1/2 gates pass, 1 deferred (not a pass) for P5" in r.output, r.output
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1 and "FAIL        B-E5" in r.output, r.output


def _ctx(setup):
    from pathlib import Path

    from bench.suites.base import Context

    return Context(Path("."), lambda *a, **k: None, None, {b_p5.OPT: setup})


def test_cli_gate_p5_and_commands(servers, tmp_path):
    live, scratch = servers
    runner = CliRunner()
    q, o = tmp_path / "q.jsonl", tmp_path / "o.jsonl"
    q.write_text("".join(json.dumps(x) + "\n" for x in QUERIES))
    o.write_text("".join(json.dumps(x) + "\n" for x in OPS))
    r = runner.invoke(app, ["e5-run", "--url", live.url, "--label", "home-local", "--role", "local",
                            "--queries", str(q), "--ops", str(o), "--scratch-url", scratch.url,
                            "--out", str(tmp_path / "local.json")])  # fmt: skip
    assert r.exit_code == 0 and "cross-store ids 0; smoke failures 0" in r.output, r.output
    assert json.loads((tmp_path / "local.json").read_text())["host_label"] == "bench-home-local"
    r = runner.invoke(app, ["e5-run", "--url", live.url, "--label", "home-local", "--role", "local",
                            "--queries", str(q), "--ops", str(o), "--scratch-url", scratch.url,
                            "--host-label", "home", "--out", str(tmp_path / "x.json")])  # fmt: skip
    assert r.exit_code == 2 and "bench-" in r.output and not (tmp_path / "x.json").exists(), (
        r.output
    )
    r = runner.invoke(app, ["e5-reconnect", "--url", live.url, "--label", "w",
                            "--host-label", "work-laptop", "--out", str(tmp_path / "rc.json")])  # fmt: skip
    assert r.exit_code == 2 and not (tmp_path / "rc.json").exists(), r.output
    r = runner.invoke(app, ["gate", "P5", "--results-dir", str(tmp_path / "res"),
                            "--e5-run", str(tmp_path / "local.json")])  # fmt: skip
    assert r.exit_code == 1 and "FAIL" in r.output and "UNAVAILABLE" in r.output, r.output
    r = runner.invoke(app, ["e5-ops", "--out", str(tmp_path / "ops.jsonl")])
    ops = e5.load_ops(tmp_path / "ops.jsonl")
    assert r.exit_code == 0 and len(ops) == 20 and {x["store"] for x in ops} == {"personal", "work"}


def test_c6_r5_flag_alone_cannot_pass_the_gate(tmp_path):
    """MEM2-41 F2. `--r5-applied` is a self-attested flag: the record still counts 12 tools under
    two servers (R5 = ONE server with a `store` param), yet overhead_ok becomes 1 at 3,000 tokens.
    A gate must not pass on a claim the record itself contradicts."""
    rec = c6.measure(_fake_counter(3000), r5_applied=True)
    assert rec["tools_n"] == 12 and len(rec["servers"]) == 2  # the record says: R5 not applied
    assert _verdict(rec, tmp_path) != 1
