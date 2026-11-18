"""P5 commands of `bench` (benchmarks-v2 §4 P5 row); registered on the main app by `bench.cli`.

e5-queries      the 50 fixed PG-v1 queries (JSONL), written once and copied to both machines
e5-ops          the 20 fixed SYN write ops (JSONL)
e5-run          B-E5 client legs on this machine (run on each machine, same files)
e5-reconnect    B-E5: restart the server under an open session, one retry allowed
c6-run          B-C6: session overhead by count_tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from bench import c6, e5
from bench import p4_setup as setup_mod
from bench.gold import pg

UrlOpt = Annotated[str, typer.Option(help="Server base URL, e.g. http://127.0.0.1:8765")]
HostOpt = Annotated[
    str,
    typer.Option(
        help="X-Mem-Host; must start with `bench-` (default bench-<label>) so P6 counts exclude it."
    ),
]


def _host_label(host_label: str, label: str) -> str:
    host = host_label or e5.bench_host_label(label)
    if not host.startswith(e5.BENCH_HOST_PREFIX):
        typer.echo(f"--host-label {host!r}: bench runs write events into the live stores; use a "
                   f"`{e5.BENCH_HOST_PREFIX}` label (e.g. bench-home, bench-work) so daily-use "
                   f"counts can exclude them ({e5.LIVE_EVENTS_FILTER})")  # fmt: skip
        raise typer.Exit(2)
    return host


def register(app: typer.Typer) -> None:
    @app.command("e5-queries")
    def e5_queries(
        out: Annotated[Path, typer.Option(help="e.g. data/p5/queries.jsonl")],
        gold_dir: Annotated[Path, typer.Option(help="Frozen PG-v1.")] = setup_mod.GOLD_DIR,
        n: Annotated[int, typer.Option(help="Spec: 50.")] = 50,
    ) -> None:
        """The first n PG-v1 questions by sha1(qid), tune and test alike (a fixed, split-neutral
        choice). PG-v1 holds no work-store content (decision 0013), so this file is safe to copy."""
        qs = sorted(pg.load(gold_dir).questions, key=lambda q: pg.sha1_key(q.qid))[:n]
        if len(qs) < n:
            typer.echo(f"PG-v1 has only {len(qs)} questions")
            raise typer.Exit(2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"qid": q.qid, "query": q.question}, ensure_ascii=False)
                               + "\n" for q in qs), encoding="utf-8")  # fmt: skip
        typer.echo(f"wrote {len(qs)} queries to {out}")

    @app.command("e5-ops")
    def e5_ops(
        out: Annotated[Path, typer.Option(help="e.g. data/p5/ops.jsonl")],
        seed: Annotated[int, typer.Option(help="SYN seed (held-out 101).")] = 101,
        n: Annotated[int, typer.Option(help="Spec: 20.")] = 20,
    ) -> None:
        """n SYN (size S, seed `seed`) `remember` ops, each introducing new entities, alternating
        personal/work (so the two stores differ and every op applies to an empty-of-it scratch copy)."""
        from bench.synth.generator import generate

        syn, ops, seen = generate(seed, "S"), [], set()
        ents = {e["key"]: e for e in syn.expected["entities"]}
        for o in syn.ops:  # first op that introduces each entity: `about` is `new:<type>:<name>`
            keys = [m["entity"] for m in o.get("mentions", [])]
            if o["tool"] == "remember" and keys and not seen & set(keys):
                seen |= set(keys)
                about = [f"new:{ents[m['entity']]['type']}:{m['surface']}" for m in o["mentions"]]
                ops.append({**o, "args": {**o["args"], "about": about}})
            if len(ops) == n:
                break
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"store": e5.STORES[i % 2], "tool": o["tool"],
                                           "args": o["args"]}, ensure_ascii=False) + "\n"
                               for i, o in enumerate(ops)), encoding="utf-8")  # fmt: skip
        typer.echo(f"wrote {len(ops)} ops to {out}")

    @app.command("e5-run")
    def e5_run(
        url: UrlOpt,
        label: Annotated[str, typer.Option(help="e.g. home-local, work-tailscale")],
        role: Annotated[str, typer.Option(help="local (home, 127.0.0.1: the baseline) | remote")],
        queries: Annotated[Path, typer.Option(help="JSONL from `bench e5-queries`.")],
        ops: Annotated[Path, typer.Option(help="JSONL from `bench e5-ops`.")],
        scratch_url: Annotated[
            str, typer.Option(help="A `mem serve --http` on COPIES of the stores (write legs).")],
        out: Annotated[Path, typer.Option(help="Record, e.g. data/p5/e5-work.json")],
        host_label: HostOpt = "",
        runs: Annotated[int, typer.Option(help="Spec: 3 runs x 50 calls.")] = 3,
        transport: Annotated[str, typer.Option(help="Free text, e.g. tailscale.")] = "",
    ) -> None:  # fmt: skip
        """B-E5 on this machine; copy `out` to the gate machine."""
        rec = e5.run_client(url, label=label, role=role, host_label=_host_label(host_label, label),
                            queries=e5.load_queries(queries), ops=e5.load_ops(ops),
                            scratch=scratch_url, runs=runs, transport=transport)  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        lat = [v for s in e5.STORES for v in rec["stores"][s]["latencies_ms"]]
        typer.echo(f"{label}: {len(lat)} recalls, p95 {e5.p95(lat):.1f} ms; "
                   f"cross-store ids {rec['cross_store_ids']}; smoke failures {rec['smoke_failures']}"
                   f"; unstable {[s for s in e5.STORES if rec['stores'][s]['unstable_qids']]}")  # fmt: skip
        typer.echo(f"record: {out}")

    @app.command("e5-reconnect")
    def e5_reconnect(
        url: UrlOpt,
        label: Annotated[str, typer.Option(help="e.g. work-tailscale")],
        out: Annotated[Path, typer.Option(help="Record, e.g. data/p5/reconnect-work.json")],
        restart_cmd: Annotated[
            str | None, typer.Option(help="Shell command that restarts the server; else asked."),
        ] = None,
        host_label: HostOpt = "",
        store: Annotated[str, typer.Option()] = "personal",
        wait_s: Annotated[float, typer.Option(help="How long the server may take to come back.")] = 180.0,
    ) -> None:  # fmt: skip
        """Open a session, restart the server, call again: pass if it works within 1 retry."""
        rec = e5.run_reconnect(url, label=label, host_label=_host_label(host_label, label),
                               restart_cmd=restart_cmd, store=store, wait_s=wait_s)  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        typer.echo(f"{'ok' if rec['ok'] else 'FAILED'} retries={rec['retries']} {rec['detail']}")
        if not rec["ok"]:
            raise typer.Exit(1)

    @app.command("c6-run")
    def c6_run(
        out: Annotated[Path, typer.Option(help="Record, e.g. data/p5/c6.json")],
        model: Annotated[str, typer.Option(help="Model for count_tokens.")] = "claude-sonnet-5-5",
        core_block: Annotated[
            Path | None, typer.Option(help="Optional SessionStart core block (P6 variant)."),
        ] = None,
        r5_applied: Annotated[
            bool, typer.Option("--r5-applied", help="One server with a `store` param is in use "
                               "(counted as such only if every tool of --tools-url takes `store`)."),
        ] = False,
        tools_url: Annotated[
            str | None, typer.Option(help="Count the tools this MCP endpoint advertises, e.g. "
                                     "http://127.0.0.1:8765/mcp (default: the frozen contract)."),
        ] = None,
    ) -> None:  # fmt: skip
        """B-C6: needs ANTHROPIC_API_KEY and `uv run --with anthropic` (count_tokens is free)."""
        try:
            counter = c6.anthropic_counter(model)
        except RuntimeError as e:
            typer.echo(f"UNAVAILABLE {e}")
            raise typer.Exit(2) from e
        tools = c6.fetch_tools(tools_url) if tools_url else None
        rec = c6.measure(counter, tools=tools, core_block=core_block.read_text(encoding="utf-8")
                         if core_block else None, r5_applied=r5_applied, model=model)  # fmt: skip
        if r5_applied and not c6.r5_shape_ok(rec):
            typer.echo(f"note: --r5-applied but the tools have no `store` param; counted as "
                       f"{len(rec['servers'])} servers / {rec['tools_n']} tools (the gate fails this)")  # fmt: skip
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        typer.echo(f"session overhead {rec['session_tokens']} tokens (tools {rec['tools_tokens']}, "
                   f"skill {rec['skill_tokens']}, core block {rec['core_block_tokens']}; "
                   f"budget {rec['budget']})")  # fmt: skip
