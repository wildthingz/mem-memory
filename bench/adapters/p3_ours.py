"""The P3 seam to the production code (benchmarks-v2 §0.3). Private, not MCP.

`P3Backend` gives the P3 suites real stores: `mem.serve.open_inprocess` over the pinned models that
the bench's OWN `mem serve --http --models-only` serves (D5: the bench process loads no model; see
`bench.adapters.lme_ours.start_models`), with the gate `mem calibrate --out` wrote. Its `.factory`
is the `ClientFactory` of `bench gate P3`; `.open_server` hands back the in-process server for the
calibration scoring (`bench syn-score-dev`) and the B-C3 timing. `loaded=` hands in models from
elsewhere instead (the T0 tests: the `mem.testing` stand-ins, tests/bench_t0/test_p3_real.py).
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from bench.adapters.lme_ours import load_config, read_gate_file, start_models
from bench.adapters.ours import FIXTURE_START, FIXTURE_TZ, OursClient, _Clock
from bench.suites.base import BackendUnavailable, Client


class P3Backend:
    def __init__(
        self, *, config_path: Path | None = None, gate_path: Path | None = None,
        models: str = "st", device: str | None = None, serve_port: int | None = None,
        cfg: Any = None, loaded: tuple[Any, Any] | None = None, unfiltered_conflicts: bool = False,
        conflict_guard: bool | None = None,
    ) -> None:  # fmt: skip
        """`loaded`: (embedder, reranker) to use instead of starting the bench's own serve.
        `unfiltered_conflicts` (syn-score-dev only): no gate file, stores get an in-memory gate with
        theta_conf = -inf so `conflicts.find` returns every candidate. Otherwise a store whose gate
        lacks theta_conf is refused: `find` would silently return no conflicts.
        `conflict_guard`: override the config's `read.conflict_guard` on the stores (the θ_read fit,
        `bench syn-fit-read-guard`, reads with the guard off to record the pairs it would judge);
        with the guard on, a gate without `read_guard` is refused (ADR 0018)."""
        try:
            from mem.serve import open_inprocess  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"mem.serve not available: {e}") from e
        self.cfg = cfg or load_config(config_path, models)
        self._models, self._device, self._port, self._config_path = (
            models, device, serve_port, config_path)  # fmt: skip
        self.gate = read_gate_file(gate_path, self.cfg) if gate_path else None
        self.unfiltered_conflicts = unfiltered_conflicts
        self.read = dict(self.cfg.read)
        if conflict_guard is not None:
            self.read["conflict_guard"] = conflict_guard
        self._loaded: tuple[Any, Any] | None = loaded
        self.serve: Any = None

    def _start(self) -> tuple[Any, Any]:
        if self._loaded is None:
            self.serve = start_models(self.cfg, self._models, config_path=self._config_path,
                                      device=self._device, port=self._port)  # fmt: skip
            self._loaded = self.serve.models
        return self._loaded

    def open_server(self, db: Path, *, scope: str = "personal", tz: str = "UTC",
                    now: str = "2026-10-22T09:00:00Z") -> tuple[Any, _Clock]:  # fmt: skip
        from mem.gate import Gate, write_gate
        from mem.serve import MemServer
        from mem.store import Store

        if self.unfiltered_conflicts:
            gate = self.gate or Gate(1.0, 0.0, 0.0, 0.0, 0.0, fit_id="unfiltered-dev",
                                     fingerprint_hash=self.cfg.fp_all())  # fmt: skip
            gate = replace(gate, theta_conf=-math.inf)
        elif self.gate is None or self.gate.theta_conf is None:
            raise BackendUnavailable(
                "no gate with theta_conf: the store would skip conflict detection (B-W2/W3/W5/C3);"
                " pass --gate with theta_conf set (bench gate-add-theta)"
            )
        else:
            gate = self.gate
            if self.read.get("conflict_guard", False) and gate.needs_read_guard():
                raise BackendUnavailable(
                    "the gate has a conflict model but no read_guard (theta_read, ADR 0018): the"
                    " read path would refuse every recall; fit it (bench syn-fit-read-guard, bench"
                    " gate-add-read-guard) or pass a config with read.conflict_guard: false"
                )
        emb, rr = self._start()
        clock = _Clock(now)
        store = Store.open(db, emb, scope=scope, fp=self.cfg.fp_embed(), tz=tz, host="bench",
                           session="bench", now=clock, retrieval=self.cfg.retrieval,
                           reranker=rr, read=self.read, ingest=self.cfg.ingest)  # fmt: skip
        if not self.unfiltered_conflicts:
            write_gate(store.conn, gate)
        store.gate = gate
        return MemServer(store), clock

    def factory(self, db_path: Path, *, scope: str = "personal", tz: str = FIXTURE_TZ,
                now: str = FIXTURE_START) -> Client:  # fmt: skip
        """The suites' `ClientFactory`: like `ours.open_client`, the fixture zone unless told."""
        server, clock = self.open_server(Path(db_path), scope=scope, tz=tz, now=now)
        return OursClient(server, Path(db_path), clock)

    def close(self) -> None:
        if self.serve is not None:
            self.serve.stop()
            self.serve, self._loaded = None, None


class OursWriteServer:
    """B-C3: a load-corpus store (L5k / L50k via `ingest_session`) and timed `remember` calls on SYN
    ops. Like `OursLoadServer` every call runs on one worker thread, so 4 concurrent clients queue
    (the 4-client total includes the wait); `timed_remember("<op index>")` returns the server wall
    time in ms (`perf_counter_ns` around the call; the two loopback model calls are inside it)."""

    def __init__(self, plan: Any, syn: Any, backend: P3Backend, *,
                 build: Callable[[], Any] | None = None) -> None:  # fmt: skip
        """`build` (P4: V5k) returns the server over an already filled store instead of `plan`."""
        self.syn, self.backend, self._build_override = syn, backend, build
        self._db = Path(tempfile.mkdtemp(prefix="bench-c3-")) / "load.db"
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="mem-serve")
        self._ops = [op for op in syn.ops if op["tool"] == "remember"]
        self._ents = {e["key"]: e for e in syn.expected["entities"]}
        try:
            self.server = self._worker.submit(self._build, plan).result()
        except BaseException:
            self.close()
            raise

    def _build(self, plan: Any):
        if self._build_override is not None:
            return self._build_override()
        from bench.datasets.corpora import build_corpus
        from mem import write

        server, _clock = self.backend.open_server(self._db, tz="UTC")
        build_corpus(plan, lambda date, rounds, sid: write.ingest_session(
            server.store, source_ref=f"lme:load:{sid}", date=date, rounds=rounds))  # fmt: skip
        return server

    def n_ops(self) -> int:
        return len(self._ops)

    def _remember(self, idx: int) -> dict[str, float]:
        op = self._ops[idx]
        ent = self._ents[op["mentions"][0]["entity"]]
        args = {**op["args"], "about": [f"new:{ent['type']}:{ent['name']}"]}
        self.server.store.now = lambda t=op["now"]: _Clock(t)()  # the op date is the clock
        t0 = time.perf_counter_ns()
        self.server.call_tool("remember", args)
        return {"total": (time.perf_counter_ns() - t0) / 1e6}

    def timed_remember(self, idx: str) -> dict[str, float]:
        return self._worker.submit(self._remember, int(idx)).result()

    def close(self) -> None:
        try:
            if getattr(self, "server", None) is not None:
                self._worker.submit(self.server.close).result()
        finally:
            self._worker.shutdown()
            shutil.rmtree(self._db.parent, ignore_errors=True)
