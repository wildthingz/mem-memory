"""V5k / V50k load stores for B-C1 and B-C3 (benchmarks-v2 B-C, decision 0012).

V5k = the vault snapshot imported into ONE store (every folder routed to `personal`, see
`p4_ours.single_store_routes`): the latency unit is chunks in one store (A19), and the live system
splits the vault over two stores, each smaller than this one, so V5k is the worst case of the two.
V50k = V5k + LongMemEval_M filler chunks up to 50,000 chunks, written with `ingest_session` under
`lme:load:<sid>` source refs, sessions in sorted id order.

Like `OursLoadServer`, every call runs on one worker thread (a sqlite connection is thread-bound):
the 4-client total includes the queueing for it.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bench.adapters import p4_ours
from bench.datasets import corpora
from bench.datasets import vault as vault_ds
from bench.gold import pg


def _fill(store: Any, lme_m: Path, vault_chunks: int) -> str:
    """Add the V50k filler to `store`; returns the filler's corpus fingerprint."""
    from mem import write

    plan = vault_ds.filler_plan(lme_m, vault_chunks)
    corpora.build_corpus(plan, lambda date, rounds, sid: write.ingest_session(
        store, source_ref=f"lme:load:{sid}", date=date, rounds=rounds))  # fmt: skip
    return corpora.corpus_fingerprint(plan)


def _chunks(conn: Any) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


class VaultLoadServer:
    """The V5k / V50k store and the production `recall` with per-stage server timings."""

    def __init__(
        self, snapshot: Path, gold: Any, questions: Sequence[pg.Question], corpus: str,
        lme_m: Path | None, *, cfg: Any = None, config_path: Path | None = None,
        gate_path: Path | None = None, models: str = "st", device: str | None = None,
        serve_port: int | None = None, routes: dict[str, str] | None = None,
        importer: Any = None, loaded: tuple[Any, Any] | None = None, gate: Any = None,
    ) -> None:  # fmt: skip
        self._tmp = Path(tempfile.mkdtemp(prefix="bench-vload-"))
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="mem-serve")
        self._filters = {q.question: q for q in questions}
        self.fingerprint = ""
        self.backend: Any = None
        try:
            self._root = vault_ds.extract(snapshot, self._tmp / "snapshot")
            self._date = gold.snapshot["snapshot_date"]
            self.backend = self._worker.submit(
                self._build,
                corpus,
                lme_m,
                cfg,
                config_path,
                gate_path,
                models,
                device,
                serve_port,
                routes or {},
                importer,
                loaded,
                gate,
            ).result()
        except BaseException:
            self.close()
            raise

    def _build(self, corpus, lme_m, cfg, config_path, gate_path, models, device, serve_port, routes,
               importer, loaded, gate):  # fmt: skip
        be = p4_ours.OursVaultBackend(
            self._tmp / "stores", self._date, cfg=cfg, config_path=config_path, gate_path=gate_path,
            models=models, device=device, serve_port=serve_port, importer=importer, loaded=loaded,
            gate=gate)  # fmt: skip
        be.import_vault(self._root, routes=p4_ours.single_store_routes(self._root, routes))
        n = _chunks(be.store("personal").conn)
        h = hashlib.sha256(f"{corpus}:{n}".encode())
        if corpus == "V50k":
            h.update(_fill(be.store("personal"), lme_m, n).encode())  # type: ignore[arg-type]
        self.fingerprint = h.hexdigest()
        return be

    def _recall(self, query: str) -> dict[str, float]:
        from mem import read

        q = self._filters.get(query)
        flt = {} if q is None else {"about": q.filters.get("about") or None,
                                    "type": q.filters.get("type") or None,
                                    "since": q.filters.get("since"), "until": q.filters.get("until")}  # fmt: skip
        stages: dict[str, float] = {}
        read.recall(self.backend.store("personal"), query=query, as_of=self._date, timings=stages,
                    **flt)  # fmt: skip
        return stages

    def timed_recall(self, query: str) -> dict[str, float]:
        return self._worker.submit(self._recall, query).result()

    def close(self) -> None:
        try:
            if self.backend is not None:
                self._worker.submit(self.backend.close).result()
        finally:
            self._worker.shutdown()
            shutil.rmtree(self._tmp, ignore_errors=True)


def vault_write_server_build(
    backend: Any, snapshot: Path, gold: Any, corpus: str, lme_m: Path | None,
    routes: dict[str, str], *, importer: Any = None,
) -> tuple[Callable[[], Any], dict[str, str]]:  # fmt: skip
    """`build` for `OursWriteServer(..., build=)` (B-C3 at V5k / V50k): runs on the worker thread,
    returns the `MemServer` over the filled personal store."""
    info = {"fingerprint": ""}
    tmp = Path(tempfile.mkdtemp(prefix="bench-vc3-"))

    def build() -> Any:
        from mem.store import Store

        root = vault_ds.extract(snapshot, tmp / "snapshot")
        server, _clock = backend.open_server(tmp / "personal.db", scope="personal", tz="UTC",
                                             now=f"{gold.snapshot['snapshot_date']}T00:00:00Z")  # fmt: skip
        emb, rr = backend._start()
        work = Store.open(tmp / "work.db", emb, scope="work", fp=backend.cfg.fp_embed(), tz="UTC",
                          host="bench", session="bench", retrieval=backend.cfg.retrieval,
                          read=backend.cfg.read, ingest=backend.cfg.ingest, reranker=rr)  # fmt: skip
        fn = importer or p4_ours.resolve_importer()
        fn(root, {"personal": server.store, "work": work},
           routes=p4_ours.single_store_routes(root, routes))  # fmt: skip
        work.close()
        n = _chunks(server.store.conn)
        h = hashlib.sha256(f"{corpus}:{n}".encode())
        if corpus == "V50k":
            h.update(_fill(server.store, lme_m, n).encode())  # type: ignore[arg-type]
        info["fingerprint"] = h.hexdigest()
        return server

    return build, info
