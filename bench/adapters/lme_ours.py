"""LongMemEval backend for the production pipeline (benchmarks-v2 §0.3). Private, not MCP.

Per question: a fresh store (`Store.open` with the config's embedder fingerprint and retrieval
section), every haystack session written with `mem.write.ingest_session`, then the production
`mem.read.recall(query, as_of=<the question's day at 23:59:59Z>)` (`lme.question_as_of`, ADR 0028)
under the gate from `mem calibrate --out`. The three lists come from the same read-path functions
`recall` calls:

- S_q: the rows `recall` returned (its structured result), and its text as the client sees it.
- F_q: `read.candidate_pool` cut at 100 instead of `pool` (same eligible set, legs, RRF, cap).
- R_q: the first `pool` of F_q (= recall's pool) by reranker logit, then the rest of F_q.
- legs: `read.fts_leg` / `read.vec_leg` on the same eligible set (baselines a, b); `rrf` = F_q.

The reranker is wrapped in a memo so R_q reuses the logits `recall` computed: one reranker pass.

Models (D5): this module never loads or constructs a model. On the first question the backend
starts the bench's own pinned `mem serve --http --models-only` (`bench.served.PinnedServe`: own
loopback port, the bench's config) and every embed and rerank of the write and read paths goes to
it over local HTTP (`mem.models_http`); `close()` stops it. The stores stay the bench's own files
(one fresh store per question). So `server_ms` is the production `recall` on the served models,
including its two loopback model calls (query embed, one rerank of the pool). Constructing a
backend is cheap (no serve, no gate). `models="hash"` serves the deterministic stand-ins of
`mem.testing` (T0 tests); `loaded=` hands in models from elsewhere (`mem calibrate --labels`
passes the person's running serve, `mem.models_http.connect`).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.datasets.lme import Haystack, question_as_of, ref
from bench.lme_eval import Retrieved
from bench.suites.base import BackendUnavailable

F_DEPTH = 100  # F_q: top 100 by RRF after the per-node cap (§0.3)
MODELS = ("st", "hash")


def load_config(config_path: Path | None, models: str):
    """The config in force: the file or the pinned defaults; the stub sections for `hash`."""
    from mem import config as mem_config

    if models not in MODELS:
        raise ValueError(f"models must be one of {MODELS}, got {models!r}")
    cfg = mem_config.load(config_path) if config_path else mem_config.default_config()
    return mem_config.stub_config(cfg) if models == "hash" else cfg


def start_models(
    cfg, models: str, *, config_path: Path | None = None, device: str | None = None,
    port: int | None = None,
):  # fmt: skip
    """The bench's own pinned `mem serve` for `cfg` (`bench.served`, D5), started and ready; its
    `.models` are the (embedder, reranker) clients. The bench process never constructs a model:
    `st` = the pinned models (the models extra and the snapshots from `mem doctor --fetch`),
    `hash` = the stand-ins, both loaded by the serve. Refuses an unpinned embedder."""
    import importlib.util

    from bench.served import PinnedServe
    from mem import catalog

    if models not in MODELS:
        raise ValueError(f"models must be one of {MODELS}, got {models!r}")
    if models == "st":
        try:
            catalog.embedder_entry(cfg.embedder)
            catalog.reranker_entry(cfg.reranker)
        except ValueError as err:
            raise BackendUnavailable(f"{err}: the bench serves only pinned models") from err
        if importlib.util.find_spec("sentence_transformers") is None:
            raise BackendUnavailable("models extra not installed: run with `uv run --extra models`")
    return PinnedServe(cfg, models=models, config_path=config_path, device=device,
                       port=port).start()  # fmt: skip


def read_gate_file(path: Path, cfg):
    """The gate `mem calibrate --out` wrote; it must belong to this config's fp_all."""
    import json

    from mem.gate import Gate

    gate = Gate(**json.loads(Path(path).read_text(encoding="utf-8")))
    if gate.fingerprint_hash != cfg.fp_all():
        raise BackendUnavailable(f"{path}: gate fitted for fp_all {gate.fingerprint_hash[:8]}, "
                                 f"the config is {cfg.fp_all()[:8]}; re-run mem calibrate")  # fmt: skip
    return gate


EXPAND_BATCH = 10  # `get` takes at most 10 ids
EXPAND_TOKENS = 1_000_000  # per `get`: no delivered row is cut (a cut is recorded, never silent)


def delivered_bodies(store, ids: Sequence[str], refs: dict[int, str],
                     dates: dict[str, str]) -> tuple[list[str], list[str]]:  # fmt: skip
    """What a client reads after `recall`: `get` on every row it returned, in recall's order.

    `recall` lists id, type, date, title (an imported chat session's title is "chat session
    <date>"); the chunk text comes from `get`. Each body is headed by its session, round and date
    (`refs`: chunk id -> `<sid>#<round>`, `chunk_refs`). Returns (bodies, ids `get` cut short).
    The one expanded path of B-E1 (`e1-run`, `--full`) and LoCoMo (`locomo_ours`)."""
    from mem import read

    bodies: list[str] = []
    cut: list[str] = []
    for i in range(0, len(ids), EXPAND_BATCH):
        res = read.get(store, list(ids[i : i + EXPAND_BATCH]), max_tokens=EXPAND_TOKENS)
        for item in res.structured["items"]:
            ident = item["id"]
            ref_ = refs.get(int(ident[1:]), "") if ident[:1] == "c" else ""
            sid, _, rnd = ref_.partition("#")
            head = f"[session {sid}, round {rnd}, {dates.get(sid, '')}]" if sid else f"[{ident}]"
            if item.get("continuation"):
                cut.append(ident)
            bodies.append(f"{head}\n{item['text']}")
    return bodies, cut


class _MemoReranker:
    """The store's reranker, remembering the last pool's logits: recall and R_q share one pass."""

    def __init__(self, inner: Any) -> None:
        self.inner, self._key, self._out = inner, None, []

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        key = (query, tuple(docs))
        if key != self._key:
            self._key, self._out = key, list(self.inner.score(query, docs))
        return list(self._out)


class _Fixed:
    def __init__(self, iso: str) -> None:
        self.t = datetime.fromisoformat(iso)

    def __call__(self) -> datetime:
        return self.t


def ingest_haystack(store, qid: str, haystack: Haystack) -> dict[int, str]:
    """Write every session with `ingest_session`; return chunk id -> chunk ref `<sid>#<round>`."""
    from mem import write

    for sid, date, rounds in haystack.sessions:
        write.ingest_session(store, source_ref=f"lme:{qid}:{sid}", date=date,
                             rounds=[(r.user, r.assistant) for r in rounds])  # fmt: skip
    return chunk_refs(store)


def chunk_refs(store) -> dict[int, str]:
    """Imported chunk id -> `<sid>#<round>`, the session id being `source_ref`'s third field
    (`lme:<qid>:<sid>`, `locomo:<sample>:<D_n>`)."""
    rows = store.conn.execute(
        "SELECT c.id, n.source_ref, c.chunk_key FROM chunks c JOIN nodes n ON n.id = c.node_id"
        " WHERE n.origin = 'import'"
    )
    return {cid: ref(src.split(":", 2)[2], int(key.split("/", 1)[1])) for cid, src, key in rows}


def later_pages(store, first, **recall_args: Any) -> list[dict[str, Any]]:
    """The rows of every page after `first` of a paged list answer (ADR 0024), in page order, asking
    `recall` with the same arguments and each page's `next_page` token; [] when `first` has none."""
    from mem import read

    rows: list[dict[str, Any]] = []
    page = first.structured.get("next_page")
    while page is not None:
        out = read.recall(store, page=page, **recall_args)
        rows += out.structured["rows"]
        page = out.structured.get("next_page")
    return rows


class StoreTools:
    """`bench.e1_agentic.ToolBox` over one store: calls go through `MemServer.call_tool`, so the
    arguments are validated against the frozen tool contract exactly as over MCP."""

    def __init__(self, store, db: Path) -> None:
        from bench.e1_agentic import TOOLS
        from mem.serve import MemServer, list_tool_schemas

        self.server, self._db = MemServer(store), db
        self.schemas = [{"name": t["name"], "description": t["description"],
                         "input_schema": t["inputSchema"]}
                        for t in list_tool_schemas() if t["name"] in TOOLS]  # fmt: skip

    def call(self, name: str, args: dict[str, Any]) -> str:
        return self.server.call_tool(name, args)[0]

    def close(self) -> None:
        self.server.close()
        self._db.unlink(missing_ok=True)


class OursLmeBackend:
    def __init__(
        self, workdir: Path | None = None, *, config_path: Path | None = None,
        gate_path: Path | None = None, models: str = "st", device: str | None = None,
        gate: Any = None, cfg: Any = None, loaded: tuple[Any, Any] | None = None,
        serve_port: int | None = None, expand: bool = False, follow_pages: bool = False,
    ) -> None:  # fmt: skip
        """`expand`: also fetch the full text of every delivered row into `Retrieved.bodies`
        (`delivered_bodies`), for the answerer of B-E1. `follow_pages`: ask for every later page of a
        paged answer (ADR 0024) into `Retrieved.paged` (`bench delivered-dump`'s whole delivery). `loaded`: (embedder, reranker) to use instead of starting the bench's own serve, e.g.
        the remote ones of a running mem serve (`mem.models_http.connect`, D5). `serve_port`:
        the own serve's port (default: a free one; refused if in use)."""
        try:
            from mem import read, write  # noqa: F401
            from mem.store import Store  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"P2 pipeline not available: {e}") from e
        self.cfg = cfg or load_config(config_path, models)
        self._models, self._device, self.expand = models, device, expand
        self.follow_pages = follow_pages
        self._config_path, self._serve_port = config_path, serve_port
        self._loaded: tuple[Any, Any] | None = loaded
        self.serve: Any = None
        self.gate = gate if gate is not None else (
            read_gate_file(gate_path, self.cfg) if gate_path else None)  # fmt: skip
        self._workdir = Path(workdir or tempfile.mkdtemp(prefix="bench-lme-"))
        self._workdir.mkdir(parents=True, exist_ok=True)

    def _open(self, name: str, as_of: str):
        from mem.store import Store

        if self._loaded is None:
            self.serve = start_models(self.cfg, self._models, config_path=self._config_path,
                                       device=self._device, port=self._serve_port)  # fmt: skip
            self._loaded = self.serve.models
        emb, rr = self._loaded
        db = self._workdir / f"{name}.db"
        db.unlink(missing_ok=True)
        return db, Store.open(db, emb, fp=self.cfg.fp_embed(), tz="UTC", host="bench",
                              session="bench", now=_Fixed(as_of), retrieval=self.cfg.retrieval,
                              read=self.cfg.read, ingest=self.cfg.ingest,
                              reranker=_MemoReranker(rr))  # fmt: skip

    def score(self, question: dict[str, Any], haystack: Haystack) -> list[tuple[str, float]]:
        """Calibration input (architecture-v2 §3.3 step 2): `mem.calibrate.score_pool` on a fresh
        store of this haystack, as (chunk ref, raw logit, cosine) in pool order. Needs no gate."""
        from mem import calibrate

        as_of = question_as_of(question["question_date"])
        db, store = self._open(f"{question['question_id']}-tune", as_of)
        try:
            refs = ingest_haystack(store, question["question_id"], haystack)
            pool = calibrate.score_pool(store, question["question"], as_of)
            return [(refs[int(i.key[1:])], i.z, i.cos, i.mcos, i.mz, i.pcos, i.pov) for i in pool]
        finally:
            store.close()
            db.unlink(missing_ok=True)

    def retrieve(
        self, question: dict[str, Any], haystack: Haystack, *, want_legs: bool
    ) -> Retrieved:
        from mem import read
        from mem.gate import write_gate

        if self.gate is None:
            raise BackendUnavailable("no gate: run `mem calibrate --labels … --out gate.json` on "
                                     "LME tune and pass --gate gate.json")  # fmt: skip
        qid, query = question["question_id"], question["question"]
        as_of = question_as_of(question["question_date"])
        db, store = self._open(f"{qid}-{len(haystack.sessions)}", as_of)
        try:
            write_gate(store.conn, self.gate)
            store.gate = self.gate
            t_ingest = time.perf_counter()
            refs = ingest_haystack(store, qid, haystack)
            ingest_ms = (time.perf_counter() - t_ingest) * 1000
            t0 = time.perf_counter()
            res = read.recall(store, query=query, as_of=as_of)
            server_ms = (time.perf_counter() - t0) * 1000
            shown = [refs[int(r["id"][1:])] for r in res.structured["rows"]]
            paged = ([refs[int(r["id"][1:])] for r in later_pages(store, res, query=query, as_of=as_of)]
                     if self.follow_pages else [])  # fmt: skip
            fused, _ = read.candidate_pool(store, query, t=as_of, size=F_DEPTH)
            f_ids = [cid for cid, _ in fused]
            n_pool = int(self.cfg.retrieval["pool"])
            zs = read.rerank(store, query, f_ids[:n_pool])  # the memo: recall's own logits
            ps = read.pool_probs(store, query, f_ids[:n_pool], zs)
            # R_q is the order recall delivers: by p under a read model, by z under a Platt gate
            key = ps if store.read_model() is not None else zs
            order = sorted(range(len(zs)), key=lambda i: (-key[i], i))
            r_ids = [f_ids[i] for i in order] + f_ids[n_pool:]
            legs: dict[str, list[str]] = {}
            if want_legs:
                k = int(self.cfg.retrieval["leg_k"])  # `allowed` is still recall's eligible set
                legs = {"bm25": [refs[c] for c in read.fts_leg(store.conn, query, k)],
                        "dense": [refs[c] for c in read.vec_leg(
                            store.conn, store.vec_table, store.query_vector(query), k)],
                        "rrf": [refs[c] for c in f_ids]}  # fmt: skip
                if store.session_leg:  # ADR 0022: the third RRF list, reported next to the others
                    legs["session"] = [refs[c] for c in read.session_leg(
                        store.conn, store.vec_table, store.query_vector(query), k,
                        int(self.cfg.retrieval["per_node_cap"]))]  # fmt: skip
            bodies: list[str] = []
            cut: list[str] = []
            if self.expand:
                dates = {sid: date for sid, date, _ in haystack.sessions}
                bodies, cut = delivered_bodies(store, [r["id"] for r in res.structured["rows"]],
                                               refs, dates)  # fmt: skip
            return Retrieved([refs[c] for c in f_ids], [refs[c] for c in r_ids], shown, res.text,
                             server_ms, sorted(ps, reverse=True), legs, bodies, cut, ingest_ms,
                             paged)  # fmt: skip
        finally:
            store.close()
            db.unlink(missing_ok=True)

    def open_tools(self, question: dict[str, Any], haystack: Haystack) -> StoreTools:
        """The production read tools (`recall`, `get`) on a fresh store of this haystack, for the
        agentic B-E1 arm. The caller closes the returned box (it deletes the store)."""
        from mem.gate import write_gate

        if self.gate is None:
            raise BackendUnavailable("no gate: pass --gate gate.json (from mem calibrate --out)")
        qid = question["question_id"]
        db, store = self._open(
            f"{qid}-{len(haystack.sessions)}", question_as_of(question["question_date"])
        )
        write_gate(store.conn, self.gate)
        store.gate = self.gate
        ingest_haystack(store, qid, haystack)
        return StoreTools(store, db)

    def close(self) -> None:
        if self.serve is not None:
            self.serve.stop()
            self.serve, self._loaded = None, None
        shutil.rmtree(self._workdir, ignore_errors=True)


class OursLoadServer:
    """A load-corpus store (L5k/L50k, every session through `ingest_session`) and the production
    `recall` with its per-stage server timings (`timings=`: eligible, fts, vec, fusion, rerank,
    total, from `perf_counter` inside `mem.read`). The models are the bench's own pinned
    `mem serve` (`start_models`, D5), started here and stopped by `close()`: `vec` includes the
    query-embed round trip and `rerank` the rerank round trip to it.

    Like `mem serve` (one connection, calls serialised by its event loop), every call runs on one
    worker thread: concurrent clients queue for it, and B-C1's 4-client total includes the wait."""

    def __init__(
        self, plan, *, config_path: Path | None = None, gate_path: Path | None = None,
        models: str = "st", device: str | None = None, gate: Any = None, cfg: Any = None,
        serve_port: int | None = None,
    ) -> None:  # fmt: skip
        cfg = cfg or load_config(config_path, models)
        gate = gate if gate is not None else (read_gate_file(gate_path, cfg) if gate_path else None)
        if gate is None:
            raise BackendUnavailable("no gate: pass --gate gate.json (from mem calibrate --out)")
        self.serve = start_models(cfg, models, config_path=config_path, device=device,
                                   port=serve_port)  # fmt: skip
        emb, rr = self.serve.models
        self._db = Path(tempfile.mkdtemp(prefix="bench-load-")) / "load.db"
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="mem-serve")
        try:
            self.store = self._worker.submit(self._build, plan, cfg, gate, emb, rr).result()
        except BaseException:
            self._worker.shutdown()
            self.serve.stop()
            shutil.rmtree(self._db.parent, ignore_errors=True)
            raise

    def _build(self, plan, cfg, gate, emb, rr):
        from bench.datasets.corpora import build_corpus
        from mem import write
        from mem.gate import write_gate
        from mem.store import Store

        store = Store.open(self._db, emb, fp=cfg.fp_embed(), tz="UTC", host="bench",
                           session="bench", retrieval=cfg.retrieval, reranker=rr, read=cfg.read,
                           ingest=cfg.ingest)  # fmt: skip
        write_gate(store.conn, gate)
        store.gate = gate
        build_corpus(plan, lambda date, rounds, sid: write.ingest_session(
            store, source_ref=f"lme:load:{sid}", date=date, rounds=rounds))  # fmt: skip
        return store

    def _recall(self, query: str) -> dict[str, float]:
        from mem import read

        stages: dict[str, float] = {}
        read.recall(self.store, query=query, timings=stages)
        return stages

    def timed_recall(self, query: str) -> dict[str, float]:
        return self._worker.submit(self._recall, query).result()

    def close(self) -> None:
        try:
            self._worker.submit(self.store.close).result()
            self._worker.shutdown()
        finally:
            self.serve.stop()
            shutil.rmtree(self._db.parent, ignore_errors=True)
