"""The vault backend: the production pipeline over an imported vault (benchmarks-v2 §0.3, P4).

Private seam to production code, like `lme_ours`: nothing here loads a model (D5: the bench's own
pinned `mem serve`, `bench.adapters.lme_ours.start_models`), and without the P4 importer every
call that needs it raises `BackendUnavailable` (a gate then reports UNAVAILABLE, never a pass).

Stores (Q3, decision 0010 of the fixture task): the vault goes to two stores, `personal.db`
(`personal/`, `projects/`) and `work.db` (`work/`), folder overrides from `config.yaml`
(`vault.routes`). A PG question is asked of the store its notes were routed to (`Question.store`),
as the two MCP servers are asked separately in real use. `OursVaultBackend` opens both under one
directory.

Importer contract (decision 0010 of MEM2-31; `mem.importer` is the importer task's):

    mem.importer.run_import(vault: Path, stores: {"personal": Store, "work": Store}, *,
                            routes: dict[str, str], render_dir: str = "export") -> report

with a report that is a mapping or an object carrying `added, changed, renamed, deleted, restored,
unchanged` (summed over both stores). `resolve_importer` is the one place to change if it moves.

Lists (benchmarks-v2 §0.3), as gold ids `path::chunk_key`:
- S_q: the rows `mem.read.recall` returned, under the gate and the question's scope filters
  (`about`, `type`, `since`, `until`) at `t = snapshot date`;
- F_q: `read.candidate_pool` cut at 100 (same eligible set, legs, RRF, per-node cap);
- R_q: the first `pool` of F_q by reranker logit, then the rest of F_q;
- legs: `read.fts_leg` / `read.vec_leg` on that eligible set (BM25 and dense).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.adapters.lme_ours import (
    F_DEPTH,
    _Fixed,
    _MemoReranker,
    load_config,
    read_gate_file,
    start_models,
)
from bench.gold import pg
from bench.gold.chunks import ChunkIndex
from bench.lme_eval import Retrieved
from bench.suites.base import BackendUnavailable

REPORT_KEYS = ("added", "changed", "renamed", "deleted", "restored", "unchanged")
STORES = ("personal", "work")
Importer = Callable[..., Any]


def normalise_report(report: Any) -> dict[str, int]:
    """The six counts of an importer report, from a mapping or an object; absent = 0."""
    get = report.get if isinstance(report, Mapping) else lambda k, d=0: getattr(report, k, d)
    return {k: int(get(k, 0)) for k in REPORT_KEYS}


def resolve_importer() -> Importer:
    try:
        from mem import importer
    except ImportError as e:
        raise BackendUnavailable(f"the P4 importer is not available (mem.importer): {e}") from e
    fn = getattr(importer, "run_import", None)
    if fn is None:
        raise BackendUnavailable("mem.importer has no run_import(vault, stores, *, routes, "
                                 "render_dir): adapt bench/adapters/p4_ours.py:resolve_importer")  # fmt: skip
    return fn


def single_store_routes(vault: Path, routes: Mapping[str, str] | None = None) -> dict[str, str]:
    """Routes that send the whole vault to `personal`: every top-level folder and every configured
    override prefix mapped to personal (the longest prefix wins). V5k / V50k are one store of all
    the vault's chunks (decision 0012): the latency unit is chunks in one store."""
    out = {p.name: "personal" for p in Path(vault).iterdir() if p.is_dir()}
    out |= dict.fromkeys(routes or {}, "personal")
    return out


class OursVaultBackend:
    """The vault stores under `root` (`personal.db`, `work.db`), on the bench's models."""

    def __init__(
        self, root: Path, snapshot_date: str, *, config_path: Path | None = None,
        gate_path: Path | None = None, models: str = "st", device: str | None = None,
        gate: Any = None, cfg: Any = None, loaded: tuple[Any, Any] | None = None,
        serve_port: int | None = None, importer: Importer | None = None,
        stores: tuple[str, ...] = STORES,
    ) -> None:  # fmt: skip
        try:
            from mem import read  # noqa: F401
            from mem.store import Store  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"mem pipeline not available: {e}") from e
        from mem.common import normalize_as_of

        self.root, self.snapshot_date = Path(root), snapshot_date
        # `t` of the direct read-path calls: recall's own reading of `as_of=snapshot_date` (ADR 0028)
        self.t = normalize_as_of(snapshot_date, "UTC")
        self.root.mkdir(parents=True, exist_ok=True)
        self.dbs = {n: self.root / f"{n}.db" for n in stores}
        self.cfg = cfg or load_config(config_path, models)
        self._models, self._device, self._config_path = models, device, config_path
        self._serve_port, self._loaded, self.serve = serve_port, loaded, None
        self._importer = importer
        self.gate = gate if gate is not None else (
            read_gate_file(gate_path, self.cfg) if gate_path else None)  # fmt: skip
        self._stores: dict[str, Any] = {}
        self._index: ChunkIndex | None = None

    # -- store lifecycle
    def _ensure_models(self) -> tuple[Any, Any]:
        if self._loaded is None:
            self.serve = start_models(self.cfg, self._models, config_path=self._config_path,
                                       device=self._device, port=self._serve_port)  # fmt: skip
            self._loaded = self.serve.models
        return self._loaded

    def store(self, name: str = "personal", *, now: str | None = None) -> Any:
        """The named store (opened once, kept). `now` is the clock for writes (import, remember)."""
        from mem.gate import write_gate
        from mem.store import Store

        if name not in self._stores:
            emb, rr = self._ensure_models()
            s = Store.open(self.dbs[name], emb, scope=name, fp=self.cfg.fp_embed(), tz="UTC",
                           host="bench", session="bench", now=_Fixed(now or self.t),
                           retrieval=self.cfg.retrieval, read=self.cfg.read, ingest=self.cfg.ingest,
                           reranker=_MemoReranker(rr))  # fmt: skip
            if self.gate is not None:
                write_gate(s.conn, self.gate)
                s.gate = self.gate
            self._stores[name] = s
        return self._stores[name]

    def stores(self, *, now: str | None = None) -> dict[str, Any]:
        return {n: self.store(n, now=now) for n in self.dbs}

    def conns(self) -> dict[str, Any]:
        return {n: s.conn for n, s in self.stores().items()}

    def import_vault(
        self, vault_root: Path, *, routes: Mapping[str, str] | None = None, now: str | None = None
    ) -> dict[str, int]:  # fmt: skip
        fn = self._importer or resolve_importer()
        report = fn(Path(vault_root), self.stores(now=now), routes=dict(routes or {}))
        self._index = None
        return normalise_report(report)

    @property
    def index(self) -> ChunkIndex:
        if self._index is None:
            self._index = ChunkIndex.from_conns(self.conns())
        return self._index

    def close(self) -> None:
        for s in self._stores.values():
            s.close()
        self._stores.clear()
        if self.serve is not None:
            self.serve.stop()
            self.serve, self._loaded = None, None

    # -- reading
    def _filters(self, q: pg.Question) -> dict[str, Any]:
        f = q.filters
        return {"about": f.get("about") or None, "type": f.get("type") or None,
                "since": f.get("since"), "until": f.get("until")}  # fmt: skip

    def _gid(self, store: str, cid: int) -> str:
        c = self.index.by_id.get((store, cid))
        return c.gold_id if c else f"c{cid}"

    def rankings(self, q: pg.Question, *, want_legs: bool = True) -> Retrieved:
        """F_q, R_q and the legs without `recall` (no gate needed; pooling and calibration use it)."""
        from mem import read

        store = self.store(q.store)
        flt = self._filters(q)
        fused, _ = read.candidate_pool(store, q.question, t=self.t, size=F_DEPTH, **flt)
        f_ids = [c for c, _ in fused]
        n_pool = int(self.cfg.retrieval["pool"])
        zs = read.rerank(store, q.question, f_ids[:n_pool]) if store.reranker else []
        ps = []
        if self.gate and zs:
            cos = (read.pool_cosines(store, q.question, f_ids[:n_pool])
                   if store.read_model() is not None else None)  # fmt: skip
            ps = read.pool_probs(store, q.question, f_ids[:n_pool], zs, cos)
        key = ps if self.gate and store.read_model() is not None else zs
        order = sorted(range(len(zs)), key=lambda i: (-key[i], i))
        r_ids = [f_ids[i] for i in order] + f_ids[len(zs) :]
        gid = lambda c: self._gid(q.store, c)
        legs: dict[str, list[str]] = {}
        if want_legs:
            k = int(self.cfg.retrieval["leg_k"])  # `allowed` is still this query's eligible set
            legs = {"bm25": [gid(c) for c in read.fts_leg(store.conn, q.question, k)],
                    "dense": [gid(c) for c in read.vec_leg(
                        store.conn, store.vec_table, store.query_vector(q.question), k)],
                    "rrf": [gid(c) for c in f_ids]}  # fmt: skip
        p = sorted(ps, reverse=True)
        return Retrieved([gid(c) for c in f_ids], [gid(c) for c in r_ids], [], "", 0.0, p, legs)

    def systems(self, q: pg.Question) -> dict[str, list[str]]:
        """The pooling input (`bench.gold.pool.PoolBackend`)."""
        r = self.rankings(q)
        return {"ours": r.R, "bm25": r.legs["bm25"], "dense": r.legs["dense"]}

    def retrieve(self, q: pg.Question) -> Retrieved:
        """One B-R3 query: `recall` (S_q, text, server time) plus F_q, R_q, legs."""
        from mem import read

        if self.gate is None:
            raise BackendUnavailable("no gate: run `mem calibrate` and pass --gate gate.json")
        store = self.store(q.store)
        t0 = time.perf_counter()
        res = read.recall(store, query=q.question, as_of=self.snapshot_date, **self._filters(q))
        server_ms = (time.perf_counter() - t0) * 1000
        r = self.rankings(q)
        r.S = [self._gid(q.store, int(row["id"][1:])) if row["id"].startswith("c") else row["id"]
               for row in res.structured["rows"]]  # fmt: skip
        r.text, r.latency_ms = res.text, server_ms
        return r

    def calibration_pool(self, q: pg.Question) -> list[tuple[str, float]]:
        """(gold id, raw logit) of the rerank pool under the question's scope filters, gate off
        (`mem.calibrate.score_pool` plus filters): the input of the PG-tune calibration records."""
        from mem import read

        store = self.store(q.store)
        pool, _ = read.candidate_pool(store, q.question, t=self.t, **self._filters(q))
        ids = [c for c, _ in pool]
        zs = read.rerank(store, q.question, ids)
        return [(self._gid(q.store, c), z) for c, z in zip(ids, zs, strict=True)]


def temp_root(prefix: str = "bench-vault-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def drop_root(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)
